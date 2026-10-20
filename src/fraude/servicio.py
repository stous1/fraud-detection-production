"""Servicio local de puntuación (FastAPI).

Rutas:

- ``POST /puntuar``: una transacción con los campos crudos de IEEE-CIS (contrato en
  ``esquema.py``; ``isFraud`` no se acepta). Calcula el vecindario con el estado en línea,
  puntúa en CPU con el modelo en producción del registro y devuelve probabilidad, decisión
  (alerta si supera el umbral de costo base), versión y los cinco motivos SHAP de más peso.
- ``POST /etiqueta``: registra el resultado de una transacción ya puntuada (un contracargo
  cuenta para las variables 30 días después de la transacción).
- ``GET /salud``, ``GET /modelo`` y ``GET /metricas`` (volumen, tasa de alertas, latencia).

El estado del vecindario arranca de una instantánea (``data/estado_dia120.pkl``, construida
por ``flujo_hito4.paridad``) y vive en memoria; un solo proceso, con un candado.

Variables de entorno: ``FRAUDE_REGISTRO`` (registro de versiones), ``FRAUDE_ESTADO``
(instantánea), ``FRAUDE_VERSION`` (forzar una versión del registro, por ejemplo el respaldo)
y ``FRAUDE_COSTO_FP`` (umbral según el costo de una falsa alarma; 10 por defecto).
"""

import json
import math
import os
import pickle
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, create_model

from fraude import esquema, estado_en_linea, modelo_xgb, registro, rutas, variables

MOTIVOS = 5


def _modelo_entrada() -> type[BaseModel]:
    campos: dict[str, tuple] = {}
    for c in (*esquema.COLUMNAS_TX, *esquema.COLUMNAS_ID):
        if c in ("isFraud",):
            continue
        if c in ("TransactionID", "TransactionDT"):
            campos[c] = (int, ...)
        elif c in (*esquema.TEXTO_TX, *esquema.TEXTO_ID):
            campos[c] = (str | None, None)
        else:
            campos[c] = (float | None, None)
    return create_model("Transaccion", __config__=ConfigDict(extra="forbid"), **campos)


Transaccion = _modelo_entrada()


class Etiqueta(BaseModel):
    TransactionID: int
    fraude: bool


class Puntuador:
    """Modelo, columnas, codificador, umbral y estado en línea de una versión del registro."""

    def __init__(
        self,
        version: registro.Version,
        estado: estado_en_linea.Estado,
        costo_fp: str = "10.0",
        raiz: Path = rutas.RAIZ,
    ) -> None:
        registro.verificar(version, raiz)
        self.version = version
        p = json.loads(version.ruta_parametros(raiz).read_text(encoding="utf-8"))
        v = p["variantes"][version.variante]
        self.columnas: list[str] = v["columnas"]
        self.codificador = variables.Codificador(p["codificadores"][v["esquema"]])
        self.umbral = float(version.umbrales[costo_fp]["umbral"])
        self.costo_fp = costo_fp
        self.modelo = modelo_xgb.cargar(version.ruta(raiz), version.sha256)
        self.estado = estado
        self.candado = threading.Lock()
        self.latencias: deque[float] = deque(maxlen=10_000)
        self.puntuadas = 0
        self.alertas = 0

    def fila(self, tx: dict[str, object], vec: dict[str, float]) -> pd.DataFrame:
        crudo = {k: (np.nan if x is None else x) for k, x in tx.items()}
        df = pd.DataFrame([{**crudo, **vec}])
        df = variables.derivadas(df)
        return variables.matriz(df, self.columnas, self.codificador)

    def puntuar(self, tx: dict[str, object]) -> dict[str, object]:
        inicio = time.perf_counter()
        with self.candado:
            try:
                vec = self.estado.procesar(tx)
            except ValueError as e:
                raise HTTPException(status_code=409, detail=str(e)) from e
            x = self.fila(tx, vec)
            prob = float(modelo_xgb.puntuar(self.modelo, x)[0])
            contrib = modelo_xgb.contribuciones(self.modelo, x).iloc[0].drop("sesgo")
            alerta = prob > self.umbral
            self.puntuadas += 1
            self.alertas += int(alerta)
        top = contrib.abs().sort_values(ascending=False).head(MOTIVOS).index
        motivos = [
            {
                "variable": c,
                "valor": None if math.isnan(float(x[c].iloc[0])) else float(x[c].iloc[0]),
                "aporte_log_odds": float(contrib[c]),
            }
            for c in top
        ]
        self.latencias.append(time.perf_counter() - inicio)
        return {
            "TransactionID": tx["TransactionID"],
            "probabilidad": prob,
            "alerta": bool(alerta),
            "umbral": self.umbral,
            "costo_fp_supuesto": float(self.costo_fp),
            "version": self.version.version,
            "motivos": motivos,
        }

    def metricas(self) -> dict[str, object]:
        lat = np.array(self.latencias) * 1000 if self.latencias else np.array([np.nan])
        return {
            "puntuadas": self.puntuadas,
            "alertas": self.alertas,
            "tasa_alertas": self.alertas / self.puntuadas if self.puntuadas else None,
            "latencia_ms_p50": float(np.nanpercentile(lat, 50)),
            "latencia_ms_p95": float(np.nanpercentile(lat, 95)),
            "contracargos_recibidos": len(self.estado.fraudes),
        }


def crear_app(puntuador: Puntuador) -> FastAPI:
    app = FastAPI(title="Puntuación de fraude (IEEE-CIS)", version=puntuador.version.version)

    @app.get("/salud")
    def salud() -> dict[str, object]:
        return {"estado": "ok", "version": puntuador.version.version}

    @app.get("/modelo")
    def modelo() -> dict[str, object]:
        v = puntuador.version
        return {
            "version": v.version,
            "variante": v.variante,
            "estado": v.estado,
            "sha256": v.sha256,
            "umbral": puntuador.umbral,
            "costo_fp_supuesto": float(puntuador.costo_fp),
            "metricas_prueba": v.metricas_prueba,
            "descripcion": v.descripcion,
        }

    @app.get("/metricas")
    def metricas() -> dict[str, object]:
        return puntuador.metricas()

    @app.post("/puntuar")
    def puntuar(tx: Transaccion) -> dict[str, object]:  # type: ignore[valid-type]
        return puntuador.puntuar(tx.model_dump())

    @app.post("/etiqueta")
    def etiqueta(e: Etiqueta) -> dict[str, object]:
        if e.TransactionID not in puntuador.estado.transacciones:
            raise HTTPException(status_code=404, detail="transacción no puntuada")
        if e.fraude:
            with puntuador.candado:
                puntuador.estado.reportar_fraude(e.TransactionID)
        return {"TransactionID": e.TransactionID, "registrado": True}

    return app


def desde_entorno() -> FastAPI:
    """App con la configuración de las variables de entorno (la usa ``uvicorn``)."""
    ruta_registro = Path(os.environ.get("FRAUDE_REGISTRO", registro.REGISTRO))
    ruta_estado = Path(os.environ.get("FRAUDE_ESTADO", rutas.DATA / "estado_dia120.pkl"))
    versiones = registro.leer(ruta_registro)
    pedida = os.environ.get("FRAUDE_VERSION")
    version = (
        next(v for v in versiones if v.version == pedida)
        if pedida
        else registro.en_produccion(versiones)
    )
    with open(ruta_estado, "rb") as f:
        estado = pickle.load(f)
    return crear_app(Puntuador(version, estado, os.environ.get("FRAUDE_COSTO_FP", "10.0")))
