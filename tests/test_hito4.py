"""Hito 4: registro, servicio (paridad con la evaluación por lotes) y monitoreo."""

import json
import math
import shutil

import numpy as np
import pandas as pd
import pydantic
import pytest
from fastapi.testclient import TestClient

from fraude import (
    clientes,
    estado_en_linea,
    flujo_hito2,
    flujo_hito3,
    flujo_hito4,
    modelo_xgb,
    monitoreo,
    registro,
    servicio,
    variables,
)


def test_psi():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 20_000))
    assert monitoreo.psi(a, pd.Series(rng.normal(0, 1, 20_000))) < 0.01
    assert monitoreo.psi(a, pd.Series(rng.normal(1, 1, 20_000))) > 0.25
    cat = pd.Series([0.0, 1.0, 2.0] * 1000)
    assert monitoreo.psi(cat, pd.Series([0.0] * 3000)) > 0.25
    con_nan = pd.Series([np.nan] * 500 + list(range(500)), dtype="float64")
    assert monitoreo.psi(a, pd.concat([a.head(500), con_nan])) > 0.1


def test_niveles():
    assert monitoreo.nivel(0.3, "psi_variable") == "critica"
    assert monitoreo.nivel(0.12, "psi_puntaje") == "aviso"
    assert monitoreo.nivel(0.05, "psi_variable") is None
    assert monitoreo.nivel(1.6, "tasa_alertas") == "critica"
    assert monitoreo.nivel(0.75, "tasa_alertas") is None
    assert monitoreo.nivel(0.8, "ap_madura_nuevos") == "aviso"
    assert monitoreo.nivel(0.6, "ap_madura_nuevos") == "critica"


def _version(**k):
    base = dict(
        variante="x",
        archivo="a",
        sha256="0",
        parametros="p",
        sha256_parametros="0",
        descripcion="",
        metricas_prueba={},
        origen="",
        umbrales={},
    )
    return registro.Version(**{**base, **k})


def test_registro_exige_una_en_produccion():
    with pytest.raises(AssertionError):
        registro.validar(
            [_version(version="1", estado="produccion"), _version(version="2", estado="produccion")]
        )
    registro.validar(
        [_version(version="1", estado="produccion"), _version(version="2", estado="respaldo")]
    )


def test_entrada_rechaza_la_etiqueta():
    with pytest.raises(pydantic.ValidationError):
        servicio.Transaccion(TransactionID=1, TransactionDT=1, isFraud=1)


@pytest.mark.integracion
def test_servicio_reproduce_la_evaluacion_por_lotes(con, tmp_path):
    df = clientes.cargar_base(con)
    tabla = pd.DataFrame({"TransactionID": df["TransactionID"], "uid": clientes.asignar(df, "uid")})
    con.register("t_cli", tabla)
    con.execute("CREATE OR REPLACE TABLE clientes AS SELECT * FROM t_cli")
    flujo_hito2.preparar(con)
    h3 = tmp_path / "h3"
    flujo_hito3.buscar(con, h3, n=1, gpu=False)
    flujo_hito3.desarrollar(con, h3, gpu=False)
    flujo_hito3.evaluar(con, h3, h3 / "parametros.json", h3 / "modelos", tag=None)
    p = json.loads((h3 / "parametros.json").read_text(encoding="utf-8"))
    umbrales = {"completo": p["variantes"]["completo"]["umbrales_por_costo_fp"]}
    versiones = registro.registrar_desde_hito3(
        h3 / "parametros.json",
        h3 / "modelos",
        h3 / "evaluacion.json",
        {"completo": ("1.0.0", "produccion", "prueba")},
        umbrales,
        destino=tmp_path / "modelos",
        raiz=tmp_path,
    )
    version = versiones[0]

    crudas = flujo_hito4.transacciones_crudas(con)
    estado = estado_en_linea.Estado(30)
    pasado = crudas[crudas["TransactionDT"] < 121 * 86_400]
    for tx in pasado.to_dict(orient="records"):
        estado.procesar(tx)
        if tx["isFraud"] == 1:
            estado.reportar_fraude(int(tx["TransactionID"]))
    puntuador = servicio.Puntuador(version, estado, "10.0", raiz=tmp_path)
    cliente_http = TestClient(servicio.crear_app(puntuador))
    assert cliente_http.get("/salud").json()["version"] == "1.0.0"

    futuro = crudas[crudas["TransactionDT"] >= 121 * 86_400].head(300)
    probs = []
    for tx in futuro.to_dict(orient="records"):
        y = tx.pop("isFraud")
        cuerpo = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in tx.items()}
        r = cliente_http.post("/puntuar", json=cuerpo)
        assert r.status_code == 200, r.text
        probs.append(r.json()["probabilidad"])
        assert len(r.json()["motivos"]) == servicio.MOTIVOS
        if y == 1:
            assert (
                cliente_http.post(
                    "/etiqueta", json={"TransactionID": tx["TransactionID"], "fraude": True}
                ).status_code
                == 200
            )

    lotes = variables.cargar(con, 182)
    lotes = lotes.set_index("TransactionID").loc[futuro["TransactionID"]].reset_index()
    modelo = modelo_xgb.cargar(version.ruta(tmp_path), version.sha256)
    cod = variables.Codificador(p["codificadores"]["realista"])
    esperado = modelo_xgb.puntuar(
        modelo, variables.matriz(lotes, p["variantes"]["completo"]["columnas"], cod)
    )
    assert np.allclose(probs, esperado, atol=1e-6)
    m = cliente_http.get("/metricas").json()
    assert m["puntuadas"] == 300 and m["latencia_ms_p50"] > 0
    # Una transacción del pasado se rechaza (fuera de orden).
    viejo = {
        k: (None if isinstance(v, float) and math.isnan(v) else v)
        for k, v in pasado.drop(columns="isFraud").iloc[0].to_dict().items()
    }
    assert cliente_http.post("/puntuar", json=viejo).status_code == 409
    shutil.rmtree(h3)


def test_registro_versionado_es_valido_y_coincide_con_los_archivos():
    versiones = registro.leer()
    prod = registro.en_produccion(versiones)
    assert prod.version == "1.1.0" and registro.respaldo(versiones) is not None
    for v in versiones:
        if v.estado != "retirado":
            registro.verificar(v)


def test_detecta_relojes_encubiertos():
    dia = np.repeat(np.arange(1, 121), 50)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"dia": dia, "reloj": dia + rng.integers(0, 3, len(dia)), "ruido": rng.random(len(dia))}
    )
    rho = monitoreo.correlacion_con_dia(df, ["reloj", "ruido"])
    assert rho["reloj"] >= monitoreo.RELOJ > abs(rho["ruido"])
