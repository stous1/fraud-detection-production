"""Flujo del hito 4: producción y monitoreo.

- ``paridad``: recorre todo train con el estado en línea, compara cada variable con la tabla
  ``vecindario`` (con la que se entrenó) y guarda la instantánea del estado al cierre del día
  120 (``data/estado_dia120.pkl``), con la que arranca el servicio.
- ``registrar``: arma ``modelos/registro.json`` con el modelo completo en producción y el
  modelo sin las variables opacas de Vesta como respaldo (plan de contingencia). El umbral del
  respaldo se fija con la tasa de alertas elegida en validación y sus puntajes de los días 91
  a 120 (sin etiqueta), como el del principal.
- ``simular``: las nueve semanas de prueba llegan una a una. Por semana: deriva contra los días
  91 a 120, tasa de alertas y, con las etiquetas que ya maduraron (30 días), rendimiento.
  Alertas según ``monitoreo.REGLAS``. Usa las variables por lotes (idénticas a las del estado
  en línea, ver ``paridad``) y los modelos del registro, en CPU.
"""

import json
import pickle
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from fraude import (
    base,
    estado_en_linea,
    graficos,
    modelo_xgb,
    monitoreo,
    particion,
    registro,
    rutas,
    util,
    variables,
    vecindario,
)
from fraude import flujo_hito3 as h3

SALIDA = rutas.reports_hito(4)
INSTANTANEA = rutas.DATA / "estado_dia120.pkl"
VARIANTES_REGISTRO = {
    "completo": (
        "1.0.0",
        "produccion",
        "XGBoost completo (transacción, C/D/V/identidad, vecindario, historia). Rango en "
        "clientes nuevos: 0,300 sin las opacas a 0,551 con ellas; W 0,161.",
    ),
    "sin_sospechosas": (
        "1.0.0-respaldo",
        "respaldo",
        "Respaldo sin las variables opacas de Vesta (C, D, V, identidad), por si dejan de "
        "llegar o derivan. AP en clientes nuevos: 0,300.",
    ),
}
TOP_VARIABLES = 20


def transacciones_crudas(con) -> pd.DataFrame:
    """Transacciones con identidad, solo con los campos crudos del esquema (más ``isFraud``),
    en orden de ``TransactionID``: lo que recibiría el servicio."""
    from fraude import esquema

    tx = ", ".join(f't."{c}"' for c in esquema.COLUMNAS_TX)
    ident = ", ".join(f'i."{c}"' for c in esquema.COLUMNAS_ID if c != "TransactionID")
    return con.execute(
        f"SELECT {tx}, {ident} FROM transacciones t "
        "LEFT JOIN identidad i USING (TransactionID) ORDER BY t.TransactionID"
    ).df()


def paridad(con=None, salida: Path = SALIDA, instantanea: Path = INSTANTANEA) -> None:
    """Paridad en línea contra lotes en todo train e instantánea al día 120."""
    propia = con is None
    con = con or base.conectar(solo_lectura=True)
    crudas = transacciones_crudas(con)
    lotes = con.execute("SELECT * FROM vecindario ORDER BY TransactionID").df()
    if propia:
        con.close()
    estado = estado_en_linea.Estado(retraso=particion.RETRASO_BASE)
    filas = []
    guardada = False
    corte = (particion.ULTIMO_DIA_DESARROLLO + 1) * 86_400
    for tx in crudas.to_dict(orient="records"):
        if not guardada and tx["TransactionDT"] >= corte:
            instantanea.parent.mkdir(parents=True, exist_ok=True)
            with open(instantanea, "wb") as f:
                pickle.dump(estado, f, protocol=pickle.HIGHEST_PROTOCOL)
            guardada = True
        filas.append({"TransactionID": tx["TransactionID"], **estado.procesar(tx)})
        if tx["isFraud"] == 1:
            # Simulación: el banco recibe el contracargo; cuenta desde ts + 30 días.
            estado.reportar_fraude(int(tx["TransactionID"]))
    en_linea = pd.DataFrame(filas).set_index("TransactionID")
    cols = vecindario.columnas(lotes, particion.RETRASO_BASE)
    a = lotes.set_index("TransactionID")[cols].astype("float64")
    b = en_linea[cols].astype("float64")
    distintas = ~((a == b) | (a.isna() & b.isna()))
    util.escribir_json(
        {
            "transacciones": int(len(a)),
            "variables": len(cols),
            "celdas_comparadas": int(distintas.size),
            "celdas_distintas": int(distintas.to_numpy().sum()),
            "variables_con_diferencias": [c for c in cols if distintas[c].any()],
            "instantanea": "estado al cierre del día 120 (no versionada, en data/)",
            "instantanea_mb": round(instantanea.stat().st_size / 1e6, 1) if guardada else None,
        },
        salida / "paridad.json",
    )


def _puntajes(version: registro.Version, df: pd.DataFrame) -> np.ndarray:
    p = json.loads(version.ruta_parametros().read_text(encoding="utf-8"))
    v = p["variantes"][version.variante]
    cod = variables.Codificador(p["codificadores"][v["esquema"]])
    modelo = modelo_xgb.cargar(version.ruta(), version.sha256)
    return modelo_xgb.puntuar(modelo, variables.matriz(df, v["columnas"], cod))


def registrar(con=None) -> None:
    """Modelos del hito 3 al registro, con umbrales para producción y respaldo."""
    p = json.loads(h3.PARAMETROS_PUBLICADOS.read_text(encoding="utf-8"))
    principal = p["variantes"][h3.PRINCIPAL]["umbrales_por_costo_fp"]
    propia = con is None
    con = con or base.conectar(solo_lectura=True)
    df = variables.cargar(con, particion.ULTIMO_DIA_DESARROLLO)
    if propia:
        con.close()
    sin_etiqueta = df[df["dia"] > particion.ULTIMO_DIA_ETIQUETADO]  # días 91 a 120
    umbrales = {"completo": principal}
    # Respaldo: misma tasa de alertas por costo, convertida con sus propios puntajes.
    cod = variables.Codificador(p["codificadores"]["realista"])
    v = p["variantes"]["sin_sospechosas"]
    modelo = modelo_xgb.cargar(h3.MODELOS / v["modelo"], v["sha256"])
    s = modelo_xgb.puntuar(modelo, variables.matriz(sin_etiqueta, v["columnas"], cod))
    from fraude import costos

    umbrales["sin_sospechosas"] = {
        cfp: {
            "tasa_alertas": u["tasa_alertas"],
            "umbral": costos.umbral_de_tasa(s, u["tasa_alertas"]),
        }
        for cfp, u in principal.items()
    }
    versiones = registro.registrar_desde_hito3(
        h3.PARAMETROS_PUBLICADOS,
        h3.MODELOS,
        h3.PUBLICADO / "evaluacion.json",
        VARIANTES_REGISTRO,
        umbrales,
    )
    registro.escribir(versiones)


def simular(con=None, salida: Path = SALIDA, costo_fp: str = "10.0") -> None:
    """Las semanas de prueba llegan una a una: deriva, operación, rendimiento y alertas."""
    versiones = registro.leer()
    propia = con is None
    con = con or base.conectar(solo_lectura=True)
    df = variables.cargar(con, particion.ULTIMO_DIA)
    if propia:
        con.close()
    h3_dev = json.loads((h3.PUBLICADO / "desarrollo.json").read_text(encoding="utf-8"))
    h3_ev = json.loads((h3.PUBLICADO / "evaluacion.json").read_text(encoding="utf-8"))
    top = [
        x["variable"] for x in h3_ev["variantes"][h3.PRINCIPAL]["shap_prueba_nuevos"]["variables"]
    ][:TOP_VARIABLES]
    grupos = {
        g: c
        for g, c in variables.grupos(df.attrs["columnas_vecindario"]).items()
        if g in ("C", "D", "V", "identidad")
    }
    grupos["M"] = tuple(f"M{i}" for i in range(1, 10))
    grupos["correo"] = ("P_emaildomain", "R_emaildomain")

    referencia = df[
        (df["dia"] > particion.ULTIMO_DIA_ETIQUETADO)
        & (df["dia"] <= particion.ULTIMO_DIA_DESARROLLO)
    ]
    prueba = df[df["dia"] >= particion.PRUEBA[0]].copy()
    prueba["semana"] = prueba["dia"].map(particion.semana_prueba)
    vistos = set(df.loc[df["dia"] < particion.PRUEBA[0], "cliente"])
    prueba["nuevo"] = ~prueba["cliente"].isin(vistos)

    resultado: dict[str, object] = {"costo_fp_supuesto": float(costo_fp), "versiones": {}}
    todas_alertas = []
    for version in (registro.en_produccion(versiones), registro.respaldo(versiones)):
        umbral = float(version.umbrales[costo_fp]["umbral"])
        tasa = float(version.umbrales[costo_fp]["tasa_alertas"])
        s_ref = _puntajes(version, referencia)
        s_prueba = _puntajes(version, prueba)
        nombre_val = version.variante
        ap_val = h3_dev["variantes"][nombre_val]["validacion"]["nuevos"]["ap"]
        semanas = []
        for w in sorted(prueba["semana"].unique()):
            m = (prueba["semana"] == w).to_numpy()
            datos = monitoreo.semana(
                referencia, prueba[m], s_ref, s_prueba[m], umbral, tasa, top, grupos
            )
            fin = int(prueba.loc[m, "dia"].max())
            maduro = (prueba["dia"] <= fin - particion.RETRASO_BASE).to_numpy()
            datos["semana"] = int(w)
            datos["dias"] = [int(prueba.loc[m, "dia"].min()), fin]
            datos["etiquetas_maduras_hasta_dia"] = fin - particion.RETRASO_BASE
            if maduro.any():
                datos["rendimiento_maduro"] = monitoreo.rendimiento_maduro(
                    prueba.loc[maduro, "isFraud"].to_numpy(),
                    s_prueba[maduro],
                    prueba.loc[maduro, "nuevo"].to_numpy(),
                    umbral,
                )
            datos["alertas"] = monitoreo.alertas(int(w), datos, ap_val)
            if version.estado == "produccion":
                todas_alertas += datos["alertas"]
            semanas.append(datos)
        resultado["versiones"][version.version] = {
            "variante": version.variante,
            "estado": version.estado,
            "umbral": umbral,
            "tasa_alertas_esperada": tasa,
            "ap_validacion_nuevos": ap_val,
            "semanas": semanas,
        }
    resultado["alertas_produccion"] = todas_alertas
    resultado["reglas"] = monitoreo.REGLAS
    util.escribir_json(resultado, salida / "monitoreo.json")
    graficos_monitoreo(salida)


def graficos_monitoreo(salida: Path = SALIDA) -> None:
    import altair as alt

    m = json.loads((salida / "monitoreo.json").read_text(encoding="utf-8"))
    prod = next(v for v in m["versiones"].values() if v["estado"] == "produccion")
    filas = []
    for s in prod["semanas"]:
        for c, v in s["psi_variables"].items():
            filas.append({"semana": s["semana"], "variable": c, "psi": v})
    psi = pd.DataFrame(filas)
    tope = psi.groupby("variable")["psi"].max().sort_values(ascending=False).head(6).index
    psi_top = psi[psi["variable"].isin(tope)]
    reglas = pd.DataFrame({"y": [0.10, 0.25], "nivel": ["aviso", "crítica"]})
    g = alt.layer(
        alt.Chart(psi_top)
        .mark_line(strokeWidth=2, point=True)
        .encode(
            x=alt.X("semana:O", title="Semana de prueba", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("psi:Q", title="PSI contra los días 91 a 120"),
            color=alt.Color("variable:N", title=None, scale=alt.Scale(range=graficos.PALETA)),
            tooltip=["variable", "semana", alt.Tooltip("psi:Q", format=".3f")],
        ),
        alt.Chart(reglas).mark_rule(strokeDash=[4, 3], color=graficos.GRIS).encode(y="y:Q"),
    ).properties(title="Deriva de las seis variables que más cambiaron", width=560, height=260)
    graficos.guardar(g, salida / "psi_semanal")

    filas = []
    for v in m["versiones"].values():
        for s in v["semanas"]:
            r = s.get("rendimiento_maduro")
            if r and r["nuevos"]["ap"] is not None:
                filas.append(
                    {"semana": s["semana"], "version": v["variante"], "ap": r["nuevos"]["ap"]}
                )
    if filas:
        g = graficos.lineas(
            pd.DataFrame(filas),
            "semana",
            "ap",
            color="version",
            titulo="AP en clientes nuevos con las etiquetas ya maduras (acumulada)",
            eje_x="Semana de prueba",
            eje_y="AP en nuevos",
            formato_y=".2f",
        )
        graficos.guardar(g, salida / "rendimiento_maduro")
    filas = [{"semana": s["semana"], "tasa": s["tasa_alertas"]} for s in prod["semanas"]]
    g = graficos.lineas(
        pd.DataFrame(filas),
        "semana",
        "tasa",
        titulo=f"Tasa de alertas por semana (esperada {prod['tasa_alertas_esperada']:.0%})",
        eje_x="Semana de prueba",
        eje_y="Transacciones marcadas",
        formato_y=".0%",
    )
    graficos.guardar(g, salida / "tasa_alertas")


PASOS: dict[str, Callable[[], None]] = {
    "paridad": paridad,
    "registrar": registrar,
    "simular": simular,
    "graficos": graficos_monitoreo,
}
