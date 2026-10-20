"""Flujo del hito 3: XGBoost (GPU), ablaciones por grupo, umbral por costo y SHAP.

**Cifra principal: AP en clientes nuevos** (sin transacciones antes del periodo). En clientes
ya vistos, el rendimiento está inflado por la propagación de etiquetas de la competencia: en
un banco real, la tarjeta se bloquea tras un contracargo. Se reporta aparte.

- ``buscar`` (GPU): búsqueda aleatoria de hiperparámetros con el modelo completo; entrena con
  los días 1 a 60 y elige por AP en clientes nuevos de la validación (días 61 a 90).
- ``desarrollar`` (GPU): cada variante de ``VARIANTES`` con los mismos hiperparámetros:
  parada temprana en validación, reajuste con los días 1 a 90, umbral por costo (principal) y
  SHAP en validación. Escribe ``parametros.json`` con las huellas de los modelos.
- ``evaluar`` (CPU): se niega a correr si ``docs/resultados/hito3/parametros.json`` no está
  congelado en ``hito3-congelado`` o si un modelo no coincide con su huella. Mide en prueba
  una sola vez.
"""

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from fraude import (
    base,
    congelados,
    costos,
    evaluacion,
    graficos,
    modelo_xgb,
    particion,
    rutas,
    util,
    variables,
)

SALIDA = rutas.reports_hito(3)
MODELOS = SALIDA / "modelos"
PUBLICADO = rutas.docs_hito(3)
PARAMETROS_PUBLICADOS = PUBLICADO / "parametros.json"
TAG = "hito3-congelado"
CONFIGURACIONES = 40
MUESTRA_SHAP = 20_000
UMBRAL_RED_GRAFO = 0.01  # aporte mínimo del vecindario (AP en nuevos, validación)


@dataclass(frozen=True)
class Variante:
    nombre: str
    grupos: tuple[str, ...]
    esquema: str = "realista"
    descripcion: str = ""


TODOS = ("transaccion", "sospechosas", "vecindario", "historia")
VARIANTES: tuple[Variante, ...] = (
    Variante("completo", TODOS, descripcion="Principal: los cuatro grupos."),
    Variante("solo_transaccion", ("transaccion",), descripcion="Solo datos de la transacción."),
    Variante(
        "transaccion_sospechosas",
        ("transaccion", "sospechosas"),
        descripcion="Transacción + C, D, V e identidad.",
    ),
    Variante(
        "sin_historia",
        ("transaccion", "sospechosas", "vecindario"),
        descripcion="Todo menos la historia de fraude del cliente.",
    ),
    Variante(
        "sin_transaccion",
        ("sospechosas", "vecindario", "historia"),
        descripcion="Todo menos los datos de la transacción.",
    ),
    Variante(
        "sin_sospechosas",
        ("transaccion", "vecindario", "historia"),
        descripcion="Todo menos C, D, V e identidad.",
    ),
    Variante(
        "sin_vecindario",
        ("transaccion", "sospechosas", "historia"),
        descripcion="Todo menos el vecindario del grafo.",
    ),
    Variante(
        "transaccion_vecindario",
        ("transaccion", "vecindario"),
        descripcion="Sin sospechosas ni historia: lo más defendible fuera de IEEE-CIS.",
    ),
    Variante(
        "completo_todas_las_etiquetas",
        TODOS,
        "todas_las_etiquetas",
        "Comparación: entrena y valida con todas las etiquetas de desarrollo.",
    ),
)
PRINCIPAL = "completo"


def _tramo(df: pd.DataFrame, tramo: tuple[int, int]) -> pd.DataFrame:
    return df[(df["dia"] >= tramo[0]) & (df["dia"] <= tramo[1])]


def visto_antes(df: pd.DataFrame, filas: pd.DataFrame, dia_inicio: int) -> np.ndarray:
    previos = set(df.loc[df["dia"] < dia_inicio, "cliente"])
    return filas["cliente"].isin(previos).to_numpy()


def columnas_de(df: pd.DataFrame, grupos_variante: tuple[str, ...]) -> list[str]:
    g = variables.grupos(df.attrs["columnas_vecindario"])
    return [c for nombre in grupos_variante for c in g[nombre]]


def _partes(df: pd.DataFrame, esquema: str) -> dict[str, object]:
    e = particion.ESQUEMAS[esquema]
    ent, val, reaj = (_tramo(df, e[k]) for k in ("entrenamiento", "validacion", "reajuste"))
    nuevos_val = ~visto_antes(df, val, e["validacion"][0])
    return {"ent": ent, "val": val, "reaj": reaj, "nuevos_val": nuevos_val}


def _dispositivo(gpu: bool) -> str:
    return "cuda" if gpu and modelo_xgb.hay_gpu() else "cpu"


def buscar(
    con: duckdb.DuckDBPyConnection | None = None,
    salida: Path = SALIDA,
    n: int = CONFIGURACIONES,
    gpu: bool = True,
) -> None:
    """Búsqueda aleatoria con el modelo completo (esquema realista) -> ``busqueda.json``."""
    propia = con is None
    con = con or base.conectar(solo_lectura=True)
    df = variables.cargar(con, particion.ULTIMO_DIA_DESARROLLO)
    if propia:
        con.close()
    p = _partes(df, "realista")
    cols = columnas_de(df, TODOS)
    cod = variables.Codificador.ajustar(p["ent"])
    x_ent = variables.matriz(p["ent"], cols, cod)
    x_val = variables.matriz(p["val"], cols, cod)
    y_ent, y_val = p["ent"]["isFraud"].to_numpy(), p["val"]["isFraud"].to_numpy()
    nuevos = p["nuevos_val"]
    dispositivo = _dispositivo(gpu)
    filas = []
    for i, hiper in enumerate(modelo_xgb.configuraciones(n)):
        modelo, arboles = modelo_xgb.entrenar(
            x_ent, y_ent, hiper, dispositivo, (x_val[nuevos], y_val[nuevos])
        )
        s = modelo_xgb.puntuar_en_memoria(modelo, x_val, arboles)
        filas.append(
            {
                "configuracion": i,
                **hiper,
                "arboles": arboles,
                "ap_validacion_nuevos": evaluacion.ap(y_val[nuevos], s[nuevos]),
                "ap_validacion_todos": evaluacion.ap(y_val, s),
                "ap_validacion_vistos": evaluacion.ap(y_val[~nuevos], s[~nuevos]),
            }
        )
        print(i, filas[-1]["ap_validacion_nuevos"], arboles, flush=True)
    mejor = max(filas, key=lambda f: (f["ap_validacion_nuevos"], -f["configuracion"]))
    util.escribir_json(
        {
            "criterio": "AP en clientes nuevos de la validación realista (días 61 a 90)",
            "dispositivo": dispositivo,
            "mejor": mejor,
            "configuraciones": filas,
            "nota": "Cifras de validación: se usaron para elegir, son optimistas.",
        },
        salida / "busqueda.json",
    )


def _hiper(fila: dict[str, object]) -> dict[str, object]:
    return {k: fila[k] for k in modelo_xgb.ESPACIO}


def _importancia_grupos(
    contrib: pd.DataFrame, grupos: dict[str, tuple[str, ...]]
) -> dict[str, object]:
    abs_medio = contrib.drop(columns="sesgo").abs().mean()
    total = float(abs_medio.sum())
    por_grupo = {
        g: float(abs_medio[[c for c in cols if c in abs_medio.index]].sum()) / total
        for g, cols in grupos.items()
    }
    top = abs_medio.sort_values(ascending=False).head(25)
    grupo_de = {c: g for g, cols in grupos.items() for c in cols}
    return {
        "fraccion_por_grupo": por_grupo,
        "variables": [
            {"variable": c, "grupo": grupo_de.get(c), "shap_abs_medio": float(v)}
            for c, v in top.items()
        ],
    }


def desarrollar(
    con: duckdb.DuckDBPyConnection | None = None,
    salida: Path = SALIDA,
    gpu: bool = True,
    hiper: dict[str, object] | None = None,
) -> None:
    """Entrena cada variante y escribe ``parametros.json`` y ``desarrollo.json``."""
    propia = con is None
    con = con or base.conectar(solo_lectura=True)
    df = variables.cargar(con, particion.ULTIMO_DIA_DESARROLLO)
    if propia:
        con.close()
    if hiper is None:
        busqueda = json.loads((salida / "busqueda.json").read_text(encoding="utf-8"))
        hiper = _hiper(busqueda["mejor"])
    dispositivo = _dispositivo(gpu)
    grupos = variables.grupos(df.attrs["columnas_vecindario"])
    modelos_dir = salida / "modelos"
    codificadores: dict[str, variables.Codificador] = {}
    parametros: dict[str, object] = {
        "principal": PRINCIPAL,
        "cifra_principal": "AP en clientes nuevos",
        "hiperparametros": hiper,
        "base_xgboost": modelo_xgb.BASE,
        "dispositivo_entrenamiento": dispositivo,
        "esquemas": particion.ESQUEMAS,
        "grupos": {g: list(c) for g, c in grupos.items()},
        "variantes": {},
    }
    desarrollo: dict[str, object] = {"variantes": {}}
    for v in VARIANTES:
        p = _partes(df, v.esquema)
        if v.esquema not in codificadores:
            codificadores[v.esquema] = variables.Codificador.ajustar(p["reaj"])
        cod_ent = variables.Codificador.ajustar(p["ent"])
        cols = columnas_de(df, v.grupos)
        y_ent, y_val = p["ent"]["isFraud"].to_numpy(), p["val"]["isFraud"].to_numpy()
        nuevos = p["nuevos_val"]
        x_ent = variables.matriz(p["ent"], cols, cod_ent)
        x_val = variables.matriz(p["val"], cols, cod_ent)
        modelo, arboles = modelo_xgb.entrenar(
            x_ent, y_ent, hiper, dispositivo, (x_val[nuevos], y_val[nuevos])
        )
        s_val = modelo_xgb.puntuar_en_memoria(modelo, x_val, arboles)
        x_reaj = variables.matriz(p["reaj"], cols, codificadores[v.esquema])
        final, _ = modelo_xgb.entrenar(
            x_reaj, p["reaj"]["isFraud"].to_numpy(), hiper, dispositivo, arboles=arboles
        )
        ruta = modelos_dir / f"{v.nombre}.ubj"
        sha = modelo_xgb.guardar(final, ruta, arboles)
        entrada: dict[str, object] = {
            "grupos": list(v.grupos),
            "esquema": v.esquema,
            "descripcion": v.descripcion,
            "columnas": cols,
            "arboles": arboles,
            "modelo": ruta.name,
            "sha256": sha,
        }
        dev: dict[str, object] = {
            "validacion": evaluacion.por_segmento(y_val, s_val, ~nuevos, intervalo=False)
        }
        if v.nombre == PRINCIPAL:
            # Umbral por costo: tasa de alertas elegida en validación, convertida en umbral
            # con el modelo reajustado sobre los días 91 a 120 (sin etiqueta, ya vistos).
            sin_etiqueta = _tramo(
                df, (particion.ULTIMO_DIA_ETIQUETADO + 1, particion.ULTIMO_DIA_DESARROLLO)
            )
            cargado = modelo_xgb.cargar(ruta, sha)
            s_sin = modelo_xgb.puntuar(
                cargado, variables.matriz(sin_etiqueta, cols, codificadores[v.esquema])
            )
            monto = p["val"]["TransactionAmt"].to_numpy()
            umbrales, curvas = {}, {}
            for cfp in costos.COSTOS_FP:
                elegido = costos.elegir_tasa(y_val, s_val, monto, cfp)
                umbrales[str(cfp)] = {
                    "tasa_alertas": elegido["tasa"],
                    "umbral": costos.umbral_de_tasa(s_sin, elegido["tasa"]),
                }
                curvas[str(cfp)] = {
                    "mejor": elegido["mejor"],
                    "sin_modelo": elegido["sin_modelo"],
                }
            entrada["umbrales_por_costo_fp"] = umbrales
            entrada["costo_fp_base"] = costos.COSTO_FP_BASE
            dev["costo_validacion"] = curvas
            muestra = x_val[nuevos].sample(
                min(MUESTRA_SHAP, int(nuevos.sum())), random_state=modelo_xgb.SEMILLA
            )
            dev["shap_validacion_nuevos"] = _importancia_grupos(
                modelo_xgb.contribuciones(modelo[:arboles], muestra), grupos
            )
        parametros["variantes"][v.nombre] = entrada
        desarrollo["variantes"][v.nombre] = dev
        print(v.nombre, arboles, dev["validacion"]["nuevos"]["ap"], flush=True)
    parametros["codificadores"] = {k: c.mapas for k, c in codificadores.items()}

    ap_nuevos = {n: d["validacion"]["nuevos"]["ap"] for n, d in desarrollo["variantes"].items()}
    aporte_vecindario = ap_nuevos[PRINCIPAL] - ap_nuevos["sin_vecindario"]
    desarrollo["regla_red_grafo"] = {
        "aporte_vecindario_ap_nuevos_validacion": aporte_vecindario,
        "umbral": UMBRAL_RED_GRAFO,
        "entrenar_red": bool(aporte_vecindario >= UMBRAL_RED_GRAFO),
        "regla": "Se entrena una red sobre el grafo solo si el vecindario aporta al menos "
        f"{UMBRAL_RED_GRAFO} de AP en clientes nuevos de la validación sobre XGBoost.",
    }
    desarrollo["nota"] = "Cifras de validación: se usaron para elegir, son optimistas."
    util.escribir_json(parametros, salida / "parametros.json")
    util.escribir_json(desarrollo, salida / "desarrollo.json")


def evaluar(
    con: duckdb.DuckDBPyConnection | None = None,
    salida: Path = SALIDA,
    ruta_parametros: Path = PARAMETROS_PUBLICADOS,
    modelos_dir: Path = MODELOS,
    tag: str | None = TAG,
) -> None:
    """Prueba (días 121 a 182), una sola vez, en CPU con los modelos congelados."""
    if tag is not None:
        congelados.verificar(tag, ruta_parametros)
    parametros = json.loads(Path(ruta_parametros).read_text(encoding="utf-8"))
    propia = con is None
    con = con or base.conectar(solo_lectura=True)
    df = variables.cargar(con, particion.ULTIMO_DIA)
    if propia:
        con.close()
    prueba = _tramo(df, particion.PRUEBA)
    visto = visto_antes(df, prueba, particion.PRUEBA[0])
    y = prueba["isFraud"].to_numpy()
    monto = prueba["TransactionAmt"].to_numpy()
    grupos_cliente = prueba["cliente"].to_numpy()
    grupos = {g: tuple(c) for g, c in parametros["grupos"].items()}
    resultados = {}
    curvas = []
    for nombre, p in parametros["variantes"].items():
        cod = variables.Codificador(parametros["codificadores"][p["esquema"]])
        modelo = modelo_xgb.cargar(modelos_dir / p["modelo"], p["sha256"])
        x = variables.matriz(prueba, p["columnas"], cod)
        s = modelo_xgb.puntuar(modelo, x)
        r: dict[str, object] = {
            "descripcion": p["descripcion"],
            "grupos": p["grupos"],
            "esquema": p["esquema"],
            "prueba": evaluacion.por_segmento(y, s, visto, grupos_cliente),
        }
        if nombre == PRINCIPAL:
            r["costo"] = {}
            for cfp, u in p["umbrales_por_costo_fp"].items():
                r["costo"][cfp] = {
                    "umbral_congelado": u,
                    **{
                        seg: {
                            "modelo": costos.costo(y[m], s[m] > u["umbral"], monto[m], float(cfp)),
                            "sin_modelo": costos.costo(
                                y[m], np.zeros(int(m.sum()), bool), monto[m], float(cfp)
                            ),
                        }
                        for seg, m in (
                            ("todos", np.ones(len(y), bool)),
                            ("nuevos", ~visto),
                            ("vistos", visto),
                        )
                    },
                }
            nuevos_x = x[~visto]
            muestra = nuevos_x.sample(
                min(MUESTRA_SHAP, len(nuevos_x)), random_state=modelo_xgb.SEMILLA
            )
            r["shap_prueba_nuevos"] = _importancia_grupos(
                modelo_xgb.contribuciones(modelo, muestra), grupos
            )
        if nombre in (PRINCIPAL, "solo_transaccion", "transaccion_vecindario"):
            for seg, m in (("nuevos", ~visto), ("vistos", visto)):
                for punto in evaluacion.curva_pr(y[m], s[m], puntos=50):
                    curvas.append({"variante": nombre, "clientes": seg, **punto})
        resultados[nombre] = r
        print(nombre, r["prueba"]["nuevos"]["ap"], flush=True)

    ap = {n: r["prueba"]["nuevos"]["ap"] for n, r in resultados.items()}
    aportes = {
        "dejar_uno_fuera": {
            "transaccion": ap[PRINCIPAL] - ap["sin_transaccion"],
            "sospechosas": ap[PRINCIPAL] - ap["sin_sospechosas"],
            "vecindario": ap[PRINCIPAL] - ap["sin_vecindario"],
            "historia": ap[PRINCIPAL] - ap["sin_historia"],
        },
        "incremental": [
            {"paso": "transaccion", "ap_nuevos": ap["solo_transaccion"]},
            {"paso": "+ sospechosas", "ap_nuevos": ap["transaccion_sospechosas"]},
            {"paso": "+ vecindario", "ap_nuevos": ap["sin_historia"]},
            {"paso": "+ historia", "ap_nuevos": ap[PRINCIPAL]},
        ],
    }
    salida_json = {
        "prueba_dias": list(particion.PRUEBA),
        "cifra_principal": "AP en clientes nuevos",
        "fraccion_tx_cliente_visto": float(visto.mean()),
        "evaluado_en": "cpu",
        "variantes": resultados,
        "aportes_por_grupo_en_nuevos": aportes,
        "curvas_pr": curvas,
    }
    util.escribir_json(salida_json, salida / "evaluacion.json")
    graficos_evaluacion(salida)


NOMBRES = {
    "completo": "Completo (principal)",
    "solo_transaccion": "Solo transacción",
    "transaccion_sospechosas": "Transacción + sospechosas",
    "sin_historia": "Sin historia de fraude",
    "sin_transaccion": "Sin transacción",
    "sin_sospechosas": "Sin sospechosas (C, D, V, identidad)",
    "sin_vecindario": "Sin vecindario",
    "transaccion_vecindario": "Transacción + vecindario",
    "completo_todas_las_etiquetas": "Completo, todas las etiquetas",
}


def graficos_evaluacion(salida: Path = SALIDA) -> None:
    """Gráficos de prueba desde ``evaluacion.json`` (sin volver a medir)."""
    e = json.loads((salida / "evaluacion.json").read_text(encoding="utf-8"))
    filas = [
        {"variante": NOMBRES[n], "clientes": f"clientes {seg}", "ap": r["prueba"][seg]["ap"]}
        for n, r in e["variantes"].items()
        for seg in ("nuevos", "vistos")
    ]
    graficos.guardar(
        graficos.barras_agrupadas(
            pd.DataFrame(filas),
            "ap",
            "variante",
            "clientes",
            titulo="Precisión promedio en prueba: la cifra principal es la de clientes nuevos",
            eje_x="AP",
            formato_x=".3f",
            orden_y=[NOMBRES[v.nombre] for v in VARIANTES],
            orden_grupo=["clientes nuevos", "clientes vistos"],
        ),
        salida / "ap_prueba",
    )
    inc = pd.DataFrame(e["aportes_por_grupo_en_nuevos"]["incremental"])
    graficos.guardar(
        graficos.barras_h(
            inc,
            "ap_nuevos",
            "paso",
            titulo="AP en clientes nuevos al sumar grupos de variables (prueba)",
            eje_x="AP en clientes nuevos",
            formato_x=".3f",
            orden=list(inc["paso"]),
        ),
        salida / "aporte_incremental_nuevos",
    )
    shap = e["variantes"][PRINCIPAL]["shap_prueba_nuevos"]
    tabla = pd.DataFrame(shap["variables"]).head(20)
    graficos.guardar(
        graficos.barras_h(
            tabla,
            "shap_abs_medio",
            "variable",
            color="grupo",
            titulo="Variables con más peso en clientes nuevos (SHAP medio absoluto, prueba)",
            eje_x="|SHAP| medio (log-odds)",
            formato_x=".3f",
            orden=list(tabla["variable"]),
            tooltip=["variable", "grupo", graficos_tooltip("shap_abs_medio")],
        ),
        salida / "shap_nuevos",
    )
    curvas = pd.DataFrame(e["curvas_pr"])
    if not curvas.empty:
        import altair as alt

        curvas = curvas.assign(variante=curvas["variante"].map(NOMBRES))
        g = (
            alt.Chart(curvas)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("recall:Q", title="Recall"),
                y=alt.Y("precision:Q", title="Precisión", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("variante:N", title=None, scale=alt.Scale(range=graficos.PALETA)),
                tooltip=[
                    "variante",
                    "clientes",
                    alt.Tooltip("recall:Q", format=".2f"),
                    alt.Tooltip("precision:Q", format=".3f"),
                ],
            )
            .properties(width=300, height=240)
            .facet(column=alt.Column("clientes:N", title=None))
            .properties(title="Curva precisión-recall en prueba")
        )
        graficos.guardar(g, salida / "curvas_pr_prueba")


def graficos_tooltip(col: str):
    import altair as alt

    return alt.Tooltip(f"{col}:Q", format=".3f")


def publicar_modelo_principal(salida: Path = SALIDA, destino: Path = PUBLICADO) -> None:
    """Copia el modelo principal a ``docs/resultados/hito3/modelos/`` (lo usará el servicio)."""
    p = json.loads((salida / "parametros.json").read_text(encoding="utf-8"))
    nombre = p["variantes"][PRINCIPAL]["modelo"]
    (destino / "modelos").mkdir(parents=True, exist_ok=True)
    shutil.copy2(salida / "modelos" / nombre, destino / "modelos" / nombre)


PASOS: dict[str, Callable[[], None]] = {
    "buscar": buscar,
    "desarrollar": desarrollar,
    "publicar-modelo": publicar_modelo_principal,
    "evaluar": evaluar,
    "graficos": graficos_evaluacion,
}
