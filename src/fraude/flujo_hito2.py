"""Flujo del hito 2: grafo, vecindario y cuánto aportan, medido una vez en prueba.

- ``preparar``: tablas ``entidades`` y ``vecindario`` (requiere ``clientes`` del hito 1).
- ``desarrollar``: con etiquetas solo de desarrollo, entrena cada variante declarada en
  ``VARIANTES`` según su esquema (``particion.ESQUEMAS``), elige la regularización en
  validación y guarda los parámetros en ``reports/hito2/parametros.json``.
- ``evaluar``: se niega a correr si ``docs/resultados/hito2/parametros.json`` no está
  congelado en el tag ``hito2-congelado``; mide cada variante en prueba una sola vez, para
  clientes nuevos y ya vistos en desarrollo.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from fraude import (
    base,
    congelados,
    evaluacion,
    graficos,
    grafo,
    modelo_lineal,
    particion,
    rutas,
    util,
    vecindario,
)

SALIDA = rutas.reports_hito(2)
PARAMETROS_PUBLICADOS = rutas.docs_hito(2) / "parametros.json"
TAG = "hito2-congelado"


@dataclass(frozen=True)
class Variante:
    nombre: str
    tipo: str  # "regla" o "logistica"
    esquema: str
    retraso: int | None  # None: sin variables de etiquetas
    descripcion: str
    columna_regla: str | None = None


VARIANTES: tuple[Variante, ...] = (
    Variante(
        "regla_cliente_fraude_r30",
        "regla",
        "realista",
        30,
        "Referencia simple: fraudes ya conocidos del mismo cliente (retraso de 30 días).",
        "cliente_fraudes_r30",
    ),
    Variante(
        "vecindario_sin_etiquetas",
        "logistica",
        "realista",
        None,
        "Solo estructura y velocidad del vecindario, sin etiquetas.",
    ),
    Variante(
        "vecindario_r30",
        "logistica",
        "realista",
        30,
        "Principal: vecindario con etiquetas conocidas a los 30 días.",
    ),
    Variante("vecindario_r15", "logistica", "realista", 15, "Sensibilidad: retraso de 15 días."),
    Variante("vecindario_r60", "logistica", "realista", 60, "Sensibilidad: retraso de 60 días."),
    Variante("vecindario_r90", "logistica", "realista", 90, "Sensibilidad: retraso de 90 días."),
    Variante(
        "vecindario_r0_filtracion",
        "logistica",
        "realista",
        0,
        "Comparación: etiquetas sin retraso (filtración en producción).",
    ),
    Variante(
        "vecindario_r30_todas_las_etiquetas",
        "logistica",
        "todas_las_etiquetas",
        30,
        "Comparación: entrena y valida con todas las etiquetas de desarrollo.",
    ),
)
PRINCIPAL = "vecindario_r30"


def preparar(con: duckdb.DuckDBPyConnection | None = None) -> None:
    propia = con is None
    con = con or base.conectar()
    grafo.crear_entidades(con)
    vecindario.construir(con)
    if propia:
        con.close()


def cargar(con: duckdb.DuckDBPyConnection, hasta_dia: int) -> pd.DataFrame:
    """Vecindario con ``dia``, ``cliente`` e ``isFraud`` hasta ``hasta_dia``."""
    return con.execute(
        f"""
        SELECT v.*, e.dia, e.cliente, t.isFraud
        FROM vecindario v JOIN entidades e USING (TransactionID)
        JOIN transacciones t USING (TransactionID)
        WHERE e.dia <= {int(hasta_dia)}
        ORDER BY v.TransactionID
        """
    ).df()


def _tramo(df: pd.DataFrame, tramo: tuple[int, int]) -> pd.DataFrame:
    return df[(df["dia"] >= tramo[0]) & (df["dia"] <= tramo[1])]


def visto_antes(df: pd.DataFrame, filas: pd.DataFrame, dia_inicio: int) -> np.ndarray:
    """¿El cliente de cada fila de ``filas`` tuvo alguna transacción antes de ``dia_inicio``?"""
    previos = set(df.loc[df["dia"] < dia_inicio, "cliente"])
    return filas["cliente"].isin(previos).to_numpy()


def columnas_variante(tabla: pd.DataFrame, v: Variante) -> list[str]:
    return vecindario.columnas(tabla, v.retraso)


def _puntaje(v: Variante, params: dict[str, object] | None, df: pd.DataFrame) -> np.ndarray:
    if v.tipo == "regla":
        return df[v.columna_regla].to_numpy(dtype="float64")
    return modelo_lineal.puntuar(modelo_lineal.ModeloLineal.desde_dict(params), df)


def desarrollar(
    con: duckdb.DuckDBPyConnection | None = None,
    salida: Path = SALIDA,
    grilla: tuple[float, ...] = modelo_lineal.GRILLA_C,
) -> None:
    """Entrena y elige en desarrollo; escribe ``parametros.json`` y ``desarrollo.json``."""
    propia = con is None
    con = con or base.conectar(solo_lectura=True)
    df = cargar(con, particion.ULTIMO_DIA_DESARROLLO)
    resumen_grafo = grafo.resumen(con)
    if propia:
        con.close()

    parametros: dict[str, object] = {
        "principal": PRINCIPAL,
        "esquemas": particion.ESQUEMAS,
        "limite_clientes_componente": vecindario.LIMITE_CLIENTES,
        "entidades_componente": list(vecindario.ENTIDADES_COMPONENTE),
        "variantes": {},
    }
    validacion: dict[str, object] = {}
    for v in VARIANTES:
        esquema = particion.ESQUEMAS[v.esquema]
        ent, val, reaj = (
            _tramo(df, esquema[k]) for k in ("entrenamiento", "validacion", "reajuste")
        )
        visto_val = visto_antes(df, val, esquema["validacion"][0])
        entrada: dict[str, object] = {
            "tipo": v.tipo,
            "esquema": v.esquema,
            "retraso": v.retraso,
            "descripcion": v.descripcion,
        }
        if v.tipo == "regla":
            entrada["columna"] = v.columna_regla
            s_val = _puntaje(v, None, val)
            ap_por_c = {}
        else:
            cols = columnas_variante(df, v)
            ap_por_c = {}
            for c in grilla:
                m = modelo_lineal.entrenar(ent, ent["isFraud"].to_numpy(), cols, c)
                ap_por_c[c] = evaluacion.ap(
                    val["isFraud"].to_numpy(), modelo_lineal.puntuar(m, val)
                )
            mejor = max(grilla, key=lambda c: (ap_por_c[c], -c))  # empate: más regularizado
            m_ent = modelo_lineal.entrenar(ent, ent["isFraud"].to_numpy(), cols, mejor)
            s_val = modelo_lineal.puntuar(m_ent, val)
            final = modelo_lineal.entrenar(reaj, reaj["isFraud"].to_numpy(), cols, mejor)
            entrada["modelo"] = final.a_dict()
        parametros["variantes"][v.nombre] = entrada
        validacion[v.nombre] = {
            "ap_por_C": {str(k): x for k, x in ap_por_c.items()},
            "validacion": evaluacion.por_segmento(
                val["isFraud"].to_numpy(), s_val, visto_val, intervalo=False
            ),
        }

    # Aporte de cada variable sola en la validación realista (más alto = más riesgo).
    val_r = _tramo(df, particion.ESQUEMAS["realista"]["validacion"])
    y_val = val_r["isFraud"].to_numpy()
    univariadas = {
        c: evaluacion.ap(y_val, val_r[c].fillna(-1).to_numpy(dtype="float64"))
        for c in vecindario.columnas(df, 30)
    }
    # Prevalencia por tamaño de componente (validación realista).
    tramos = pd.cut(
        val_r["componente_clientes"],
        [0, 1, 2, 5, 20, 100, np.inf],
        labels=["1", "2", "3-5", "6-20", "21-100", "más de 100"],
    )
    por_componente = (
        val_r.groupby(tramos, observed=True)["isFraud"].agg(["size", "mean"]).reset_index()
    )
    con_fraude = pd.cut(
        val_r["componente_clientes_fraude_r30"],
        [-1, 0, 1, 5, np.inf],
        labels=["0", "1", "2-5", "más de 5"],
    )
    por_fraude_componente = (
        val_r.groupby(con_fraude, observed=True)["isFraud"].agg(["size", "mean"]).reset_index()
    )
    desarrollo = {
        "grafo": resumen_grafo,
        "componente_gigante_clientes": int(df["componente_clientes"].max()),
        "validacion_por_variante": validacion,
        "ap_univariada_validacion_realista": dict(
            sorted(univariadas.items(), key=lambda kv: -kv[1])
        ),
        "prevalencia_por_tamano_componente": [
            {
                "clientes": str(r.componente_clientes),
                "transacciones": int(r.size),
                "p_fraude": r.mean,
            }
            for r in por_componente.itertuples()
        ],
        "prevalencia_por_clientes_con_fraude_en_componente_r30": [
            {
                "clientes_con_fraude": str(r.componente_clientes_fraude_r30),
                "transacciones": int(r.size),
                "p_fraude": r.mean,
            }
            for r in por_fraude_componente.itertuples()
        ],
        "nota": "Cifras de validación: se usaron para elegir, son optimistas.",
    }
    util.escribir_json(parametros, salida / "parametros.json")
    util.escribir_json(desarrollo, salida / "desarrollo.json")
    print(f"desarrollo -> {salida}")


def evaluar(
    con: duckdb.DuckDBPyConnection | None = None,
    salida: Path = SALIDA,
    ruta_parametros: Path = PARAMETROS_PUBLICADOS,
    tag: str | None = TAG,
) -> None:
    """Mide cada variante congelada en prueba, una vez. ``tag=None`` solo en pruebas."""
    if tag is not None:
        congelados.verificar(tag, ruta_parametros)
    import json

    parametros = json.loads(Path(ruta_parametros).read_text(encoding="utf-8"))
    propia = con is None
    con = con or base.conectar(solo_lectura=True)
    df = cargar(con, particion.ULTIMO_DIA)
    if propia:
        con.close()
    prueba = _tramo(df, particion.PRUEBA)
    visto = visto_antes(df, prueba, particion.PRUEBA[0])
    y = prueba["isFraud"].to_numpy()
    grupos = prueba["cliente"].to_numpy()
    resultados = {}
    curvas = []
    for v in VARIANTES:
        p = parametros["variantes"][v.nombre]
        s = _puntaje(v, p.get("modelo"), prueba)
        resultados[v.nombre] = {
            "descripcion": v.descripcion,
            "esquema": v.esquema,
            "retraso": v.retraso,
            "prueba": evaluacion.por_segmento(y, s, visto, grupos),
        }
        if v.nombre in (PRINCIPAL, "regla_cliente_fraude_r30", "vecindario_sin_etiquetas"):
            for seg, m in (("nuevos", ~visto), ("vistos", visto)):
                for punto in evaluacion.curva_pr(y[m], s[m], puntos=50):
                    curvas.append({"variante": v.nombre, "clientes": seg, **punto})
    salida_json = {
        "prueba_dias": list(particion.PRUEBA),
        "fraccion_tx_cliente_visto": float(visto.mean()),
        "variantes": resultados,
        "curvas_pr": curvas,
    }
    util.escribir_json(salida_json, salida / "evaluacion.json")
    _graficos_evaluacion(resultados, pd.DataFrame(curvas), salida)
    print(f"evaluación -> {salida}")


NOMBRES = {
    "regla_cliente_fraude_r30": "Regla: fraude conocido del cliente",
    "vecindario_sin_etiquetas": "Vecindario sin etiquetas",
    "vecindario_r30": "Vecindario, retraso 30 (principal)",
    "vecindario_r15": "Vecindario, retraso 15",
    "vecindario_r60": "Vecindario, retraso 60",
    "vecindario_r90": "Vecindario, retraso 90",
    "vecindario_r0_filtracion": "Vecindario, sin retraso (filtración)",
    "vecindario_r30_todas_las_etiquetas": "Vecindario, retraso 30, todas las etiquetas",
}


def _graficos_evaluacion(resultados: dict, curvas: pd.DataFrame, salida: Path) -> None:
    filas = []
    for nombre, r in resultados.items():
        for seg in ("nuevos", "vistos"):
            filas.append(
                {
                    "variante": NOMBRES[nombre],
                    "clientes": f"clientes {seg}",
                    "ap": r["prueba"][seg]["ap"],
                }
            )
    tabla = pd.DataFrame(filas)
    graficos.guardar(
        graficos.barras_agrupadas(
            tabla,
            "ap",
            "variante",
            "clientes",
            titulo="Precisión promedio en prueba (días 121 a 182)",
            eje_x="AP",
            formato_x=".3f",
            orden_y=[NOMBRES[v.nombre] for v in VARIANTES],
            orden_grupo=["clientes nuevos", "clientes vistos"],
        ),
        salida / "ap_prueba",
    )
    if not curvas.empty:
        curvas = curvas.assign(variante=curvas["variante"].map(NOMBRES))
        import altair as alt

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


PASOS: dict[str, Callable[[], None]] = {
    "preparar": preparar,
    "desarrollar": desarrollar,
    "evaluar": evaluar,
}
