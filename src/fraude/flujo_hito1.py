"""Flujo del hito 1: preparar, explorar, clientes y sospechas."""

from collections.abc import Callable
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from fraude import (
    base,
    clientes,
    exploracion,
    graficos,
    ingesta,
    particion,
    rutas,
    sospechas,
    splink_ref,
    util,
)

SALIDA = rutas.reports_hito(1)
UMBRALES_SPLINK = (0.5, 0.9, 0.99)
NOMBRES_NIVEL = {
    "tarjeta": "card1",
    "tarjeta_direccion": "card1 + addr1",
    "uid": "uid (card1 + addr1 + D1n)",
    "uid_tolerancia_1": "uid con ±1 día",
    "uid_veto_correo": "uid con veto de correo",
    "splink": "Splink (umbral 0,9)",
}


def preparar() -> None:
    """CSV de train -> ``data/fraude.duckdb``, con verificaciones y huellas."""
    con = base.conectar()
    ingesta.cargar(con, rutas.CSV_TRANSACCIONES, rutas.CSV_IDENTIDAD)
    particion.crear_vistas(con)
    info = ingesta.verificar(con)
    info["huellas_sha256"] = {
        r.name: ingesta.huella(r) for r in (rutas.CSV_TRANSACCIONES, rutas.CSV_IDENTIDAD)
    }
    info["particion"] = particion.resumen(con)
    util.escribir_json(info, SALIDA / "carga.json")
    con.close()
    print(info)


def _registros(df: pd.DataFrame) -> list[dict[str, object]]:
    return df.to_dict(orient="records")


def explorar(con: duckdb.DuckDBPyConnection | None = None, salida: Path = SALIDA) -> None:
    """Exploración (etiquetas solo de desarrollo) -> ``exploracion.json`` y gráficos."""
    propia = con is None
    con = con or base.conectar(solo_lectura=True)
    sem = exploracion.semanal(con)
    producto = exploracion.por_categoria(con, "ProductCD")
    identidad = exploracion.por_categoria(
        con, "CASE WHEN i.TransactionID IS NULL THEN 'sin identidad' ELSE 'con identidad' END"
    )
    dispositivo = exploracion.por_categoria(con, "i.DeviceType")
    tipo_tarjeta = exploracion.por_categoria(con, "card6")
    marca = exploracion.por_categoria(con, "card4")
    correo = exploracion.por_categoria(con, "P_emaildomain", minimo=2000)
    dec = exploracion.decimales(con)
    falt = exploracion.faltantes(con)
    datos = {
        "desarrollo": exploracion.resumen_desarrollo(con),
        "semanal": _registros(sem),
        "producto": _registros(producto),
        "identidad": _registros(identidad),
        "tipo_dispositivo": _registros(dispositivo),
        "tipo_tarjeta": _registros(tipo_tarjeta),
        "marca_tarjeta": _registros(marca),
        "correo_comprador": _registros(correo),
        "decimales_por_producto": _registros(dec),
        "montos": _registros(exploracion.montos(con)),
        "faltantes": _registros(falt),
        "bloques_v": exploracion.bloques_v(con),
    }
    if propia:
        con.close()
    util.escribir_json(datos, salida / "exploracion.json")

    g = graficos
    g.guardar(
        g.barras_v(
            sem,
            "semana",
            "transacciones",
            color="periodo",
            orden_color=["desarrollo", "mixta", "prueba"],
            titulo="Transacciones por semana (la semana 18 se reparte entre desarrollo y prueba)",
            eje_x="Semana",
            eje_y="Transacciones",
            formato_y=",.0f",
        ),
        salida / "volumen_semanal",
    )
    g.guardar(
        g.lineas(
            sem.dropna(subset=["prevalencia_desarrollo"]),
            "semana",
            "prevalencia_desarrollo",
            titulo="Prevalencia de fraude por semana (solo desarrollo)",
            eje_x="Semana",
            eje_y="Fraude",
            formato_y=".1%",
        ),
        salida / "prevalencia_semanal",
    )
    g.guardar(
        g.barras_h(
            producto,
            "prevalencia",
            "categoria",
            titulo="Prevalencia por producto (desarrollo)",
            eje_x="Fraude",
            formato_x=".1%",
            tooltip=["categoria", "transacciones", alt_fmt("prevalencia")],
        ),
        salida / "prevalencia_producto",
    )
    dec_w_c = dec[dec["transacciones"] >= 1000].assign(
        etiqueta=lambda d: d["producto"] + " · " + d["decimales"]
    )
    g.guardar(
        g.barras_h(
            dec_w_c,
            "prevalencia",
            "etiqueta",
            titulo="Prevalencia según los decimales del monto (desarrollo)",
            eje_x="Fraude",
            formato_x=".1%",
            orden=list(dec_w_c["etiqueta"]),
            tooltip=["etiqueta", "transacciones", alt_fmt("prevalencia")],
        ),
        salida / "prevalencia_decimales",
    )
    g.guardar(
        g.barras_agrupadas(
            falt,
            "faltante",
            "grupo",
            "periodo",
            titulo="Celdas faltantes por grupo de columnas",
            eje_x="Faltante",
            formato_x=".0%",
            orden_y=list(dict.fromkeys(falt["grupo"])),
            orden_grupo=["desarrollo", "prueba"],
        ),
        salida / "faltantes",
    )
    print(f"exploración -> {salida}")


def alt_fmt(col: str, formato: str = ".2%"):
    import altair as alt

    return alt.Tooltip(f"{col}:Q", format=formato)


def guardar_clientes(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, uid: np.ndarray) -> None:
    """Tabla ``clientes`` (``TransactionID``, ``uid``) para los hitos siguientes."""
    tabla = pd.DataFrame({"TransactionID": df["TransactionID"].to_numpy(), "uid": uid})
    con.register("tabla_clientes", tabla)
    con.execute("CREATE OR REPLACE TABLE clientes AS SELECT * FROM tabla_clientes")
    con.unregister("tabla_clientes")


def paso_clientes(
    con: duckdb.DuckDBPyConnection | None = None,
    salida: Path = SALIDA,
    umbrales_splink: tuple[float, ...] = UMBRALES_SPLINK,
) -> None:
    """Niveles de reconstrucción, variantes descartadas, Splink y clientes nuevos en prueba.

    No lee etiquetas. Deja la tabla ``clientes`` en la base.
    """
    propia = con is None
    con = con or base.conectar()
    df = clientes.cargar_base(con)
    ids = {nivel: clientes.asignar(df, nivel) for nivel in clientes.NIVELES}
    uid = ids[clientes.NIVEL_PRINCIPAL]
    guardar_clientes(con, df, uid)

    validacion = ("P_emaildomain", "D10n", "D15n", "DeviceInfo")
    refs = clientes.referencias(df, validacion)
    niveles = {}
    for nivel, v in ids.items():
        niveles[nivel] = {
            "fragmentacion": clientes.fragmentacion(v),
            "concordancia": clientes.concordancia(df, v),
        }
    variantes = {
        "uid_tolerancia_1": clientes.cambios(df, uid, ids["uid_tolerancia_1"], refs),
        "uid_veto_correo": clientes.cambios(df, uid, ids["uid_veto_correo"], refs),
    }

    splink = []
    ids_splink_09 = None
    for umbral in umbrales_splink:
        s_ids, info = splink_ref.resolver(df, umbral=umbral)
        if umbral == splink_ref.UMBRAL:
            ids_splink_09 = s_ids
        splink.append(
            {
                **{k: v for k, v in info.items() if k != "parametros"},
                "fragmentacion": clientes.fragmentacion(s_ids),
                "acuerdo_con_uid": splink_ref.comparar_pares(uid, s_ids),
                "concordancia": clientes.concordancia(df, s_ids, validacion),
                "parametros": info["parametros"],
            }
        )

    # Clientes de prueba ya vistos en desarrollo (sin etiquetas).
    vistos_dev = set(uid[df["dia"].to_numpy() <= particion.ULTIMO_DIA_DESARROLLO])
    prueba = df["dia"] > particion.ULTIMO_DIA_DESARROLLO
    semana = df.loc[prueba, "dia"].map(particion.semana_prueba)
    visto = pd.Series(uid[prueba.to_numpy()], index=semana.index).isin(vistos_dev)
    por_semana = (
        pd.DataFrame({"semana_prueba": semana, "visto": visto})
        .groupby("semana_prueba")["visto"]
        .agg(["size", "mean"])
        .reset_index()
        .rename(columns={"size": "transacciones", "mean": "fraccion_cliente_visto"})
    )
    nuevos = {
        "fraccion_tx_prueba_cliente_visto_en_desarrollo": float(visto.mean()),
        "por_semana": _registros(por_semana),
    }

    datos = {
        "atributos_validacion": list(validacion),
        "referencias": refs,
        "niveles": niveles,
        "variantes_descartadas": variantes,
        "splink": splink,
        "clientes_en_prueba": nuevos,
    }
    util.escribir_json(datos, salida / "clientes.json")
    if propia:
        con.close()

    filas = []
    todos = {**ids, "splink": ids_splink_09}
    for nivel, v in todos.items():
        c = clientes.concordancia(df, v, validacion)
        for atributo in validacion:
            filas.append(
                {
                    "nivel": NOMBRES_NIVEL[nivel],
                    "atributo": atributo,
                    "concordancia": c[atributo]["concordancia"],
                }
            )
    tabla = pd.DataFrame(filas)
    graficos.guardar(
        graficos.barras_agrupadas(
            tabla,
            "concordancia",
            "nivel",
            "atributo",
            titulo="Concordancia entre transacciones consecutivas del mismo cliente",
            eje_x="Pares que coinciden",
            formato_x=".0%",
            orden_y=[NOMBRES_NIVEL[n] for n in todos],
            orden_grupo=list(validacion),
        ),
        salida / "clientes_concordancia",
    )
    graficos.guardar(
        graficos.barras_v(
            por_semana,
            "semana_prueba",
            "fraccion_cliente_visto",
            titulo="Transacciones de prueba cuyo cliente (uid) ya apareció en desarrollo",
            eje_x="Semana de prueba",
            eje_y="Cliente ya visto",
            formato_y=".0%",
        ),
        salida / "clientes_vistos_prueba",
    )
    print(f"clientes -> {salida}")


def paso_sospechas(con: duckdb.DuckDBPyConnection | None = None, salida: Path = SALIDA) -> None:
    """Propagación (etiquetas de desarrollo), monotonía y prueba de futuro (sin etiquetas)."""
    propia = con is None
    con = con or base.conectar(solo_lectura=True)
    numericas = (
        [f"C{i}" for i in range(1, 15)]
        + [f"D{i}" for i in range(1, 16)]
        + [f"V{i}" for i in range(1, 340)]
        + ["TransactionAmt", "dist1", "dist2"]
    )
    columnas = ", ".join(f"t.{c}" for c in numericas)
    todo = con.execute(
        f"""
        SELECT t.TransactionID, t.TransactionDT, t.dia, t.ProductCD, c.uid, {columnas}
        FROM transacciones t JOIN clientes c USING (TransactionID)
        ORDER BY t.TransactionDT, t.TransactionID
        """
    ).df()
    dev = con.execute(
        """
        SELECT d.TransactionID, d.TransactionDT, d.isFraud, c.uid
        FROM desarrollo d JOIN clientes c USING (TransactionID)
        ORDER BY d.TransactionDT, d.TransactionID
        """
    ).df()
    if propia:
        con.close()

    uid = todo["uid"].to_numpy()
    prop = sospechas.propagacion(dev, dev["uid"].to_numpy())
    mono = sospechas.monotonia(
        todo, uid, [f"C{i}" for i in range(1, 15)] + [f"D{i}" for i in range(1, 16)]
    )
    primeras = sospechas.primeras_de_tarjetas_nuevas(todo, uid)
    rho = sospechas.prueba_futuro(primeras, [c for c in numericas if c != "D1"])
    datos = {
        "propagacion_desarrollo": prop,
        "retraso_base_dias": sospechas.RETRASO_BASE,
        "monotonia": mono,
        "prueba_futuro": {
            "tarjetas_nuevas": int(len(primeras)),
            "fraccion_con_actividad_futura": float((primeras["futuras"] > 0).mean()),
            "resumen": sospechas.resumen_futuro(rho),
            "rho": rho,
        },
        "grupos": [
            {
                "nombre": g.nombre,
                "columnas": len(g.columnas),
                "sospechosa": g.sospechosa,
                "motivo": g.motivo,
            }
            for g in sospechas.GRUPOS
        ],
    }
    util.escribir_json(datos, salida / "sospechas.json")

    tramos = pd.DataFrame(prop["por_dias_desde_primer_fraude"])
    graficos.guardar(
        graficos.barras_h(
            tramos,
            "p_fraude",
            "dias_desde_primer_fraude",
            titulo="Fraude según los días desde el primer fraude del mismo cliente (desarrollo)",
            eje_x="Transacciones marcadas como fraude",
            formato_x=".0%",
            orden=list(tramos["dias_desde_primer_fraude"]),
            tooltip=["dias_desde_primer_fraude", "transacciones", alt_fmt("p_fraude", ".1%")],
        ),
        salida / "propagacion_tramos",
    )
    regla = pd.DataFrame(prop["regla_fraude_previo"]).assign(
        retraso=lambda d: d["retraso_dias"].astype(str) + " días"
    )
    graficos.guardar(
        graficos.barras_h(
            regla,
            "fraccion_de_fraudes_cubiertos",
            "retraso",
            titulo="Fraudes cubiertos por la regla «el cliente ya tuvo un fraude conocido»",
            eje_x="Fracción de los fraudes de desarrollo",
            formato_x=".1%",
            orden=list(regla["retraso"]),
            tooltip=[
                "retraso",
                "transacciones_marcadas",
                alt_fmt("precision", ".1%"),
                alt_fmt("fraccion_de_fraudes_cubiertos", ".1%"),
            ],
        ),
        salida / "propagacion_retraso",
    )
    top = (
        pd.Series(rho, name="rho")
        .rename_axis("columna")
        .reset_index()
        .assign(abs_rho=lambda d: d["rho"].abs())
        .sort_values("abs_rho", ascending=False)
        .head(15)
    )
    graficos.guardar(
        graficos.barras_h(
            top,
            "rho",
            "columna",
            titulo="Correlación con la actividad futura en la primera transacción de una tarjeta",
            eje_x="ρ de Spearman con las transacciones de los 30 días siguientes",
            formato_x=".3f",
            orden=list(top["columna"]),
        ),
        salida / "prueba_futuro",
    )
    print(f"sospechas -> {salida}")


def todo() -> None:
    explorar()
    paso_clientes()
    paso_sospechas()


PASOS: dict[str, Callable[[], None]] = {
    "preparar": preparar,
    "explorar": explorar,
    "clientes": paso_clientes,
    "sospechas": paso_sospechas,
    "todo": todo,
}
