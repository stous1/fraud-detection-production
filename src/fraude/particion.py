"""Partición temporal fija, decidida antes de mirar etiquetas.

Los 182 días de train (``dia`` 1 a 182) se dividen así:

- desarrollo = días 1 a 120 (cuatro bloques de 30 días):
  - entrenamiento = días 1 a 90;
  - validación = días 91 a 120 (parada temprana, hiperparámetros y umbral);
- prueba = días 121 a 182: ocho semanas completas y una novena de seis días. Se mide una sola
  vez, tras congelar con un tag, y en el hito 4 llega semana a semana.

Toda decisión que mire etiquetas se toma en desarrollo. De prueba, antes de congelar, solo se
miran volúmenes y faltantes, nunca ``isFraud``.
"""

import duckdb

ULTIMO_DIA_ENTRENAMIENTO = 90
ULTIMO_DIA_DESARROLLO = 120
ULTIMO_DIA = 182
DIAS_SEMANA = 7

PERIODOS = ("entrenamiento", "validacion", "prueba")


def periodo(dia: int) -> str:
    """Periodo de un día."""
    if not 1 <= dia <= ULTIMO_DIA:
        raise ValueError(f"día fuera de rango: {dia}")
    if dia <= ULTIMO_DIA_ENTRENAMIENTO:
        return "entrenamiento"
    return "validacion" if dia <= ULTIMO_DIA_DESARROLLO else "prueba"


def semana_prueba(dia: int) -> int | None:
    """Semana de prueba (1 a 9) de un día, o ``None`` si es de desarrollo."""
    if dia <= ULTIMO_DIA_DESARROLLO:
        return None
    return (dia - ULTIMO_DIA_DESARROLLO - 1) // DIAS_SEMANA + 1


def mes(dia: int) -> int:
    """Bloque de 30 días (1 a 7; el séptimo tiene dos días)."""
    return (dia - 1) // 30 + 1


SQL_PERIODO = (
    f"CASE WHEN dia <= {ULTIMO_DIA_ENTRENAMIENTO} THEN 'entrenamiento' "
    f"WHEN dia <= {ULTIMO_DIA_DESARROLLO} THEN 'validacion' ELSE 'prueba' END"
)
SQL_SEMANA_PRUEBA = (
    f"CASE WHEN dia > {ULTIMO_DIA_DESARROLLO} "
    f"THEN (dia - {ULTIMO_DIA_DESARROLLO} - 1) // {DIAS_SEMANA} + 1 END"
)


def crear_vistas(con: duckdb.DuckDBPyConnection) -> None:
    """Vistas ``tiempo`` (periodo y semana de prueba de cada transacción) y ``desarrollo``.

    ``desarrollo`` es la única vista con etiquetas que usan la exploración y los
    diagnósticos del hito 1.
    """
    con.execute(
        f"""
        CREATE OR REPLACE VIEW tiempo AS
        SELECT TransactionID, dia, {SQL_PERIODO} AS periodo,
               {SQL_SEMANA_PRUEBA} AS semana_prueba, (dia - 1) // 30 + 1 AS mes
        FROM transacciones
        """
    )
    con.execute(
        f"CREATE OR REPLACE VIEW desarrollo AS "
        f"SELECT * FROM transacciones WHERE dia <= {ULTIMO_DIA_DESARROLLO}"
    )


def resumen(con: duckdb.DuckDBPyConnection) -> list[dict[str, object]]:
    """Volumen por periodo (sin etiquetas)."""
    filas = con.execute(
        "SELECT periodo, min(dia), max(dia), count(*) FROM tiempo GROUP BY 1 ORDER BY 2"
    ).fetchall()
    return [
        {"periodo": p, "dia_inicio": a, "dia_fin": b, "transacciones": n} for p, a, b, n in filas
    ]


# --- Esquemas de entrenamiento con retraso de etiquetas (decididos en el hito 2) -----------
#
# El modelo entra a producción el día 121. Con un retraso de contracargo de 30 días, ese día
# solo se conocen las etiquetas de las transacciones hasta el día 90: el esquema principal
# ("realista") entrena, valida y elige umbral solo con ellas. Los días 91 a 120 sirven como
# historia sin etiqueta. "todas_las_etiquetas" usa todo desarrollo y se reporta solo como
# comparación, para medir cuánto infla.

RETRASO_BASE = 30
ULTIMO_DIA_ETIQUETADO = ULTIMO_DIA_DESARROLLO - RETRASO_BASE  # 90

ESQUEMAS: dict[str, dict[str, tuple[int, int]]] = {
    "realista": {
        "entrenamiento": (1, 60),
        "validacion": (61, ULTIMO_DIA_ETIQUETADO),
        "reajuste": (1, ULTIMO_DIA_ETIQUETADO),
    },
    "todas_las_etiquetas": {
        "entrenamiento": (1, ULTIMO_DIA_ENTRENAMIENTO),
        "validacion": (ULTIMO_DIA_ENTRENAMIENTO + 1, ULTIMO_DIA_DESARROLLO),
        "reajuste": (1, ULTIMO_DIA_DESARROLLO),
    },
}
ESQUEMA_PRINCIPAL = "realista"
PRUEBA: tuple[int, int] = (ULTIMO_DIA_DESARROLLO + 1, ULTIMO_DIA)
