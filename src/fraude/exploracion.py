"""Exploración del hito 1.

Las funciones con etiquetas leen solo la vista ``desarrollo`` (días 1 a 120). Las que miran
prueba (volumen y faltantes) no tocan ``isFraud``.
"""

import duckdb
import pandas as pd

from fraude import esquema

GRUPOS_COLUMNAS: dict[str, tuple[str, ...]] = {
    "card": tuple(f"card{i}" for i in range(1, 7)),
    "addr_dist": ("addr1", "addr2", "dist1", "dist2"),
    "correo": ("P_emaildomain", "R_emaildomain"),
    "C": tuple(f"C{i}" for i in range(1, 15)),
    "D": tuple(f"D{i}" for i in range(1, 16)),
    "M": tuple(f"M{i}" for i in range(1, 10)),
    "V": tuple(f"V{i}" for i in range(1, 340)),
}

SQL_DECIMALES = (
    "CASE WHEN abs(TransactionAmt - round(TransactionAmt)) < 1e-9 THEN 'entero' "
    "WHEN abs(TransactionAmt * 100 - round(TransactionAmt * 100)) < 1e-6 THEN 'centavos' "
    "ELSE '3 o más decimales' END"
)


def _df(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


def semanal(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Volumen por semana (días 1-7 = semana 1) en todo train y prevalencia solo en
    desarrollo."""
    return _df(
        con,
        """
        SELECT (t.dia - 1) // 7 + 1 AS semana, min(t.dia) AS dia_inicio, max(t.dia) AS dia_fin,
               count(*) AS transacciones,
               CASE WHEN max(t.dia) <= 120 THEN 'desarrollo'
                    WHEN min(t.dia) > 120 THEN 'prueba' ELSE 'mixta' END AS periodo,
               avg(d.isFraud) AS prevalencia_desarrollo
        FROM transacciones t LEFT JOIN desarrollo d USING (TransactionID)
        GROUP BY 1 ORDER BY 1
        """,
    )


def por_categoria(con: duckdb.DuckDBPyConnection, expresion: str, minimo: int = 0) -> pd.DataFrame:
    """Transacciones, fraudes y prevalencia por ``expresion`` (en desarrollo)."""
    return _df(
        con,
        f"""
        SELECT coalesce(CAST({expresion} AS VARCHAR), '(faltante)') AS categoria,
               count(*) AS transacciones, sum(isFraud) AS fraudes, avg(isFraud) AS prevalencia
        FROM desarrollo d LEFT JOIN identidad i USING (TransactionID)
        GROUP BY 1 HAVING count(*) >= {minimo} ORDER BY 2 DESC
        """,
    )


def decimales(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Prevalencia por producto y número de decimales del monto (en desarrollo)."""
    return _df(
        con,
        f"""
        SELECT ProductCD AS producto, {SQL_DECIMALES} AS decimales,
               count(*) AS transacciones, avg(isFraud) AS prevalencia
        FROM desarrollo GROUP BY 1, 2 ORDER BY 1, 2
        """,
    )


def montos(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Cuantiles del monto según la etiqueta (en desarrollo)."""
    return _df(
        con,
        """
        SELECT isFraud AS fraude, count(*) AS transacciones,
               quantile_cont(TransactionAmt, 0.1) AS p10,
               quantile_cont(TransactionAmt, 0.5) AS p50,
               quantile_cont(TransactionAmt, 0.9) AS p90,
               quantile_cont(TransactionAmt, 0.99) AS p99
        FROM desarrollo GROUP BY 1 ORDER BY 1
        """,
    )


def faltantes(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Fracción de celdas faltantes por grupo de columnas, en desarrollo y en prueba
    (sin etiquetas)."""
    filas = []
    for grupo, cols in GRUPOS_COLUMNAS.items():
        suma = " + ".join(f'("{c}" IS NULL)::INT' for c in cols)
        res = con.execute(
            f"""
            SELECT dia > 120 AS es_prueba, avg(({suma}) / {len(cols)}) AS faltante
            FROM transacciones GROUP BY 1 ORDER BY 1
            """
        ).fetchall()
        for es_prueba, f in res:
            filas.append(
                {"grupo": grupo, "periodo": "prueba" if es_prueba else "desarrollo", "faltante": f}
            )
    cols_id = [c for c in esquema.COLUMNAS_ID if c != "TransactionID"]
    suma = " + ".join(f'("{c}" IS NULL)::INT' for c in cols_id)
    res = con.execute(
        f"""
        SELECT t.dia > 120, avg(CASE WHEN i.TransactionID IS NULL THEN 1.0
                                     ELSE ({suma}) / {len(cols_id)} END)
        FROM transacciones t LEFT JOIN identidad i USING (TransactionID) GROUP BY 1 ORDER BY 1
        """
    ).fetchall()
    for es_prueba, f in res:
        filas.append(
            {
                "grupo": "identidad",
                "periodo": "prueba" if es_prueba else "desarrollo",
                "faltante": f,
            }
        )
    return pd.DataFrame(filas)


def bloques_v(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    """Las V vienen en bloques que faltan juntos: columnas agrupadas por su número de
    faltantes en todo train (sin etiquetas)."""
    cols = GRUPOS_COLUMNAS["V"]
    conteos = con.execute(
        "SELECT " + ", ".join(f'count(*) - count("{c}")' for c in cols) + " FROM transacciones"
    ).fetchone()
    n = con.execute("SELECT count(*) FROM transacciones").fetchone()[0]
    bloques: dict[int, list[str]] = {}
    for c, k in zip(cols, conteos, strict=True):
        bloques.setdefault(int(k), []).append(c)
    return {
        "bloques": len(bloques),
        "detalle": [
            {"faltante": k / n, "columnas": len(v), "primera": v[0], "ultima": v[-1]}
            for k, v in sorted(bloques.items())
        ],
    }


def resumen_desarrollo(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    n, fraudes, prev = con.execute(
        "SELECT count(*), sum(isFraud), avg(isFraud) FROM desarrollo"
    ).fetchone()
    por_mes = con.execute(
        "SELECT (dia - 1) // 30 + 1, count(*), avg(isFraud) FROM desarrollo GROUP BY 1 ORDER BY 1"
    ).fetchall()
    return {
        "transacciones": n,
        "fraudes": int(fraudes),
        "prevalencia": prev,
        "por_bloque_30_dias": [
            {"bloque": b, "transacciones": t, "prevalencia": p} for b, t, p in por_mes
        ],
    }
