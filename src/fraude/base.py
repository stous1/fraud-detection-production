"""Conexión a DuckDB con los límites del proyecto."""

from pathlib import Path

import duckdb

from fraude.rutas import BASE_DUCKDB, DATA_TMP

MEMORIA = "24GB"


def conectar(
    ruta: Path | str = BASE_DUCKDB,
    solo_lectura: bool = False,
    memoria: str = MEMORIA,
    temporal: Path = DATA_TMP,
) -> duckdb.DuckDBPyConnection:
    """Conexión con memoria limitada, desborde a disco y zona horaria UTC.

    ``ruta=":memory:"`` abre una base en memoria (pruebas).
    """
    if str(ruta) != ":memory:":
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    temporal.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(ruta), read_only=solo_lectura)
    con.execute(f"SET memory_limit = '{memoria}'")
    con.execute(f"SET temp_directory = '{temporal.as_posix()}'")
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET TimeZone = 'UTC'")
    con.execute("SET enable_progress_bar = false")
    return con
