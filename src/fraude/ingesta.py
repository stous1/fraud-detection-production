"""Ingesta de los CSV de train de IEEE-CIS a DuckDB, con tipos explícitos y verificaciones.

Solo se cargan ``train_transaction.csv`` y ``train_identity.csv``. Los archivos de prueba de
la competencia no traen etiqueta y **no se leen**: ``verificar_origen`` los rechaza.
"""

import hashlib
from pathlib import Path

import duckdb

from fraude import esquema

SEGUNDOS_DIA = 86_400
PROHIBIDOS = ("test_", "sample_submission")


def verificar_origen(ruta: Path) -> None:
    """Falla si ``ruta`` es un archivo de prueba de la competencia (sin etiqueta)."""
    nombre = Path(ruta).name.lower()
    if any(nombre.startswith(p) for p in PROHIBIDOS):
        raise ValueError(f"{nombre}: los archivos de prueba de la competencia no se usan")


def huella(ruta: Path, bloque: int = 1 << 20) -> str:
    """SHA-256 del archivo."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        while datos := f.read(bloque):
            h.update(datos)
    return h.hexdigest()


def _leer_csv(ruta: Path, tipos: dict[str, str]) -> str:
    columnas = ", ".join(f"'{c}': '{t}'" for c, t in tipos.items())
    return (
        f"read_csv('{Path(ruta).as_posix()}', header = true, auto_detect = false, "
        f"columns = {{{columnas}}}, nullstr = '')"
    )


def cargar(con: duckdb.DuckDBPyConnection, csv_tx: Path, csv_id: Path) -> None:
    """Crea ``transacciones`` (con ``dia`` y ``hora`` derivados) e ``identidad``.

    ``dia = TransactionDT // 86400``; la fecha de calendario es desconocida y no se inventa.
    Las tablas quedan ordenadas por tiempo y ``TransactionID``.
    """
    for ruta in (csv_tx, csv_id):
        verificar_origen(ruta)
    tipos_tx = {c: esquema.tipo_tx(c) for c in esquema.COLUMNAS_TX}
    tipos_id = {c: esquema.tipo_id(c) for c in esquema.COLUMNAS_ID}
    con.execute(
        f"""
        CREATE OR REPLACE TABLE transacciones AS
        SELECT *,
               (TransactionDT // {SEGUNDOS_DIA})::INTEGER AS dia,
               ((TransactionDT % {SEGUNDOS_DIA}) // 3600)::TINYINT AS hora
        FROM {_leer_csv(csv_tx, tipos_tx)}
        ORDER BY TransactionDT, TransactionID
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE identidad AS
        SELECT * FROM {_leer_csv(csv_id, tipos_id)} ORDER BY TransactionID
        """
    )


def verificar(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    """Comprobaciones de integridad; lanza ``AssertionError`` si alguna falla."""
    n_tx, n_unicos, etiquetas_validas = con.execute(
        "SELECT count(*), count(DISTINCT TransactionID), "
        "bool_and(isFraud IN (0, 1)) FROM transacciones"
    ).fetchone()
    assert n_tx == n_unicos, "TransactionID repetido en transacciones"
    assert etiquetas_validas, "isFraud fuera de {0, 1}"
    n_id, n_id_unicos, huerfanos = con.execute(
        "SELECT count(*), count(DISTINCT i.TransactionID), count(*) - count(t.TransactionID) "
        "FROM identidad i LEFT JOIN transacciones t USING (TransactionID)"
    ).fetchone()
    assert n_id == n_id_unicos, "TransactionID repetido en identidad"
    assert huerfanos == 0, "identidades sin transacción"
    # TransactionID crece con el tiempo: el orden por identificador es el orden temporal.
    desordenados = con.execute(
        """
        SELECT count(*) FROM (
            SELECT TransactionDT < lag(TransactionDT) OVER (ORDER BY TransactionID) AS atras
            FROM transacciones
        ) WHERE atras
        """
    ).fetchone()[0]
    dia_min, dia_max, n_dias = con.execute(
        "SELECT min(dia), max(dia), count(DISTINCT dia) FROM transacciones"
    ).fetchone()
    return {
        "transacciones": n_tx,
        "identidades": n_id,
        "cobertura_identidad": n_id / n_tx,
        "id_desordenados_en_tiempo": desordenados,
        "dia_min": dia_min,
        "dia_max": dia_max,
        "dias_con_datos": n_dias,
    }
