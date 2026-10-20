"""Esquema explícito de los CSV de IEEE-CIS (train).

Se fija en código, no se infiere: el mismo esquema sirve de contrato para el servicio de
puntuación. Los tipos salieron de revisar cada columna completa: toda columna que no es texto
se lee como ``DOUBLE`` salvo los identificadores, la etiqueta, el tiempo y los códigos de
tarjeta y dirección, que son enteros.
"""

TEXTO_TX: tuple[str, ...] = (
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    *(f"M{i}" for i in range(1, 10)),
)
ENTEROS_TX: dict[str, str] = {
    "TransactionID": "BIGINT",
    "isFraud": "TINYINT",
    "TransactionDT": "BIGINT",
    "card1": "INTEGER",
    "card2": "INTEGER",
    "card3": "INTEGER",
    "card5": "INTEGER",
    "addr1": "INTEGER",
    "addr2": "INTEGER",
}
NUMERICAS_TX: tuple[str, ...] = (
    "TransactionAmt",
    "dist1",
    "dist2",
    *(f"C{i}" for i in range(1, 15)),
    *(f"D{i}" for i in range(1, 16)),
    *(f"V{i}" for i in range(1, 340)),
)

TEXTO_ID: tuple[str, ...] = (
    "id_12",
    "id_15",
    "id_16",
    "id_23",
    *(f"id_{i}" for i in range(27, 32)),
    *(f"id_{i}" for i in range(33, 39)),
    "DeviceType",
    "DeviceInfo",
)
NUMERICAS_ID: tuple[str, ...] = tuple(
    f"id_{i:02d}" for i in range(1, 39) if f"id_{i:02d}" not in TEXTO_ID
)

COLUMNAS_TX: tuple[str, ...] = (
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    *(f"card{i}" for i in range(1, 7)),
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "P_emaildomain",
    "R_emaildomain",
    *(f"C{i}" for i in range(1, 15)),
    *(f"D{i}" for i in range(1, 16)),
    *(f"M{i}" for i in range(1, 10)),
    *(f"V{i}" for i in range(1, 340)),
)
COLUMNAS_ID: tuple[str, ...] = (
    "TransactionID",
    *(f"id_{i:02d}" for i in range(1, 39)),
    "DeviceType",
    "DeviceInfo",
)


def tipo_tx(columna: str) -> str:
    """Tipo de DuckDB de una columna de transacciones."""
    if columna in ENTEROS_TX:
        return ENTEROS_TX[columna]
    return "VARCHAR" if columna in TEXTO_TX else "DOUBLE"


def tipo_id(columna: str) -> str:
    """Tipo de DuckDB de una columna de identidad."""
    if columna == "TransactionID":
        return "BIGINT"
    return "VARCHAR" if columna in TEXTO_ID else "DOUBLE"
