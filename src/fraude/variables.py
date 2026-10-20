"""Matriz de variables del hito 3 y sus grupos para las ablaciones.

Grupos (``GRUPOS``):

- ``transaccion``: datos de la transacción al pagar (monto, producto, tarjeta, dirección,
  distancias, dominios de correo, M), hora, día de la semana y decimales del monto.
- ``sospechosas``: C, D, V e identidad (opacas o dependientes de cómo se recolectaron; ver
  ``sospechas.GRUPOS``).
- ``vecindario``: variables del grafo del hito 2 con retraso de 30 días, sin la historia de
  fraude del propio cliente.
- ``historia``: fraudes ya conocidos del propio cliente (retraso de 30 días). En IEEE-CIS
  está inflada por la propagación de etiquetas.

Nunca entran ``dia`` ni ``TransactionDT`` (el modelo no debe aprender la fecha), ni la
etiqueta, ni ``D1n`` (el día de alta absoluto identificaría al cliente).

Las categorías se codifican con ``Codificador``: código = rango de frecuencia en el periodo de
ajuste (0 = la más frecuente); un valor nuevo recibe el código siguiente al último; un
faltante queda como NaN.
"""

from dataclasses import dataclass, field

import duckdb
import numpy as np
import pandas as pd

from fraude import esquema, vecindario

RETRASO = 30
HISTORIA: tuple[str, ...] = (
    f"cliente_etiquetadas_r{RETRASO}",
    f"cliente_fraudes_r{RETRASO}",
    f"cliente_clientes_fraude_r{RETRASO}",
)
TRANSACCION: tuple[str, ...] = (
    "TransactionAmt",
    "ProductCD",
    *(f"card{i}" for i in range(1, 7)),
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "P_emaildomain",
    "R_emaildomain",
    *(f"M{i}" for i in range(1, 10)),
    "hora",
    "dia_semana",
    "decimales",
)
SOSPECHOSAS: tuple[str, ...] = (
    *(f"C{i}" for i in range(1, 15)),
    *(f"D{i}" for i in range(1, 16)),
    *(f"V{i}" for i in range(1, 340)),
    *(c for c in esquema.COLUMNAS_ID if c != "TransactionID"),
)
CATEGORICAS: tuple[str, ...] = tuple(
    c for c in (*esquema.TEXTO_TX, *esquema.TEXTO_ID) if c in (*TRANSACCION, *SOSPECHOSAS)
)


SUBGRUPOS_SOSPECHOSAS: dict[str, tuple[str, ...]] = {
    "C": tuple(f"C{i}" for i in range(1, 15)),
    "D": tuple(f"D{i}" for i in range(1, 16)),
    "V": tuple(f"V{i}" for i in range(1, 340)),
    "identidad": tuple(c for c in esquema.COLUMNAS_ID if c != "TransactionID"),
}
PRINCIPALES: tuple[str, ...] = ("transaccion", "sospechosas", "vecindario", "historia")


def grupos(columnas_vecindario: list[str]) -> dict[str, tuple[str, ...]]:
    """Grupos de variables; ``columnas_vecindario`` = columnas de la tabla ``vecindario``.

    Incluye los cuatro grupos principales (``PRINCIPALES``) y, además, los subgrupos de las
    sospechosas (C, D, V, identidad), que se solapan con ``sospechosas``.
    """
    vec = [c for c in vecindario.columnas(columnas_vecindario, RETRASO) if c not in HISTORIA]
    return {
        "transaccion": TRANSACCION,
        "sospechosas": SOSPECHOSAS,
        "vecindario": tuple(vec),
        "historia": HISTORIA,
        **SUBGRUPOS_SOSPECHOSAS,
    }


def cargar(con: duckdb.DuckDBPyConnection, hasta_dia: int) -> pd.DataFrame:
    """Transacciones con identidad, vecindario, cliente y etiqueta, hasta ``hasta_dia``."""
    cols_vec = [
        c
        for c in con.execute("SELECT * FROM vecindario LIMIT 0").df().columns
        if c != "TransactionID"
    ]
    vec = ", ".join(f'v."{c}"' for c in cols_vec)
    ident = ", ".join(f'i."{c}"' for c in esquema.COLUMNAS_ID if c != "TransactionID")
    tx = ", ".join(f't."{c}"' for c in esquema.COLUMNAS_TX if c not in ("TransactionID",))
    df = con.execute(
        f"""
        SELECT t.TransactionID, t.dia, c.uid AS cliente,
               e.tarjeta_direccion AS id_tarjeta_direccion, {tx}, {ident}, {vec}
        FROM transacciones t
        JOIN clientes c USING (TransactionID)
        JOIN entidades e USING (TransactionID)
        JOIN vecindario v USING (TransactionID)
        LEFT JOIN identidad i USING (TransactionID)
        WHERE t.dia <= {int(hasta_dia)}
        ORDER BY t.TransactionID
        """
    ).df()
    df = derivadas(df)
    df.attrs["columnas_vecindario"] = cols_vec
    return df


def derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega ``hora``, ``dia_semana`` y ``decimales`` (mismo código en el servicio)."""
    t = df["TransactionDT"].astype("int64")
    df["hora"] = ((t % 86_400) // 3600).astype("int8")
    df["dia_semana"] = ((t // 86_400) % 7).astype("int8")
    monto = df["TransactionAmt"].astype("float64")
    df["decimales"] = np.select(
        [np.isclose(monto, monto.round(0)), np.isclose(monto * 100, (monto * 100).round(0))],
        [0, 2],
        3,
    ).astype("int8")
    return df


@dataclass
class Codificador:
    """Códigos de categorías por frecuencia, ajustados en un periodo y congelados."""

    mapas: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def ajustar(cls, df: pd.DataFrame, columnas: tuple[str, ...] = CATEGORICAS) -> "Codificador":
        mapas = {}
        for c in columnas:
            conteo = df[c].dropna().astype(str).value_counts()
            # Empates de frecuencia: orden alfabético, para que no dependa del orden de lectura.
            orden = sorted(conteo.items(), key=lambda kv: (-kv[1], kv[0]))
            mapas[c] = [k for k, _ in orden]
        return cls(mapas)

    def aplicar(self, df: pd.DataFrame) -> pd.DataFrame:
        salida = df.copy()
        for c, valores in self.mapas.items():
            if c not in df:
                continue
            codigos = {v: i for i, v in enumerate(valores)}
            serie = df[c]
            cod = serie.astype(str).map(codigos).fillna(len(valores))
            salida[c] = cod.where(serie.notna()).astype("float32")
        return salida


def matriz(df: pd.DataFrame, columnas: list[str], codificador: Codificador) -> pd.DataFrame:
    """Variables en ``float32``, en el orden de ``columnas``."""
    prohibidas = {
        "dia",
        "TransactionDT",
        "isFraud",
        "cliente",
        "TransactionID",
        "D1n",
        "id_tarjeta_direccion",
    }
    assert not set(columnas) & prohibidas, set(columnas) & prohibidas
    return codificador.aplicar(df[columnas]).astype("float32")
