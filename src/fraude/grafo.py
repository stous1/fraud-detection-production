"""Grafo de identidad compartida construido desde la tabla: transacciones unidas a entidades.

Cada transacción se conecta con seis entidades (nodos del grafo heterogéneo):

- ``cliente``: el ``uid`` del hito 1 (``card1 + addr1 + D1n``).
- ``tarjeta_direccion``: ``card1 + addr1``; une los fragmentos de un ``uid`` (por ejemplo,
  días con ``D1 = 0``) y también clientes distintos.
- ``tarjeta``: perfil de tarjeta ``card1`` a ``card6`` (mediana de 2 clientes por perfil).
- ``huella``: dispositivo = ``DeviceInfo + id_30 + id_31 + id_33 + id_19 + id_20``. Solo
  existe con identidad (24 %); es la entidad más específica (mediana de 1 cliente).
- ``direccion``: ``addr1 + addr2``, una región (437 valores).
- ``correo``: dominio del correo del comprador. **IEEE-CIS no trae la dirección de correo,
  solo el dominio** (59 valores): es un atributo, no un identificador.

Los identificadores son enteros numerados por la primera aparición de la entidad (orden de
``TransactionID``), así que no dependen de la función de hash de DuckDB ni del futuro.
"""

import duckdb

ENTIDADES: dict[str, str] = {
    "tarjeta_direccion": "concat_ws('|', t.card1, coalesce(t.addr1::VARCHAR, '-'))",
    "tarjeta": (
        "concat_ws('|', t.card1, coalesce(t.card2::VARCHAR, '-'), coalesce(t.card3::VARCHAR, '-'),"
        " coalesce(t.card4, '-'), coalesce(t.card5::VARCHAR, '-'), coalesce(t.card6, '-'))"
    ),
    "huella": (
        "CASE WHEN i.DeviceInfo IS NOT NULL THEN concat_ws('|', i.DeviceInfo,"
        " coalesce(i.id_30, '-'), coalesce(i.id_31, '-'), coalesce(i.id_33, '-'),"
        " coalesce(i.id_19::VARCHAR, '-'), coalesce(i.id_20::VARCHAR, '-')) END"
    ),
    "direccion": (
        "CASE WHEN t.addr1 IS NOT NULL"
        " THEN concat_ws('|', t.addr1, coalesce(t.addr2::VARCHAR, '-')) END"
    ),
    "correo": "t.P_emaildomain",
}
TIPOS: tuple[str, ...] = ("cliente", *ENTIDADES)


def crear_entidades(con: duckdb.DuckDBPyConnection) -> None:
    """Tabla ``entidades``: ``TransactionID``, ``TransactionDT``, ``dia``, ``cliente`` y un
    identificador entero por tipo de entidad (``NULL`` si la transacción no la tiene)."""
    claves = ",\n".join(f"{expr} AS {nombre}" for nombre, expr in ENTIDADES.items())
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE claves_entidad AS
        SELECT t.TransactionID, t.TransactionDT, t.dia, c.uid AS cliente,
               {claves}
        FROM transacciones t
        JOIN clientes c USING (TransactionID)
        LEFT JOIN identidad i USING (TransactionID)
        """
    )
    ids = []
    for nombre in ENTIDADES:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE id_{nombre} AS
            SELECT {nombre} AS clave,
                   (row_number() OVER (ORDER BY min(TransactionID)) - 1)::INTEGER AS id
            FROM claves_entidad WHERE {nombre} IS NOT NULL GROUP BY 1
            """
        )
        ids.append(f"id_{nombre}.id AS {nombre}")
    uniones = "\n".join(f"LEFT JOIN id_{n} ON id_{n}.clave = k.{n}" for n in ENTIDADES)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE entidades AS
        SELECT k.TransactionID, k.TransactionDT, k.dia, k.cliente, {", ".join(ids)}
        FROM claves_entidad k
        {uniones}
        ORDER BY k.TransactionID
        """
    )
    for nombre in ENTIDADES:
        con.execute(f"DROP TABLE id_{nombre}")
    con.execute("DROP TABLE claves_entidad")


def resumen(con: duckdb.DuckDBPyConnection) -> list[dict[str, object]]:
    """Por tipo: cuántas transacciones la tienen, cuántas entidades y cuántos clientes por
    entidad (mediana, percentil 99, máximo). Sin etiquetas."""
    filas = []
    for tipo in TIPOS:
        r = con.execute(
            f"""
            SELECT count(*), sum(n), quantile_cont(nc, 0.5), quantile_cont(nc, 0.99), max(nc)
            FROM (SELECT {tipo}, count(*) AS n, count(DISTINCT cliente) AS nc
                  FROM entidades WHERE {tipo} IS NOT NULL GROUP BY 1)
            """
        ).fetchone()
        filas.append(
            {
                "tipo": tipo,
                "entidades": r[0],
                "transacciones_con_entidad": int(r[1]),
                "mediana_clientes": r[2],
                "p99_clientes": r[3],
                "max_clientes": r[4],
            }
        )
    return filas
