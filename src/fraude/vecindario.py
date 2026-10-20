"""Variables de vecindario sobre el grafo de identidad, **solo con información del pasado**.

Convención: el pasado de una transacción son las transacciones con ``TransactionID`` menor
(el hito 1 verificó que ese es el orden temporal). Las etiquetas entran con retraso: el
fraude de una transacción con tiempo ``ts`` se conoce desde ``ts + R`` días, así que en el
tiempo ``t`` solo cuentan las de ``ts <= t - R``. ``R = 0`` (sin retraso) se calcula solo
como comparación: es filtración en producción.

Por entidad (``cliente`` y cada tipo de ``grafo.ENTIDADES``), con prefijo ``{tipo}_``:

- ``tx_previas``, ``tx_24h``, ``tx_7d``: transacciones previas de la entidad (todas, en las
  últimas 24 horas y en los últimos 7 días);
- ``dias_desde_primera``: antigüedad de la entidad en los datos;
- ``clientes_previos`` y ``cliente_nuevo``: cuántos clientes distintos la usaron antes y si el
  cliente de esta transacción es nuevo para ella (no aplica a ``cliente``);
- con retraso ``R``: ``etiquetadas_rR`` (transacciones con etiqueta ya conocida),
  ``fraudes_rR`` y ``clientes_fraude_rR`` (clientes con un fraude ya conocido en la entidad).

Del cliente a dos saltos: ``cliente_huellas_previas``, ``cliente_correos_previos`` y
``cliente_tarjetas_previas`` (cuántas distintas usó antes).

Componentes (``componentes_en_linea``): unión de clientes que comparten huella o perfil de
tarjeta, construida en orden de tiempo. Una entidad deja de unir cuando ya tiene ``K``
clientes (los concentradores unirían a todos); aun así hay un componente gigante.
"""

from collections.abc import Sequence

import duckdb
import numpy as np
import pandas as pd

from fraude.grafo import ENTIDADES

SEGUNDOS_DIA = 86_400
RETRASOS: tuple[int, ...] = (0, 15, 30, 60, 90)
LIMITE_CLIENTES = 5
ENTIDADES_COMPONENTE: tuple[str, ...] = ("huella", "tarjeta")


def _sql_entidad(tipo: str, retrasos: Sequence[int]) -> str:
    """Variables de una entidad. ``entidades`` + ``etiquetas`` -> una fila por transacción
    que tiene la entidad."""
    p = f"{tipo}_"
    previas = (
        "PARTITION BY ent ORDER BY TransactionID ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING"
    )

    def hasta(segundos: int) -> str:
        return (
            f"PARTITION BY ent ORDER BY TransactionDT "
            f"RANGE BETWEEN UNBOUNDED PRECEDING AND {segundos} PRECEDING"
        )

    columnas = [
        f"count(*) OVER ({previas}) AS {p}tx_previas",
        f"count(*) OVER ({previas}) - count(*) OVER ({hasta(SEGUNDOS_DIA + 1)}) AS {p}tx_24h",
        f"count(*) OVER ({previas}) - count(*) OVER ({hasta(7 * SEGUNDOS_DIA + 1)}) AS {p}tx_7d",
        f"(TransactionDT - first_value(TransactionDT) OVER (PARTITION BY ent ORDER BY "
        f"TransactionID)) / {SEGUNDOS_DIA} AS {p}dias_desde_primera",
    ]
    if tipo != "cliente":
        columnas += [
            f"coalesce(sum(cliente_nuevo) OVER ({previas}), 0) AS {p}clientes_previos",
            f"cliente_nuevo AS {p}cliente_nuevo",
        ]
    for r in retrasos:
        marco = previas if r == 0 else hasta(r * SEGUNDOS_DIA)
        columnas += [
            f"count(*) OVER ({marco}) AS {p}etiquetadas_r{r}",
            f"coalesce(sum(y) OVER ({marco}), 0) AS {p}fraudes_r{r}",
            f"coalesce(sum(primer_fraude) OVER ({marco}), 0) AS {p}clientes_fraude_r{r}",
        ]
    return f"""
        SELECT TransactionID, {", ".join(columnas)}
        FROM (
            SELECT e.TransactionID, e.TransactionDT, e.{tipo} AS ent, e.cliente, y.isFraud AS y,
                   (row_number() OVER (PARTITION BY e.{tipo}, e.cliente
                                       ORDER BY e.TransactionID) = 1)::INTEGER AS cliente_nuevo,
                   (y.isFraud = 1 AND row_number() OVER (
                        PARTITION BY e.{tipo}, e.cliente, y.isFraud ORDER BY e.TransactionID) = 1
                   )::INTEGER AS primer_fraude
            FROM entidades e JOIN etiquetas y USING (TransactionID)
            WHERE e.{tipo} IS NOT NULL
        )
    """


def _sql_distintos(valor: str) -> str:
    """Cuántos valores distintos de ``valor`` usó el cliente antes de cada transacción."""
    return f"""
        SELECT TransactionID,
               coalesce(sum(nuevo) OVER (PARTITION BY cliente ORDER BY TransactionID
                        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0)
                   AS cliente_{valor}s_previ{"a" if valor in ("huella", "tarjeta") else "o"}s
        FROM (
            SELECT TransactionID, cliente,
                   ({valor} IS NOT NULL AND row_number() OVER (
                       PARTITION BY cliente, {valor} ORDER BY TransactionID) = 1)::INTEGER AS nuevo
            FROM entidades
        )
    """


def crear_etiquetas(con: duckdb.DuckDBPyConnection) -> None:
    """Vista ``etiquetas`` (``TransactionID``, ``isFraud``) de todo train.

    Solo la usan las variables con retraso: en el tiempo ``t`` ven etiquetas de ``ts <= t - R``.
    """
    con.execute(
        "CREATE OR REPLACE VIEW etiquetas AS SELECT TransactionID, isFraud FROM transacciones"
    )


def variables_sql(
    con: duckdb.DuckDBPyConnection, retrasos: Sequence[int] = RETRASOS
) -> pd.DataFrame:
    """Variables de entidad y de dos saltos, una fila por transacción (orden de
    ``TransactionID``). Requiere ``entidades`` y ``etiquetas``."""
    tipos = ("cliente", *ENTIDADES)
    partes = [f"v_{t} AS ({_sql_entidad(t, retrasos)})" for t in tipos]
    partes += [f"d_{v} AS ({_sql_distintos(v)})" for v in ("huella", "correo", "tarjeta")]
    uniones = "\n".join(f"LEFT JOIN v_{t} USING (TransactionID)" for t in tipos)
    uniones += "\n" + "\n".join(
        f"LEFT JOIN d_{v} USING (TransactionID)" for v in ("huella", "correo", "tarjeta")
    )
    return con.execute(
        f"""
        WITH {", ".join(partes)}
        SELECT e.TransactionID, * EXCLUDE (TransactionID, TransactionDT, dia, cliente,
               {", ".join(ENTIDADES)})
        FROM entidades e
        {uniones}
        ORDER BY e.TransactionID
        """
    ).df()


class _Union:
    """Unión-búsqueda con contadores por componente."""

    def __init__(self, n: int, n_contadores: int) -> None:
        self.padre = np.arange(n)
        self.clientes = np.zeros(n, dtype=np.int64)
        self.contadores = np.zeros((n_contadores, n), dtype=np.int64)

    def raiz(self, x: int) -> int:
        padre = self.padre
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    def unir(self, a: int, b: int) -> None:
        ra, rb = self.raiz(a), self.raiz(b)
        if ra == rb:
            return
        if self.clientes[ra] < self.clientes[rb]:
            ra, rb = rb, ra
        self.padre[rb] = ra
        self.clientes[ra] += self.clientes[rb]
        self.contadores[:, ra] += self.contadores[:, rb]


def componentes_en_linea(
    df: pd.DataFrame,
    y: np.ndarray,
    retrasos: Sequence[int] = RETRASOS,
    entidades: Sequence[str] = ENTIDADES_COMPONENTE,
    limite: int = LIMITE_CLIENTES,
) -> pd.DataFrame:
    """Componente del cliente de cada transacción, en orden de tiempo.

    ``df``: ``entidades`` en orden de ``TransactionID``; ``y``: etiqueta de cada fila (solo se
    lee con retraso). Por transacción: ``componente_clientes`` y, por retraso,
    ``componente_fraudes_rR`` y ``componente_clientes_fraude_rR``. Las aristas de la
    transacción actual sí cuentan (se conocen al pagar); su etiqueta no.
    """
    n = len(df)
    cliente = df["cliente"].to_numpy()
    n_clientes = int(cliente.max()) + 1 if n else 0
    desplazamiento = {}
    total = n_clientes
    columnas_ent = {}
    for e in entidades:
        valores = df[e].to_numpy(dtype="float64", na_value=np.nan)
        columnas_ent[e] = valores
        desplazamiento[e] = total
        total += int(np.nanmax(valores)) + 1 if np.isfinite(valores).any() else 0
    uf = _Union(total, 2 * len(retrasos))
    uf.clientes[:n_clientes] = 1
    miembros: dict[int, set[int]] = {}
    t = df["TransactionDT"].to_numpy()
    fraudes = np.flatnonzero(y == 1)
    # Llegada de cada etiqueta de fraude: ts + R (R = 0: justo después de la transacción).
    punteros = [0] * len(retrasos)
    llegadas = [t[fraudes] + r * SEGUNDOS_DIA for r in retrasos]
    conocido = np.zeros((len(retrasos), n_clientes), dtype=bool)
    salida = np.zeros((n, 1 + 2 * len(retrasos)), dtype=np.int64)

    def registrar(k: int, fila: int) -> None:
        raiz = uf.raiz(int(cliente[fila]))
        uf.contadores[2 * k, raiz] += 1
        if not conocido[k, cliente[fila]]:
            conocido[k, cliente[fila]] = True
            uf.contadores[2 * k + 1, raiz] += 1

    for i in range(n):
        for k, r in enumerate(retrasos):
            if r == 0:
                continue
            lleg = llegadas[k]
            while punteros[k] < len(lleg) and lleg[punteros[k]] <= t[i]:
                registrar(k, int(fraudes[punteros[k]]))
                punteros[k] += 1
        c = int(cliente[i])
        for e in entidades:
            v = columnas_ent[e][i]
            if np.isnan(v):
                continue
            nodo = desplazamiento[e] + int(v)
            socios = miembros.setdefault(nodo, set())
            if c in socios:
                uf.unir(c, nodo)
            elif len(socios) < limite:
                socios.add(c)
                uf.unir(c, nodo)
        raiz = uf.raiz(c)
        salida[i, 0] = uf.clientes[raiz]
        salida[i, 1:] = uf.contadores[:, raiz]
        if y[i] == 1:
            for k, r in enumerate(retrasos):
                if r == 0:
                    registrar(k, i)
    nombres = ["componente_clientes"]
    for r in retrasos:
        nombres += [f"componente_fraudes_r{r}", f"componente_clientes_fraude_r{r}"]
    return pd.DataFrame(salida, columns=nombres).assign(
        TransactionID=df["TransactionID"].to_numpy()
    )[["TransactionID", *nombres]]


def construir(con: duckdb.DuckDBPyConnection, retrasos: Sequence[int] = RETRASOS) -> pd.DataFrame:
    """Todas las variables de vecindario; deja la tabla ``vecindario`` en la base."""
    crear_etiquetas(con)
    sql = variables_sql(con, retrasos)
    ent = con.execute(
        "SELECT e.*, y.isFraud FROM entidades e JOIN etiquetas y USING (TransactionID) "
        "ORDER BY e.TransactionID"
    ).df()
    comp = componentes_en_linea(ent, ent["isFraud"].to_numpy(), retrasos)
    tabla = sql.merge(comp, on="TransactionID", how="left", validate="one_to_one")
    con.register("tabla_vecindario", tabla)
    con.execute("CREATE OR REPLACE TABLE vecindario AS SELECT * FROM tabla_vecindario")
    con.unregister("tabla_vecindario")
    return tabla


NO_VARIABLES: frozenset[str] = frozenset(
    {"TransactionID", "TransactionDT", "dia", "cliente", "isFraud"}
)


def columnas(tabla: pd.DataFrame | Sequence[str], retraso: int | None) -> list[str]:
    """Columnas de vecindario sin etiquetas (``retraso=None``) o con las del retraso dado.

    Nunca devuelve identificadores, tiempo ni la etiqueta, aunque ``tabla`` los traiga.
    """
    nombres = list(tabla.columns if isinstance(tabla, pd.DataFrame) else tabla)
    nombres = [c for c in nombres if c not in NO_VARIABLES]
    con_etiqueta = tuple(f"_r{r}" for r in RETRASOS)
    sin_etiquetas = [c for c in nombres if not c.endswith(con_etiqueta)]
    if retraso is None:
        return sin_etiquetas
    return sin_etiquetas + [c for c in nombres if c.endswith(f"_r{retraso}")]
