"""Reconstrucción del cliente detrás de las transacciones.

IEEE-CIS no trae identificador de cliente. ``card1`` es un código de tarjeta que comparten
muchos clientes, ``addr1`` es la dirección de facturación y ``D1`` son los días desde el alta
de la tarjeta, así que ``D1n = dia - D1`` es el día de alta: constante para una tarjeta.

Niveles, de menos a más estricto:

- ``tarjeta``: ``card1``.
- ``tarjeta_direccion``: ``card1 + addr1``.
- ``uid``: ``card1 + addr1 + D1n`` exacto (la clave popular en la competencia). **Es el
  cliente que usa el resto del proyecto.** Se calcula con lo que trae la transacción, así que
  no mira el futuro.
- ``uid_tolerancia_1``: ``card1 + addr1 + D1n`` con un día de tolerancia, asignado en orden de
  tiempo (cada transacción se une a un cliente ya visto o abre uno nuevo). **Descartado:** los
  pares que agrega no coinciden en ``D10n`` ni ``D15n`` (une tarjetas distintas dadas de alta
  en días consecutivos).
- ``uid_veto_correo``: ``uid`` en orden de tiempo, pero un correo distinto veta la unión (uno
  faltante no veta). **Descartado:** separa sobre todo transacciones del mismo cliente, la
  misma lección que dejaron los vetos del proyecto 1.

Faltantes: ``addr1`` faltante es un valor más (no se une con direcciones conocidas) y una
transacción sin ``D1`` va al cliente "sin D1" de su ``card1 + addr1``.

Sin verdad conocida, la calidad se mide con atributos que **no** se usan para unir
(``concordancia``): el correo, ``D10n``, ``D15n`` y el dispositivo deberían coincidir entre
transacciones consecutivas del mismo cliente. ``card2``-``card6`` y ``addr2`` se reportan
pero no discriminan: coinciden en más del 99 % aun agrupando solo por ``card1``.

Para comparar dos agrupaciones, ``cambios`` mide la concordancia de los pares que una une y
la otra no, y la ubica entre la de pares del mismo ``uid`` y la de pares de ``uid`` distintos
dentro de la misma ``card1 + addr1`` (``fraccion_mismo_cliente``).
"""

import math
from collections.abc import Sequence

import duckdb
import numpy as np
import pandas as pd

NIVELES: tuple[str, ...] = (
    "tarjeta",
    "tarjeta_direccion",
    "uid",
    "uid_tolerancia_1",
    "uid_veto_correo",
)
ATRIBUTOS_VALIDACION: tuple[str, ...] = (
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr2",
    "P_emaildomain",
    "D10n",
    "D15n",
    "DeviceInfo",
)
NIVEL_PRINCIPAL = "uid"


def cargar_base(con: duckdb.DuckDBPyConnection, hasta_dia: int | None = None) -> pd.DataFrame:
    """Columnas para reconstruir y validar clientes, en orden de tiempo. Sin etiquetas."""
    filtro = f"WHERE t.dia <= {int(hasta_dia)}" if hasta_dia is not None else ""
    return con.execute(
        f"""
        SELECT t.TransactionID, t.TransactionDT, t.dia, t.ProductCD,
               t.card1, t.card2, t.card3, t.card4, t.card5, t.card6, t.addr1, t.addr2,
               t.P_emaildomain, t.D1, t.dia - t.D1 AS D1n, t.dia - t.D10 AS D10n,
               t.dia - t.D15 AS D15n, i.DeviceInfo
        FROM transacciones t LEFT JOIN identidad i USING (TransactionID)
        {filtro}
        ORDER BY t.TransactionDT, t.TransactionID
        """
    ).df()


def clave_exacta(df: pd.DataFrame, columnas: Sequence[str]) -> np.ndarray:
    """Cliente = combinación exacta de ``columnas`` (faltante = un valor más).

    Numerados por orden de primera aparición, así que no dependen del futuro.
    """
    return df.groupby(list(columnas), dropna=False, sort=False).ngroup().to_numpy()


def _orden_deltas(tolerancia: int) -> list[int]:
    # Exacto primero; después el ancla del día anterior (la deriva observada es hacia +1).
    deltas = [0]
    for k in range(1, tolerancia + 1):
        deltas += [-k, k]
    return deltas


def asignar_en_linea(
    df: pd.DataFrame, tolerancia: int = 0, veto_correo: bool = False
) -> np.ndarray:
    """Asigna cada transacción (``df`` en orden de tiempo) a un cliente, sin mirar el futuro.

    Cada cliente guarda su ancla (el ``D1n`` de su primera transacción); una transacción se une
    al cliente de su ``card1 + addr1`` cuyo ancla esté a ``tolerancia`` días o menos. El ancla
    no se mueve, así que la tolerancia no encadena clientes.
    """
    card1 = df["card1"].to_numpy()
    addr1 = df["addr1"].to_numpy(dtype="float64", na_value=np.nan)
    d1n = df["D1n"].to_numpy(dtype="float64", na_value=np.nan)
    correos = df["P_emaildomain"].tolist() if veto_correo else [None] * len(df)
    deltas = _orden_deltas(tolerancia)
    anclas: dict[tuple[int, int, int], list[list]] = {}
    sin_d1: dict[tuple[int, int], int] = {}
    salida = np.empty(len(df), dtype=np.int64)
    siguiente = 0
    for i in range(len(df)):
        c = int(card1[i])
        a = -1 if math.isnan(addr1[i]) else int(addr1[i])
        if math.isnan(d1n[i]):
            if (c, a) not in sin_d1:
                sin_d1[(c, a)] = siguiente
                siguiente += 1
            salida[i] = sin_d1[(c, a)]
            continue
        d = int(d1n[i])
        e = correos[i] if isinstance(correos[i], str) else None
        elegido = None
        for delta in deltas:
            for cliente in anclas.get((c, a, d + delta), ()):
                if e is None or cliente[1] is None or cliente[1] == e:
                    elegido = cliente
                    break
            if elegido is not None:
                break
        if elegido is None:
            elegido = [siguiente, e]
            anclas.setdefault((c, a, d), []).append(elegido)
            siguiente += 1
        elif elegido[1] is None and e is not None:
            elegido[1] = e
        salida[i] = elegido[0]
    return salida


def asignar(df: pd.DataFrame, nivel: str) -> np.ndarray:
    """Identificador de cliente según ``nivel`` (ver ``NIVELES``)."""
    if nivel == "tarjeta":
        return clave_exacta(df, ["card1"])
    if nivel == "tarjeta_direccion":
        return clave_exacta(df, ["card1", "addr1"])
    if nivel == "uid":
        return clave_exacta(df, ["card1", "addr1", "D1n"])
    if nivel == "uid_tolerancia_1":
        return asignar_en_linea(df, tolerancia=1)
    if nivel == "uid_veto_correo":
        return asignar_en_linea(df, veto_correo=True)
    raise ValueError(f"nivel desconocido: {nivel}")


def concordancia(
    df: pd.DataFrame, cliente: np.ndarray, atributos: Sequence[str] = ATRIBUTOS_VALIDACION
) -> dict[str, dict[str, float]]:
    """Para cada atributo: fracción de pares consecutivos del mismo cliente que coinciden.

    Solo cuenta pares con el atributo presente en ambas transacciones. ``df`` en orden de
    tiempo.
    """
    g = pd.Series(cliente, index=df.index)
    salida: dict[str, dict[str, float]] = {}
    for col in atributos:
        actual = df[col]
        previo = actual.groupby(g).shift()
        validos = actual.notna() & previo.notna()
        n = int(validos.sum())
        iguales = (actual[validos] == previo[validos]).mean() if n else float("nan")
        salida[col] = {"pares": n, "concordancia": float(iguales)}
    return salida


def fragmentacion(cliente: np.ndarray) -> dict[str, float]:
    """Cuántos clientes, qué tan grandes y qué parte de las transacciones queda sola."""
    tamanos = np.bincount(cliente)
    tamanos = tamanos[tamanos > 0]
    n_tx = int(tamanos.sum())
    return {
        "clientes": int(len(tamanos)),
        "transacciones_por_cliente": n_tx / len(tamanos),
        "fraccion_clientes_unicos": float((tamanos == 1).mean()),
        "fraccion_tx_en_clientes_unicos": float(tamanos[tamanos == 1].sum() / n_tx),
        "mediana_tamano": float(np.median(tamanos)),
        "p99_tamano": float(np.quantile(tamanos, 0.99)),
        "max_tamano": int(tamanos.max()),
    }


def tabla_clientes(df: pd.DataFrame) -> pd.DataFrame:
    """``TransactionID`` con el cliente de cada nivel."""
    salida = df[["TransactionID", "dia"]].copy()
    for nivel in NIVELES:
        salida[nivel] = asignar(df, nivel)
    return salida


def _pares_consecutivos(
    df: pd.DataFrame, agrupacion: np.ndarray, col: str
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Valor actual, valor de la transacción anterior del mismo grupo y máscara de válidos."""
    actual = df[col]
    previo = actual.groupby(pd.Series(agrupacion, index=df.index)).shift()
    return actual, previo, actual.notna() & previo.notna()


def referencias(
    df: pd.DataFrame, atributos: Sequence[str] = ("P_emaildomain", "D10n", "D15n", "DeviceInfo")
) -> dict[str, dict[str, float]]:
    """Concordancia de pares del mismo ``uid`` y de ``uid`` distintos en la misma
    ``card1 + addr1`` (consecutivos en el tiempo)."""
    uid = asignar(df, "uid")
    td = asignar(df, "tarjeta_direccion")
    uid_previo = pd.Series(uid, index=df.index).groupby(td).shift()
    distinto = uid_previo.notna() & (uid_previo != uid)
    salida = {}
    for col in atributos:
        a, p, v = _pares_consecutivos(df, uid, col)
        a2, p2, v2 = _pares_consecutivos(df, td, col)
        m = v2 & distinto
        salida[col] = {
            "mismo_uid": float((a[v] == p[v]).mean()),
            "uid_distinto": float((a2[m] == p2[m]).mean()),
        }
    return salida


def cambios(
    df: pd.DataFrame,
    base: np.ndarray,
    variante: np.ndarray,
    refs: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Pares consecutivos que ``variante`` une y ``base`` no (agregados), y pares que
    ``base`` une y ``variante`` no (separados), con su concordancia por atributo.

    ``fraccion_mismo_cliente`` interpola entre las referencias: 1 = como pares del mismo
    ``uid``, 0 = como pares de clientes distintos.
    """
    salida: dict[str, dict[str, float]] = {}
    for nombre, une, otra in (("agregados", variante, base), ("separados", base, variante)):
        otra_previa = pd.Series(otra, index=df.index).groupby(une).shift()
        cambio = otra_previa.notna() & (otra_previa != otra)
        fila: dict[str, float] = {"pares": int(cambio.sum())}
        for col, r in refs.items():
            a, p, v = _pares_consecutivos(df, une, col)
            m = v & cambio
            c = float((a[m] == p[m]).mean()) if m.any() else float("nan")
            fila[f"{col}_concordancia"] = c
            fila[f"{col}_fraccion_mismo_cliente"] = (c - r["uid_distinto"]) / (
                r["mismo_uid"] - r["uid_distinto"]
            )
        salida[nombre] = fila
    return salida
