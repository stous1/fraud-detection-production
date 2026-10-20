"""Diagnósticos de variables que pueden filtrar el futuro o depender de cómo se armó la
competencia.

- ``propagacion``: tras un fraude, ¿las transacciones posteriores del mismo cliente quedan
  marcadas como fraude? Los organizadores describieron esa regla de etiquetado. Se mide
  también qué parte de la señal sobrevive al retraso del contracargo. **Lee etiquetas: solo
  sobre desarrollo.**
- ``monotonia``: dentro del cliente, ¿la columna baja, se mantiene o sube entre transacciones
  consecutivas? Un contador en línea no baja; un agregado de toda la ventana es constante.
- ``prueba_futuro``: en la primera transacción de una tarjeta nueva (``D1 = 0``, producto W),
  ¿la columna anticipa cuántas transacciones hará el cliente en los 30 días siguientes? Una
  variable calculada en línea no tiene historia que mirar; una que agrega la ventana completa
  sí correlaciona. No lee etiquetas.

``GRUPOS`` declara cada grupo de variables con o sin sospecha y el motivo; los modelos del
hito 3 se reportan con y sin los sospechosos.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

SEGUNDOS_DIA = 86_400
RETRASOS: tuple[int, ...] = (0, 15, 30, 60, 90)
RETRASO_BASE = 30
TRAMOS_DESDE_FRAUDE: tuple[tuple[float, float, str], ...] = (
    (0, 1, "0-1"),
    (1, 7, "1-7"),
    (7, 30, "7-30"),
    (30, 60, "30-60"),
    (60, np.inf, "60+"),
)


@dataclass(frozen=True)
class Grupo:
    nombre: str
    columnas: tuple[str, ...]
    sospechosa: bool
    motivo: str


GRUPOS: tuple[Grupo, ...] = (
    Grupo(
        "historia_etiquetas_cliente",
        (),
        True,
        "Fraudes previos del mismo cliente (con retraso). Disponibles en producción, pero "
        "inflados por la propagación de etiquetas de la competencia.",
    ),
    Grupo(
        "C",
        tuple(f"C{i}" for i in range(1, 15)),
        True,
        "Conteos opacos de Vesta; podrían agregar la ventana completa.",
    ),
    Grupo(
        "D",
        tuple(f"D{i}" for i in range(1, 16)),
        True,
        "Diferencias de tiempo opacas; D1 se usa para reconstruir al cliente.",
    ),
    Grupo(
        "V",
        tuple(f"V{i}" for i in range(1, 340)),
        True,
        "339 variables construidas por Vesta, sin definición pública.",
    ),
    Grupo(
        "identidad",
        tuple(f"id_{i:02d}" for i in range(1, 39)) + ("DeviceType", "DeviceInfo"),
        True,
        "Solo existe para parte de las transacciones; su presencia depende de cómo se "
        "recolectó y ya se asocia con el fraude.",
    ),
    Grupo(
        "transaccion",
        (
            "TransactionAmt",
            "ProductCD",
            "card1",
            "card2",
            "card3",
            "card4",
            "card5",
            "card6",
            "addr1",
            "addr2",
            "dist1",
            "dist2",
            "P_emaildomain",
            "R_emaildomain",
            *(f"M{i}" for i in range(1, 10)),
        ),
        False,
        "Datos de la transacción en el momento de pagar.",
    ),
)


def _primer_fraude_previo(df: pd.DataFrame, cliente: np.ndarray) -> pd.Series:
    """Momento del primer fraude del cliente **antes** de cada transacción (NaN si no hubo)."""
    t_fraude = df["TransactionDT"].where(df["isFraud"] == 1).astype("float64")
    g = pd.Series(cliente, index=df.index)
    return t_fraude.fillna(np.inf).groupby(g).cummin().groupby(g).shift().replace(np.inf, np.nan)


def propagacion(
    df: pd.DataFrame, cliente: np.ndarray, retrasos: Sequence[int] = RETRASOS
) -> dict[str, object]:
    """Probabilidad de fraude según la historia de fraude del cliente.

    ``df`` en orden de tiempo con ``TransactionDT`` e ``isFraud`` (solo desarrollo).
    """
    y = df["isFraud"]
    g = pd.Series(cliente, index=df.index)
    primera = g.groupby(g).cumcount() == 0
    previo = _primer_fraude_previo(df, cliente)
    con_previo = previo.notna()
    salida: dict[str, object] = {
        "transacciones": int(len(df)),
        "prevalencia": float(y.mean()),
        "p_fraude_primera_del_cliente": float(y[primera].mean()),
        "p_fraude_sin_fraude_previo": float(y[~primera & ~con_previo].mean()),
        "p_fraude_con_fraude_previo": float(y[con_previo].mean()),
    }
    clientes_fraude = y.groupby(g).max()
    salida["clientes"] = int(len(clientes_fraude))
    salida["clientes_con_fraude"] = int(clientes_fraude.sum())
    por_retraso = []
    for r in retrasos:
        conocido = (df["TransactionDT"] - previo) >= r * SEGUNDOS_DIA
        por_retraso.append(
            {
                "retraso_dias": r,
                "transacciones_marcadas": int(conocido.sum()),
                "precision": float(y[conocido].mean()) if conocido.any() else float("nan"),
                "fraccion_de_fraudes_cubiertos": float(y[conocido].sum() / y.sum()),
            }
        )
    salida["regla_fraude_previo"] = por_retraso
    dias = (df["TransactionDT"] - previo) / SEGUNDOS_DIA
    tramos = []
    for a, b, nombre in TRAMOS_DESDE_FRAUDE:
        m = (dias >= a) & (dias < b)
        tramos.append(
            {
                "dias_desde_primer_fraude": nombre,
                "transacciones": int(m.sum()),
                "p_fraude": float(y[m].mean()) if m.any() else float("nan"),
            }
        )
    salida["por_dias_desde_primer_fraude"] = tramos
    return salida


def monotonia(
    df: pd.DataFrame, cliente: np.ndarray, columnas: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Fracción de pares consecutivos del cliente en que la columna baja, se mantiene o sube."""
    g = pd.Series(cliente, index=df.index)
    salida = {}
    for col in columnas:
        previo = df[col].groupby(g).shift()
        v = df[col].notna() & previo.notna()
        d = (df[col] - previo)[v]
        salida[col] = {
            "pares": int(v.sum()),
            "baja": float((d < 0).mean()) if len(d) else float("nan"),
            "igual": float((d == 0).mean()) if len(d) else float("nan"),
            "sube": float((d > 0).mean()) if len(d) else float("nan"),
        }
    return salida


def primeras_de_tarjetas_nuevas(
    df: pd.DataFrame, cliente: np.ndarray, ventana: int = 30, ultimo_dia: int = 182
) -> pd.DataFrame:
    """Primera transacción de cada cliente con ``D1 = 0`` y producto W (tarjeta nueva, sin
    historia), con ``futuras`` = transacciones del cliente en los ``ventana`` días siguientes.

    Solo clientes cuya primera transacción deja ``ventana`` días de datos por delante.
    """
    g = pd.Series(cliente, index=df.index, name="cliente")
    orden = g.groupby(g).cumcount()
    primeras = df[
        (orden == 0)
        & (df["D1"] == 0)
        & (df["ProductCD"] == "W")
        & (df["dia"] <= ultimo_dia - ventana)
    ].assign(cliente=g)
    todas = pd.DataFrame({"cliente": g, "dia": df["dia"]})
    cruce = todas.merge(primeras[["cliente", "dia"]].rename(columns={"dia": "d0"}), on="cliente")
    en_ventana = cruce[(cruce["dia"] >= cruce["d0"]) & (cruce["dia"] <= cruce["d0"] + ventana)]
    futuras = en_ventana.groupby("cliente").size() - 1
    return primeras.assign(futuras=primeras["cliente"].map(futuras).fillna(0).astype(int))


def prueba_futuro(
    primeras: pd.DataFrame, columnas: Sequence[str], minimo: int = 200
) -> dict[str, float]:
    """Correlación de Spearman entre cada columna y ``futuras`` (ver
    ``primeras_de_tarjetas_nuevas``). Columnas con menos de ``minimo`` valores o constantes
    quedan fuera."""
    rango_futuras = primeras["futuras"].rank()
    salida = {}
    for col in columnas:
        x = primeras[col]
        v = x.notna()
        if v.sum() < minimo or x[v].nunique() < 2:
            continue
        salida[col] = float(x[v].rank().corr(rango_futuras[v]))
    return salida


def resumen_futuro(rho: dict[str, float], umbral: float = 0.3) -> dict[str, object]:
    """Por grupo: cuántas columnas superan ``|rho| >= umbral`` y la de mayor ``|rho|``."""
    salida = {}
    for grupo in GRUPOS:
        cols = [c for c in grupo.columnas if c in rho]
        if not cols:
            continue
        mayor = max(cols, key=lambda c: abs(rho[c]))
        salida[grupo.nombre] = {
            "columnas_evaluadas": len(cols),
            "columnas_sobre_umbral": sum(abs(rho[c]) >= umbral for c in cols),
            "mayor": mayor,
            "rho_mayor": rho[mayor],
        }
    return {"umbral": umbral, "grupos": salida}
