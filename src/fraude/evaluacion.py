"""Métricas de evaluación. Nunca exactitud.

- ``ap``: precisión promedio (área bajo la curva precisión-recall).
- ``recall_a_fpr``: recall cuando se acepta una tasa fija de falsas alarmas (fracción de
  legítimas marcadas), con el umbral tomado de los puntajes de las legítimas.
- ``curva_pr``: puntos de la curva precisión-recall, resumidos.
- ``intervalo_ap``: intervalo del 95 % por bootstrap **de clientes** (todas las transacciones
  de un cliente salen o entran juntas: no son independientes).
- ``por_segmento``: todo, clientes nuevos y clientes ya vistos antes del periodo.
"""

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

TASAS_FALSAS_ALARMAS: tuple[float, ...] = (0.01, 0.05)
REPETICIONES = 200
SEMILLA = 11


def ap(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s)) if y.any() else float("nan")


def recall_a_fpr(y: np.ndarray, s: np.ndarray, fpr: float) -> dict[str, float]:
    """Recall y precisión marcando lo que supera el umbral que deja ``fpr`` de legítimas
    marcadas (empates en el umbral: no se marcan)."""
    negativos = np.sort(s[y == 0])[::-1]
    k = int(np.floor(fpr * len(negativos)))
    umbral = negativos[k] if k < len(negativos) else -np.inf
    marcadas = s > umbral
    vp = int((marcadas & (y == 1)).sum())
    return {
        "fpr_objetivo": fpr,
        "umbral": float(umbral),
        "fpr_real": float((marcadas & (y == 0)).sum() / max(1, len(negativos))),
        "recall": vp / max(1, int(y.sum())),
        "precision": vp / max(1, int(marcadas.sum())),
    }


def curva_pr(y: np.ndarray, s: np.ndarray, puntos: int = 100) -> list[dict[str, float]]:
    """Curva precisión-recall con ``puntos`` puntos espaciados en recall."""
    precision, recall, _ = precision_recall_curve(y, s)
    objetivo = np.linspace(0, 1, puntos + 1)
    salida = []
    for r in objetivo:
        validos = recall >= r
        salida.append({"recall": float(r), "precision": float(precision[validos].max())})
    return salida


def intervalo_ap(
    y: np.ndarray,
    s: np.ndarray,
    grupos: np.ndarray,
    repeticiones: int = REPETICIONES,
    semilla: int = SEMILLA,
) -> tuple[float, float]:
    """Intervalo del 95 % de la AP remuestreando clientes con reemplazo."""
    unicos, inverso = np.unique(grupos, return_inverse=True)
    orden = np.argsort(inverso, kind="stable")
    limites = np.searchsorted(inverso[orden], np.arange(len(unicos) + 1))
    rng = np.random.default_rng(semilla)
    valores = []
    for _ in range(repeticiones):
        elegidos = rng.integers(len(unicos), size=len(unicos))
        filas = np.concatenate([orden[limites[g] : limites[g + 1]] for g in elegidos])
        valores.append(ap(y[filas], s[filas]))
    return float(np.nanpercentile(valores, 2.5)), float(np.nanpercentile(valores, 97.5))


def metricas(
    y: np.ndarray,
    s: np.ndarray,
    grupos: np.ndarray | None = None,
    tasas: Sequence[float] = TASAS_FALSAS_ALARMAS,
    intervalo: bool = True,
) -> dict[str, object]:
    """AP (con intervalo si hay ``grupos``), prevalencia y recall a tasas de falsas alarmas."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype="float64")
    salida: dict[str, object] = {
        "transacciones": int(len(y)),
        "fraudes": int(y.sum()),
        "prevalencia": float(y.mean()) if len(y) else float("nan"),
        "ap": ap(y, s),
    }
    if intervalo and grupos is not None and y.any():
        salida["ap_ic95"] = list(intervalo_ap(y, s, np.asarray(grupos)))
    salida["recall_a_fpr"] = [recall_a_fpr(y, s, f) for f in tasas]
    return salida


def por_segmento(
    y: np.ndarray,
    s: np.ndarray,
    visto: np.ndarray,
    grupos: np.ndarray | None = None,
    intervalo: bool = True,
) -> dict[str, dict[str, object]]:
    """Métricas para todo el periodo, clientes nuevos y clientes ya vistos antes de él."""
    y, s, visto = np.asarray(y), np.asarray(s), np.asarray(visto, dtype=bool)
    g = np.asarray(grupos) if grupos is not None else None
    salida = {}
    for nombre, m in (("todos", np.ones(len(y), bool)), ("nuevos", ~visto), ("vistos", visto)):
        salida[nombre] = metricas(y[m], s[m], None if g is None else g[m], intervalo=intervalo)
    return salida
