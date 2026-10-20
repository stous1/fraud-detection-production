"""Costo total esperado según el umbral: un falso positivo también cuesta.

- Fraude no detectado (falso negativo): se pierde el monto de la transacción.
- Alerta sobre una legítima (falso positivo): ``costo_fp`` US$ por revisión y fricción con el
  cliente (venta perdida, llamada). Supuesto base: 10 US$; sensibilidad 5, 25 y 50.
- Fraude detectado: se evita su monto (el costo de revisarlo se ignora; es igual con o sin
  modelo cuando hay alerta).

El umbral se elige en validación como **tasa de alertas** (fracción de transacciones
marcadas), porque el modelo final se reajusta con más días y sus puntajes cambian de escala.
Al desplegar, la tasa se convierte en umbral con los puntajes del modelo reajustado sobre
días ya vistos y sin etiqueta (91 a 120): eso existe el día 121.
"""

from collections.abc import Sequence

import numpy as np

COSTO_FP_BASE = 10.0
COSTOS_FP: tuple[float, ...] = (5.0, 10.0, 25.0, 50.0)
TASAS_CANDIDATAS: tuple[float, ...] = tuple(
    float(x)
    for x in np.round(
        np.concatenate([np.arange(0.001, 0.02, 0.001), np.arange(0.02, 0.201, 0.005)]), 4
    )
)


def costo(
    y: np.ndarray, marcadas: np.ndarray, monto: np.ndarray, costo_fp: float
) -> dict[str, float]:
    """Costo de marcar ``marcadas``: montos de fraudes no marcados más ``costo_fp`` por legítima
    marcada."""
    y = np.asarray(y).astype(bool)
    marcadas = np.asarray(marcadas, dtype=bool)
    perdido = float(monto[y & ~marcadas].sum())
    fp = int((~y & marcadas).sum())
    vp = int((y & marcadas).sum())
    return {
        "costo_total": perdido + costo_fp * fp,
        "monto_perdido": perdido,
        "falsos_positivos": fp,
        "verdaderos_positivos": vp,
        "alertas": int(marcadas.sum()),
        "tasa_alertas": float(marcadas.mean()),
        "recall": vp / max(1, int(y.sum())),
        "precision": vp / max(1, int(marcadas.sum())),
        "recall_monto": float(monto[y & marcadas].sum() / max(1e-9, monto[y].sum())),
    }


def umbral_de_tasa(puntajes: np.ndarray, tasa: float) -> float:
    """Umbral que marca (con ``>``) las ``round(tasa * n)`` transacciones de mayor puntaje;
    con empates en el umbral, marca menos."""
    orden = np.sort(np.asarray(puntajes, dtype="float64"))[::-1]
    k = int(round(tasa * len(orden)))
    return float(orden[k]) if k < len(orden) else float("-inf")


def elegir_tasa(
    y: np.ndarray,
    s: np.ndarray,
    monto: np.ndarray,
    costo_fp: float,
    tasas: Sequence[float] = TASAS_CANDIDATAS,
) -> dict[str, object]:
    """Tasa de alertas de menor costo en (y, s); empate: la menor tasa."""
    filas = []
    for t in tasas:
        u = umbral_de_tasa(s, t)
        filas.append({"tasa": t, **costo(y, s > u, monto, costo_fp)})
    mejor = min(filas, key=lambda f: (f["costo_total"], f["tasa"]))
    sin_modelo = costo(y, np.zeros(len(y), bool), monto, costo_fp)
    return {"tasa": mejor["tasa"], "mejor": mejor, "sin_modelo": sin_modelo, "curva": filas}
