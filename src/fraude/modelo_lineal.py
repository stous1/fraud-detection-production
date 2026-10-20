"""Regresión logística sobre variables de vecindario, con parámetros guardables en JSON.

Sirve para medir en el hito 2 cuánto aportan el grafo y la historia de etiquetas con un
modelo simple; el modelo base (XGBoost) llega en el hito 3. Transformación: ``log1p`` de cada
variable (todas son conteos o días, no negativas), un indicador por cada columna que puede
faltar (la transacción no tiene esa entidad) y estandarización con media y escala de
entrenamiento. Puntuar solo usa numpy con los parámetros guardados: se reproduce exacto.
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

GRILLA_C: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0)


@dataclass
class ModeloLineal:
    columnas: list[str]
    indicadores: list[str]
    media: list[float]
    escala: list[float]
    coeficientes: list[float]
    intercepto: float
    C: float

    def a_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def desde_dict(cls, d: dict[str, object]) -> "ModeloLineal":
        return cls(**d)


def _matriz(df: pd.DataFrame, columnas: Sequence[str], indicadores: Sequence[str]) -> np.ndarray:
    x = np.log1p(df[list(columnas)].astype("float64").clip(lower=0).fillna(0).to_numpy())
    faltan = df[list(indicadores)].isna().to_numpy(dtype="float64")
    return np.hstack([x, faltan])


def entrenar(df: pd.DataFrame, y: np.ndarray, columnas: Sequence[str], C: float) -> ModeloLineal:
    indicadores = [c for c in columnas if df[c].isna().any()]
    x = _matriz(df, columnas, indicadores)
    media = x.mean(axis=0)
    escala = x.std(axis=0)
    escala[escala == 0] = 1.0
    z = (x - media) / escala
    lr = LogisticRegression(C=C, max_iter=5000, solver="lbfgs")
    lr.fit(z, y)
    return ModeloLineal(
        columnas=list(columnas),
        indicadores=indicadores,
        media=media.tolist(),
        escala=escala.tolist(),
        coeficientes=lr.coef_[0].tolist(),
        intercepto=float(lr.intercept_[0]),
        C=C,
    )


def puntuar(modelo: ModeloLineal, df: pd.DataFrame) -> np.ndarray:
    """Puntaje lineal (log-odds) con los parámetros guardados."""
    x = _matriz(df, modelo.columnas, modelo.indicadores)
    z = (x - np.array(modelo.media)) / np.array(modelo.escala)
    return z @ np.array(modelo.coeficientes) + modelo.intercepto
