"""XGBoost: entrenamiento en GPU (o CPU), búsqueda de hiperparámetros, guardado y puntaje.

- Entrena con ``device="cuda"`` si se pide y hay GPU; la CI y las pruebas usan CPU.
- La parada temprana mira la AP (``aucpr``) de un conjunto de evaluación: en el flujo, los
  **clientes nuevos** de la validación, que es la cifra principal del proyecto.
- ``puntuar`` y ``contribuciones`` cargan el modelo congelado y corren **siempre en CPU**: las
  cifras se reproducen exactas aunque el entrenamiento haya sido en GPU.
- ``contribuciones`` es TreeSHAP exacto de XGBoost (``pred_contribs``).
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

SEMILLA = 17
BASE: dict[str, object] = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "max_bin": 256,
    "seed": SEMILLA,
}
ESPACIO: dict[str, tuple] = {
    "max_depth": (4, 6, 8, 10, 12),
    "learning_rate": (0.02, 0.03, 0.05, 0.08),
    "min_child_weight": (1, 3, 10, 30),
    "subsample": (0.6, 0.8, 1.0),
    "colsample_bytree": (0.3, 0.5, 0.7, 0.9),
    "reg_lambda": (0.5, 1.0, 3.0, 10.0),
    "gamma": (0.0, 0.1, 1.0),
}
MAX_ARBOLES = 3000
PARADA = 100


def hay_gpu() -> bool:
    return bool(xgb.build_info().get("USE_CUDA"))


def configuraciones(n: int, semilla: int = SEMILLA) -> list[dict[str, object]]:
    """``n`` configuraciones al azar (sin repetir) del ``ESPACIO``."""
    rng = np.random.default_rng(semilla)
    vistas, salida = set(), []
    while len(salida) < n:
        c = {k: v[int(rng.integers(len(v)))] for k, v in ESPACIO.items()}
        clave = tuple(sorted(c.items()))
        if clave not in vistas:
            vistas.add(clave)
            salida.append(c)
    return salida


def entrenar(
    x: pd.DataFrame,
    y: np.ndarray,
    hiper: dict[str, object],
    dispositivo: str = "cpu",
    evaluacion: tuple[pd.DataFrame, np.ndarray] | None = None,
    arboles: int | None = None,
) -> tuple[xgb.Booster, int]:
    """Entrena; con ``evaluacion`` usa parada temprana y devuelve el mejor número de árboles."""
    params = {**BASE, **hiper, "device": dispositivo}
    dtrain = xgb.QuantileDMatrix(x, label=y)
    if evaluacion is not None:
        deval = xgb.QuantileDMatrix(evaluacion[0], label=evaluacion[1], ref=dtrain)
        modelo = xgb.train(
            params,
            dtrain,
            num_boost_round=arboles or MAX_ARBOLES,
            evals=[(deval, "eval")],
            early_stopping_rounds=PARADA,
            verbose_eval=False,
        )
        return modelo, int(modelo.best_iteration) + 1
    n = arboles or MAX_ARBOLES
    return xgb.train(params, dtrain, num_boost_round=n, verbose_eval=False), n


def guardar(modelo: xgb.Booster, ruta: Path, arboles: int) -> str:
    """Guarda solo los primeros ``arboles`` árboles en UBJ; devuelve el SHA-256."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    modelo[:arboles].save_model(str(ruta))
    return huella(ruta)


def huella(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def cargar(ruta: Path, sha256: str | None = None) -> xgb.Booster:
    """Carga un modelo para puntuar en CPU; con ``sha256`` verifica que sea el congelado."""
    if sha256 is not None and huella(ruta) != sha256:
        raise RuntimeError(f"{ruta} no coincide con la huella congelada")
    modelo = xgb.Booster()
    modelo.load_model(str(ruta))
    modelo.set_param({"device": "cpu", "nthread": 8})
    return modelo


def puntuar(modelo: xgb.Booster, x: pd.DataFrame) -> np.ndarray:
    """Probabilidad de fraude, en CPU."""
    return modelo.predict(xgb.DMatrix(x)).astype("float64")


def puntuar_en_memoria(modelo: xgb.Booster, x: pd.DataFrame, arboles: int) -> np.ndarray:
    """Puntaje de un modelo recién entrenado (validación), con sus primeros ``arboles``."""
    return modelo.predict(xgb.DMatrix(x), iteration_range=(0, arboles)).astype("float64")


def contribuciones(modelo: xgb.Booster, x: pd.DataFrame) -> pd.DataFrame:
    """TreeSHAP en log-odds: una columna por variable más ``sesgo``; cada fila suma el
    puntaje del modelo en log-odds."""
    c = modelo.predict(xgb.DMatrix(x), pred_contribs=True)
    return pd.DataFrame(c, columns=[*x.columns, "sesgo"], index=x.index)
