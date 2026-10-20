"""Utilidades pequeñas: JSON estable."""

import json
import math
from pathlib import Path

import numpy as np


def _limpiar(x: object) -> object:
    if isinstance(x, dict):
        return {str(k): _limpiar(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_limpiar(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating | float):
        v = float(x)
        return None if math.isnan(v) else round(v, 6)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def escribir_json(datos: object, ruta: Path) -> None:
    """JSON en UTF-8, con LF, claves en orden de inserción y flotantes a 6 decimales."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8", newline="\n") as f:
        json.dump(_limpiar(datos), f, ensure_ascii=False, indent=2)
        f.write("\n")
