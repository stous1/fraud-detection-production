"""Monitoreo de deriva y rendimiento, semana a semana, con alertas.

- **Deriva de datos:** índice de estabilidad poblacional (PSI) de cada variable importante
  y del puntaje, contra una referencia sin etiquetas (los días 91 a 120, los últimos antes de
  desplegar), y cambio de la fracción de faltantes por grupo. PSI con 10 cubetas por cuantiles
  de la referencia (o una por valor si hay 20 o menos) y una cubeta para faltantes.
- **Operación:** tasa de alertas contra la esperada (la elegida por costo).
- **Rendimiento, con etiquetas maduras:** al cierre de la semana ``w`` solo se conocen las
  etiquetas de transacciones con 30 días o más. El rendimiento se mide sobre lo maduro, en
  clientes nuevos y ya vistos, y se compara con la validación.

Reglas (``REGLAS``), declaradas antes de correr la simulación:

| Señal | Aviso | Crítica |
|---|---|---|
| PSI de una variable del top SHAP | >= 0,10 | >= 0,25 |
| PSI del puntaje | >= 0,10 | >= 0,25 |
| Faltantes de un grupo (cambio absoluto) | >= 0,10 | >= 0,20 |
| Tasa de alertas / esperada | fuera de [0,7; 1,3] | fuera de [0,5; 1,5] |
| AP madura en nuevos / AP de validación en nuevos | < 0,85 | < 0,70 |

Se implementa aquí (unas decenas de líneas) en lugar de usar Evidently: el cálculo queda
explícito, sin dependencias pesadas, y se prueba como el resto.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from fraude import evaluacion

CUBETAS = 10
EPS = 1e-4
REGLAS: dict[str, dict[str, object]] = {
    "psi_variable": {"aviso": 0.10, "critica": 0.25},
    "psi_puntaje": {"aviso": 0.10, "critica": 0.25},
    "faltantes_grupo": {"aviso": 0.10, "critica": 0.20},
    "tasa_alertas": {"aviso": (0.7, 1.3), "critica": (0.5, 1.5)},
    "ap_madura_nuevos": {"aviso": 0.85, "critica": 0.70},
}


def cortes_psi(referencia: pd.Series) -> np.ndarray | None:
    """Bordes de las cubetas (``None``: una cubeta por valor)."""
    r = referencia.dropna().to_numpy(dtype="float64")
    if len(np.unique(r)) <= 20:
        return None
    return np.unique(np.quantile(r, np.linspace(0, 1, CUBETAS + 1)[1:-1]))


def _distribucion(x: pd.Series, bordes: np.ndarray | None, valores: np.ndarray) -> np.ndarray:
    falt = x.isna().to_numpy()
    v = x.to_numpy(dtype="float64")[~falt]
    if bordes is None:
        idx = np.searchsorted(valores, v)
        idx = np.where(
            (idx < len(valores)) & (valores[np.minimum(idx, len(valores) - 1)] == v),
            idx,
            len(valores),
        )  # valores nuevos: cubeta propia
        cuenta = np.bincount(idx, minlength=len(valores) + 1).astype(float)
    else:
        cuenta = np.bincount(
            np.searchsorted(bordes, v, side="right"), minlength=len(bordes) + 1
        ).astype(float)
    cuenta = np.append(cuenta, falt.sum())
    return cuenta / max(1.0, cuenta.sum())


def psi(referencia: pd.Series, actual: pd.Series) -> float:
    """PSI entre dos muestras de una variable (los faltantes son una cubeta más)."""
    bordes = cortes_psi(referencia)
    valores = np.unique(referencia.dropna().to_numpy(dtype="float64"))
    p = _distribucion(referencia, bordes, valores) + EPS
    q = _distribucion(actual, bordes, valores) + EPS
    return float(np.sum((q - p) * np.log(q / p)))


def nivel(valor: float, regla: str) -> str | None:
    """``"critica"``, ``"aviso"`` o ``None`` según ``REGLAS``."""
    r = REGLAS[regla]
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return None
    if regla == "tasa_alertas":
        bajo, alto = r["critica"]
        if not bajo <= valor <= alto:
            return "critica"
        bajo, alto = r["aviso"]
        return None if bajo <= valor <= alto else "aviso"
    if regla == "ap_madura_nuevos":
        return "critica" if valor < r["critica"] else ("aviso" if valor < r["aviso"] else None)
    return "critica" if valor >= r["critica"] else ("aviso" if valor >= r["aviso"] else None)


def faltantes(df: pd.DataFrame, grupos: dict[str, Sequence[str]]) -> dict[str, float]:
    return {g: float(df[list(c)].isna().to_numpy().mean()) for g, c in grupos.items()}


def semana(
    referencia: pd.DataFrame,
    actual: pd.DataFrame,
    s_ref: np.ndarray,
    s_act: np.ndarray,
    umbral: float,
    tasa_esperada: float,
    variables: Sequence[str],
    grupos: dict[str, Sequence[str]],
) -> dict[str, object]:
    """Deriva y operación de una semana (sin etiquetas)."""
    psis = {c: psi(referencia[c], actual[c]) for c in variables}
    f_ref, f_act = faltantes(referencia, grupos), faltantes(actual, grupos)
    tasa = float((s_act > umbral).mean())
    return {
        "transacciones": int(len(actual)),
        "psi_variables": psis,
        "psi_puntaje": psi(pd.Series(s_ref), pd.Series(s_act)),
        "faltantes": {
            g: {"referencia": f_ref[g], "actual": f_act[g], "cambio": f_act[g] - f_ref[g]}
            for g in grupos
        },
        "tasa_alertas": tasa,
        "tasa_alertas_relativa": tasa / tasa_esperada,
    }


def rendimiento_maduro(
    y: np.ndarray, s: np.ndarray, nuevo: np.ndarray, umbral: float
) -> dict[str, object]:
    """AP y recall/precisión al umbral, para todo, nuevos y vistos (solo filas maduras)."""
    salida = {}
    for nombre, m in (("todos", np.ones(len(y), bool)), ("nuevos", nuevo), ("vistos", ~nuevo)):
        yy, ss = y[m], s[m]
        marcadas = ss > umbral
        vp = int((marcadas & (yy == 1)).sum())
        salida[nombre] = {
            "transacciones": int(m.sum()),
            "fraudes": int(yy.sum()),
            "ap": evaluacion.ap(yy, ss) if yy.any() else None,
            "recall_al_umbral": vp / max(1, int(yy.sum())),
            "precision_al_umbral": vp / max(1, int(marcadas.sum())),
        }
    return salida


def alertas(
    numero_semana: int,
    datos: dict[str, object],
    ap_validacion_nuevos: float,
) -> list[dict[str, object]]:
    """Alertas de una semana según ``REGLAS``."""
    salida = []

    def agregar(regla: str, objeto: str, valor: float) -> None:
        n = nivel(valor, regla)
        if n:
            salida.append(
                {
                    "semana": numero_semana,
                    "nivel": n,
                    "regla": regla,
                    "objeto": objeto,
                    "valor": valor,
                }
            )

    for c, v in datos["psi_variables"].items():
        agregar("psi_variable", c, v)
    agregar("psi_puntaje", "puntaje", datos["psi_puntaje"])
    for g, f in datos["faltantes"].items():
        agregar("faltantes_grupo", g, abs(f["cambio"]))
    agregar("tasa_alertas", "tasa de alertas / esperada", datos["tasa_alertas_relativa"])
    madura = datos.get("rendimiento_maduro")
    if madura and madura["nuevos"]["ap"] is not None:
        agregar(
            "ap_madura_nuevos",
            "AP madura en nuevos / validación",
            madura["nuevos"]["ap"] / ap_validacion_nuevos,
        )
    return salida
