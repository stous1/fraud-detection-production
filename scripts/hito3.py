"""Hito 3: XGBoost en GPU, ablaciones por grupo, umbral por costo y SHAP.

Uso:
    uv run python scripts/hito3.py buscar           # hiperparámetros en GPU (validación 61-90)
    uv run python scripts/hito3.py desarrollar      # variantes, umbral por costo y SHAP (GPU)
    uv run python scripts/hito3.py publicar-modelo  # modelo principal -> docs/resultados/hito3
    uv run python scripts/hito3.py evaluar          # prueba, una vez, en CPU, con hito3-congelado
    uv run python scripts/hito3.py graficos         # gráficos desde evaluacion.json
"""

import argparse

from fraude import flujo_hito3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paso", choices=flujo_hito3.PASOS)
    args = parser.parse_args()
    flujo_hito3.PASOS[args.paso]()


if __name__ == "__main__":
    main()
