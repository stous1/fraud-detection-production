"""Hito 2: grafo de identidad, vecindario y cuánto aporta (medido una vez en prueba).

Uso:
    uv run python scripts/hito2.py preparar     # entidades y vecindario -> data/fraude.duckdb
    uv run python scripts/hito2.py desarrollar  # parámetros en reports/hito2 (solo desarrollo)
    uv run python scripts/hito2.py evaluar      # prueba, una vez, con el tag hito2-congelado
"""

import argparse

from fraude import flujo_hito2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paso", choices=flujo_hito2.PASOS)
    args = parser.parse_args()
    flujo_hito2.PASOS[args.paso]()


if __name__ == "__main__":
    main()
