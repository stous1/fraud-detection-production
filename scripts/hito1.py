"""Hito 1: ingesta, exploración, sospechas y reconstrucción del cliente.

Uso:
    uv run python scripts/hito1.py preparar   # CSV de train -> data/fraude.duckdb
"""

import argparse

from fraude import flujo_hito1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paso", choices=flujo_hito1.PASOS)
    args = parser.parse_args()
    flujo_hito1.PASOS[args.paso]()


if __name__ == "__main__":
    main()
