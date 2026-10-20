"""Hito 4: producciÃ³n y monitoreo.

Uso:
    uv run python scripts/hito4.py paridad    # en línea contra lotes + instantánea del día 120
    uv run python scripts/hito4.py registrar  # modelos/registro.json (producción y respaldo)
    uv run python scripts/hito4.py simular    # semanas de prueba una a una: deriva y alertas
    uv run python scripts/hito4.py graficos   # gráficos desde monitoreo.json
"""

import argparse

from fraude import flujo_hito4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paso", choices=flujo_hito4.PASOS)
    args = parser.parse_args()
    flujo_hito4.PASOS[args.paso]()


if __name__ == "__main__":
    main()
