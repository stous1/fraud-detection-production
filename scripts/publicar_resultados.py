"""Copia los resultados finales de ``reports/hitoN`` a ``docs/resultados/hitoN`` (Pages).

Uso:
    uv run python scripts/publicar_resultados.py --hito 2 --solo-parametros  # para congelar
    uv run python scripts/publicar_resultados.py --hito 2                    # todo, tras evaluar

Publica los ``.json``, ``.png`` y ``.html`` de la carpeta del hito (sin subcarpetas salvo
``modelos/``). Los archivos congelados (``parametros.json`` y ``modelos/``) no se pisan si
ya existen, salvo con ``--solo-parametros``.
"""

import argparse
import shutil

from fraude.rutas import docs_hito, reports_hito

CONGELADOS = ("parametros.json",)
EXCLUIDOS: tuple[str, ...] = ()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hito", required=True, type=int, choices=range(1, 6))
    parser.add_argument("--solo-parametros", action="store_true")
    args = parser.parse_args()

    origen, destino = reports_hito(args.hito), docs_hito(args.hito)
    if not origen.exists():
        raise SystemExit(f"No existe {origen}")
    if args.solo_parametros:
        archivos = [origen / n for n in CONGELADOS if (origen / n).exists()]
        archivos += (
            sorted((origen / "modelos").glob("*.json")) if (origen / "modelos").exists() else []
        )
    else:
        archivos = [
            a
            for a in sorted(origen.iterdir())
            if a.is_file() and a.suffix in (".json", ".png", ".html") and a.name not in EXCLUIDOS
        ]
    for a in archivos:
        rel = a.relative_to(origen)
        dest = destino / rel
        congelado = rel.name in CONGELADOS or rel.parts[0] == "modelos"
        if congelado and dest.exists() and not args.solo_parametros:
            continue  # lo congelado no se pisa al publicar el resto
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a, dest)
        print(f"{a} -> {dest}")


if __name__ == "__main__":
    main()
