"""Parámetros congelados con un tag de git antes de medir en prueba.

``verificar(tag, ruta)`` falla si el tag no existe o si el archivo cambió desde el tag. Las
evaluaciones en prueba la llaman antes de leer etiquetas de prueba.
"""

import subprocess
from pathlib import Path

from fraude.rutas import RAIZ


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=RAIZ, capture_output=True, text=True, encoding="utf-8"
    )


def contenido_en_tag(tag: str, ruta: Path) -> str | None:
    """Texto de ``ruta`` en el tag (``None`` si el tag o el archivo no existen)."""
    rel = Path(ruta).resolve().relative_to(RAIZ).as_posix()
    r = _git("show", f"{tag}:{rel}")
    return r.stdout if r.returncode == 0 else None


def verificar(tag: str, ruta: Path) -> None:
    """Falla si ``ruta`` no está congelada en ``tag`` o si cambió desde entonces."""
    en_tag = contenido_en_tag(tag, ruta)
    if en_tag is None:
        raise RuntimeError(f"{ruta} no está congelado en el tag {tag}")
    actual = Path(ruta).read_text(encoding="utf-8")
    if actual.replace("\r\n", "\n") != en_tag.replace("\r\n", "\n"):
        raise RuntimeError(f"{ruta} cambió desde el tag {tag}")
