"""Verifica que GitHub Pages sirva exactamente lo publicado en ``docs/``.

Pide cada archivo de ``docs/`` a la URL pública (sin credenciales) y compara su SHA-256 con el
local, normalizando los finales de línea de los archivos de texto (git los guarda con LF).
Sale con código 1 si alguno falta o difiere.

Motivo: en el hito 3, GitHub no lanzó la compilación de Pages tras un push y el sitio quedó en
la versión anterior sin ningún aviso.

Uso:
    uv run python scripts/verificar_pages.py
"""

import hashlib
import sys
import urllib.error
import urllib.request

from fraude.rutas import DOCS

BASE = "https://stous1.github.io/fraud-detection-production/"
TEXTO = {".json", ".html", ".md", ".css", ".js", ".csv", ".txt", ".svg"}


def _huella(datos: bytes, texto: bool) -> str:
    if texto:
        datos = datos.replace(b"\r\n", b"\n")
    return hashlib.sha256(datos).hexdigest()


def main() -> int:
    fallas = 0
    archivos = sorted(p for p in DOCS.rglob("*") if p.is_file())
    for ruta in archivos:
        rel = ruta.relative_to(DOCS).as_posix()
        texto = ruta.suffix in TEXTO
        local = _huella(ruta.read_bytes(), texto)
        try:
            with urllib.request.urlopen(f"{BASE}{rel}?v={local[:8]}", timeout=30) as r:
                remoto = _huella(r.read(), texto)
            estado = "ok" if remoto == local else "DIFIERE"
        except urllib.error.HTTPError as e:
            estado = f"HTTP {e.code}"
        if estado != "ok":
            fallas += 1
            print(f"{estado:8s} {rel}")
    print(f"{len(archivos) - fallas} de {len(archivos)} archivos coinciden con Pages")
    return 1 if fallas else 0


if __name__ == "__main__":
    sys.exit(main())
