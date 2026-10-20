"""Rutas del proyecto, centralizadas y relativas a la raíz del repositorio."""

from pathlib import Path

RAIZ: Path = Path(__file__).resolve().parents[2]

DATA: Path = RAIZ / "data"
DATA_RAW: Path = DATA / "raw"
CSV_TRANSACCIONES: Path = DATA_RAW / "train_transaction.csv"
CSV_IDENTIDAD: Path = DATA_RAW / "train_identity.csv"
BASE_DUCKDB: Path = DATA / "fraude.duckdb"
DATA_TMP: Path = DATA / "tmp"

REPORTS: Path = RAIZ / "reports"
DOCS: Path = RAIZ / "docs"
DOCS_RESULTADOS: Path = DOCS / "resultados"


def reports_hito(n: int) -> Path:
    """Carpeta de trabajo (no versionada) del hito ``n``."""
    return REPORTS / f"hito{n}"


def docs_hito(n: int) -> Path:
    """Carpeta publicada (versionada, Pages) del hito ``n``."""
    return DOCS_RESULTADOS / f"hito{n}"
