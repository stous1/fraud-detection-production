"""Fixtures comunes: mini-IEEE sintético cargado en una DuckDB en memoria."""

from pathlib import Path

import duckdb
import pytest

from fraude import base, ingesta, particion, sintetico


@pytest.fixture(scope="session")
def mini() -> sintetico.Mini:
    return sintetico.generar(n_clientes=300, semilla=7)


@pytest.fixture(scope="session")
def csv_mini(mini: sintetico.Mini, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return sintetico.escribir_csv(mini, tmp_path_factory.mktemp("raw"))


@pytest.fixture()
def con(csv_mini: tuple[Path, Path], tmp_path: Path) -> duckdb.DuckDBPyConnection:
    c = base.conectar(":memory:", temporal=tmp_path / "tmp")
    ingesta.cargar(c, *csv_mini)
    particion.crear_vistas(c)
    yield c
    c.close()
