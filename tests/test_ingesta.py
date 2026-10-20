"""Ingesta: tipos, verificaciones y rechazo de los archivos de prueba de la competencia."""

from pathlib import Path

import pytest

from fraude import esquema, ingesta, rutas


def test_carga_y_verificacion(con, mini):
    info = ingesta.verificar(con)
    assert info["transacciones"] == len(mini.transacciones)
    assert info["identidades"] == len(mini.identidad)
    assert info["id_desordenados_en_tiempo"] == 0
    tipos = dict(
        con.execute("SELECT column_name, column_type FROM (DESCRIBE transacciones)").fetchall()
    )
    assert tipos["card2"] == "INTEGER" and tipos["TransactionAmt"] == "DOUBLE"
    assert tipos["M4"] == "VARCHAR" and tipos["dia"] == "INTEGER"


def test_dia_derivado(con):
    malos = con.execute(
        "SELECT count(*) FROM transacciones WHERE dia <> TransactionDT // 86400"
    ).fetchone()[0]
    assert malos == 0


@pytest.mark.parametrize(
    "nombre", ["test_transaction.csv", "test_identity.csv", "sample_submission.csv"]
)
def test_rechaza_archivos_de_prueba(nombre):
    with pytest.raises(ValueError):
        ingesta.verificar_origen(Path(nombre))


def test_codigo_no_menciona_archivos_de_prueba():
    """Ningún archivo de src/ ni scripts/ apunta a los CSV sin etiqueta de la competencia."""
    prohibidos = ("test_transaction", "test_identity", "sample_submission")
    for carpeta in (rutas.RAIZ / "src", rutas.RAIZ / "scripts"):
        for py in carpeta.rglob("*.py"):
            texto = py.read_text(encoding="utf-8")
            if py.name == "ingesta.py":
                continue  # solo los nombra para rechazarlos
            assert not any(p in texto for p in prohibidos), py


def test_esquema_completo():
    assert len(esquema.COLUMNAS_TX) == 394 and len(esquema.COLUMNAS_ID) == 41
    assert len(set(esquema.COLUMNAS_TX)) == 394
