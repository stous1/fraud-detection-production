"""Partición temporal fija (verificada contra los valores decididos antes de mirar etiquetas)."""

import pytest

from fraude import particion


def test_cortes_fijos():
    assert particion.ULTIMO_DIA_ENTRENAMIENTO == 90
    assert particion.ULTIMO_DIA_DESARROLLO == 120
    assert particion.ULTIMO_DIA == 182


@pytest.mark.parametrize(
    ("dia", "periodo", "semana"),
    [
        (1, "entrenamiento", None),
        (90, "entrenamiento", None),
        (91, "validacion", None),
        (120, "validacion", None),
        (121, "prueba", 1),
        (127, "prueba", 1),
        (128, "prueba", 2),
        (176, "prueba", 8),
        (177, "prueba", 9),
        (182, "prueba", 9),
    ],
)
def test_periodo_y_semana(dia, periodo, semana):
    assert particion.periodo(dia) == periodo
    assert particion.semana_prueba(dia) == semana


def test_sql_coincide_con_python(con):
    filas = con.execute("SELECT DISTINCT dia, periodo, semana_prueba, mes FROM tiempo").fetchall()
    for dia, periodo, semana, mes in filas:
        assert periodo == particion.periodo(dia)
        assert semana == particion.semana_prueba(dia)
        assert mes == particion.mes(dia)


def test_vista_desarrollo_sin_prueba(con):
    assert con.execute("SELECT max(dia) FROM desarrollo").fetchone()[0] <= 120
