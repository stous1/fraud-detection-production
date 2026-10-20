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


def test_esquema_realista_solo_usa_etiquetas_conocidas_al_desplegar():
    """Despliegue el día 121 con 30 días de retraso: etiquetas hasta el día 90."""
    assert particion.ULTIMO_DIA_ETIQUETADO == 90
    for tramo in particion.ESQUEMAS["realista"].values():
        assert tramo[1] <= particion.ULTIMO_DIA_ETIQUETADO
    assert particion.ESQUEMA_PRINCIPAL == "realista"
    for esquema in particion.ESQUEMAS.values():
        ent, val = esquema["entrenamiento"], esquema["validacion"]
        assert ent[1] + 1 == val[0]  # validación justo después del entrenamiento
        assert esquema["reajuste"] == (ent[0], val[1])
        assert val[1] < particion.PRUEBA[0]
