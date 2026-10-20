"""Flujo completo del hito 1 sobre el mini-IEEE y reglas sobre quién lee etiquetas."""

import json

import pytest

from fraude import flujo_hito1, rutas

QUIEN_LEE_ETIQUETAS = {
    "esquema.py",  # declara la columna
    "ingesta.py",  # verifica que sea 0 o 1
    "particion.py",  # solo la nombra en la documentación
    "sintetico.py",  # la genera
    "exploracion.py",  # solo sobre la vista desarrollo
    "sospechas.py",  # propagación, solo sobre desarrollo
    "flujo_hito1.py",  # pasa a sospechas la vista desarrollo
}


def test_solo_modulos_declarados_leen_etiquetas():
    for py in (rutas.RAIZ / "src" / "fraude").glob("*.py"):
        if "isFraud" in py.read_text(encoding="utf-8"):
            assert py.name in QUIEN_LEE_ETIQUETAS, py.name


def test_etiquetas_del_flujo_vienen_de_desarrollo():
    texto = (rutas.RAIZ / "src" / "fraude" / "flujo_hito1.py").read_text(encoding="utf-8")
    for linea in texto.splitlines():
        if "isFraud" in linea:
            assert "d.isFraud" in linea, linea  # alias de la vista desarrollo


@pytest.mark.integracion
def test_flujo_completo(con, tmp_path):
    flujo_hito1.explorar(con, tmp_path)
    flujo_hito1.paso_clientes(con, tmp_path, umbrales_splink=(0.9,))
    flujo_hito1.paso_sospechas(con, tmp_path)
    exp = json.loads((tmp_path / "exploracion.json").read_text(encoding="utf-8"))
    assert exp["desarrollo"]["transacciones"] > 0
    cli = json.loads((tmp_path / "clientes.json").read_text(encoding="utf-8"))
    assert set(cli["niveles"]) >= {"uid", "uid_tolerancia_1", "uid_veto_correo"}
    assert 0 < cli["clientes_en_prueba"]["fraccion_tx_prueba_cliente_visto_en_desarrollo"] < 1
    sos = json.loads((tmp_path / "sospechas.json").read_text(encoding="utf-8"))
    assert sos["propagacion_desarrollo"]["p_fraude_con_fraude_previo"] > 0.3
    for nombre in ("volumen_semanal", "clientes_concordancia", "propagacion_retraso"):
        assert (tmp_path / f"{nombre}.png").exists()
