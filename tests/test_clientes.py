"""Reconstrucción del cliente sobre el mini-IEEE con clientes plantados."""

import numpy as np
import pandas as pd
import pytest

from fraude import clientes, splink_ref


@pytest.fixture()
def base_mini(con, mini):
    df = clientes.cargar_base(con)
    verdad = df["TransactionID"].map(mini.cliente).to_numpy()
    return df, verdad


def _pureza(ids: np.ndarray, verdad: np.ndarray) -> float:
    """Fracción de pares unidos por ``ids`` que son del mismo cliente verdadero."""
    return splink_ref.comparar_pares(ids, verdad)["precision"]


def _cobertura(ids: np.ndarray, verdad: np.ndarray) -> float:
    return splink_ref.comparar_pares(ids, verdad)["recall"]


def test_uid_recupera_clientes(base_mini):
    df, verdad = base_mini
    uid = clientes.asignar(df, "uid")
    assert _pureza(uid, verdad) > 0.99
    assert _cobertura(uid, verdad) > 0.99


def test_tolerancia_une_tarjetas_hermanas(base_mini):
    df, verdad = base_mini
    tol = clientes.asignar(df, "uid_tolerancia_1")
    assert _pureza(tol, verdad) < _pureza(clientes.asignar(df, "uid"), verdad)


def test_veto_correo_parte_clientes(base_mini):
    df, verdad = base_mini
    veto = clientes.asignar(df, "uid_veto_correo")
    assert _cobertura(veto, verdad) < 0.97


def test_en_linea_sin_tolerancia_es_la_clave_exacta(base_mini):
    df, _ = base_mini
    a = clientes.asignar_en_linea(df, tolerancia=0)
    b = clientes.asignar(df, "uid")
    assert splink_ref.comparar_pares(a, b) == {**splink_ref.comparar_pares(a, b), "precision": 1.0}
    assert splink_ref.comparar_pares(a, b)["recall"] == 1.0


@pytest.mark.parametrize("nivel", clientes.NIVELES)
def test_no_mira_el_futuro(base_mini, nivel):
    """Truncar los datos en un día no cambia la asignación de lo anterior."""
    df, _ = base_mini
    completo = clientes.asignar(df, nivel)
    corte = df["dia"] <= 100
    truncado = clientes.asignar(df[corte].reset_index(drop=True), nivel)
    assert (completo[corte.to_numpy()] == truncado).all()


def test_cambios_ubica_pares_entre_referencias(base_mini):
    df, _ = base_mini
    refs = clientes.referencias(df, ["D10n", "D15n"])
    assert refs["D10n"]["mismo_uid"] > 0.95 > 0.2 > refs["D10n"]["uid_distinto"]
    uid = clientes.asignar(df, "uid")
    tol = clientes.cambios(df, uid, clientes.asignar(df, "uid_tolerancia_1"), refs)
    veto = clientes.cambios(df, uid, clientes.asignar(df, "uid_veto_correo"), refs)
    assert tol["agregados"]["pares"] > 0 and veto["separados"]["pares"] > 0
    assert tol["agregados"]["D10n_fraccion_mismo_cliente"] < 0.2
    assert veto["separados"]["D10n_fraccion_mismo_cliente"] > 0.8


def test_fragmentacion():
    f = clientes.fragmentacion(np.array([0, 0, 1, 2, 2, 2]))
    assert f["clientes"] == 3 and f["max_tamano"] == 3
    assert f["fraccion_clientes_unicos"] == pytest.approx(1 / 3)


def test_comparar_pares():
    a = np.array([0, 0, 0, 1])
    b = np.array([0, 0, 1, 1])
    r = splink_ref.comparar_pares(a, b)
    assert (r["pares_a"], r["pares_b"], r["pares_ambos"]) == (3, 2, 1)


def test_concordancia_ignora_faltantes():
    df = pd.DataFrame({"x": [1.0, np.nan, 1.0, 2.0]})
    c = clientes.concordancia(df, np.array([0, 0, 0, 0]), ["x"])
    assert c["x"] == {"pares": 1, "concordancia": 0.0}


@pytest.mark.integracion
def test_splink_resuelve_el_mini(base_mini):
    df, verdad = base_mini
    ids, info = splink_ref.resolver(df, max_pares_u=2e5)
    assert len(ids) == len(df) and info["pares_sobre_umbral"] > 0
    assert _pureza(ids, verdad) > 0.9
