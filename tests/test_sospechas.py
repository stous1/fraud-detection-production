"""Diagnósticos de sospecha sobre el mini-IEEE (propagación y variables con futuro)."""

import numpy as np
import pandas as pd

from fraude import clientes, sospechas


def _base(con):
    df = con.execute(
        "SELECT TransactionID, TransactionDT, dia, ProductCD, card1, addr1, D1, dia - D1 AS D1n, "
        "C1, C13, D10, TransactionAmt, isFraud FROM transacciones "
        "ORDER BY TransactionDT, TransactionID"
    ).df()
    return df, clientes.clave_exacta(df, ["card1", "addr1", "D1n"])


def test_propagacion_detectada(con):
    df, uid = _base(con)
    p = sospechas.propagacion(df, uid)
    # El fraude aislado (sin propagación) diluye la señal, pero sigue muy por encima.
    assert p["p_fraude_con_fraude_previo"] > 10 * p["p_fraude_sin_fraude_previo"]
    cubiertos = [r["fraccion_de_fraudes_cubiertos"] for r in p["regla_fraude_previo"]]
    assert cubiertos == sorted(cubiertos, reverse=True)  # más retraso, menos cobertura


def test_primer_fraude_previo_no_incluye_la_actual():
    df = pd.DataFrame({"TransactionDT": [0, 10, 20], "isFraud": [0, 1, 1]})
    previo = sospechas._primer_fraude_previo(df, np.array([0, 0, 0]))
    assert np.isnan(previo[0]) and np.isnan(previo[1]) and previo[2] == 10


def test_prueba_futuro_separa_agregado_de_contador(con):
    df, uid = _base(con)
    primeras = sospechas.primeras_de_tarjetas_nuevas(df, uid)
    assert len(primeras) > 30
    rho = sospechas.prueba_futuro(primeras, ["C1", "C13", "TransactionAmt"], minimo=30)
    assert "C1" not in rho  # contador en línea: vale 1 en la primera transacción
    assert rho["C13"] > 0.5  # total del cliente en la ventana: usa el futuro
    assert abs(rho["TransactionAmt"]) < 0.3


def test_monotonia_contador(con, mini):
    df, _ = _base(con)
    verdad = df["TransactionID"].map(mini.cliente).to_numpy()
    m = sospechas.monotonia(df, verdad, ["C1", "C13"])
    assert m["C1"]["baja"] == 0.0 and m["C13"]["igual"] == 1.0


def test_grupos_cubren_el_esquema():
    from fraude import esquema

    declaradas = {c for g in sospechas.GRUPOS for c in g.columnas}
    for c in (*esquema.COLUMNAS_TX, *esquema.COLUMNAS_ID):
        if c not in ("TransactionID", "isFraud", "TransactionDT"):
            assert c in declaradas, c
