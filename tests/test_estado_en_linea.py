"""Paridad exacta entre el estado en línea (servicio) y las variables por lotes (entrenamiento)."""

import math

import numpy as np
import pandas as pd
import pytest

from fraude import clientes, esquema, estado_en_linea, flujo_hito2, vecindario


def transacciones_crudas(con) -> pd.DataFrame:
    ident = ", ".join(f'i."{c}"' for c in esquema.COLUMNAS_ID if c != "TransactionID")
    return con.execute(
        f"SELECT t.*, {ident} FROM transacciones t LEFT JOIN identidad i USING (TransactionID) "
        "ORDER BY t.TransactionID"
    ).df()


def recorrer(estado: estado_en_linea.Estado, crudas: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for tx in crudas.to_dict(orient="records"):
        v = estado.procesar(tx)
        if tx["isFraud"] == 1:  # oráculo: el contracargo solo cuenta a los 30 días
            estado.reportar_fraude(int(tx["TransactionID"]))
        filas.append({"TransactionID": tx["TransactionID"], **v})
    return pd.DataFrame(filas)


@pytest.fixture()
def lotes_y_crudas(con):
    df = clientes.cargar_base(con)
    tabla = pd.DataFrame({"TransactionID": df["TransactionID"], "uid": clientes.asignar(df, "uid")})
    con.register("t_cli", tabla)
    con.execute("CREATE OR REPLACE TABLE clientes AS SELECT * FROM t_cli")
    flujo_hito2.preparar(con)
    lotes = con.execute("SELECT * FROM vecindario ORDER BY TransactionID").df()
    return lotes, transacciones_crudas(con)


def test_paridad_exacta_con_el_vecindario_por_lotes(lotes_y_crudas):
    lotes, crudas = lotes_y_crudas
    en_linea = recorrer(estado_en_linea.Estado(retraso=30), crudas)
    columnas = vecindario.columnas(lotes, 30)
    assert set(columnas) <= set(en_linea.columns), set(columnas) - set(en_linea.columns)
    a = lotes.set_index("TransactionID")[columnas].astype("float64")
    b = en_linea.set_index("TransactionID")[columnas].astype("float64")
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_contracargo_tardio_cuenta_al_llegar():
    est = estado_en_linea.Estado(retraso=30)
    base = {"card1": 1, "addr1": 2, "D1": 0.0, "P_emaildomain": "x.com"}
    est.procesar({**base, "TransactionID": 1, "TransactionDT": 86_400})
    est.reportar_fraude(1)
    antes = est.procesar({**base, "TransactionID": 2, "TransactionDT": 86_400 * 10, "D1": 9.0})
    despues = est.procesar({**base, "TransactionID": 3, "TransactionDT": 86_400 * 32, "D1": 31.0})
    assert antes["cliente_fraudes_r30"] == 0 and antes["componente_fraudes_r30"] == 0
    assert despues["cliente_fraudes_r30"] == 1 and despues["componente_fraudes_r30"] == 1
    assert despues["cliente_tx_previas"] == 2


def test_rechaza_fuera_de_orden():
    est = estado_en_linea.Estado()
    est.procesar({"TransactionID": 1, "TransactionDT": 100, "card1": 1})
    with pytest.raises(ValueError):
        est.procesar({"TransactionID": 2, "TransactionDT": 50, "card1": 1})


def test_claves_normalizan_faltantes():
    k = estado_en_linea.claves(
        {"TransactionDT": 86_400 * 5, "card1": 7.0, "addr1": float("nan"), "D1": 2.0}
    )
    assert k["cliente"] == (7, None, 3) and k["huella"] is None and k["direccion"] is None
    assert not any(isinstance(x, float) and math.isnan(x) for x in k["tarjeta"] if x is not None)
    assert np.isnan(float("nan"))
