"""Grafo y vecindario: sin información del futuro, etiquetas solo con retraso."""

import numpy as np
import pandas as pd
import pytest

from fraude import clientes, grafo, vecindario


@pytest.fixture()
def con_grafo(con):
    df = clientes.cargar_base(con)
    tabla = pd.DataFrame({"TransactionID": df["TransactionID"], "uid": clientes.asignar(df, "uid")})
    con.register("t_cli", tabla)
    con.execute("CREATE OR REPLACE TABLE clientes AS SELECT * FROM t_cli")
    grafo.crear_entidades(con)
    return con


def _vecindario(con, retrasos=(0, 30)):
    vecindario.crear_etiquetas(con)
    return vecindario.variables_sql(con, retrasos)


def test_entidades_una_fila_por_transaccion(con_grafo):
    n, n_ent = con_grafo.execute(
        "SELECT (SELECT count(*) FROM transacciones), count(*) FROM entidades"
    ).fetchone()
    assert n == n_ent
    resumen = {r["tipo"]: r for r in grafo.resumen(con_grafo)}
    assert set(resumen) == set(grafo.TIPOS)
    assert resumen["cliente"]["max_clientes"] == 1


def test_conteos_basicos(con_grafo):
    v = _vecindario(con_grafo)
    e = con_grafo.execute("SELECT * FROM entidades ORDER BY TransactionID").df()
    esperado = e.groupby("cliente").cumcount().to_numpy()
    assert (v["cliente_tx_previas"].to_numpy() == esperado).all()
    assert (v["tarjeta_clientes_previos"] >= 0).all()


def test_no_mira_el_futuro_sin_etiquetas(con_grafo):
    """Truncar en el día 100 no cambia las variables de lo anterior."""
    completo = _vecindario(con_grafo).set_index("TransactionID")
    con_grafo.execute(
        "CREATE OR REPLACE TABLE entidades AS SELECT * FROM entidades WHERE dia <= 100"
    )
    truncado = _vecindario(con_grafo).set_index("TransactionID")
    pd.testing.assert_frame_equal(
        completo.loc[truncado.index], truncado, check_dtype=False, check_exact=False
    )


def test_etiquetas_recientes_no_cuentan(con_grafo):
    """Cambiar las etiquetas de los últimos R días antes de t no cambia las variables con
    retraso R en t (sí las de R = 0)."""
    t_corte = con_grafo.execute(
        "SELECT quantile_disc(TransactionDT, 0.7) FROM entidades"
    ).fetchone()[0]
    base = _vecindario(con_grafo).set_index("TransactionID")
    con_grafo.execute(
        f"""CREATE OR REPLACE TABLE transacciones AS SELECT * REPLACE (
            CASE WHEN TransactionDT > {t_corte} - 30 * 86400 THEN 1 - isFraud ELSE isFraud END
            AS isFraud) FROM transacciones"""
    )
    cambiado = _vecindario(con_grafo).set_index("TransactionID")
    ids = con_grafo.execute(
        f"SELECT TransactionID FROM entidades WHERE TransactionDT <= {t_corte}"
    ).df()["TransactionID"]
    cols30 = [c for c in base.columns if c.endswith("_r30")]
    pd.testing.assert_frame_equal(base.loc[ids, cols30], cambiado.loc[ids, cols30])
    cols0 = [c for c in base.columns if c.endswith("_r0")]
    assert not base.loc[ids, cols0].equals(cambiado.loc[ids, cols0])


def test_componentes_en_linea_con_retraso():
    # Clientes 0 y 1 comparten huella 7; el cliente 2 está solo.
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [0, 10, 40 * 86400, 41 * 86400],
            "cliente": [0, 1, 2, 1],
            "huella": [7.0, 7.0, np.nan, 7.0],
            "tarjeta": [np.nan, np.nan, np.nan, np.nan],
        }
    )
    y = np.array([1, 0, 0, 0])
    c = vecindario.componentes_en_linea(df, y, retrasos=(0, 30))
    assert list(c["componente_clientes"]) == [1, 2, 1, 2]
    assert list(c["componente_fraudes_r0"]) == [0, 1, 0, 1]  # la propia no cuenta
    assert list(c["componente_fraudes_r30"]) == [0, 0, 0, 1]  # llega a los 30 días
    assert list(c["componente_clientes_fraude_r30"]) == [0, 0, 0, 1]


def test_limite_de_clientes_por_entidad():
    df = pd.DataFrame(
        {
            "TransactionID": range(4),
            "TransactionDT": range(4),
            "cliente": [0, 1, 2, 3],
            "huella": [5.0] * 4,
            "tarjeta": [np.nan] * 4,
        }
    )
    c = vecindario.componentes_en_linea(df, np.zeros(4, int), retrasos=(30,), limite=2)
    assert list(c["componente_clientes"]) == [1, 2, 1, 1]


def test_columnas_por_retraso(con_grafo):
    v = _vecindario(con_grafo)
    sin = vecindario.columnas(v, None)
    assert sin and not any(c.endswith(("_r0", "_r30")) for c in sin)
    assert set(vecindario.columnas(v, 30)) - set(sin) == {c for c in v if c.endswith("_r30")}
