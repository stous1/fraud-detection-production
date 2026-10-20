"""Hito 2: modelo lineal guardable, congelamiento y flujo completo sobre el mini-IEEE."""

import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from fraude import clientes, congelados, flujo_hito2, modelo_lineal, rutas


def test_modelo_lineal_se_reproduce_desde_json():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"a": rng.poisson(2, 500).astype(float), "b": rng.random(500)})
    df.loc[::7, "b"] = np.nan
    y = (df["a"] > 3).astype(int).to_numpy()
    m = modelo_lineal.entrenar(df, y, ["a", "b"], C=1.0)
    assert m.indicadores == ["b"]
    copia = modelo_lineal.ModeloLineal.desde_dict(json.loads(json.dumps(m.a_dict())))
    assert (modelo_lineal.puntuar(copia, df) == modelo_lineal.puntuar(m, df)).all()


def test_variantes_declaradas():
    nombres = [v.nombre for v in flujo_hito2.VARIANTES]
    assert flujo_hito2.PRINCIPAL in nombres and len(set(nombres)) == len(nombres)
    principal = next(v for v in flujo_hito2.VARIANTES if v.nombre == flujo_hito2.PRINCIPAL)
    assert principal.esquema == "realista" and principal.retraso == 30


def test_verificar_tag_inexistente(tmp_path):
    ruta = rutas.RAIZ / "pyproject.toml"
    with pytest.raises(RuntimeError):
        congelados.verificar("tag-que-no-existe-xyz", ruta)


def test_parametros_congelados_no_cambiaron():
    """Si el hito 2 ya se congeló, los parámetros publicados son los del tag."""
    existe = subprocess.run(
        ["git", "tag", "-l", flujo_hito2.TAG], cwd=rutas.RAIZ, capture_output=True, text=True
    ).stdout.strip()
    if not existe:
        pytest.skip("hito 2 aún no congelado")
    congelados.verificar(flujo_hito2.TAG, flujo_hito2.PARAMETROS_PUBLICADOS)


@pytest.mark.integracion
def test_flujo_completo(con, tmp_path):
    df = clientes.cargar_base(con)
    tabla = pd.DataFrame({"TransactionID": df["TransactionID"], "uid": clientes.asignar(df, "uid")})
    con.register("t_cli", tabla)
    con.execute("CREATE OR REPLACE TABLE clientes AS SELECT * FROM t_cli")
    flujo_hito2.preparar(con)
    flujo_hito2.desarrollar(con, tmp_path, grilla=(0.1,))
    flujo_hito2.evaluar(con, tmp_path, tmp_path / "parametros.json", tag=None)
    ev = json.loads((tmp_path / "evaluacion.json").read_text(encoding="utf-8"))
    assert set(ev["variantes"]) == {v.nombre for v in flujo_hito2.VARIANTES}
    r = ev["variantes"]["vecindario_r30"]["prueba"]
    assert set(r) == {"todos", "nuevos", "vistos"}
    # En el mini-IEEE la etiqueta se propaga: los clientes ya vistos se detectan mejor.
    assert r["vistos"]["ap"] > r["vistos"]["prevalencia"]
    assert (tmp_path / "ap_prueba.png").exists()


@pytest.mark.integracion
def test_parametros_sin_etiqueta(con, tmp_path):
    df = clientes.cargar_base(con)
    tabla = pd.DataFrame({"TransactionID": df["TransactionID"], "uid": clientes.asignar(df, "uid")})
    con.register("t_cli", tabla)
    con.execute("CREATE OR REPLACE TABLE clientes AS SELECT * FROM t_cli")
    flujo_hito2.preparar(con)
    flujo_hito2.desarrollar(con, tmp_path, grilla=(0.1,))
    p = json.loads((tmp_path / "parametros.json").read_text(encoding="utf-8"))
    for v in p["variantes"].values():
        if "modelo" in v:
            assert not set(v["modelo"]["columnas"]) & {"isFraud", "dia", "cliente"}
