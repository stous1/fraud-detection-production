"""Hito 3: costos, codificador, XGBoost guardado y puntuado en CPU, y flujo en el mini-IEEE."""

import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from fraude import (
    clientes,
    congelados,
    costos,
    flujo_hito2,
    flujo_hito3,
    modelo_xgb,
    rutas,
    variables,
)


def test_costo_y_tasa():
    y = np.array([1, 0, 0, 1, 0])
    monto = np.array([100.0, 5, 5, 50, 5])
    c = costos.costo(y, np.array([True, True, False, False, False]), monto, 10.0)
    assert c["costo_total"] == 50 + 10 and c["recall"] == 0.5 and c["falsos_positivos"] == 1
    s = np.array([0.9, 0.1, 0.2, 0.8, 0.3])
    e = costos.elegir_tasa(y, s, monto, 10.0, tasas=(0.2, 0.4, 0.6))
    assert e["tasa"] == 0.4 and e["mejor"]["costo_total"] == 0.0
    assert e["sin_modelo"]["costo_total"] == 150.0


def test_umbral_de_tasa():
    s = np.arange(100) / 100
    assert (s > costos.umbral_de_tasa(s, 0.1)).sum() == 10
    assert (s > costos.umbral_de_tasa(s, 1.0)).sum() == 100


def test_codificador_por_frecuencia_y_valores_nuevos():
    df = pd.DataFrame({"ProductCD": ["W", "W", "C", None, "H"]})
    cod = variables.Codificador.ajustar(df, ("ProductCD",))
    assert cod.mapas["ProductCD"] == ["W", "C", "H"]  # empate C-H: alfabético
    out = cod.aplicar(pd.DataFrame({"ProductCD": ["C", "Z", None]}))["ProductCD"]
    assert out.iloc[0] == 1 and out.iloc[1] == 3 and np.isnan(out.iloc[2])


def test_matriz_rechaza_columnas_prohibidas():
    with pytest.raises(AssertionError):
        variables.matriz(pd.DataFrame({"dia": [1]}), ["dia"], variables.Codificador())


def test_xgb_guardado_en_cpu_reproduce(tmp_path):
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.random((2000, 5)), columns=list("abcde")).astype("float32")
    y = (x["a"] + rng.normal(0, 0.2, 2000) > 0.8).astype(int).to_numpy()
    hiper = modelo_xgb.configuraciones(1)[0]
    m, n = modelo_xgb.entrenar(x, y, hiper, "cpu", (x, y))
    sha = modelo_xgb.guardar(m, tmp_path / "m.ubj", n)
    cargado = modelo_xgb.cargar(tmp_path / "m.ubj", sha)
    a = modelo_xgb.puntuar(cargado, x)
    assert np.allclose(a, modelo_xgb.puntuar_en_memoria(m, x, n))
    assert (a == modelo_xgb.puntuar(modelo_xgb.cargar(tmp_path / "m.ubj"), x)).all()
    contrib = modelo_xgb.contribuciones(cargado, x.head(50))
    logit = np.log(a[:50] / (1 - a[:50]))
    assert np.allclose(contrib.sum(axis=1), logit, atol=1e-4)
    with pytest.raises(RuntimeError):
        modelo_xgb.cargar(tmp_path / "m.ubj", "0" * 64)


def test_configuraciones_deterministas_y_distintas():
    a = modelo_xgb.configuraciones(10)
    assert a == modelo_xgb.configuraciones(10)
    assert len({tuple(sorted(c.items())) for c in a}) == 10


def test_variantes_y_grupos():
    nombres = [v.nombre for v in flujo_hito3.VARIANTES]
    assert flujo_hito3.PRINCIPAL in nombres and len(set(nombres)) == len(nombres)
    g = variables.grupos(["cliente_fraudes_r30", "huella_tx_previas", "cliente_fraudes_r0"])
    assert g["historia"] == variables.HISTORIA and g["vecindario"] == ("huella_tx_previas",)
    assert not set(variables.TRANSACCION) & set(variables.SOSPECHOSAS)


def test_parametros_congelados_no_cambiaron():
    existe = subprocess.run(
        ["git", "tag", "-l", flujo_hito3.TAG], cwd=rutas.RAIZ, capture_output=True, text=True
    ).stdout.strip()
    if not existe:
        pytest.skip("hito 3 aún no congelado")
    congelados.verificar(flujo_hito3.TAG, flujo_hito3.PARAMETROS_PUBLICADOS)


@pytest.mark.integracion
def test_flujo_completo(con, tmp_path):
    df = clientes.cargar_base(con)
    tabla = pd.DataFrame({"TransactionID": df["TransactionID"], "uid": clientes.asignar(df, "uid")})
    con.register("t_cli", tabla)
    con.execute("CREATE OR REPLACE TABLE clientes AS SELECT * FROM t_cli")
    flujo_hito2.preparar(con)
    flujo_hito3.buscar(con, tmp_path, n=2, gpu=False)
    flujo_hito3.desarrollar(con, tmp_path, gpu=False)
    flujo_hito3.evaluar(con, tmp_path, tmp_path / "parametros.json", tmp_path / "modelos", tag=None)
    ev = json.loads((tmp_path / "evaluacion.json").read_text(encoding="utf-8"))
    assert set(ev["variantes"]) == {v.nombre for v in flujo_hito3.VARIANTES}
    assert set(ev["aportes_por_grupo"]["nuevos"]["dejar_uno_fuera"]) >= set(flujo_hito3.TODOS)
    assert "nuevos_W" in ev["variantes"]["completo"]["prueba_cortes"]
    p = json.loads((tmp_path / "parametros.json").read_text(encoding="utf-8"))
    for v in p["variantes"].values():
        assert not set(v["columnas"]) & {"isFraud", "dia", "TransactionDT", "cliente"}
    assert "10.0" in p["variantes"]["completo"]["umbrales_por_costo_fp"]
    assert (tmp_path / "shap_nuevos.png").exists()
