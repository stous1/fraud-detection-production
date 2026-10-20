"""Métricas: AP, recall a tasa fija de falsas alarmas, bootstrap por cliente y segmentos."""

import numpy as np
import pytest

from fraude import evaluacion


def test_recall_a_fpr_exacto():
    y = np.array([0] * 100 + [1] * 10)
    s = np.concatenate([np.arange(100) / 100, np.full(10, 0.995)])
    r = evaluacion.recall_a_fpr(y, s, 0.01)
    assert r["fpr_real"] <= 0.01 and r["recall"] == 1.0


def test_ap_perfecto_y_aleatorio():
    y = np.array([0, 0, 1, 1])
    assert evaluacion.ap(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    rng = np.random.default_rng(0)
    y = rng.random(20_000) < 0.05
    assert evaluacion.ap(y, rng.random(20_000)) == pytest.approx(0.05, abs=0.01)


def test_intervalo_contiene_la_ap_y_es_determinista():
    rng = np.random.default_rng(1)
    g = rng.integers(500, size=5000)
    y = (rng.random(5000) < 0.1).astype(int)
    s = y + rng.normal(0, 1, 5000)
    a = evaluacion.intervalo_ap(y, s, g, repeticiones=50)
    assert a == evaluacion.intervalo_ap(y, s, g, repeticiones=50)
    assert a[0] < evaluacion.ap(y, s) < a[1]


def test_por_segmento():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    r = evaluacion.por_segmento(y, s, np.array([True, True, False, False]), intervalo=False)
    assert r["todos"]["transacciones"] == 4 and r["vistos"]["fraudes"] == 1


def test_curva_pr_monotona():
    y = np.array([0, 1, 0, 1, 1])
    c = evaluacion.curva_pr(y, np.array([0.1, 0.9, 0.5, 0.4, 0.7]), puntos=10)
    precisiones = [p["precision"] for p in c]
    assert precisiones == sorted(precisiones, reverse=True)
