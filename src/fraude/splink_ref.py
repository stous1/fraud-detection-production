"""Referencia fuera de línea para la reconstrucción del cliente: Fellegi-Sunter con Splink.

Splink ve toda la ventana (también el futuro), así que **no alimenta al modelo**: solo sirve
para cuantificar qué tanto se parece el ``uid`` en línea a una resolución de entidades
probabilística entrenada sin etiquetas (EM). Los registros son transacciones: Splink puede
unir ``uid`` distintos (por ejemplo, días con ``D1 = 0``) y también separar un ``uid``.

Comparaciones: ``D1n``, ``D10n``, ``D15n`` (exacto, un día, siete días), correo y producto.
``DeviceInfo`` queda fuera para validar las dos agrupaciones con un atributo que ninguna usa.
"""

import logging

import duckdb
import numpy as np
import pandas as pd
import splink.comparison_level_library as cll
import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on

UMBRAL = 0.9
SEMILLA = 3


def _dias(col: str) -> cl.CustomComparison:
    return cl.CustomComparison(
        comparison_levels=[
            cll.NullLevel(col),
            cll.ExactMatchLevel(col),
            cll.AbsoluteDifferenceLevel(col, 1),
            cll.AbsoluteDifferenceLevel(col, 7),
            cll.ElseLevel(),
        ],
        output_column_name=col,
    )


def configuracion() -> SettingsCreator:
    """Modelo de deduplicación de transacciones dentro de la misma ``card1 + addr1``."""
    return SettingsCreator(
        link_type="dedupe_only",
        unique_id_column_name="TransactionID",
        comparisons=[
            _dias("D1n"),
            _dias("D10n"),
            _dias("D15n"),
            cl.ExactMatch("P_emaildomain"),
            cl.ExactMatch("ProductCD"),
        ],
        blocking_rules_to_generate_predictions=[
            block_on("card1", "addr1", "D1n"),
            block_on("card1", "addr1", "D10n"),
            block_on("card1", "addr1", "D15n"),
        ],
        retain_intermediate_calculation_columns=False,
    )


def resolver(
    df: pd.DataFrame, umbral: float = UMBRAL, max_pares_u: float = 5e6
) -> tuple[np.ndarray, dict[str, object]]:
    """Entrena con EM (sin etiquetas) y agrupa las transacciones; devuelve el cluster de cada
    fila de ``df`` (numerado por orden de aparición) y los parámetros del modelo."""
    columnas = ["TransactionID", "card1", "addr1", "D1n", "D10n", "D15n"]
    entrada = df[columnas + ["P_emaildomain", "ProductCD"]].copy()
    # addr1 faltante como valor propio (igual que en uid): el bloqueo lo compara con "=".
    entrada["addr1"] = entrada["addr1"].fillna(-1)
    con = duckdb.connect()
    con.execute("SET threads = 8")
    db = DuckDBAPI(con)
    tabla = db.register(entrada, table_name="tx")
    linker = Linker(tabla, configuracion(), log_level=logging.WARNING)
    linker.training.estimate_probability_two_random_records_match(
        [block_on("card1", "addr1", "D1n", "D10n")], recall=0.7
    )
    linker.training.estimate_u_using_random_sampling(max_pairs=max_pares_u, seed=SEMILLA)
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("card1", "addr1", "D1n")
    )
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("card1", "addr1", "D10n")
    )
    pares = linker.inference.predict(threshold_match_probability=umbral)
    clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
        pares, threshold_match_probability=umbral
    ).as_pandas_dataframe()[["TransactionID", "cluster_id"]]
    n_pares = con.execute(f"SELECT count(*) FROM {pares.physical_name}").fetchone()[0]
    modelo = linker.misc.save_model_to_json()
    con.close()
    orden = df[["TransactionID"]].merge(clusters, on="TransactionID", how="left")
    ids = pd.factorize(orden["cluster_id"].astype(str), sort=False)[0]
    pesos = {
        c["output_column_name"]: [
            {
                "nivel": n.get("label_for_charts"),
                "m": n.get("m_probability"),
                "u": n.get("u_probability"),
            }
            for n in c["comparison_levels"]
            if not n.get("is_null_level")
        ]
        for c in modelo["comparisons"]
    }
    info = {
        "umbral": umbral,
        "pares_sobre_umbral": int(n_pares),
        "probabilidad_previa": modelo["probability_two_random_records_match"],
        "parametros": pesos,
    }
    return ids, info


def comparar_pares(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Acuerdo por pares entre dos agrupaciones de las mismas transacciones.

    ``precision`` = de los pares que ``a`` une, qué fracción une ``b``; ``recall`` = de los
    pares que ``b`` une, qué fracción une ``a``.
    """
    tabla = pd.DataFrame({"a": a, "b": b})

    def pares(n: pd.Series) -> int:
        return int((n * (n - 1) // 2).sum())

    ambos = pares(tabla.groupby(["a", "b"]).size())
    en_a = pares(tabla.groupby("a").size())
    en_b = pares(tabla.groupby("b").size())
    return {
        "pares_a": en_a,
        "pares_b": en_b,
        "pares_ambos": ambos,
        "precision": ambos / en_a if en_a else float("nan"),
        "recall": ambos / en_b if en_b else float("nan"),
    }
