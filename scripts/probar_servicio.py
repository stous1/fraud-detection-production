"""Prueba de humo del servicio en Docker: una semana de prueba por HTTP.

Con el contenedor arriba (``docker compose up -d --build``), envía en orden las transacciones
de la primera semana de prueba (o las primeras ``--n``) a ``/puntuar`` y reporta cada fraude a
``/etiqueta``; compara cada probabilidad con la puntuación por lotes del modelo en producción
y mide la latencia de punta a punta. Escribe ``reports/hito4/servicio.json``.

Uso:
    uv run python scripts/probar_servicio.py --n 3000
"""

import argparse
import math
import time

import httpx
import numpy as np

from fraude import base, flujo_hito4, modelo_xgb, registro, rutas, util, variables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=3000)
    args = parser.parse_args()

    con = base.conectar(solo_lectura=True)
    crudas = flujo_hito4.transacciones_crudas(con)
    lotes = variables.cargar(con, 127)
    con.close()
    semana1 = crudas[(crudas["TransactionDT"] >= 121 * 86_400)].head(args.n)

    cliente = httpx.Client(base_url=args.url, timeout=30)
    info = cliente.get("/modelo").json()
    probs, latencias = [], []
    for tx in semana1.to_dict(orient="records"):
        y = tx.pop("isFraud")
        cuerpo = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in tx.items()}
        inicio = time.perf_counter()
        r = cliente.post("/puntuar", json=cuerpo)
        latencias.append(time.perf_counter() - inicio)
        r.raise_for_status()
        probs.append(r.json()["probabilidad"])
        if y == 1:
            cliente.post("/etiqueta", json={"TransactionID": tx["TransactionID"], "fraude": True})
    metricas = cliente.get("/metricas").json()

    version = next(v for v in registro.leer() if v.version == info["version"])
    import json

    p = json.loads(version.ruta_parametros().read_text(encoding="utf-8"))
    v = p["variantes"][version.variante]
    filas = lotes.set_index("TransactionID").loc[semana1["TransactionID"]].reset_index()
    esperado = modelo_xgb.puntuar(
        modelo_xgb.cargar(version.ruta(), version.sha256),
        variables.matriz(
            filas, v["columnas"], variables.Codificador(p["codificadores"]["realista"])
        ),
    )
    dif = np.abs(np.array(probs) - esperado)
    lat = np.array(latencias) * 1000
    salida = {
        "url": args.url,
        "version": info["version"],
        "variante": info["variante"],
        "transacciones": len(probs),
        "diferencia_max_con_lotes": float(dif.max()),
        "iguales_a_1e-6": bool((dif <= 1e-6).all()),
        "latencia_ms_cliente": {
            "p50": float(np.percentile(lat, 50)),
            "p95": float(np.percentile(lat, 95)),
            "p99": float(np.percentile(lat, 99)),
        },
        "metricas_servicio": metricas,
    }
    util.escribir_json(salida, rutas.reports_hito(4) / "servicio.json")
    print(salida)


if __name__ == "__main__":
    main()
