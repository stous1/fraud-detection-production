"""Flujo del hito 1."""

from collections.abc import Callable

from fraude import base, ingesta, particion, rutas, util


def preparar() -> None:
    """CSV de train -> ``data/fraude.duckdb``, con verificaciones y huellas."""
    con = base.conectar()
    ingesta.cargar(con, rutas.CSV_TRANSACCIONES, rutas.CSV_IDENTIDAD)
    particion.crear_vistas(con)
    info = ingesta.verificar(con)
    info["huellas_sha256"] = {
        r.name: ingesta.huella(r) for r in (rutas.CSV_TRANSACCIONES, rutas.CSV_IDENTIDAD)
    }
    info["particion"] = particion.resumen(con)
    util.escribir_json(info, rutas.reports_hito(1) / "carga.json")
    con.close()
    print(info)


PASOS: dict[str, Callable[[], None]] = {"preparar": preparar}
