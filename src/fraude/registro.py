"""Registro de versiones del modelo (``modelos/registro.json``).

Cada versión apunta a un modelo congelado del hito 3 (archivo UBJ con su SHA-256), a los
parámetros publicados (columnas, codificador y umbrales, también con su SHA-256) y guarda sus
métricas de prueba y su estado: ``produccion`` (el que sirve el servicio), ``respaldo`` (listo
para el plan de contingencia) o ``retirado``. Solo puede haber una versión en producción.

Es un registro propio en un JSON versionado, sin servidor: alcanza para un servicio local y
deja la trazabilidad en git. MLflow cumpliría el mismo papel con más infraestructura.
"""

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from fraude import rutas

DIR_MODELOS = rutas.RAIZ / "modelos"
REGISTRO = DIR_MODELOS / "registro.json"
ESTADOS = ("produccion", "respaldo", "retirado")


def sha256(ruta: Path) -> str:
    datos = (
        Path(ruta).read_bytes().replace(b"\r\n", b"\n")
        if Path(ruta).suffix == ".json"
        else Path(ruta).read_bytes()
    )
    return hashlib.sha256(datos).hexdigest()


@dataclass(frozen=True)
class Version:
    version: str
    variante: str
    estado: str
    archivo: str  # relativo a la raíz del repositorio
    sha256: str
    parametros: str  # relativo a la raíz
    sha256_parametros: str
    descripcion: str
    metricas_prueba: dict
    origen: str
    umbrales: dict  # costo de una falsa alarma (US$, texto) -> {"tasa_alertas", "umbral"}

    def ruta(self, raiz: Path = rutas.RAIZ) -> Path:
        return raiz / self.archivo

    def ruta_parametros(self, raiz: Path = rutas.RAIZ) -> Path:
        return raiz / self.parametros


def leer(ruta: Path = REGISTRO) -> list[Version]:
    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    versiones = [Version(**v) for v in datos["versiones"]]
    validar(versiones)
    return versiones


def validar(versiones: list[Version]) -> None:
    nombres = [v.version for v in versiones]
    assert len(set(nombres)) == len(nombres), "versiones repetidas"
    assert all(v.estado in ESTADOS for v in versiones), "estado desconocido"
    assert sum(v.estado == "produccion" for v in versiones) == 1, "debe haber una en producción"


def en_produccion(versiones: list[Version]) -> Version:
    return next(v for v in versiones if v.estado == "produccion")


def respaldo(versiones: list[Version]) -> Version | None:
    return next((v for v in versiones if v.estado == "respaldo"), None)


def verificar(version: Version, raiz: Path = rutas.RAIZ) -> None:
    """Falla si el modelo o los parámetros no coinciden con sus huellas."""
    if sha256(version.ruta(raiz)) != version.sha256:
        raise RuntimeError(f"{version.archivo}: el modelo no coincide con el registro")
    if sha256(version.ruta_parametros(raiz)) != version.sha256_parametros:
        raise RuntimeError(f"{version.parametros}: los parámetros no coinciden con el registro")


def escribir(versiones: list[Version], ruta: Path = REGISTRO) -> None:
    validar(versiones)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    datos = {"versiones": [v.__dict__ for v in versiones]}
    with open(ruta, "w", encoding="utf-8", newline="\n") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)
        f.write("\n")


def registrar_desde_hito3(
    parametros: Path,
    modelos_hito3: Path,
    evaluacion: Path,
    variantes: dict[str, tuple[str, str, str]],
    umbrales: dict[str, dict],
    destino: Path = DIR_MODELOS,
    raiz: Path = rutas.RAIZ,
) -> list[Version]:
    """Copia los modelos congelados del hito 3 a ``destino`` y arma el registro.

    ``variantes``: nombre de la variante -> (versión, estado, descripción); ``umbrales``:
    nombre de la variante -> umbrales por costo de falsa alarma. Verifica cada modelo contra la
    huella de ``parametros.json`` antes de copiarlo.
    """
    p = json.loads(Path(parametros).read_text(encoding="utf-8"))
    ev = json.loads(Path(evaluacion).read_text(encoding="utf-8"))
    sha_param = sha256(parametros)
    versiones = []
    destino.mkdir(parents=True, exist_ok=True)
    for variante, (version, estado, descripcion) in variantes.items():
        entrada = p["variantes"][variante]
        origen = Path(modelos_hito3) / entrada["modelo"]
        if sha256(origen) != entrada["sha256"]:
            raise RuntimeError(f"{origen} no coincide con la huella congelada")
        archivo = destino / f"v{version}_{variante}.ubj"
        shutil.copy2(origen, archivo)
        r = ev["variantes"][variante]
        metricas = {
            "ap_nuevos": r["prueba"]["nuevos"]["ap"],
            "ap_nuevos_ic95": r["prueba"]["nuevos"].get("ap_ic95"),
            "ap_nuevos_W": r["prueba_cortes"]["nuevos_W"]["ap"],
            "ap_vistos_inflada": r["prueba"]["vistos"]["ap"],
            "ap_todos": r["prueba"]["todos"]["ap"],
        }
        versiones.append(
            Version(
                version=version,
                variante=variante,
                estado=estado,
                archivo=archivo.relative_to(raiz).as_posix(),
                sha256=entrada["sha256"],
                parametros=Path(parametros).resolve().relative_to(raiz).as_posix(),
                sha256_parametros=sha_param,
                descripcion=descripcion,
                metricas_prueba=metricas,
                origen="tag hito3-congelado",
                umbrales=umbrales[variante],
            )
        )
    return versiones
