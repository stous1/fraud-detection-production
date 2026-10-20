"""Estado en línea del vecindario: las variables del hito 2, transacción por transacción.

En producción no se puede recalcular ``vecindario`` con una consulta por lotes: cada
transacción llega sola y hay que puntuarla con lo que se sabe en ese instante. ``Estado``
guarda, por entidad, lo necesario para calcular exactamente las mismas variables que
``vecindario.py`` (con el retraso base) y se actualiza al procesar cada transacción y al
recibir cada contracargo. ``tests/test_estado_en_linea.py`` verifica la **paridad exacta** con
las variables por lotes con que se entrenó el modelo.

Uso:

1. ``avanzar(t)``: aplica a los componentes los fraudes cuyo contracargo ya "llegó"
   (``ts + R <= t``).
2. ``variables(tx)``: variables de la transacción, con el estado previo.
3. ``agregar(tx)``: la incorpora al estado.
4. ``reportar_fraude(TransactionID)``: registra un contracargo (en cualquier momento; solo
   cuenta para variables de transacciones con ``t >= ts + R``).

``procesar(tx)`` hace 1 a 3.
"""

import heapq
import math
from bisect import bisect_right, insort
from dataclasses import dataclass, field

from fraude.vecindario import ENTIDADES_COMPONENTE, LIMITE_CLIENTES, SEGUNDOS_DIA

TIPOS: tuple[str, ...] = (
    "cliente",
    "tarjeta_direccion",
    "tarjeta",
    "huella",
    "direccion",
    "correo",
)


def _valor(x: object) -> object:
    """Normaliza faltantes (``None`` o NaN) a ``None`` y enteros representados como float."""
    if x is None:
        return None
    if isinstance(x, float):
        if math.isnan(x):
            return None
        if x.is_integer():
            return int(x)
    return x


def claves(tx: dict[str, object]) -> dict[str, tuple | None]:
    """Clave de cada entidad de una transacción (``None`` si no la tiene). Mismas reglas que
    ``clientes.uid`` y ``grafo.ENTIDADES``."""
    v = {
        k: _valor(tx.get(k))
        for k in (
            "card1",
            "card2",
            "card3",
            "card4",
            "card5",
            "card6",
            "addr1",
            "addr2",
            "P_emaildomain",
            "D1",
            "DeviceInfo",
            "id_30",
            "id_31",
            "id_33",
            "id_19",
            "id_20",
        )
    }
    dia = int(tx["TransactionDT"]) // SEGUNDOS_DIA
    d1n = None if v["D1"] is None else dia - v["D1"]
    return {
        "cliente": (v["card1"], v["addr1"], d1n),
        "tarjeta_direccion": (v["card1"], v["addr1"]),
        "tarjeta": tuple(v[f"card{i}"] for i in range(1, 7)),
        "huella": None
        if v["DeviceInfo"] is None
        else tuple(v[k] for k in ("DeviceInfo", "id_30", "id_31", "id_33", "id_19", "id_20")),
        "direccion": None if v["addr1"] is None else (v["addr1"], v["addr2"]),
        "correo": v["P_emaildomain"],
    }


@dataclass
class _Entidad:
    ts: list[int] = field(default_factory=list)
    clientes: set = field(default_factory=set)
    fraudes_ts: list[int] = field(default_factory=list)
    primer_fraude: dict = field(default_factory=dict)  # cliente -> ts del primer fraude
    primer_fraude_ts: list[int] = field(default_factory=list)


class _Union:
    """Unión-búsqueda con diccionarios (crece con los datos)."""

    def __init__(self) -> None:
        self.padre: dict = {}
        self.clientes: dict = {}
        self.fraudes: dict = {}
        self.clientes_fraude: dict = {}

    def nodo(self, x: object, es_cliente: bool) -> None:
        if x not in self.padre:
            self.padre[x] = x
            self.clientes[x] = int(es_cliente)
            self.fraudes[x] = 0
            self.clientes_fraude[x] = 0

    def raiz(self, x: object) -> object:
        padre = self.padre
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    def unir(self, a: object, b: object) -> None:
        ra, rb = self.raiz(a), self.raiz(b)
        if ra == rb:
            return
        if self.clientes[ra] < self.clientes[rb]:
            ra, rb = rb, ra
        self.padre[rb] = ra
        self.clientes[ra] += self.clientes[rb]
        self.fraudes[ra] += self.fraudes[rb]
        self.clientes_fraude[ra] += self.clientes_fraude[rb]


class Estado:
    """Estado del vecindario con un retraso de etiquetas ``retraso`` (días)."""

    def __init__(self, retraso: int = 30, limite: int = LIMITE_CLIENTES) -> None:
        self.retraso = retraso
        self.limite = limite
        self.entidades: dict[str, dict[object, _Entidad]] = {t: {} for t in TIPOS}
        self.distintos: dict[object, dict[str, set]] = {}  # cliente -> valores vistos
        self.transacciones: dict[int, tuple[int, dict]] = {}  # TransactionID -> (ts, claves)
        self.fraudes: set[int] = set()
        self.union = _Union()
        self.miembros: dict[object, set] = {}
        self.pendientes: list[tuple[int, int]] = []  # (ts + R, TransactionID) para componentes
        self.conocidos: set = set()  # clientes con un fraude ya conocido en su componente
        self.ultimo_t: int | None = None

    # --- contracargos ---------------------------------------------------------------
    def reportar_fraude(self, tid: int) -> None:
        """Registra un contracargo de una transacción ya procesada."""
        if tid in self.fraudes or tid not in self.transacciones:
            return
        self.fraudes.add(tid)
        ts, k = self.transacciones[tid]
        for tipo in TIPOS:
            clave = k[tipo]
            if clave is None:
                continue
            e = self.entidades[tipo][clave]
            insort(e.fraudes_ts, ts)
            cliente = k["cliente"]
            previo = e.primer_fraude.get(cliente)
            if previo is None or ts < previo:
                if previo is not None:
                    e.primer_fraude_ts.remove(previo)
                e.primer_fraude[cliente] = ts
                insort(e.primer_fraude_ts, ts)
        heapq.heappush(self.pendientes, (ts + self.retraso * SEGUNDOS_DIA, tid))

    def avanzar(self, t: int) -> None:
        """Aplica a los componentes los contracargos que ya llegaron en ``t``."""
        while self.pendientes and self.pendientes[0][0] <= t:
            _, tid = heapq.heappop(self.pendientes)
            cliente = self.transacciones[tid][1]["cliente"]
            raiz = self.union.raiz(("cliente", cliente))
            self.union.fraudes[raiz] += 1
            if cliente not in self.conocidos:
                self.conocidos.add(cliente)
                self.union.clientes_fraude[raiz] += 1

    # --- variables ------------------------------------------------------------------
    def variables(self, tx: dict[str, object]) -> dict[str, float]:
        """Variables de vecindario de ``tx`` con el estado actual (sin incorporarla)."""
        t = int(tx["TransactionDT"])
        k = claves(tx)
        corte = t - self.retraso * SEGUNDOS_DIA
        r = self.retraso
        salida: dict[str, float] = {}
        for tipo in TIPOS:
            p = f"{tipo}_"
            clave = k[tipo]
            nombres = [
                "tx_previas",
                "tx_24h",
                "tx_7d",
                "dias_desde_primera",
                f"etiquetadas_r{r}",
                f"fraudes_r{r}",
                f"clientes_fraude_r{r}",
            ]
            if tipo != "cliente":
                nombres += ["clientes_previos", "cliente_nuevo"]
            if clave is None:
                salida.update({p + n: math.nan for n in nombres})
                continue
            e = self.entidades[tipo].get(clave)
            if e is None:
                e = _Entidad()
            n = len(e.ts)
            salida[p + "tx_previas"] = n
            salida[p + "tx_24h"] = n - bisect_right(e.ts, t - SEGUNDOS_DIA - 1)
            salida[p + "tx_7d"] = n - bisect_right(e.ts, t - 7 * SEGUNDOS_DIA - 1)
            salida[p + "dias_desde_primera"] = (t - e.ts[0]) / SEGUNDOS_DIA if n else 0.0
            if tipo != "cliente":
                salida[p + "clientes_previos"] = len(e.clientes)
                salida[p + "cliente_nuevo"] = int(k["cliente"] not in e.clientes)
            salida[p + f"etiquetadas_r{r}"] = bisect_right(e.ts, corte)
            salida[p + f"fraudes_r{r}"] = bisect_right(e.fraudes_ts, corte)
            salida[p + f"clientes_fraude_r{r}"] = bisect_right(e.primer_fraude_ts, corte)
        vistos = self.distintos.get(k["cliente"], {})
        salida["cliente_huellas_previas"] = len(vistos.get("huella", ()))
        salida["cliente_correos_previos"] = len(vistos.get("correo", ()))
        salida["cliente_tarjetas_previas"] = len(vistos.get("tarjeta", ()))
        # Componente: con las aristas de esta transacción (se conocen al pagar).
        raiz = self._unir_aristas(k)
        salida["componente_clientes"] = self.union.clientes[raiz]
        salida[f"componente_fraudes_r{r}"] = self.union.fraudes[raiz]
        salida[f"componente_clientes_fraude_r{r}"] = self.union.clientes_fraude[raiz]
        return salida

    def _unir_aristas(self, k: dict[str, tuple | None]) -> object:
        nodo_c = ("cliente", k["cliente"])
        self.union.nodo(nodo_c, True)
        for tipo in ENTIDADES_COMPONENTE:
            clave = k[tipo]
            if clave is None:
                continue
            nodo = (tipo, clave)
            self.union.nodo(nodo, False)
            socios = self.miembros.setdefault(nodo, set())
            if k["cliente"] in socios:
                self.union.unir(nodo_c, nodo)
            elif len(socios) < self.limite:
                socios.add(k["cliente"])
                self.union.unir(nodo_c, nodo)
        return self.union.raiz(nodo_c)

    def agregar(self, tx: dict[str, object]) -> None:
        """Incorpora ``tx`` al estado (después de calcular sus variables)."""
        t = int(tx["TransactionDT"])
        tid = int(tx["TransactionID"])
        k = claves(tx)
        for tipo in TIPOS:
            clave = k[tipo]
            if clave is None:
                continue
            e = self.entidades[tipo].setdefault(clave, _Entidad())
            e.ts.append(t)
            e.clientes.add(k["cliente"])
        vistos = self.distintos.setdefault(k["cliente"], {})
        for tipo in ("huella", "correo", "tarjeta"):
            if k[tipo] is not None:
                vistos.setdefault(tipo, set()).add(k[tipo])
        self.transacciones[tid] = (t, k)
        self.ultimo_t = t

    def procesar(self, tx: dict[str, object]) -> dict[str, float]:
        """Avanza el reloj, calcula las variables de ``tx`` y la incorpora."""
        t = int(tx["TransactionDT"])
        if self.ultimo_t is not None and t < self.ultimo_t:
            raise ValueError(f"transacción fuera de orden: {t} < {self.ultimo_t}")
        self.avanzar(t)
        v = self.variables(tx)
        self.agregar(tx)
        return v
