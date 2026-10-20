"""Mini-IEEE sintético con el formato de los CSV reales, para pruebas y CI.

Planta lo que el hito 1 quiere detectar:

- clientes con tarjeta (``card1`` compartido entre varios clientes), dirección, correo y día
  de alta ``D1n``; ``D1 = dia - D1n`` con ruido de un día en el 10 % de las transacciones;
- ``D10`` y ``card2``-``card6`` estables por cliente (sirven para validar sin usarlos);
- correo faltante en algunas transacciones (no debe partir al cliente);
- **propagación de etiquetas**: en un cliente fraudulento, todo lo posterior al primer fraude
  es fraude;
- ``C1`` como contador en línea (cuántas transacciones lleva el cliente) y ``C13`` como
  agregado global (total del cliente en toda la ventana: usa el futuro).

``generar`` devuelve también la verdad del cliente, que no va en los CSV.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fraude import esquema

PRIMER_ID = 2_987_000
DOMINIOS = ("gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "aol.com")
MARCAS = ("visa", "mastercard", "discover", "american express")


@dataclass(frozen=True)
class Mini:
    transacciones: pd.DataFrame
    identidad: pd.DataFrame
    cliente: pd.Series  # TransactionID -> cliente verdadero


def generar(
    n_clientes: int = 300,
    semilla: int = 0,
    dias: int = 182,
    fraccion_fraude: float = 0.08,
) -> Mini:
    """Genera transacciones e identidades con clientes plantados."""
    rng = np.random.default_rng(semilla)
    n_tarjetas = max(2, n_clientes // 3)  # ~3 clientes por card1
    filas = []
    for c in range(n_clientes):
        card1 = int(1000 + rng.integers(n_tarjetas))
        perfil = {
            "card1": card1,
            "card2": int(100 + rng.integers(500)),
            "card3": 150,
            "card4": MARCAS[rng.integers(len(MARCAS))],
            "card5": int(100 + rng.integers(126)),
            "card6": "credit" if rng.random() < 0.3 else "debit",
            "addr1": int(100 + rng.integers(400)) if rng.random() < 0.9 else None,
            "addr2": 87,
            "P_emaildomain": DOMINIOS[rng.integers(len(DOMINIOS))],
            "D1n": int(rng.integers(-400, dias - 5)),
            "D10n": int(rng.integers(-200, dias - 5)),
            "dispositivo": f"Disp{int(rng.integers(40))}",
        }
        inicio = max(1, perfil["D1n"])
        n_tx = int(rng.integers(1, 25))
        dias_tx = np.sort(rng.integers(inicio, dias + 1, size=n_tx))
        fraudulento = rng.random() < fraccion_fraude
        dia_fraude = int(dias_tx[rng.integers(n_tx)]) if fraudulento else None
        for k, d in enumerate(dias_tx):
            d = int(d)
            correo = perfil["P_emaildomain"] if rng.random() > 0.15 else None
            fraude = dia_fraude is not None and d >= dia_fraude
            if not fraude and rng.random() < 0.005:
                fraude = True  # fraude aislado, sin cliente comprometido
            filas.append(
                {
                    "cliente": c,
                    "dia": d,
                    "segundo": int(rng.integers(86_400)),
                    "isFraud": int(fraude),
                    "TransactionAmt": round(float(rng.lognormal(4, 1)), 3),
                    "ProductCD": "W" if rng.random() < 0.7 else "C",
                    **{k2: v for k2, v in perfil.items() if k2 not in ("D1n", "D10n")},
                    "P_emaildomain": correo,
                    "D1": d - perfil["D1n"] - (1 if rng.random() < 0.1 else 0),
                    "D10": d - perfil["D10n"],
                    "C1": k + 1,
                    "C13": n_tx,
                    "tiene_identidad": rng.random() < 0.25,
                }
            )
    df = pd.DataFrame(filas)
    df["D1"] = df["D1"].clip(lower=0)
    df["D10"] = df["D10"].clip(lower=0)
    df["TransactionDT"] = df["dia"] * 86_400 + df["segundo"]
    df = df.sort_values(["TransactionDT", "cliente"], kind="stable").reset_index(drop=True)
    df["TransactionID"] = PRIMER_ID + np.arange(len(df))

    tx = pd.DataFrame({c: df[c] if c in df else np.nan for c in esquema.COLUMNAS_TX})
    for c in ("card2", "card3", "card5", "addr1", "addr2", "C1", "C13", "D1", "D10"):
        tx[c] = tx[c].astype("float64")  # como en el CSV real: "404.0"
    for c in esquema.TEXTO_TX:
        tx[c] = tx[c].astype("object")

    con_id = df[df["tiene_identidad"]]
    ident = pd.DataFrame(
        {c: np.nan for c in esquema.COLUMNAS_ID}, index=con_id.index, dtype="object"
    )
    ident["TransactionID"] = con_id["TransactionID"]
    ident["id_01"] = -5.0
    ident["DeviceType"] = "desktop"
    ident["DeviceInfo"] = con_id["dispositivo"]
    ident = ident.reset_index(drop=True)
    cliente = pd.Series(df["cliente"].to_numpy(), index=df["TransactionID"], name="cliente")
    return Mini(tx, ident, cliente)


def escribir_csv(mini: Mini, directorio: Path) -> tuple[Path, Path]:
    """Escribe ``train_transaction.csv`` y ``train_identity.csv`` en ``directorio``."""
    directorio.mkdir(parents=True, exist_ok=True)
    ruta_tx = directorio / "train_transaction.csv"
    ruta_id = directorio / "train_identity.csv"
    mini.transacciones.to_csv(ruta_tx, index=False, lineterminator="\n")
    mini.identidad.to_csv(ruta_id, index=False, lineterminator="\n")
    return ruta_tx, ruta_id
