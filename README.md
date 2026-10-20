# Detección de fraude con desbalance extremo, en producción

Caso 3 del portafolio de ciencia de datos aplicado a banca, sobre **IEEE-CIS Fraud Detection**
(Vesta, 590.540 transacciones con 3,5 % de fraude): reconstrucción del cliente detrás de las
transacciones, grafo de identificadores compartidos, modelo con umbral elegido por costo y un
servicio local con monitoreo de deriva y ficha del modelo.

**Estado:** hitos 1 y 2 de 5 terminados (datos, cliente, sospechas, grafo y vecindario).
El detalle con todas las cifras está en [INFORME_TECNICO.md](INFORME_TECNICO.md).

## Lo que dejó el hito 1

- **Partición temporal fijada antes de mirar etiquetas** (tag `particion-fijada`): desarrollo
  = días 1 a 120, prueba = días 121 a 182, medida una sola vez al final.
- **El cliente** es `card1 + addr1 + D1n` (día de alta de la tarjeta), exacto y calculable en
  el momento de pagar. Dos variantes que parecían razonables se midieron con atributos que no
  se usan para unir y se descartaron: un día de tolerancia une tarjetas distintas y un veto
  por correo parte clientes. Splink (EM sin etiquetas, con acceso al futuro) no lo mejora en
  el atributo que ninguno de los dos usa.
- **La etiqueta se propaga por cliente**, como en la regla de etiquetado de la competencia:
  con un fraude conocido 30 días antes (el retraso de contracargo supuesto), el 58 % de las
  transacciones siguientes del mismo cliente son fraude, 16 veces la prevalencia. Sin
  retraso, esa regla cubriría el 60 % de los fraudes: sería filtración. Y el 40 % de las
  transacciones de prueba son de clientes ya vistos, así que los resultados del modelo se
  reportarán aparte para clientes nuevos y ya vistos.
- Las variables opacas C, D y V no muestran señales de usar el futuro en una prueba directa
  (|ρ| ≤ 0,13), pero siguen declaradas sospechosas y tendrán ablación.

## Lo que dejó el hito 2

- **Esquema realista:** al desplegar el día 121, con 30 días de retraso de contracargo, solo
  se conocen etiquetas hasta el día 90; con ellas se entrena y se valida. Usar todas las
  etiquetas de desarrollo queda solo como comparación.
- **Grafo de identidad** desde la tabla: cliente, tarjeta, huella de dispositivo, región y
  dominio de correo (IEEE-CIS no trae la dirección de correo). 139 variables de vecindario
  solo con el pasado, verificadas con pruebas de truncado y de etiquetas recientes.
- **En prueba, una sola vez, tras el tag `hito2-congelado`:** una logística de vecindario
  llega a AP 0,360 (azar: 0,034), pero **0,182 en clientes nuevos y 0,784 en ya vistos**.
  Sin retraso de etiquetas daría 0,681 (filtración); con todas las etiquetas de desarrollo,
  0,382.

## Cómo correrlo

Requiere [uv](https://docs.astral.sh/uv/) y los CSV de train de IEEE-CIS en `data/raw/`.

```bash
uv sync
uv run pytest                              # pruebas sobre un mini-IEEE sintético
uv run python scripts/hito1.py preparar    # CSV -> data/fraude.duckdb
uv run python scripts/hito1.py todo        # exploración, clientes y sospechas -> reports/hito1
uv run python scripts/hito2.py preparar    # grafo y vecindario
uv run python scripts/hito2.py desarrollar # variantes en desarrollo
uv run python scripts/hito2.py evaluar     # prueba, con los parámetros congelados
```

Guía de trabajo y reglas de evaluación: [CLAUDE.md](CLAUDE.md). Licencia MIT.
