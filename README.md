# Detección de fraude con desbalance extremo, en producción

Caso 3 del portafolio de ciencia de datos aplicado a banca, sobre **IEEE-CIS Fraud Detection**
(Vesta, 590.540 transacciones con 3,5 % de fraude): reconstrucción del cliente detrás de las
transacciones, grafo de identificadores compartidos, modelo con umbral elegido por costo y un
servicio local con monitoreo de deriva y ficha del modelo.

**Estado:** hitos 1 a 3 de 5 terminados (datos, cliente, grafo y modelo). El detalle con
todas las cifras está en [INFORME_TECNICO.md](INFORME_TECNICO.md).

## La cifra principal

**AP 0,551 [0,526–0,574] en clientes nuevos**, en prueba (días 121 a 182, medida una vez con
XGBoost congelado; el azar da 0,044). Con 1 % de falsas alarmas, detecta el 44,6 % de los
fraudes de clientes nuevos. Con un costo supuesto de 10 US$ por falsa alarma, el umbral
elegido en validación reduce el costo total en 62 % frente a no tener modelo.

**Por qué clientes nuevos y no todos.** IEEE-CIS etiquetó por propagación: tras un
contracargo, marcó como fraude lo que siguió en la misma cuenta, correo o dirección. Un
modelo que reconoce a un cliente ya marcado se ve muy bien en los datos (AP 0,831 en clientes
ya vistos) y no serviría en un banco, donde la tarjeta se bloquea tras el contracargo. Esa
cifra se reporta aparte y no se presenta como rendimiento.

**Y un matiz más.** El 71 % de las transacciones de clientes nuevos son del producto W, y ahí
la AP es 0,161. En los demás productos (C, R, H, S) llega a 0,58-0,80, probablemente porque
las variables opacas de Vesta (C, V) agregan por identificadores de cuenta que IEEE-CIS no
publica, por los que también se propagó la etiqueta. La cifra de W es la cota conservadora.

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

## Lo que dejó el hito 3

- **XGBoost en GPU** (RTX 4080 SUPER), búsqueda de 40 configuraciones elegida por AP en
  clientes nuevos de la validación, y **evaluación en CPU con los pesos congelados** (tag
  `hito3-congelado`, huellas SHA-256).
- **Cuánto aporta cada grupo en clientes nuevos** (AP del completo menos AP sin el grupo):
  variables sospechosas C/D/V/identidad +0,250, transacción +0,021, historia de fraude del
  cliente +0,002 y vecindario del grafo −0,008. Sin las sospechosas, el mejor modelo llega a
  0,300. La red sobre el grafo no se entrenó: la regla declarada exigía que el vecindario
  aportara al menos 0,01 en validación, y aportó 0,002.
- **SHAP** (TreeSHAP de XGBoost): en clientes nuevos, las sospechosas son el 57 % del peso;
  las variables principales son el tipo de tarjeta, el monto y las C1, C13 y C14.

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
uv run python scripts/hito3.py buscar      # hiperparámetros (GPU si hay)
uv run python scripts/hito3.py desarrollar # variantes, umbral por costo, SHAP
uv run python scripts/hito3.py evaluar     # prueba, en CPU, con los modelos congelados
```

Guía de trabajo y reglas de evaluación: [CLAUDE.md](CLAUDE.md). Licencia MIT.
