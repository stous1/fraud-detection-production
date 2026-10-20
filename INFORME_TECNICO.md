# Informe técnico

Detalle de cada hito, con todas las cifras. Cada número sale de un JSON en
`docs/resultados/`; los gráficos están en la misma carpeta, en PNG y en HTML (con tooltip).

| Hito | Qué hay | Resultado principal |
|---|---|---|
| [1. Datos y cliente](#hito-1-ingesta-exploración-sospechas-y-reconstrucción-del-cliente) | Ingesta, partición, exploración, sospechas y cliente | El cliente es `card1 + addr1 + D1n` exacto. **La etiqueta se propaga por cliente:** con un fraude conocido 30 días antes, el 58 % de las transacciones siguientes del mismo cliente son fraude (16 veces la prevalencia) |

---

## Hito 1: ingesta, exploración, sospechas y reconstrucción del cliente

### Datos y partición

- `train_transaction.csv` y `train_identity.csv` de IEEE-CIS: **590.540 transacciones**,
  144.233 con identidad (24,4 %). Huellas SHA-256 en
  [carga.json](docs/resultados/hito1/carga.json). Los archivos de prueba de la competencia no
  traen etiqueta y no se leen: la ingesta los rechaza y una prueba verifica que el código no
  los nombre.
- Esquema explícito (`esquema.py`): texto, enteros (identificadores, tiempo, códigos de tarjeta
  y dirección) y el resto `DOUBLE`. Sale de revisar cada columna completa.
- `dia = TransactionDT // 86400` va de 1 a 182; `TransactionID` crece con el tiempo (ningún
  par fuera de orden). La fecha de calendario es desconocida.
- **Partición fijada antes de mirar etiquetas** (tag `particion-fijada`):

| Periodo | Días | Transacciones |
|---|---|---|
| Entrenamiento | 1 a 90 | 315.927 |
| Validación | 91 a 120 | 98.615 |
| Prueba | 121 a 182 (8 semanas y una de 6 días) | 175.998 |

- **Simplificación declarada:** el modelo se entrenará con todas las etiquetas de desarrollo,
  como si al congelarlo (día 120) ya se conocieran. Con un retraso de contracargo de 30 días,
  en producción solo se conocerían las de los días 1 a 90. El hito 3 medirá cuánto cambia si
  se entrena solo con los días 1 a 90.

### Exploración (desarrollo, días 1 a 120)

[exploracion.json](docs/resultados/hito1/exploracion.json)

- **414.542 transacciones y 14.600 fraudes: prevalencia 3,52 %.** El primer bloque de 30
  días tiene 2,53 % y los tres siguientes, entre 3,95 % y 4,04 %. Las semanas 3 y 4 tienen el
  mayor volumen (unas 35 mil transacciones) y la menor prevalencia (2,1-2,4 %); es
  compatible con una temporada de compras, pero la fecha real no se conoce.

![Volumen semanal](docs/resultados/hito1/volumen_semanal.png)
![Prevalencia semanal](docs/resultados/hito1/prevalencia_semanal.png)

- **Producto:** C tiene 11,2 % de fraude, S 6,0 %, H 4,5 %, R 3,5 % y W 2,1 %; W es el 72 %
  del volumen.
- **Decimales del monto:** casi todo C tiene tres o más decimales (45.328 de 50.136), lo que
  sugiere conversión de moneda. En W, los montos con centavos tienen 0,78 % de fraude y los
  enteros, 3,34 %. Es información disponible al pagar: no se trata como sospechosa, pero sí
  como posible artefacto de cómo Vesta registró los montos.

![Prevalencia por decimales](docs/resultados/hito1/prevalencia_decimales.png)

- **Identidad:** con identidad hay 7,34 % de fraude; sin ella, 2,14 %. En móvil, 9,87 %; en
  escritorio, 5,96 %. Crédito tiene 6,61 % y débito, 2,41 %.
- **Monto:** la mediana es 68,95 en legítimas y 76,02 en fraudes; el percentil 90, 270 y 335.
- **Faltantes:** las V faltan en 15 bloques que faltan juntos. Entre desarrollo y prueba (sin
  etiquetas), las columnas **M pasan de 55,7 % a 36,3 % de faltantes** y la identidad, de
  83,1 % a 87,8 %: deriva que el monitoreo del hito 4 tiene que ver.

![Faltantes](docs/resultados/hito1/faltantes.png)

### Reconstrucción del cliente

[clientes.json](docs/resultados/hito1/clientes.json) · el paso no lee etiquetas.

`D1` son los días desde el alta de la tarjeta, así que `D1n = dia - D1` es el día de alta y
debería ser constante para una tarjeta. Sin verdad conocida, la calidad se mide con atributos
que **no** se usan para unir: correo, `D10n`, `D15n` y dispositivo deberían coincidir entre
transacciones consecutivas del mismo cliente. `card2` a `card6` y `addr2` no sirven para esto:
coinciden en más del 99 % aun agrupando solo por `card1` (dependen del código de tarjeta).

| Nivel | Clientes | Tx en clientes de una sola tx | Correo | D10n | D15n | Dispositivo |
|---|---|---|---|---|---|---|
| `card1` | 13.553 | 0,6 % | 0,429 | 0,289 | 0,249 | 0,465 |
| `card1 + addr1` | 39.974 | 2,7 % | 0,478 | 0,312 | 0,285 | 0,496 |
| **`uid` = `card1 + addr1 + D1n`** | **217.850** | 21,2 % | **0,781** | **0,666** | **0,695** | **0,640** |
| `uid` con ±1 día (descartado) | 187.997 | 16,8 % | 0,727 | 0,609 | 0,632 | 0,597 |
| `uid` con veto de correo (descartado) | 254.704 | 26,8 % | 0,999* | 0,678 | 0,714 | 0,715 |

\* El veto usa el correo, así que su concordancia de correo no vale como validación.

![Concordancia por nivel](docs/resultados/hito1/clientes_concordancia.png)

**Dos variantes medidas y descartadas.** Para leerlas, sirven dos referencias: los pares
consecutivos del mismo `uid` coinciden en `D10n` el 66,6 % de las veces, y los pares de `uid`
distintos dentro de la misma `card1 + addr1`, el 7,3 %.

1. **Tolerancia de un día en `D1n`.** La propuse en el plan pensando en la hora y la zona
   horaria. Los datos dicen otra cosa: los 50.715 pares que agrega coinciden en `D10n` el
   4,4 % de las veces y en `D15n` el 4,5 %, el nivel de clientes distintos. Tampoco cerca de
   la medianoche: entre las 0 y las 3 h, coinciden en `D10n` el 5,1 % (11.788 pares). Une
   tarjetas distintas dadas de alta en días consecutivos. `D1n` es exacto.
2. **Veto por correo distinto.** Separa 70.634 pares que coinciden en `D10n` el 58,0 % y en
   `D15n` el 55,5 %: por esas dos, entre el 79 % y el 85 % serían del mismo cliente. Es la
   lección de los vetos del proyecto 1. Pero en dispositivo coinciden solo el 26,6 %, como
   clientes distintos: la misma tarjeta usada desde otro dispositivo y otro correo. Por eso el
   `uid` es la **cuenta de tarjeta**, y un cambio de correo o de dispositivo es información
   para el modelo, no un corte. Mirando etiquetas de desarrollo, el cambio no eleva el fraude
   por sí solo: con correo distinto al de la transacción anterior del cliente, 4,23 %; con el
   mismo, 4,35 %. Con el mismo dispositivo el fraude es mayor (10,1 % frente a 6,8 %), algo
   compatible con la propagación (el defraudador repite dispositivo). Ver
   `cambio_de_atributo_desarrollo` en sospechas.json.

**`D1 = 0` es frecuente** (47 % de las transacciones; 89 % en R y 95 % en H). Solo el 34 % de
esas transacciones tiene después otra del mismo `uid` con `D1 > 0`, como cabría esperar del
alta de una tarjeta que sigue activa (37 % en W, 18 % en H, 13 % en R). En H y R parece que
`D1` no se actualiza, y el `uid` de esos productos se parte por día. Queda como
limitación: el grafo del hito 2 conecta esos fragmentos por correo, dispositivo y dirección.

**Splink como referencia fuera de línea.** Fellegi-Sunter entrenado con EM y sin etiquetas,
sobre transacciones, con bloqueo dentro de `card1 + addr1` y comparaciones de `D1n`, `D10n`,
`D15n`, correo y producto. Ve toda la ventana (también el futuro), así que no entra al modelo.
`DeviceInfo` queda fuera de Splink para validar las dos agrupaciones con algo que ninguna usa.

| Umbral de Splink | Grupos | Pares de `uid` que Splink une | Pares de Splink dentro de un `uid` | Dispositivo |
|---|---|---|---|---|
| 0,5 | 281.441 | 81,5 % | 70,7 % | 0,644 |
| 0,9 | 314.696 | 44,6 % | 89,4 % | 0,624 |
| 0,99 | 342.388 | 36,4 % | 100,0 % | 0,614 |
| `uid` (en línea) | 217.850 | | | 0,640 |

Splink parte sobre todo los `uid` grandes y une poco entre ellos. En el atributo que ninguno
usa, no mejora al `uid` en ningún umbral (0,614 a 0,644 contra 0,640). Una resolución
probabilística con acceso al futuro no justifica algo más complejo que la clave en línea.

**Clientes de prueba ya vistos.** El 40,3 % de las transacciones de prueba son de un `uid` que
apareció en desarrollo: 51,7 % en la primera semana de prueba y 31,7 % en la novena. Es la
base del reporte separado de clientes nuevos y ya vistos del hito 3.

![Clientes vistos en prueba](docs/resultados/hito1/clientes_vistos_prueba.png)

### Sospechas de filtración

[sospechas.json](docs/resultados/hito1/sospechas.json)

**1. Propagación de la etiqueta por cliente (confirmada).** Los organizadores contaron que,
tras un contracargo, marcaron como fraude transacciones posteriores ligadas a la misma cuenta,
correo o dirección. En desarrollo, con el `uid`:

| Situación de la transacción | Fraude |
|---|---|
| Todas | 3,52 % |
| Primera del cliente | 2,64 % |
| El cliente tiene historia, pero sin fraude previo | 0,65 % |
| El cliente ya tuvo un fraude antes | 44,1 % |

| Días desde el primer fraude del cliente | Transacciones | Fraude |
|---|---|---|
| 0 a 1 | 12.366 | 32,3 % |
| 1 a 7 | 2.459 | 73,1 % |
| 7 a 30 | 2.620 | 60,3 % |
| 30 a 60 | 1.350 | 57,7 % |
| 60 o más | 1.032 | 57,8 % |

![Propagación](docs/resultados/hito1/propagacion_tramos.png)

**Con el retraso del contracargo.** En producción, el fraude de una transacción se sabe días
después. La regla "el cliente ya tuvo un fraude conocido" con cada retraso:

| Retraso | Transacciones marcadas | Precisión | Fraudes cubiertos |
|---|---|---|---|
| 0 días (no realista) | 19.827 | 44,1 % | 59,9 % |
| 15 días | 3.713 | 57,9 % | 14,7 % |
| **30 días (base)** | 2.382 | **57,7 %** | **9,4 %** |
| 60 días | 1.032 | 57,8 % | 4,1 % |
| 90 días | 322 | 56,8 % | 1,3 % |

![Retraso](docs/resultados/hito1/propagacion_retraso.png)

Consecuencias:

- Una variable de "fraudes previos del cliente" **sin retraso** cubriría el 60 % de los
  fraudes: es filtración. Con el retraso base de 30 días, la señal es legítima en producción
  pero pequeña (9,4 % de los fraudes) e inflada por la regla de etiquetado de la competencia
  (en la realidad, la tarjeta se bloquea).
- El modelo puede **memorizar clientes**: el 40 % de las transacciones de prueba son de
  clientes ya vistos. El hito 3 reporta clientes nuevos y ya vistos por separado y hace
  ablación de las variables de identidad e historia del cliente.

> **Corrección.** Durante el trabajo comuniqué en el chat una precisión del 90 % para esta
> regla. Venía de un cálculo exploratorio con un error: `cummin` de pandas deja NaN en las
> filas sin fraude, así que solo miraba si la transacción inmediatamente anterior era fraude.
> La cifra correcta es la de la tabla, calculada por `sospechas.propagacion` con una prueba
> que verifica que el primer fraude previo no incluye la transacción actual.

**2. ¿Las C, D o V agregan el futuro? Sin evidencia.** En la primera transacción de 93.713
tarjetas nuevas (`D1 = 0`, producto W, con 30 días de datos por delante), se correlacionó cada
columna con el número de transacciones del cliente en los 30 días siguientes. Una variable en
línea no tiene historia que mirar en ese momento; una que agrega la ventana completa sí
correlaciona (en el mini-IEEE, un total plantado da ρ > 0,5). Ninguna de las 159 columnas
evaluadas llega a |ρ| = 0,3: la mayor es D3 (0,127; probablemente los días desde la
transacción anterior), después D15 (0,078), y entre las C, C1 (0,065). El monto da −0,031.

![Prueba de futuro](docs/resultados/hito1/prueba_futuro.png)

Dentro del cliente, las C casi siempre se mantienen (C1 igual en el 77 % de los pares
consecutivos, C3 en el 99,8 %) pero a veces bajan (C1 en el 10 %): no son contadores
acumulados puros, quizá conteos por ventana. Esto **no prueba** que estén limpias: son
opacas y siguen declaradas sospechosas, con ablación en el hito 3.

**3. Registro de grupos** (`sospechas.GRUPOS`, verificado por una prueba que cubre todas las
columnas del esquema): historia de etiquetas del cliente, C, D, V e identidad, sospechosos;
datos de la transacción (monto, producto, tarjeta, dirección, distancia, correo, M), no
sospechosos.

### Lo que deja el hito 1 para los siguientes

- Tabla `clientes` (`TransactionID`, `uid`) en `data/fraude.duckdb`.
- Hito 2: los fragmentos de `uid` (sobre todo H y R) se conectan en el grafo por correo,
  dispositivo y dirección. Las variables de vecindario y de historia de etiquetas usan solo
  el pasado, y las etiquetas con el retraso de contracargo.
- Hito 3: reporte con y sin sospechosas, y para clientes nuevos y ya vistos.
- Hito 4: deriva en las M y en la identidad entre desarrollo y prueba.
