# Ficha del modelo: detección de fraude transaccional (v1.1.0)

Proyecto de portafolio sobre datos públicos (IEEE-CIS, Vesta). **No es un modelo en uso en
ningún banco** ni se entrenó con datos de clientes reales.

| | |
|---|---|
| Versión en producción | **1.1.0** (`sin_vecindario`), XGBoost, 2026-10-20 |
| Respaldo | **1.1.0-respaldo** (`solo_transaccion`) |
| Retirada | 1.0.0 (`completo`): relojes encubiertos, ver "Historia" |
| Artefactos | `modelos/registro.json` (SHA-256 de modelos y parámetros), tag `hito3-congelado` |
| Servicio | FastAPI en Docker (`docker compose up -d --build`), puntúa en CPU |

## Propósito

Puntuar cada transacción con tarjeta, en el momento del pago, con la probabilidad de que sea
fraude, y marcar para revisión las que superan un umbral elegido por costo. Pensado para
**priorizar revisiones**, no para rechazar pagos de forma automática.

**Usos fuera de alcance:** decisiones automáticas sobre clientes sin revisión humana;
productos, países o comercios distintos de los de IEEE-CIS sin reentrenar y revalidar;
cualquier uso como "puntaje del cliente".

## Datos

- IEEE-CIS Fraud Detection, solo `train` (590.540 transacciones, 182 días, 3,5 % de fraude).
  Los archivos de prueba de la competencia no traen etiqueta y no se usan.
- **Partición temporal**: desarrollo días 1 a 120 y prueba días 121 a 182, medida una sola
  vez tras congelar. **Esquema realista**: el modelo entra el día 121 y, con 30 días de
  retraso del contracargo, solo conoce etiquetas hasta el día 90 (entrena 1-60, valida
  61-90, reajusta 1-90).
- Cliente reconstruido: `uid = card1 + addr1 + D1n` (día de alta de la tarjeta).
- Variables (437): datos de la transacción (26), variables opacas de Vesta C, D, V e
  identidad (408) e historia de fraudes ya conocidos del cliente, con 30 días de retraso (3).
  La versión 1.1.0 no usa el vecindario del grafo (no aportaba y traía relojes encubiertos).

## Métricas (prueba, días 121 a 182)

**La cifra principal es la AP en clientes nuevos** (sin transacciones antes del periodo).
Se presenta como un rango:

| | AP en clientes nuevos |
|---|---|
| Solo datos de la transacción (respaldo, sin opacas ni relojes) | 0,235 [0,215–0,257] |
| Sin las variables opacas de Vesta (incluye relojes encubiertos) | 0,300 [0,278–0,326] |
| **v1.1.0 en producción** | **0,558 [0,532–0,584]** |
| Cota conservadora: nuevos del producto W (71 % de los nuevos), v1.1.0 | 0,177 |
| Azar | 0,044 |

- Con 1 % de falsas alarmas, la v1.1.0 detecta el 46,3 % de los fraudes de clientes nuevos.
- **Clientes ya vistos: AP 0,825, inflada.** La competencia marcó como fraude lo que siguió a
  un contracargo en la misma cuenta; en un banco la tarjeta se bloquea. No es rendimiento.
- Por qué un rango: casi toda la diferencia entre 0,235-0,300 y 0,558 son variables opacas
  de Vesta. Hipótesis: agregan por identificadores de cuenta no publicados, los mismos por
  los que se propagó la etiqueta, y podrían delatar cuentas ya marcadas. No se puede
  separar con estos datos.

## Umbral y costo

Supuesto: un fraude no detectado cuesta su monto; una falsa alarma, **10 US$** (revisión y
fricción). Umbral elegido en validación como tasa de alertas (14 %) y fijado con los
puntajes de los días 91 a 120. En la simulación semanal de prueba, la v1.1.0 marcó entre
12,6 % y 15,8 % de las transacciones; con las etiquetas maduras al cierre (días 121 a 152),
detectó el 81,5 % de los fraudes de clientes nuevos con una precisión de 17,7 %. Si una falsa
alarma cuesta 25 US$ o más, el umbral óptimo marca 6 % o menos (ver el informe técnico).

## Explicaciones

Cada respuesta del servicio trae los cinco aportes SHAP (TreeSHAP exacto, en log-odds) de
más peso. En clientes nuevos pesan sobre todo el tipo de tarjeta, el monto y las C1, C13 y
C14 de Vesta. Los aportes SHAP explican el puntaje del modelo; **no son causas**.

## Monitoreo y alertas

Semana a semana, contra los días 91 a 120 (`monitoreo.REGLAS`):

| Señal | Aviso | Crítica |
|---|---|---|
| PSI de las 20 variables con más peso | ≥ 0,10 | ≥ 0,25 |
| PSI del puntaje | ≥ 0,10 | ≥ 0,25 |
| Cambio de faltantes por grupo | ≥ 0,10 | ≥ 0,20 |
| Tasa de alertas / esperada | fuera de [0,7; 1,3] | fuera de [0,5; 1,5] |
| AP madura en nuevos / validación | < 0,85 | < 0,70 |
| Variable con \|Spearman con el día\| ≥ 0,8 (reloj encubierto) | se marca al registrar | |

El rendimiento solo se mide con etiquetas maduras (30 días después de la transacción).

## Limitaciones

- Datos públicos de un solo emisor y seis meses; sin calendario real.
- Las variables opacas de Vesta no tienen definición pública: si cambian de fuente o de
  cálculo, el modelo cambia de comportamiento sin aviso semántico. Son el 57 % del peso SHAP.
- La AP en clientes nuevos de C, R, H y S puede estar inflada por propagación (ver arriba).
- `id_31` (versión del navegador) deriva de forma creciente (PSI 0,49 en la semana 7): el
  codificador de categorías se congela y no conoce las versiones nuevas.
- **Equidad:** IEEE-CIS no trae atributos protegidos (edad, sexo, etnia), así que no se pudo
  medir equidad en ese sentido. Solo se reportan diferencias por producto.
- El retraso del contracargo se supuso fijo (30 días); en la realidad varía y llega a meses.
- Los costos son supuestos y el umbral depende mucho de ellos.

## Plan de contingencia

| Situación | Señal | Acción |
|---|---|---|
| Deriva de una variable concreta | PSI crítico sostenido 2 semanas | Revisar la fuente; si es un reloj o un cambio de cálculo, pasar a una versión sin esa variable ya validada |
| Las variables opacas de Vesta dejan de llegar o cambian | Faltantes de C/D/V/identidad ≥ 0,20 o PSI crítico en varias | Activar **1.1.0-respaldo** (`FRAUDE_VERSION=1.1.0-respaldo`): AP en nuevos 0,235, sin opacas ni relojes |
| Caída de rendimiento | AP madura en nuevos < 0,70 de la validación | Volver a umbral conservador (costo de falsa alarma 25 US$, `FRAUDE_COSTO_FP=25.0`) y reentrenar con las etiquetas maduras |
| Tasa de alertas fuera de rango | Crítica | Revisar el umbral con los puntajes de las últimas 4 semanas sin etiqueta (mismo procedimiento que al desplegar) |
| El servicio falla | `/salud` no responde | Las transacciones pasan sin puntuar y se aplica la regla simple (fraude ya conocido del cliente); se reinicia desde la instantánea del estado |

Cambiar de versión es cambiar `modelos/registro.json` (una sola en producción, con huellas)
y reiniciar el servicio; el servicio verifica el SHA-256 del modelo al arrancar.

## Historia

- **1.0.0** (`completo`, AP en nuevos 0,551): en producción desde el día 121. En la primera
  semana, sin necesitar etiquetas, el monitoreo marcó PSI de 4 a 7 en la antigüedad de
  región, correo y perfil de tarjeta. Con entidades gruesas que aparecen desde el día 1, esa
  antigüedad es el día calendario (Spearman con el día de 0,87 a 1,00): un reloj encubierto
  que en producción toma valores nunca vistos al entrenar. **Retirada.**
- **1.1.0** (`sin_vecindario`): la misma familia sin el vecindario; validación en nuevos
  0,640 contra 0,642 del completo. Decidida con la alerta de la semana 1 y la validación de
  desarrollo; su cifra de prueba ya se había medido una vez, congelada, en el hito 3.
