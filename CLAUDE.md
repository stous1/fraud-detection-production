# CLAUDE.md

Guía para trabajar en este repositorio.

## Objetivo

Caso de uso 3 del portafolio de ciencia de datos aplicado a banca: **detección de fraude con
desbalance extremo, llevada a producción con monitoreo y gobierno**. Sobre IEEE-CIS (Vesta),
el proyecto reconstruye al cliente detrás de las transacciones, arma un grafo de
identificadores compartidos, entrena un modelo con umbral elegido por costo y lo sirve en un
servicio local con monitoreo de deriva y una ficha del modelo.

Hitos:

1. Ingesta, exploración, sospechas de filtración y reconstrucción del cliente (tarjeta,
   dirección y fechas), con Splink como referencia fuera de línea.
2. Grafo de identidad compartida (tarjeta, correo, dispositivo, dirección) y variables de
   vecindario sin información del futuro.
3. Modelo: XGBoost en GPU como base, frente a una referencia simple y a una red sobre el grafo
   si aporta; umbral por costo y SHAP.
4. Producción: FastAPI en Docker, registro de versiones, monitoreo de deriva y rendimiento
   con las semanas de prueba llegando una a una, alertas y ficha del modelo.
5. Página interactiva (estilo `../causal-customer-analytics/docs/demo.html`), README final,
   verificación de cifras y tag v1.0.

## Stack

- Python 3.12, manejado por **uv** (`.python-version` fija 3.12).
- **DuckDB** local (`data/fraude.duckdb`), limitado a 24 GB con desborde a `data/tmp/`
  (`fraude.base.conectar`). Sin servicios externos.
- pandas, numpy, Splink 5 (sobre DuckDB); Altair + vl-convert para gráficos HTML y PNG.
- **pytest** para pruebas y **ruff** para lint y formato.
- **GPU (RTX 4080 SUPER, 16 GB), cuando entren XGBoost y PyTorch:**
  - XGBoost con `device="cuda"` es el modelo base, y la búsqueda de hiperparámetros corre en
    GPU. CatBoost en GPU vale como comparación.
  - Una red sobre el grafo usa PyTorch Geometric en GPU. torch en dos grupos de uv
    excluyentes, como en `../aml-graph-detection`: `cpu` (predeterminado, CI) y `gpu`
    (`uv sync --no-group cpu --group gpu` y después `uv run --no-sync ...`).
  - Siempre hay respaldo en CPU para la CI. **La evaluación carga los pesos congelados y
    puntúa en CPU**, para que las cifras se reproduzcan exactas. Las pruebas fuerzan CPU.

## Comandos

```bash
uv sync                                       # crea .venv e instala dependencias (incluye dev)
uv run pytest                                 # todas las pruebas (mini-IEEE sintético)
uv run pytest -m "not integracion"            # solo pruebas rápidas
uv run ruff check . && uv run ruff format .   # lint y formato
uv run python scripts/hito1.py preparar       # CSV de train -> data/fraude.duckdb (~15 s)
uv run python scripts/hito1.py explorar       # -> reports/hito1/exploracion.json y gráficos
uv run python scripts/hito1.py clientes       # uid, variantes, Splink -> tabla clientes (~1 min)
uv run python scripts/hito1.py sospechas      # propagación, monotonía, prueba de futuro
uv run python scripts/hito1.py todo           # explorar + clientes + sospechas
uv run python scripts/hito2.py preparar       # entidades y vecindario (~20 s; requiere clientes)
uv run python scripts/hito2.py desarrollar    # variantes en desarrollo -> reports/hito2/parametros.json
uv run python scripts/publicar_resultados.py --hito 2 --solo-parametros   # congelar (+ tag)
uv run python scripts/hito2.py evaluar        # prueba, una vez; se niega sin el tag hito2-congelado
uv run python scripts/hito2.py graficos       # rehace los gráficos desde evaluacion.json
uv run python scripts/hito3.py buscar         # 40 configuraciones en GPU (~10 min)
uv run python scripts/hito3.py desarrollar    # 13 variantes en GPU, umbral por costo, SHAP (~5 min)
uv run python scripts/publicar_resultados.py --hito 3 --solo-parametros   # congelar (+ tag)
uv run python scripts/hito3.py publicar-modelo  # modelo principal -> docs/resultados/hito3/modelos
uv run python scripts/hito3.py evaluar        # prueba, una vez, en CPU; exige hito3-congelado
uv run python scripts/hito4.py paridad        # en línea vs lotes + data/estado_dia120.pkl
uv run python scripts/hito4.py registrar      # modelos/registro.json
uv run python scripts/hito4.py simular        # monitoreo semana a semana -> reports/hito4
docker compose up -d --build                  # servicio en localhost:8000; docker compose down
uv run python scripts/probar_servicio.py --n 3000   # prueba de humo por HTTP
uv run python scripts/verificar_pages.py      # docs/ contra las URL públicas de Pages
uv run python scripts/publicar_resultados.py --hito 1   # reports/hitoN -> docs/resultados/hitoN
```

## Datos

- `data/raw/train_transaction.csv` y `data/raw/train_identity.csv` de IEEE-CIS. **No se
  versionan.** Los `test_*.csv` y `sample_submission.csv` de la competencia no traen etiqueta
  y **no se usan**: `ingesta.verificar_origen` los rechaza y una prueba verifica que el código
  no los nombre.
- 590.540 transacciones, 144.233 con identidad (24 %). `TransactionID` crece con el tiempo.
- `dia = TransactionDT // 86400` (1 a 182). La fecha de calendario es desconocida: no se
  inventa.
- El esquema se fija en `esquema.py` (no se infiere); sirve de contrato para el servicio.
- **Cliente = `uid` = `card1 + addr1 + D1n` exacto**, con `D1n = dia - D1` (día de alta de la
  tarjeta). Tabla `clientes` en la base. La tolerancia de un día y el veto por correo se
  midieron y se descartaron (ver `INFORME_TECNICO.md`). `D1 = 0` es frecuente (95 % en H,
  89 % en R): ahí el `uid` se parte por día.
- `card2`-`card6` y `addr2` dependen de `card1`: no sirven para validar clientes.

## Estructura

- `src/fraude/`: toda la lógica, como funciones reutilizables.
  - `rutas.py`: rutas del proyecto. **Toda ruta se obtiene de aquí.**
  - `base.py`: conexión a DuckDB con memoria limitada, desborde a disco y UTC.
  - `esquema.py`, `ingesta.py`: CSV a DuckDB (`transacciones`, `identidad`).
  - `particion.py`: cortes fijos y vistas `tiempo` y `desarrollo`.
  - `sintetico.py`: mini-IEEE con clientes plantados (tarjetas hermanas, cambios de correo,
    propagación de etiquetas, un contador en línea y un total con futuro), para pruebas y CI.
  - `exploracion.py`, `graficos.py` (Altair, números en español, tooltip en el HTML).
  - `clientes.py`: niveles de cliente, asignación en línea, validación con atributos que no
    se usan para unir (`concordancia`, `referencias`, `cambios`).
  - `splink_ref.py`: Fellegi-Sunter con EM, sin etiquetas, como referencia fuera de línea
    (ve el futuro: **no alimenta al modelo**).
  - `sospechas.py`: propagación de etiquetas (con retraso), monotonía, prueba de futuro y
    registro `GRUPOS` de variables con o sin sospecha (verificado por una prueba).
  - `flujo_hito1.py`: pasos del hito 1. `util.py`: JSON estable.
  - **Quién lee `isFraud`:** solo los módulos listados en `tests/test_flujo_hito1.py`, y en
    el hito 1 siempre a través de la vista `desarrollo`.
  - Hito 2: `grafo.py` (tabla `entidades`: cliente, tarjeta y dirección, perfil de tarjeta,
    huella de dispositivo, región, dominio de correo), `vecindario.py` (tabla `vecindario`,
    139 variables solo con pasado; etiquetas con retraso `R` en 0, 15, 30, 60 y 90;
    componentes en línea con límite de 5 clientes por entidad), `evaluacion.py` (AP con
    bootstrap por cliente, recall a tasa fija de falsas alarmas, curva PR, nuevos y vistos),
    `modelo_lineal.py` (logística con parámetros en JSON), `congelados.py` (verifica un
    archivo contra un tag) y `flujo_hito2.py` (`VARIANTES` declaradas). Congelado en
    `hito2-congelado` con `docs/resultados/hito2/parametros.json`.
  - Hito 3: `variables.py` (matriz, grupos `transaccion`, `sospechosas` con subgrupos C, D,
    V, identidad, `vecindario` e `historia`; `Codificador` de categorías congelado),
    `modelo_xgb.py` (entrenar en GPU o CPU, búsqueda aleatoria, guardar UBJ con SHA-256,
    puntuar y TreeSHAP siempre en CPU), `costos.py` (costo total y umbral como tasa de
    alertas) y `flujo_hito3.py` (`VARIANTES`, cortes de nuevos por producto y estrictos).
    Congelado en `hito3-congelado`; el modelo principal se versiona en
    `docs/resultados/hito3/modelos/completo.ubj`, las ablaciones quedan en
    `reports/hito3/modelos/` con su huella en `parametros.json`.
  - Hito 4: `estado_en_linea.py` (vecindario incremental; paridad exacta con
    `vecindario.py`, en `tests/test_estado_en_linea.py` y en todo train con
    `flujo_hito4.paridad`), `registro.py` + `modelos/registro.json` (versiones con SHA-256;
    una sola en producción), `servicio.py` (FastAPI; `desde_entorno` lee `FRAUDE_REGISTRO`,
    `FRAUDE_ESTADO`, `FRAUDE_VERSION`, `FRAUDE_COSTO_FP`), `monitoreo.py` (PSI, faltantes,
    tasa de alertas, rendimiento maduro, relojes encubiertos, `REGLAS`) y `flujo_hito4.py`.
    Ficha en `docs/ficha_modelo.md`. En producción: **1.1.0 (`sin_vecindario`)**; la 1.0.0
    se retiró por relojes encubiertos en el vecindario.
  - **Relojes encubiertos:** ninguna variable nueva con |Spearman con el día| >= 0,8
    (`monitoreo.correlacion_con_dia`). La antigüedad de entidades gruesas lo es.
  - **Pasado = `TransactionID` menor.** `vecindario.columnas` nunca devuelve la etiqueta ni
    identificadores (en el hito 2 se coló `isFraud` antes de congelar; hay pruebas).
- `scripts/`: puntos de entrada delgados que orquestan funciones de `src/`.
- `notebooks/`: solo exploración. Nada en `src/` ni `scripts/` importa de notebooks.
- `tests/`: pytest. Las lentas se marcan con `@pytest.mark.integracion`. Todas corren sobre el
  mini-IEEE: CI no necesita los datos reales.
- `data/` y `reports/` (ignoradas): datos, base DuckDB y salidas de trabajo.
- `docs/`: fuente de GitHub Pages (rama `main`, carpeta `/docs`). `docs/resultados/` se
  versiona y solo se actualiza con `scripts/publicar_resultados.py`.
- `README.md` es la puerta de entrada corta; el detalle con todas las cifras está en
  `INFORME_TECNICO.md`.

## Convenciones

- La lógica vive en `src/` como funciones reutilizables **con type hints**.
- Las pruebas no dependen de internet ni de `data/raw`.
- **Las pruebas son deterministas**: semillas fijas y nada que dependa del reloj ni de la
  zona horaria. La conexión fija `TimeZone = 'UTC'`; el tiempo es `TransactionDT` en
  segundos, sin zona.
- Documentación, docstrings y mensajes de commit **en español**.
- Rutas centralizadas en `fraude.rutas`; **nunca rutas absolutas** en el código.
- Commits pequeños y enfocados. En modo autónomo no se pide confirmación antes de cada commit:
  al cerrar cada hito se hace push y se entrega un resumen de lo subido.

## Regla de evaluación

- **Partición temporal** fija (`particion.py`, tag `particion-fijada`, decidida antes de mirar
  etiquetas): desarrollo = días 1 a 120; prueba = días 121 a 182 (ocho semanas y una novena
  de seis días).
- **Esquema realista (principal, desde el hito 2):** el modelo entra a producción el día 121
  y, con 30 días de retraso, solo conoce etiquetas hasta el día 90. Entrenamiento 1 a 60,
  validación 61 a 90 (regularización, hiperparámetros, umbral), reajuste 1 a 90. Los días 91
  a 120 son historia sin etiqueta. `todas_las_etiquetas` (entrena 1 a 90, valida 91 a 120,
  reajusta 1 a 120) se reporta **solo como comparación**, para medir cuánto infla.
- **Resultados en clientes nuevos como rango**: sin las variables opacas de Vesta
  (`sin_sospechosas`), con ellas (completo) y W como cota conservadora. Nunca una cifra
  única.
- **Cifra principal: AP en clientes nuevos.** El rendimiento en ya vistos está inflado por la
  propagación de etiquetas de la competencia (en un banco real la tarjeta se bloquea tras un
  contracargo): se reporta aparte, nunca como rendimiento. Los hiperparámetros se eligen por
  AP en nuevos de la validación. En nuevos, W (la mayoría) es la cota conservadora.
- **Siempre por separado: clientes nuevos y ya vistos** (con transacciones antes del periodo;
  en prueba, en desarrollo).
- Toda decisión que mire etiquetas (variables, hiperparámetros, umbral) se toma en
  **desarrollo**. De prueba, antes de congelar, solo se miran volúmenes y faltantes, nunca
  `isFraud`. El modelo se congela con un tag y la prueba se mide **una sola vez**. Cifras de
  desarrollo se presentan marcadas como optimistas.
- Métricas: curva precisión-recall, **AP**, **recall a una tasa fija de falsas alarmas** y
  **costo total esperado según el umbral** (un falso positivo también cuesta). **Nunca
  exactitud.** Siempre contra una **referencia simple**.
- **Retraso del contracargo:** la etiqueta de una transacción se conoce 30 días después
  (supuesto base), con sensibilidad a 15, 60 y 90. Toda variable que use etiquetas pasadas, y
  el monitoreo de rendimiento, respetan ese retraso.
- **Propagación de etiquetas por cliente:** tras un contracargo, la competencia marcó como
  fraude transacciones posteriores ligadas a la misma cuenta, correo o dirección. El modelo
  puede memorizar clientes: en prueba se reporta aparte para clientes nuevos y ya vistos, y se
  hace ablación de las variables de identidad.
- **Variables sospechosas** (pueden usar el futuro o depender de cómo se armó la
  competencia): C1-C14, D1-D15, V1-V339 y las de identidad. Se diagnostican en el hito 1, se
  declaran con o sin sospecha y cada modelo se reporta con y sin ellas.

## Regla de pruebas antes de un commit (para Claude y cualquier subagente)

- **Nunca encadenar pytest (ni ruff) con tuberías que oculten el código de salida**, como
  `uv run pytest | tail -1 && git commit`: el código que ve `&&` es el de `tail`.
- Antes de cada commit, ejecutar las pruebas sin tubería (o redirigir a un archivo) y
  **verificar el código de salida real** (`echo $?` en bash, `$LASTEXITCODE` en PowerShell).
  Solo se hace el commit si es 0.

## Regla de procesos (para Claude y cualquier subagente)

- **Nunca terminar procesos que no hayas iniciado tú.** Nada de `taskkill /IM`, `pkill`,
  `killall` ni `kill` generales por nombre.
- Si un comando se traba, termina solo ese proceso por su identificador.
- Al delegar en un subagente, incluye esta regla en sus instrucciones.

## Regla de credenciales (para Claude y cualquier subagente)

- Las credenciales guardadas en la máquina solo se usan para `git push`. Para cualquier otra
  cosa que use credenciales hay que preguntarle antes al usuario.
- **Excepción (autorizada por el usuario en el hito 3):** se permiten lecturas públicas **sin
  token** de la API de GitHub (por ejemplo `curl https://api.github.com/repos/stous1/...`)
  para ver el estado de CI y de Pages de este repo y de cualquier otro repo público del
  usuario. Nunca con credenciales: sin `gh`, sin cabecera `Authorization`, sin tokens.
- GitHub Pages está activo (rama `main`, carpeta `/docs`): se verifica con peticiones a las
  URL públicas (`https://stous1.github.io/fraud-detection-production/...`), sin credenciales.
  Tras cada push que toque `docs/`, correr `uv run python scripts/verificar_pages.py`
  (compara cada archivo con su versión pública por SHA-256). En el hito 3, GitHub no lanzó
  la compilación de Pages tras un push y el sitio quedó en la versión anterior sin aviso; si
  vuelve a pasar, revisar las ejecuciones con la API pública y hacer un nuevo push.

## Regla de git

- Nunca `git push --force`, borrar ramas o tags ni reescribir la historia.

## Regla dura de datos

- **Nunca usar datos reales de clientes.** Solo datasets públicos (IEEE-CIS) o sintéticos.
- **Tampoco** código, reglas ni ideas de proyectos internos de ningún banco.
- Los repositorios hermanos (`../identity-resolution`, `../aml-graph-detection`,
  `../causal-customer-analytics`) son solo de referencia: no se modifican desde aquí ni se
  corre git en ellos.

## Docker

`Dockerfile` y `docker-compose.yml` levantan el servicio de puntuación (FastAPI, CPU, con
`xgboost-cpu`). La instantánea `data/estado_dia120.pkl` se monta en solo lectura (no va en la
imagen). Solo se detienen los contenedores propios (`docker compose down`).
