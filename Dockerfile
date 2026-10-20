# Servicio de puntuación de fraude (FastAPI), en CPU.
# La instantánea del estado del vecindario (data/estado_dia120.pkl) no se copia: se monta.
FROM python:3.12-slim

# XGBoost en Linux necesita OpenMP del sistema.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencias primero (capa reutilizable), después el código.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev

COPY modelos ./modelos
COPY docs/resultados/hito3/parametros.json docs/resultados/hito3/parametros.json

ENV FRAUDE_ESTADO=/app/data/estado_dia120.pkl
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/salud')"
CMD ["/app/.venv/bin/uvicorn", "fraude.servicio:desde_entorno", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
