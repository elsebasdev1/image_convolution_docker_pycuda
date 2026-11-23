# syntax=docker/dockerfile:1.7
#
# Servicio de convolución GPU (FastAPI + PyCUDA).
#
# Compatibilidad (ver README §Compatibilidad):
#   * La imagen aporta el TOOLKIT CUDA 13.4.3 (nvcc, cabeceras, libcudart). PyCUDA compila
#     los kernels en tiempo de ejecución, por eso se usa la variante -devel y no -runtime.
#   * El anfitrión aporta el CONTROLADOR NVIDIA (libcuda.so y el módulo del kernel). El
#     NVIDIA Container Toolkit lo inyecta al arrancar con --gpus all; NO se instala aquí.
#   * Requisito: controlador del anfitrión con soporte CUDA >= 13.4.
FROM nvidia/cuda:13.4.3-devel-ubuntu24.04

ARG GIT_COMMIT=desconocido
LABEL org.opencontainers.image.title="convolucion-gpu-web" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.base.name="nvidia/cuda:13.4.3-devel-ubuntu24.04"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GIT_COMMIT=${GIT_COMMIT} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Python 3.12 del sistema (Ubuntu 24.04) + herramientas para compilar PyCUDA desde el sdist
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends build-essential python3 python3-dev && \
    rm -rf /var/lib/apt/lists/*

# uv con versión fija
COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=/usr/bin/python3 \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Capa de dependencias (se reutiliza mientras no cambien pyproject.toml / uv.lock)
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project

# Código de la aplicación
COPY app ./app
COPY web ./web
COPY bench ./bench
COPY tests ./tests

# Usuario sin privilegios; HOME propio para la caché de compilación de PyCUDA
RUN useradd --create-home --uid 10001 app && mkdir -p /app/resultados && chown -R app /app/resultados
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"

# Un solo proceso uvicorn (un contexto CUDA), sin --reload ni depurador.
CMD ["python", "-m", "app"]
