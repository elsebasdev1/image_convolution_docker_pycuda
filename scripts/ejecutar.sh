#!/usr/bin/env bash
# Arranca el servicio con acceso a GPU en http://localhost:8000
set -euo pipefail
docker run --rm --gpus all --name convolucion-gpu -p 8000:8000 \
  -v "$(pwd)/datos:/app/datos:ro" -v "$(pwd)/resultados:/app/resultados" \
  convolucion-gpu:1.0 "$@"
