#!/usr/bin/env bash
# Protocolo completo de mediciones y validación con la versión confirmada (commit) actual.
# Orden: directo en el anfitrión -> directo en contenedor -> servicio en contenedor -> validación.
set -euo pipefail
cd "$(dirname "$0")/.."
export GIT_COMMIT="$(git rev-parse HEAD)"   # etiqueta de la imagen = commit medido
[ -z "$(git status --porcelain)" ] || { echo "Árbol con cambios: haga commit antes de medir." >&2; exit 1; }

make down >/dev/null 2>&1 || true
docker compose build
echo "== 8.1/8.2/8.3 directo en el anfitrión";  uv run python -m bench.directo --modo directo_host
echo "== 8.1/8.2/8.3 directo en el contenedor"
docker run --rm --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp -e GIT_COMMIT="$(git rev-parse HEAD)" \
  -v "$PWD/datos:/app/datos:ro" -v "$PWD/resultados:/app/resultados" \
  convolucion-gpu:1.0 python -m bench.directo --modo directo_contenedor
make up
echo "== 8.1/8.2/8.3/8.4 contra el servicio contenerizado"
uv run python -m bench.cliente --modo servicio_contenedor --concurrencias 1,2,4
uv run python -m bench.resumir
echo "== validación funcional y numérica"
uv run python -m bench.validacion
{ echo "# docker ps"; docker ps --format 'table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}\t{{.Names}}'
  echo; echo "# imagen"; docker image inspect convolucion-gpu:1.0 --format 'id={{.Id}}
revision={{index .Config.Labels "org.opencontainers.image.revision"}}
base={{index .Config.Labels "org.opencontainers.image.base.name"}}'
  echo; echo "# nvidia-smi dentro del contenedor"; docker exec convolucion-gpu nvidia-smi
  echo; echo "# nvcc"; docker exec convolucion-gpu nvcc --version | tail -2
  echo; echo "# python / pycuda"; docker exec convolucion-gpu python -c "import sys,pycuda; print(sys.version.split()[0], pycuda.VERSION_TEXT)"
  echo; echo "# health"; curl -s localhost:8000/api/health | python3 -m json.tool; } > resultados/contenedor_gpu.txt 2>&1
echo "commit: $(git rev-parse HEAD)"
