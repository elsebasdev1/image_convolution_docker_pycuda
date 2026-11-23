#!/usr/bin/env bash
# Arranque en frío del motor: cuda.init + make_context + compilación de kernels (campo inicio_ms).
# 5 arranques en el anfitrión (con caché de compilación de PyCUDA) y 5 en contenedores nuevos (sin caché).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=resultados/arranque.csv
echo "modo,repeticion,inicio_ms" > "$OUT"
for i in 1 2 3 4 5; do
  ms=$(uv run python -m app.cli --info | python3 -c "import sys,json; print(json.load(sys.stdin)['inicio_ms'])")
  echo "directo_host,$i,$ms" >> "$OUT"
done
for i in 1 2 3 4 5; do
  ms=$(docker run --rm --gpus all convolucion-gpu:1.0 python -m app.cli --info 2>/dev/null | sed -n '/^{/,$p' | python3 -c "import sys,json; print(json.load(sys.stdin)['inicio_ms'])")
  echo "directo_contenedor,$i,$ms" >> "$OUT"
done
cat "$OUT"
