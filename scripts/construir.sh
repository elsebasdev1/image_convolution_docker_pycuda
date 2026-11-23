#!/usr/bin/env bash
# Construye la imagen etiquetando el commit exacto. Falla si hay cambios sin confirmar.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -n "$(git status --porcelain)" ]; then echo "Hay cambios sin confirmar: haga commit antes de construir." >&2; exit 1; fi
COMMIT=$(git rev-parse HEAD)
docker build --build-arg GIT_COMMIT="$COMMIT" -t convolucion-gpu:1.0 -t "convolucion-gpu:${COMMIT:0:12}" .
echo "commit: $COMMIT"
echo "imagen: $(docker image inspect convolucion-gpu:1.0 --format '{{.Id}}')"
