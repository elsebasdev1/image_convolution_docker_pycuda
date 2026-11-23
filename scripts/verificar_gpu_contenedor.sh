#!/usr/bin/env bash
# Comprueba que el contenedor ve la GPU (controlador del anfitrión + toolkit de la imagen).
set -euo pipefail
echo "== nvidia-smi dentro del contenedor"; docker run --rm --gpus all convolucion-gpu:1.0 nvidia-smi || true
echo "== nvcc"; docker run --rm --gpus all convolucion-gpu:1.0 nvcc --version | tail -2
echo "== PyCUDA / motor"; docker run --rm --gpus all convolucion-gpu:1.0 python -m app.cli --info
