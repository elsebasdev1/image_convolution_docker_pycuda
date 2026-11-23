#!/usr/bin/env bash
# Evidencia del diagnóstico sin GPU: el mismo contenedor SIN --gpus all debe arrancar,
# informar el motivo en /api/health y responder 503 (nunca procesar en CPU).
set -euo pipefail
cd "$(dirname "$0")/.."
SALIDA=${1:-resultados/sin_gpu.txt}
IMG=${IMAGEN:-convolucion-gpu:1.0}
PUERTO=8001
docker rm -f conv-sin-gpu >/dev/null 2>&1 || true
docker run -d --rm --name conv-sin-gpu -p $PUERTO:8000 "$IMG" >/dev/null
trap 'docker rm -f conv-sin-gpu >/dev/null 2>&1 || true' EXIT
for _ in $(seq 1 60); do curl -sf localhost:$PUERTO/api/health >/dev/null 2>&1 && break; sleep 1; done
PNG=$(mktemp --suffix=.png); TXT=$(mktemp --suffix=.txt); echo "esto no es una imagen" > "$TXT"
python3 - "$PNG" <<'PY'
import struct, sys, zlib
w = h = 8
raw = b"".join(b"\x00" + bytes([(x * 32) % 256 for x in range(w)]) for _ in range(h))
def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
open(sys.argv[1], "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
PY
{
  echo "# docker run (sin --gpus all) de $IMG"
  echo; echo "# GET /api/health"; curl -s localhost:$PUERTO/api/health | python3 -c "import sys,json; h=json.load(sys.stdin); print(json.dumps({'estado':h['estado'],'gpu':h['gpu']}, indent=2, ensure_ascii=False))"
  echo; echo "# POST /api/v1/convolucion con imagen válida"
  curl -s -w '\nHTTP %{http_code}\n' -F imagen=@"$PNG" -F filtro=media -F mascara=3 localhost:$PUERTO/api/v1/convolucion
  echo; echo "# POST con archivo inválido (se valida antes de la GPU)"
  curl -s -w '\nHTTP %{http_code}\n' -F imagen=@"$TXT" -F filtro=media localhost:$PUERTO/api/v1/convolucion
} > "$SALIDA"
cat "$SALIDA"
