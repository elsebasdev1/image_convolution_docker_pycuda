"""Evidencia visual con la imagen cuadrada de Arduino (600x600) procesada por el servicio.

    uv run python -m bench.evidencia_visual --url http://localhost:8000
Guarda original y resultados en docs/evidencia/arduino/.
"""

import argparse

from PIL import Image

from . import comun
from .validacion import http_get, http_post

DESTINO = comun.RAIZ / "docs" / "evidencia" / "arduino"
CONFIGS = (("media9", "media", 9), ("nitidez", "nitidez", 3), ("laplaciano", "laplaciano", 3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    a = ap.parse_args()
    DESTINO.mkdir(parents=True, exist_ok=True)
    ruta = comun.DIR_IMAGENES / "arduino_cuadrada.png"
    datos = ruta.read_bytes()
    Image.open(ruta).convert("L").save(DESTINO / "original.png")
    for nombre, filtro, m in CONFIGS:
        st, j = http_post(a.url, datos, {"filtro": filtro, "mascara": m, "formato": "png"}, "arduino_cuadrada.png", "image/png")
        assert st == 200, j
        (DESTINO / f"{nombre}.png").write_bytes(http_get(a.url, j["salida"]["url"]))
        print(nombre, j["salida"]["sha256"][:16], f"kernel {j['tiempos']['kernel_ms']} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
