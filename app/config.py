"""Límites y parámetros del servicio. Todos se pueden sobrescribir con variables de entorno."""

import os
from pathlib import Path


def _int(nombre: str, defecto: int) -> int:
    return int(os.environ.get(nombre, defecto))


# --- Límites de entrada -------------------------------------------------------
# 6000x6000 = 36 Mpx es la mayor resolución exigida por la práctica.
MAX_UPLOAD_BYTES = _int("MAX_UPLOAD_MB", 32) * 1024 * 1024
MAX_PIXELES = _int("MAX_PIXELES", 36_000_000)
MAX_LADO = _int("MAX_LADO", 8000)
MIN_LADO = 3
FORMATOS_ENTRADA = {"JPEG", "PNG", "BMP"}

# --- Parámetros de procesamiento permitidos -----------------------------------
FILTROS = ("media", "nitidez", "laplaciano")
MASCARAS_MEDIA = (3, 9, 21)
HILOS_PERMITIDOS = (64, 128, 256, 512, 1024)
HILOS_DEFECTO = 256
FORMATOS_SALIDA = ("png", "jpeg", "raw32")

# --- Cola y trabajador GPU ----------------------------------------------------
# Un único trabajador es dueño del contexto CUDA; las solicitudes esperan en una
# cola acotada. Si se llena, el servicio rechaza con 429 (no acumula memoria).
TAMANO_COLA = _int("TAMANO_COLA", 8)
TIEMPO_MAX_COLA_S = _int("TIEMPO_MAX_COLA_S", 120)
GPU_ORDINAL = _int("GPU_ORDINAL", 0)

# --- Resultados temporales ----------------------------------------------------
DIRECTORIO_RESULTADOS = Path(os.environ.get("DIRECTORIO_RESULTADOS", "/tmp/convolucion-resultados"))
TTL_RESULTADOS_S = _int("TTL_RESULTADOS_S", 900)
MAX_RESULTADOS = _int("MAX_RESULTADOS", 20)
