"""Lectura, validación y codificación de imágenes (Pillow solo para E/S y conversión)."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

from . import config

Image.MAX_IMAGE_PIXELS = config.MAX_PIXELES * 2  # defensa extra contra bombas de descompresión


class ErrorEntrada(Exception):
    """Entrada inválida; `http` y `codigo` los traduce la capa web."""

    def __init__(self, http: int, codigo: str, mensaje: str):
        super().__init__(mensaje)
        self.http, self.codigo, self.mensaje = http, codigo, mensaje


@dataclass
class ImagenGris:
    pixeles: bytes  # u8, fila a fila, ancho*alto bytes
    ancho: int
    alto: int
    formato: str
    bytes_entrada: int
    sha256: str


def sha256(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def decodificar(datos: bytes) -> ImagenGris:
    """Valida formato y dimensiones leyendo solo la cabecera, y luego decodifica a gris.

    La conversión a escala de grises es `convert("L")`, igual que la práctica 02.
    """
    if not datos:
        raise ErrorEntrada(400, "archivo_invalido", "El archivo está vacío.")
    try:
        img = Image.open(io.BytesIO(datos))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise ErrorEntrada(400, "archivo_invalido", "El archivo no es una imagen válida.")

    if img.format not in config.FORMATOS_ENTRADA:
        raise ErrorEntrada(
            415, "formato_no_soportado",
            f"Formato {img.format!r} no soportado. Use: {', '.join(sorted(config.FORMATOS_ENTRADA))}.",
        )
    ancho, alto = img.size  # solo cabecera: todavía no se decodificó el raster
    if min(ancho, alto) < config.MIN_LADO:
        raise ErrorEntrada(422, "dimensiones_invalidas", f"Cada lado debe medir al menos {config.MIN_LADO} px.")
    if max(ancho, alto) > config.MAX_LADO or ancho * alto > config.MAX_PIXELES:
        raise ErrorEntrada(
            413, "imagen_demasiado_grande",
            f"{ancho}x{alto} excede el límite ({config.MAX_PIXELES:,} px, lado máx. {config.MAX_LADO}).",
        )
    try:
        gris = img.convert("L")
        pixeles = gris.tobytes()
    except (OSError, ValueError, Image.DecompressionBombError):
        raise ErrorEntrada(400, "archivo_invalido", "La imagen está corrupta o truncada.")
    return ImagenGris(pixeles, ancho, alto, img.format, len(datos), sha256(datos))


def codificar(pixeles: bytes, ancho: int, alto: int, formato: str) -> tuple[bytes, str]:
    """Codifica el resultado. PNG es sin pérdida; JPEG calidad 100 como en la práctica 02."""
    img = Image.frombytes("L", (ancho, alto), pixeles)
    buf = io.BytesIO()
    if formato == "png":
        img.save(buf, "PNG", compress_level=1)
        return buf.getvalue(), "image/png"
    if formato == "jpeg":
        img.save(buf, "JPEG", quality=100)
        return buf.getvalue(), "image/jpeg"
    raise ValueError(formato)
