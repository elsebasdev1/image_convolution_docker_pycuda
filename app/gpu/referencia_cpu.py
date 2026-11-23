"""Convolución secuencial en CPU (referencia de la práctica 02, sin NumPy).

Solo se usa en pruebas con imágenes pequeñas: es O(w*h*k^2) en Python puro.
"""


def convolucion_cpu(img_in, mascara, mask_size: int, width: int, height: int) -> list[float]:
    offset = mask_size // 2
    img_out = [0.0] * (width * height)
    for r in range(height):
        for c in range(width):
            suma = 0.0
            for mr in range(-offset, offset + 1):
                img_r = r + mr
                if 0 <= img_r < height:
                    fila_img = img_r * width
                    fila_mask = (mr + offset) * mask_size
                    for mc in range(-offset, offset + 1):
                        img_c = c + mc
                        if 0 <= img_c < width:
                            suma += img_in[fila_img + img_c] * mascara[fila_mask + (mc + offset)]
            img_out[r * width + c] = suma
    return img_out


def a_uint8(valores, desplazamiento: float = 0.0) -> bytes:
    """Mismo tratamiento de visualización que float_a_u8 (truncar y recortar)."""
    return bytes(max(0, min(255, int(v + desplazamiento))) for v in valores)


def calcular_errores(a, b) -> tuple[float, float]:
    """Error absoluto máximo y medio. Un NaN cuenta como error infinito."""
    import math

    maximo, suma = 0.0, 0.0
    for x, y in zip(a, b):
        d = math.fabs(x - y)
        if not (d <= maximo):
            maximo = d
        suma += d
    return maximo, suma / len(a)
