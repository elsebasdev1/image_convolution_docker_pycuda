"""Kernels CUDA propios y definición de máscaras.

El kernel `convolucionGPU` es el de la práctica 02 (commit fb95088 de
elsebasdev1/image_convolution_processing) sin cambios: un píxel de salida por
hilo, distribución 1D en X, extensión con ceros, acumulación en float32.

Los kernels `u8_a_float` y `float_a_u8` son auxiliares nuevos de esta práctica:
mueven la conversión uint8<->float32 a la GPU para no recorrer 36 Mpx en Python
y reducir 4x la transferencia. No son filtros; no sustituyen la convolución.
"""

KERNEL_SOURCE = r"""
extern "C" __global__
void convolucionGPU(const float *img_in, float *img_out, const float *mascara,
                    int mask_size, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = width * height;
    if (idx >= n) return;

    int row = idx / width;
    int col = idx % width;
    int offset = mask_size / 2;
    float suma = 0.0f;

    for (int mr = -offset; mr <= offset; ++mr) {
        for (int mc = -offset; mc <= offset; ++mc) {
            int img_r = row + mr;
            int img_c = col + mc;
            if (img_r >= 0 && img_r < height && img_c >= 0 && img_c < width) {
                suma += img_in[img_r * width + img_c]
                      * mascara[(mr + offset) * mask_size + (mc + offset)];
            }
        }
    }
    img_out[idx] = suma;
}

extern "C" __global__
void u8_a_float(const unsigned char *src, float *dst, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = (float)src[idx];
}

// Mismo tratamiento que la práctica 02: sumar desplazamiento (128 en el
// laplaciano), truncar hacia cero como int() de Python y recortar a [0, 255].
// Un NaN (píxel no procesado) se convierte en 0 por la comparación negada.
extern "C" __global__
void float_a_u8(const float *src, unsigned char *dst, int n, float desplazamiento)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int v = (int)(src[idx] + desplazamiento);
    if (!(v > 0)) v = 0;
    if (v > 255) v = 255;
    dst[idx] = (unsigned char)v;
}
"""

# Firmas para PyCUDA (P = puntero, i = int32, f = float32)
FIRMAS = {
    "convolucionGPU": "PPPiii",
    "u8_a_float": "PPi",
    "float_a_u8": "PPif",
}


def crear_mascaras() -> dict[str, list[float]]:
    """Máscaras de la práctica 02 como arreglos 1D."""
    return {
        "media_3": [1.0 / 9.0] * 9,
        "media_9": [1.0 / 81.0] * 81,
        "media_21": [1.0 / 441.0] * 441,
        "nitidez": [0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0],
        "laplaciano": [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0],
    }


def seleccionar_mascara(filtro: str, mascara: int) -> tuple[str, int, float]:
    """Devuelve (clave_de_mascara, tamano, desplazamiento_de_visualizacion)."""
    if filtro == "media":
        return f"media_{mascara}", mascara, 0.0
    if filtro == "nitidez":
        return "nitidez", 3, 0.0
    if filtro == "laplaciano":
        return "laplaciano", 3, 128.0
    raise ValueError(f"filtro desconocido: {filtro}")
