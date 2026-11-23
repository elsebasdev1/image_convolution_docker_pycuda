"""Motor de convolución en GPU con PyCUDA. No conoce HTTP.

Modelo de contexto CUDA
-----------------------
* `MotorGPU.iniciar()` crea UN contexto CUDA (`device.make_context()`), lo deja
  activo en el hilo que lo llamó y compila los kernels una sola vez.
* Todas las llamadas CUDA posteriores (`procesar`, `cerrar`) deben hacerse desde
  ese mismo hilo: un contexto de PyCUDA pertenece al hilo que lo hizo activo.
  El servicio lo garantiza con un único hilo trabajador (ver `trabajador.py`);
  la ejecución directa (`app.cli`) lo usa desde el hilo principal.
* `cerrar()` hace `pop()` y `detach()` para liberar el contexto.

Ciclo de una convolución (`procesar`)
-------------------------------------
1. Reserva de VRAM (`mem_alloc`): entrada u8, entrada f32, salida f32, salida u8 y máscara.
2. H2D: máscara (f32) e imagen en escala de grises (u8).      -> evento CUDA
3. Kernel auxiliar u8 -> f32 y relleno de la salida con NaN (0xFF) para detectar
   píxeles sin procesar.
4. Kernel `convolucionGPU` (el de la práctica 02).             -> evento CUDA
5. Kernel auxiliar f32 -> u8 (desplazar, truncar, recortar).   -> evento CUDA
6. (opcional) D2H de la salida f32 antes del recorte, para medir error numérico.
7. D2H de la salida u8.                                        -> evento CUDA
8. Sincronización: `evento_final.synchronize()` antes de leer los tiempos.
9. Liberación de VRAM en `finally`.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from time import perf_counter

from .kernels import FIRMAS, KERNEL_SOURCE, crear_mascaras, seleccionar_mascara

FLOAT_SIZE = 4


class ErrorGPU(Exception):
    """Error controlado del motor; `codigo` es estable y lo usa la capa HTTP."""

    codigo = "error_procesamiento"


class SinGPU(ErrorGPU):
    codigo = "gpu_no_disponible"


class MemoriaGPUInsuficiente(ErrorGPU):
    codigo = "memoria_gpu_insuficiente"


class ErrorProcesamiento(ErrorGPU):
    codigo = "error_procesamiento"


@dataclass
class TiemposGPU:
    """Tiempos en milisegundos. Los *_ms de GPU provienen de eventos CUDA."""

    reserva_ms: float = 0.0   # mem_alloc (reloj de pared, fuera de los eventos)
    h2d_ms: float = 0.0
    kernel_ms: float = 0.0
    auxiliares_ms: float = 0.0  # u8->f32 y f32->u8
    d2h_ms: float = 0.0
    d2h_float_ms: float | None = None  # solo si se pidieron flotantes (validación)
    gpu_total_ms: float = 0.0  # h2d + kernel + auxiliares + d2h
    seccion_ms: float = 0.0   # reloj de pared de todo procesar()


@dataclass
class ResultadoGPU:
    pixeles: bytes
    ancho: int
    alto: int
    bloques: int
    hilos: int
    tiempos: TiemposGPU
    flotantes: array | None = None
    detalles: dict = field(default_factory=dict)


class MotorGPU:
    def __init__(self, ordinal: int = 0):
        self.ordinal = ordinal
        self._cuda = None
        self._ctx = None
        self._device = None
        self._funciones: dict = {}
        self._mascaras = crear_mascaras()
        self._eventos: list = []
        self.info: dict = {}

    # ------------------------------------------------------------------ inicio
    def iniciar(self) -> dict:
        """Inicializa CUDA, crea el contexto y compila los kernels (arranque en frío)."""
        t0 = perf_counter()
        try:
            import pycuda.driver as cuda
        except ImportError as e:  # libcuda.so ausente o PyCUDA mal instalado
            raise SinGPU(
                "No se pudo importar PyCUDA/libcuda ("
                f"{e}). En Docker verifique --gpus all y el NVIDIA Container Toolkit; "
                "en el anfitrión verifique el controlador NVIDIA."
            ) from e
        self._cuda = cuda

        try:
            cuda.init()
            total = cuda.Device.count()
            if total == 0:
                raise SinGPU("CUDA inicializó pero no hay dispositivos visibles (0 GPUs).")
            if not 0 <= self.ordinal < total:
                raise SinGPU(f"GPU_ORDINAL={self.ordinal} fuera de rango; hay {total} GPU(s).")
            self._device = cuda.Device(self.ordinal)
            self._ctx = self._device.make_context()
        except cuda.Error as e:
            raise SinGPU(
                f"CUDA no pudo inicializarse: {e}. Causas típicas: contenedor sin --gpus all, "
                "NVIDIA Container Toolkit sin configurar, o desajuste entre el módulo del kernel "
                "y las bibliotecas del controlador (reiniciar el anfitrión)."
            ) from e

        try:
            from pycuda.compiler import SourceModule

            modulo = SourceModule(KERNEL_SOURCE)
            for nombre, firma in FIRMAS.items():
                f = modulo.get_function(nombre)
                f.prepare(firma)
                self._funciones[nombre] = f
            self._eventos = [cuda.Event() for _ in range(8)]
        except Exception as e:
            self.cerrar()
            raise ErrorProcesamiento(f"Falló la compilación/carga de los kernels: {e}") from e

        dev = self._device
        cc = dev.compute_capability()
        self.info = {
            "nombre": dev.name(),
            "compute_capability": f"{cc[0]}.{cc[1]}",
            "memoria_total_mb": dev.total_memory() // (1024 * 1024),
            "multiprocesadores": dev.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT),
            "max_hilos_por_bloque": dev.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK),
            "ordinal": self.ordinal,
            "cuda_driver_api": ".".join(map(str, _version(cuda.get_driver_version()))),
            "cuda_pycuda_compilado_con": ".".join(map(str, cuda.get_version())),
            "inicio_ms": round((perf_counter() - t0) * 1000.0, 1),
        }
        return self.info

    def cerrar(self) -> None:
        """Libera el contexto CUDA (debe llamarse desde el hilo dueño)."""
        if self._ctx is not None:
            try:
                self._ctx.pop()
                self._ctx.detach()
            except Exception:  # el contexto ya podía estar roto
                pass
            self._ctx = None

    def memoria_libre_mb(self) -> int | None:
        if self._cuda is None or self._ctx is None:
            return None
        libre, _ = self._cuda.mem_get_info()
        return libre // (1024 * 1024)

    # --------------------------------------------------------------- procesado
    def procesar(
        self,
        pixeles: bytes,
        ancho: int,
        alto: int,
        filtro: str,
        mascara: int = 3,
        hilos: int = 256,
        devolver_flotantes: bool = False,
    ) -> ResultadoGPU:
        if self._ctx is None:
            raise SinGPU("El contexto CUDA no está inicializado.")
        cuda = self._cuda
        n = ancho * alto
        if len(pixeles) != n:
            raise ErrorProcesamiento(f"Se esperaban {n} bytes (u8) y llegaron {len(pixeles)}.")
        if hilos > self.info["max_hilos_por_bloque"]:
            raise ErrorProcesamiento(
                f"{hilos} hilos/bloque supera el máximo de la GPU ({self.info['max_hilos_por_bloque']})."
            )

        clave, k, desplazamiento = seleccionar_mascara(filtro, mascara)
        h_mask = array("f", self._mascaras[clave])
        bytes_f32, bytes_mask = n * FLOAT_SIZE, k * k * FLOAT_SIZE
        necesarios = 2 * bytes_f32 + 2 * n + bytes_mask

        libre, _ = cuda.mem_get_info()
        if necesarios > libre:
            raise MemoriaGPUInsuficiente(
                f"Se requieren {necesarios // 2**20} MB de VRAM y hay {libre // 2**20} MB libres."
            )

        t_seccion = perf_counter()
        t = TiemposGPU()
        bloques = (n + hilos - 1) // hilos
        grid, block = (bloques, 1, 1), (hilos, 1, 1)
        ev = self._eventos
        f_conv, f_cast_in, f_cast_out = (
            self._funciones["convolucionGPU"],
            self._funciones["u8_a_float"],
            self._funciones["float_a_u8"],
        )
        d_u8_in = d_f_in = d_f_out = d_u8_out = d_mask = None
        h_out = bytearray(n)
        flotantes = array("f", bytes(bytes_f32)) if devolver_flotantes else None
        try:
            t_r = perf_counter()
            d_u8_in = cuda.mem_alloc(n)
            d_f_in = cuda.mem_alloc(bytes_f32)
            d_f_out = cuda.mem_alloc(bytes_f32)
            d_u8_out = cuda.mem_alloc(n)
            d_mask = cuda.mem_alloc(bytes_mask)
            t.reserva_ms = (perf_counter() - t_r) * 1000.0

            # H2D
            ev[0].record()
            cuda.memcpy_htod(d_mask, h_mask)
            cuda.memcpy_htod(d_u8_in, pixeles)
            ev[1].record()

            # u8 -> f32 (y salida inicializada a NaN para detectar píxeles sin cubrir)
            ev[2].record()
            f_cast_in.prepared_call(grid, block, d_u8_in, d_f_in, n)
            ev[3].record()
            cuda.memset_d8(d_f_out, 0xFF, bytes_f32)

            # Convolución (kernel de la práctica 02)
            ev[4].record()
            f_conv.prepared_call(grid, block, d_f_in, d_f_out, d_mask, k, ancho, alto)
            ev[5].record()

            # f32 -> u8
            ev[6].record()
            f_cast_out.prepared_call(grid, block, d_f_out, d_u8_out, n, desplazamiento)
            ev[7].record()
            ev[7].synchronize()
            t.kernel_ms = ev[4].time_till(ev[5])
            t.auxiliares_ms = ev[2].time_till(ev[3]) + ev[6].time_till(ev[7])

            # D2H opcional de los flotantes (validación numérica, antes del recorte)
            if devolver_flotantes:
                t_f = perf_counter()
                cuda.memcpy_dtoh(flotantes, d_f_out)
                t.d2h_float_ms = (perf_counter() - t_f) * 1000.0

            # D2H de la imagen final
            d2h_ini, d2h_fin = cuda.Event(), cuda.Event()
            d2h_ini.record()
            cuda.memcpy_dtoh(h_out, d_u8_out)
            d2h_fin.record()
            d2h_fin.synchronize()  # sincronizar antes de consultar tiempos/resultados
            t.d2h_ms = d2h_ini.time_till(d2h_fin)
            t.h2d_ms = ev[0].time_till(ev[1])
        except cuda.MemoryError as e:
            raise MemoriaGPUInsuficiente(f"cuMemAlloc falló: {e}") from e
        except cuda.Error as e:
            raise ErrorProcesamiento(f"Error CUDA durante el procesamiento: {e}") from e
        finally:
            for d in (d_u8_in, d_f_in, d_f_out, d_u8_out, d_mask):
                if d is not None:
                    d.free()

        t.gpu_total_ms = t.h2d_ms + t.kernel_ms + t.auxiliares_ms + t.d2h_ms
        t.seccion_ms = (perf_counter() - t_seccion) * 1000.0
        return ResultadoGPU(
            pixeles=bytes(h_out), ancho=ancho, alto=alto, bloques=bloques, hilos=hilos,
            tiempos=t, flotantes=flotantes,
            detalles={"mascara_clave": clave, "desplazamiento": desplazamiento},
        )


def _version(v) -> tuple:
    """get_driver_version() devuelve un entero (12040 -> 12.4); get_version() una tupla."""
    if isinstance(v, int):
        return (v // 1000, (v % 1000) // 10)
    return tuple(v)
