"""Fixtures. Las pruebas HTTP usan un trabajador falso SOLO aquí (nunca en el servicio)."""

import io
import sys
import threading
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.gpu.convolucion import ResultadoGPU, SinGPU, TiemposGPU  # noqa: E402
from app.gpu.kernels import crear_mascaras, seleccionar_mascara  # noqa: E402
from app.gpu.referencia_cpu import a_uint8, convolucion_cpu  # noqa: E402
from app.gpu.trabajador import ColaLlena  # noqa: E402
from app.main import create_app  # noqa: E402


class MotorFalso:
    """Calcula con la referencia CPU. Permite probar la capa HTTP sin GPU."""

    info = {"nombre": "GPU-falsa", "compute_capability": "0.0", "max_hilos_por_bloque": 1024}

    def procesar(self, pixeles, ancho, alto, filtro, mascara=3, hilos=256, devolver_flotantes=False):
        clave, k, desp = seleccionar_mascara(filtro, mascara)
        flot = convolucion_cpu([float(b) for b in pixeles], crear_mascaras()[clave], k, ancho, alto)
        from array import array

        return ResultadoGPU(a_uint8(flot, desp), ancho, alto, (ancho * alto + hilos - 1) // hilos, hilos,
                            TiemposGPU(kernel_ms=1.0, gpu_total_ms=3.0, seccion_ms=3.5),
                            array("f", flot) if devolver_flotantes else None)


class TrabajadorFalso:
    def __init__(self, disponible=True, cola_llena=False, retardo=None):
        self.disponible, self.cola_llena, self.retardo = disponible, cola_llena, retardo
        self.error_inicio = None if disponible else SinGPU("CUDA no pudo inicializarse: prueba.")
        self.info, self.en_cola, self.procesadas, self.rechazadas = MotorFalso.info, 0, 0, 0
        self.llamadas = 0
        self._motor = MotorFalso()
        self._lock = threading.Lock()

    def enviar(self, funcion):
        if self.cola_llena:
            raise ColaLlena("cola llena de prueba")
        fut = Future()
        with self._lock:  # serializa como el trabajador real
            self.llamadas += 1
            fut.set_result((0.5, funcion(self._motor)))
        return fut

    def iniciar(self):
        pass


def imagen_bytes(ancho=40, alto=30, formato="PNG", modo="L", semilla=0) -> bytes:
    im = Image.new(modo, (ancho, alto))
    px = im.load()
    for y in range(alto):
        for x in range(ancho):
            v = (x * 7 + y * 13 + semilla * 31) % 256
            px[x, y] = v if modo == "L" else (v, (v * 2) % 256, 255 - v)
    buf = io.BytesIO()
    im.save(buf, formato)
    return buf.getvalue()


@pytest.fixture
def cliente(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DIRECTORIO_RESULTADOS", tmp_path / "res")
    trabajador = TrabajadorFalso()
    with TestClient(create_app(trabajador)) as c:
        c.trabajador = trabajador
        yield c


@pytest.fixture
def cliente_sin_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DIRECTORIO_RESULTADOS", tmp_path / "res")
    with TestClient(create_app(TrabajadorFalso(disponible=False))) as c:
        yield c


def hay_gpu() -> bool:
    try:
        from app.gpu.convolucion import MotorGPU

        m = MotorGPU(0)
        m.iniciar()
        m.cerrar()
        return True
    except Exception:
        return False
