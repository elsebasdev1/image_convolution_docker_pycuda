"""Pruebas con GPU real (se omiten si no hay GPU). Cubren la sección 7 de la guía."""

import math
from array import array

import pytest
from fastapi.testclient import TestClient

from conftest import hay_gpu, imagen_bytes
from app import config
from app.gpu.convolucion import MotorGPU
from app.gpu.kernels import crear_mascaras, seleccionar_mascara
from app.gpu.referencia_cpu import a_uint8, calcular_errores, convolucion_cpu
from app.main import create_app

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not hay_gpu(), reason="sin GPU NVIDIA operativa")]
TOLERANCIA = 1e-2  # misma de la práctica 02; float32 con hasta 441 sumandos de magnitud <=255


@pytest.fixture(scope="module")
def motor():
    m = MotorGPU(0)
    m.iniciar()
    yield m
    m.cerrar()


def pixeles_sinteticos(w, h) -> bytes:
    return bytes((i * 31 + (i // w) * 17) % 256 for i in range(w * h))


def test_matriz_manual_laplaciano(motor):
    """Prueba pequeña de la práctica 02: matriz 3x3 resuelta a mano."""
    entrada = bytes([1, 2, 3, 4, 5, 6, 7, 8, 9])
    esperado = [2, 1, -4, -3, 0, -7, -16, -11, -22]
    r = motor.procesar(entrada, 3, 3, "laplaciano", 3, 4, devolver_flotantes=True)
    assert list(r.flotantes) == esperado
    assert list(r.pixeles) == [max(0, min(255, v + 128)) for v in esperado]


@pytest.mark.parametrize("filtro,mascara", [("media", 3), ("media", 9), ("nitidez", 3), ("laplaciano", 3)])
def test_limites_dimension_no_multiplo_de_hilos(motor, filtro, mascara):
    """301x203 = 61 103 px (no múltiplo de 256): todos cubiertos y igual a la CPU."""
    w, h, tpb = 301, 203, 256
    assert (w * h) % tpb != 0
    pix = pixeles_sinteticos(w, h)
    r = motor.procesar(pix, w, h, filtro, mascara, tpb, devolver_flotantes=True)
    assert not any(math.isnan(v) for v in r.flotantes), "hay píxeles sin procesar"
    clave, k, desp = seleccionar_mascara(filtro, mascara)
    cpu = convolucion_cpu([float(b) for b in pix], crear_mascaras()[clave], k, w, h)
    emax, emed = calcular_errores(cpu, r.flotantes)
    assert emax <= TOLERANCIA, (emax, emed)
    # La CPU de referencia acumula en double y la GPU en float32: al truncar a entero, un valor
    # como 61.99999 vs 62.00001 cambia el byte en 1. Nunca más que eso.
    esperado = a_uint8(cpu, desp)
    assert max(abs(a - b) for a, b in zip(r.pixeles, esperado)) <= 1


def test_bordes_extension_con_ceros(motor):
    """Esquina de una imagen de unos con media 3x3: 4/9 (solo 4 vecinos dentro)."""
    r = motor.procesar(bytes([1] * 25), 5, 5, "media", 3, 64, devolver_flotantes=True)
    f = r.flotantes
    assert math.isclose(f[0], 4 / 9, rel_tol=1e-6) and math.isclose(f[2], 6 / 9, rel_tol=1e-6)
    assert math.isclose(f[12], 1.0, rel_tol=1e-6)


def test_equivalencia_directa_servicio(tmp_path, monkeypatch, motor):
    """Mismo kernel por dos caminos: motor directo vs. POST al servicio (raw32, antes del recorte)."""
    monkeypatch.setattr(config, "DIRECTORIO_RESULTADOS", tmp_path)
    datos = imagen_bytes(500, 400, formato="PNG", modo="L")
    from app.imagen import decodificar

    img = decodificar(datos)
    with TestClient(create_app()) as c:
        for filtro, mascara in (("media", 9), ("nitidez", 3), ("laplaciano", 3)):
            directo = motor.procesar(img.pixeles, img.ancho, img.alto, filtro, mascara, 256, devolver_flotantes=True)
            j = c.post("/api/v1/convolucion", files={"imagen": ("i.png", datos, "image/png")},
                       data={"filtro": filtro, "mascara": str(mascara), "formato": "raw32"}).json()
            servicio = array("f")
            servicio.frombytes(c.get(j["salida"]["url"]).content)
            emax, emed = calcular_errores(directo.flotantes, servicio)
            assert emax == 0.0 and emed == 0.0, (filtro, emax, emed)


def test_servicio_real_errores_y_aislamiento(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DIRECTORIO_RESULTADOS", tmp_path)
    a, b = imagen_bytes(64, 48, semilla=1), imagen_bytes(80, 60, semilla=2)
    with TestClient(create_app()) as c:
        assert c.get("/api/health").json()["gpu"]["disponible"]
        ra = c.post("/api/v1/convolucion", files={"imagen": ("a.png", a, "image/png")}, data={"filtro": "media", "id_solicitud": "gpu-test-a-1"}).json()
        rb = c.post("/api/v1/convolucion", files={"imagen": ("b.png", b, "image/png")}, data={"filtro": "laplaciano", "id_solicitud": "gpu-test-b-1"}).json()
        assert ra["tiempos"]["kernel_ms"] > 0 and rb["id_resultado"] != ra["id_resultado"]
        assert (ra["entrada"]["ancho"], rb["entrada"]["ancho"]) == (64, 80)
        assert c.post("/api/v1/convolucion", files={"imagen": ("a.png", b"no", "image/png")}, data={"filtro": "media"}).status_code == 400
