"""Hilo trabajador dueño del contexto CUDA y cola acotada de trabajos.

Política: FIFO, un trabajo a la vez (la GPU se usa en exclusiva), cola de
`TAMANO_COLA` plazas. Cola llena -> `ColaLlena` (HTTP 429). Un trabajo que espera
más de `TIEMPO_MAX_COLA_S` se descarta (HTTP 504) en lugar de ocupar la GPU.
"""

from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

from .convolucion import ErrorGPU, MotorGPU


class ColaLlena(Exception):
    pass


class TiempoColaAgotado(Exception):
    pass


@dataclass
class _Trabajo:
    funcion: Callable[[MotorGPU], object]
    futuro: Future
    t_encolado: float


class TrabajadorGPU:
    def __init__(self, ordinal: int, tam_cola: int, tiempo_max_cola_s: float):
        self.ordinal = ordinal
        self.tiempo_max_cola_s = tiempo_max_cola_s
        self._cola: queue.Queue[_Trabajo | None] = queue.Queue(maxsize=tam_cola)
        self._hilo = threading.Thread(target=self._bucle, name="trabajador-gpu", daemon=True)
        self._listo = threading.Event()
        self.motor = MotorGPU(ordinal)
        self.error_inicio: ErrorGPU | Exception | None = None
        self.info: dict = {}
        self.procesadas = 0
        self.rechazadas = 0

    def iniciar(self, esperar_s: float = 300.0) -> None:
        self._hilo.start()
        self._listo.wait(esperar_s)

    @property
    def disponible(self) -> bool:
        return self._listo.is_set() and self.error_inicio is None

    @property
    def en_cola(self) -> int:
        return self._cola.qsize()

    def enviar(self, funcion: Callable[[MotorGPU], object]) -> Future:
        """Encola `funcion(motor)`. Se ejecutará en el hilo dueño del contexto."""
        fut: Future = Future()
        try:
            self._cola.put_nowait(_Trabajo(funcion, fut, perf_counter()))
        except queue.Full:
            self.rechazadas += 1
            raise ColaLlena(f"La cola de la GPU está llena ({self._cola.maxsize} solicitudes).")
        return fut

    def detener(self) -> None:
        self._cola.put(None)
        self._hilo.join(timeout=10)

    def _bucle(self) -> None:
        try:
            self.info = self.motor.iniciar()
        except Exception as e:  # SinGPU u otro: el servicio sigue vivo y lo informa
            self.error_inicio = e
            self._listo.set()
            self._vaciar_con_error(e)
            return
        self._listo.set()
        try:
            while True:
                trabajo = self._cola.get()
                if trabajo is None:
                    break
                if not trabajo.futuro.set_running_or_notify_cancel():
                    continue  # el cliente canceló mientras esperaba
                espera = perf_counter() - trabajo.t_encolado
                if espera > self.tiempo_max_cola_s:
                    trabajo.futuro.set_exception(TiempoColaAgotado(f"esperó {espera:.1f} s en cola"))
                    continue
                try:
                    trabajo.futuro.set_result((espera * 1000.0, trabajo.funcion(self.motor)))
                    self.procesadas += 1
                except BaseException as e:
                    trabajo.futuro.set_exception(e)
        finally:
            self.motor.cerrar()

    def _vaciar_con_error(self, e: Exception) -> None:
        """Sin GPU: los trabajos que lleguen reciben el diagnóstico en vez de esperar."""
        while True:
            trabajo = self._cola.get()
            if trabajo is None:
                return
            if trabajo.futuro.set_running_or_notify_cancel():
                trabajo.futuro.set_exception(e)
