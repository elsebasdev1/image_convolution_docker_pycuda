"""Almacén de resultados temporales: un archivo por resultado, ID único, TTL y tope."""

from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Resultado:
    id: str
    ruta: Path
    tipo_mime: str
    nombre_descarga: str
    creado: float
    meta: dict


class AlmacenResultados:
    def __init__(self, directorio: Path, ttl_s: int, maximo: int):
        self.directorio, self.ttl_s, self.maximo = directorio, ttl_s, maximo
        self._items: dict[str, Resultado] = {}
        self._lock = threading.Lock()
        shutil.rmtree(directorio, ignore_errors=True)  # no heredar archivos de otra ejecución
        directorio.mkdir(parents=True, exist_ok=True)

    def guardar(self, datos: bytes, extension: str, tipo_mime: str, nombre: str, meta: dict) -> Resultado:
        rid = uuid.uuid4().hex  # ID aleatorio: dos solicitudes nunca comparten archivo
        ruta = self.directorio / f"{rid}.{extension}"
        tmp = ruta.with_suffix(".tmp")
        tmp.write_bytes(datos)
        os.replace(tmp, ruta)  # escritura atómica
        res = Resultado(rid, ruta, tipo_mime, nombre, time.time(), meta)
        with self._lock:
            self._items[rid] = res
            self._purgar_locked()
        return res

    def obtener(self, rid: str) -> Resultado | None:
        with self._lock:
            self._purgar_locked()
            return self._items.get(rid)

    def cantidad(self) -> int:
        with self._lock:
            return len(self._items)

    def _purgar_locked(self) -> None:
        ahora = time.time()
        for rid in [r for r, x in self._items.items() if ahora - x.creado > self.ttl_s]:
            self._borrar_locked(rid)
        while len(self._items) > self.maximo:
            self._borrar_locked(next(iter(self._items)))  # el más antiguo

    def _borrar_locked(self, rid: str) -> None:
        res = self._items.pop(rid, None)
        if res:
            res.ruta.unlink(missing_ok=True)
