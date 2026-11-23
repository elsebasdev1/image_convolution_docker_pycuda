"""Definiciones compartidas por los scripts de medición: columnas CSV, experimentos, entorno."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DIR_IMAGENES = Path(os.environ.get("DIR_IMAGENES", RAIZ / "datos" / "imagenes"))
DIR_RESULTADOS = Path(os.environ.get("DIR_RESULTADOS", RAIZ / "resultados"))

COLUMNAS = (
    "id_solicitud", "modo_ejecucion", "experimento", "fase", "filtro", "resolucion", "mascara",
    "hilos_por_bloque", "bloques", "repeticion", "concurrencia", "ronda",
    "h2d_ms", "kernel_ms", "auxiliares_gpu_ms", "d2h_ms", "gpu_total_ms",
    "decodificacion_ms", "cola_ms", "codificacion_ms", "servidor_ms", "cliente_post_ms",
    "descarga_ms", "cliente_total_ms", "unidad_tiempo", "estado", "http", "error",
    "sha256_entrada", "sha256_salida", "entorno_id", "marca_tiempo_utc",
)

TAMANOS = (2000, 4000, 6000)
FILTROS_8_3 = (("media", 3), ("media", 9), ("media", 21), ("nitidez", 3), ("laplaciano", 3))
HILOS = 256
REPS_DEFECTO = 10


def ruta_imagen(lado: int) -> Path:
    return DIR_IMAGENES / f"extra2_{lado}.jpeg"


def configuraciones(exp: str):
    """Rinde (filtro, mascara, lado) de cada experimento de la sección 8 (un factor a la vez)."""
    if exp == "8.1":
        yield ("media", 9, 4000)
    elif exp == "8.2":
        for lado in TAMANOS:
            yield ("media", 9, lado)
    elif exp == "8.3":
        for f, m in FILTROS_8_3:
            yield (f, m, 4000)
    else:
        raise ValueError(exp)


def abrir_csv(ruta: Path):
    """Abre en modo append y escribe la cabecera solo si el archivo es nuevo."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not ruta.exists() or ruta.stat().st_size == 0
    f = ruta.open("a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=COLUMNAS)
    if nuevo:
        w.writeheader()
    return f, w


def fila_base(**kw) -> dict:
    fila = {c: "" for c in COLUMNAS}
    fila.update(unidad_tiempo="ms", id_solicitud=uuid.uuid4().hex,
                marca_tiempo_utc=datetime.now(timezone.utc).isoformat(timespec="milliseconds"))
    fila.update(kw)
    return fila


def registrar_error(modo: str, contexto: dict, detalle: str) -> None:
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    with (DIR_RESULTADOS / "errores.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "modo": modo,
                            "contexto": contexto, "detalle": detalle}, ensure_ascii=False) + "\n")


def _cmd(*args) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def entorno(etiqueta: str) -> dict:
    """Captura del entorno (sección 6 de la guía). `etiqueta` identifica el entorno en el CSV."""
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "")
    ram_kb = next((int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemTotal")), 0)
    try:
        import pycuda, pycuda.driver as cuda  # noqa: E401

        pyc = pycuda.VERSION_TEXT
        cuda.init()
        d = cuda.Device(0)
        gpu = {"nombre": d.name(), "memoria_mb": d.total_memory() // 2**20,
               "compute_capability": "%d.%d" % d.compute_capability(),
               "cuda_driver_api": cuda.get_driver_version(), "pycuda_compilado_con": list(cuda.get_version())}
    except Exception as e:
        pyc, gpu = None, {"error": str(e)}
    return {
        "etiqueta": etiqueta,
        "cpu": cpu, "nucleos_logicos": os.cpu_count(), "ram_gb": round(ram_kb / 2**20, 1),
        "gpu": gpu, "controlador_nvidia": _cmd("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")
        or (open("/proc/driver/nvidia/version").read().splitlines()[0] if Path("/proc/driver/nvidia/version").exists() else ""),
        "sistema": platform.platform(), "python": sys.version.split()[0], "pycuda": pyc,
        "nvcc": next((l for l in _cmd("nvcc", "--version").splitlines() if "release" in l), ""),
        "docker": _cmd("docker", "--version"),
        "en_contenedor": Path("/.dockerenv").exists(),
        "commit": os.environ.get("GIT_COMMIT") or _cmd("git", "-C", str(RAIZ), "rev-parse", "HEAD"),
        "marca_tiempo_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def guardar_entorno(etiqueta: str) -> dict:
    e = entorno(etiqueta)
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    ruta = DIR_RESULTADOS / f"entorno_{etiqueta}.json"
    ruta.write_text(json.dumps(e, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return e
