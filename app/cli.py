"""Ejecución directa (sin HTTP) del mismo motor que usa el servicio.

    uv run python -m app.cli imagen.jpeg --filtro media --mascara 9 --salida salida.png
    uv run python -m app.cli --info
"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

from . import config, imagen
from .gpu.convolucion import ErrorGPU, MotorGPU


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("imagen", nargs="?", type=Path)
    ap.add_argument("--filtro", choices=config.FILTROS, default="media")
    ap.add_argument("--mascara", type=int, default=3, choices=config.MASCARAS_MEDIA)
    ap.add_argument("--hilos", type=int, default=config.HILOS_DEFECTO, choices=config.HILOS_PERMITIDOS)
    ap.add_argument("--salida", type=Path)
    ap.add_argument("--reps", type=int, default=1, help="repeticiones (la primera cuenta como calentamiento si >1)")
    ap.add_argument("--info", action="store_true", help="solo mostrar la GPU y salir")
    a = ap.parse_args(argv)

    motor = MotorGPU(config.GPU_ORDINAL)
    try:
        info = motor.iniciar()  # crea el contexto CUDA y compila los kernels
    except ErrorGPU as e:
        print(f"[{e.codigo}] {e}", file=sys.stderr)
        return 2
    try:
        print(json.dumps(info, indent=2, ensure_ascii=False))
        if a.info or a.imagen is None:
            return 0
        datos = a.imagen.read_bytes()
        for i in range(a.reps):
            t0 = perf_counter()
            img = imagen.decodificar(datos)
            t1 = perf_counter()
            r = motor.procesar(img.pixeles, img.ancho, img.alto, a.filtro, a.mascara, a.hilos)
            t2 = perf_counter()
            t = r.tiempos
            print(
                f"rep {i}: decod {1000*(t1-t0):.2f} ms | H2D {t.h2d_ms:.3f} | kernel {t.kernel_ms:.3f} | "
                f"aux {t.auxiliares_ms:.3f} | D2H {t.d2h_ms:.3f} | GPU total {t.gpu_total_ms:.3f} | "
                f"sección {t.seccion_ms:.3f} ms ({img.ancho}x{img.alto}, {r.bloques} bloques x {r.hilos} hilos)"
            )
        if a.salida:
            cuerpo, _ = imagen.codificar(r.pixeles, r.ancho, r.alto, "jpeg" if a.salida.suffix.lower() in (".jpg", ".jpeg") else "png")
            a.salida.write_bytes(cuerpo)
            print(f"guardado: {a.salida} (sha256 {imagen.sha256(cuerpo)[:16]}…)")
        return 0
    except ErrorGPU as e:
        print(f"[{e.codigo}] {e}", file=sys.stderr)
        return 1
    finally:
        motor.cerrar()


if __name__ == "__main__":
    raise SystemExit(main())
