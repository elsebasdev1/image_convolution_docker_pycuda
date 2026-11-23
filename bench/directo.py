"""PyCUDA directo (sin HTTP). Se ejecuta igual en el anfitrión y dentro del contenedor.

    uv run python -m bench.directo --modo directo_host
    docker run --rm --gpus all -v $PWD/datos:/app/datos -v $PWD/resultados:/app/resultados \\
        convolucion-gpu:1.0 python -m bench.directo --modo directo_contenedor
"""

import argparse
import sys
from time import perf_counter

from app import imagen
from app.gpu.convolucion import ErrorGPU, MotorGPU

from . import comun


def ejecutar(motor, datos: bytes, filtro, mascara, hilos):
    """Mismo trabajo que el servicio: decodificar -> GPU -> codificar PNG. Devuelve campos de la fila."""
    t0 = perf_counter()
    img = imagen.decodificar(datos)
    t1 = perf_counter()
    r = motor.procesar(img.pixeles, img.ancho, img.alto, filtro, mascara, hilos)
    t2 = perf_counter()
    cuerpo, _ = imagen.codificar(r.pixeles, r.ancho, r.alto, "png")
    t3 = perf_counter()
    t = r.tiempos
    total = (t3 - t0) * 1000.0
    return dict(
        bloques=r.bloques, h2d_ms=t.h2d_ms, kernel_ms=t.kernel_ms, auxiliares_gpu_ms=t.auxiliares_ms,
        d2h_ms=t.d2h_ms, gpu_total_ms=t.gpu_total_ms, decodificacion_ms=(t1 - t0) * 1000.0,
        cola_ms=0.0, codificacion_ms=(t3 - t2) * 1000.0, servidor_ms=total, cliente_total_ms=total,
        sha256_entrada=img.sha256, sha256_salida=imagen.sha256(cuerpo),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", default="directo_host", choices=["directo_host", "directo_contenedor"])
    ap.add_argument("--experimentos", default="8.1,8.2,8.3")
    ap.add_argument("--reps", type=int, default=comun.REPS_DEFECTO)
    ap.add_argument("--salida", default=str(comun.DIR_RESULTADOS / "mediciones.csv"))
    a = ap.parse_args(argv)

    env = comun.guardar_entorno(a.modo)
    motor = MotorGPU()
    try:
        t0 = perf_counter()
        info = motor.iniciar()
        print(f"Contexto + compilación: {1000*(perf_counter()-t0):.0f} ms | {info['nombre']}")
    except ErrorGPU as e:
        print(f"[{e.codigo}] {e}", file=sys.stderr)
        return 2

    f, w = comun.abrir_csv(__import__("pathlib").Path(a.salida))
    primera = True
    try:
        for exp in a.experimentos.split(","):
            for filtro, mascara, lado in comun.configuraciones(exp):
                datos = comun.ruta_imagen(lado).read_bytes()
                ctx = dict(modo_ejecucion=a.modo, experimento=exp, filtro=filtro, resolucion=f"{lado}x{lado}",
                           mascara=mascara, hilos_por_bloque=comun.HILOS, concurrencia=1, entorno_id=a.modo)
                # fase: frío (primera ejecución del proceso), calentamiento (1) y medición (reps)
                secuencia = [("calentamiento", 0)] + [("medicion", i) for i in range(1, a.reps + 1)]
                if primera:
                    secuencia = [("frio", 0)] + secuencia
                    primera = False
                for fase, rep in secuencia:
                    try:
                        campos = ejecutar(motor, datos, filtro, mascara, comun.HILOS)
                        w.writerow(comun.fila_base(**ctx, fase=fase, repeticion=rep, estado="ok", **campos))
                    except Exception as e:
                        w.writerow(comun.fila_base(**ctx, fase=fase, repeticion=rep, estado="fallo", error=str(e)))
                        comun.registrar_error(a.modo, ctx, f"{type(e).__name__}: {e}")
                    f.flush()
                print(f"{exp} {filtro} {mascara}x{mascara} {lado}² -> ok")
    finally:
        f.close()
        motor.cerrar()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
