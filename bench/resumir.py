"""Resume mediciones.csv (solo fase 'medicion' y estado 'ok'): mediana, desviación, IQR, min, max."""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from . import comun

METRICAS = ("h2d_ms", "kernel_ms", "auxiliares_gpu_ms", "d2h_ms", "gpu_total_ms", "decodificacion_ms",
            "cola_ms", "codificacion_ms", "servidor_ms", "cliente_total_ms")


def main(entrada=None, salida=None) -> int:
    entrada = Path(entrada or comun.DIR_RESULTADOS / "mediciones.csv")
    salida = Path(salida or comun.DIR_RESULTADOS / "resumen.csv")
    grupos = defaultdict(list)
    fallos = defaultdict(int)
    with entrada.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["fase"] != "medicion":
                continue
            clave = (r["modo_ejecucion"], r["experimento"], r["filtro"], r["resolucion"], r["mascara"],
                     r["hilos_por_bloque"], r["concurrencia"])
            if r["estado"] == "ok":
                grupos[clave].append(r)
            else:
                fallos[clave] += 1
    cab = ["modo_ejecucion", "experimento", "filtro", "resolucion", "mascara", "hilos_por_bloque", "concurrencia",
           "n_ok", "n_no_ok", "unidad_tiempo"]
    for m in METRICAS:
        cab += [f"{m}_mediana", f"{m}_desv", f"{m}_q1", f"{m}_q3", f"{m}_min", f"{m}_max"]
    with salida.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cab)
        for clave in sorted(set(grupos) | set(fallos)):
            filas = grupos.get(clave, [])
            linea = [*clave, len(filas), fallos.get(clave, 0), "ms"]
            for m in METRICAS:
                v = [float(r[m]) for r in filas if r[m] != ""]
                if len(v) >= 2:
                    q = statistics.quantiles(v, n=4, method="inclusive")
                    linea += [f"{statistics.median(v):.4f}", f"{statistics.stdev(v):.4f}", f"{q[0]:.4f}",
                              f"{q[2]:.4f}", f"{min(v):.4f}", f"{max(v):.4f}"]
                else:
                    linea += [""] * 6
            w.writerow(linea)
    print(f"resumen -> {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
