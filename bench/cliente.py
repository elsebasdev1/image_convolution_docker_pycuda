"""Cliente de medición contra el servicio HTTP (experimentos 8.1 -servicio-, 8.2, 8.3 y 8.4).

    uv run python -m bench.cliente --url http://localhost:8000 --modo servicio_contenedor
"""

import argparse
import csv
import http.client
import json
import threading
import uuid
from pathlib import Path
from time import perf_counter
from urllib.parse import urlparse

from . import comun


def multipart(campos: dict, nombre: str, datos: bytes) -> tuple[bytes, str]:
    b = uuid.uuid4().hex
    partes = [f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in campos.items()]
    partes.append(f'--{b}\r\nContent-Disposition: form-data; name="imagen"; filename="{nombre}"\r\n'
                  f'Content-Type: image/jpeg\r\n\r\n'.encode() + datos + b"\r\n")
    partes.append(f"--{b}--\r\n".encode())
    return b"".join(partes), f"multipart/form-data; boundary={b}"


def solicitar(url: str, datos: bytes, nombre: str, filtro: str, mascara: int, hilos: int, descargar=True) -> dict:
    """Una solicitud completa: POST + descarga del resultado. Devuelve campos de la fila."""
    u = urlparse(url)
    id_sol = uuid.uuid4().hex
    campos = {"filtro": filtro, "hilos": hilos, "formato": "png", "id_solicitud": id_sol}
    if filtro == "media":
        campos["mascara"] = mascara
    cuerpo, ctype = multipart(campos, nombre, datos)
    out = {"id_solicitud": id_sol}
    t0 = perf_counter()
    con = http.client.HTTPConnection(u.hostname, u.port, timeout=300)
    try:
        con.request("POST", "/api/v1/convolucion", body=cuerpo, headers={"Content-Type": ctype})
        r = con.getresponse()
        texto = r.read()
        t1 = perf_counter()
        out["http"] = r.status
        out["cliente_post_ms"] = (t1 - t0) * 1000.0
        j = json.loads(texto)
        if r.status != 200:
            out.update(estado="rechazo" if r.status in (429, 503, 504) else "fallo",
                       error=f'{j.get("error", {}).get("codigo")}: {j.get("error", {}).get("mensaje")}',
                       cliente_total_ms=out["cliente_post_ms"])
            return out
        if descargar:
            con.request("GET", j["salida"]["url"])
            rr = con.getresponse()
            rr.read()
            out["descarga_ms"] = (perf_counter() - t1) * 1000.0
        out["cliente_total_ms"] = (perf_counter() - t0) * 1000.0
        t = j["tiempos"]
        out.update(
            estado="ok", bloques=j["bloques"], h2d_ms=t["h2d_ms"], kernel_ms=t["kernel_ms"],
            auxiliares_gpu_ms=t["auxiliares_gpu_ms"], d2h_ms=t["d2h_ms"], gpu_total_ms=t["gpu_total_ms"],
            decodificacion_ms=t["decodificacion_ms"], cola_ms=t["cola_ms"], codificacion_ms=t["codificacion_ms"],
            servidor_ms=t["servidor_ms"], sha256_entrada=j["entrada"]["sha256"], sha256_salida=j["salida"]["sha256"],
        )
    except Exception as e:  # red, tiempo de espera, JSON inválido…
        out.update(estado="fallo", error=f"{type(e).__name__}: {e}", cliente_total_ms=(perf_counter() - t0) * 1000.0)
    finally:
        con.close()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--modo", default="servicio_contenedor", choices=["servicio_contenedor", "servicio_host"])
    ap.add_argument("--experimentos", default="8.1,8.2,8.3,8.4")
    ap.add_argument("--reps", type=int, default=comun.REPS_DEFECTO)
    ap.add_argument("--concurrencias", default="1,2", help="niveles de 8.4, p. ej. 1,2 o 1,2,4")
    ap.add_argument("--ubicacion", default="localhost (loopback, mismo equipo)")
    ap.add_argument("--salida", default=str(comun.DIR_RESULTADOS / "mediciones.csv"))
    a = ap.parse_args(argv)

    entorno_id = a.modo
    env = comun.guardar_entorno("cliente_" + a.modo)
    env["ubicacion_cliente"] = a.ubicacion
    (comun.DIR_RESULTADOS / f"entorno_cliente_{a.modo}.json").write_text(json.dumps(env, indent=2, ensure_ascii=False) + "\n")
    f, w = comun.abrir_csv(Path(a.salida))
    lock = threading.Lock()

    def anotar(ctx, fase, rep, campos, concurrencia=1, ronda=""):
        with lock:
            w.writerow(comun.fila_base(**{**ctx, **campos}, fase=fase, repeticion=rep, concurrencia=concurrencia,
                                       ronda=ronda, entorno_id=entorno_id))
            f.flush()
            if campos.get("estado") != "ok":
                comun.registrar_error(a.modo, ctx, campos.get("error", ""))

    primera = True
    try:
        for exp in [e for e in a.experimentos.split(",") if e in ("8.1", "8.2", "8.3")]:
            for filtro, mascara, lado in comun.configuraciones(exp):
                datos = comun.ruta_imagen(lado).read_bytes()
                ctx = dict(modo_ejecucion=a.modo, experimento=exp, filtro=filtro, resolucion=f"{lado}x{lado}",
                           mascara=mascara, hilos_por_bloque=comun.HILOS)
                secuencia = [("calentamiento", 0)] + [("medicion", i) for i in range(1, a.reps + 1)]
                if primera:
                    secuencia = [("frio", 0)] + secuencia  # primera solicitud tras arrancar el servicio
                    primera = False
                for fase, rep in secuencia:
                    anotar(ctx, fase, rep, solicitar(a.url, datos, f"extra2_{lado}.jpeg", filtro, mascara, comun.HILOS))
                print(f"{exp} {filtro} {mascara}x{mascara} {lado}² -> hecho")

        if "8.4" in a.experimentos.split(","):
            datos = comun.ruta_imagen(4000).read_bytes()
            ctx = dict(modo_ejecucion=a.modo, experimento="8.4", filtro="media", resolucion="4000x4000",
                       mascara=9, hilos_por_bloque=comun.HILOS)
            rondas_csv = comun.DIR_RESULTADOS / "concurrencia_rondas.csv"
            nuevo = not rondas_csv.exists()
            with rondas_csv.open("a", newline="", encoding="utf-8") as rf:
                rw = csv.writer(rf)
                if nuevo:
                    rw.writerow(["modo_ejecucion", "concurrencia", "ronda", "exitosas", "rechazos", "fallos",
                                 "intervalo_s", "solicitudes_exitosas_por_s", "entorno_id"])
                solicitar(a.url, datos, "extra2_4000.jpeg", "media", 9, comun.HILOS)  # calentamiento
                for n in (int(x) for x in a.concurrencias.split(",")):
                    for ronda in range(1, a.reps + 1):  # un factor a la vez: solo cambia n
                        barrera, res = threading.Barrier(n + 1), [None] * n

                        def cliente(i):
                            barrera.wait()
                            res[i] = solicitar(a.url, datos, "extra2_4000.jpeg", "media", 9, comun.HILOS)

                        hs = [threading.Thread(target=cliente, args=(i,)) for i in range(n)]
                        [h.start() for h in hs]
                        barrera.wait()
                        t0 = perf_counter()  # inicio de la ronda: todos liberados a la vez
                        [h.join() for h in hs]
                        intervalo = perf_counter() - t0
                        ok = sum(r["estado"] == "ok" for r in res)
                        rech = sum(r["estado"] == "rechazo" for r in res)
                        rw.writerow([a.modo, n, ronda, ok, rech, n - ok - rech, f"{intervalo:.6f}",
                                     f"{ok / intervalo:.6f}", entorno_id])
                        rf.flush()
                        for i, r in enumerate(res):
                            anotar(ctx, "medicion", ronda, r, concurrencia=n, ronda=ronda)
                    print(f"8.4 concurrencia={n} -> hecho")
    finally:
        f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
