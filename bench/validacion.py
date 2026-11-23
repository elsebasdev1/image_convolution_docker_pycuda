"""Validación funcional y numérica (sección 7 de la guía) contra el servicio en marcha.

    uv run python -m bench.validacion --url http://localhost:8000

Genera resultados/validacion.csv y las imágenes de docs/evidencia/.
Compara el motor PyCUDA ejecutado directamente (anfitrión) con el servicio (contenedor).
"""

import argparse
import csv
import http.client
import io
import json
import math
import os
import threading
from array import array
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

from app import imagen
from app.gpu.convolucion import MotorGPU
from app.gpu.kernels import crear_mascaras, seleccionar_mascara
from app.gpu.referencia_cpu import a_uint8, calcular_errores, convolucion_cpu

from . import comun
from .cliente import multipart

EVID = comun.RAIZ / "docs" / "evidencia"
FILTROS = (("media", 3), ("media", 9), ("media", 21), ("nitidez", 3), ("laplaciano", 3))
filas: list[dict] = []


def anotar(prueba, detalle, resultado, **extra):
    filas.append({"prueba": prueba, "detalle": detalle, "resultado": resultado, **extra})
    print(f"[{resultado}] {prueba}: {detalle} {extra if extra else ''}")


def http_post(url, datos, campos, nombre="x.jpeg", mime="image/jpeg"):
    u = urlparse(url)
    cuerpo, ctype = multipart(campos, nombre, datos)
    con = http.client.HTTPConnection(u.hostname, u.port, timeout=300)
    try:
        con.request("POST", "/api/v1/convolucion", body=cuerpo, headers={"Content-Type": ctype})
        r = con.getresponse()
        return r.status, json.loads(r.read() or b"{}")
    finally:
        con.close()


def http_get(url, ruta):
    u = urlparse(url)
    con = http.client.HTTPConnection(u.hostname, u.port, timeout=300)
    try:
        con.request("GET", ruta)
        return con.getresponse().read()
    finally:
        con.close()


def png_bytes(w, h):
    b = io.BytesIO()
    Image.new("L", (w, h), 128).save(b, "PNG")
    return b.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    a = ap.parse_args()
    (EVID / "completas").mkdir(parents=True, exist_ok=True)
    motor = MotorGPU()
    motor.iniciar()
    datos4k = comun.ruta_imagen(4000).read_bytes()
    img4k = imagen.decodificar(datos4k)
    masc = crear_mascaras()
    try:
        # ---- 1. Matriz pequeña y límites (motor directo) --------------------
        r = motor.procesar(bytes(range(1, 10)), 3, 3, "laplaciano", 3, 4, devolver_flotantes=True)
        esperado = [2, 1, -4, -3, 0, -7, -16, -11, -22]
        anotar("matriz_manual_3x3", "laplaciano, 4 hilos/bloque", "OK" if list(r.flotantes) == esperado else "FALLO",
               obtenido=list(r.flotantes))
        w, h, tpb = 1237, 911, 256
        pix = bytes((i * 31 + (i // w) * 17) % 256 for i in range(w * h))
        for filtro, m in (("media", 9), ("nitidez", 3), ("laplaciano", 3)):
            rg = motor.procesar(pix, w, h, filtro, m, tpb, devolver_flotantes=True)
            nan = sum(1 for v in rg.flotantes if math.isnan(v))
            clave, k, desp = seleccionar_mascara(filtro, m)
            cpu = convolucion_cpu([float(b) for b in pix], masc[clave], k, w, h)
            emax, emed = calcular_errores(cpu, rg.flotantes)
            dif = max(abs(x - y) for x, y in zip(rg.pixeles, a_uint8(cpu, desp)))
            anotar("limites_no_multiplo", f"{w}x{h}={w*h} px ({(w*h) % tpb} sobrantes de {tpb}), {filtro} {m}x{m}",
                   "OK" if nan == 0 and emax <= 1e-2 else "FALLO", pixeles_nan=nan, err_max_cpu_vs_gpu=emax,
                   err_medio_cpu_vs_gpu=emed, dif_max_bytes_tras_recorte=dif, bloques=rg.bloques)

        # ---- 2. Equivalencia directa (anfitrión) vs servicio (contenedor) ---
        for filtro, m in FILTROS:
            d = motor.procesar(img4k.pixeles, img4k.ancho, img4k.alto, filtro, m, 256, devolver_flotantes=True)
            st, j = http_post(a.url, datos4k, {"filtro": filtro, "mascara": m, "formato": "raw32"})
            s = array("f")
            s.frombytes(http_get(a.url, j["salida"]["url"]))
            emax, emed = calcular_errores(d.flotantes, s)
            nan = sum(1 for v in s if math.isnan(v))
            st2, j2 = http_post(a.url, datos4k, {"filtro": filtro, "mascara": m, "formato": "png"})
            cuerpo_dir, _ = imagen.codificar(d.pixeles, d.ancho, d.alto, "png")
            igual = imagen.sha256(cuerpo_dir) == j2["salida"]["sha256"]
            anotar("equivalencia_directa_servicio", f"4000x4000 {filtro} {m}x{m}", "OK" if emax == 0 and igual else "FALLO",
                   err_max=emax, err_medio=emed, pixeles_nan=nan, png_identico=igual)

        # ---- 3. Resultado visual -------------------------------------------
        original = Image.open(io.BytesIO(datos4k)).convert("L")
        original.save(EVID / "completas" / "original_4000.png")
        for filtro, m in (("media", 9), ("nitidez", 3), ("laplaciano", 3)):
            st, j = http_post(a.url, datos4k, {"filtro": filtro, "mascara": m, "formato": "png"})
            (EVID / "completas" / f"{filtro}{m if filtro == 'media' else ''}_4000.png").write_bytes(http_get(a.url, j["salida"]["url"]))
        anotar("resultado_visual", "original + media 9x9, nitidez, laplaciano guardados en docs/evidencia", "OK")

        # ---- 4. Entradas inválidas -----------------------------------------
        base = {"filtro": "media", "mascara": 3}
        casos = [
            ("archivo_invalido", b"esto no es una imagen", base, 400),
            ("archivo_vacio", b"", base, 400),
            ("archivo_demasiado_grande", os.urandom(40 * 2**20), base, 413),
            ("imagen_demasiado_grande_lado", png_bytes(9000, 100), base, 413),
            ("filtro_desconocido", png_bytes(50, 50), {"filtro": "gaussiano"}, 422),
            ("mascara_no_permitida_media_5", png_bytes(50, 50), {"filtro": "media", "mascara": 5}, 422),
            ("mascara_no_permitida_nitidez_9", png_bytes(50, 50), {"filtro": "nitidez", "mascara": 9}, 422),
            ("hilos_no_permitidos", png_bytes(50, 50), {**base, "hilos": 300}, 422),
        ]
        for nombre, datos, campos, esperado_http in casos:
            try:
                st, j = http_post(a.url, datos, campos)
            except (ConnectionError, OSError) as e:  # el servidor puede cortar la subida de 40 MB
                st, j = 413, {"error": {"codigo": "conexion_cortada", "mensaje": str(e)}}
            e = j.get("error", {})
            anotar("entrada_invalida", nombre, "OK" if st == esperado_http else "FALLO", http=st,
                   codigo=e.get("codigo"), mensaje=e.get("mensaje"))

        # ---- 5. Aislamiento de solicitudes ----------------------------------
        datos2k = comun.ruta_imagen(2000).read_bytes()
        img2k = imagen.decodificar(datos2k)
        pares = [("A", datos4k, img4k, "media", 9), ("B", datos2k, img2k, "laplaciano", 3),
                 ("C", datos2k, img2k, "media", 21), ("D", datos4k, img4k, "nitidez", 3)]
        salida = {}

        def enviar(clave, datos, filtro, m):
            st, j = http_post(a.url, datos, {"filtro": filtro, "mascara": m, "id_solicitud": f"aislamiento-{clave}-0001"})
            salida[clave] = (st, j)

        hs = [threading.Thread(target=enviar, args=(c, d, f, m)) for c, d, _, f, m in pares]
        [t.start() for t in hs]
        [t.join() for t in hs]
        ids = {j["id_resultado"] for _, j in salida.values()}
        for c, d, im, f, m in pares:
            st, j = salida[c]
            ref = motor.procesar(im.pixeles, im.ancho, im.alto, f, m, 256)
            cuerpo_ref, _ = imagen.codificar(ref.pixeles, ref.ancho, ref.alto, "png")
            ok = st == 200 and j["id_solicitud"] == f"aislamiento-{c}-0001" and j["salida"]["sha256"] == imagen.sha256(cuerpo_ref) \
                and j["entrada"]["sha256"] == im.sha256 and j["filtro"] == f
            anotar("aislamiento", f"solicitud {c}: {im.ancho}x{im.alto} {f} {m}x{m}", "OK" if ok else "FALLO",
                   id_resultado=j.get("id_resultado"), sha256_salida=j.get("salida", {}).get("sha256", "")[:16])
        anotar("aislamiento", "4 solicitudes simultáneas -> 4 id_resultado distintos", "OK" if len(ids) == 4 else "FALLO")

        # ---- 6. Idempotencia ------------------------------------------------
        campos = {"filtro": "media", "mascara": 3, "id_solicitud": "idempotencia-0001"}
        _, j1 = http_post(a.url, datos2k, campos)
        _, j2 = http_post(a.url, datos2k, campos)
        anotar("solicitud_duplicada", "mismo id_solicitud dos veces", "OK" if j1["id_resultado"] == j2["id_resultado"] else "FALLO",
               id_resultado=j1["id_resultado"])
    finally:
        motor.cerrar()

    cols = ["prueba", "detalle", "resultado", "err_max", "err_medio", "err_max_cpu_vs_gpu", "err_medio_cpu_vs_gpu",
            "pixeles_nan", "dif_max_bytes_tras_recorte", "bloques", "png_identico", "http", "codigo", "mensaje",
            "id_resultado", "sha256_salida", "obtenido"]
    with (comun.DIR_RESULTADOS / "validacion.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)
    fallos = [x for x in filas if x["resultado"] != "OK"]
    print(f"\n{len(filas) - len(fallos)}/{len(filas)} pruebas OK")
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
