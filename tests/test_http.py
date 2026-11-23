"""Capa HTTP: validación de entradas, errores, idempotencia, aislamiento. No requiere GPU."""

import io
import threading

from conftest import imagen_bytes
from app import config

URL = "/api/v1/convolucion"


def post(c, datos=None, nombre="a.png", mime="image/png", **campos):
    campos.setdefault("filtro", "media")
    files = {"imagen": (nombre, datos if datos is not None else imagen_bytes(), mime)}
    return c.post(URL, files=files, data=campos)


# ---------------------------------------------------------------- caso feliz
def test_procesa_y_devuelve_tiempos_y_resultado(cliente):
    r = post(cliente, mascara="9", hilos="128")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["estado"] == "ok" and j["tiempos"]["unidad"] == "ms"
    assert j["tiempos"]["servidor_ms"] > 0 and j["tiempos"]["cola_ms"] is not None
    assert (j["entrada"]["ancho"], j["entrada"]["alto"]) == (40, 30)
    assert j["bloques"] == (40 * 30 + 127) // 128
    img = cliente.get(j["salida"]["url"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    desc = cliente.get(j["salida"]["url_descarga"])
    assert "attachment" in desc.headers["content-disposition"]
    assert cliente.get(j["salida"]["url_meta"]).json()["id_resultado"] == j["id_resultado"]


def test_formatos_de_salida(cliente):
    for fmt, mime in (("png", "image/png"), ("jpeg", "image/jpeg")):
        j = post(cliente, formato=fmt).json()
        assert cliente.get(j["salida"]["url"]).headers["content-type"] == mime
    j = post(cliente, formato="raw32").json()
    assert j["salida"]["bytes"] == 40 * 30 * 4


# ----------------------------------------------------------- entradas inválidas
def test_archivo_invalido(cliente):
    r = post(cliente, datos=b"esto no es una imagen")
    assert r.status_code == 400 and r.json()["error"]["codigo"] == "archivo_invalido"


def test_archivo_vacio(cliente):
    assert post(cliente, datos=b"").status_code == 400


def test_formato_no_soportado(cliente):
    r = post(cliente, datos=imagen_bytes(formato="GIF"), nombre="a.gif", mime="image/gif")
    assert r.status_code == 415 and r.json()["error"]["codigo"] == "formato_no_soportado"


def test_archivo_demasiado_grande_por_bytes(cliente, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 1000)
    r = post(cliente, datos=imagen_bytes(200, 200))
    assert r.status_code == 413 and r.json()["error"]["codigo"] == "archivo_demasiado_grande"


def test_imagen_demasiado_grande_por_pixeles(cliente, monkeypatch):
    monkeypatch.setattr(config, "MAX_PIXELES", 1000)
    r = post(cliente, datos=imagen_bytes(100, 100))
    assert r.status_code == 413 and r.json()["error"]["codigo"] == "imagen_demasiado_grande"


def test_imagen_demasiado_pequena(cliente):
    r = post(cliente, datos=imagen_bytes(2, 2))
    assert r.status_code == 422 and r.json()["error"]["codigo"] == "dimensiones_invalidas"


def test_filtro_desconocido(cliente):
    r = post(cliente, filtro="gaussiano")
    assert r.status_code == 422 and "desconocido" in r.json()["error"]["mensaje"]


def test_filtro_ausente(cliente):
    r = cliente.post(URL, files={"imagen": ("a.png", imagen_bytes(), "image/png")})
    assert r.status_code == 422


def test_mascara_no_permitida(cliente):
    for m in ("5", "4", "0", "-3", "abc", "100"):
        r = post(cliente, mascara=m)
        assert r.status_code == 422, m
    assert post(cliente, filtro="nitidez", mascara="9").status_code == 422
    assert post(cliente, filtro="laplaciano", mascara="21").status_code == 422
    assert post(cliente, filtro="nitidez", mascara="3").status_code == 200


def test_hilos_y_formato_invalidos(cliente):
    assert post(cliente, hilos="300").status_code == 422
    assert post(cliente, hilos="2000").status_code == 422
    assert post(cliente, formato="gif").status_code == 422


def test_falta_la_imagen(cliente):
    assert cliente.post(URL, data={"filtro": "media"}).status_code == 422


def test_id_solicitud_invalido(cliente):
    assert post(cliente, id_solicitud="../../etc").status_code == 422


def test_resultado_inexistente_o_malformado(cliente):
    assert cliente.get("/api/v1/resultados/" + "0" * 32).status_code == 404
    assert cliente.get("/api/v1/resultados/..%2f..%2fetc%2fpasswd").status_code == 404


# ------------------------------------------------------------ GPU / cola
def test_sin_gpu_da_diagnostico_claro_y_no_usa_cpu(cliente_sin_gpu):
    r = post(cliente_sin_gpu)
    assert r.status_code == 503
    e = r.json()["error"]
    assert e["codigo"] == "gpu_no_disponible" and "NO procesa en CPU" in e["mensaje"]
    h = cliente_sin_gpu.get("/api/health").json()
    assert h["estado"] == "sin_gpu" and h["gpu"]["disponible"] is False and h["gpu"]["error"]


def test_entrada_invalida_se_valida_aun_sin_gpu(cliente_sin_gpu):
    assert post(cliente_sin_gpu, datos=b"basura").status_code == 400
    assert post(cliente_sin_gpu, filtro="x").status_code == 422


def test_cola_llena_devuelve_429(cliente):
    cliente.trabajador.cola_llena = True
    r = post(cliente)
    assert r.status_code == 429 and r.headers["retry-after"] and r.json()["error"]["codigo"] == "cola_llena"


def test_health(cliente):
    h = cliente.get("/api/health").json()
    assert h["estado"] == "ok" and h["limites"]["mascaras_media"] == [3, 9, 21]


# ------------------------------------------------- idempotencia y aislamiento
def test_id_solicitud_duplicado_no_reprocesa(cliente):
    datos = imagen_bytes()
    a = post(cliente, datos=datos, id_solicitud="abcdefgh1234")
    b = post(cliente, datos=datos, id_solicitud="abcdefgh1234")
    assert a.status_code == b.status_code == 200
    assert b.headers["x-solicitud-duplicada"] == "true"
    assert a.json()["id_resultado"] == b.json()["id_resultado"]
    assert cliente.trabajador.llamadas == 1


def test_id_solicitud_reutilizado_con_otra_imagen_es_conflicto(cliente):
    post(cliente, id_solicitud="abcdefgh1234")
    r = post(cliente, datos=imagen_bytes(semilla=5), id_solicitud="abcdefgh1234")
    assert r.status_code == 409


def test_solicitud_fallida_permite_reintento(cliente):
    assert post(cliente, datos=b"x", id_solicitud="reintento-001").status_code == 400
    assert post(cliente, id_solicitud="reintento-001").status_code == 200


def test_aislamiento_cada_solicitud_recibe_su_resultado(cliente):
    """Dos imágenes y filtros distintos, enviadas en paralelo: cada ID devuelve lo suyo."""
    img_a, img_b = imagen_bytes(40, 30, semilla=1), imagen_bytes(50, 20, semilla=2)
    salida = {}

    def enviar(clave, datos, filtro):
        salida[clave] = post(cliente, datos=datos, filtro=filtro, formato="raw32", id_solicitud=f"cliente-{clave}-xx").json()

    hs = [threading.Thread(target=enviar, args=("a", img_a, "media")),
          threading.Thread(target=enviar, args=("b", img_b, "laplaciano"))]
    [h.start() for h in hs]
    [h.join() for h in hs]
    a, b = salida["a"], salida["b"]
    assert a["id_resultado"] != b["id_resultado"] and a["id_solicitud"] != b["id_solicitud"]
    assert (a["entrada"]["ancho"], a["filtro"]) == (40, "media")
    assert (b["entrada"]["ancho"], b["filtro"]) == (50, "laplaciano")
    assert len(cliente.get(a["salida"]["url"]).content) == 40 * 30 * 4
    assert len(cliente.get(b["salida"]["url"]).content) == 50 * 20 * 4


def test_resultados_expirados_se_eliminan(cliente, monkeypatch):
    j = post(cliente).json()
    cliente.app.state.almacen.ttl_s = -1
    assert cliente.get(j["salida"]["url"]).status_code == 404
    assert not list(config.DIRECTORIO_RESULTADOS.glob("*.png"))


def test_tope_de_resultados_en_disco(cliente):
    cliente.app.state.almacen.maximo = 2
    ids = [post(cliente, id_solicitud=f"solicitud-{i:03d}").json() for i in range(4)]
    assert len(list(config.DIRECTORIO_RESULTADOS.glob("*.png"))) == 2
    assert cliente.get(ids[0]["salida"]["url"]).status_code == 404
    assert cliente.get(ids[3]["salida"]["url"]).status_code == 200


def test_frontend_servido(cliente):
    r = cliente.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
