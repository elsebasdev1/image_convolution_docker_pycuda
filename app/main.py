"""Capa HTTP (FastAPI). Solo valida, encola y responde; la convolución vive en app.gpu."""

from __future__ import annotations

import asyncio
import platform
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import config, imagen
from .almacen import AlmacenResultados
from .gpu.convolucion import ErrorGPU, MemoriaGPUInsuficiente
from .gpu.trabajador import ColaLlena, TiempoColaAgotado, TrabajadorGPU
from .imagen import ErrorEntrada

RAIZ = Path(__file__).resolve().parent.parent
RE_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
RE_RESULTADO = re.compile(r"^[0-9a-f]{32}$")
EXTENSIONES = {"png": ("png", "image/png"), "jpeg": ("jpg", "image/jpeg"), "raw32": ("f32", "application/octet-stream")}


def _error(http: int, codigo: str, mensaje: str, id_solicitud: str | None = None, headers=None):
    cuerpo = {"estado": "error", "error": {"codigo": codigo, "mensaje": mensaje}}
    if id_solicitud:
        cuerpo["id_solicitud"] = id_solicitud
    return JSONResponse(cuerpo, status_code=http, headers=headers)


class MiddlewareLlegada:
    """ASGI puro: marca el instante de llegada (reloj monotónico) y rechaza cuerpos enormes
    por Content-Length antes de que Starlette los reciba y los guarde."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            scope.setdefault("state", {})["t_llegada"] = perf_counter()
            if scope["method"] == "POST":
                cl = dict(scope["headers"]).get(b"content-length")
                if cl and cl.isdigit() and int(cl) > config.MAX_UPLOAD_BYTES + 64 * 1024:
                    resp = _error(
                        413, "archivo_demasiado_grande",
                        f"El cuerpo supera el límite de {config.MAX_UPLOAD_BYTES // 2**20} MB.",
                    )
                    await resp(scope, receive, send)
                    return
        await self.app(scope, receive, send)


def create_app(trabajador=None) -> FastAPI:
    """`trabajador` solo se inyecta en pruebas; en producción se crea el real."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.trabajador = trabajador or TrabajadorGPU(
            config.GPU_ORDINAL, config.TAMANO_COLA, config.TIEMPO_MAX_COLA_S
        )
        if trabajador is None:
            await run_in_threadpool(app.state.trabajador.iniciar)
        app.state.almacen = AlmacenResultados(
            config.DIRECTORIO_RESULTADOS, config.TTL_RESULTADOS_S, config.MAX_RESULTADOS
        )
        app.state.solicitudes = {}  # id_solicitud -> (huella, futuro, instante)
        yield
        if trabajador is None:
            app.state.trabajador.detener()

    app = FastAPI(
        title="Convolución de imágenes en GPU",
        description="Filtros de media, nitidez y laplaciano con kernels CUDA propios (PyCUDA).",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(MiddlewareLlegada)

    @app.exception_handler(ErrorEntrada)
    async def _h_entrada(request: Request, exc: ErrorEntrada):
        return _error(exc.http, exc.codigo, exc.mensaje)

    @app.exception_handler(RequestValidationError)
    async def _h_validacion(request: Request, exc: RequestValidationError):
        return _error(422, "parametro_invalido", "Solicitud mal formada (¿multipart/form-data con el campo 'imagen'?).")

    # ----------------------------------------------------------------- estado
    @app.get("/api/health", tags=["estado"], summary="Estado del servicio, GPU, cola y límites")
    async def health(request: Request):
        t: TrabajadorGPU = request.app.state.trabajador
        gpu = {"disponible": t.disponible}
        if t.disponible:
            gpu.update(t.info)
        else:
            gpu["error"] = str(t.error_inicio) if t.error_inicio else "Inicializando contexto CUDA…"
        return {
            "estado": "ok" if t.disponible else "sin_gpu",
            "gpu": gpu,
            "cola": {
                "en_cola": t.en_cola,
                "capacidad": config.TAMANO_COLA,
                "procesadas": t.procesadas,
                "rechazadas": t.rechazadas,
            },
            "limites": {
                "max_upload_mb": config.MAX_UPLOAD_BYTES // 2**20,
                "max_pixeles": config.MAX_PIXELES,
                "max_lado": config.MAX_LADO,
                "formatos_entrada": sorted(config.FORMATOS_ENTRADA),
                "filtros": list(config.FILTROS),
                "mascaras_media": list(config.MASCARAS_MEDIA),
                "hilos_por_bloque": list(config.HILOS_PERMITIDOS),
                "hilos_por_defecto": config.HILOS_DEFECTO,
                "formatos_salida": list(config.FORMATOS_SALIDA),
                "ttl_resultados_s": config.TTL_RESULTADOS_S,
            },
            "versiones": _versiones(),
        }

    # ------------------------------------------------------------ procesamiento
    @app.post(
        "/api/v1/convolucion",
        tags=["procesamiento"],
        summary="Aplica un filtro de convolución en la GPU",
        responses={
            200: {"description": "Procesada. JSON con identificadores, tiempos (ms) y URL del resultado."},
            400: {"description": "archivo_invalido"},
            409: {"description": "id_solicitud reutilizado con otro contenido"},
            413: {"description": "archivo_demasiado_grande | imagen_demasiado_grande"},
            415: {"description": "formato_no_soportado"},
            422: {"description": "parametro_invalido (filtro, máscara, hilos, formato, id) | dimensiones_invalidas"},
            429: {"description": "cola_llena (cabecera Retry-After)"},
            503: {"description": "gpu_no_disponible | memoria_gpu_insuficiente"},
            504: {"description": "tiempo_cola_agotado"},
        },
    )
    async def convolucion(
        request: Request,
        imagen_archivo: UploadFile | None = File(None, alias="imagen"),
        filtro: str | None = Form(None),
        mascara: str | None = Form(None),
        hilos: str | None = Form(None),
        formato: str | None = Form(None),
        id_solicitud: str | None = Form(None),
    ):
        t_handler = perf_counter()
        recepcion_ms = (t_handler - request.scope["state"]["t_llegada"]) * 1000.0
        id_sol = id_solicitud or request.headers.get("x-id-solicitud") or uuid.uuid4().hex

        # 1. Validar parámetros (barato) antes de tocar la imagen
        if not RE_ID.match(id_sol):
            raise ErrorEntrada(422, "parametro_invalido", "id_solicitud debe cumplir [A-Za-z0-9_-]{8,64}.")
        p = _validar_parametros(filtro, mascara, hilos, formato)
        if imagen_archivo is None:
            raise ErrorEntrada(422, "parametro_invalido", "Falta el campo 'imagen' (multipart/form-data).")
        datos = await imagen_archivo.read()
        if len(datos) > config.MAX_UPLOAD_BYTES:
            raise ErrorEntrada(
                413, "archivo_demasiado_grande", f"El archivo supera {config.MAX_UPLOAD_BYTES // 2**20} MB."
            )

        # 2. Idempotencia: el mismo id_solicitud no se procesa dos veces
        huella = (imagen.sha256(datos), p["filtro"], p["mascara"], p["hilos"], p["formato"])
        registro = request.app.state.solicitudes
        _purgar_solicitudes(registro)
        previa = registro.get(id_sol)
        if previa:
            if previa[0] != huella:
                return _error(409, "id_solicitud_en_conflicto",
                              "Ese id_solicitud ya se usó con otra imagen o parámetros.", id_sol)
            try:
                payload = await asyncio.shield(previa[1])
            except Exception:
                return _error(409, "solicitud_original_fallida",
                              "La solicitud original con ese id falló; use un id nuevo.", id_sol)
            return JSONResponse(payload, headers={"X-Solicitud-Duplicada": "true", "X-Id-Solicitud": id_sol})

        futuro = asyncio.get_running_loop().create_future()
        registro[id_sol] = (huella, futuro, time.time())
        try:
            payload, headers = await _ejecutar(request, datos, p, id_sol, huella[0], t_handler, recepcion_ms)
        except BaseException as e:
            registro.pop(id_sol, None)  # un fallo no bloquea el reintento
            futuro.set_exception(e if isinstance(e, Exception) else RuntimeError("cancelada"))
            futuro.exception()  # marcar como consumida
            if isinstance(e, Exception):
                return _manejar_fallo(e, id_sol)
            raise
        futuro.set_result(payload)
        return JSONResponse(payload, headers=headers)

    # --------------------------------------------------------------- resultados
    @app.get("/api/v1/resultados/{rid}", tags=["procesamiento"], summary="Descarga/visualiza el resultado")
    async def resultado(request: Request, rid: str, descargar: bool = False):
        res = _buscar(request, rid)
        if res is None:
            return _error(404, "resultado_no_encontrado", "No existe o ya expiró (TTL).")
        return FileResponse(
            res.ruta, media_type=res.tipo_mime,
            filename=res.nombre_descarga if descargar else None,
            headers={} if descargar else {"Content-Disposition": "inline"},
        )

    @app.get("/api/v1/resultados/{rid}/meta", tags=["procesamiento"], summary="Metadatos y tiempos del resultado")
    async def resultado_meta(request: Request, rid: str):
        res = _buscar(request, rid)
        if res is None:
            return _error(404, "resultado_no_encontrado", "No existe o ya expiró (TTL).")
        return res.meta

    app.mount("/", StaticFiles(directory=RAIZ / "web", html=True), name="web")
    return app


# ------------------------------------------------------------------ auxiliares
def _versiones() -> dict:
    v = {"python": platform.python_version(), "sistema": platform.platform()}
    try:
        import pycuda

        v["pycuda"] = pycuda.VERSION_TEXT
    except Exception:
        v["pycuda"] = None
    return v


def _validar_parametros(filtro, mascara, hilos, formato) -> dict:
    def inv(msg):
        return ErrorEntrada(422, "parametro_invalido", msg)

    if filtro not in config.FILTROS:
        raise inv(f"filtro {filtro!r} desconocido. Permitidos: {', '.join(config.FILTROS)}.")
    m = 3
    if mascara not in (None, ""):
        try:
            m = int(mascara)
        except ValueError:
            raise inv(f"mascara {mascara!r} no es un entero.")
    if filtro == "media" and m not in config.MASCARAS_MEDIA:
        raise inv(f"máscara {m} no permitida para media. Permitidas: {list(config.MASCARAS_MEDIA)}.")
    if filtro != "media" and m != 3:
        raise inv(f"{filtro} solo admite máscara 3x3 (recibido {m}).")
    h = config.HILOS_DEFECTO
    if hilos not in (None, ""):
        try:
            h = int(hilos)
        except ValueError:
            raise inv(f"hilos {hilos!r} no es un entero.")
    if h not in config.HILOS_PERMITIDOS:
        raise inv(f"hilos por bloque {h} no permitido. Permitidos: {list(config.HILOS_PERMITIDOS)}.")
    f = (formato or "png").lower()
    if f not in config.FORMATOS_SALIDA:
        raise inv(f"formato de salida {formato!r} no válido. Permitidos: {list(config.FORMATOS_SALIDA)}.")
    return {"filtro": filtro, "mascara": m, "hilos": h, "formato": f}


def _purgar_solicitudes(registro: dict) -> None:
    ahora = time.time()
    for k in [k for k, v in registro.items() if ahora - v[2] > config.TTL_RESULTADOS_S and v[1].done()]:
        registro.pop(k, None)


def _buscar(request: Request, rid: str):
    if not RE_RESULTADO.match(rid):
        return None
    return request.app.state.almacen.obtener(rid)


def _manejar_fallo(e: Exception, id_sol: str):
    if isinstance(e, ErrorEntrada):
        return _error(e.http, e.codigo, e.mensaje, id_sol)
    if isinstance(e, ColaLlena):
        return _error(429, "cola_llena", str(e), id_sol, headers={"Retry-After": "2"})
    if isinstance(e, TiempoColaAgotado):
        return _error(504, "tiempo_cola_agotado", str(e), id_sol)
    if isinstance(e, MemoriaGPUInsuficiente):
        return _error(503, e.codigo, str(e), id_sol)
    if isinstance(e, ErrorGPU):
        return _error(503 if e.codigo == "gpu_no_disponible" else 500, e.codigo, str(e), id_sol)
    return _error(500, "error_interno", f"{type(e).__name__}: {e}", id_sol)


async def _ejecutar(request, datos: bytes, p: dict, id_sol: str, sha_entrada: str, t_handler: float, recepcion_ms: float):
    trabajador = request.app.state.trabajador
    almacen: AlmacenResultados = request.app.state.almacen

    t0 = perf_counter()
    img = await run_in_threadpool(imagen.decodificar, datos)
    decodificacion_ms = (perf_counter() - t0) * 1000.0

    if not trabajador.disponible:
        raise ErrorGPU_sin_gpu(trabajador)

    raw = p["formato"] == "raw32"
    fut = trabajador.enviar(
        lambda motor: motor.procesar(
            img.pixeles, img.ancho, img.alto, p["filtro"], p["mascara"], p["hilos"], devolver_flotantes=raw
        )
    )
    cola_ms, res = await asyncio.wrap_future(fut)

    t0 = perf_counter()
    if raw:
        cuerpo, mime = res.flotantes.tobytes(), "application/octet-stream"
    else:
        cuerpo, mime = await run_in_threadpool(imagen.codificar, res.pixeles, res.ancho, res.alto, p["formato"])
    codificacion_ms = (perf_counter() - t0) * 1000.0

    ext = EXTENSIONES[p["formato"]][0]
    t = res.tiempos
    r3 = lambda x: None if x is None else round(x, 3)  # noqa: E731
    meta = {
        "estado": "ok",
        "id_solicitud": id_sol,
        "filtro": p["filtro"],
        "mascara": p["mascara"],
        "hilos_por_bloque": res.hilos,
        "bloques": res.bloques,
        "entrada": {
            "ancho": img.ancho, "alto": img.alto, "formato": img.formato,
            "bytes": img.bytes_entrada, "sha256": sha_entrada,
        },
        "salida": {
            "formato": p["formato"], "mime": mime, "bytes": len(cuerpo),
            "sha256": imagen.sha256(cuerpo),
        },
        "gpu": {"nombre": trabajador.info.get("nombre"), "compute_capability": trabajador.info.get("compute_capability")},
    }
    t_fin = perf_counter()
    meta["tiempos"] = {
        "unidad": "ms",
        "recepcion_ms": r3(recepcion_ms),          # llegada -> inicio del manejador (incluye subida)
        "decodificacion_ms": r3(decodificacion_ms),  # Pillow: decodificar + gris
        "cola_ms": r3(cola_ms),                    # espera hasta que el trabajador toma el trabajo
        "reserva_ms": r3(t.reserva_ms),
        "h2d_ms": r3(t.h2d_ms),
        "kernel_ms": r3(t.kernel_ms),
        "auxiliares_gpu_ms": r3(t.auxiliares_ms),
        "d2h_ms": r3(t.d2h_ms),
        "d2h_flotantes_ms": r3(t.d2h_float_ms),
        "gpu_total_ms": r3(t.gpu_total_ms),
        "seccion_gpu_ms": r3(t.seccion_ms),
        "codificacion_ms": r3(codificacion_ms),
        "servidor_ms": r3((t_fin - t_handler) * 1000.0),  # manejador completo (reloj monotónico)
    }
    nombre = f"convolucion_{p['filtro']}{p['mascara'] if p['filtro'] == 'media' else ''}_{id_sol[:8]}.{ext}"
    almacen_res = almacen.guardar(cuerpo, ext, mime, nombre, meta)
    meta["id_resultado"] = almacen_res.id
    meta["salida"]["url"] = f"/api/v1/resultados/{almacen_res.id}"
    meta["salida"]["url_descarga"] = f"/api/v1/resultados/{almacen_res.id}?descargar=true"
    meta["salida"]["url_meta"] = f"/api/v1/resultados/{almacen_res.id}/meta"
    meta["tiempos"]["servidor_ms"] = r3((perf_counter() - t_handler) * 1000.0)
    headers = {
        "X-Id-Solicitud": id_sol,
        "Server-Timing": f"gpu;dur={t.gpu_total_ms:.3f}, cola;dur={cola_ms:.3f}, servidor;dur={meta['tiempos']['servidor_ms']}",
    }
    return meta, headers


def ErrorGPU_sin_gpu(trabajador):
    from .gpu.convolucion import SinGPU

    detalle = str(trabajador.error_inicio) if trabajador.error_inicio else "El contexto CUDA aún se está inicializando."
    return SinGPU(f"GPU no disponible: {detalle} El servicio NO procesa en CPU.")


app = create_app()
