# Práctica 03 — Servicio web de convolución de imágenes (Docker + PyCUDA)

Computación Paralela · UPS · Grupo F. Continúa la práctica 02: los filtros de media (3×3, 9×9, 21×21),
nitidez 3×3 y laplaciano 3×3 se ejecutan con **kernels CUDA propios** compilados por PyCUDA y se ofrecen
como servicio HTTP dentro de un contenedor con acceso a la GPU.

```
Navegador ──POST──▶ FastAPI/uvicorn ──cola──▶ hilo trabajador ──PyCUDA──▶ GPU
   ▲                  (valida, decodifica)     (dueño del contexto)           │
   └──────────── JSON con tiempos + imagen PNG/JPEG ◀──────── resultado ◀────┘
```

## Trazabilidad

| Elemento | Valor |
|---|---|
| Repositorio práctica 02 | https://github.com/elsebasdev1/image_convolution_processing |
| Commit de origen práctica 02 | `fb95088d827a9e62ab2ba80d0b6fe78bc625a069` (kernel `convolucionGPU` copiado sin cambios) |
| Commit práctica 03 | ver `git rev-parse HEAD`; la imagen lo guarda en la etiqueta `org.opencontainers.image.revision` |

## Estructura

```
app/            servicio y motor (sin NumPy en la lógica de convolución)
  gpu/kernels.py        kernels CUDA propios + máscaras
  gpu/convolucion.py    MotorGPU: contexto, memoria, transferencias, eventos (sin HTTP)
  gpu/trabajador.py     hilo dueño del contexto + cola acotada
  gpu/referencia_cpu.py convolución secuencial de referencia (pruebas)
  main.py               rutas HTTP · imagen.py E/S Pillow · almacen.py resultados temporales
  cli.py                ejecución directa (sin HTTP) del mismo motor
web/            interfaz (HTML + CSS + JS sin framework)
bench/          medición: directo.py, cliente.py (HTTP + concurrencia), resumir.py
tests/          pruebas (HTTP sin GPU; motor y equivalencia con GPU)
datos/imagenes/ imágenes de entrada (NO versionadas, ver «Datos de entrada»)
resultados/     CSV de mediciones, resumen, validación, entornos y evidencia del contenedor
```

## Cambios respecto a la práctica 02

* **Kernel de convolución idéntico** (un píxel de salida por hilo, distribución 1D en X, extensión con ceros, `float32`).
* **Dos kernels auxiliares nuevos** (`u8_a_float`, `float_a_u8`): la conversión uint8↔float32 y el tratamiento del
  resultado (desplazamiento +128 del laplaciano, truncar, recortar a [0,255]) se hacen en la GPU. Así no se recorren
  36 Mpx en Python y la transferencia es 4× menor. Se miden aparte (`auxiliares_gpu_ms`). No son filtros.
* La práctica 02 transfería `float32`; aquí H2D/D2H mueven `uint8`. El kernel de convolución se mide igual (eventos CUDA).
* Diferencia numérica CPU(double) vs GPU(float32) al truncar a entero: ≤ 1 nivel (ver `tests/test_gpu.py`).

## Compatibilidad

| Componente | Quién lo aporta | Versión usada |
|---|---|---|
| Controlador NVIDIA (módulo del kernel + `libcuda.so`) | **Anfitrión** (inyectado por NVIDIA Container Toolkit) | 620.73.04 (CUDA 13.5 soportado) |
| NVIDIA Container Toolkit | Anfitrión | 1.20.2 |
| Toolkit CUDA (`nvcc`, cabeceras, runtime) | **Imagen** `nvidia/cuda:13.4.3-devel-ubuntu24.04` | 13.4.3 |
| Python | Imagen (Ubuntu 24.04) y `.python-version` | 3.12 |
| PyCUDA, FastAPI, uvicorn, Pillow, NumPy (dependencia de PyCUDA) | `uv.lock` (versiones exactas) | ver `uv.lock` |
| GPU | Anfitrión | RTX 3060 Laptop, CC 8.6, 6 GB |

Regla: el controlador del anfitrión debe soportar una versión CUDA ≥ a la del toolkit de la imagen (13.4).
La imagen es `-devel` porque PyCUDA invoca `nvcc` al arrancar para compilar los kernels.

## Ejecución directa (anfitrión)

```bash
uv sync                                   # crea .venv con las versiones de uv.lock
uv run python -m app.cli --info           # GPU y versiones
uv run python -m app.cli datos/imagenes/extra2_4000.jpeg --filtro media --mascara 9 --reps 6 --salida /tmp/salida.png
uv run python -m app                      # servicio en http://localhost:8000
uv run pytest                             # pruebas (las de GPU se omiten si no hay GPU)
```

## Inicio rápido (Makefile + Docker Compose)

```bash
make up       # construye, arranca con GPU y espera: web en http://localhost:8000
make down     # detiene
make help     # logs, status, gpu, test, shell, bench
```

## Docker

Requisitos del anfitrión: Docker, controlador NVIDIA y NVIDIA Container Toolkit
(`sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`).

```bash
scripts/construir.sh                      # docker build con --build-arg GIT_COMMIT (exige árbol limpio)
scripts/verificar_gpu_contenedor.sh       # nvidia-smi / nvcc / motor dentro del contenedor
scripts/ejecutar.sh                       # docker run --gpus all -p 8000:8000 …  → http://localhost:8000
```

Equivalente manual: `docker build -t convolucion-gpu:1.0 .` y `docker run --rm --gpus all -p 8000:8000 convolucion-gpu:1.0`.
Sin `--gpus all` el servicio **arranca igualmente** pero responde `503 gpu_no_disponible` con el diagnóstico; nunca cae a CPU.
Servidor: `uvicorn` con un solo proceso (un contexto CUDA), sin `--reload` ni depurador.

Verificar desde una copia limpia:

```bash
git clone https://github.com/elsebasdev1/image_convolution_docker_pycuda.git /tmp/p3 && cd /tmp/p3
docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t convolucion-gpu:1.0 .
scripts/verificar_gpu_contenedor.sh && scripts/probar_sin_gpu.sh
```

## Servicio

* Interfaz: `GET /` · Documentación OpenAPI: `GET /docs` · Estado: `GET /api/health`
* Procesar: `POST /api/v1/convolucion` (multipart). Detalle en la sección «API HTTP».

```bash
curl -F imagen=@datos/imagenes/extra2_4000.jpeg -F filtro=media -F mascara=9 http://localhost:8000/api/v1/convolucion
```

Límites (variables de entorno en `app/config.py`): 32 MB por archivo, 36 Mpx, lado ≤ 8000, formatos JPEG/PNG/BMP,
máscaras de media {3, 9, 21}, hilos {64…1024}, cola de 8, resultados 15 min / 20 archivos.

Concurrencia: **un único hilo trabajador** posee el contexto CUDA y atiende una cola FIFO acotada; el decodificado y
la codificación de imagen ocurren fuera de ese hilo, así que se solapan con la GPU de otra solicitud.

## Mediciones

```bash
# 8.1 / 8.2 / 8.3 directo en el anfitrión
uv run python -m bench.directo --modo directo_host
# 8.1 / 8.2 / 8.3 directo dentro del contenedor
docker run --rm --gpus all --user $(id -u):$(id -g) -e HOME=/tmp -v "$PWD/datos:/app/datos:ro" -v "$PWD/resultados:/app/resultados" \
  convolucion-gpu:1.0 python -m bench.directo --modo directo_contenedor
# 8.1 (servicio), 8.2, 8.3 y 8.4 (concurrencia) contra el servicio contenerizado
uv run python -m bench.cliente --modo servicio_contenedor --url http://localhost:8000
uv run python -m bench.resumir
```

Protocolo: 1 ejecución en frío (primera tras arrancar), 1 de calentamiento, 10 repeticiones medidas por configuración;
mediana, desviación, cuartiles y extremos en `resultados/resumen.csv`. Tiempos de GPU con eventos CUDA; servidor y
cliente con `time.perf_counter()` (monotónico). Datos crudos: `resultados/mediciones.csv` (cualquier fallo se guardaría también en `resultados/errores.jsonl`; no hubo ninguno). El protocolo completo se repite con `make medir`.

## Datos de entrada

Las imágenes no se versionan. Para reproducir las mediciones se usan **las mismas de la práctica 02** (`imagenes/extras/extra2_{2000,4000,6000}.jpeg`
de [image_convolution_processing](https://github.com/elsebasdev1/image_convolution_processing)): copiarlas a `datos/imagenes/` y comprobar el SHA-256:

| Archivo | Resolución | SHA-256 |
|---|---|---|
| `extra2_2000.jpeg` | 2000×2000 | `2bd12124f9266285012baf4be2e65e6e931fff60cd22ece5b94b889a1d92906b` |
| `extra2_4000.jpeg` | 4000×4000 | `1bfe684b4fc7629d0aa7125b730801d2a6dbaa963f1ca9173d1ea504729c5429` |
| `extra2_6000.jpeg` | 6000×6000 | `a6ecf0cd9abdd2871e00cf03df7aeddeb45f3aaa4137505c693ff22ebc1ce5aa` |

```bash
cd datos/imagenes && sha256sum extra2_*.jpeg
```

## Resultados

`resultados/`: `mediciones.csv` (una fila por ejecución: modo, filtro, resolución, máscara, hilos, repetición, concurrencia, tiempos en ms, estado, entorno),
`resumen.csv` (mediana, desviación, cuartiles), `concurrencia_rondas.csv`, `arranque.csv`, `validacion.csv` (incluye las entradas inválidas y sus códigos de error), `entorno_*.json`,
`contenedor_gpu.txt` (`docker ps`, id de la imagen, `nvidia-smi`, `nvcc` y `/api/health` del contenedor),
`sin_gpu.txt` (el mismo contenedor sin `--gpus all`: `/api/health` y `503 gpu_no_disponible`) y
`construccion_imagen.txt` (construcción de la imagen desde cero).

## API HTTP

Base: `http://<host>:8000`. Todos los errores devuelven JSON:
`{"estado":"error","error":{"codigo":"…","mensaje":"…"},"id_solicitud":"…"}`.

### POST `/api/v1/convolucion`

`multipart/form-data`

| Campo | Obligatorio | Valores | Notas |
|---|---|---|---|
| `imagen` | sí | JPEG, PNG, BMP; ≤ 32 MB; ≤ 36 Mpx; lado 3…8000 | Se convierte a gris con `convert("L")` |
| `filtro` | sí | `media`, `nitidez`, `laplaciano` | |
| `mascara` | no (3) | media: `3`, `9`, `21`; nitidez/laplaciano: solo `3` | |
| `hilos` | no (256) | `64,128,256,512,1024` | hilos por bloque; bloques = ⌈píxeles/hilos⌉ en X |
| `formato` | no (`png`) | `png`, `jpeg` (q=100), `raw32` | `raw32` = `float32` little-endian antes del recorte (validación) |
| `id_solicitud` | no | `[A-Za-z0-9_-]{8,64}` (o cabecera `X-Id-Solicitud`) | idempotencia; si falta lo genera el servidor |

**200** — ejemplo abreviado:

```json
{"estado":"ok","id_solicitud":"…","id_resultado":"…","filtro":"media","mascara":9,
 "hilos_por_bloque":256,"bloques":62500,
 "entrada":{"ancho":4000,"alto":4000,"formato":"JPEG","bytes":3383378,"sha256":"…"},
 "salida":{"formato":"png","mime":"image/png","bytes":5584431,"sha256":"…","url":"/api/v1/resultados/<id>",
           "url_descarga":"…?descargar=true","url_meta":"…/meta"},
 "tiempos":{"unidad":"ms","recepcion_ms":…,"decodificacion_ms":…,"cola_ms":…,"reserva_ms":…,"h2d_ms":…,
            "kernel_ms":…,"auxiliares_gpu_ms":…,"d2h_ms":…,"gpu_total_ms":…,"seccion_gpu_ms":…,
            "codificacion_ms":…,"servidor_ms":…}}
```

Tiempos (todos en **ms**): `h2d/kernel/auxiliares/d2h` con eventos CUDA; `recepcion` (llegada→manejador, incluye la
subida), `decodificacion`, `cola`, `reserva`, `seccion_gpu`, `codificacion` y `servidor` con reloj monotónico.
Cabeceras: `X-Id-Solicitud`, `Server-Timing`. Un `id_solicitud` repetido con el mismo contenido devuelve la respuesta
original con `X-Solicitud-Duplicada: true` sin reprocesar.

| HTTP | `codigo` | Causa |
|---|---|---|
| 400 | `archivo_invalido` | vacío, no es imagen o está corrupto |
| 409 | `id_solicitud_en_conflicto` | mismo id con otra imagen/parámetros |
| 413 | `archivo_demasiado_grande`, `imagen_demasiado_grande` | bytes, píxeles o lado sobre el límite |
| 415 | `formato_no_soportado` | formato distinto de JPEG/PNG/BMP |
| 422 | `parametro_invalido`, `dimensiones_invalidas` | filtro, máscara, hilos, formato o id inválidos; lado < 3 |
| 429 | `cola_llena` | cola de la GPU llena (`Retry-After: 2`) |
| 500 | `error_procesamiento` | fallo CUDA durante el procesamiento |
| 503 | `gpu_no_disponible`, `memoria_gpu_insuficiente` | sin GPU/contexto (diagnóstico en `mensaje`) o VRAM insuficiente |
| 504 | `tiempo_cola_agotado` | esperó más de `TIEMPO_MAX_COLA_S` en cola |

### GET `/api/v1/resultados/{id}` · `?descargar=true` · `/meta`

Imagen del resultado (inline o adjunta) y su JSON. `404 resultado_no_encontrado` si no existe o expiró (TTL 15 min).

### GET `/api/health`

Estado (`ok` | `sin_gpu`), datos de la GPU (o el error de inicialización), cola (`en_cola`, `capacidad`, `procesadas`,
`rechazadas`), límites y versiones.
