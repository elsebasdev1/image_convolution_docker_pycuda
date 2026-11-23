"use strict";
const $ = (id) => document.getElementById(id);
const el = {
  form: $("form"), archivo: $("archivo"), drop: $("drop"), enviar: $("enviar"), estado: $("estado"),
  msg: $("mensaje"), notaMascara: $("nota-mascara"),
  imgO: $("img-original"), imgR: $("img-resultado"), vacioO: $("vacio-original"), vacioR: $("vacio-resultado"),
  infoO: $("info-original"), infoR: $("info-resultado"), descargar: $("descargar"), gpu: $("gpu"), gpuTexto: $("gpu-texto"),
  barra: $("barra"), leyenda: $("leyenda"), tabla: $("tabla"), ids: $("ids"),
};
let limites = { max_upload_mb: 32, formatos_entrada: ["JPEG", "PNG", "BMP"] };
let archivo = null, urlOriginal = null, enVuelo = false;
let ultimoIntento = null; // {huella, id}: reutiliza el id si se reintenta lo mismo tras un fallo de red

const fmtBytes = (b) => (b >= 1048576 ? (b / 1048576).toFixed(2) + " MB" : (b / 1024).toFixed(1) + " KB");
const fmtMs = (v) => (v === null || v === undefined ? "—" : v.toLocaleString("es", { minimumFractionDigits: 3, maximumFractionDigits: 3 }));
function uuid() {
  if (crypto.randomUUID) return crypto.randomUUID().replaceAll("-", "");
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, "0")).join("");
}

// ---------------------------------------------------------------- estado GPU
async function refrescarGpu() {
  try {
    const r = await fetch("/api/health", { cache: "no-store" });
    const h = await r.json();
    limites = h.limites;
    if (h.gpu.disponible) {
      el.gpu.className = "status ok";
      el.gpuTexto.textContent = `${h.gpu.nombre} · CC ${h.gpu.compute_capability} · cola ${h.cola.en_cola}/${h.cola.capacidad}`;
    } else {
      el.gpu.className = "status bad";
      el.gpuTexto.textContent = "GPU no disponible";
      el.gpu.title = h.gpu.error || "";
    }
  } catch {
    el.gpu.className = "status bad";
    el.gpuTexto.textContent = "Servicio sin respuesta";
  }
}
refrescarGpu();
setInterval(refrescarGpu, 8000);

// ------------------------------------------------------------ elegir archivo
function mostrarError(codigo, mensaje) {
  el.msg.className = "msg error";
  const c = document.createElement("b");
  c.textContent = codigo;
  el.msg.replaceChildren(c, " " + mensaje);
}
function mostrarEstado(texto) { el.msg.className = "msg"; el.msg.textContent = texto; }
const limpiarError = () => mostrarEstado("");
const valor = (n) => document.querySelector(`input[name="${n}"]:checked`).value;

function elegir(f) {
  limpiarError();
  if (!f) return;
  if (f.size > limites.max_upload_mb * 1048576) {
    return mostrarError("archivo_demasiado_grande", `El archivo pesa ${fmtBytes(f.size)}; el máximo es ${limites.max_upload_mb} MB.`);
  }
  if (!/^image\/(jpeg|png|bmp)$/.test(f.type)) {
    return mostrarError("formato_no_soportado", "Use JPEG, PNG o BMP.");
  }
  archivo = f;
  if (urlOriginal) URL.revokeObjectURL(urlOriginal);
  urlOriginal = URL.createObjectURL(f);
  el.imgO.src = urlOriginal;
  el.imgO.hidden = false; el.vacioO.hidden = true;
  el.imgO.onload = () => { el.infoO.textContent = `${el.imgO.naturalWidth}×${el.imgO.naturalHeight} px · ${fmtBytes(f.size)}`; };
  $("drop-titulo").textContent = f.name;
  $("drop-detalle").textContent = fmtBytes(f.size);
  el.imgR.hidden = true; el.vacioR.hidden = false; marcarDescarga(null); el.infoR.textContent = "—";
  pintarTiempos(null);
  el.enviar.disabled = enVuelo;
}
el.archivo.addEventListener("change", () => elegir(el.archivo.files[0]));
["dragenter", "dragover"].forEach((e) => el.drop.addEventListener(e, (ev) => { ev.preventDefault(); el.drop.classList.add("activo"); }));
["dragleave", "drop"].forEach((e) => el.drop.addEventListener(e, (ev) => { ev.preventDefault(); el.drop.classList.remove("activo"); }));
el.drop.addEventListener("drop", (ev) => elegir(ev.dataTransfer.files[0]));

const filtroActual = () => valor("filtro");
function actualizarMascara() {
  const media = filtroActual() === "media";
  document.querySelectorAll('input[name="mascara"]').forEach((r) => { r.disabled = !media; });
  el.notaMascara.textContent = media ? "" : "fija 3 × 3";
}
document.querySelectorAll('input[name="filtro"]').forEach((r) => r.addEventListener("change", actualizarMascara));
actualizarMascara();
function marcarDescarga(href) {
  el.descargar.classList.toggle("disabled", !href);
  el.descargar.setAttribute("aria-disabled", String(!href));
  if (href) el.descargar.href = href; else el.descargar.removeAttribute("href");
}

// ----------------------------------------------------------------- procesar
// Etapas agrupadas en 4 categorías de color: transferencia, cómputo GPU, servidor y cola.
const CAT = { transf: ["Transferencia", "--g1"], gpu: ["Cómputo GPU", "--g2"], srv: ["Servidor (CPU)", "--g4"], cola: ["Cola", "--g3"] };
const ETAPAS = [
  ["recepcion_ms", "Recepción de la subida", "reloj monotónico", "srv"],
  ["decodificacion_ms", "Decodificación + gris", "reloj monotónico", "srv"],
  ["cola_ms", "Espera en cola", "reloj monotónico", "cola"],
  ["h2d_ms", "CPU → GPU", "eventos CUDA", "transf"],
  ["kernel_ms", "Kernel de convolución", "eventos CUDA", "gpu"],
  ["auxiliares_gpu_ms", "Conversión u8 ↔ f32", "eventos CUDA", "gpu"],
  ["d2h_ms", "GPU → CPU", "eventos CUDA", "transf"],
  ["codificacion_ms", "Codificación de salida", "reloj monotónico", "srv"],
];

el.form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  if (enVuelo || !archivo) return; // evita envíos duplicados accidentales
  enVuelo = true;
  el.enviar.disabled = true;
  mostrarEstado("Procesando…");

  const params = { filtro: filtroActual(), mascara: valor("mascara"), hilos: valor("hilos"), formato: valor("formato") };
  const huella = [archivo.name, archivo.size, archivo.lastModified, ...Object.values(params)].join("|");
  const id = ultimoIntento && ultimoIntento.huella === huella ? ultimoIntento.id : uuid();
  ultimoIntento = { huella, id };

  const fd = new FormData();
  fd.append("imagen", archivo);
  fd.append("filtro", params.filtro);
  if (params.filtro === "media") fd.append("mascara", params.mascara);
  fd.append("hilos", params.hilos);
  fd.append("formato", params.formato);
  fd.append("id_solicitud", id);

  const t0 = performance.now();
  try {
    const r = await fetch("/api/v1/convolucion", { method: "POST", body: fd });
    const j = await r.json();
    const clienteMs = performance.now() - t0;
    if (!r.ok) {
      mostrarError(j.error?.codigo ?? "error", j.error?.mensaje ?? `HTTP ${r.status}`);
      if (![429, 503, 504].includes(r.status)) ultimoIntento = null;
      return;
    }
    ultimoIntento = null; // éxito: el próximo envío es una solicitud nueva
    mostrarEstado(`Listo en ${fmtMs(clienteMs)} ms (cliente).`);
    mostrarResultado(j, clienteMs);
  } catch (e) {
    mostrarError("sin_conexion", "No se pudo contactar con el servicio. Reintentar reutiliza el mismo id de solicitud.");
  } finally {
    enVuelo = false;
    el.enviar.disabled = !archivo;
    refrescarGpu();
  }
});

function mostrarResultado(j, clienteMs) {
  el.imgR.src = j.salida.url + "?t=" + j.id_resultado;
  el.imgR.hidden = false; el.vacioR.hidden = true;
  el.infoR.textContent = `${j.entrada.ancho}×${j.entrada.alto} px · ${fmtBytes(j.salida.bytes)}`;
  marcarDescarga(j.salida.url_descarga);
  pintarTiempos(j, clienteMs);
}

const celda = (txt, cls) => { const td = document.createElement("td"); td.textContent = txt; if (cls) td.className = cls; return td; };
function fila(nombre, ms, medido, total = false) {
  const tr = document.createElement("tr");
  if (total) tr.className = "total";
  tr.append(celda(nombre), celda(ms === null ? "—" : fmtMs(ms), "num"), celda(medido, "med"));
  return tr;
}

// Sin resultado, la tabla y la leyenda ya existen con guiones: el panel no cambia de tamaño al procesar.
function pintarTiempos(j, clienteMs) {
  const t = j ? j.tiempos : {};
  const filas = ETAPAS.map(([k, nombre, medido, cat]) => ({ nombre, medido, cat, v: j ? t[k] ?? 0 : null }));
  const suma = j ? filas.reduce((a, f) => a + f.v, 0) : 0;

  el.barra.replaceChildren(...(j ? filas.filter((f) => f.v > 0).map((f) => {
    const d = document.createElement("div");
    d.style.cssText = `width:${(100 * f.v) / suma}%;background:var(${CAT[f.cat][1]})`;
    d.title = `${f.nombre}: ${fmtMs(f.v)} ms`;
    return d;
  }) : []));
  el.leyenda.replaceChildren(...Object.values(CAT).map(([nombre, color]) => {
    const li = document.createElement("li");
    const i = document.createElement("i"); i.style.background = `var(${color})`;
    li.append(i, nombre);
    return li;
  }));

  const filasTabla = filas.map((f) => fila(f.nombre, f.v, f.medido));
  filasTabla.push(fila("Total en el servidor", j ? t.servidor_ms : null, "reloj monotónico", true));
  filasTabla.push(fila("Total en el navegador", j ? clienteMs : null, "performance.now()", true));
  el.tabla.replaceChildren(...filasTabla);

  const ids = [
    ["solicitud", j?.id_solicitud], ["resultado", j?.id_resultado],
    ["filtro", j && (j.filtro === "media" ? `media ${j.mascara}×${j.mascara}` : j.filtro)],
    ["hilos × bloques", j && `${j.hilos_por_bloque} × ${j.bloques.toLocaleString("es")}`],
    ["sha256 entrada", j?.entrada.sha256], ["sha256 salida", j?.salida.sha256],
  ];
  el.ids.replaceChildren(...ids.flatMap(([k, v]) => {
    const dt = document.createElement("dt"), dd = document.createElement("dd");
    dt.textContent = k; dd.textContent = v ?? "—"; dd.title = v ?? "";
    return [dt, dd];
  }));
}
pintarTiempos(null);
