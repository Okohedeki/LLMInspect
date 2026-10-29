// The neuron atlas, at the top of the Neurons view: every MLP neuron of the model at once, one cell each,
// coloured by a map (llminspect/recorder/atlas.py). Layers run down the page and scroll; inside a layer
// the neurons are packed by node, the weight anatomy's neuron clusters that Trace draws as "c" nodes, so a
// node is a tile of its own neurons. Nodes mode draws each tile as one block. Clicking a neuron opens its
// card below and draws its direct weight links; clicking a node opens it beside the atlas, zoomed to its
// neurons. Ctrl + wheel (or − and +) zooms; the page scrolls as usual.

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const vis = (s) => (s ?? "").replace(/\n/g, "⏎").replace(/\t/g, "⇥").replace(/^ /, "·") || "∅";
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const fmt = (x) => (x == null ? "–" : Math.abs(x) >= 100 ? x.toFixed(0) : Math.abs(x) >= 1 ? x.toFixed(2) : x.toPrecision(2));
async function api(path, opts) {
  const res = await fetch(path, opts);
  const text = await res.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch { /* not JSON */ }
  if (!res.ok) throw new Error(body?.detail || text || `${res.status}`);
  return body;
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}) });

const MAPS = [
  { id: "write", label: "Write", hint: "How much each neuron writes per unit of activation (weights only)" },
  { id: "spread", label: "Spread", hint: "How widely each neuron's activation varies on text" },
  { id: "mean", label: "Mean", hint: "Each neuron's average activation on text: below zero blue, above orange" },
  { id: "changed", label: "Changed", hint: "How much each neuron's weights differ from the originals while edits or a checkpoint are applied" },
  { id: "fires", label: "Fires", hint: "Each neuron's peak on a text you give, as a z-score against its usual activity" },
];
const GUTTER = 58;      // CSS px for the layer labels
const CHUNK = 256;      // neurons per tile when a layer has no clusters yet
const NODE_ZOOM = 220;  // CSS px the node panel's grid may use

const A = {
  layout: null, maps: {}, map: "write", mode: "neurons", cell: 0, loading: false, error: "",
  geo: null, bitmaps: new Map(), colors: null, hover: null, sel: null, node: null,
  links: null, fires: null, firing: false, try: null, trying: false, frame: 0, model: null, seq: 0,
};

// ---------------------------------------------------------------- colours
function hexToRgb(h) {
  h = h.replace("#", "");
  if (h.length === 3) h = h.split("").map((c) => c + c).join("");
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
}
function lerpStops(stops, n = 256) {
  const rgb = stops.map(hexToRgb), out = [];
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * (rgb.length - 1), k = Math.min(Math.floor(x), rgb.length - 2), f = x - k;
    out.push(rgb[k].map((v, j) => Math.round(v + (rgb[k + 1][j] - v) * f)));
  }
  return out;
}
function readColors() {
  const dark = cssVar("--scheme-dark") === "1";
  const stops = cssVar("--ramp").split(",").map((s) => s.trim());
  const neg = cssVar("--div-neg"), pos = cssVar("--div-pos"), mid = cssVar("--surface");  // zero recedes into the stage
  A.colors = {
    dark, bg: cssVar("--surface"), ink: cssVar("--ink"), ink2: cssVar("--ink-2"), muted: cssVar("--muted"), grid: cssVar("--grid"),
    none: hexToRgb(cssVar("--no-data")), select: cssVar("--select-ring"), halo: cssVar("--select-halo"), hover: cssVar("--hover-ring"),
    edgeIn: cssVar("--edge-in"), edgeOut: cssVar("--edge-out"), glow: cssVar("--glow"),
    seq: lerpStops(dark ? stops : stops.slice(1)), div: lerpStops([neg, mid, pos]),
  };
  A.bitmaps.clear();
}
const lut = (m) => (m?.signed ? A.colors.div : A.colors.seq);
function colorOf(m, b) {
  if (!m.signed && b === 0 && m.scale === "log") return A.colors.none;
  return lut(m)[b];
}
/** A byte back to the value it stands for, from the map's range. */
function value(m, b) {
  if (!m) return null;
  if (m.signed) return ((b - 127.5) / 127.5) * m.hi;
  if (m.scale === "log") return b === 0 ? 0 : m.lo * Math.exp(((b - 1) / 254) * Math.log(m.hi / m.lo));
  return (b / 255) * m.hi;
}

// ---------------------------------------------------------------- data
const b64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));

async function loadLayout() {
  const seq = ++A.seq;
  A.loading = true; A.error = ""; renderState();
  try {
    const lay = await api("/llminspect/neurons/atlas");
    if (seq !== A.seq) return;
    for (const L of lay.layers) L.assign = L.clusters ? b64(L.clusters) : null;
    A.layout = lay; A.geo = null; A.bitmaps.clear();
    await loadMap(A.map === "fires" && !A.fires ? "write" : A.map);
  } catch (err) {
    if (seq === A.seq) A.error = err.message;
  }
  if (seq === A.seq) { A.loading = false; renderState(); renderHead(); layoutNow(); }
}

async function loadMap(name, { force = false } = {}) {
  if (name === "fires") { if (A.fires) { A.map = name; A.bitmaps.clear(); renderHead(); schedule(); } return; }
  if (!force && A.maps[name]) { A.map = name; A.bitmaps.clear(); renderHead(); schedule(); return; }
  const seq = A.seq;
  A.loading = true; renderState();
  try {
    const m = await api(`/llminspect/neurons/atlas/${name}`);
    if (seq !== A.seq) return;
    m.bytes = Object.fromEntries(Object.entries(m.layers).map(([k, v]) => [k, b64(v)]));
    A.maps[name] = m; A.map = name; A.error = "";
  } catch (err) { A.error = err.message; }
  A.loading = false; A.bitmaps.clear(); renderState(); renderHead(); renderNode(); schedule();
}

async function readText(text) {
  text = String(text || "").trim();
  if (!text) return;
  A.firing = true; renderHead();
  try {
    const f = await post("/llminspect/neurons/fires", { text });
    f.bytes = Object.fromEntries(Object.entries(f.layers).map(([k, v]) => [k, b64(v)]));
    A.fires = f; A.maps.fires = f; A.map = "fires"; A.error = "";
  } catch (err) { A.error = err.message; }
  A.firing = false; A.bitmaps.clear(); renderState(); renderHead(); renderFires(); renderNode(); schedule();
}

const current = () => A.maps[A.map] || null;

// ---------------------------------------------------------------- geometry
// Device pixels throughout. A layer is a band; its nodes are tiles packed onto shelves, largest first;
// inside a tile the neurons run in index order, row by row.
function tilesOf(L) {
  const groups = new Map();
  if (L.assign) for (let n = 0; n < L.width; n++) { const c = L.assign[n]; if (!groups.has(c)) groups.set(c, []); groups.get(c).push(n); }
  else for (let n = 0; n < L.width; n++) { const c = Math.floor(n / CHUNK); if (!groups.has(c)) groups.set(c, []); groups.get(c).push(n); }
  return [...groups.entries()].map(([c, ns]) => ({ c, members: Int32Array.from(ns) }))
    .sort((a, b) => b.members.length - a.members.length || a.c - b.c);
}

function autoCell(widthCss) {
  const widest = Math.max(1, ...A.layout.layers.map((L) => L.width));
  return clamp(Math.round(Math.sqrt((64 * widthCss) / widest)), 1, 14);
}

function layoutNow() {
  const scroll = $("#at-scroll");
  if (!A.layout || !scroll || !scroll.clientWidth) return;  // hidden: laid out when shown
  const dpr = window.devicePixelRatio || 1, wCss = Math.max(120, scroll.clientWidth - GUTTER - 14);
  if (!A.zoomed) A.cell = autoCell(wCss);  // fits the width until the user zooms
  const cd = Math.max(1, Math.round(A.cell * dpr)), gap = Math.max(2, Math.round(Math.max(2, A.cell) * dpr)),
    W = Math.floor(wCss * dpr), pad = Math.round(14 * dpr);
  let y = pad;
  const layers = A.layout.layers.map((L) => {
    const tiles = tilesOf(L), pos = new Int32Array(L.width * 2), tileOf = new Int32Array(L.width);
    let x = 0, sy = 0, shelf = 0;
    tiles.forEach((t, ti) => {
      const k = t.members.length, cols = Math.max(1, Math.min(Math.ceil(Math.sqrt(k)), Math.floor(W / cd))), rows = Math.ceil(k / cols);
      const tw = cols * cd, th = rows * cd;
      if (x > 0 && x + tw > W) { x = 0; sy += shelf + gap; shelf = 0; }
      Object.assign(t, { x, y: sy, cols, rows, w: tw, h: th });
      t.members.forEach((n, j) => { pos[2 * n] = x + (j % cols) * cd; pos[2 * n + 1] = sy + Math.floor(j / cols) * cd; tileOf[n] = ti; });
      x += tw + gap; shelf = Math.max(shelf, th);
    });
    const band = { L, tiles, pos, tileOf, y, h: sy + shelf };
    y += band.h + pad * 2;
    return band;
  });
  A.geo = { dpr, cd, gap, W, layers, height: y, gutter: Math.round(GUTTER * dpr), byLayer: new Map(layers.map((b) => [b.L.layer, b])) };
  $("#at-spacer").style.height = `${Math.ceil(y / dpr)}px`;
  A.bitmaps.clear();
  sizeCanvas();
  schedule();
}

function sizeCanvas() {
  const cv = $("#at-canvas"), scroll = $("#at-scroll"), dpr = window.devicePixelRatio || 1;
  const w = scroll.clientWidth, h = scroll.clientHeight;
  cv.style.width = `${w}px`; cv.style.height = `${h}px`;
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
}

/** The mean of a node's bytes, or its most extreme one for the maps where one neuron is the point. */
function nodeByte(m, bytes, t) {
  const peak = A.map === "changed" || A.map === "fires";
  let s = 0, best = m.signed ? 127.5 : 0;
  for (const n of t.members) {
    const b = bytes[n];
    if (m.signed) { if (Math.abs(b - 127.5) > Math.abs(best - 127.5)) best = b; s += b; }
    else { if (b > best) best = b; s += b; }
  }
  return Math.round(peak || m.signed ? best : s / t.members.length);
}

function bitmap(band) {
  const m = current(), key = band.L.layer;
  if (!m) return null;
  if (A.bitmaps.has(key)) return A.bitmaps.get(key);
  const { W, cd, dpr } = A.geo, H = Math.max(1, band.h), bytes = m.bytes[String(key)];
  const cv = document.createElement("canvas");
  cv.width = W; cv.height = H;
  const ctx = cv.getContext("2d");
  if (!bytes) { A.bitmaps.set(key, cv); return cv; }
  if (A.mode === "nodes") {
    const r = Math.min(4 * dpr, cd * 2);
    for (const t of band.tiles) {
      const [cr, cg, cb] = colorOf(m, nodeByte(m, bytes, t));
      ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
      ctx.beginPath(); ctx.roundRect(t.x, t.y, t.w, t.h, r); ctx.fill();
    }
  } else {
    const img = ctx.createImageData(W, H), px = img.data, inner = cd >= 4 ? cd - 1 : cd;
    for (let n = 0; n < band.L.width; n++) {
      const [r, g, b] = colorOf(m, bytes[n]), x0 = band.pos[2 * n], y0 = band.pos[2 * n + 1];
      for (let yy = 0; yy < inner; yy++) {
        let o = ((y0 + yy) * W + x0) * 4;
        for (let xx = 0; xx < inner; xx++, o += 4) { px[o] = r; px[o + 1] = g; px[o + 2] = b; px[o + 3] = 255; }
      }
    }
    ctx.putImageData(img, 0, 0);
  }
  A.bitmaps.set(key, cv);
  return cv;
}

// ---------------------------------------------------------------- drawing
function schedule() {
  if (A.frame) return;
  A.frame = requestAnimationFrame(() => { A.frame = 0; draw(); });
}

/** Where a neuron's cell is on the canvas (device px), or null when its layer isn't laid out. */
function cellAt(layer, n, top) {
  const band = A.geo?.byLayer.get(layer);
  if (!band || n >= band.L.width) return null;
  return { x: A.geo.gutter + band.pos[2 * n], y: band.y - top + band.pos[2 * n + 1], band };
}

function draw() {
  const cv = $("#at-canvas"), scroll = $("#at-scroll");
  if (!cv || !A.geo || !A.colors) return;
  const ctx = cv.getContext("2d"), { dpr, gutter, cd } = A.geo, C = A.colors;
  const top = Math.round(scroll.scrollTop * dpr), H = cv.height;
  ctx.clearRect(0, 0, cv.width, H);
  const m = current();
  const keep = new Set();
  ctx.font = `${Math.round(11 * dpr)}px ${cssVar("--mono") || "monospace"}`;
  ctx.textBaseline = "top";
  for (const band of A.geo.layers) {
    const y = band.y - top;
    if (y > H + 4 * H || y + band.h < -4 * H) continue;
    keep.add(band.L.layer);
    if (y > H || y + band.h < 0) continue;
    const bm = m ? bitmap(band) : null;
    if (bm) ctx.drawImage(bm, gutter, y);
    else {  // still loading: the bands, faintly
      ctx.fillStyle = C.grid;
      for (const t of band.tiles) ctx.fillRect(gutter + t.x, y + t.y, t.w, t.h);
    }
    const strong = A.sel?.layer === band.L.layer || A.node?.layer === band.L.layer;
    ctx.fillStyle = strong ? C.ink : C.muted;
    ctx.fillText(`L${band.L.layer}`, Math.round(8 * dpr), y);
    ctx.fillStyle = C.muted;
    ctx.font = `${Math.round(9.5 * dpr)}px ${cssVar("--mono") || "monospace"}`;
    ctx.fillText(`${band.tiles.length}${band.L.assign ? " nodes" : ""}`, Math.round(8 * dpr), y + Math.round(15 * dpr));
    ctx.font = `${Math.round(11 * dpr)}px ${cssVar("--mono") || "monospace"}`;
  }
  for (const k of A.bitmaps.keys()) if (!keep.has(k)) A.bitmaps.delete(k);

  // The focus: dim the rest, draw the links additively, then ring what's selected.
  const focus = A.sel && A.links?.neuron === label(A.sel.layer, A.sel.n);
  if (focus || A.node) {
    ctx.fillStyle = C.dark ? "rgba(7,8,11,0.55)" : "rgba(255,255,255,0.6)";
    ctx.fillRect(gutter, 0, cv.width - gutter, H);
  }
  if (A.node) {
    const band = A.geo.byLayer.get(A.node.layer), t = band?.tiles.find((x) => x.c === A.node.c);
    if (t) {
      const bm = bitmap(band), x = gutter + t.x, y = band.y - top + t.y;
      if (bm) ctx.drawImage(bm, t.x, t.y, t.w, t.h, x, y, t.w, t.h);
      ring(ctx, x, y, t.w, t.h, C.select, 2 * dpr, C.halo);
    }
  }
  if (focus) {
    const src = cellAt(A.sel.layer, A.sel.n, top);
    const all = [...A.links.reads_it.map((l) => ({ ...l, col: C.edgeOut })), ...A.links.it_reads.map((l) => ({ ...l, col: C.edgeIn }))];
    const most = Math.max(1e-9, ...all.map((l) => Math.abs(l.cos)));
    ctx.save();
    if (C.dark) ctx.globalCompositeOperation = "lighter";
    ctx.lineWidth = Math.max(1, dpr);
    for (const l of all) {
      const dst = cellAt(l.layer, l.index, top);
      if (!src || !dst) continue;
      const a = 0.25 + 0.75 * (Math.abs(l.cos) / most);
      const sx = src.x + cd / 2, sy = src.y + cd / 2, dx = dst.x + cd / 2, dy = dst.y + cd / 2, bend = gutter * 0.6 + Math.abs(dy - sy) * 0.08;
      ctx.strokeStyle = l.col; ctx.globalAlpha = a * 0.8;
      ctx.beginPath(); ctx.moveTo(sx, sy); ctx.bezierCurveTo(sx - bend, sy + (dy - sy) * 0.3, dx - bend, dy - (dy - sy) * 0.3, dx, dy); ctx.stroke();
      ctx.globalAlpha = a; ctx.fillStyle = l.col;
      const r = Math.max(2 * dpr, cd * 0.9);
      ctx.beginPath(); ctx.arc(dx, dy, r, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
  }
  if (A.map === "fires" && A.fires && !focus && !A.node) {  // the top of a read: a soft glow on each
    ctx.save();
    if (C.dark) ctx.globalCompositeOperation = "lighter";
    ctx.fillStyle = C.glow;
    for (const f of A.fires.top.slice(0, 24)) {
      const p = cellAt(f.layer, f.index, top);
      if (!p) continue;
      ctx.globalAlpha = 0.35; ctx.beginPath(); ctx.arc(p.x + cd / 2, p.y + cd / 2, Math.max(5 * dpr, cd * 2.2), 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
  }
  if (A.sel) {
    const p = cellAt(A.sel.layer, A.sel.n, top), r = Math.max(3 * dpr, cd);
    if (p) ring(ctx, p.x - r / 2, p.y - r / 2, cd + r, cd + r, C.select, 2 * dpr, C.halo);
  }
  if (A.hover) {
    const { band, tile, n } = A.hover, y = band.y - top;
    if (A.mode === "nodes" || n == null) ring(ctx, gutter + tile.x - dpr, y + tile.y - dpr, tile.w + 2 * dpr, tile.h + 2 * dpr, C.hover, dpr);
    else { const p = cellAt(band.L.layer, n, top); ring(ctx, p.x - dpr, p.y - dpr, cd + 2 * dpr, cd + 2 * dpr, C.hover, dpr); }
  }
}

function ring(ctx, x, y, w, h, color, lw, halo) {
  ctx.save();
  if (halo) { ctx.strokeStyle = halo; ctx.lineWidth = lw * 4; ctx.strokeRect(x, y, w, h); }
  ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.strokeRect(x, y, w, h);
  ctx.restore();
}

// ---------------------------------------------------------------- hit testing
const label = (layer, n) => `L${layer} N${n}`;

function hit(ev) {
  const scroll = $("#at-scroll");
  if (!A.geo) return null;
  const r = scroll.getBoundingClientRect(), { dpr, gutter, cd } = A.geo;
  const x = (ev.clientX - r.left) * dpr - gutter, y = (ev.clientY - r.top + scroll.scrollTop) * dpr;
  if (x < 0) return null;
  const band = A.geo.layers.find((b) => y >= b.y && y < b.y + b.h);
  if (!band) return null;
  const ly = y - band.y;
  const tile = band.tiles.find((t) => x >= t.x && x < t.x + t.w && ly >= t.y && ly < t.y + t.h);
  if (!tile) return null;
  const j = Math.floor((ly - tile.y) / cd) * tile.cols + Math.floor((x - tile.x) / cd);
  return { band, tile, n: j < tile.members.length ? tile.members[j] : null };
}

const unitOf = (band, tile) => (band.L.assign && band.L.offset != null ? band.L.offset + tile.c : null);
function nodeName(band, tile) {
  if (band.L.assign) return `L${band.L.layer} · c${tile.c}`;  // as Trace names the cluster
  const a = tile.members[0], b = tile.members[tile.members.length - 1];
  return `L${band.L.layer} · N${a}–${b}`;
}

function tip(ev, h) {
  const el = $("#at-tip");
  if (!h) { el.hidden = true; return; }
  const m = current(), bytes = m?.bytes[String(h.band.L.layer)];
  const what = MAPS.find((x) => x.id === A.map)?.label.toLowerCase();
  let body;
  if (A.mode === "nodes" || h.n == null) {
    const v = bytes ? value(m, nodeByte(m, bytes, h.tile)) : null;
    body = `<b>${esc(nodeName(h.band, h.tile))}</b><span>${h.tile.members.length} neurons${bytes ? ` · ${what} ${fmt(v)}` : ""}</span>`;
  } else {
    body = `<b>${label(h.band.L.layer, h.n)}</b><span>${esc(nodeName(h.band, h.tile))}${bytes ? ` · ${what} ${fmt(value(m, bytes[h.n]))}` : ""}</span>`;
  }
  el.innerHTML = body;
  el.hidden = false;
  const stage = $("#at-stage").getBoundingClientRect();
  const x = ev.clientX - stage.left, y = ev.clientY - stage.top;
  el.style.left = `${Math.min(x + 14, stage.width - el.offsetWidth - 8)}px`;
  el.style.top = `${y + 16 + el.offsetHeight > stage.height ? y - el.offsetHeight - 10 : y + 16}px`;
}

// ---------------------------------------------------------------- selection
function selectNode(layer, c) {
  A.node = layer == null ? null : { layer, c };
  A.try = null;
  renderNode(); schedule();
}

/** A neuron opened here or anywhere else: ring it, open its node and draw its weight links. */
async function selectNeuron(name, { scroll = true } = {}) {
  const m = /^L(\d+)\s*N(\d+)$/i.exec(String(name || "").trim());
  if (!m || !A.geo) return;
  const layer = Number(m[1]), n = Number(m[2]), band = A.geo.byLayer.get(layer);
  if (!band || n >= band.L.width) return;
  A.sel = { layer, n };
  A.node = { layer, c: band.tiles[band.tileOf[n]].c };
  A.try = null;
  if (scroll) reveal(band.y + band.pos[2 * n + 1]);
  renderNode(); schedule();
  const key = label(layer, n);
  if (A.links?.neuron === key) return;
  A.links = null;
  try {
    const links = await api(`/llminspect/neurons/${encodeURIComponent(key)}/links`);
    if (A.sel && label(A.sel.layer, A.sel.n) === key) { A.links = links; renderNode(); schedule(); }
  } catch { /* the card says what went wrong */ }
}

/** Scroll the atlas so a content y (device px) is in view, gliding unless motion is reduced. */
function reveal(yDev) {
  const scroll = $("#at-scroll"), y = yDev / A.geo.dpr, h = scroll.clientHeight;
  if (y > scroll.scrollTop + 40 && y < scroll.scrollTop + h - 40) return;
  const smooth = !matchMedia("(prefers-reduced-motion: reduce)").matches;
  scroll.scrollTo({ top: Math.max(0, y - h * 0.4), behavior: smooth ? "smooth" : "auto" });
}

/** A Trace unit (a cluster's node id) to its layer and cluster here. */
function showUnit(g) {
  const L = A.layout?.layers.find((x) => x.assign && x.offset != null && g >= x.offset && g < x.offset + x.n_clusters);
  if (!L || !A.geo) return false;
  const band = A.geo.byLayer.get(L.layer), t = band.tiles.find((x) => x.c === g - L.offset);
  if (!t) return false;
  A.sel = null; A.links = null;
  selectNode(L.layer, t.c);
  reveal(band.y + t.y);
  return true;
}

// ---------------------------------------------------------------- the node panel
function renderNode() {
  const box = $("#at-node");
  if (!box) return;
  const band = A.node && A.geo?.byLayer.get(A.node.layer), t = band?.tiles.find((x) => x.c === A.node.c);
  if (!t) { box.hidden = true; return; }
  const m = current(), bytes = m?.bytes[String(band.L.layer)], g = unitOf(band, t);
  const what = MAPS.find((x) => x.id === A.map)?.label.toLowerCase();
  const ranked = bytes ? [...t.members].sort((a, b) => (m.signed ? Math.abs(bytes[b] - 127.5) - Math.abs(bytes[a] - 127.5) : bytes[b] - bytes[a])) : [...t.members];
  const sel = A.sel && A.sel.layer === band.L.layer && t.members.includes(A.sel.n) ? label(A.sel.layer, A.sel.n) : null;
  const linkRow = (title, list, col) => (list?.length ? `<div class="at-lk"><span class="at-lk-h" style="--c:${col}">${title}</span>
      ${list.slice(0, 6).map((l) => `<button type="button" class="nv-link" data-open-neuron="${l.neuron}">${l.neuron} <span>${l.cos >= 0 ? "+" : "−"}${Math.abs(l.cos).toFixed(2)}</span></button>`).join("")}</div>` : "");
  box.innerHTML = `
    <header><div><span class="eyebrow">${band.L.assign ? "Node · MLP cluster" : "Neurons"}</span><h3>${esc(nodeName(band, t))}</h3>
      <span class="muted small">${t.members.length} neurons${bytes ? ` · ${what} ${A.map === "changed" || A.map === "fires" ? "peak" : m.signed ? "most extreme" : "mean"} ${fmt(value(m, nodeByte(m, bytes, t)))}` : ""}</span></div>
      <button type="button" class="tbtn icon" data-close aria-label="Close the node">✕</button></header>
    <canvas class="at-zoom" id="at-zoom" aria-label="The node's neurons, larger: click one to open it"></canvas>
    <div class="at-sub">${A.map === "fires" ? "Fire most" : "Highest"} here</div>
    <div class="nv-links">${ranked.slice(0, 10).map((n) => `<button type="button" class="nv-link${sel === label(band.L.layer, n) ? " on" : ""}" data-open-neuron="${label(band.L.layer, n)}">N${n}${bytes ? ` <span>${fmt(value(m, bytes[n]))}</span>` : ""}</button>`).join("")}</div>
    ${sel ? `<div class="at-sel"><div class="at-sub">${esc(sel)} <span class="dim">direct weight links${A.links ? "" : ": reading…"}</span></div>
      ${linkRow("read by", A.links?.reads_it, A.colors.edgeOut)}${linkRow("reads", A.links?.it_reads, A.colors.edgeIn)}
      <div class="at-acts"><button type="button" class="btn sm" data-try ${A.trying ? "disabled" : ""} title="Answer the prompt under Steer and verify at dose −1, unsteered and +1">${A.trying ? "Answering…" : "Try dose −1 · 0 · +1"}</button>
        <button type="button" class="link" data-card>Card ↓</button></div>
      ${A.try?.error ? `<p class="small warn-text">${esc(A.try.error)}</p>` : A.try ? `<div class="at-try">${A.try.map((r) => `<div><b>${r.factor === 0 ? "0" : r.factor > 0 ? "+1" : "−1"}</b><p>${esc(r.text)}</p></div>`).join("")}</div>` : ""}</div>` : ""}
    <div class="at-acts">${g != null ? `<button type="button" class="btn sm" data-trace="${g}" title="Show this node in the Trace graph">Open in Trace</button>` : ""}</div>`;
  box.hidden = false;
  // Dock on the side away from what's selected, so the panel never covers it.
  const at = A.sel && A.sel.layer === band.L.layer ? band.pos[2 * A.sel.n] : t.x + t.w / 2;
  box.classList.toggle("left", at > A.geo.W * 0.5);
  drawZoom(band, t, bytes, m);
}

function drawZoom(band, t, bytes, m) {
  const cv = $("#at-zoom");
  if (!cv) return;
  const dpr = window.devicePixelRatio || 1, z = clamp(Math.floor(NODE_ZOOM / t.cols), 2, 14);
  const w = t.cols * z, h = t.rows * z;
  cv.style.width = `${w}px`; cv.style.height = `${h}px`;
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  cv.dataset.z = z;
  const ctx = cv.getContext("2d"), zd = z * dpr, inner = z >= 5 ? zd - dpr : zd;
  t.members.forEach((n, j) => {
    const [r, g, b] = bytes ? colorOf(m, bytes[n]) : A.colors.none;
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect((j % t.cols) * zd, Math.floor(j / t.cols) * zd, inner, inner);
    if (A.sel && A.sel.layer === band.L.layer && A.sel.n === n) {
      ctx.strokeStyle = A.colors.select; ctx.lineWidth = 2 * dpr;
      ctx.strokeRect((j % t.cols) * zd - dpr, Math.floor(j / t.cols) * zd - dpr, zd + dpr, zd + dpr);
    }
  });
}

function zoomHit(ev) {
  const cv = ev.target.closest("#at-zoom"), band = A.node && A.geo?.byLayer.get(A.node.layer), t = band?.tiles.find((x) => x.c === A.node.c);
  if (!cv || !t) return null;
  const r = cv.getBoundingClientRect(), z = Number(cv.dataset.z);
  const j = Math.floor((ev.clientY - r.top) / z) * t.cols + Math.floor((ev.clientX - r.left) / z);
  return j >= 0 && j < t.members.length ? label(band.L.layer, t.members[j]) : null;
}

async function tryDoses() {
  if (!A.sel || A.trying) return;
  const prompt = $("#nv-prompt")?.value.trim() || "Tell me about yourself in two sentences.";
  A.trying = true; renderNode();
  try {
    A.try = await post("/llminspect/neurons/sweep", { prompt, steers: [{ neuron: label(A.sel.layer, A.sel.n), dose: 1 }], factors: [-1, 0, 1], max_tokens: 40 });
  } catch (err) { A.try = { error: err.message }; }
  A.trying = false; renderNode();
}

// ---------------------------------------------------------------- the head, legend and state
function renderHead() {
  const maps = $("#at-maps");
  if (!maps) return;
  maps.innerHTML = MAPS.map((x) => `<button type="button" data-map="${x.id}" aria-pressed="${A.map === x.id}" title="${esc(x.hint)}"${x.id === "fires" && !A.fires ? " disabled" : ""}>${x.label}</button>`).join("");
  for (const b of $("#at-mode").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.mode === A.mode));
  $("#at-read").disabled = A.firing;
  $("#at-read").textContent = A.firing ? "Reading…" : "Read";
  const lay = A.layout, sub = $("#at-sub");
  if (lay) {
    const total = lay.layers.reduce((s, L) => s + L.width, 0), nodes = lay.layers.some((L) => L.assign);
    sub.textContent = `${total.toLocaleString()} neurons in ${lay.layers.length} layers${nodes ? ", packed by the nodes Trace draws" : ", in blocks of 256 until the weight anatomy is built"}${lay.edited ? " · edits applied" : ""}`;
  }
  renderLegend();
}

function renderLegend() {
  const el = $("#at-legend"), m = current();
  if (!el) return;
  if (!m || !A.colors) { el.innerHTML = ""; return; }
  const stops = (m.signed ? A.colors.div : A.colors.seq).filter((_, i) => i % 32 === 0 || i === 255).map(([r, g, b]) => `rgb(${r},${g},${b})`).join(",");
  const lo = m.signed ? -m.hi : m.lo, unit = A.map === "fires" ? "z " : "";
  el.innerHTML = `<span class="at-bar" style="background:linear-gradient(90deg,${stops})"></span>
    <span class="mono small">${unit}${fmt(lo)}${m.scale === "log" ? " ··log·· " : " — "}${fmt(m.hi)}</span>
    <span class="muted small">${esc(m.note || "")}${A.map === "fires" ? ` · ${A.fires.tokens} tokens` : ""}</span>`;
}

function renderState() {
  const el = $("#at-state");
  if (!el) return;
  el.className = `at-state${A.error ? " error" : ""}`;
  el.textContent = A.error ? A.error : A.loading && !A.layout ? "Reading every neuron…" : A.loading ? "Drawing the map…" : "";
  el.hidden = !el.textContent;
}

function renderFires() {
  const el = $("#at-fires");
  if (!el) return;
  if (!A.fires || A.map !== "fires") { el.innerHTML = ""; return; }
  el.innerHTML = `<span class="at-sub">Fire most on your text</span>${A.fires.top.slice(0, 16).map((f) => `<button type="button" class="nv-link" data-open-neuron="${f.neuron}" title="Peaks at “${esc(vis(f.token))}” after “${esc(f.before)}”">${f.neuron} <span>z ${f.z.toFixed(1)} · ${esc(vis(f.token))}</span></button>`).join("")}`;
}

// ---------------------------------------------------------------- wiring
function setZoom(cell, anchorY) {
  const scroll = $("#at-scroll"), old = A.geo;
  cell = clamp(Math.round(cell), 1, 16);
  if (!old || cell === A.cell) return;
  const yContent = (scroll.scrollTop + (anchorY ?? scroll.clientHeight / 2)) * old.dpr;
  const band = old.layers.find((b) => yContent < b.y + b.h + 30 * old.dpr) || old.layers.at(-1), frac = (yContent - band.y) / Math.max(1, band.h);
  A.cell = cell; A.zoomed = true;
  layoutNow();
  const nb = A.geo.byLayer.get(band.L.layer);
  scroll.scrollTop = (nb.y + frac * nb.h) / A.geo.dpr - (anchorY ?? scroll.clientHeight / 2);
}

function reset() {
  A.seq++;
  Object.assign(A, { layout: null, maps: {}, map: "write", cell: 0, zoomed: false, geo: null, sel: null, node: null, links: null, fires: null, try: null, hover: null, error: "" });
  A.bitmaps.clear();
  renderNode(); renderFires(); renderHead(); schedule();
}

function boot() {
  const scroll = $("#at-scroll");
  if (!scroll) return;
  readColors();
  renderHead();
  scroll.addEventListener("scroll", () => { A.hover = null; $("#at-tip").hidden = true; schedule(); }, { passive: true });
  scroll.addEventListener("pointermove", (ev) => {
    const h = hit(ev);
    const same = h && A.hover && h.band === A.hover.band && h.tile === A.hover.tile && h.n === A.hover.n;
    if (!same) { A.hover = h; schedule(); }
    scroll.style.cursor = h ? "pointer" : "";
    tip(ev, h);
  });
  scroll.addEventListener("pointerleave", () => { A.hover = null; $("#at-tip").hidden = true; schedule(); });
  scroll.addEventListener("click", (ev) => {
    const h = hit(ev);
    if (!h) return;  // a gap: Esc or ✕ closes the node
    if (A.mode === "nodes" || h.n == null) { A.sel = null; A.links = null; selectNode(h.band.L.layer, h.tile.c); }
    else document.dispatchEvent(new CustomEvent("llminspect:neuron", { detail: label(h.band.L.layer, h.n) }));
  });
  scroll.addEventListener("wheel", (ev) => {
    if (!ev.ctrlKey) return;
    ev.preventDefault();
    setZoom(A.cell * (ev.deltaY < 0 ? 1.25 : 0.8) + (ev.deltaY < 0 ? 0.5 : -0.5), ev.clientY - scroll.getBoundingClientRect().top);
  }, { passive: false });
  scroll.addEventListener("keydown", (ev) => {
    if (ev.key === "+" || ev.key === "=") { ev.preventDefault(); setZoom(A.cell + 1); }
    else if (ev.key === "-") { ev.preventDefault(); setZoom(A.cell - 1); }
    else if (ev.key === "Escape") { A.sel = null; A.links = null; selectNode(null); }
  });
  $("#at-zin").addEventListener("click", () => setZoom(A.cell + 1));
  $("#at-zout").addEventListener("click", () => setZoom(A.cell - 1));
  $("#at-maps").addEventListener("click", (ev) => { const b = ev.target.closest("[data-map]"); if (b && !b.disabled) { loadMap(b.dataset.map); renderFires(); } });
  $("#at-mode").addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-mode]");
    if (!b || b.dataset.mode === A.mode) return;
    A.mode = b.dataset.mode; A.bitmaps.clear(); renderHead(); schedule();
  });
  $("#at-fform").addEventListener("submit", (ev) => { ev.preventDefault(); readText($("#at-text").value); });
  $("#at-node").addEventListener("click", (ev) => {
    if (ev.target.closest("[data-close]")) { A.sel = null; A.links = null; selectNode(null); return; }
    if (ev.target.closest("[data-try]")) { tryDoses(); return; }
    if (ev.target.closest("[data-card]")) { $("#h-neuron")?.scrollIntoView({ behavior: "smooth", block: "start" }); return; }
    const tr = ev.target.closest("[data-trace]");
    if (tr) {
      document.dispatchEvent(new CustomEvent("llminspect:show", { detail: "weights" }));
      document.dispatchEvent(new CustomEvent("llminspect:trace", { detail: { unit: Number(tr.dataset.trace) } }));
      return;
    }
    const name = zoomHit(ev);
    if (name) document.dispatchEvent(new CustomEvent("llminspect:neuron", { detail: name }));
  });
  new ResizeObserver(() => { if (A.geo) layoutNow(); }).observe(scroll);
  const recolor = () => { readColors(); renderLegend(); renderNode(); schedule(); };
  new MutationObserver(recolor).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", recolor);
  // A neuron opened anywhere (ui/neurons.js opens its card) is ringed here too.
  document.addEventListener("llminspect:neuron", (ev) => { if (A.geo) selectNeuron(ev.detail); else A.pending = ev.detail; });
  document.addEventListener("llminspect:atlas", (ev) => {  // a node named in Trace
    const g = Number(ev.detail?.unit);
    if (!showUnit(g)) A.pendingUnit = g;
  });
  document.addEventListener("llminspect:view", async (ev) => {
    if (ev.detail !== "neurons") return;
    if (!A.layout) await loadLayout();
    else { requestAnimationFrame(layoutNow); if (A.map === "changed") loadMap("changed", { force: true }); }
    if (A.pending) { const p = A.pending; A.pending = null; selectNeuron(p); }
    if (A.pendingUnit != null) { const g = A.pendingUnit; A.pendingUnit = null; showUnit(g); }
  });
  document.addEventListener("llminspect:workbench", () => {  // edits changed: the changed map and the layout's flag are stale
    delete A.maps.changed;
    if (A.layout && !$("#view-neurons").hidden) loadLayout();
    else if (A.layout) A.layout = null;
  });
  document.addEventListener("llminspect:model", (ev) => {
    if (ev.detail?.name === A.model) return;
    A.model = ev.detail?.name ?? null;
    reset();
    if (!$("#view-neurons").hidden) loadLayout();
  });
  if (!$("#view-neurons").hidden) loadLayout();
}
boot();
