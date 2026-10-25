// Brainscan weights: the model as a live, zoomable graph. Nodes are units (attention heads and MLP
// neuron clusters) laid out along the residual stream; they glow by how much they wrote into the
// stream at the current token, and edges join units whose weights compose. A dock below follows the
// run token by token; clicking a node opens its weights, what it writes, and when it was most active.

import { WeightGraph } from "/ui/graph.js";
import { explainPaths, findPaths } from "/ui/paths.js";
import { defineBar, noteBar, refreshBar } from "/ui/actionbar.js";

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const visTok = (s) => (s ?? "").replace(/\n/g, "⏎").replace(/\t/g, "⇥").replace(/ /g, "·") || "∅";
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const fmtP = (p) => (p >= 0.001 || p === 0 ? p.toFixed(3) : p.toExponential(1));  // small probabilities stay readable
const num = (x, d = 3) => (x == null || Number.isNaN(x) ? "–"
  : x !== 0 && (Math.abs(x) >= 1e4 || Math.abs(x) < 1e-3) ? Number(x).toExponential(2) : Number(x).toFixed(d));
async function api(path, opts) {
  const res = await fetch(path, opts);
  const text = await res.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch { /* not JSON */ }
  if (!res.ok) { const err = new Error(body?.detail || text || `${res.status}`); err.status = res.status; throw err; }
  return body;
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}) });

const HIT_Z = 3;           // a unit counts as hit when it is this many SDs above its own run average
const Z_FULL = 4;          // z at full brightness
const SD_FLOOR = 4;        // in bytes (half a doubling), so very steady units don't light up on noise
const TRAIL_DECAY = 0.82;  // per token
const ENC = { lo: -16, steps: 8 };
const value = (code) => 2 ** (code / ENC.steps + ENC.lo);

const A = {
  atlas: null, U: 0, layers: [], units: [], kind: null, layerOf: null, idxOf: null, normT: null, edited: new Set(),
  runId: null, loading: null, buffer: [], tok: [], view: [], events: [], meta: null, skipped: 0,
  rows: new Map(), buf: new Uint8Array(0), n: 0, sum: null, sq: null, hist: new Float64Array(256),
  hitCache: new Map(), hitBase: 0, heat: null, lastAnat: 0,
  cursor: null, follow: true, playing: null, selected: null, trail: null, trailOn: true, mode: "rel", edgeMode: "circuit",
  colW: 4, scrollX: 0, fit: true, plotW: 600, traceHover: null,
  visible: false, raf: 0, detail: null, detailMode: null, statsTimer: 0, params: null,
  // tracing
  sel: new Set(),                    // multi-selected units
  attr: null, attrN: null,           // per-unit attribution of the next token at the cursor (raw, scaled to [-1, 1])
  attrInfo: null, attrTarget: null, attrSeq: 0, attrTimer: 0,
  cmp: false, attrOrig: null, attrDiff: null,  // "vs original": attribution with the weight edits lifted, and the change
  attrOf: null,                      // the slot of the word being explained, when it was clicked in the conversation
  trace: null, traceOn: true, frameTrace: false, followTimer: 0,  // the explained token's routes, START to END
  editsOpen: false, applied: 0,     // weight edits applied now (from ui/edits.js)
  panelSeq: 0, knockouts: [], koKey: null,
  paths: [], pathMode: "down", pathHops: 3, pathBranch: 4, pathActivity: true, pathsOpen: false,
  expanded: new Map(),               // cluster g -> Float32Array neuron activity in [-1, 1] at the cursor
  tokenTimer: 0, tokenSeq: 0,
};
const H = { base: null, tile: null, vals: null, stack: [], mode: "auto", drag: null, lo: 0, scale: 1, seq: 0 };
let graph = null;

// ---------------------------------------------------------------- colors
let C = {};
function hexToRgb(h) { h = h.replace("#", ""); return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)); }
function ramp(stops) {
  const rgb = stops.map(hexToRgb), out = [];
  for (let i = 0; i < 256; i++) {
    const x = (i / 255) * (rgb.length - 1), k = Math.min(Math.floor(x), rgb.length - 2), f = x - k;
    out.push(rgb[k].map((v, j) => Math.round(v + (rgb[k + 1][j] - v) * f)));
  }
  return out;
}
const list = (name) => cssVar(name).split(",").map((s) => s.trim());
function readColors() {
  const v = cssVar;
  C = {
    dark: v("--scheme-dark") === "1",
    surface: v("--surface"), raised: v("--raised"), ink: v("--ink"), ink2: v("--ink-2"), muted: v("--muted"), grid: v("--grid"),
    axis: v("--axis"), focus: v("--focus"), serious: v("--serious"), critical: v("--critical"), good: v("--good"),
    noData: v("--no-data"), s1: v("--series-1"), s2: v("--series-2"),
    roles: { system: v("--role-system"), user: v("--role-user"), assistant: v("--role-assistant"), reasoning: v("--role-reasoning"), tool: v("--role-tool"), template: v("--role-template") },
    rampStops: list("--ramp"),
    neg: hexToRgb(v("--div-neg")), pos: hexToRgb(v("--div-pos")), mid: hexToRgb(v("--surface")),
    // graph
    nodeNoData: v("--node-nodata"), gridDot: v("--grid-dot"), podFill: v("--pod-fill"), podStroke: v("--pod-stroke"),
    busA: v("--bus-a"), busB: v("--bus-b"), busFlow: v("--bus-flow"), writeEdge: v("--write-edge"),
    edgeIn: v("--edge-in"), edgeOut: v("--edge-out"), edgeCircuit: v("--edge-circuit"), edgeCircuitHot: v("--edge-circuit-hot"),
    glow: v("--glow"), chipBg: v("--chip-bg"), cardBg: v("--card-bg"), headRing: v("--head-ring"), trail: v("--trail"),
    edited: v("--serious"), hoverRing: v("--hover-ring"), selectRing: v("--select-ring"), selectHalo: v("--select-halo"),
    path: v("--path"), pathFocus: v("--path-focus"), multiSelect: v("--multi-select"), multiSelectFill: v("--multi-select-fill"),
  };
  C.ramp = ramp(C.rampStops);
  C.graphStops = list("--graph-ramp");
  C.graphRamp = ramp(C.graphStops).map(([r, g, b]) => `rgb(${r},${g},${b})`);
  // Attribution: pushes the next token down (cool) -> no effect (idle node color) -> pushes it up (warm).
  C.attrStops = [v("--div-neg"), C.graphStops[0], v("--div-pos")];
  C.attrRamp = ramp(C.attrStops).map(([r, g, b]) => `rgb(${r},${g},${b})`);
  C.div = Array.from({ length: 256 }, (_, i) => {
    const x = i / 127.5 - 1, end = x < 0 ? C.neg : C.pos, f = Math.abs(x);
    return [0, 1, 2].map((k) => Math.round(C.mid[k] + (end[k] - C.mid[k]) * f));
  });
  graph?.setColors(C);
}

// ---------------------------------------------------------------- half floats and bytes from the server
const HALF = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    t[h] = e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024);
  }
  return t;
})();
function b64Bytes(b64) { const bin = atob(b64), out = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i); return out; }
function b64Halves(b64) { const u8 = b64Bytes(b64), u16 = new Uint16Array(u8.buffer, 0, u8.length >> 1); return Float32Array.from(u16, (h) => HALF[h]); }

// ---------------------------------------------------------------- the unit map and its edges
async function loadAtlas() {
  if (A.atlas) return true;
  if (A._loadingAtlas) return A._loadingAtlas;  // single-flight: a retry + a status frame must not double-load
  A._loadingAtlas = (async () => {
   try {
    const atlas = await api("/brainscan/anatomy");
    A.atlas = atlas; A.U = atlas.n_units; A.layers = atlas.layers; A.units = atlas.units;
    ENC.lo = atlas.encoding.lo; ENC.steps = atlas.encoding.steps;
    A.kind = new Uint8Array(A.U); A.layerOf = new Uint16Array(A.U); A.idxOf = new Uint16Array(A.U);
    for (const u of A.units) { A.kind[u.g] = u.kind === "head" ? 0 : 1; A.layerOf[u.g] = u.layer; A.idxOf[u.g] = u.i; }
    // weight size: log norm, scaled within heads and within clusters
    A.normT = new Float32Array(A.U);
    for (const k of [0, 1]) {
      const logs = A.units.filter((u) => A.kind[u.g] === k).map((u) => Math.log(Math.max(u.norm, 1e-9)));
      const lo = Math.min(...logs), hi = Math.max(...logs);
      for (const u of A.units) if (A.kind[u.g] === k) A.normT[u.g] = hi > lo ? (Math.log(Math.max(u.norm, 1e-9)) - lo) / (hi - lo) : 0.5;
    }
    A.edited = new Set(atlas.edited || []);
    A.trail = new Float32Array(A.U); A.heat = new Float32Array(A.U).fill(NaN);
    graph.setAtlas(atlas);
    pushState();
    renderLegend();
    api("/brainscan/anatomy/edges").then((e) => {
      if (e.available) { e.up = Int32Array.from(e.up); e.down = Int32Array.from(e.down); e.up_x = Float32Array.from(e.up_x); e.down_x = Float32Array.from(e.down_x); }
      A.edges = e; graph.setEdges(e); renderLegend();
      if (A.pathsOpen) runPathSearch(true);
      else updateTrace();  // a word clicked before the edges arrived
    }).catch(() => {});
    return true;
   } catch (err) {
    A.atlasError = err;
    return false;
   } finally {
    A._loadingAtlas = null;
   }
  })();
  return A._loadingAtlas;
}

async function refreshEdited() {
  try { A.edited = new Set((await api("/brainscan/anatomy/edited")).edited); } catch { /* recorder busy */ }
  resetThumbs();
  pushState();
}

// ---------------------------------------------------------------- run data
function resetRunData() {
  A.tok = []; A.view = []; A.events = []; A.rows = new Map(); A.n = 0; A.buf = new Uint8Array(0);
  A.sum = new Float64Array(A.U); A.sq = new Float64Array(A.U); A.hist = new Float64Array(256);
  A.hitCache = new Map(); A.hitBase = 0; A.cursor = null; A.skipped = 0; A.statN = -1; A.meta = null;
  if (A.trail) A.trail.fill(0);
  // Token-level tracing belongs to the old run; the selection and path settings carry over.
  A.attr = null; A.attrN = null; A.attrInfo = null; A.attrTarget = null; A.attrOrig = null; A.attrDiff = null;
  A.attrSeq++; clearTimeout(A.attrTimer);
  A.attrOf = null; A.knockouts = []; A.koKey = null;
  A.expanded.clear();
}

function ensureRows(n) {
  if (A.buf.length >= n * A.U) return;
  const next = new Uint8Array(Math.max(n, Math.ceil(A.buf.length / A.U * 1.6) + 64) * A.U);
  next.set(A.buf.subarray(0, A.n * A.U));
  A.buf = next;
}

function addSlice(ts, bytes) {
  ensureRows(A.n + ts.length);
  const U = A.U;
  for (let k = 0; k < ts.length; k++) {
    const t = ts[k], src = bytes.subarray(k * U, (k + 1) * U);
    let row = A.rows.get(t);
    if (row == null) { row = A.n++; A.rows.set(t, row); } else {
      const old = A.buf.subarray(row * U, (row + 1) * U);
      for (let g = 0; g < U; g++) { A.sum[g] -= old[g]; A.sq[g] -= old[g] * old[g]; A.hist[old[g]]--; }
    }
    A.buf.set(src, row * U);
    for (let g = 0; g < U; g++) { const c = src[g]; A.sum[g] += c; A.sq[g] += c * c; A.hist[c]++; }
  }
  if (A.n > A.hitBase * 1.25 + 8) { A.hitCache.clear(); A.hitBase = A.n; }  // averages moved: recount hits
  updateAbsRange();
}

function updateAbsRange() {
  let total = 0; for (let i = 0; i < 256; i++) total += A.hist[i];
  let acc = 0, lo = 0, hi = 255;
  for (let i = 0; i < 256; i++) { acc += A.hist[i]; if (acc >= total * 0.02) { lo = i; break; } }
  acc = 0;
  for (let i = 255; i >= 0; i--) { acc += A.hist[i]; if (acc >= total * 0.005) { hi = i; break; } }
  A.absLo = lo; A.absHi = Math.max(hi, lo + 1);
}

function unitStats() {
  if (A.statN === A.n && A.mean) return;
  A.mean = new Float32Array(A.U); A.isd = new Float32Array(A.U);
  for (let g = 0; g < A.U; g++) {
    const m = A.sum[g] / Math.max(A.n, 1), v = Math.max(A.sq[g] / Math.max(A.n, 1) - m * m, 0);
    A.mean[g] = m; A.isd[g] = 1 / Math.max(Math.sqrt(v), SD_FLOOR);
  }
  A.statN = A.n;
}
function zOf(g, code) {
  if (A.statN !== A.n) unitStats();
  return (code - A.mean[g]) * A.isd[g];
}
const rowOf = (t) => (t == null ? null : A.rows.get(t) ?? null);
const codeAt = (row, g) => A.buf[row * A.U + g];

function hitCount(t) {
  let c = A.hitCache.get(t);
  if (c != null) return c;
  const row = rowOf(t); if (row == null) return null;
  c = 0;
  for (let g = 0; g < A.U; g++) if (zOf(g, codeAt(row, g)) >= HIT_Z) c++;
  A.hitCache.set(t, c);
  return c;
}

function rebuildView() {
  A.view = [];
  for (const r of A.tok) if (r && r.ph !== "replay") A.view.push(r.t);
}

async function openRun(id) {
  if (!A.atlas) return;  // activate() opens the selected run once the unit map is loaded
  A.runId = id || null; A.loading = id || null; A.buffer = [];
  stopPlay();
  resetRunData();
  schedule();
  if (!id) { A.loading = null; renderConvo(true); setCursor(null); return; }
  try {
    const [run, tape, bin] = await Promise.all([
      api(`/brainscan/runs/${encodeURIComponent(id)}`),
      api(`/brainscan/runs/${encodeURIComponent(id)}/tape`),
      fetch(`/brainscan/runs/${encodeURIComponent(id)}/anatomy`).then((r) => (r.ok ? r.arrayBuffer() : null)),
    ]);
    if (A.runId !== id) return;
    A.meta = run.meta; A.events = run.events || [];
    for (const r of tape) A.tok[r.t] = { t: r.t, tx: r.tx, r: r.r, ph: r.ph, pos: r.pos, req: r.req, p: r.p };
    rebuildView();
    if (bin) {
      const dv = new DataView(bin), hlen = dv.getUint32(0, true);
      const head = JSON.parse(new TextDecoder().decode(new Uint8Array(bin, 4, hlen)));
      A.skipped = head.skipped;
      if (head.units === A.U && head.n) addSlice(head.t, new Uint8Array(bin, 4 + hlen, head.n * head.units));
    }
  } catch (err) {
    console.warn("weights: could not load run", err);
  } finally {
    if (A.loading === id) {
      const buffered = A.buffer; A.loading = null; A.buffer = [];
      buffered.forEach(handle);
    }
  }
  renderConvo(true);
  setCursor(lastRecorded(), "jump");
  if (A.selected != null) renderRunStats();
  if (A.pendingWord != null && A.runId === id) { const w = A.pendingWord; A.pendingWord = null; if (A.tok[w]) traceWord(w); }
}

function lastRecorded() {
  for (let i = A.view.length - 1; i >= 0; i--) if (A.rows.has(A.view[i])) return A.view[i];
  return null;
}

// ---------------------------------------------------------------- live feed
function handle(msg) {
  if (A.loading && msg.run === A.loading && ["toks", "anat", "ev"].includes(msg.k)) { A.buffer.push(msg); return; }
  switch (msg.k) {
    case "toks":
      if (msg.run !== A.runId) break;
      {
        let reordered = false;
        for (const r of msg.items) {
          A.tok[r.t] = { t: r.t, tx: r.tx, r: r.r, ph: r.ph, pos: r.pos, req: r.req, p: r.p };
          if (r.ph !== "replay") { if (!A.view.length || r.t > A.view[A.view.length - 1]) A.view.push(r.t); else { rebuildView(); reordered = true; } }
        }
        renderConvo(reordered);
      }
      schedule();
      break;
    case "anat": {
      if (msg.run !== A.runId || !A.atlas || msg.u !== A.U) break;
      addSlice(msg.t, b64Bytes(msg.b64));
      A.lastAnat = performance.now();
      if (!A.visible) { A.pendingCursor = true; break; }  // keep the data; skip redraw/heat recompute while hidden
      if (A.follow && !A.playing) setCursor(msg.t[msg.t.length - 1], msg.t.length === 1 ? "step" : "jump");
      else schedule();
      if (A.selected != null) throttledRunStats();
      break;
    }
    case "ev":
      if (msg.run === A.runId && !A.events.some((e) => e.ts === msg.ev.ts && e.type === msg.ev.type)) { A.events.push(msg.ev); schedule(); }
      break;
    case "workbench":
      refreshEdited();
      if (A.selected != null && A.visible) selectUnit(A.selected, true);  // the weights changed
      if (A.mode === "attr" && A.visible) loadExplain();  // hidden: activate() re-traces on return
      document.dispatchEvent(new CustomEvent("brainscan:workbench"));  // the edit panels (ui/edits.js) refresh
      break;
    case "status":
      if (!A.atlas && A.visible) activate();
      break;
  }
}

function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/brainscan/live`);
  ws.onclose = () => setTimeout(connect, 1500);
  ws.onmessage = (ev) => { try { for (const msg of JSON.parse(ev.data)) handle(msg); } catch (err) { console.warn("weights: bad live frame", err); } };
}

// ---------------------------------------------------------------- cursor, heat, trail, replay
function computeHeat() {
  const h = A.heat; if (!h) return;
  const row = rowOf(A.cursor);
  if (A.mode === "norm") { h.set(A.normT); return; }
  if (A.mode === "attr") {  // glow by how strongly a unit moved the next token either way
    if (A.attrN && A.attrInfo?.t === A.cursor) for (let g = 0; g < A.U; g++) h[g] = Math.abs(A.attrN[g]);
    else h.fill(NaN);
    return;
  }
  if (row == null) { h.fill(NaN); return; }
  for (let g = 0; g < A.U; g++) {
    const c = codeAt(row, g);
    h[g] = A.mode === "abs" ? clamp((c - A.absLo) / (A.absHi - A.absLo), 0, 1) : clamp(zOf(g, c) / Z_FULL, 0, 1);
  }
}

/** Activity of a unit at the cursor in [0, 1] (vs. its run average), whatever the color mode shows. */
function activityAt(g) {
  const row = rowOf(A.cursor);
  return row == null ? 0 : clamp(zOf(g, codeAt(row, g)) / Z_FULL, 0, 1);
}

function pushState() {
  if (!graph || !A.atlas) return;
  computeHeat();
  const signed = A.mode === "attr" && A.attrN && A.attrInfo?.t === A.cursor ? A.attrN : null;
  graph.setState({ heat: A.heat, trail: A.trail, trailOn: A.trailOn && A.mode !== "attr", mode: A.mode, normT: A.normT,
    selected: A.selected, edited: A.edited, edgeMode: A.edgeMode, live: isLive(), visible: A.visible,
    signed, selection: A.sel, paths: A.paths, expanded: A.expanded,
    trace: A.trace && !A.pathsOpen ? { start: A.trace.start, end: A.trace.end } : null,
    insets: { l: A.pathsOpen || A.editsOpen ? 396 : 0, r: $("#gv-drawer").hidden ? 0 : 452 } });
  refreshBar("weights");
}
const isLive = () => A.follow && performance.now() - A.lastAnat < 2500;

function setCursor(t, how = "jump") {
  A.cursor = t;
  const row = rowOf(t);
  if (A.trail) {
    if (row == null) { if (how === "jump") A.trail.fill(0); }
    else for (let g = 0; g < A.U; g++) {
      const hot = clamp(zOf(g, codeAt(row, g)) / Z_FULL, 0, 1);
      A.trail[g] = how === "step" ? Math.max(A.trail[g] * TRAIL_DECAY, hot) : hot;
    }
  }
  if (t != null && !A.fit) {  // keep the cursor on screen
    const i = viewIndex(t), x = i * A.colW;
    if (x < A.scrollX || x > A.scrollX + A.plotW - A.colW) A.scrollX = x - A.plotW * 0.8;
  }
  if (A.attrInfo?.t !== t) { A.attrTarget = null; A.attrOf = null; }  // a new token explains its own next token by default
  if (A.trace && A.trace.t !== t) clearTrace();  // redrawn when this token's explanation arrives
  pushState();
  renderReadout();
  schedule();
  scheduleTokenWork();
  markConvo(true);
}

function setFollow(on) {
  A.follow = on;
  $("#an-follow").setAttribute("aria-pressed", String(on));
  if (on) { stopPlay(); setCursor(lastRecorded(), "jump"); }
  renderReadout();
}

function stopPlay() {
  if (A.playing) { clearInterval(A.playing); A.playing = null; }
  const b = $("#an-play"); b.innerHTML = ICON.play; b.setAttribute("aria-label", "Replay the run");
  scheduleTokenWork();  // tracing waits for replay to stop
}

function togglePlay() {
  if (A.playing) { stopPlay(); return; }
  const seq = A.view.filter((t) => A.rows.has(t));
  if (!seq.length) return;
  setFollow(false);
  let i = A.cursor != null ? seq.indexOf(A.cursor) : -1;
  if (i < 0 || i >= seq.length - 1) { i = 0; setCursor(seq[0], "jump"); }
  const b = $("#an-play"); b.innerHTML = ICON.pause; b.setAttribute("aria-label", "Pause");
  A.playing = setInterval(() => {
    i++;
    if (i >= seq.length) { stopPlay(); return; }
    setCursor(seq[i], "step");
  }, 55);
}

function step(d) {
  if (A.cursor == null || !A.view.length) return;
  const i = viewIndex(A.cursor) + d;
  if (i >= 0 && i < A.view.length) { setFollow(false); stopPlay(); setCursor(A.view[i], "step"); }
}

function renderReadout() {
  const el = $("#an-readout");
  if (!A.atlas) { el.textContent = ""; return; }
  $("#an-follow").classList.toggle("streaming", isLive());
  if (A.mode === "norm") { el.innerHTML = `<span class="muted">Weight size: each unit's output-weight norm, log scale. No run needed.</span>`; return; }
  const t = A.cursor, rec = t != null ? A.tok[t] : null;
  if (rec == null) {
    const why = !A.runId ? `No run yet. Point any harness at <code>${esc(location.origin)}/v1</code>; the graph lights up as tokens arrive.`
      : A.skipped ? "This run was recorded with a different unit map (another model or cluster count)."
        : A.view.length ? "No unit activity in this run: it was recorded before tracing was added. New requests record it."
          : "No tokens in this run yet.";
    el.innerHTML = `<span class="muted">${why}</span>`;
    return;
  }
  const hits = hitCount(t);
  let html = `<span class="tok-chip">${esc(visTok(rec.tx))}</span><span class="meta">t${t} · ${esc(rec.r)}</span>`;
  if (A.mode !== "attr") {
    html += hits == null ? `<span class="muted">no activity stored for this token</span>` : `<span class="hits"><b>${hits}</b> units hit</span>`;
    html += `<button type="button" class="link" id="an-explain-btn" title="Color the graph by what each unit did to the next token">Explain next token</button>`;
  } else {
    const info = A.attrInfo;
    if (A.playing) html += `<span class="muted">Pause to trace this token</span>`;
    else if (!info || info.t !== t || info.loading) html += `<span class="muted">Tracing the next token…</span>`;
    else if (info.busy) html += `<span class="muted">The model is generating; tracing when it pauses…</span>`;
    else if (info.error) html += `<span class="warn-text">${esc(info.error)}</span>`;
    else {
      const o = info.orig, sign = (x) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(2);
      html += `<span class="arrow">→</span><span class="tok-chip target">${esc(visTok(info.target.text))}</span>
        ${o ? `<span class="meta" title="original weights → edited weights">p ${fmtP(o.target.prob)} → ${fmtP(info.target.prob)} · logit ${sign(info.target.logit - o.logit)}</span>`
          : `<span class="meta">p ${fmtP(info.target.prob)} · logit ${info.target.logit.toFixed(2)}</span>`}
        <span class="alts">${info.alternatives.filter((a) => a.id !== info.target.id).slice(0, 4).map((a) =>
          `<button type="button" class="alt" data-id="${a.id}" title="Explain this candidate instead (p ${fmtP(a.prob)})">${esc(visTok(a.text))}</button>`).join("")}</span>`;
      html += `<button type="button" class="tbtn cmp" id="an-why" title="Layer by layer, the units behind this token, and knockout tests">Why?</button>`;
      if (info.weights_edited) {
        html += `<button type="button" class="tbtn cmp" id="an-cmp" aria-pressed="${A.cmp}" title="Color units by how the applied weight edits changed their effect on this token">vs original</button>`;
      }
      if (!info.weights_match && !o) html += `<span class="warn-text" title="Tracing uses the current weights">weights differ from the recording</span>`;
    }
  }
  el.innerHTML = html;
  $("#an-explain-btn")?.addEventListener("click", () => setMode("attr"));
  $("#an-cmp")?.addEventListener("click", toggleCompare);
  $("#an-why")?.addEventListener("click", openTokenPanel);
  el.querySelectorAll(".alt[data-id]").forEach((b) => { b.onclick = () => { A.attrTarget = b.dataset.id; loadExplain(); }; });
}

// ---------------------------------------------------------------- tracing: attribution, selection, paths
function setMode(mode) {
  A.mode = mode;
  setSeg("#gv-mode", "mode", mode);
  if (mode !== "attr") clearTrace();
  pushState(); renderLegend(); renderReadout();
  if (mode === "attr") loadExplain();
}

// ---------------------------------------------------------------- the trace: START (input token) → units → END (output token)
/** For each of the units that pushed the explained token up the most (or, comparing with the
 * original weights, that the edits changed most), its strongest route in from earlier layers
 * through units active at this token. Drawn on the graph from the embedding (START, the input
 * token) to the unembedding (END, the output token), unless the path explorer is showing its own. */
function updateTrace() {
  const info = A.attrInfo;
  const want = A.mode === "attr" && A.traceOn && A.attr && info?.target && info.t === A.cursor && A.edges?.available;
  if (!want) { clearTrace(); return; }
  const scores = A.attrDiff || A.attr, cmp = !!A.attrDiff;
  const roots = [...scores.keys()].filter((g) => (cmp ? Math.abs(scores[g]) > 1e-6 : scores[g] > 0))
    .sort((a, b) => (cmp ? Math.abs(scores[b]) - Math.abs(scores[a]) : scores[b] - scores[a])).slice(0, 5);
  const weight = rowOf(A.cursor) != null ? (g) => 0.15 + 0.85 * activityAt(g) : null;
  const routes = roots.map((g) => {
    const best = findPaths(A.edges, [g], { dir: "up", hops: 3, branch: 3, weight, limit: 1 })[0];
    return { units: best ? best.units : [g], xs: best ? best.xs : [], score: Math.abs(scores[g]), root: g, value: scores[g], ends: true };
  });
  stopFollow();
  A.trace = { t: info.t, start: visTok(A.tok[info.t]?.tx ?? info.token), end: visTok(info.target.text), routes, cmp };
  if (!A.pathsOpen) {
    A.traceFocus = 0;  // one route at a time reads clearly; the others stay faint
    A.paths = routes.map((r, i) => ({ ...r, on: true, focus: i === 0 }));
    if (A.frameTrace) { A.frameTrace = false; requestAnimationFrame(() => graph.fitTrace(routeUnits())); }
  }
  pathsChanged();
}

const routeUnits = () => [...new Set((A.trace?.routes || []).flatMap((r) => r.units))];

function clearTrace() {
  stopFollow();
  if (!A.trace) return;
  A.trace = null;
  if (!A.pathsOpen) A.paths = [];
  pathsChanged();
}

/** Light up route i (hover); `pin` keeps it lit; un-hovering goes back to the pinned route. */
function focusRoute(i, on, pin = false) {
  if (A.pathsOpen || !A.trace) return;
  if (pin) A.traceFocus = i;
  const lit = on ? i : A.traceFocus;
  A.paths.forEach((p, j) => { p.focus = j === lit; });
  document.querySelectorAll(".routes li[data-i]").forEach((li) => li.classList.toggle("pinned", Number(li.dataset.i) === A.traceFocus));
  pushState();
}

function stopFollow() {
  clearTimeout(A.followTimer); A.followTimer = 0;
  document.querySelectorAll(".routes .now").forEach((el) => el.classList.remove("now"));
}

/** Walk a route step by step: START, each unit in order, END. */
function followRoute(i) {
  stopFollow();
  const r = A.trace?.routes[i];
  if (!r) return;
  focusRoute(i, true, true);
  const steps = ["start", ...r.units, "end"];
  const li = document.querySelector(`.routes li[data-i="${i}"]`);
  const marks = li ? [...li.querySelectorAll("[data-step]")] : [];
  let k = 0;
  const next = () => {
    marks.forEach((el) => el.classList.toggle("now", Number(el.dataset.step) === k));
    if (k >= steps.length) { A.followTimer = 0; return; }
    const st = steps[k];
    if (st === "start") graph.flyToEnd(0);
    else if (st === "end") graph.flyToEnd(1);
    else graph.flyTo(st, 2.2);
    k++;
    A.followTimer = setTimeout(next, 1200);
  };
  next();
}

function routesHtml(info) {
  if (!A.traceOn) return `<section class="d-sec"><h4>The trace</h4><p class="hint">Hidden. <button type="button" class="link" id="tp-trace-on">Draw it on the graph</button></p></section>`;
  const tr = A.trace;
  if (!tr || tr.t !== info.t) return A.edges?.available === false ? "" : `<section class="d-sec"><h4>The trace, start to end</h4><p class="hint">Mapping the routes…</p></section>`;
  const sgn = (x) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(2)}`;
  const rows = tr.routes.map((r, i) => {
    let step = 0;
    const units = r.units.map((g) => `<span class="rt-arrow">→</span><span class="pchip ${A.units[g].kind}${g === r.root ? " root" : ""}" data-g="${g}" data-step="${++step}" title="${esc(unitName(g))}${g === r.root ? " · writes to the output directly" : ""}">${shortName(g)}</span>`).join("");
    return `<li data-i="${i}"><span class="rt-n">${i + 1}</span><div class="rt-flow">
        <span class="rt-end start" data-step="0" title="The input token at this position. Every unit reads it from the residual stream (dashed on the graph).">START <code>${esc(tr.start)}</code></span>
        ${units}<span class="rt-arrow">→</span>
        <span class="rt-end end" data-step="${step + 1}" title="The output token: the last unit's direct write to it">END <code>${esc(tr.end)}</code></span>
        <b class="rt-val ${r.value >= 0 ? "pos" : "neg"}" title="${tr.cmp ? "Change in the last unit's direct effect" : "The last unit's direct effect on the output token"}">${tr.cmp ? "Δ" : ""}${sgn(r.value)}</b></div></li>`;
  }).join("");
  return `<section class="d-sec"><h4>The trace, start to end <span class="dim">${tr.cmp ? "routes behind the change" : `${tr.routes.length} strongest routes`}</span></h4>
    <ol class="routes">${rows || `<li class="muted">No unit pushed this token up.</li>`}</ol>
    <div class="d-actions wrap">
      ${tr.routes.length ? `<button type="button" class="btn sm" id="tp-follow" title="Walk route 1 on the graph: START, each unit, END">▶ Follow route 1</button>` : ""}
      <button type="button" class="btn sm ghost" id="tp-frame">Frame the trace</button>
      <button type="button" class="link" id="tp-trace-off">Hide trace</button>
    </div>
    <p class="hint">Read left to right. A route starts at the input token, which every unit reads from the residual stream (dashed on the graph); each hop is a unit whose input weights read the previous unit's output, favoring units active at this token; it ends with the last unit's direct write to the output token. Hover a route to light it up on the graph, click a unit to open it.</p></section>`;
}

/** Once the cursor settles: refresh the attribution, the open unit's token details and expanded clusters. */
function scheduleTokenWork() {
  clearTimeout(A.tokenTimer);
  if (A.playing || A.cursor == null || !A.runId) return;
  A.tokenTimer = setTimeout(() => {
    if (A.mode === "attr") loadExplain();
    if (A.detailMode === "unit" && A.selected != null) renderTokenSection();
    refreshExpanded();
    if (A.pathsOpen && (A.pathMode === "explain" || A.pathActivity)) runPathSearch(false);
  }, 260);
}

const tokenUrl = (t) => `/brainscan/runs/${encodeURIComponent(A.runId)}/tokens/${t}`;
function fetchExplain(t, target, weights) {
  const q = new URLSearchParams();
  if (target != null) q.set("target", target);
  if (weights) q.set("weights", weights);
  const qs = q.toString();
  return api(`${tokenUrl(t)}/explain${qs ? `?${qs}` : ""}`);
}

/** Scale to [-1, 1] on a square-root scale, so mid-sized effects stay visible; returns [scaled, max]. */
function sqrtScale(arr) {
  let m = 1e-9;
  for (const x of arr) m = Math.max(m, Math.abs(x));
  return [arr.map((x) => Math.sign(x) * Math.sqrt(Math.abs(x) / m)), m];
}

/** What the graph colors and the next-token paths start from: the attribution, or its change. */
const attrView = () => A.attrDiff || A.attr;

async function loadExplain() {
  const t = A.cursor, seq = ++A.attrSeq, target = A.attrTarget;
  clearTimeout(A.attrTimer);
  if (t == null || !A.runId) return;
  A.attrInfo = { loading: true, t };
  renderReadout();
  renderTokenPanel();
  try {
    const ex = await fetchExplain(t, target);
    if (seq !== A.attrSeq) return;
    // Before/after: the same target token, traced again with the applied weight edits lifted.
    const orig = A.cmp && ex.weights_edited ? await fetchExplain(t, ex.target.id, "original") : null;
    if (seq !== A.attrSeq) return;
    A.attr = Float32Array.from(ex.contrib);
    A.attrOrig = orig ? Float32Array.from(orig.contrib) : null;
    A.attrDiff = orig ? A.attr.map((x, g) => x - A.attrOrig[g]) : null;
    const [scaled, m] = sqrtScale(attrView());
    A.attrN = scaled;
    A.attrInfo = { ...ex, t, max: m, orig: orig && { target: orig.target, logit: orig.logit } };
  } catch (err) {
    if (seq !== A.attrSeq) return;
    A.attrInfo = err.status === 409 ? { busy: true, t } : { error: err.message, t };
    if (err.status === 409) A.attrTimer = setTimeout(() => { if (A.cursor === t) loadExplain(); }, 1500);
  }
  updateTrace();
  pushState(); renderReadout(); renderLegend(); renderTokenPanel(); markConvo(false);
  if (A.pathsOpen) renderPathControls();
  if (A.pathsOpen && A.pathMode === "explain") runPathSearch(false);
}

function toggleCompare() {
  A.cmp = !A.cmp;
  loadExplain();
  if (A.detailMode === "unit" && A.selected != null) renderTokenSection();
}

const shortName = (g) => { const u = A.units[g]; return `L${u.layer} ${u.kind === "head" ? "h" : "c"}${u.i}`; };

function toggleSelect(g) { if (A.sel.has(g)) A.sel.delete(g); else A.sel.add(g); afterSelection(); }
function boxSelect(units, remove) { for (const g of units) { if (remove) A.sel.delete(g); else A.sel.add(g); } afterSelection(); }
function clearSelection() { A.sel.clear(); afterSelection(); }
function afterSelection() {
  pushState(); renderSelChip();
  if (A.pathsOpen && A.pathMode !== "explain") runPathSearch(false);
}
function renderSelChip() {
  const n = A.sel.size;
  $("#gv-selchip").hidden = !n;
  if (n) $("#gv-selcount").textContent = `${n} selected`;
}

const PATH_MODES = {
  down: "Strongest routes out of the selected units into later layers. Each hop is a unit whose inputs read the previous unit's outputs.",
  up: "Strongest routes into the selected units from earlier layers: what feeds them.",
  between: "Routes that start at one selected unit and end at another, through any units in between.",
  explain: "The circuit behind the next token: the units that pushed it up the most, and the active units feeding them.",
};

function openPaths(mode) {
  A.pathsOpen = true;
  if (mode) A.pathMode = mode;
  else if (A.mode === "attr" && !pathSources().length) A.pathMode = "explain";  // nothing selected: explain the token
  $("#gv-paths").hidden = false;
  $("#gv-stage").classList.add("paths-open");
  $("#gv-paths-btn").setAttribute("aria-pressed", "true");
  renderPathControls();
  runPathSearch(true);
}
function closePaths() {
  A.pathsOpen = false;
  A.paths = A.trace ? A.trace.routes.map((r) => ({ ...r, on: true, focus: false })) : [];  // the trace comes back
  $("#gv-paths").hidden = true;
  $("#gv-stage").classList.remove("paths-open");
  $("#gv-paths-btn").setAttribute("aria-pressed", "false");
  pathsChanged();
}
function renderPathControls() {
  setSeg("#gv-path-mode", "pm", A.pathMode);
  $("#gv-path-hint").textContent = A.pathMode === "explain" && A.attrDiff
    ? "The circuit behind the change: the units whose effect on the next token the weight edits changed most, and the active units feeding them."
    : PATH_MODES[A.pathMode];
  $("#gv-path-hops").value = A.pathHops; $("#gv-path-hops-v").textContent = A.pathHops;
  $("#gv-path-branch").value = A.pathBranch; $("#gv-path-branch-v").textContent = A.pathBranch;
  $("#gv-path-activity").checked = A.pathActivity;
}

function pathSources() {
  const s = [...A.sel];
  if (!s.length && A.selected != null) s.push(A.selected);
  return s;
}

function pathsChanged() { pushState(); renderLegend(); }

function pathMessage(text) {
  A.paths = []; pathsChanged();
  $("#gv-path-head").innerHTML = "";
  $("#gv-path-list").innerHTML = `<li class="empty">${text}</li>`;
}

function runPathSearch(frame) {
  if (!A.pathsOpen) return;
  const e = A.edges;
  if (!e) { pathMessage("Loading the graph's edges…"); return; }  // loadAtlas reruns the search
  if (!e.available) { pathMessage("Edges are not available for this model yet."); return; }
  const weight = A.pathActivity && rowOf(A.cursor) != null ? (g) => 0.15 + 0.85 * activityAt(g) : null;
  const opts = { hops: A.pathHops, branch: A.pathBranch, weight, limit: 60 };
  let found;
  if (A.pathMode === "explain") {
    if (!A.runId || A.cursor == null) { pathMessage("Pick a token on the timeline first."); return; }
    if (!A.attr || A.attrInfo?.t !== A.cursor || A.attrInfo?.loading) {
      pathMessage("Tracing the next token…");
      if (!A.attrInfo?.loading || A.attrInfo?.t !== A.cursor) loadExplain();
      return;
    }
    // Comparing: start from the units whose effect the edits changed most, either way.
    found = explainPaths(e, A.attrDiff ? A.attrDiff.map(Math.abs) : A.attr, opts);
  } else {
    const src = pathSources();
    if (!src.length) { pathMessage("Select units first: click one, shift-click to add more, or shift-drag a box on the graph."); return; }
    if (A.pathMode === "between") {
      if (src.length < 2) { pathMessage("Select at least two units in different layers to find the routes between them."); return; }
      found = findPaths(e, src, { ...opts, dir: "down", targets: new Set(src), hops: Math.max(A.pathHops, 2), beam: 256 });
    } else {
      found = findPaths(e, src, { ...opts, dir: A.pathMode });
    }
  }
  if (!found.length) { pathMessage("No path above chance strength from here. Try more hops, more branches, or turn off activity weighting."); return; }
  A.paths = found.map((p, i) => ({ ...p, on: i < 12, focus: false }));
  pathsChanged();
  renderPathList();
  if (frame) frameShownPaths();
}

function frameShownPaths() {
  const units = [...new Set(A.paths.filter((p) => p.on).flatMap((p) => p.units))];
  if (units.length) graph.fitUnits(units);
}

function renderPathList() {
  const shown = A.paths.filter((p) => p.on).length, max = Math.max(...A.paths.map((p) => p.score), 1e-9);
  $("#gv-path-head").innerHTML = `<span><b>${A.paths.length}</b> paths · ${shown} shown</span>
    <span class="path-actions">
      <button type="button" class="link" data-act="all">Select all</button>
      <button type="button" class="link" data-act="none">None</button>
      <button type="button" class="link" data-act="frame">Frame</button>
      <button type="button" class="link" data-act="select" title="Add the units on the shown paths to the selection">Select units</button>
      <button type="button" class="link" data-act="copy" title="Copy the shown paths as JSON">Copy</button>
      <button type="button" class="link" data-act="ko" title="Remove the units on the shown paths and see what happens to the next token">Knock out</button>
    </span>`;
  $("#gv-path-list").innerHTML = A.paths.map((p, i) => {
    const chips = p.units.map((g) => `<span class="pchip ${A.units[g].kind}">${shortName(g)}</span>`).join(`<span class="parrow">→</span>`);
    const v = p.root != null && A.attr ? attrView()[p.root] : null;
    const extra = v != null ? `<span class="pattr" title="${A.attrDiff ? "change from the original weights" : "effect on the next token"}">${A.attrDiff ? "Δ" : ""}${v >= 0 ? "+" : ""}${v.toFixed(2)}</span>` : "";
    return `<li data-i="${i}" class="${p.on ? "on" : ""}${p.focus ? " focus" : ""}">
      <input type="checkbox" ${p.on ? "checked" : ""} aria-label="Show path ${i + 1}">
      <span class="ptext">${chips}</span>
      <span class="pscore">${extra}<span class="pbar"><i style="width:${Math.max(4, (p.score / max) * 100).toFixed(0)}%"></i></span></span></li>`;
  }).join("");
  $("#gv-path-head").querySelectorAll("[data-act]").forEach((b) => { b.onclick = () => pathAction(b.dataset.act); });
  $("#gv-path-list").querySelectorAll("li[data-i]").forEach((li) => {
    const p = A.paths[Number(li.dataset.i)];
    li.querySelector("input").onchange = (ev) => { p.on = ev.target.checked; li.classList.toggle("on", p.on); pathsChanged(); renderPathHeadCount(); };
    li.onmouseenter = () => { p.focus = true; pushState(); };
    li.onmouseleave = () => { p.focus = false; pushState(); };
    li.querySelector(".ptext").onclick = () => { p.on = true; li.querySelector("input").checked = true; li.classList.add("on"); graph.fitUnits(p.units); pushState(); };
  });
}
function renderPathHeadCount() {
  const span = $("#gv-path-head span");
  if (span) span.innerHTML = `<b>${A.paths.length}</b> paths · ${A.paths.filter((p) => p.on).length} shown`;
}

function pathAction(act) {
  const shown = A.paths.filter((p) => p.on);
  if (act === "all" || act === "none") { for (const p of A.paths) p.on = act === "all"; renderPathList(); pathsChanged(); }
  else if (act === "frame") frameShownPaths();
  else if (act === "select") { for (const p of shown) for (const g of p.units) A.sel.add(g); pushState(); renderSelChip(); }
  else if (act === "ko") {
    const units = [...new Set(shown.flatMap((p) => p.units))];
    if (!units.length || A.cursor == null) return;
    if (A.mode !== "attr") setMode("attr");
    openTokenPanel();
    knockOutSet(units, `units on ${shown.length} shown path${shown.length === 1 ? "" : "s"}`);
  }
  else if (act === "copy") {
    const data = shown.map((p) => ({ path: p.units.map(shortName), units: p.units, score: +p.score.toFixed(5),
      edges_x_chance: p.xs.map((x) => +x.toFixed(3)),
      ...(p.root != null && A.attr ? { attribution: +A.attr[p.root].toFixed(4) } : {}),
      ...(p.root != null && A.attrDiff ? { attribution_original: +A.attrOrig[p.root].toFixed(4) } : {}) }));
    navigator.clipboard?.writeText(JSON.stringify({ run: A.runId, token: A.cursor, mode: A.pathMode, paths: data }, null, 2));
  }
}

// ---------------------------------------------------------------- tracing: one unit at the cursor token
const dbar = (x) => {
  const v = clamp(Number.isFinite(x) ? x : 0, -1, 1);
  return `<span class="dbar"><i class="${v >= 0 ? "pos" : "neg"}" style="left:${v >= 0 ? 50 : 50 + v * 50}%;width:${Math.abs(v) * 50}%"></i></span>`;
};

async function renderTokenSection() {
  const box = $("#an-token-sec");
  if (!box || A.selected == null || A.detailMode !== "unit") return;
  const g = A.selected, t = A.cursor, seq = ++A.tokenSeq, u = A.units[g];
  if (t == null || !A.runId) { box.innerHTML = `<p class="empty">Pick a token on the timeline to see what this unit did there.</p>`; return; }
  if (!box.innerHTML.trim()) box.innerHTML = `<div class="skeleton short"></div>`;
  try {
    const target = A.mode === "attr" ? A.attrTarget : null;
    const [ex, more] = await Promise.all([
      fetchExplain(t, target),
      api(`${tokenUrl(t)}/${u.kind === "cluster" ? "neurons" : "attention"}?unit=${g}${u.kind === "cluster" && target != null ? `&target=${encodeURIComponent(target)}` : ""}`),
    ]);
    const orig = A.cmp && ex.weights_edited ? await fetchExplain(t, ex.target.id, "original") : null;
    if (seq !== A.tokenSeq || A.selected !== g) return;
    const c = ex.contrib[g], rank = 1 + ex.contrib.filter((x) => Math.abs(x) > Math.abs(c)).length;
    const fmt = (x) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(3)}`;
    const before = orig ? orig.contrib[g] : null;
    box.innerHTML = `<div class="tok-line"><span class="tok-chip">${esc(visTok(ex.token))}</span><span class="arrow">→</span>
        <span class="tok-chip target">${esc(visTok(ex.target.text))}</span>
        <span class="attr-val ${c >= 0 ? "pos" : "neg"}">${fmt(c)} logits</span></div>
      <p class="hint">${Math.abs(c) < 5e-4
        ? `No direct effect on this candidate${A.edited.has(g) ? "; its weights are changed by an applied edit" : ""}.`
        : `${c >= 0 ? "Pushed" : "Pulled"} the model's next token ${c >= 0 ? "toward" : "away from"} this candidate, by its direct write. #${rank} of ${A.U} units by size of effect.`}</p>
      ${orig ? `<div class="cmp-line"><span>original weights <b>${fmt(before)}</b></span><span class="arrow">→</span>
        <span>edited <b>${fmt(c)}</b></span><span class="attr-val ${c - before >= 0 ? "pos" : "neg"}">Δ ${fmt(c - before)}</span></div>
        <p class="hint">p(${esc(visTok(ex.target.text))}) ${fmtP(orig.target.prob)} → ${fmtP(ex.target.prob)} with the applied edits.
        Below is the edited model.</p>` : ""}
      <div class="d-actions"><button type="button" class="btn sm" id="an-ko-btn" title="Re-read the context without this unit and compare this token with the intact model">Knock out this unit</button>
        <button type="button" class="btn sm ghost" id="an-why-btn">Why ${esc(visTok(ex.target.text))}?</button></div>
      <ul class="ko-list" id="an-ko"></ul>
      ${u.kind === "cluster" ? neuronsHtml(g, more) : attentionHtml(more)}`;
    box.querySelector("#an-ko-btn").onclick = () => runKnockout([g], shortName(g));
    box.querySelector("#an-why-btn").onclick = () => { if (A.mode !== "attr") setMode("attr"); openTokenPanel(); };
    renderKnockouts();
    box.querySelector("#an-expand")?.addEventListener("click", () => toggleExpand(g));
    box.querySelectorAll("[data-t]").forEach((el) => {
      const tt = Number(el.dataset.t);
      if (tt >= 0) el.onclick = () => { setFollow(false); stopPlay(); setCursor(tt, "jump"); };
    });
  } catch (err) {
    if (seq !== A.tokenSeq) return;
    box.innerHTML = `<p class="empty">${err.status === 409 ? "The model is generating; this fills in when it pauses." : esc(err.message)}</p>`;
    if (err.status === 409) setTimeout(() => { if (A.selected === g && A.cursor === t) renderTokenSection(); }, 1500);
  }
}

function neuronsHtml(g, nn) {
  let ma = 1e-9, mc = 1e-9;
  for (const n of nn.neurons) { ma = Math.max(ma, Math.abs(n.act)); mc = Math.max(mc, Math.abs(n.contrib)); }
  const layer = A.units[g].layer;
  const rows = nn.neurons.slice(0, 24).map((n) => `<li><button type="button" class="link mono" data-open-neuron="L${layer} N${n.n}" title="Open L${layer} N${n.n} in the Neurons view">${n.n}</button>${dbar(n.act / ma)}${dbar(n.contrib / mc)}
      <span class="v ${n.contrib >= 0 ? "pos" : "neg"}">${n.contrib >= 0 ? "+" : ""}${n.contrib.toFixed(3)}</span></li>`).join("");
  return `<div class="facts"><span title="Fewest neurons that make up 90% of the cluster's write at this token"><b>${nn.active}</b> of ${nn.neurons.length} neurons carry 90% of its write</span>
      <span>cluster effect <b>${nn.contrib >= 0 ? "+" : ""}${nn.contrib.toFixed(3)}</b></span></div>
    <div class="nn-head"><span>neuron</span><span>activation</span><span>effect on next token</span><span></span></div>
    <ul class="nn-list">${rows}</ul>
    <button type="button" class="btn sm" id="an-expand">${A.expanded.has(g) ? "Collapse in graph" : "Expand neurons in graph"}</button>`;
}

function attentionHtml(at) {
  if (!at.available) return `<p class="hint">${esc(at.reason)}</p>`;
  const w = at.weights, max = Math.max(...w, 1e-9), start = Math.max(0, w.length - 160);
  const strip = w.slice(start).map((x, i) => {
    const p = start + i;
    return `<span class="att${p === at.pos ? " self" : ""}" style="--a:${(x / max).toFixed(3)}" data-t="${at.t_of[p]}" title="position ${p} · weight ${x.toFixed(4)}">${esc(visTok(at.tokens[p]))}</span>`;
  }).join("");
  const top = at.top.slice(0, 8).map(([p, x]) => `<li data-t="${at.t_of[p]}"><span class="mono">${esc(visTok(at.tokens[p]))}</span>
      <span class="pbar"><i style="width:${((x / max) * 100).toFixed(0)}%"></i></span><span class="v">${(x * 100).toFixed(1)}%</span></li>`).join("");
  return `<p class="hint">Where this head looked from this token: darker means more attention.${start ? ` Showing the last 160 of ${w.length} tokens.` : ""} Click a token to jump there.</p>
    <div class="att-strip">${strip}</div>
    <h5>Most attended</h5><ul class="att-top">${top}</ul>`;
}

async function loadExpanded(g) {
  const t = A.cursor;
  if (t == null || !A.runId) return;
  const nn = await api(`${tokenUrl(t)}/neurons?unit=${g}`);
  if (A.cursor !== t || !A.expanded.has(g)) return;  // moved on, or collapsed meanwhile
  const byId = [...nn.neurons].sort((a, b) => a.n - b.n);
  let m = 1e-9;
  for (const n of byId) m = Math.max(m, Math.abs(n.act));
  A.expanded.set(g, Float32Array.from(byId, (n) => Math.sign(n.act) * Math.sqrt(Math.abs(n.act) / m)));
  pushState();
}

async function toggleExpand(g) {
  if (A.expanded.has(g)) { A.expanded.delete(g); pushState(); renderTokenSection(); return; }
  if (A.expanded.size >= 8) A.expanded.delete(A.expanded.keys().next().value);
  A.expanded.set(g, new Float32Array(A.units[g].size).fill(NaN));
  try { await loadExpanded(g); } catch { /* the grid stays empty until the next token */ }
  graph.flyTo(g, 6);
  renderTokenSection();
}

function refreshExpanded() { for (const g of A.expanded.keys()) loadExpanded(g).catch(() => {}); }

// ---------------------------------------------------------------- conversation: the run's text, word by word
const CV = { rendered: 0, block: null, blockKey: null, showTemplate: false, open: true };

/** How surprising a token was to the model, in [0, 1] (0 to 10 bits). */
const surprise = (p) => (p == null ? 0 : clamp(-Math.log2(Math.max(p, 1e-9)) / 10, 0, 1));
const ROLE_LABEL = { reasoning: "thinking", template: "template", assistant: "model" };

function tokenSpan(rec) {
  const s = surprise(rec.p);
  const tip = `t${rec.t} · ${rec.p != null ? `p ${fmtP(rec.p)}, ${(-Math.log2(Math.max(rec.p, 1e-9))).toFixed(1)} bits` : "no prediction (start of the context)"}`;
  return `<span class="ct${s > 0.33 ? " surprising" : ""}" data-t="${rec.t}"${s ? ` style="--s:${s.toFixed(2)}"` : ""} title="${tip}">${esc(rec.tx)}</span>`;
}

/** Render the run's tokens grouped into turns; `full` redraws, otherwise appends what arrived since. */
function renderConvo(full) {
  const body = $("#tv-convo-body");
  if (!body) return;
  $("#tv-convo-meta").textContent = A.runId ? (A.meta?.label || A.runId) : "";
  if (full) { body.innerHTML = ""; CV.rendered = 0; CV.block = null; CV.blockKey = null; }
  if (!A.runId || !A.view.length) {
    body.innerHTML = `<p class="empty">${!A.runId ? "No run yet. Ask the model below" : "No tokens in this run yet. Ask the model below"}, or point any harness at <code>${esc(location.origin)}/v1</code>.</p>`;
    CV.rendered = 0;
    return;
  }
  body.querySelector(".empty")?.remove();
  const nearBottom = body.scrollHeight - body.scrollTop - body.clientHeight < 80;
  let html = "";
  const flush = () => { if (html && CV.block) CV.block.insertAdjacentHTML("beforeend", html); html = ""; };
  for (let i = CV.rendered; i < A.view.length; i++) {
    const rec = A.tok[A.view[i]];
    if (!rec) continue;
    if (rec.r === "template" && !CV.showTemplate) { flush(); CV.blockKey = null; continue; }
    const key = `${rec.req}|${rec.r}`;
    if (key !== CV.blockKey) {
      flush();
      const div = document.createElement("div");
      div.className = `cb role-${rec.r}`;
      div.innerHTML = `<div class="cb-role">${esc(ROLE_LABEL[rec.r] || rec.r)}</div><div class="cb-text"></div>`;
      body.appendChild(div);
      CV.block = div.querySelector(".cb-text"); CV.blockKey = key;
    }
    html += tokenSpan(rec);
  }
  flush();
  CV.rendered = A.view.length;
  if (full) markConvo(true);
  else if (nearBottom || A.follow) body.scrollTop = body.scrollHeight;
}

/** The token after slot t in the same request: what explaining t traces by default. */
function nextSlot(t) {
  const i = A.view.indexOf(t), rec = A.tok[t];
  const n = i >= 0 ? A.tok[A.view[i + 1]] : null;
  return n && rec && n.req === rec.req && n.pos === rec.pos + 1 ? n.t : null;
}

function markConvo(scroll) {
  const body = $("#tv-convo-body");
  if (!body) return;
  body.querySelectorAll(".ct.cursor, .ct.target").forEach((el) => el.classList.remove("cursor", "target"));
  if (A.cursor == null) return;
  const of = A.attrOf ?? nextSlot(A.cursor);
  const c = body.querySelector(`.ct[data-t="${A.cursor}"]`), g = of != null ? body.querySelector(`.ct[data-t="${of}"]`) : null;
  c?.classList.add("cursor");
  g?.classList.add("target");
  if (scroll && CV.open) (g || c)?.scrollIntoView({ block: "nearest" });
}

/** Why did the model write token t? Trace the position before it with t as the target. */
async function traceWord(t) {
  if (!A.runId) return;
  let src;
  try { src = await api(`${tokenUrl(t)}/producer`); } catch (err) { convoNote(err.message); return; }
  setFollow(false); stopPlay();
  setCursor(src.t, "jump");  // schedules the trace, which picks up the target set next
  A.attrTarget = String(src.target.id); A.attrOf = t;
  A.frameTrace = true;  // frame the whole trace, START to END, once it is drawn
  if (A.mode !== "attr") { A.mode = "attr"; setSeg("#gv-mode", "mode", "attr"); pushState(); renderLegend(); renderReadout(); }
  openTokenPanel();
  markConvo(false);
}

function convoNote(text) {
  const el = $("#tv-convo .convo-hint");
  el.textContent = text; el.classList.add("warn-text");
  setTimeout(() => { el.textContent = "Click any word to see which units and layers produced it. Underlines mark words the model found surprising."; el.classList.remove("warn-text"); }, 4000);
}

function setConvo(open) {
  CV.open = open;
  $("#view-weights").classList.toggle("convo-hidden", !open);
  $("#gv-convo-btn").setAttribute("aria-pressed", String(open));
  try { localStorage.setItem("brainscan-convo", open ? "1" : "0"); } catch { /* storage unavailable */ }
  schedule();
  if (open) markConvo(true);
}

// ---------------------------------------------------------------- why this token: layers, lens, units, knockouts
function openTokenPanel() {
  A.detailMode = "token"; A.selected = null; A.detail = null;
  pushState(); renderLegend(); schedule();
  openDrawer(true);
  $("#an-unit-where").textContent = "Why this token";
  renderTokenPanel();
}

async function renderTokenPanel() {
  if (A.detailMode !== "token") return;
  const box = $("#an-detail"), info = A.attrInfo, t = A.cursor, seq = ++A.panelSeq;
  if (t == null || !A.runId) { box.innerHTML = `<p class="empty">Click a word in the conversation, or a token on the timeline.</p>`; return; }
  if (!info || info.t !== t || info.loading) { box.innerHTML = `<p class="empty">Tracing…</p><div class="skeleton"></div><div class="skeleton tall"></div>`; return; }
  if (info.busy) { box.innerHTML = `<p class="empty">The model is generating; this fills in when it pauses.</p>`; return; }
  if (info.error) { box.innerHTML = `<p class="empty">${esc(info.error)}</p>`; return; }
  const of = A.attrOf ?? nextSlot(t);
  let lens = null;
  if (of != null) {
    try { const rec = await api(tokenUrl(of)); if (rec.tok === info.target.id && rec.lens_p) lens = rec; } catch { /* no lens */ }
  }
  if (seq !== A.panelSeq || A.detailMode !== "token" || A.attrInfo !== info) return;
  box.innerHTML = tokenPanelHtml(info, lens);
  box.querySelectorAll(".alt[data-id]").forEach((b) => { b.onclick = () => { A.attrTarget = b.dataset.id; A.attrOf = null; loadExplain(); }; });
  box.querySelectorAll("[data-layer]").forEach((el) => { el.onclick = () => graph.flyToLayer(Number(el.dataset.layer)); });
  box.querySelectorAll(".tp-units li[data-g]").forEach((li) => {
    li.onclick = (ev) => {
      const g = Number(li.dataset.g);
      if (ev.shiftKey) { toggleSelect(g); return; }
      selectUnit(g); graph.flyTo(g, Math.max(graph.target.s, 1.6));
    };
  });
  box.querySelectorAll(".routes li[data-i]").forEach((li) => {
    const i = Number(li.dataset.i), r = A.trace?.routes[i];
    li.onmouseenter = () => focusRoute(i, true);
    li.onmouseleave = () => { if (!A.followTimer) focusRoute(i, false); };  // back to the pinned route
    li.onclick = (ev) => {
      const chip = ev.target.closest(".pchip[data-g]");
      if (chip) { const g = Number(chip.dataset.g); selectUnit(g); graph.flyTo(g, Math.max(graph.target.s, 1.6)); return; }
      if (r) { focusRoute(i, true, true); graph.fitTrace(r.units); }
    };
    li.classList.toggle("pinned", i === A.traceFocus);
  });
  $("#tp-follow")?.addEventListener("click", () => followRoute(A.traceFocus || 0));
  $("#tp-frame")?.addEventListener("click", () => graph.fitTrace(routeUnits()));
  $("#tp-trace-off")?.addEventListener("click", () => { A.traceOn = false; clearTrace(); renderTokenPanel(); });
  $("#tp-trace-on")?.addEventListener("click", () => { A.traceOn = true; updateTrace(); renderTokenPanel(); });
  $("#tp-circuit").onclick = () => openPaths("explain");
  $("#tp-ko-top").onclick = () => runKnockout(topUnits(info, 5, true), "top 5 units that pushed it up");
  $("#tp-ko-sel")?.addEventListener("click", () => knockOutSet([...A.sel], "selected units"));
  $("#tp-add-check").onclick = addCheckFromToken;
  $("#tp-forget").onclick = () => forgetFromToken(info);
  $("#tp-copy").onclick = () => navigator.clipboard?.writeText(JSON.stringify(traceJson(info, lens), null, 2));
  renderKnockouts();
}

function topUnits(info, n, positive) {
  const c = info.contrib;
  return [...c.keys()].filter((g) => (positive ? c[g] > 0 : c[g] < 0)).sort((a, b) => (positive ? c[b] - c[a] : c[a] - c[b])).slice(0, n);
}

function tokenPanelHtml(info, lens) {
  const o = info.orig, fmtL = (x) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(2)}`;
  const ctxTok = A.tok[info.t]?.tx ?? info.token;
  // The word being explained: one the model generated, one from the prompt (typed by you or a tool),
  // or a runner-up picked from the chips.
  const ofRec = A.tok[A.attrOf ?? nextSlot(info.t)];
  const chosen = A.attrOf == null && A.attrTarget != null && !(ofRec && A.attrInfo?.target?.text === ofRec.tx);
  const typed = !chosen && !!ofRec && ofRec.ph !== "gen";
  const alts = info.alternatives.filter((a) => a.id !== info.target.id).slice(0, 5).map((a) =>
    `<button type="button" class="alt" data-id="${a.id}" title="Explain this candidate instead">${esc(visTok(a.text))} <span class="dim">${fmtP(a.prob)}</span></button>`).join("");
  const unitRows = (gs) => {
    const max = Math.max(...gs.map((g) => Math.abs(info.contrib[g])), 1e-9);
    return gs.map((g) => {
      const c = info.contrib[g], u = A.units[g];
      return `<li data-g="${g}" title="Open this unit (shift-click to add it to the selection)"><span class="tp-kind ${u.kind}">${u.kind === "head" ? "head" : "MLP"}</span>
        <span class="tp-name">${shortName(g)}</span><span class="pbar ${c < 0 ? "neg" : ""}"><i style="width:${Math.max(3, Math.abs(c) / max * 100).toFixed(0)}%"></i></span>
        <span class="v ${c >= 0 ? "pos" : "neg"}">${fmtL(c)}</span></li>`;
    }).join("");
  };
  return `<div class="d-hero">
      <div class="d-kicker">${typed
        ? `You wrote this after <span class="mono">${esc(visTok(ctxTok))}</span> (t${info.t}): how the model predicted it`
        : `After <span class="mono">${esc(visTok(ctxTok))}</span> (t${info.t}) the model ${chosen ? "could have written" : "wrote"}`}</div>
      <div class="tp-target"><span class="tok-chip target big">${esc(visTok(info.target.text))}</span></div>
      ${typed && info.target.rank > 1 ? `<p class="hint tp-note">The model didn't expect this word: it ranked it #${info.target.rank} and expected
        <b>${esc(visTok(info.alternatives[0].text))}</b> (${fmtP(info.alternatives[0].prob)}) instead.</p>` : ""}
      <div class="d-stats">
        <div><span>probability</span><b>${fmtP(info.target.prob)}</b></div>
        <div><span>rank</span><b>#${info.target.rank}</b></div>
        <div><span>logit</span><b>${info.target.logit.toFixed(2)}</b></div>
      </div>
      ${o ? `<div class="cmp-line"><span>original weights p <b>${fmtP(o.target.prob)}</b></span><span class="arrow">→</span><span>edited <b>${fmtP(info.target.prob)}</b></span><span class="attr-val ${info.target.logit - o.logit >= 0 ? "pos" : "neg"}">logit ${fmtL(info.target.logit - o.logit)}</span></div>` : ""}
      ${alts ? `<div class="chips tp-alts"><span class="lbl">runners-up</span>${alts}</div>` : ""}
    </div>
    ${routesHtml(info)}
    <div class="d-sec tp-tools">
      <div class="d-actions wrap">
        <button type="button" class="btn sm" id="tp-circuit" title="Open the path explorer on this token: up to 60 routes, with hops and branches to tune">All routes</button>
        <button type="button" class="btn sm" id="tp-forget" title="Act on it: find the units that carry this word, remove them, and measure what else changed">Forget this fact →</button>
        <button type="button" class="btn sm" id="tp-ko-top" title="Remove the five units that pushed it up most and re-run">Knock out top 5</button>
        ${A.sel.size ? `<button type="button" class="btn sm" id="tp-ko-sel">Knock out selection (${A.sel.size})</button>` : ""}
        <button type="button" class="btn sm ghost" id="tp-add-check" title="Save as a behavior check: after this context, this word. Checks show what an edit changes.">Add as a check</button>
        <button type="button" class="btn sm ghost" id="tp-copy" title="Copy this trace as JSON">Copy JSON</button>
      </div>
    </div>
    <section class="d-sec"><h4>Where the logit came from <span class="dim">direct effect by layer</span></h4>
      ${layerChart(info)}
      <p class="hint">Each layer's attention heads and MLP clusters, by how much they wrote toward this token (read through the final norm and the unembedding). The line adds them up from the embedding to the final logit. Click a layer to fly to it.</p></section>
    <section class="d-sec"><h4>When the model knew <span class="dim">logit lens</span></h4>${lensStrip(lens, info)}</section>
    <section class="d-sec"><h4>Units that pushed it up</h4><ul class="tp-units">${unitRows(topUnits(info, 8, true))}</ul>
      <h4 class="sub">Units that pulled it down</h4><ul class="tp-units">${unitRows(topUnits(info, 5, false))}</ul>
      <p class="hint">Direct effects add up to the logit. They don't show what a unit does through later layers: a knockout does.</p></section>
    <section class="d-sec"><h4>Knockout tests <span class="dim">remove units, re-run, compare</span></h4>
      <ul class="ko-list" id="tp-ko-list"></ul>
      <p class="hint">A knockout reads the whole context again with the units' writes set to zero and compares this token with the intact model. The total effect includes everything that flows through later layers, so the gap to the direct effect shows how much the rest of the network makes up for them. Select units on the graph (shift-click or shift-drag) to test your own set.</p></section>`;
}

function layerChart(info) {
  const L = info.layers.length, W = 392, H = 150, pad = 18, top = 10, bottom = H - 20;
  if (!L) return "";
  const vals = info.layers.flat(), maxBar = Math.max(...vals.map(Math.abs), 1e-6);
  const cum = [info.embed];
  for (const [h, c] of info.layers) cum.push(cum[cum.length - 1] + h + c);
  const cmin = Math.min(...cum, 0), cmax = Math.max(...cum, 1e-6);
  const bw = (W - pad * 2) / L, mid = (top + bottom) / 2, half = (bottom - top) / 2;
  const yb = (v) => mid - (v / maxBar) * half;
  const yc = (v) => bottom - ((v - cmin) / (cmax - cmin || 1)) * (bottom - top);
  let bars = "";
  info.layers.forEach(([h, c], l) => {
    const x = pad + l * bw;
    const rect = (v, dx, cls) => { const y0 = yb(Math.max(v, 0)), y1 = yb(Math.min(v, 0)); return `<rect class="${cls}" x="${(x + dx).toFixed(1)}" y="${y0.toFixed(1)}" width="${Math.max(bw * 0.38, 1).toFixed(1)}" height="${Math.max(y1 - y0, 0.5).toFixed(1)}"/>`; };
    bars += `<g data-layer="${l}" class="lc-layer"><title>Layer ${l}: heads ${h >= 0 ? "+" : ""}${h.toFixed(2)}, MLP ${c >= 0 ? "+" : ""}${c.toFixed(2)}</title>
      <rect class="lc-hit" x="${x.toFixed(1)}" y="${top}" width="${bw.toFixed(1)}" height="${bottom - top}"/>${rect(h, bw * 0.08, "lc-head")}${rect(c, bw * 0.5, "lc-mlp")}</g>`;
  });
  const line = cum.map((v, i) => `${(pad + Math.max(i - 0.5, 0) * bw).toFixed(1)},${yc(v).toFixed(1)}`).join(" ");
  const ticks = [0, Math.floor(L / 4), Math.floor(L / 2), Math.floor(3 * L / 4), L - 1].filter((v, i, a) => a.indexOf(v) === i)
    .map((l) => `<text x="${(pad + (l + 0.5) * bw).toFixed(1)}" y="${H - 6}" class="lc-tick">L${l}</text>`).join("");
  return `<svg class="layer-chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Direct effect on the token by layer, attention and MLP, with the running total">
    <line x1="${pad}" x2="${W - pad}" y1="${mid}" y2="${mid}" class="lc-zero"/>${bars}
    <polyline points="${line}" class="lc-cum"/>
    <text x="${W - pad}" y="${top + 8}" class="lc-label" text-anchor="end">logit ${cum[cum.length - 1].toFixed(1)}</text>
    ${ticks}</svg>
    <div class="lc-legend"><span><i class="lc-head"></i>attention heads</span><span><i class="lc-mlp"></i>MLP clusters</span><span><i class="lc-cum"></i>running total</span></div>`;
}

function lensStrip(lens, info) {
  if (!lens) {
    return `<p class="hint">The logit lens is recorded for the token the model actually produced${A.attrTarget != null && !A.attrOf ? "; this is a different candidate" : ""}.</p>`;
  }
  const L = lens.lens_p.length, first = lens.lens_top.findIndex((x) => x === lens.tok);
  const cells = lens.lens_p.map((p, l) => {
    const a = clamp((Math.log10(Math.max(p, 1e-6)) + 6) / 6, 0, 1);
    return `<span class="lens-cell${lens.lens_top[l] === lens.tok ? " top" : ""}" style="--a:${a.toFixed(2)}" data-layer="${l}"
      title="Layer ${l}: p ${fmtP(p)} · top guess ${esc(visTok(lens.lens_top_text[l]))}"></span>`;
  }).join("");
  return `<div class="lens-strip" style="--n:${L}">${cells}</div>
    <div class="lens-axis"><span>L0</span><span>L${L - 1}</span></div>
    <p class="hint">${first >= 0 ? `The token is the top guess from layer <b>${first}</b> on (outlined); ` : "No layer has it as the top guess; "}darker means more probable when each layer's residual is read through the final norm and unembedding. Final p ${fmtP(info.target.prob)}.</p>`;
}

function traceJson(info, lens) {
  return { run: A.runId, t: info.t, token: info.token, target: info.target, alternatives: info.alternatives,
    logit: info.logit, embed: info.embed, layers: info.layers,
    top_units: info.top.map(([g, c]) => ({ unit: shortName(g), g, contrib: c })),
    lens: lens ? { p: lens.lens_p, top: lens.lens_top_text } : null,
    knockouts: (A.knockouts || []).filter((k) => k.res).map((k) => ({ label: k.label, units: k.units.map(shortName), ...k.res })) };
}

/** Test removing units (factor 0) or scaling their writes (0.5 dampens, 2 boosts) on this token. */
async function runKnockout(units, label, factor = 0) {
  const t = A.cursor;
  if (t == null || !A.runId || !units.length) return;
  const key = `${A.runId}|${t}|${A.attrTarget ?? ""}`;
  if (A.koKey !== key) { A.koKey = key; A.knockouts = []; }
  const item = { label, units: [...units], factor, pending: true };
  A.knockouts.unshift(item);
  renderKnockouts();
  $("#tp-ko-list")?.scrollIntoView({ block: "nearest", behavior: "smooth" });  // the result shows here
  try {
    item.res = await post(`${tokenUrl(t)}/knockout`, { units, target: A.attrTarget, factor });
  } catch (err) {
    item.error = err.status === 409 ? "The model is generating; try again when it pauses." : err.message;
  }
  item.pending = false;
  if (A.koKey === key) renderKnockouts();
}

const MAX_KO = 64;  // the most units one knockout removes (MAX_KNOCKOUT in recorder/tracing.py)

/** Knock out a set of units. Past the limit, the ones with the largest direct effect on the token
 * are kept, and the label says how many of the set were removed. */
function knockOutSet(units, what, factor = 0) {
  if (!units.length) return;
  const how = factor ? ` ×${factor}` : "";
  if (units.length <= MAX_KO) { runKnockout(units, `${units.length} ${what}${how}`, factor); return; }
  const c = A.attrInfo?.t === A.cursor ? A.attrInfo.contrib : null;
  const kept = c ? [...units].sort((a, b) => Math.abs(c[b]) - Math.abs(c[a])).slice(0, MAX_KO) : units.slice(0, MAX_KO);
  runKnockout(kept, `${MAX_KO} of ${units.length} ${what}${how}${c ? ", the largest direct effects" : ""}`, factor);
}

/** Scale tests offered for a selection: [factor, what it does]. */
const SCALES = [[0.5, "dampen"], [0.25, "dampen more"], [1.5, "boost"], [2, "boost more"]];

/** Remove a knockout's units from the weights for real: one grouped edit, applied now. */
async function makePermanent(k) {
  if (!k?.res || k.permanent) return;
  k.permanent = "pending"; k.permError = null;
  renderKnockouts();
  try {
    await post("/brainscan/anatomy/lesion", { units: k.units, factor: k.factor || 0, apply: true,
      note: `${k.factor ? `scale test ×${k.factor}` : "knockout"} for ${visTok(k.res.target.text)}` });
    k.permanent = "done";
    A.cmp = true;  // the trace now shows what the edit changed on this token
    document.dispatchEvent(new CustomEvent("brainscan:checks", { detail: { run: true } }));
    setEditsPanel(true);
  } catch (err) {
    k.permanent = null; k.permError = err.message;
  }
  renderKnockouts();
}

function knockoutHtml(k, i) {
  const names = k.units.length <= 4 ? k.units.map(shortName).join(" + ") : `${k.units.slice(0, 3).map(shortName).join(" + ")} + ${k.units.length - 3} more`;
  if (k.pending) return `<li class="pending"><b>${esc(k.label)}</b><span class="dim">${names}</span><span class="muted">Re-running without them…</span></li>`;
  if (k.error) return `<li class="err"><b>${esc(k.label)}</b><span class="warn-text">${esc(k.error)}</span></li>`;
  const r = k.res, d = r.direct_effect, tot = r.total_effect, sgn = (x) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(2)}`;
  let why;
  if (k.factor) why = Math.abs(tot) < 0.05 ? `Scaling them ×${k.factor} barely changes this token.`
    : `Scaling them ×${k.factor} moves its logit ${sgn(-tot)}${Math.abs(d) >= 0.05 ? `; their own writes alone would move it ${sgn(-d)}` : ""}.`;
  else if (Math.abs(d) < 0.05 && Math.abs(tot) < 0.05) why = "Removing them barely changes this token.";
  else if (d > 0 && tot >= 0 && tot < d) why = `The rest of the network made up ${(d - tot).toFixed(2)} of their ${d.toFixed(2)} direct effect (${Math.round((1 - tot / d) * 100)}%).`;
  else if (d > 0 && tot > d) why = `Removing them costs more than their direct writes: they also help this token through later layers (+${(tot - d).toFixed(2)}).`;
  else if (d > 0 && tot < 0) why = "Without them the token gets more likely: their indirect effect outweighs their direct push.";
  else why = `Direct effect ${sgn(d)}, total ${sgn(tot)}.`;
  const after = r.top_after.slice(0, 3).map((a) => `${esc(visTok(a.text))} ${fmtP(a.prob)}`).join(" · ");
  return `<li><div class="ko-head"><b>${esc(k.label)}</b><span class="dim">${names}</span></div>
    <div class="ko-nums"><span>p ${fmtP(r.before.prob)} → <b>${fmtP(r.after.prob)}</b></span><span>logit ${sgn(-tot)}</span>
      <span>rank #${r.before.rank} → #${r.after.rank}</span></div>
    <div class="ko-bars"><span>direct</span>${dbar(d / Math.max(Math.abs(d), Math.abs(tot), 1e-6))}<span class="v">${sgn(d)}</span>
      <span>total</span>${dbar(tot / Math.max(Math.abs(d), Math.abs(tot), 1e-6))}<span class="v">${sgn(tot)}</span></div>
    <p class="hint">${why} Most likely now: ${after}.</p>
    <div class="ko-actions">${k.permanent === "done"
      ? `<span class="pill ok">made permanent: applied in the weights</span>`
      : `<button type="button" class="btn sm" data-ko-perm="${i}" ${k.permanent ? "disabled" : ""} title="${k.factor ? `Scale these units ×${k.factor} in the weights for real` : "Remove these units from the weights for real"}: one edit in the stack, applied now. Restore any time from Edits.">${k.permanent ? "Applying…" : "Make permanent"}</button>`}
      ${k.permError ? `<span class="warn-text small">${esc(k.permError)}</span>` : ""}</div></li>`;
}

function renderKnockouts() {
  const list = $("#tp-ko-list"), kos = A.knockouts || [];
  refreshBar("weights");
  const onClick = (ev) => {
    const b = ev.target.closest("[data-ko-perm]");
    if (b) makePermanent(kos[Number(b.dataset.koPerm)]);
  };
  if (list) {
    list.innerHTML = kos.map(knockoutHtml).join("") || `<li class="muted">No tests yet at this token.</li>`;
    list.onclick = onClick;
  }
  const unitBox = $("#an-ko");
  if (unitBox && A.selected != null) {
    const k = kos.find((x) => x.units.length === 1 && x.units[0] === A.selected);
    unitBox.innerHTML = k ? knockoutHtml(k, kos.indexOf(k)) : "";
    unitBox.onclick = onClick;
  }
}

// ---------------------------------------------------------------- edits panel (rendered by ui/edits.js)
function setEditsPanel(open) {
  A.editsOpen = open;
  $("#gv-edits").hidden = !open;
  $("#gv-edits-btn").setAttribute("aria-pressed", String(open));
  if (open) document.dispatchEvent(new CustomEvent("brainscan:workbench"));
  pushState();
}

/** Hand the traced word to the Actions view's "Make it forget a fact". */
function forgetFromToken(info) {
  const i = A.view.indexOf(info.t);
  const lead = i < 0 ? "" : A.view.slice(Math.max(0, i - 14), i + 1).map((t) => A.tok[t]?.tx ?? "").join("");
  document.dispatchEvent(new CustomEvent("brainscan:action", { detail: {
    action: "forget", run: A.runId, t: info.t, target: info.target.id, of: A.attrOf ?? nextSlot(info.t),
    word: info.target.text, lead } }));
}

async function addCheckFromToken() {
  const info = A.attrInfo, btn = $("#tp-add-check");
  if (!info?.target || !A.runId) return;
  if (btn) btn.disabled = true;
  try {
    await post("/brainscan/checks", { run: A.runId, t: info.t, target: info.target.id });
    if (btn) btn.textContent = "Saved as a check ✓";
    noteBar("weights", `Saved “${visTok(info.target.text)}” as a behavior check`);
    document.dispatchEvent(new CustomEvent("brainscan:checks", { detail: { run: true } }));
  } catch (err) {
    if (btn) { btn.textContent = "Couldn't save"; btn.title = err.message; btn.disabled = false; }
    noteBar("weights", `Couldn't save the check: ${err.message}`);
  }
}

// ---------------------------------------------------------------- the action bar (ui/actionbar.js)
/** The traced word, once its explanation has loaded. */
const traced = () => {
  const info = A.attrInfo;
  return A.mode === "attr" && info && info.t === A.cursor && !info.loading && info.target ? info : null;
};
const needWord = () => (!A.runId ? ($("#run-select")?.value ? "a word: click one in Trace's conversation" : "a run: ask the model in the conversation")
  : traced() ? null : A.cursor == null ? "a word: click one in the conversation" : "the word's trace to load (click a word)");
/** The latest knockout on this token that hasn't been made permanent. */
const lastKnockout = () => (A.koKey === `${A.runId}|${A.cursor}|${A.attrTarget ?? ""}`
  ? (A.knockouts || []).find((k) => k.res && !k.permanent) : null);

async function lesionSelection() {
  const units = [...A.sel];
  try {
    await post("/brainscan/anatomy/lesion", { units, apply: true, note: `lesion of ${units.length} selected units` });
    A.cmp = true;
    setEditsPanel(true);
    document.dispatchEvent(new CustomEvent("brainscan:checks", { detail: { run: true } }));
    noteBar("weights", `Removed ${units.length} unit${units.length === 1 ? "" : "s"} from the weights: one edit, applied`);
  } catch (err) { noteBar("weights", `Couldn't lesion: ${err.message}`); }
}

async function restoreOriginal() {
  try {
    await post("/brainscan/workbench/revert");
    document.dispatchEvent(new CustomEvent("brainscan:workbench"));
    document.dispatchEvent(new CustomEvent("brainscan:checks", { detail: { run: true } }));
    noteBar("weights", "Original weights restored");
  } catch (err) { noteBar("weights", `Couldn't restore: ${err.message}`); }
}

/** Open the Edits panel and press its "Re-run this chat" once it has rendered. */
function rerunChat() {
  setEditsPanel(true);
  let tries = 0;
  const press = () => {
    const b = $("#gv-edits [data-replay]");
    if (b && !b.disabled) { b.click(); return; }
    if (++tries < 20) setTimeout(press, 100);
  };
  setTimeout(press, 100);
}

function setupBar() {
  const koKey = () => `${A.runId}|${A.cursor}|${A.attrTarget ?? ""}`;
  defineBar("weights", {
    groups: ["Read", "Test", "Change", "Verify"],
    // What the action steps read (ui/verbs.js): the traced word, the selection, a test on this word.
    state: () => ({ word: traced() ? visTok(traced().target.text) : null, selection: A.sel.size,
      tested: A.koKey === koKey() && (A.knockouts || []).some((k) => k.res) }),
    context: () => {
      const info = traced(), n = A.sel.size;
      const sel = n ? ` · <b>${n}</b> unit${n === 1 ? "" : "s"} selected` : "";
      if (info) {
        const prev = A.tok[info.t]?.tx ?? info.token;
        return `Acting on <b>“${esc(visTok(info.target.text))}”</b> after “${esc(visTok(prev))}” · p ${fmtP(info.target.prob)}${sel}`;
      }
      if (n) return `Acting on <b>${n}</b> selected unit${n === 1 ? "" : "s"}: ${[...A.sel].slice(0, 4).map(shortName).join(", ")}${n > 4 ? "…" : ""}`;
      return !A.runId ? "No run open · ask the model in the conversation" : "Nothing selected · click a word in the conversation, or shift-drag units on the graph";
    },
    actions: [
      { id: "why", group: "Read", label: "Why this word", hint: "Where its logit came from, layer by layer, and the units behind it (W)",
        need: needWord, run: () => openTokenPanel() },
      { id: "runner", group: "Read", label: "Runner-up", hint: "Trace a word the model also considered instead",
        need: () => needWord() || (traced().alternatives.some((a) => a.id !== traced().target.id) ? null : "a word with runners-up"),
        choices: () => traced().alternatives.filter((a) => a.id !== traced().target.id).slice(0, 5).map((a) => ({
          label: `${visTok(a.text)} (p ${fmtP(a.prob)})`,
          run: () => { A.attrTarget = String(a.id); A.attrOf = null; openTokenPanel(); loadExplain(); } })) },
      { id: "follow", group: "Read", label: "Follow route", hint: "Walk the strongest route on the graph: START, each unit, END",
        need: () => needWord() || (A.trace?.routes?.length ? null : "the word's routes to load"), run: () => followRoute(A.traceFocus || 0) },
      { id: "paths", group: "Read", label: "Find paths", hint: "Routes out of or into the selection, between units, or behind the next token (P)",
        need: () => (A.atlas ? null : "the model's unit map"), run: () => (A.pathsOpen ? runPathSearch(true) : openPaths()) },
      { id: "jump", group: "Read", label: "Jump to a unit", hint: "Find a unit or a parameter by name (/)",
        need: () => (A.atlas ? null : "the model's unit map"), run: openPalette },
      { id: "ko5", group: "Test", label: "Knock out top 5", hint: "Remove the five units that pushed this word up and run it again",
        need: needWord, run: () => { openTokenPanel(); runKnockout(topUnits(traced(), 5, true), "top 5 units that pushed it up"); } },
      { id: "kosel", group: "Test", label: "Knock out selection", hint: "Remove the selected units and run the word again",
        need: () => (!A.sel.size ? "units selected on the graph (shift-click or shift-drag)" : A.cursor == null ? "a word: click one in the conversation" : null),
        run: () => { if (A.mode !== "attr") setMode("attr"); openTokenPanel(); knockOutSet([...A.sel], "selected units"); } },
      { id: "kopaths", group: "Test", label: "Knock out path units", hint: "Remove the units on the paths shown in the path explorer",
        need: () => (!(A.pathsOpen && A.paths.some((p) => p.on)) ? "paths shown in the path explorer" : A.cursor == null ? "a word: click one in the conversation" : null),
        run: () => pathAction("ko") },
      { id: "scale", group: "Test", label: "Scale selection", hint: "Scale the selected units' writes and read the word again: ×0.5 dampens, ×2 boosts",
        need: () => (!A.sel.size ? "units selected on the graph (shift-click or shift-drag)" : A.cursor == null ? "a word: click one in the conversation" : null),
        choices: () => SCALES.map(([f, what]) => ({ label: `×${f}: ${what}`,
          run: () => { if (A.mode !== "attr") setMode("attr"); openTokenPanel(); knockOutSet([...A.sel], "selected units", f); } })) },
      { id: "perm", group: "Change", label: "Make permanent", hint: "Turn the latest test (a knockout or a scale) into a weight edit, applied now",
        need: () => (lastKnockout() ? null : "a test on this word to keep: knock out or scale units first"), run: () => makePermanent(lastKnockout()) },
      { id: "lesion", group: "Change", label: "Lesion selection", hint: "Remove the selected units from the weights now, as one edit",
        need: () => (A.sel.size ? null : "units selected on the graph (shift-click or shift-drag)"), run: lesionSelection },
      { id: "forget", group: "Change", label: "Forget this fact", hint: "Guided: find the units that carry this word, try without them, measure, keep",
        need: needWord, run: () => forgetFromToken(traced()) },
      { id: "restore", group: "Change", label: "Restore original", hint: "Undo every applied weight edit, exactly",
        need: () => (A.applied ? null : "an applied weight edit"), run: restoreOriginal },
      { id: "check", group: "Verify", label: "Add as a check", hint: "Save: after this context, this word. Checks show what every edit changes",
        need: needWord, run: addCheckFromToken },
      { id: "rerun", group: "Verify", label: "Re-run the chat", hint: "Ask the chat's question again on the original and the edited weights",
        need: () => (!A.applied ? "an applied weight edit" : !A.runId ? "a chat" : null), run: rerunChat },
      { id: "checks", group: "Verify", label: "Run checks", hint: "Every saved behavior check, before and after the applied edits",
        run: () => { setEditsPanel(true); document.dispatchEvent(new CustomEvent("brainscan:checks", { detail: { run: true } })); } },
      { id: "cmp", group: "Verify", label: () => (A.cmp ? "vs original: on" : "vs original"), hint: "Color units by how the applied edits changed their effect on this word",
        need: () => (!A.applied ? "an applied weight edit" : needWord()), run: toggleCompare },
    ],
  });
  document.addEventListener("brainscan:edit-state", (ev) => { A.applied = ev.detail?.applied || 0; refreshBar("weights"); });
}

// ---------------------------------------------------------------- dock trace
function schedule() { if (!A.raf && A.visible) A.raf = requestAnimationFrame(() => { A.raf = 0; drawTrace(); overlay(); }); }

function sizeCanvas(cv, Hh) {
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth;
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(Hh * dpr)) {
    cv.width = Math.round(W * dpr); cv.height = Math.round(Hh * dpr); cv.style.height = `${Hh}px`;
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, W };
}

const TG = 12;
function viewIndex(t) {
  let lo = 0, hi = A.view.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (A.view[m] < t) lo = m + 1; else hi = m; }
  return Math.min(lo, Math.max(0, A.view.length - 1));
}
function traceLayout() { return { ev: [4, 10], bars: [18, 46], roles: [66, 3], ruler: [72, 14], height: 88 }; }

function drawTrace() {
  const cv = $("#an-trace"), lay = traceLayout(), { ctx, W } = sizeCanvas(cv, lay.height);
  ctx.clearRect(0, 0, W, lay.height);
  A.plotW = Math.max(50, W - 2 * TG);
  const n = A.view.length;
  if (A.fit) { A.colW = Math.max(0.05, A.plotW / Math.max(1, n)); A.scrollX = 0; }
  A.scrollX = clamp(A.scrollX, 0, Math.max(0, n * A.colW - A.plotW));
  const cw = A.colW, i0 = Math.max(0, Math.floor(A.scrollX / cw)), i1 = Math.min(n, Math.ceil((A.scrollX + A.plotW) / cw) + 1);
  const X = (i) => TG + i * cw - A.scrollX;
  $("#an-trace-note").textContent = A.selected != null ? `${unitName(A.selected)} · z per token, hits ≥ ${HIT_Z}σ marked` : `units hit per token (≥ ${HIT_Z}σ above their run average)`;
  if (!n) return;
  ctx.save(); ctx.beginPath(); ctx.rect(TG, 0, A.plotW, lay.height); ctx.clip();
  const gap = cw >= 4 ? 1 : 0;
  for (let i = i0; i < i1; i++) { const r = A.tok[A.view[i]]; ctx.fillStyle = C.roles[r.r] || C.roles.template; ctx.fillRect(X(i), lay.roles[0], Math.max(cw - gap, 0.6), lay.roles[1]); }
  const [by, bh] = lay.bars, sel = A.selected;
  let maxHits = 1;
  if (sel == null) for (let i = i0; i < i1; i++) { const hh = hitCount(A.view[i]); if (hh != null && hh > maxHits) maxHits = hh; }
  for (let i = i0; i < i1; i++) {
    const t = A.view[i], row = rowOf(t), x = X(i), w = Math.max(cw - gap, 0.6);
    if (row == null) { ctx.fillStyle = C.noData; ctx.fillRect(x, by + bh - 2, w, 2); continue; }
    if (sel != null) {
      const z = zOf(sel, codeAt(row, sel)), hh = clamp(z, 0, 6) / 6 * bh;
      ctx.fillStyle = z >= HIT_Z ? C.edgeCircuitHot : C.s1;
      ctx.globalAlpha = z >= HIT_Z ? 1 : 0.65;
      ctx.fillRect(x, by + bh - hh, w, hh);
    } else {
      const hh = (hitCount(t) / maxHits) * bh;
      ctx.fillStyle = C.graphRamp[Math.round(clamp(hitCount(t) / maxHits, 0.15, 1) * 255)];
      ctx.fillRect(x, by + bh - hh, w, hh);
    }
  }
  ctx.globalAlpha = 1;
  for (const e of A.events) {
    const tone = eventTone(e); if (!tone) continue;
    const x = X(viewIndex(e.t)) + cw / 2; if (x < TG - 4 || x > TG + A.plotW + 4) continue;
    ctx.fillStyle = tone; ctx.beginPath(); ctx.arc(x, lay.ev[0] + 4, 3, 0, Math.PI * 2); ctx.fill();
  }
  ctx.font = "10px 'JetBrains Mono', ui-monospace, monospace"; ctx.textBaseline = "top"; ctx.fillStyle = C.muted;
  const [uy] = lay.ruler;
  if (cw >= 22) {
    for (let i = i0; i < i1; i++) {
      const r = A.tok[A.view[i]];
      ctx.fillStyle = r.ph === "gen" ? C.ink : C.muted;
      ctx.save(); ctx.beginPath(); ctx.rect(X(i), uy, cw - 2, 14); ctx.clip(); ctx.fillText(visTok(r.tx), X(i) + 1, uy + 2); ctx.restore();
    }
  } else {
    const stepN = niceStep(Math.ceil(80 / cw));
    for (let i = Math.ceil(i0 / stepN) * stepN; i < i1; i += stepN) ctx.fillText(`t${A.view[i]}`, X(i) + 2, uy + 2);
  }
  const mark = (t, color, w) => {
    if (t == null) return; const i = viewIndex(t); if (A.view[i] !== t) return;
    ctx.fillStyle = color; ctx.fillRect(X(i) + cw / 2 - w / 2, 0, w, lay.roles[0] + lay.roles[1] + 1);
  };
  if (A.traceHover != null) mark(A.traceHover, C.ink2, 1);
  mark(A.cursor, C.selectRing, 2);
  ctx.restore();
}
function niceStep(raw) { const p = 10 ** Math.floor(Math.log10(raw)); return [1, 2, 5, 10].map((m) => m * p).find((s) => s >= raw); }
function eventTone(e) {
  const d = e.data || {};
  switch (e.type) {
    case "tool_call": return C.ink;
    case "tool_result": return C.ink2;
    case "error": return C.critical;
    case "test_result": return (d.failed || 0) + (d.errors || 0) ? C.critical : C.good;
    case "verdict": return d.success ? C.good : C.critical;
    case "cutoff": case "cancelled": case "file_edit": case "monitor": return C.serious;
    case "think_end": case "final_answer": return C.ink2;
    default: return null;
  }
}

function traceHit(mx) {
  if (mx < TG || mx > TG + A.plotW || !A.view.length) return null;
  const i = Math.floor((mx - TG + A.scrollX) / A.colW);
  return i >= 0 && i < A.view.length ? A.view[i] : null;
}

function placeTip(tip, wrap, html, mx, my) {
  tip.innerHTML = html; tip.hidden = false;
  const box = wrap.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = `${clamp(mx + 14, 4, box.width - w - 4)}px`;
  tip.style.top = `${my + 14 + h > box.height ? Math.max(4, my - h - 10) : my + 14}px`;
}

function showTraceTip(t, mx, my) {
  const tip = $("#an-trace-tip");
  if (t == null) { tip.hidden = true; return; }
  const r = A.tok[t], row = rowOf(t);
  let html = `<div><span class="tok">${esc(visTok(r.tx))}</span> <span class="muted">${esc(r.r)} · t${t}</span></div>`;
  if (row == null) html += `<div class="muted">no weight activity stored</div>`;
  else if (A.selected != null) { const c = codeAt(row, A.selected), z = zOf(A.selected, c); html += `<div class="row"><span>${unitName(A.selected)}</span><span>${z >= 0 ? "+" : ""}${z.toFixed(1)}σ · ${num(value(c))}</span></div>`; }
  else html += `<div class="row"><span>units ≥ ${HIT_Z}σ</span><span>${hitCount(t)}</span></div>`;
  const near = A.events.filter((e) => eventTone(e) && Math.abs(viewIndex(e.t) - viewIndex(t)) * A.colW < 6);
  for (const e of near.slice(0, 3)) html += `<div class="muted">${esc(e.type.replace("_", " "))}${e.data?.name ? ` · ${esc(e.data.name)}` : ""}</div>`;
  const wrap = $("#an-trace-wrap"), box = wrap.getBoundingClientRect();
  placeTip(tip, wrap, html, mx, Math.min(my, box.height - 4));
}

function overlay() {
  const ov = $("#an-overlay");
  let html = null;
  if (!A.atlas) {
    const err = A.atlasError;
    html = err?.status === 404 ? `<b>Weight anatomy is off</b><span>Start the recorder without <code>BRAINSCAN_ANATOMY=0</code>.</span>`
      : err && err.status !== 503 ? `<b>Could not load the weight graph</b><span>${esc(err.message)}</span>`
        : `<div class="spinner"></div><b>Mapping the weights…</b><span>The first start on a model clusters its MLP neurons and scores unit-to-unit edges once, then caches them.</span>`;
  }
  ov.hidden = html == null;
  if (html) ov.innerHTML = html;
}

// ---------------------------------------------------------------- graph hover tooltip
function unitName(g) { const u = A.units[g]; return `L${u.layer} · ${u.kind === "head" ? "head" : "cluster"} ${u.i}`; }

function graphTip(hit, mx, my) {
  const tip = $("#gv-tip");
  if (!hit || hit.g == null) {
    if (hit?.end) { placeTip(tip, $("#gv-stage"), `<div><b>${hit.end === "embed" ? "Token embedding" : "Unembedding"}</b></div><div class="muted">Click to open the matrix.</div>`, mx, my); return; }
    tip.hidden = true; return;
  }
  const g = hit.g, u = A.units[g], L = A.layers[u.layer], row = rowOf(A.cursor);
  const what = u.kind === "head" ? `${L.type === "linear_attention" ? "linear-attention" : "attention"} head · ${u.size} dims` : `MLP cluster · ${u.size} neurons`;
  let html = `<div class="tip-title">${unitName(g)}</div><div class="muted">${what}</div>`;
  if (row != null && A.mode !== "norm") {
    const c = codeAt(row, g), z = zOf(g, c);
    html += `<div class="row"><span>${u.kind === "head" ? "write norm" : "write mass"}</span><span>${num(value(c))}</span></div><div class="row"><span>vs. its run average</span><span>${z >= 0 ? "+" : ""}${z.toFixed(1)}σ</span></div>`;
  }
  html += `<div class="row"><span>‖output weights‖</span><span>${num(u.norm)}</span></div>`;
  if (u.pos) html += `<div class="toks"><span class="muted">↑</span> ${u.pos.map((s) => `<code>${esc(visTok(s))}</code>`).join(" ")}</div><div class="toks"><span class="muted">↓</span> ${u.neg.map((s) => `<code>${esc(visTok(s))}</code>`).join(" ")}</div>`;
  if (A.edges?.available) {
    const k = A.edges.k, a = A.edges.up[g * k], x = A.edges.up_x[g * k];
    if (x > 0) html += `<div class="row"><span>strongest input</span><span>${unitName(a)} · ${x.toFixed(1)}×</span></div>`;
  }
  if (A.edited.has(g)) html += `<div class="warn-line">Changed by an applied workbench edit</div>`;
  placeTip(tip, $("#gv-stage"), html, mx, my);
}

// ---------------------------------------------------------------- legend and toolbar
function renderLegend() {
  const attr = A.mode === "attr";
  const g = `linear-gradient(90deg, ${((attr ? C.attrStops : C.graphStops) || []).join(",")})`;
  const info = A.attrInfo;
  const scale = attr
    ? (info?.target && info.orig ? `edits weakened <b>${esc(visTok(info.target.text))}</b> ← · → strengthened it (edited − original, √ scale, max Δ ${(info.max ?? 0).toFixed(2)} logits)`
      : info?.target ? `pulls <b>${esc(visTok(info.target.text))}</b> down ← · → pushes it up (√ scale, max ${(info.max ?? 0).toFixed(2)} logits)` : "effect on the next token")
    : A.mode === "rel" ? "at or below its run average → 4σ above" : A.mode === "abs" ? "activity, log scale" : "output-weight norm, log scale";
  const e = A.edges;
  $("#gv-legend").innerHTML = `
    <div class="lg-row"><span class="lg-grad" style="background:${g}"></span><span>${scale}</span></div>
    ${A.trace && !A.pathsOpen ? `<div class="lg-row"><span class="lg-line" style="background:${C.busA}"></span><span>residual stream: carries the text left to right, from START <b>${esc(A.trace.start)}</b> to END <b>${esc(A.trace.end)}</b></span></div>
      <div class="lg-row"><span class="lg-line dashed" style="border-color:${C.path}"></span><span>START feeds the route's first unit</span></div>
      <div class="lg-row"><span class="lg-line" style="background:${C.pathFocus}"></span><span>each unit reads the one before; the last writes into END (numbered: the lit route)</span></div>`
      : A.paths.some((p) => p.on) ? `<div class="lg-row"><span class="lg-line" style="background:${C.path}"></span><span>highlighted paths</span><span class="lg-ring" style="border-color:${C.multiSelect}"></span><span>selected</span></div>` : ""}
    ${A.edgeMode !== "off" && !(A.trace && !A.pathsOpen && A.selected == null) ? `<div class="lg-row"><span class="lg-line" style="background:${C.edgeCircuit}"></span><span>${A.selected != null ? `<span style="color:${C.edgeIn}">inputs</span> · <span style="color:${C.edgeOut}">outputs</span> of the selected unit` : A.edgeMode === "circuit" ? "composition between units active now" : "edges of the unit under the cursor"}</span></div>` : ""}
    <div class="lg-row"><span class="lg-ring" style="border-color:${C.trail}"></span><span>hit in the last few tokens</span><span class="lg-dot" style="background:${C.edited}"></span><span>edited</span></div>
    <details><summary>How to read this</summary>
      <p><b>Nodes.</b> Attention heads sit above the residual stream, MLP neuron clusters below; one column per layer. ${esc(A.atlas?.measures?.head || "")} ${esc(A.atlas?.measures?.cluster || "")} ${esc(A.atlas?.measures?.clusters || "")}</p>
      <p><b>Color.</b> ${attr
        ? `Attribution: how much each unit's write at this token moved the logit of the target token, read through the final norm (held at its recorded scale) and the unembedding. Orange pushed it up, blue pulled it down. The units plus the embedding add up to the logit exactly; this is the direct effect only, not effects routed through later layers.${A.attrDiff
          ? " With <b>vs original</b> on, each unit is colored by how its effect changed when the applied weight edits are lifted: the same token traced on both weight sets, edited minus original. A lesioned unit shows its whole former effect, reversed; later units show how their own direct effect shifted once their inputs changed."
          : " When weight edits are applied, <b>vs original</b> in the dock compares against the unedited model."}`
        : `Each unit against its own average over the run, on a log scale; a unit counts as hit at ${HIT_Z}σ. Every weight takes part in every token, so this is how much a unit wrote, not an on/off firing.`}</p>
      <p><b>Selecting.</b> Click a unit to open it. Shift-click adds it to a selection; shift-drag a box to select many (alt-drag removes). Open <b>Paths</b> to trace routes from the selection, between units, or behind the next token.</p>
      <p><b>Edges.</b> ${esc(e?.measure || "")}${e?.available ? ` Median ${e.p50}×, 99th percentile ${e.p99}×.` : ""}</p>
      <p><b>Labels.</b> ${esc(A.atlas?.measures?.labels || "")}</p>
    </details>`;
}

function setSeg(group, attr, value) {
  document.querySelectorAll(`${group} [data-${attr}]`).forEach((b) => b.setAttribute("aria-pressed", String(b.dataset[attr] === value)));
}

function updateZoomLabel() {
  const z = $("#gv-zoom"), text = graph ? `${graph.zoomPercent()}%` : "";
  if (z && z.textContent !== text) z.textContent = text;  // runs every frame
}

// ---------------------------------------------------------------- detail drawer
function openDrawer(on) {
  $("#gv-drawer").hidden = !on;
  $("#gv-stage").classList.toggle("drawer-open", on);
  graph?.setState({ insets: { l: A.pathsOpen || A.editsOpen ? 396 : 0, r: on ? 452 : 0 } });
}

async function selectUnit(g, keepView = false) {
  if (g == null) { A.selected = null; A.detailMode = null; openDrawer(false); pushState(); renderLegend(); schedule(); return; }
  A.selected = g; A.detailMode = "unit";
  pushState(); renderLegend(); schedule();
  openDrawer(true);
  $("#an-unit-where").textContent = unitName(g);
  if (!keepView) $("#an-detail").innerHTML = `<div class="skeleton"></div><div class="skeleton short"></div><div class="skeleton tall"></div>`;
  try {
    const d = await api(`/brainscan/anatomy/units/${g}`);
    if (A.selected !== g) return;
    A.detail = d;
    renderDetail();
    if (keepView && H.base?.unit === g && d.matrices.some((m) => m.key === H.base.matrix)) {
      // The weights changed under an open view: reload the same window, keep the zoom history.
      const { stack, tile } = H;
      openView(H.base, tile?.window);
      H.stack = stack;
    } else openView({ unit: g, matrix: "writes" });
  } catch (err) { $("#an-detail").innerHTML = `<p class="empty">${esc(err.message)}</p>`; }
}

function renderDetail() {
  const d = A.detail, cluster = d.kind === "cluster";
  const layerType = d.layer_type === "linear_attention" ? "linear-attention layer" : d.layer_type === "full_attention" ? "full-attention layer" : "";
  const size = cluster ? `${d.size} neurons` : `${d.size} dims`;
  const chips = (arr) => (arr || []).map((s) => `<span class="chip">${esc(visTok(s))}</span>`).join("");
  const mtabs = d.matrices.map((m) => `<button type="button" role="tab" data-key="${esc(m.key)}" title="${esc(m.desc)}">${esc(m.title.replace("Input weights · ", ""))} <span class="dim">${m.shape[0]}×${m.shape[1]}</span></button>`).join("");
  const e = A.edges, edgeRows = (dir) => {
    if (!e?.available) return "";
    const k = e.k, ids = dir === "in" ? e.up : e.down, xs = dir === "in" ? e.up_x : e.down_x, out = [];
    for (let j = 0; j < Math.min(k, 6); j++) { const o = ids[d.g * k + j], x = xs[d.g * k + j]; if (x > 0) out.push(`<li data-g="${o}"><span>${unitName(o)}</span><span class="bar"><i style="width:${clamp(x / (e.p99 * 1.4), 0.04, 1) * 100}%"></i></span><span class="x">${x.toFixed(1)}×</span></li>`); }
    return out.join("") || `<li class="muted">none</li>`;
  };
  $("#an-detail").innerHTML = `
    <div class="d-hero">
      <div class="d-kicker">${cluster ? "MLP neuron cluster" : "Attention head"}${layerType ? ` · ${layerType}` : ""}</div>
      <div class="d-title">${esc(d.title)}</div>
      <div class="d-stats">
        <div><span>size</span><b>${size}</b></div>
        <div><span>‖output‖</span><b>${num(d.norm)}</b></div>
        <div title="Participation ratio of the squared singular values of the output weights"><span>eff. rank</span><b>${d.effective_rank}</b></div>
      </div>
      ${d.edited ? `<div class="pill warn">Changed by an applied edit</div>` : ""}
      <div class="d-actions">
        <button type="button" id="an-lesion" class="btn" title="Adds an edit that zeroes this unit's output weights to the edit stack in Tune. Nothing changes until you apply the stack.">Lesion in Tune</button>
        <button type="button" id="an-focus" class="btn ghost">Center in graph</button>
        ${A.mode === "attr" && A.attrInfo?.target ? `<button type="button" id="an-back-token" class="btn ghost" title="Back to the token this unit helped explain">← ${esc(visTok(A.attrInfo.target.text))}</button>` : ""}
      </div>
      <div class="wb-msg" id="an-lesion-msg" role="status"></div>
    </div>
    <section class="d-sec"><h4>At this token <span class="dim">neurons, attention, effect on the next token</span></h4><div id="an-token-sec"></div></section>
    <section class="d-sec"><h4>In this run</h4><div id="an-run-stats"></div></section>
    <section class="d-sec"><h4>What it writes</h4>
      <div class="chips"><span class="lbl">pushes up</span>${chips(d.pos_now)}</div>
      <div class="chips"><span class="lbl">pushes down</span>${chips(d.neg_now)}</div>
      <p class="hint">${cluster ? "The cluster's sign-aligned mean output direction" : "The head's top output direction (its largest singular vector; the sign is arbitrary, so up and down can swap)"} read through the final norm and unembedding, on the current weights. Read from the weights, not measured on text.</p>
    </section>
    ${e?.available ? `<section class="d-sec"><h4>Connections <span class="dim">weight-based composition, × chance</span></h4>
      <div class="conn"><div><div class="conn-h" style="--c:${C.edgeIn}">Reads from</div><ul class="conn-list">${edgeRows("in")}</ul></div>
      <div><div class="conn-h" style="--c:${C.edgeOut}">Read by</div><ul class="conn-list">${edgeRows("out")}</ul></div></div></section>` : ""}
    <section class="d-sec"><h4>Weights</h4>
      <div class="mtabs" role="tablist" aria-label="Weight matrix">${mtabs}</div>
      ${heatSection()}</section>
    ${cluster ? `<section class="d-sec"><h4>Neurons <span class="dim">${d.neurons.length < d.n_neurons ? `largest ${d.neurons.length} of ${d.n_neurons}` : d.n_neurons}</span></h4>
    <div class="ntable-wrap"><table class="ntable"><thead><tr><th>neuron</th><th>‖w‖</th><th title="|cosine| between the neuron's output weights and the cluster direction">fit</th><th>pushes up</th></tr></thead><tbody>
    ${d.neurons.map((n) => `<tr data-row="${n.row}" title="Show this neuron's weights"><td>${n.n}</td><td>${num(n.norm)}</td><td>${n.cos.toFixed(2)}</td><td class="t">${n.top.map((s) => esc(visTok(s))).join(" ")}</td></tr>`).join("")}
    </tbody></table></div></section>` : ""}`;
  $("#an-lesion").onclick = lesion;
  $("#an-focus").onclick = () => graph.flyTo(d.g, Math.max(graph.target.s, 2.6));
  $("#an-back-token")?.addEventListener("click", openTokenPanel);
  renderTokenSection();
  document.querySelectorAll("#an-detail .conn-list li[data-g]").forEach((li) => { li.onclick = () => { const g = Number(li.dataset.g); selectUnit(g); graph.flyTo(g, Math.max(graph.target.s, 1.6)); }; });
  document.querySelectorAll("#an-detail .mtabs button").forEach((b) => { b.onclick = () => openView({ unit: d.g, matrix: b.dataset.key }); });
  document.querySelectorAll("#an-detail .ntable tbody tr").forEach((tr) => {
    tr.onclick = () => { const r = Number(tr.dataset.row); openView({ unit: d.g, matrix: H.base?.matrix || "writes" }, [r, r + 1, 0, H.tile?.cols || 1e9], true); };
  });
  wireHeat();
  renderRunStats();
}

function heatSection() {
  return `<p class="hint" id="an-view-desc"></p>
    <div class="heat-tools">
      <select id="an-heat-mode" aria-label="What each bin shows when it holds many weights">
        <option value="auto">RMS per bin</option><option value="mean">mean per bin</option><option value="absmax">largest magnitude</option></select>
      <button type="button" class="btn sm" id="an-zoom-out" disabled>Zoom out</button>
      <button type="button" class="btn sm" id="an-zoom-reset" disabled>Reset</button>
    </div>
    <div class="crumb" id="an-crumb"></div>
    <div class="heat-wrap" id="an-heat-wrap"><canvas id="an-heat" aria-label="Weight heatmap; drag to zoom into a region"></canvas><div class="tooltip" id="an-heat-tip" hidden></div></div>
    <div class="legend" id="an-heat-legend"></div>
    <div class="mini-charts"><div id="an-hist"></div><div id="an-spec"></div></div>
    <div class="facts" id="an-stats"></div>`;
}

function renderRunStats() {
  const box = $("#an-run-stats"); if (!box || A.selected == null) return;
  const g = A.selected;
  if (!A.n) { box.innerHTML = `<p class="empty">${A.runId ? "No weight activity recorded for this run yet." : "No run selected."}</p>`; return; }
  const scored = [];
  let sumV = 0, peak = 0, hits = 0;
  for (const t of A.view) {
    const row = rowOf(t); if (row == null) continue;
    const c = codeAt(row, g), z = zOf(g, c), v = value(c);
    sumV += v; peak = Math.max(peak, v); if (z >= HIT_Z) hits++;
    scored.push([z, t]);
  }
  scored.sort((a, b) => b[0] - a[0]);
  const picked = [];
  for (const [z, t] of scored) { if (picked.length >= 8) break; if (picked.every(([, u]) => Math.abs(viewIndex(u) - viewIndex(t)) > 2)) picked.push([z, t]); }
  const ctx = (t) => {
    const i = viewIndex(t), part = (a, b) => A.view.slice(Math.max(0, a), Math.max(0, b)).map((u) => esc(visTok(A.tok[u].tx))).join("");
    return `${part(i - 6, i)}<b>${esc(visTok(A.tok[t].tx))}</b>${part(i + 1, i + 4)}`;
  };
  box.innerHTML = `<div class="d-stats small"><div><span>average</span><b>${num(sumV / Math.max(1, scored.length))}</b></div><div><span>peak</span><b>${num(peak)}</b></div>
      <div><span>hit at</span><b>${hits} / ${scored.length}</b></div></div>
    <ul class="toplist">${picked.map(([z, t]) => `<li data-t="${t}"><span class="dim">t${t}</span><span class="ctx">${ctx(t)}</span><span class="z">${z >= 0 ? "+" : ""}${z.toFixed(1)}σ</span></li>`).join("")}</ul>`;
  box.querySelectorAll("li[data-t]").forEach((li) => { li.onclick = () => { setFollow(false); stopPlay(); setCursor(Number(li.dataset.t), "jump"); }; });
}

function throttledRunStats() {
  if (A.statsTimer) return;
  A.statsTimer = setTimeout(() => { A.statsTimer = 0; renderRunStats(); }, 1000);
}

async function lesion(ev) {
  const g = A.selected; ev.target.disabled = true;
  const msg = $("#an-lesion-msg");
  try {
    const e = await post(`/brainscan/anatomy/units/${g}/lesion`, { factor: 0 });
    msg.className = "wb-msg ok";
    msg.innerHTML = `Added “${esc(e.summary)}” to the edit stack in Tune. Apply it there; recordings made while it is applied are tagged.`;
  } catch (err) { msg.className = "wb-msg error"; msg.textContent = err.message; }
  ev.target.disabled = false;
}

function openParam(name) {
  A.detailMode = "param"; A.selected = null; A.detail = null; pushState(); renderLegend(); schedule();
  openDrawer(true);
  const p = (A.params || []).find((x) => x.name === name);
  $("#an-unit-where").textContent = "Parameter";
  $("#an-detail").innerHTML = `<div class="d-hero"><div class="d-kicker">Parameter</div><div class="d-title mono">${esc(name)}</div>
    ${p ? `<div class="d-stats"><div><span>shape</span><b>${p.shape.join(" × ")}</b></div><div><span>weights</span><b>${p.numel.toLocaleString()}</b></div><div><span>dtype</span><b>${esc(p.dtype)}</b></div></div>` : ""}</div>
    <section class="d-sec"><h4>Weights</h4>${heatSection()}</section>`;
  wireHeat();
  openView({ param: name });
}

async function openEnd(which) {
  await loadParams();
  const name = which === "embed" ? A.atlas.embed?.name : A.atlas.unembed?.name;
  if (name) openParam(name);
}

// ---------------------------------------------------------------- heatmap of weights
function wireHeat() {
  $("#an-heat-mode").value = H.mode;
  $("#an-heat-mode").onchange = (ev) => { H.mode = ev.target.value; loadTile(H.tile?.window); };
  $("#an-zoom-out").onclick = () => { const prev = H.stack.pop(); loadTile(prev); };
  $("#an-zoom-reset").onclick = () => { H.stack = []; loadTile(null); };
  const cv = $("#an-heat");
  cv.onpointerdown = (ev) => { const p = heatPoint(ev); if (!p) return; H.drag = { a: p, b: p }; cv.setPointerCapture(ev.pointerId); };
  cv.onpointermove = (ev) => {
    const p = heatPoint(ev);
    if (H.drag) { H.drag.b = p || H.drag.b; drawHeat(); return; }
    heatTip(p);
  };
  cv.onpointerleave = () => { $("#an-heat-tip").hidden = true; };
  cv.onpointerup = () => {
    const d = H.drag; H.drag = null;
    if (!d || !H.tile) return;
    const t = H.tile, [r0, , c0] = t.window;
    const i0 = Math.min(d.a.i, d.b.i), i1 = Math.max(d.a.i, d.b.i), j0 = Math.min(d.a.j, d.b.j), j1 = Math.max(d.a.j, d.b.j);
    if (i1 - i0 < 1 && j1 - j0 < 1 && (t.bh > 1 || t.bw > 1)) {  // a click: zoom in 4x around the point
      const ch = Math.max(1, Math.round((t.window[1] - t.window[0]) / 4)), cw = Math.max(1, Math.round((t.window[3] - t.window[2]) / 4));
      const rc = r0 + (i0 + 0.5) * t.bh, cc = c0 + (j0 + 0.5) * t.bw;
      zoomTo([Math.round(rc - ch / 2), Math.round(rc + ch / 2), Math.round(cc - cw / 2), Math.round(cc + cw / 2)]);
      return;
    }
    if (i1 - i0 < 1 && j1 - j0 < 1) { drawHeat(); return; }
    zoomTo([r0 + i0 * t.bh, r0 + (i1 + 1) * t.bh, c0 + j0 * t.bw, c0 + (j1 + 1) * t.bw]);
  };
}

function zoomTo(win) {
  const t = H.tile; if (!t) return;
  let [a, b, c, d] = win;
  a = clamp(a, 0, t.rows - 1); b = clamp(b, a + 1, t.rows); c = clamp(c, 0, t.cols - 1); d = clamp(d, c + 1, t.cols);
  H.stack.push(t.window);
  loadTile([a, b, c, d]);
}

// Show a view (a unit's matrix or a parameter). With push, the current window goes on the zoom
// stack first, so "Zoom out" comes back to it.
function openView(base, win = null, push = false) {
  const same = JSON.stringify(base) === JSON.stringify(H.base);
  if (push && same && H.tile) H.stack.push(H.tile.window); else H.stack = [];
  H.base = base;
  document.querySelectorAll("#an-detail .mtabs button").forEach((b) => {
    const on = b.dataset.key === base.matrix; b.classList.toggle("active", on); b.setAttribute("aria-selected", String(on));
  });
  loadTile(win);
}

async function loadTile(win) {
  const cv = $("#an-heat"); if (!cv || !H.base) return;
  const seq = ++H.seq;
  const plotW = Math.max(64, cv.clientWidth - 40);
  const body = { ...H.base, mode: H.mode, w: Math.min(1024, Math.round(plotW)), h: 320 };
  if (win) Object.assign(body, { r0: win[0], r1: win[1], c0: win[2], c1: win[3] });
  $("#an-crumb").textContent = "loading…";
  try {
    const tile = await post("/brainscan/anatomy/tile", body);
    if (seq !== H.seq) return;
    H.tile = tile; H.vals = b64Halves(tile.data);
    // Color range: RMS bins from their 0.5th to 99.5th percentile (they are all positive and often
    // similar, so starting at zero would wash them out); signed values symmetric around zero.
    const rms = tile.mode === "rms";
    const vals = Array.from(H.vals, rms ? (x) => x : Math.abs).filter(Number.isFinite).sort((a, b) => a - b);
    const pct = (q) => (vals.length ? vals[Math.min(vals.length - 1, Math.floor(vals.length * q))] : 0);
    H.lo = rms ? pct(0.005) : 0;
    H.scale = pct(0.995) || vals[vals.length - 1] || 1;
    if (H.scale <= H.lo) H.scale = H.lo + (Math.abs(H.lo) || 1) * 1e-3;
    drawHeat(); renderHeatInfo();
  } catch (err) { $("#an-crumb").textContent = err.message; }
}

const HEAT = { left: 30, bottom: 32, top: 2 };
function heatGeometry() {
  const t = H.tile, cv = $("#an-heat"), W = cv.clientWidth;
  const plotW = W - HEAT.left - 2;
  const plotH = clamp(t.h * Math.max(1, Math.floor(plotW / Math.max(t.w, 1))), 110, 300);
  return { W, plotW, plotH, height: HEAT.top + plotH + HEAT.bottom, x0: HEAT.left, y0: HEAT.top, cw: plotW / t.w, rh: plotH / t.h };
}

function heatPoint(ev) {
  if (!H.tile) return null;
  const g = heatGeometry(), r = $("#an-heat").getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
  const j = Math.floor((mx - g.x0) / g.cw), i = Math.floor((my - g.y0) / g.rh);
  if (i < 0 || j < 0 || i >= H.tile.h || j >= H.tile.w) return null;
  return { i, j, mx, my };
}

function drawHeat() {
  const t = H.tile, cv = $("#an-heat"); if (!t || !cv) return;
  const g = heatGeometry(), { ctx, W } = sizeCanvas(cv, g.height);
  ctx.clearRect(0, 0, W, g.height);
  // the weights, one pixel per bin, scaled up without smoothing: RMS on the sequential ramp, signed
  // values on a diverging one with zero at the surface color
  const img = new ImageData(t.w, t.h), px = img.data, signed = t.mode !== "rms", inv = 1 / (H.scale - H.lo);
  for (let k = 0; k < t.w * t.h; k++) {
    const raw = H.vals[k], v = Number.isFinite(raw) ? (raw - H.lo) * inv : 0;  // a NaN weight must not poison the ImageData
    const c = signed ? C.div[Math.round((clamp(v, -1, 1) + 1) * 127.5)] : C.ramp[Math.round(clamp(v, 0, 1) * 255)];
    px[4 * k] = c[0]; px[4 * k + 1] = c[1]; px[4 * k + 2] = c[2]; px[4 * k + 3] = 255;
  }
  const off = new OffscreenCanvas(t.w, t.h); off.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.save(); roundRectPath(ctx, g.x0, g.y0, g.plotW, g.plotH, 8); ctx.clip();
  ctx.drawImage(off, g.x0, g.y0, g.plotW, g.plotH); ctx.restore();
  ctx.strokeStyle = C.axis; ctx.lineWidth = 1; roundRectPath(ctx, g.x0 - 0.5, g.y0 - 0.5, g.plotW + 1, g.plotH + 1, 8); ctx.stroke();
  // RMS of each row bin (left) and column bin (below): outlier rows and residual dims stand out
  const rmax = Math.max(...t.row_rms, 1e-12), cmax = Math.max(...t.col_rms, 1e-12);
  ctx.fillStyle = C.edgeIn;
  t.row_rms.forEach((v, i) => { const w = (v / rmax) * (HEAT.left - 8); ctx.fillRect(HEAT.left - 4 - w, g.y0 + i * g.rh, w, Math.max(g.rh - (g.rh > 3 ? 0.5 : 0), 0.5)); });
  const cy = g.y0 + g.plotH + 4, ch = 16;
  ctx.fillStyle = C.edgeOut;
  t.col_rms.forEach((v, j) => { const hh = (v / cmax) * ch; ctx.fillRect(g.x0 + j * g.cw, cy + ch - hh, Math.max(g.cw - (g.cw > 3 ? 0.5 : 0), 0.5), hh); });
  ctx.fillStyle = C.muted; ctx.font = "10px Inter, system-ui, sans-serif"; ctx.textBaseline = "top"; ctx.textAlign = "left";
  ctx.fillText(`${t.cols_label} ${t.window[2]}–${t.window[3] - 1} · RMS per column`, g.x0, cy + ch + 2);
  if (H.drag && H.drag.a && H.drag.b) {
    const a = H.drag.a, b = H.drag.b;
    const x = g.x0 + Math.min(a.j, b.j) * g.cw, y = g.y0 + Math.min(a.i, b.i) * g.rh;
    ctx.strokeStyle = C.selectRing; ctx.lineWidth = 2;
    ctx.strokeRect(x, y, (Math.abs(a.j - b.j) + 1) * g.cw, (Math.abs(a.i - b.i) + 1) * g.rh);
  }
}

function roundRectPath(ctx, x, y, w, h, r) {
  ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

function binRange(t, i, j) {
  const [r0, r1, c0, c1] = t.window;
  return { ra: r0 + i * t.bh, rb: Math.min(r0 + (i + 1) * t.bh, r1) - 1, ca: c0 + j * t.bw, cb: Math.min(c0 + (j + 1) * t.bw, c1) - 1 };
}

function rowName(r) {
  const d = A.detail;
  if (A.detailMode === "unit" && d?.neuron_ids && H.base?.unit === d.g) return `neuron ${d.neuron_ids[r]}`;
  return `${H.tile.rows_label === "head dims" ? "head dim" : "row"} ${r}`;
}

function heatTip(p) {
  const tip = $("#an-heat-tip");
  if (!p) { tip.hidden = true; return; }
  const t = H.tile, b = binRange(t, p.i, p.j), v = H.vals[p.i * t.w + p.j];
  const rows = b.ra === b.rb ? rowName(b.ra) : `rows ${b.ra}–${b.rb}`, cols = b.ca === b.cb ? `dim ${b.ca}` : `cols ${b.ca}–${b.cb}`;
  const what = t.mode === "value" ? "weight" : t.mode === "rms" ? "RMS" : t.mode === "absmax" ? "largest magnitude" : "mean";
  placeTip(tip, $("#an-heat-wrap"), `<div>${esc(rows)} · ${esc(cols)}</div><div class="row"><span>${what}</span><span>${num(v, 4)}</span></div>
    ${t.exact ? "" : `<div class="muted">${(b.rb - b.ra + 1) * (b.cb - b.ca + 1)} weights in this bin · drag to zoom</div>`}`, p.mx, p.my);
}

function renderHeatInfo() {
  const t = H.tile, s = t.stats;
  $("#an-view-desc").textContent = t.desc;
  $("#an-zoom-out").disabled = !H.stack.length; $("#an-zoom-reset").disabled = !H.stack.length;
  const [r0, r1, c0, c1] = t.window;
  $("#an-crumb").textContent = `${t.rows_label} ${r0}–${r1 - 1} × ${t.cols_label} ${c0}–${c1 - 1}` + (t.exact ? " · one weight per pixel" : ` · ${t.bh}×${t.bw} weights per bin`);
  const grad = t.mode === "rms" ? `linear-gradient(90deg, ${C.rampStops.join(",")})` : `linear-gradient(90deg, rgb(${C.neg}), rgb(${C.mid}), rgb(${C.pos}))`;
  $("#an-heat-legend").innerHTML = `<span class="item"><span class="grad" style="background:${grad}"></span>${t.mode === "rms" ? `${num(H.lo, 4)} → ${num(H.scale, 4)}` : `−${num(H.scale, 4)} → +${num(H.scale, 4)}`}
    <span class="dim">${t.mode === "value" ? "each weight" : t.mode === "rms" ? "RMS per bin" : t.mode === "absmax" ? "largest magnitude per bin" : "mean per bin"},
    ${t.mode === "rms" ? "0.5th–99.5th percentile" : "99.5th percentile of |value|"}; values outside saturate</span></span>`;
  $("#an-stats").innerHTML = `<span><b>${s.n.toLocaleString()}</b> weights</span><span>‖W‖ <b>${num(s.norm)}</b></span>
    <span>mean <b>${num(s.mean, 5)}</b></span><span>std <b>${num(s.std, 5)}</b></span><span>|max| <b>${num(s.absmax, 4)}</b></span>${s.zeros ? `<span>zeros <b>${s.zeros.toLocaleString()}</b></span>` : ""}`;
  $("#an-hist").innerHTML = histogramSvg(s);
  $("#an-spec").innerHTML = A.detailMode === "unit" && A.detail ? spectrumSvg(A.detail) : "";
}

function histogramSvg(s) {
  const w = 190, h = 70, n = s.hist.length, max = Math.max(...s.hist, 1), bw = (w - 8) / n;
  const bars = s.hist.map((c, i) => { const bh = c ? Math.max(1, (Math.log1p(c) / Math.log1p(max)) * (h - 26)) : 0; return `<rect x="${4 + i * bw}" y="${h - 16 - bh}" width="${Math.max(bw - 0.5, 0.5)}" height="${bh}" rx="0.5" fill="${C.edgeIn}"/>`; }).join("");
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="Histogram of all weight values in this view">
    <text x="4" y="10" font-size="10" fill="${C.ink2}">values (log counts)</text>${bars}
    <line x1="${w / 2}" x2="${w / 2}" y1="14" y2="${h - 16}" stroke="${C.axis}" stroke-dasharray="2 2"/>
    <text x="4" y="${h - 3}" font-size="9" fill="${C.muted}">${num(s.range[0], 3)}</text><text x="${w - 4}" y="${h - 3}" font-size="9" fill="${C.muted}" text-anchor="end">${num(s.range[1], 3)}</text></svg>`;
}

function spectrumSvg(d) {
  const sv = d.spectrum || []; if (!sv.length) return "";
  const w = 190, h = 70, max = sv[0] || 1, bw = (w - 8) / sv.length;
  const bars = sv.map((v, i) => { const bh = Math.max(0.5, (v / max) * (h - 26)); return `<rect x="${4 + i * bw}" y="${h - 16 - bh}" width="${Math.max(bw - 0.5, 0.5)}" height="${bh}" rx="0.5" fill="${C.edgeOut}"/>`; }).join("");
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="Singular values of the output weights">
    <text x="4" y="10" font-size="10" fill="${C.ink2}">singular values</text>${bars}
    <text x="4" y="${h - 3}" font-size="9" fill="${C.muted}">top ${sv.length} · eff. rank ${d.effective_rank}</text></svg>`;
}

// ---------------------------------------------------------------- weight thumbnails on cards
const thumbQueue = [], thumbPending = new Set();
let thumbActive = 0;
function needThumb(g, level) {
  const key = `${g}:${level}`;
  if (thumbPending.has(key)) return;
  thumbPending.add(key); thumbQueue.push([g, level, key]);
  pumpThumbs();
}
function pumpThumbs() {
  while (thumbActive < 3 && thumbQueue.length) {
    const [g, level, key] = thumbQueue.pop();  // newest first: what is on screen now
    thumbActive++;
    post("/brainscan/anatomy/tile", { unit: g, matrix: "writes", mode: "rms", h: level ? 48 : 16, w: level ? 160 : 56 })
      .then(async (t) => {
        const v = b64Halves(t.data), sorted = Array.from(v).sort((a, b) => a - b);
        const lo = sorted[Math.floor(sorted.length * 0.01)], hi = sorted[Math.floor(sorted.length * 0.99)] || lo + 1e-6;
        const img = new ImageData(t.w, t.h);
        for (let k = 0; k < v.length; k++) { const x = (v[k] - lo) / (hi - lo); const c = C.ramp[Number.isFinite(x) ? Math.round(clamp(x, 0, 1) * 255) : 0]; img.data.set([c[0], c[1], c[2], 255], 4 * k); }
        graph.setThumb(key, await createImageBitmap(img));
      })
      .catch(() => {})
      .finally(() => { thumbActive--; thumbPending.delete(key); pumpThumbs(); });  // allow a re-request after clear
  }
  if (thumbQueue.length > 40) { for (const [, , key] of thumbQueue.splice(0, thumbQueue.length - 40)) thumbPending.delete(key); }
}
// Drop cached thumbnails (and in-flight bookkeeping) so they regenerate — after a theme change or weight edit.
function resetThumbs() { graph.clearThumbs(); thumbPending.clear(); thumbQueue.length = 0; }

// ---------------------------------------------------------------- jump-to palette (⌘K)
async function loadParams() {
  if (A.params) return;
  try { A.params = await api("/brainscan/anatomy/params"); } catch { A.params = []; }
}
const P = { items: [], active: 0 };
const bare = (s) => (s || "").replace(/^[\sĠ▁·]+/, "").toLowerCase();
function openPalette() {
  if (!A.atlas) return;
  $("#gv-palette").hidden = false;
  const q = $("#gv-q"); q.value = ""; q.focus();
  loadParams().then(searchPalette);
  searchPalette();
}
function closePalette() { $("#gv-palette").hidden = true; }
function searchPalette() {
  const q = $("#gv-q").value.trim().toLowerCase(), items = [];
  const m = q.match(/^l?\s*(\d+)\s*[\s.:·-]?\s*(h|head|c|cluster)?\s*(\d+)?$/);
  if (m && A.layers[Number(m[1])]) {
    const l = A.layers[Number(m[1])], kind = m[2] ? m[2][0] : null, idx = m[3] != null ? Number(m[3]) : null;
    for (let i = 0; i < l.heads + l.clusters && items.length < 10; i++) {
      const g = l.offset + i, u = A.units[g], k = u.kind === "head" ? "h" : "c";
      if ((kind && kind !== k) || (idx != null && u.i !== idx)) continue;
      items.push({ type: "unit", g });
    }
  } else if (q) {
    for (const u of A.units) {
      if (items.length >= 10) break;
      if ([...(u.pos || []), ...(u.neg || [])].some((s) => bare(s) === q || (q.length >= 3 && bare(s).startsWith(q)))) items.push({ type: "unit", g: u.g });
    }
  }
  for (const p of A.params || []) {
    if (items.length >= 18) break;
    if (q && p.name.toLowerCase().includes(q)) items.push({ type: "param", name: p.name, shape: p.shape });
  }
  if (!q) items.push(...["embed", "unembed"].map((w) => ({ type: "end", end: w })));
  P.items = items; P.active = 0;
  renderPalette();
}
function renderPalette() {
  $("#gv-results").innerHTML = P.items.map((it, i) => {
    const on = i === P.active ? " class=\"on\"" : "";
    if (it.type === "unit") { const u = A.units[it.g]; return `<li${on} data-i="${i}" role="option"><span class="kind">${u.kind === "head" ? "head" : "cluster"}</span><b>${unitName(it.g)}</b><span class="dim">${(u.pos || []).slice(0, 3).map((s) => esc(visTok(s))).join(" ")}</span></li>`; }
    if (it.type === "param") return `<li${on} data-i="${i}" role="option"><span class="kind">param</span><b class="mono">${esc(it.name)}</b><span class="dim">${it.shape.join("×")}</span></li>`;
    return `<li${on} data-i="${i}" role="option"><span class="kind">matrix</span><b>${it.end === "embed" ? "Token embedding" : "Unembedding"}</b><span class="dim">${(A.atlas[it.end]?.shape || []).join("×")}</span></li>`;
  }).join("") || `<li class="none">No match. Try “L12 c7”, “L3 h5”, a token, or a parameter name.</li>`;
  $("#gv-results").querySelectorAll("li[data-i]").forEach((li) => { li.onclick = () => choose(Number(li.dataset.i)); });
}
function choose(i) {
  const it = P.items[i]; if (!it) return;
  closePalette();
  if (it.type === "unit") { selectUnit(it.g); graph.flyTo(it.g, Math.max(graph.target.s, 2.6)); }
  else if (it.type === "param") openParam(it.name);
  else openEnd(it.end);
}

// ---------------------------------------------------------------- icons
const ICON = {
  play: `<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M4 2.5v11l9-5.5z" fill="currentColor"/></svg>`,
  pause: `<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M4 2.5h3v11H4zM9 2.5h3v11H9z" fill="currentColor"/></svg>`,
};

// ---------------------------------------------------------------- input wiring
function setupInput() {
  const tr = $("#an-trace");
  let drag = null;
  tr.tabIndex = 0;
  tr.addEventListener("pointerdown", (ev) => { drag = { x: ev.clientX, scroll: A.scrollX, moved: false }; tr.setPointerCapture(ev.pointerId); });
  tr.addEventListener("pointermove", (ev) => {
    const r = tr.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
    if (drag) {
      const dx = ev.clientX - drag.x;
      if (Math.abs(dx) > 3) drag.moved = true;
      if (drag.moved) { A.fit = false; A.scrollX = drag.scroll - dx; schedule(); return; }
    }
    const t = traceHit(mx);
    if (t !== A.traceHover) { A.traceHover = t; schedule(); }
    showTraceTip(t, mx, my);
  });
  tr.addEventListener("pointerleave", () => { A.traceHover = null; $("#an-trace-tip").hidden = true; schedule(); });
  tr.addEventListener("pointerup", (ev) => {
    const moved = drag?.moved; drag = null;
    if (moved) return;
    const r = tr.getBoundingClientRect(), t = traceHit(ev.clientX - r.left);
    if (t != null) { setFollow(false); stopPlay(); setCursor(t, "jump"); }
  });
  tr.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    const r = tr.getBoundingClientRect(), mx = ev.clientX - r.left;
    if (Math.abs(ev.deltaX) > Math.abs(ev.deltaY)) { A.fit = false; A.scrollX += ev.deltaX; schedule(); return; }
    const anchor = (mx - TG + A.scrollX) / A.colW;
    A.fit = false;
    A.colW = clamp(A.colW * (ev.deltaY < 0 ? 1.25 : 0.8), 0.05, 40);
    A.scrollX = anchor * A.colW - (mx - TG);
    schedule();
  }, { passive: false });
  $("#an-fit").onclick = () => { A.fit = true; schedule(); };
  document.querySelectorAll("#gv-mode [data-mode]").forEach((b) => { b.onclick = () => setMode(b.dataset.mode); });
  $("#gv-convo-btn").onclick = () => setConvo(!CV.open);
  $("#gv-edits-btn").onclick = () => setEditsPanel(!A.editsOpen);
  $("#gv-edits-close").onclick = () => setEditsPanel(false);
  $("#tv-convo-hide").onclick = () => setConvo(false);
  $("#tv-show-template").onchange = (ev) => { CV.showTemplate = ev.target.checked; renderConvo(true); };
  $("#tv-convo-body").addEventListener("click", (ev) => {
    const el = ev.target.closest(".ct[data-t]");
    if (el) traceWord(Number(el.dataset.t));
  });
  $("#gv-selko").onclick = () => {
    if (A.mode !== "attr") setMode("attr");
    openTokenPanel();
    knockOutSet([...A.sel], "selected units");
  };
  $("#gv-paths-btn").onclick = () => (A.pathsOpen ? closePaths() : openPaths());
  $("#gv-paths-close").onclick = closePaths;
  $("#gv-selclear").onclick = clearSelection;
  $("#gv-selpaths").onclick = () => openPaths(A.pathMode === "explain" ? "down" : A.pathMode);
  document.querySelectorAll("#gv-path-mode [data-pm]").forEach((b) => { b.onclick = () => { A.pathMode = b.dataset.pm; renderPathControls(); runPathSearch(true); }; });
  const pathInput = (id, key) => { $(id).oninput = (ev) => { A[key] = Number(ev.target.value); $(`${id}-v`).textContent = A[key]; runPathSearch(false); }; };
  pathInput("#gv-path-hops", "pathHops");
  pathInput("#gv-path-branch", "pathBranch");
  $("#gv-path-activity").onchange = (ev) => { A.pathActivity = ev.target.checked; runPathSearch(false); };
  document.querySelectorAll("#gv-edges [data-edges]").forEach((b) => { b.onclick = () => { A.edgeMode = b.dataset.edges; setSeg("#gv-edges", "edges", A.edgeMode); pushState(); renderLegend(); }; });
  $("#gv-trail").onclick = () => { A.trailOn = !A.trailOn; $("#gv-trail").setAttribute("aria-pressed", String(A.trailOn)); pushState(); };
  $("#gv-fit").onclick = () => graph.fit();
  $("#gv-zoom-in").onclick = () => graph.zoomAt(graph.W / 2, graph.H / 2, 1.6);
  $("#gv-zoom-out").onclick = () => graph.zoomAt(graph.W / 2, graph.H / 2, 1 / 1.6);
  $("#gv-search").onclick = openPalette;
  $("#gv-close").onclick = () => selectUnit(null);
  $("#an-follow").onclick = () => setFollow(!A.follow);
  $("#an-play").onclick = togglePlay;
  $("#an-prev").onclick = () => step(-1);
  $("#an-next").onclick = () => step(1);
  $("#gv-q").addEventListener("input", searchPalette);
  $("#gv-q").addEventListener("keydown", (ev) => {
    if (ev.key === "ArrowDown") { P.active = Math.min(P.items.length - 1, P.active + 1); renderPalette(); ev.preventDefault(); }
    if (ev.key === "ArrowUp") { P.active = Math.max(0, P.active - 1); renderPalette(); ev.preventDefault(); }
    if (ev.key === "Enter") { choose(P.active); ev.preventDefault(); }
    if (ev.key === "Escape") closePalette();
  });
  $("#gv-palette").addEventListener("pointerdown", (ev) => { if (ev.target.id === "gv-palette") closePalette(); });
  document.addEventListener("keydown", (ev) => {
    if (!A.visible) return;
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || "");
    if (typing) return;
    if (ev.key === "/" && !ev.shiftKey) { ev.preventDefault(); openPalette(); }  // shift-/ is ? (the guide)
    else if (ev.key === "?" || ev.key === "/") { /* the guide handles it */ }
    else if (ev.key === "Escape") {
      if (!$("#gv-palette").hidden) closePalette();
      else if (A.sel.size) clearSelection();
      else if (A.selected != null || A.detailMode) selectUnit(null);
      else if (A.editsOpen) setEditsPanel(false);
      else if (A.pathsOpen) closePaths();
    }
    else if (ev.key === "e" || ev.key === "E") setEditsPanel(!A.editsOpen);
    else if (ev.key === "p" || ev.key === "P") (A.pathsOpen ? closePaths() : openPaths());
    else if (ev.key === "t" || ev.key === "T") setConvo(!CV.open);
    else if (ev.key === "w" || ev.key === "W") { if (A.mode !== "attr") setMode("attr"); openTokenPanel(); }
    else if (ev.key === "a" || ev.key === "A") setMode(A.mode === "attr" ? "rel" : "attr");
    else if (ev.key === "f" || ev.key === "F") graph.fit();
    else if (ev.key === "+" || ev.key === "=") graph.zoomAt(graph.W / 2, graph.H / 2, 1.6);
    else if (ev.key === "-") graph.zoomAt(graph.W / 2, graph.H / 2, 1 / 1.6);
    else if (ev.key === " ") { ev.preventDefault(); togglePlay(); }
    else if (ev.key === "ArrowLeft") step(-1);
    else if (ev.key === "ArrowRight") step(1);
  });
  new ResizeObserver(() => { schedule(); if (H.tile && $("#an-heat")) drawHeat(); }).observe($("#view-weights"));
  setInterval(() => { if (A.visible) { $("#an-follow").classList.toggle("streaming", isLive()); graph.setState({ live: isLive() }); } }, 1000);
}

/** Another model was loaded: everything here describes the old one's units, so start over. */
function resetModel() {
  if (!A.atlas) return;  // nothing loaded yet; activate() fetches the new map
  stopPlay();
  if (A.pathsOpen) closePaths();
  A.sel.clear(); renderSelChip();
  selectUnit(null);
  resetThumbs();
  A.atlas = null; A.atlasError = null; A.edges = null; A.params = null; A.runId = null; A.edited = new Set();
  A.attr = null; A.attrN = null; A.attrInfo = null; A.attrTarget = null; A.attrOrig = null; A.attrDiff = null;
  A.attrSeq++; clearTimeout(A.attrTimer);
  A.expanded.clear();
  graph.setEdges(null);
  schedule();
  if (A.visible) activate();
}

async function activate() {
  readColors();
  graph.setState({ visible: true });
  if (!A.atlas) {
    schedule();
    const ok = await loadAtlas();
    if (!ok) { schedule(); if (A.atlasError?.status === 503) setTimeout(() => { if (A.visible && !A.atlas) activate(); }, 2500); return; }
  }
  const id = $("#run-select").value || null;
  if (id !== A.runId) await openRun(id);
  else {
    if (A.pendingCursor && A.follow && !A.playing) setCursor(lastRecorded(), "jump");  // catch up after being hidden
    renderReadout(); pushState(); schedule();
    scheduleTokenWork();  // weights may have been edited in Tune meanwhile: re-trace (cached if not)
  }
  A.pendingCursor = false;
  graph.invalidate();
}

function boot() {
  readColors();
  graph = new WeightGraph($("#gv-canvas"), $("#gv-minimap"), {
    onSelect: (g, how) => (how?.shift ? (g != null && toggleSelect(g)) : selectUnit(g)),
    onBoxSelect: (units, remove) => boxSelect(units, remove),
    onHover: (hit, mx, my) => graphTip(hit, mx, my),
    onEnd: (which) => openEnd(which),
    needThumb,
    onFrame: updateZoomLabel,
  });
  graph.setColors(C);
  setupInput();
  setupBar();
  stopPlay();
  document.addEventListener("brainscan:view", (ev) => {
    A.visible = ev.detail === "weights";
    graph.setState({ visible: A.visible, live: A.visible && isLive() });  // hidden → loop stops
    if (A.visible) activate(); else stopPlay();
  });
  // A hidden view catches up in activate(), so switching runs in the Scanner costs nothing here.
  document.addEventListener("brainscan:run", (ev) => { if (A.atlas && A.visible && ev.detail !== A.runId) openRun(ev.detail); });
  document.addEventListener("brainscan:focus-layer", (ev) => { if (A.atlas) graph.flyToLayer(Number(ev.detail)); });
  document.addEventListener("brainscan:model", resetModel);
  // A message sent from a composer: follow its tokens live (ui/chat.js).
  document.addEventListener("brainscan:expect-run", () => { if (A.atlas) setFollow(true); });
  const convoPref = (() => { try { return localStorage.getItem("brainscan-convo"); } catch { return null; } })();
  setConvo(convoPref ? convoPref === "1" : innerWidth >= 1000);
  renderConvo(true);
  document.addEventListener("brainscan:trace", (ev) => {  // from the guide
    const d = ev.detail || {};
    if (d.cmp) A.cmp = true;  // e.g. the first word an edit changed: compare with the original weights
    if (d.word != null) {  // "why this token" for a token picked in another view
      if (A.atlas && A.runId && A.runId === $("#run-select").value && A.tok[d.word]) traceWord(d.word);
      else A.pendingWord = d.word;  // traced once the run is open here
    }
    if (d.mode) setMode(d.mode);
    if (d.paths) openPaths(d.paths === true ? undefined : d.paths);
    if (d.palette) openPalette();
    if (d.unit != null && A.atlas) selectUnit(Number(d.unit));  // a unit named elsewhere (a neuron's cluster)
  });
  const recolor = () => { readColors(); renderLegend(); schedule(); resetThumbs(); if (H.tile && $("#an-heat")) { drawHeat(); renderHeatInfo(); } };
  $("#theme-toggle").addEventListener("click", recolor);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", recolor);
  connect();
  if (!$("#view-weights").hidden) { A.visible = true; activate(); }
}
boot();
