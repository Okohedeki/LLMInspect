// Compare checkpoints (Tune). Load a candidate checkpoint of the loaded model's architecture (a fine-tune,
// a quantized copy, a downloaded build) and see where its weights differ from the base: a change map by
// weight type and layer, the neurons, heads and token rows that moved. The candidate also sits in the edit
// stack as one "checkpoint" edit; applying it puts its weights in place, so Compare, Checks and Trace
// "vs original" then compare the base with the candidate (llminspect/recorder/checkpoints.py).
//
// API, under /llminspect/workbench/checkpoint: POST {spec} starts a job; GET returns
//   { status: none|building|ready|error, step, progress, error, report, edit }; GET .../reports lists saved ones.
// The report, as normalize() reads it: { id, created, base: { name }, candidate: { spec, name, files: [{ file, sha256 }] },
//   tensors: [{ name, layer, module, rel, max_abs, changed_frac, lowrank }], counts: { changed, identical, not_compared },
//   not_compared: [names], neurons: { top: [{ neuron, layer, index, rel }], layer_max: [{ layer, index, rel }] },
//   heads: { top: [{ head, layer, index, rel }] }, tokens: { embed: [{ token_id, text, rel }], lm_head: [...] }, notes,
//   summary: { total_rel, pattern: identical|concentrated|spread|mixed, basis: matrix|1d, matrices, changed_matrices,
//   tensors, top_tensors, top_share, max_share, layers: [{ layer, mean_rel }] } }.
// The stack comes from the status poll (workbench: { edits, applied, checkpoint: { id, name } }) and /llminspect/workbench.
import { extendBar, refreshBar } from "/ui/actionbar.js";

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const vis = (s) => (s ?? "").replace(/\n/g, "⏎").replace(/\t/g, "⇥").replace(/^ /, "·") || "∅";
async function api(path, opts) {
  const res = await fetch(path, opts);
  const text = await res.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch { /* not JSON */ }
  if (!res.ok) { const err = new Error(body?.detail || text || `${res.status}`); err.status = res.status; throw err; }
  return body;
}
const send = (method, path, body) => api(path, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}) });
const JOB = "/llminspect/workbench/checkpoint";
const KEEP = "llminspect-checkpoint-spec";

const MODULES = ["q", "k", "v", "o", "gate", "up", "down", "attn_norm", "mlp_norm"];
const LABEL = { attn_norm: "attn norm", mlp_norm: "mlp norm", neuron: "top neuron" };
const STRIP = [["embed", "embed"], ["lm_head", "lm_head"], ["final_norm", "final_norm"]];

const C = { supported: true, job: null, stack: null, models: [], modelsAt: 0, history: [], busy: null, timer: 0, fail: 0, pending: false, was: null,
  said: null, view: null, shown: null, d: null, more: false, nLayers: 24, base: "", acts: "" };
const M = { canvas: null, ctx: null, box: null, ro: null, d: null, g: null, hover: null, cur: null, last: "", progress: 0, target: 0, t0: 0, done: false, raf: 0 };
let P = null;  // theme colors, read again on a theme change

// ---------------------------------------------------------------- numbers and colors
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const num = (x) => (x == null || x === "" || !Number.isFinite(Number(x)) ? null : Number(x));
const pct = (x) => { const p = x * 100; return `${p === 0 ? 0 : p >= 10 ? Math.round(p) : p >= 1 ? +p.toFixed(1) : +p.toFixed(2)}%`; };
const rel = (x) => (x == null ? "–" : x === 0 ? "0" : x >= 0.001 ? pct(x) : x.toExponential(1));
const arr = (x, ...keys) => (Array.isArray(x) ? x : keys.map((k) => x?.[k]).find(Array.isArray) || []);
const leaf = (p) => String(p || "").replace(/[\\/]+$/, "").split(/[\\/]/).pop();
const hexToRgb = (h) => { h = h.replace("#", ""); return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)); };
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function colors() {
  // The graph's ramp, from its second stop: the first is the "no data" grey.
  const stops = cssVar("--graph-ramp").split(",").map((s) => hexToRgb(s.trim())).slice(1);
  return { ink: cssVar("--ink"), ink2: cssVar("--ink-2"), muted: cssVar("--muted"), idle: cssVar("--grid"), accent: hexToRgb(cssVar("--accent")),
    stops, mono: cssVar("--mono"), reduced: matchMedia("(prefers-reduced-motion: reduce)").matches };
}
function ramp(t) {
  const s = P.stops, x = clamp(t) * (s.length - 1), k = Math.min(Math.floor(x), s.length - 2), f = x - k;
  return s[k].map((a, j) => Math.round(a + (s[k + 1][j] - a) * f));
}
const rgb = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;
const luma = (c) => (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255;

// ---------------------------------------------------------------- the report
/** The one place that reads the report: tensors by (weight type, layer), counts, lists, the color domain. */
function normalize(r, nModel) {
  const s = r.summary || {}, cnt = r.counts || {}, cand = r.candidate || {};
  const tensors = (r.tensors || []).map((t) => ({ name: String(t.name ?? ""), layer: num(t.layer), module: t.module || "other",
    rel: num(t.rel) ?? 0, max_abs: num(t.max_abs), frac: num(t.changed_frac), low: num(t.lowrank) }));
  const top = (list) => list.map((x) => ({ ...x, layer: num(x.layer), index: num(x.index), rel: num(x.rel) ?? 0 }));
  const tok = r.tokens || {};
  const d = { id: r.id, spec: cand.spec || r.spec || "", tensors, summary: s, base: r.base?.name || "", notes: r.notes || "",
    name: cand.name || r.name || leaf(cand.spec || r.spec || cand.folder || r.folder) || "candidate",
    neurons: top(arr(r.neurons, "top")), layerMax: top(arr(r.neurons, "layer_max", "per_layer")), heads: top(arr(r.heads, "top")),
    embed: top(arr(tok, "embed", "embedding")), head: top(arr(tok, "lm_head", "output")) };
  d.changed = num(cnt.changed) ?? tensors.filter((t) => t.rel > 0).length;
  d.identical = num(cnt.identical) ?? tensors.filter((t) => t.rel === 0).length;
  const nc = r.not_compared ?? cnt.not_compared;  // the names may be cut short of the count
  d.skipped = Array.isArray(nc) ? nc : [];
  d.notCompared = num(cnt.not_compared) ?? (Array.isArray(nc) ? nc.length : num(nc) ?? 0);
  const fl = cand.files ?? r.files ?? {};
  d.files = (Array.isArray(fl) ? fl.map((f) => ({ name: f.name ?? f.file ?? f.path ?? "", sha: f.sha256 ?? f.sha ?? f.hash ?? "" }))
    : Object.entries(fl).map(([name, sha]) => ({ name, sha: String(sha) }))).filter((f) => f.name || f.sha);
  d.total = num(s.total_rel ?? s.total);
  d.pattern = s.pattern ?? r.pattern ?? null;
  d.hasHead = tensors.some((t) => t.module === "lm_head");
  // Grid cells: the largest tensor of each (type, layer); the strip: whole-model tensors.
  d.cell = new Map(); d.strip = {};
  let maxLayer = -1;
  const keep = (old, t) => (old && t.rel <= old.rel ? (old.count++, old) : { ...t, count: (old?.count || 0) + 1 });
  for (const t of tensors) {
    if (t.layer != null) maxLayer = Math.max(maxLayer, t.layer);
    if (t.layer != null && MODULES.includes(t.module)) d.cell.set(`${t.module}:${t.layer}`, keep(d.cell.get(`${t.module}:${t.layer}`), t));
    else if (t.layer == null && STRIP.some(([m]) => m === t.module)) d.strip[t.module] = keep(d.strip[t.module], t);
  }
  d.n = Math.max(nModel || 0, maxLayer + 1, ...d.layerMax.map((x) => (x.layer ?? -1) + 1), 1);
  d.peak = Array(d.n).fill(null);
  for (const x of d.layerMax) if (x.layer != null && x.layer >= 0 && x.layer < d.n) d.peak[x.layer] = x;
  // Layers by mean change: the report ranks them; worked out from the tensors when it doesn't.
  d.layers = arr(s.layers).map((x) => ({ layer: num(x.layer), rel: num(x.mean_rel ?? x.rel) ?? 0 })).filter((x) => x.layer != null && x.rel > 0);
  if (!d.layers.length) {
    const per = new Map();
    for (const t of tensors) if (t.layer != null) { const p = per.get(t.layer) || [0, 0]; p[0] += t.rel; p[1]++; per.set(t.layer, p); }
    d.layers = [...per].map(([layer, [sum, k]]) => ({ layer, rel: sum / k })).filter((x) => x.rel > 0).sort((a, b) => b.rel - a.rel);
  }
  // One color domain (log10 of the relative change) for the map, the legend and the lists.
  const logs = [...tensors.map((t) => t.rel), ...d.layerMax.map((x) => x.rel), ...[d.neurons, d.heads, d.embed, d.head].flat().map((x) => x.rel)]
    .filter((v) => v > 0).map(Math.log10);
  d.hi = logs.length ? Math.max(...logs) : -3;
  d.lo = Math.max(logs.length ? Math.min(...logs) : -4, d.hi - 5);
  if (d.hi - d.lo < 1) d.lo = d.hi - 1;
  return d;
}
const pos = (d, r) => (r > 0 ? clamp((Math.log10(r) - d.lo) / (d.hi - d.lo)) : 0);

/** One plain sentence with the numbers, then what it is typical of. Never a verdict. */
function headline(d) {
  const s = d.summary, b = (x) => `<b class="num">${x}</b>`;
  const share = num(s.top_share), most = num(s.max_share);
  const frac = num(s.top_frac) ?? (num(s.top_tensors) && num(s.tensors) ? s.top_tensors / s.tensors : 0.05);  // the tensors that carry `share`
  // The pattern is read from the matrices; say so, and how many of them moved, next to what it is typical of.
  const mats = num(s.matrices), moved = num(s.changed_matrices) ?? mats;
  const where = s.basis === "1d" ? "1-D weights" : "matrices";
  const of = mats && s.basis !== "1d" ? ` ${b(moved)} of ${b(mats)} ${where} moved` : "";
  if (!d.changed) return ["Identical: every compared tensor matches the base exactly.", "Applying it would leave the weights as they are."];
  if (d.pattern === "concentrated") {
    const total = num(s.tensors);
    if (total && d.changed <= num(s.top_tensors)) {  // every changed tensor is among the top: name the count itself
      return [`Concentrated: all of the change is in ${b(d.changed)} of ${b(total)} tensors.`, "Typical of a targeted edit."];
    }
    return [share != null ? `Concentrated: ${b(pct(share))} of the change is in ${b(pct(frac))} of the tensors${of}.` : `Concentrated: a few tensors carry most of the change${of}.`,
      "Typical of a targeted edit."];
  }
  if (d.pattern === "spread") {
    return [`Spread: the change is spread over the ${where}${of}${most != null ? `, and no one carries more than ${b(pct(most))}` : ""}.`,
      "Typical of quantization or a full fine-tune."];
  }
  if (d.pattern === "mixed") {
    return [`Mixed: some ${where} changed more than others${of}${share != null ? `, with ${b(pct(frac))} of the tensors carrying ${b(pct(share))}` : ""}.`,
      "It is neither confined to a few tensors nor uniform."];
  }
  return [`${b(d.changed)} of ${b(d.changed + d.identical)} tensors differ from the base.`, ""];
}

// ---------------------------------------------------------------- the stack
const stackEdits = () => C.stack?.edits || [];
const editName = (e) => e.checkpoint?.name || /^checkpoint:\s*([^,]+)/.exec(e.summary || "")?.[1]?.trim() || "candidate";
/** The checkpoint edit in the stack, or null. */
function cpEdit() {
  const e = stackEdits().find((x) => x.kind === "checkpoint");
  if (e) return { id: e.id, name: editName(e), enabled: e.enabled !== false, applied: !!e.applied || (C.stack.applied || []).some((a) => a.id === e.id) };
  const c = C.stack?.checkpoint;  // { id, name, report, applied }
  return c ? { id: c.id, name: c.name || "candidate", enabled: true, applied: !!c.applied } : null;
}

async function fetchStack() {
  try { C.stack = await api("/llminspect/workbench"); } catch { /* the recorder is loading: the next poll tells */ }
  renderActions();
}
function say(text, tone = "") {
  const el = $("#cp-msg");
  el.className = `wb-msg ${tone}`;
  el.textContent = text || "";
}

/** Apply or restore, with the buttons locked and the edit panels told. */
async function act(label, fn) {
  C.busy = label; renderActions(); renderForm();
  try { await fn(); } catch (err) { say(err.message, "error"); }
  C.busy = null;
  await fetchStack(); renderForm();
  document.dispatchEvent(new CustomEvent("llminspect:workbench"));  // the edit stack and Tune's bar refresh
}
const applyCandidate = () => act("Applying", async () => {
  const e = cpEdit();
  if (!e) throw new Error("The candidate isn't in the edit stack: load it again.");
  if (!e.enabled) await send("PATCH", `/llminspect/workbench/edits/${e.id}`, { enabled: true });
  const res = await send("POST", "/llminspect/workbench/apply");
  say(`Applied in ${res?.seconds ?? "?"} s.${res?.snapshot_bytes != null ? ` The base values of the touched weights are kept in memory (${(res.snapshot_bytes / 1e9).toFixed(2)} GB) for an exact restore.` : ""}`, "ok");
});
const restoreBase = () => act("Restoring", async () => { await send("POST", "/llminspect/workbench/revert"); say("Base weights restored.", "ok"); });

function renderActions() {
  const el = $("#cp-actions"), e = cpEdit();
  let html = "";
  if (e) {
    const applied = (C.stack.applied || []).length, others = stackEdits().filter((x) => x.enabled && x.id !== e.id).length, off = !!C.busy;
    const tail = e.applied
      ? `<button type="button" class="link" data-pick="compare">Compare</button>, <button type="button" class="link" data-pick="check">Checks</button> and Trace › vs original now compare the base with the candidate.`
      : others ? `Applies the whole stack: the candidate's weights first, then ${others} other edit${others === 1 ? "" : "s"} on top. Restore base undoes it exactly.`
        : "Puts the candidate's weights in place. Restore base undoes it exactly.";
    html = `<div class="cp-edit"><span class="cp-edit-name">checkpoint: ${esc(e.name)}</span>
        ${e.applied ? `<span class="pill ok">applied</span>` : `<span class="muted small">in the stack, not applied</span>`}</div>
      <div class="cp-btns">
        <button type="button" class="primary" data-apply ${off || e.applied ? "disabled" : ""}>${C.busy === "Applying" ? "Applying…" : "Apply candidate"}</button>
        <button type="button" data-revert ${off || !applied ? "disabled" : ""}>${C.busy === "Restoring" ? "Restoring…" : "Restore base"}</button></div>
      <p class="hint cp-tail">${tail}</p>`;
  } else if (C.stack && C.view === "report" && C.d?.spec) {  // the report is still here, its edit was removed from the stack
    html = `<div class="cp-edit"><span class="muted small">This candidate isn't in the edit stack any more, so there is nothing to apply.</span></div>
      <div class="cp-btns"><button type="button" data-again ${C.busy ? "disabled" : ""}>Load it again</button></div>`;
  }
  el.hidden = !html;
  if (html !== C.acts) { C.acts = html; el.innerHTML = html; }  // the status poll calls this: leave a pressed button alone
}

// ---------------------------------------------------------------- the job
/** Read the job again in `ms`; one timer at a time, whichever read asked last. */
function poll(ms) { clearTimeout(C.timer); C.timer = setTimeout(fetchJob, ms); }
async function fetchJob() {
  clearTimeout(C.timer);
  let ok = true;
  try { C.job = await api(JOB); C.supported = true; C.fail = 0; }
  catch (err) {
    ok = false;
    if (err.status === 404 || err.status === 405) C.supported = false;
    else if (err.status !== 503 && ++C.fail > 4) { say(`Lost contact with the recorder: ${err.message}`, "error"); C.was = null; }
  }
  const now = C.job?.status;
  if (now === "building" || (!ok && C.was === "building" && C.fail <= 4)) poll(now === "building" ? 1500 : 3000);
  else if (C.was === "building" && ok) {  // the job finished: its edit is in the stack now
    await fetchStack();
    document.dispatchEvent(new CustomEvent("llminspect:workbench"));
    loadHistory();
  }
  if (ok) {
    if (now === "error" && C.said !== C.job.error) { say(C.job.error || "The comparison failed.", "error"); C.said = C.job.error; }
    else if (now === "ready" && C.was === "building") say("");
    C.was = now;
  }
  renderAll();
}

const when = (iso) => { const t = new Date(iso); return Number.isNaN(+t) ? "" : t.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }); };
/** The saved reports for this model: a candidate worth comparing again goes back in the box. */
async function loadHistory() {
  try { C.history = await api(`${JOB}/reports`); } catch { C.history = []; }
  const el = $("#cp-hist"), rows = (C.history || []).slice(0, 12);
  el.hidden = !rows.length;
  el.innerHTML = rows.length ? `<details class="cp-det"><summary>Earlier comparisons <span class="dim">${C.history.length}</span></summary>
    <ul class="cp-hashes">${rows.map((r, i) => `<li><button type="button" class="link mono" data-history="${i}" title="Put ${esc(r.spec)} in the box">${esc(r.candidate)}</button>
      <span class="muted small">${esc(r.pattern)}${num(r.total_rel) != null ? ` · ${rel(num(r.total_rel))}` : ""} · ${esc(when(r.created))}</span></li>`).join("")}</ul></details>` : "";
}

async function loadModels() {
  if (Date.now() - C.modelsAt < 60000) return;
  C.modelsAt = Date.now();
  try {
    const d = await api("/llminspect/models");
    C.models = (d.models || []).filter((m) => !m.current);
    $("#cp-models").innerHTML = C.models.map((m) => `<option value="${esc(m.spec)}" label="${esc(m.name)}"></option>`).join("");
  } catch { C.modelsAt = 0; }
}

/** Start a comparison. The recorder refuses a bad candidate here (pickle, a different shape, not cached, a
 * checkpoint already applied) and says why; a candidate loaded before and not applied is replaced by this one. */
async function load(ev) {
  ev?.preventDefault();
  const spec = $("#cp-spec").value.trim();
  if (!spec) { say("Name a candidate: a folder, a .gguf file, ollama:tag or a Hugging Face repo id.", "error"); return; }
  try { localStorage.setItem(KEEP, spec); } catch { /* storage unavailable */ }
  say(""); C.said = null; C.busy = "Starting"; renderForm();
  try {
    C.job = await send("POST", JOB, { spec });
    C.was = "building"; C.fail = 0;
  } catch (err) {
    say(err.message, "error");
    if (err.status === 409) { C.busy = null; await fetchJob(); return; }  // one is running already: follow it
  }
  C.busy = null;
  if (C.job?.status === "building") poll(1500);
  renderAll();
}

// ---------------------------------------------------------------- drawing the map
const GUT = 76, PAD = 2, RH = 22, RG = 3;
function layout(w, n) {
  const cw = (w - GUT - PAD) / n, gap = cw > 10 ? 2 : cw > 5 ? 1 : 0, rows = [];
  let y = 0;
  for (const m of MODULES) { rows.push({ key: m, y, h: RH }); y += RH + RG; }
  y += 6; rows.push({ key: "neuron", y, h: RH }); y += RH + 16;
  const strip = { y, h: RH }; y += RH + 10;
  const axis = { y: y + 1 }; y += 16;
  return { w, n, cw, gap, rows, strip, axis, h: y, bw: (w - GUT - PAD - 12) / 3 };
}
/** The rectangle of a cell: { r, c } with r the row (rows.length is the strip) and c the column. */
function rectOf(g, h) {
  if (h.r === g.rows.length) return [GUT + h.c * (g.bw + 6), g.strip.y, g.bw, g.strip.h];
  return [GUT + h.c * g.cw + g.gap / 2, g.rows[h.r].y, Math.max(1, g.cw - g.gap), g.rows[h.r].h];
}
function hit(x, y) {
  const g = M.g;
  if (!g || !M.d || x < GUT) return null;
  const c = Math.floor((x - GUT) / g.cw);
  const r = g.rows.findIndex((row) => y >= row.y && y <= row.y + row.h);
  if (r >= 0 && c >= 0 && c < g.n) return { r, c };
  if (y >= g.strip.y && y <= g.strip.y + g.strip.h) { const k = Math.floor((x - GUT) / (g.bw + 6)); if (k >= 0 && k < 3) return { r: g.rows.length, c: k }; }
  return null;
}
/** The text, cut with an ellipsis to fit `room` pixels (nothing when even a few letters don't fit). */
function fit(ctx, text, room) {
  if (ctx.measureText(text).width <= room) return text;
  let k = text.length;
  while (k > 2 && ctx.measureText(`${text.slice(0, k)}…`).width > room) k--;
  return k > 2 ? `${text.slice(0, k)}…` : "";
}
function box(ctx, [x, y, w, h], fill, alpha = 1, stroke) {
  const r = Math.min(4, w / 3, h / 3);
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x, y, w, h, r); else ctx.rect(x, y, w, h);
  if (fill) { ctx.globalAlpha = alpha; ctx.fillStyle = fill; ctx.fill(); ctx.globalAlpha = 1; }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.stroke(); }
}

function drawMap() {
  cancelAnimationFrame(M.raf);
  if (!M.canvas?.isConnected) return;
  const w = Math.floor(M.box.clientWidth);
  if (w < 120) return;
  const d = M.d, n = d ? d.n : C.nLayers, g = (M.g = layout(w, n)), dpr = devicePixelRatio || 1, ctx = M.ctx;
  if (M.canvas.width !== Math.round(w * dpr) || M.canvas.height !== Math.round(g.h * dpr)) {
    M.canvas.width = Math.round(w * dpr); M.canvas.height = Math.round(g.h * dpr);
    M.canvas.style.width = `${w}px`; M.canvas.style.height = `${g.h}px`;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, g.h);
  let moving = false;
  if (!d) {  // the quiet grid, with the build's progress sweeping across it
    const d2 = M.target - M.progress;
    if (P.reduced || Math.abs(d2) < 0.002) M.progress = M.target; else { M.progress += d2 * 0.1; moving = true; }
  }
  let reveal = 1;
  if (d && !M.done) { reveal = P.reduced ? 1 : clamp((performance.now() - M.t0) / 750); if (reveal < 1) moving = true; else M.done = true; }
  const ease = 1 - (1 - reveal) ** 3, fade = (c) => (reveal >= 1 ? 1 : clamp((ease * (n + 10) - c) / 10));
  const hov = M.hover;
  ctx.font = `10.5px ${P.mono}`; ctx.textBaseline = "middle";
  const wipe = (c) => clamp(M.progress * n - c);
  const paint = (rec, rect, c) => {
    if (!d) { box(ctx, rect, P.idle, 0.55); const k = wipe(c); if (k > 0) box(ctx, rect, `rgb(${P.accent})`, 0.34 * k); return; }
    if (!rec) return;
    box(ctx, rect, rec.rel > 0 ? rgb(ramp(pos(d, rec.rel))) : P.idle, fade(c));
  };
  g.rows.forEach((row, r) => {
    ctx.fillStyle = hov?.r === r ? P.ink : P.muted; ctx.textAlign = "left";
    ctx.fillText(LABEL[row.key] || row.key, 0, row.y + row.h / 2);
    for (let c = 0; c < n; c++) paint(d ? (row.key === "neuron" ? d.peak[c] : d.cell.get(`${row.key}:${c}`)) : null, rectOf(g, { r, c }), c);
  });
  STRIP.forEach(([key, label], k) => {
    const rect = rectOf(g, { r: g.rows.length, c: k }), rec = d?.strip[key], col = rec && rec.rel > 0 ? ramp(pos(d, rec.rel)) : null;
    if (!d) box(ctx, rect, P.idle, 0.55);
    else if (rec) box(ctx, rect, col ? rgb(col) : P.idle, fade(0));
    else box(ctx, rect, null, 1, P.idle);  // not in this model (tied): an outline
    const room = rect[2] - 16, text = d && !rec && key === "lm_head" ? "lm_head · tied" : label, val = rec ? rel(rec.rel) : "";
    const both = val && ctx.measureText(`${text}   ${val}`).width < room;  // on a narrow screen the value waits for the tooltip
    ctx.textAlign = "left"; ctx.fillStyle = col ? (luma(col) < 0.55 ? "#fff" : "#0b0d12") : P.muted;
    ctx.fillText(fit(ctx, text, room), rect[0] + 8, rect[1] + rect[3] / 2);
    if (both) { ctx.textAlign = "right"; ctx.fillText(val, rect[0] + rect[2] - 8, rect[1] + rect[3] / 2); }
  });
  ctx.textAlign = "center";
  const want = 28 / g.cw, step = [1, 2, 5, 10, 20, 25, 50, 100].find((s) => s >= want) || 100;  // labels at least ~28px apart
  for (let c = 0; c < n; c++) {
    const on = hov && hov.r < g.rows.length && hov.c === c;
    if (c % step && !on) continue;
    ctx.fillStyle = on ? P.ink : P.muted;
    ctx.fillText(String(c), GUT + c * g.cw + g.cw / 2, g.axis.y + 6);
  }
  ctx.textAlign = "left"; ctx.fillStyle = P.muted; ctx.fillText("layer", 0, g.axis.y + 6);
  if (hov) box(ctx, rectOf(g, hov), null, 1, P.ink);
  if (moving) M.raf = requestAnimationFrame(drawMap);
}

/** The words for a cell, for the tooltip and for screen readers. */
function recordAt(h) {
  const d = M.d, g = M.g;
  if (!d || !h) return null;
  const info = (title, t) => (t.rel
    ? { title, name: t.name, lines: [["relative change", rel(t.rel)], ...(t.frac != null ? [["elements changed", pct(t.frac)]] : []),
      ...(t.low != null ? [["fits in 8 directions", pct(t.low)]] : []), ...(t.max_abs != null ? [["largest single change", t.max_abs.toExponential(1)]] : [])] }
    : { title, name: t.name, lines: [["", "identical to the base"]] });
  if (h.r === g.rows.length) {
    const [key] = STRIP[h.c], rec = d.strip[key];
    return rec ? info(key, rec) : { title: key, lines: [["", key === "lm_head" ? "tied to the embeddings" : "not in the report"]] };
  }
  const row = g.rows[h.r];
  if (row.key === "neuron") {
    const p = d.peak[h.c];
    return p ? { title: `MLP neuron peak · layer ${h.c}`, lines: [["strongest neuron", `N${p.index}`], ["relative change", rel(p.rel)]] } : null;
  }
  const rec = d.cell.get(`${row.key}:${h.c}`);
  return rec ? info(`${LABEL[row.key] || row.key} · layer ${h.c}${rec.count > 1 ? ` (largest of ${rec.count} tensors)` : ""}`, rec) : null;
}

function setHover(h, speak = false) {
  const tip = $("#cp-tip"), rec = recordAt(h);
  M.hover = rec ? h : null;
  drawMap();
  if (!rec) { tip.hidden = true; return; }
  tip.innerHTML = `<b>${esc(rec.title)}</b>${rec.lines.map(([k, v]) => (k ? `<div class="row"><span>${esc(k)}</span><span class="mono">${esc(v)}</span></div>` : `<div class="muted">${esc(v)}</div>`)).join("")}
    ${rec.name ? `<div class="cp-tipname mono">${esc(rec.name)}</div>` : ""}`;
  tip.hidden = false;
  const [x, y, w, hh] = rectOf(M.g, h), tw = tip.offsetWidth;
  tip.style.left = `${Math.max(0, Math.min(x + w / 2 - tw / 2, M.g.w - tw))}px`;
  tip.style.top = `${y + hh + 8 + tip.offsetHeight > M.g.h ? Math.max(0, y - tip.offsetHeight - 8) : y + hh + 8}px`;
  if (speak) $("#cp-sr").textContent = `${rec.title}. ${rec.lines.map(([k, v]) => (k ? `${k} ${v}` : v)).join(", ")}`;
}

function mountMap(d) {
  M.box = $("#cp-map"); M.canvas = $("#cp-canvas"); M.ctx = M.canvas.getContext("2d");
  M.d = d; M.hover = M.cur = M.g = null; M.t0 = performance.now(); M.done = false;
  M.ro?.disconnect();
  M.ro = new ResizeObserver(() => drawMap());
  M.ro.observe(M.box);
  M.last = "";
  const point = (ev) => {  // a tap on a phone is a pointerdown with no move
    const r = M.canvas.getBoundingClientRect(), h = hit(ev.clientX - r.left, ev.clientY - r.top), key = h ? `${h.r}:${h.c}` : "";
    if (key !== M.last) { M.last = key; setHover(h); }
  };
  M.canvas.addEventListener("pointermove", point);
  M.canvas.addEventListener("pointerdown", point);
  M.canvas.addEventListener("pointerleave", () => { M.last = ""; setHover(null); });
  M.canvas.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") { M.cur = null; setHover(null); return; }
    const dr = { ArrowUp: -1, ArrowDown: 1 }[ev.key] ?? 0, dc = { ArrowLeft: -1, ArrowRight: 1 }[ev.key] ?? 0;
    if (!M.d || (!dr && !dc)) return;
    ev.preventDefault();
    const g = M.g, r = clamp((M.cur?.r ?? 0) + dr, 0, g.rows.length);
    M.cur = M.cur ? { r, c: clamp(M.cur.c + dc, 0, (r === g.rows.length ? 3 : g.n) - 1) } : { r: 0, c: 0 };  // the first press lands on the first cell
    setHover(M.cur, true);
  });
  M.canvas.addEventListener("blur", () => { M.cur = null; setHover(null); });
  drawMap();
}

// ---------------------------------------------------------------- the panel
function paintLegend() {
  const el = $("#cp-ramp");
  if (el) el.style.background = `linear-gradient(90deg, ${P.stops.map(rgb).join(",")})`;
}
const CANVAS = `<canvas id="cp-canvas" tabindex="0" role="img" aria-label="Change map: a row for each kind of weight, a column for each layer; color is the relative change from the base. Arrow keys move between cells."></canvas>
  <div class="tooltip" id="cp-tip" hidden></div><div class="cp-sr" id="cp-sr" aria-live="polite"></div>`;

function ghost(title, sub, progress) {
  const body = $("#cp-body");
  if (C.view !== "ghost") {
    body.innerHTML = `<div class="cp-hero"><div class="cp-map" id="cp-map">${CANVAS}
      <div class="cp-empty"><div><b id="cp-g-title"></b><span id="cp-g-sub"></span><div class="cp-prog" id="cp-g-prog" hidden><i></i></div></div></div></div></div>`;
    C.view = "ghost"; C.shown = null;
    M.progress = 0;
    mountMap(null);
  }
  $("#cp-g-title").textContent = title;
  $("#cp-g-sub").textContent = sub || "";
  const bar = $("#cp-g-prog");
  bar.hidden = progress == null;
  if (progress != null) bar.firstElementChild.style.setProperty("--p", `${Math.round(clamp(progress) * 100)}%`);
  M.target = progress == null ? 0 : clamp(progress);
  drawMap();
}

function renderBody() {
  const j = C.job;
  if (!C.supported) return ghost("This recorder can't compare checkpoints yet.", "Restart it with the latest code, then reopen Tune.");
  if (j?.status === "building") {
    const p = num(j.progress), step = String(j.step || "working");
    return ghost(`${step.charAt(0).toUpperCase()}${step.slice(1)}${p != null ? ` · ${Math.round(clamp(p) * 100)}%` : ""}`, "The model is busy until this finishes.", p ?? 0);
  }
  if (j?.status === "ready" && j.report) return showReport(j.report);
  return ghost("Load a candidate to see where its weights differ from this model.", "A fine-tune, a quantized copy or a downloaded build of the same architecture.");
}

function showReport(report) {
  if (C.view === "report" && C.shown === report.id) return;
  const d = normalize(report, C.nLayers);
  C.view = "report"; C.shown = report.id; C.more = false;
  const [lead, typ] = headline(d);
  const top = d.layers.slice(0, 3).map((x) => x.layer);
  const [lo, hi] = [rel(10 ** d.lo), rel(10 ** d.hi)];
  const files = d.files.length ? `<details class="cp-det" ${d.files.length <= 4 ? "open" : ""}><summary>Candidate files <span class="dim">${d.files.length} · sha256</span></summary>
      <ul class="cp-hashes">${d.files.map((f) => `<li><span class="cp-fn mono" title="${esc(f.name)}">${esc(f.name)}</span>${f.sha
        ? `<button type="button" class="link mono" data-copy="${esc(f.sha)}" title="Copy the full sha256: ${esc(f.sha)}">${esc(f.sha.slice(0, 12))}…</button>` : ""}</li>`).join("")}</ul></details>` : "";
  const skipped = d.skipped.length ? `<details class="cp-det"><summary>Not compared <span class="dim">${d.notCompared} tensors with no match in this model</span></summary>
      <p class="cp-nc mono">${d.skipped.slice(0, 40).map(esc).join("<br>")}${d.skipped.length > 40 ? `<br>+${d.skipped.length - 40} more` : ""}</p></details>` : "";
  $("#cp-body").innerHTML = `
    <div class="cp-head">
      <div class="cp-who"><span class="eyebrow">Candidate</span><b class="mono">${esc(d.name)}</b>${d.base || C.base ? `<span>against ${esc(d.base || C.base)}</span>` : ""}</div>
      <h3 class="cp-headline">${lead}${typ ? `<span class="typ">${esc(typ)}</span>` : ""}</h3></div>
    <div class="cp-hero">
      <div class="cp-map" id="cp-map">${CANVAS}</div>
      ${d.changed ? `<div class="cp-legend"><span class="sw"></span><span>identical</span>
        <span class="mono">${lo}</span><span class="cp-ramp" id="cp-ramp"></span><span class="mono">${hi}</span>
        <span>relative change, log scale</span></div>` : ""}
      <details class="cp-how"><summary>How to read the map</summary>
        <p class="hint">Each column is a layer and each row a kind of weight: q, k, v and o are attention; gate, up and down are the MLP; the
        norms scale activations. Color is how large the change is next to the base weights. Gray cells are identical to the base. The
        <b>top neuron</b> row shows, for each layer, the one MLP neuron whose weights changed most. Hover a cell, or use the arrow keys.</p>
        ${d.notes ? `<p class="hint">${esc(d.notes)}</p>` : ""}</details>
    </div>
    <dl class="cp-stats">
      <div><dt>tensors changed</dt><dd>${d.changed}</dd></div><div><dt>identical</dt><dd class="dim">${d.identical}</dd></div>
      <div><dt>not compared</dt><dd class="dim">${d.notCompared}</dd></div>
      ${d.total != null ? `<div title="‖Δ‖ ÷ ‖W‖ over everything compared"><dt>total change</dt><dd>${rel(d.total)}</dd></div>` : ""}
      ${top.length ? `<div><dt>layers that changed most</dt><dd>${top.map((l) => `L${l}`).join(" · ")}</dd></div>` : ""}</dl>
    <div class="cp-lists" id="cp-lists"></div>
    <div class="cp-foot">${files}${skipped}</div>`;
  C.d = d;
  mountMap(d);
  paintLegend();
  renderLists();
}

function renderLists() {
  const d = C.d, el = $("#cp-lists");
  if (!d || !el) return;
  const cap = C.more ? 25 : 8;
  const bar = (r) => `<span class="cp-bar"><i style="--w:${pos(d, r).toFixed(3)};--c:${rgb(ramp(pos(d, r)))}"></i></span>`;
  const col = (title, sub, items, name, empty) => `<section class="cp-col"><h4>${title} <span class="dim">${sub}</span></h4>
    <ul class="cp-rank">${items.slice(0, cap).map((x) => `<li><span class="cp-l">${name(x)}</span>${bar(x.rel)}<span class="cp-val">${rel(x.rel)}</span></li>`).join("")
      || `<li class="muted small">${empty}</li>`}</ul></section>`;
  const neuron = (x) => x.neuron ?? `L${x.layer} N${x.index}`;
  const tokName = (x) => `<span class="cp-tok" title="token ${esc(x.token_id)}">${esc(vis(x.text))}</span>`;
  const more = Math.max(d.neurons.length, d.heads.length, d.embed.length, d.head.length) > 8;
  el.innerHTML = col("MLP neurons", "biggest relative change", d.neurons,
      (x) => `<button type="button" class="nv-link" data-open-neuron="${esc(neuron(x))}" title="Open ${esc(neuron(x))}">${esc(neuron(x))}</button>`, "none changed")
    + col("Attention heads", "biggest relative change", d.heads, (x) => `<span class="mono">${esc(x.head ?? `L${x.layer} h${x.index}`)}</span>`, "none changed")
    + col("Input embeddings", "token rows", d.embed, tokName, "none changed")
    + col("Output weights", "token rows", d.head, tokName, d.hasHead ? "none changed" : "tied to the embeddings")
    + (more ? `<button type="button" class="link cp-more" data-more>${C.more ? "Show fewer" : "Show 25 of each"}</button>` : "");
}

function renderForm() {
  const building = C.job?.status === "building", off = building || !!C.busy || !C.supported;
  $("#cp-load").disabled = off;
  $("#cp-load").textContent = building || C.busy === "Starting" ? "Reading…" : "Load candidate";
  $("#cp-spec").disabled = building;
  $("#cp-base").textContent = C.base && C.view !== "report" ? `against ${C.base}` : "";  // a report names its own base
}
function renderAll() { renderBody(); renderForm(); renderActions(); refreshBar("workbench"); }

// ---------------------------------------------------------------- boot
function boot() {
  if (!$("#cp-panel")) return;
  P = colors();
  const recolor = () => { P = colors(); paintLegend(); drawMap(); renderLists(); };
  new MutationObserver(recolor).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", recolor);
  extendBar("workbench", {
    groups: [],
    actions: [{ id: "checkpoint", group: "Change", label: "Compare a checkpoint",
      hint: "Load a fine-tune, a quantized copy or a downloaded build and see where its weights differ from this model",
      need: () => (!C.supported ? "a recorder with checkpoint comparison: restart it" : C.job?.status === "building" ? "the comparison running now to finish" : null),
      run: () => { $("#h-checkpoint")?.scrollIntoView({ behavior: "smooth", block: "start" }); setTimeout(() => $("#cp-spec")?.focus({ preventScroll: true }), 350); } }],
  });
  const panel = $("#cp-panel");
  panel.addEventListener("keydown", (ev) => ev.stopPropagation());  // typing a path never drives the graph
  $("#cp-form").addEventListener("submit", load);
  $("#cp-spec").addEventListener("focus", loadModels);
  panel.addEventListener("click", (ev) => {
    const t = ev.target;
    if (t.closest("[data-apply]")) applyCandidate();
    else if (t.closest("[data-revert]")) restoreBase();
    else if (t.closest("[data-again]")) { if (C.d?.spec) { $("#cp-spec").value = C.d.spec; load(); } }
    else if (t.closest("[data-history]")) { const r = C.history[Number(t.closest("[data-history]").dataset.history)]; if (r) { $("#cp-spec").value = r.spec; $("#cp-spec").focus(); } }
    else if (t.closest("[data-more]")) { C.more = !C.more; renderLists(); }
    else if (t.closest("[data-pick]")) document.dispatchEvent(new CustomEvent("llminspect:pick", { detail: t.closest("[data-pick]").dataset.pick }));
    else if (t.closest("[data-copy]")) {
      const b = t.closest("[data-copy]"), old = b.textContent;
      navigator.clipboard.writeText(b.dataset.copy).then(() => { b.textContent = "copied"; }, () => { b.textContent = "select and copy"; })
        .finally(() => setTimeout(() => { b.textContent = old; }, 1400));
    }
  });
  try { $("#cp-spec").value = localStorage.getItem(KEEP) || ""; } catch { /* storage unavailable */ }
  const enter = () => { fetchJob(); fetchStack(); loadModels(); loadHistory(); };
  document.addEventListener("llminspect:view", (ev) => { if (ev.detail === "workbench") enter(); });
  document.addEventListener("llminspect:workbench", fetchStack);  // the stack changed elsewhere
  document.addEventListener("llminspect:status", (ev) => {  // every poll (ui/app.js)
    const st = ev.detail || {};
    if (st.model) {
      const n = st.model.n_layers || C.nLayers;
      if (st.model.name !== C.base || n !== C.nLayers) { C.base = st.model.name || ""; C.nLayers = n; renderForm(); if (C.view === "ghost") drawMap(); }
    }
    if (st.workbench) { C.stack = st.workbench; renderActions(); }
    // The page opened while the recorder was loading: the first read of the job failed, so read it now.
    if (st.status === "ready" && !C.job && C.supported && !C.pending && !$("#view-workbench").hidden) {
      C.pending = true;
      fetchJob().finally(() => { C.pending = false; });
    }
  });
  document.addEventListener("llminspect:model", () => {  // a report and a candidate belong to one base model
    C.job = null; C.view = null; C.shown = null; C.was = null; C.said = null; C.d = null;
    say(""); enter(); renderAll();
  });
  renderAll();
  if (!$("#view-workbench").hidden) enter();
}
boot();
