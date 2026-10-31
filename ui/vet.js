// Vet: check a model you didn't train before running it (llminspect/check.py), as four actions, the steps of
// the Vet action in the top bar: open it safely (what loading it runs), its origin (which model its weights come
// from), its edits (what moved, and where) and its behavior (what the edit does, against the loaded base).
//
// The stage is Trace's graph (ui/graph.js) drawn for the checked model: layers left to right along the
// residual stream, the attention tensor above it and the MLP tensor below, each lit by how far it moved from
// the base (1 − the fingerprint's correlation). The drawer holds the four answers and the findings of the
// action on now. Nothing is loaded to draw it: the values come from the fingerprint.
//
// API: GET /llminspect/check/models; POST /llminspect/check {candidate, base, deep} and /check/all start a job;
//   GET /llminspect/check/jobs/{job}; GET /llminspect/check/reports/{id} and /{id}/map; /{id}/html is the page.
import { defineBar, noteBar, refreshBar, openStep, currentStep } from "/ui/actionbar.js";
import { WeightGraph } from "/ui/graph.js";
import { graphColors } from "/ui/weights.js";

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
async function api(path, opts) {
  const res = await fetch(path, opts);
  const text = await res.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch { /* not JSON */ }
  if (!res.ok) throw new Error(body?.detail || text || `${res.status}`);
  return body;
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const KEEP = "llminspect-vet", KEEP_JOB = "llminspect-vet-job";
const remember = (key, value) => { try { if (value) localStorage.setItem(key, value); else localStorage.removeItem(key); } catch { /* storage unavailable */ } };

const STEPS = [["open", "Open safely"], ["origin", "Origin"], ["edits", "Edits"], ["behavior", "Behavior"]];
const AREAS = { open: ["files"], origin: ["origin"], edits: ["outside_weights"], behavior: ["inside_weights"] };
const V = { models: [], base: null, sel: null, job: null, report: null, map: null, act: "open", timer: 0, ticks: 0, loaded: false, visible: false };
let graph = null, units = [];

/** A model's state, from its latest check. */
function tone(m) {
  const c = m.last?.counts;
  if (!c) return "none";
  return c.high ? "high" : c.medium ? "medium" : "clear";
}
const GROUP = { high: "Look first", medium: "Worth a look", clear: "Checked, nothing found", none: "Not checked yet" };

// ---------------------------------------------------------------- the map

/** The graph's atlas for a vetted model: one attention and one MLP node per layer. */
function atlasOf(map) {
  units = [];
  const byLayer = new Map(map.layers.map((l) => [l.layer, l]));
  const n = map.layers.length ? Math.max(...map.layers.map((l) => l.layer)) + 1 : 0;
  const layers = [];
  for (let i = 0; i < n; i++) {
    const l = byLayer.get(i) || { layer: i }, offset = units.length;
    if (l.attn) units.push({ kind: "head", i: 0, layer: i, size: 1, label: "attn", title: "Attention", t: l.attn });
    if (l.mlp) units.push({ kind: "cluster", i: 0, layer: i, size: 1, label: "mlp", title: "MLP", t: l.mlp });
    layers.push({ layer: i, heads: l.attn ? 1 : 0, clusters: l.mlp ? 1 : 0, offset });
  }
  for (const u of units) u.sub = `L${u.layer} · ${movedText(u.t.corr)}`;
  return { n_units: units.length, units, layers, podCols: 1, captions: false, nodeScale: 1.7,
    embed: { label: "Embedding", shape: map.embed?.shape }, unembed: { label: "Output" } };
}

const moved = (corr) => (corr == null ? null : Math.max(0, 1 - corr));
function movedText(corr) {
  if (corr == null) return "no base to compare";
  return corr >= 0.99999 ? "same as the base" : `moved · r ${corr.toFixed(4)}`;
}

/** Heat per unit: how far it moved from the base, against the most any tensor moved (at least 2%). */
function paint() {
  if (!graph || !V.map) return;
  const ms = units.map((u) => moved(u.t.corr));
  const top = Math.max(0.02, ...ms.filter((m) => m != null));
  const heat = new Float32Array(units.length), edited = new Set();
  ms.forEach((m, g) => {
    if (m == null) heat[g] = NaN;
    else if (m < 1e-5) heat[g] = 0.06;
    else { heat[g] = 0.3 + 0.7 * Math.min(1, m / top); edited.add(g); }
  });
  graph.setState({ heat, edited, visible: V.visible });
}

function drawMap() {
  const map = V.map, empty = $("#vt-empty");
  if (!graph) return;
  if (!map || !map.layers?.length) {
    graph.setState({ visible: false });
    $("#vt-canvas").style.visibility = "hidden";
    empty.hidden = false;
    empty.innerHTML = emptyHtml();
    renderLegend();
    return;
  }
  empty.hidden = true;
  $("#vt-canvas").style.visibility = "";
  graph.setAtlas(atlasOf(map));
  paint();
  frame(true);
  renderLegend();
}

/** Fit the whole model into the part of the stage the drawer doesn't cover. */
function frame(now = false) {
  if (!graph?.bounds) return;
  graph.insets = { l: 0, r: innerWidth > 900 ? $("#vt-drawer").offsetWidth + 12 : 0 };
  graph.fitTrace([]);
  if (now) graph.cam = { ...graph.target };
}

function emptyHtml() {
  if (!V.loaded) return '<div class="spinner" aria-hidden="true"></div><span>Reading the model folders…</span>';
  if (!V.models.length) return "<b>No models found</b><span>Add the folders your models are in with LLMINSPECT_MODEL_DIRS.</span>";
  const m = picked();
  if (!m) return "<b>Pick a model to vet</b><span>Every model on this machine is in the list at the top left, the ones to look at first on top.</span>";
  if (V.job) return '<div class="spinner" aria-hidden="true"></div><span>Checking…</span>';
  if (!m.last) return `<b>${esc(m.name)} isn't checked yet</b><span>Check reads every file and fingerprints the weights. Nothing is loaded.</span>
    <button type="button" class="primary" data-vet-check>Check it</button>`;
  if (V.map?.error) return `<b>No layer map</b><span>${esc(V.map.error)}</span>`;
  return "<b>No layer map</b><span>The fingerprint found no attention or MLP tensors it reads.</span>";
}

function renderLegend() {
  const el = $("#vt-legend");
  if (!V.map?.layers?.length) { el.hidden = true; return; }
  el.hidden = false;
  const C = graphColors();
  const grad = `linear-gradient(90deg, ${C.graphStops.join(",")})`;
  const base = V.map.base;
  el.innerHTML = base
    ? `<div class="lg-row"><span class="lg-grad" style="background:${grad}"></span>same as ${esc(base)} → moved most</div>
       <div class="lg-row"><span class="lg-dot" style="background:${C.edited}"></span>moved from the base</div>
       <details><summary>How to read this</summary><p>Each layer has its attention tensor above the stream and its MLP tensor below.
       A node's color is how far its weights moved from ${esc(base)}${V.map.base_is === "closest" ? " (the closest model on this machine, not a confirmed base)" : ""}:
       1 − the correlation of 256 values read from both files. One sharp column is a targeted edit; every layer moving a little is a fine-tune.</p></details>`
    : `<div class="lg-row"><span class="lg-dot" style="background:${C.nodeNoData}"></span>no base on this machine to compare with</div>`;
}

function tip(hit, mx, my) {
  const el = $("#vt-tip");
  if (!hit) { el.hidden = true; return; }
  let html;
  if (hit.end) {
    const e = V.map.embed;
    html = hit.end === "embed" && e ? `<div class="tip-title">Embedding</div><code>${esc(e.tensor)}</code><div>${esc(movedText(e.corr))}</div>`
      : `<div class="tip-title">Output</div><div>The unembedding isn't fingerprinted.</div>`;
  } else {
    const u = units[hit.g];
    html = `<div class="tip-title">${u.title} · L${u.layer}</div><code>${esc(u.t.tensor)}</code> <span class="muted">${esc(u.t.shape.join(" × "))}</span>
      <div>${esc(movedText(u.t.corr))}${u.t.corr != null && V.map.base ? ` against ${esc(V.map.base)}` : ""}</div>`;
  }
  el.innerHTML = html;
  el.hidden = false;
  const box = $("#vt-stage").getBoundingClientRect();
  el.style.left = `${Math.min(mx + 14, box.width - 320)}px`;
  el.style.top = `${Math.max(8, my - 12)}px`;
}

// ---------------------------------------------------------------- the drawer: four answers, then the one on now

/** The action a finding belongs to: the fingerprint's "edited in these layers" is an edit, not an origin. */
const areaOf = (f) => (f.area === "origin" && /^Edited in /.test(f.title) ? "outside_weights" : f.area);
const findingsIn = (areas) => (V.report?.findings || []).filter((f) => areas.includes(areaOf(f)));
const why = (area) => (V.report?.not_checked || []).find((x) => x.area === area)?.why;

function span(layers) {
  const s = [...layers].sort((a, b) => a - b), runs = [];
  for (const l of s) { const r = runs[runs.length - 1]; if (r && l === r[1] + 1) r[1] = l; else runs.push([l, l]); }
  return runs.map(([a, b]) => (a === b ? `L${a}` : `L${a}–L${b}`)).join(", ");
}

/** Each action's one-line answer and its tone, from the report and the map. */
function answer(id) {
  const r = V.report;
  if (!r) return { tone: "todo", text: "Not checked yet" };
  const fs = findingsIn(AREAS[id]), worst = fs.find((f) => f.severity !== "note");
  if (id === "open") {
    const s = r.sections.files;
    if (worst) return { tone: worst.severity, text: worst.short || worst.title };
    return { tone: "clear", text: s ? `Nothing runs on load: ${s.files} files, ${s.pickles} pickle-based` : "Not checked" };
  }
  if (id === "origin") {
    const o = r.sections.origin;
    if (!o) return { tone: "todo", text: why("origin") || "Not checked" };
    if (o.closest && o.reading?.startsWith("derived")) return { tone: "clear", text: `Its weights are ${o.closest}'s` };
    const lead = worst || fs[0];
    return { tone: worst ? worst.severity : "clear", text: lead ? lead.short || lead.title : o.reading };
  }
  if (id === "edits") {
    const ts = units.filter((u) => u.t.corr != null), mv = ts.filter((u) => moved(u.t.corr) >= 1e-5);
    const outside = fs.filter((f) => f.severity !== "note");
    if (outside.length) return { tone: outside[0].severity, text: outside[0].short || outside[0].title };
    if (!ts.length) return { tone: "todo", text: "Needs a base to compare with" };
    if (!mv.length) return { tone: "clear", text: "Nothing moved: the same weights as the base" };
    const layers = [...new Set(mv.map((u) => u.layer))], total = new Set(ts.map((u) => u.layer)).size;
    return layers.length >= total * 0.8 ? { tone: "clear", text: "Every layer moved a little, as in a fine-tune" }
      : { tone: "moved", text: `${mv.length} of ${ts.length} tensors moved, in ${span(layers)}` };
  }
  if (!r.sections.inside_weights) return { tone: "todo", text: V.base ? `Not run: compares it with ${V.base.name}, loaded now` : "Not run: needs its base loaded" };
  return worst ? { tone: worst.severity, text: worst.short || worst.title } : { tone: "clear", text: "Nothing the weight checks look for" };
}

function findingHtml(f) {
  const call = f.next?.tool ? `${f.next.tool}(${Object.entries(f.next.args || {}).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ")})` : "";
  return `<article class="vt-f ${f.severity}">
    <h3>${esc(f.title)}</h3>${f.detail ? `<p>${esc(f.detail)}</p>` : ""}
    ${f.evidence?.length ? `<ul class="vt-ev">${f.evidence.map((e) => `<li>${esc(e)}</li>`).join("")}</ul>` : ""}
    ${f.where ? `<p class="vt-where">in <code>${esc(f.where)}</code></p>` : ""}
    ${f.next?.do ? `<p class="vt-next">→ ${esc(f.next.do)}</p>` : ""}
    ${call ? `<code class="vt-call">${esc(call)}</code>` : ""}</article>`;
}

/** What the action on now found beyond the findings: its files, the moved tensors, the neurons that moved most. */
function extraHtml(id) {
  if (id === "edits" && units.length) {
    const mv = units.filter((u) => moved(u.t.corr) >= 1e-5).sort((a, b) => a.t.corr - b.t.corr).slice(0, 12);
    if (!mv.length) return "";
    return `<h4>Moved most</h4><ul class="vt-list">${mv.map((u) => `<li><button type="button" class="link" data-fly="${units.indexOf(u)}">L${u.layer} ${u.title}</button>
      <span>r ${u.t.corr.toFixed(4)}</span></li>`).join("")}</ul>`;
  }
  if (id === "behavior" && V.map?.deep?.neurons?.length) {
    return `<h4>Neurons that moved most</h4><ul class="vt-list">${V.map.deep.neurons.map((n) => `<li><span>${esc(n.neuron)}</span><span>${(n.rel * 100).toFixed(1)}%</span></li>`).join("")}</ul>`;
  }
  if (id === "open" && V.report?.sections.files) {
    const s = V.report.sections.files;
    return `<h4>Its files</h4><p class="vt-files">${Object.entries(s.formats || {}).map(([k, n]) => `<span>${esc(k)} ${n}</span>`).join("")}</p>`;
  }
  return "";
}

function renderDrawer() {
  const body = $("#vt-body"), m = picked(), r = V.report;
  $("#vt-where").textContent = m ? m.name : "Vet";
  $("#vt-page").hidden = !r;
  if (r) $("#vt-page").href = `/llminspect/check/reports/${encodeURIComponent(r.id)}/html`;
  if (!m) { body.innerHTML = '<p class="vt-calm">Pick a model at the top left to see what loading it runs, where it comes from, what was edited and what the edit does.</p>'; return; }
  const head = r ? `<p class="vt-headline">${esc(r.summary.headline)}</p>
    <p class="vt-meta">${r.base ? `base ${esc(r.base.name)} (by ${esc(r.base.chosen_by)}) · ` : ""}checked ${esc(r.created.slice(0, 16).replace("T", " "))}</p>`
    : '<p class="vt-headline">Not checked yet.</p>';
  const rows = STEPS.map(([id, label], i) => {
    const a = answer(id);
    return `<li><button type="button" class="vt-act${V.act === id ? " on" : ""}" data-act="${id}"${V.act === id ? ' aria-current="step"' : ""}>
      <span class="n">${i + 1}</span><span class="lbl">${label}</span><span class="ans ${a.tone}"><i aria-hidden="true"></i>${esc(a.text)}</span></button></li>`;
  }).join("");
  let detail = "";
  if (r) {
    const fs = findingsIn(AREAS[V.act]), not = AREAS[V.act].map(why).find(Boolean);
    detail = (fs.length ? fs.map(findingHtml).join("")
      : `<p class="vt-calm">${not ? `${esc(not[0].toUpperCase() + not.slice(1))}.` : "Nothing these checks look for turned up here."}</p>`)
      + extraHtml(V.act)
      + `<p class="vt-agent">agent <code>check_model(candidate=${esc(JSON.stringify(m.path))}${V.act === "behavior" ? ", deep=true" : ""})</code></p>`;
  }
  body.innerHTML = `${head}<ol class="vt-acts" aria-label="The four checks">${rows}</ol><section class="vt-detail" aria-live="polite">${detail}</section>`;
}

// ---------------------------------------------------------------- models, checks, reports

const picked = () => V.models.find((x) => x.path === V.sel) || (V.other?.path === V.sel ? V.other : undefined);

/** A folder that isn't under the model folders, with its latest check, if any, from the checks run before. */
async function other(path) {
  const trim = (x) => x.trim().replace(/[\\/]+$/, "");
  const p = trim(path);
  let last = null;
  try {
    const h = (await api("/llminspect/check/reports")).find((x) => trim(x.path) === p);
    if (h) last = { id: h.id, headline: h.headline, counts: h.counts, created: h.created, deep: h.deep };
  } catch { /* no history */ }
  V.other = { name: p.split(/[\\/]/).pop(), path: p, gb: 0, formats: [], last };
  return p;
}

async function pickOther() {
  const path = prompt("Path of a model folder to vet");
  if (path?.trim()) await pick(await other(path));
  renderPicker();
}

function renderPicker() {
  const groups = Object.keys(GROUP).map((t) => {
    const ms = V.models.filter((m) => tone(m) === t).sort((a, b) => a.name.localeCompare(b.name));
    return ms.length ? `<optgroup label="${GROUP[t]} (${ms.length})">${ms.map((m) => `<option value="${esc(m.path)}">${esc(m.name)}${
      V.base?.path === m.path ? " · loaded" : ""}</option>`).join("")}</optgroup>` : "";
  }).join("");
  const sel = $("#vt-model");
  const other = V.other ? `<optgroup label="Another folder"><option value="${esc(V.other.path)}">${esc(V.other.name)}</option></optgroup>` : "";
  sel.innerHTML = `<option value="">${V.loaded ? `Pick a model (${V.models.length})` : "Reading models…"}</option>${groups}${other}
    <option value="__other">Another folder…</option>`;
  sel.value = picked() ? V.sel : "";
}

async function pick(path) {
  V.sel = path || null;
  remember(KEEP, V.sel);
  V.report = null; V.map = null; units = [];
  const m = picked();
  if (m?.last) await loadReport(m.last.id);
  else { drawMap(); renderDrawer(); refreshBar("vet"); }
}

async function loadReport(id) {
  try {
    const [r, map] = await Promise.all([api(`/llminspect/check/reports/${encodeURIComponent(id)}`),
      api(`/llminspect/check/reports/${encodeURIComponent(id)}/map`).catch((err) => ({ layers: [], error: err.message }))]);
    V.report = r; V.map = map;
  } catch (err) { V.report = null; V.map = { layers: [], error: err.message }; }
  drawMap();
  // Outside the Vet action, open on the first check with something to look at.
  if (currentStep().verb !== "vet") V.act = STEPS.map(([s]) => s).find((s) => ["high", "medium", "moved"].includes(answer(s).tone)) || "open";
  renderDrawer(); refreshBar("vet");
}

async function loadModels() {
  try {
    const d = await api("/llminspect/check/models");
    V.models = d.models; V.base = d.base;
  } catch (err) { noteBar("vet", err.message); }
  V.loaded = true;
  if (V.sel && !picked()) await other(V.sel);  // picked last time from another folder, or checked since
  else if (V.other?.path === V.sel) await other(V.sel);
  renderPicker();
  const m = picked();
  if (m?.last && m.last.id !== V.report?.id) await loadReport(m.last.id);
  else { drawMap(); renderDrawer(); refreshBar("vet"); }
}

async function start(deep = false) {
  if (!V.sel || V.job) return;
  try {
    const job = await post("/llminspect/check", { candidate: V.sel, base: deep ? "loaded" : "auto", deep });
    V.job = job.job; remember(KEEP_JOB, V.job);
    progress(job.step);
    if (!V.map?.layers?.length) drawMap();
    poll();
  } catch (err) { noteBar("vet", err.message); }
  refreshBar("vet");
}

async function startAll() {
  if (V.job) return;
  try {
    const job = await post("/llminspect/check/all", {});
    V.job = job.job; V.ticks = 0; remember(KEEP_JOB, V.job);
    progress(job.step);
    poll();
  } catch (err) { noteBar("vet", err.message); }
  refreshBar("vet");
}

async function poll() {
  clearTimeout(V.timer);
  if (!V.job) return;
  let job;
  try { job = await api(`/llminspect/check/jobs/${V.job}`); } catch {  // gone (the server restarted): forget it
    V.job = null; remember(KEEP_JOB, null); progress(null); refreshBar("vet"); return;
  }
  const all = Array.isArray(job.done);
  if (job.status === "running") {
    progress(all ? `${job.step} · ${job.done.length} done` : job.step);
    if (all && ++V.ticks % 6 === 0) loadModels();
    V.timer = setTimeout(poll, 700);
    return;
  }
  V.job = null; remember(KEEP_JOB, null);
  progress(null);
  if (job.status === "error") { noteBar("vet", job.error); drawMap(); refreshBar("vet"); return; }
  await loadModels();
  if (all) {
    const high = job.done.filter((d) => d.counts.high).length;
    noteBar("vet", `Checked ${job.done.length} models${high ? `: ${high} to look at first` : ""}`);
  } else {
    noteBar("vet", "Check done");
    await loadReport(job.report);
  }
}

function progress(step) {
  const el = $("#vt-progress");
  el.hidden = !step;
  el.innerHTML = step ? `<span class="vt-pulse" aria-hidden="true"></span>${esc(step)}` : "";
}

// ---------------------------------------------------------------- boot

function setAct(id) {
  if (!AREAS[id] || id === V.act) return;
  V.act = id;
  renderDrawer();
}

function boot() {
  if (!$("#view-vet")) return;
  try { V.sel = localStorage.getItem(KEEP); V.job = localStorage.getItem(KEEP_JOB); } catch { /* storage unavailable */ }
  const cur = currentStep();
  if (cur.verb === "vet" && AREAS[cur.step]) V.act = cur.step;
  graph = new WeightGraph($("#vt-canvas"), $("#vt-minimap"), {
    onHover: tip,
    onSelect: (g) => { if (g != null) graph.flyTo(g, 3); },
    onFrame: () => { $("#vt-zoom").textContent = `${graph.zoomPercent()}%`; },
  });
  graph.setColors(graphColors());
  graph.setState({ visible: false });
  $("#vt-zoom-in").addEventListener("click", () => graph.zoomAt(graph.W / 2, graph.H / 2, 1.5));
  $("#vt-zoom-out").addEventListener("click", () => graph.zoomAt(graph.W / 2, graph.H / 2, 1 / 1.5));
  $("#vt-fit").addEventListener("click", () => frame());
  $("#vt-model").addEventListener("change", (ev) => (ev.target.value === "__other" ? pickOther() : pick(ev.target.value)));
  $("#vt-form").addEventListener("submit", (ev) => { ev.preventDefault(); start(false); });
  $("#vt-empty").addEventListener("click", (ev) => { if (ev.target.closest("[data-vet-check]")) start(false); });
  $("#vt-body").addEventListener("click", (ev) => {
    const row = ev.target.closest("[data-act]");
    if (row) { setAct(row.dataset.act); openStep("vet", row.dataset.act); return; }
    const fly = ev.target.closest("[data-fly]");
    if (fly) graph.flyTo(Number(fly.dataset.fly), 3);
  });
  document.addEventListener("llminspect:step", (ev) => { if (ev.detail.verb === "vet") setAct(ev.detail.step); });
  // The graph follows the app's theme: the toggle (data-theme) or, with none set, the system's.
  const retheme = () => { graph.setColors(graphColors()); renderLegend(); };
  new MutationObserver(retheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", retheme);
  document.addEventListener("llminspect:view", (ev) => {
    V.visible = ev.detail === "vet";
    graph.setState({ visible: V.visible && !!V.map?.layers?.length });
    if (V.visible && !V.loaded) loadModels();
    if (V.visible) requestAnimationFrame(() => frame(true));
  });
  document.addEventListener("llminspect:model", () => { V.loaded = false; if (V.visible) loadModels(); });

  defineBar("vet", {
    groups: ["Check", "Read"],
    context: () => (V.job ? "Checking…" : picked() ? `<b>${esc(picked().name)}</b>` : "No model picked"),
    state: () => {
      const s = V.report?.sections || {};
      return { vet_files: !!s.files, vet_origin: !!s.origin, vet_edits: !!s.outside_weights || units.some((u) => u.t.corr != null),
        vet_behavior: !!s.inside_weights };
    },
    actions: [
      { id: "check", group: "Check", label: "Check", hint: "What loading it runs, where it comes from, what changed. Nothing is loaded.",
        need: () => (V.job ? "the check running now to finish" : V.sel ? null : "a model picked at the top left"), run: () => start(false) },
      { id: "all", group: "Check", label: "Check every model", hint: "Check every model on this machine, files only",
        need: () => (V.job ? "the check running now to finish" : V.models.length ? null : "models found on this machine"), run: startAll },
      { id: "deep", group: "Check", label: "Test against the loaded base", hint: "What the edit does: run it beside the loaded model on probe text",
        need: () => (V.job ? "the check running now to finish" : !V.base ? "its base model loaded (pick it in the model menu)"
          : !V.sel ? "a model picked at the top left" : V.sel === V.base.path ? "a model other than the loaded one" : null),
        run: () => start(true) },
      { id: "open", group: "Read", label: "Full report", hint: "The report as a page of its own, to save or share",
        need: () => (V.report ? null : "a check done"), run: () => window.open($("#vt-page").href, "_blank") },
      { id: "refresh", group: "Read", label: "Refresh models", hint: "Read the model folders again", run: () => { V.loaded = false; drawMap(); loadModels(); } },
    ],
  });
  renderPicker();
  renderDrawer();
  if (V.job) poll();  // a check still running from before a reload
  if (!$("#view-vet").hidden) { V.visible = true; loadModels(); }
}
boot();
