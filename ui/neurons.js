// The Neurons view: a secondary screen for single MLP neurons ("L5 N120"). A card says what a
// neuron does (logit lens, steering effect at ±dose, where it fires, what feeds it and what it feeds);
// a steer set doses one or more neurons with dose scaling and answers a prompt at several dose levels
// to verify, or steers the chat itself (brainscan/recorder/neurons.py). Any neuron named in the app
// opens here: a [data-open-neuron] link anywhere, or the brainscan:neuron event.
// A steer set can be saved as a knob (brainscan/recorder/knobs.py): a named set of neurons with doses,
// per model, that Steer, Ablate and Monitor use without finding the neurons again. The Knobs panel
// lists them; the tools that take a knob (choices) are on this screen's bar and in those actions.
import { defineBar, noteBar, refreshBar } from "/ui/actionbar.js";

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const vis = (s) => (s ?? "").replace(/\n/g, "⏎").replace(/\t/g, "⇥").replace(/^ /, "·") || "∅";
const n2 = (x, d = 2) => (x == null ? "–" : Number(x).toFixed(d));
const sign = (x, d = 2) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}`;
async function api(path, opts) {
  const res = await fetch(path, opts);
  const text = await res.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch { /* not JSON */ }
  if (!res.ok) throw new Error(body?.detail || text || `${res.status}`);
  return body;
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}) });

const N = { name: null, dose: 1, card: null, loading: null, history: [], set: [], sweep: null, busy: false, steering: [],
  index: null, results: null, searching: null, timer: 0,
  model: null, knobs: [], knobSeq: 0, factors: {}, confirm: null, kbusy: null, from: null };
const SET_KEY = "brainscan-neuron-set";  // + ":<model name>": a steer set belongs to one model

function message(text, tone = "") {
  const el = $("#nv-msg");
  if (el) { el.className = `wb-msg ${tone}`; el.textContent = text || ""; }
}

// ---------------------------------------------------------------- the card
async function open(name, { push = true } = {}) {
  name = String(name || "").trim();
  if (!/^L\d+\s*N\d+$/i.test(name)) { message("Name a neuron like L5 N120.", "error"); return; }
  name = name.toUpperCase().replace(/\s*N/, " N");
  if (push && N.name && N.name !== name) N.history.push(N.name);
  N.name = name;
  N.dose = Number($("#nv-dose").value) || 1;
  $("#nv-q").value = name;
  N.loading = name;
  N.card = null;
  render();
  try {
    const card = await api(`/brainscan/neurons/${encodeURIComponent(name)}?dose=${N.dose}`);
    if (N.loading !== name) return;
    N.card = card;
    message("");
  } catch (err) {
    if (N.loading === name) message(err.message, "error");
  }
  N.loading = null;
  render();
}

function chips(list, cls) {
  return list.map((t) => `<span class="nv-chip ${cls}" title="${esc(t.text)}: ${t.v}">${esc(vis(t.text))}</span>`).join("");
}
function links(list, key, fmt) {
  return list.length ? list.map((x) => `<button type="button" class="nv-link" data-open-neuron="${x.neuron}" title="Open ${x.neuron}">${x.neuron} <span>${fmt(x[key])}</span></button>`).join("")
    : `<span class="muted small">none</span>`;
}

function render() {
  refreshBar("neurons");
  renderSet();
  const body = $("#nv-body");
  if (!body) return;
  if (N.loading) {
    body.innerHTML = `<p class="muted">Reading ${esc(N.loading)} on the calibration texts: its lenses, where it fires, and what it feeds and is fed by…</p>`;
    return;
  }
  const c = N.card;
  if (!c) {
    body.innerHTML = `<p class="muted">Open a neuron by name, or from a cluster's neurons in Trace. A neuron is one channel of a layer's MLP output: it writes its activation × its output weights into the residual stream.</p>`;
    return;
  }
  const sl = c.steer_effect, a = c.activity, fb = c.fed_by, fd = c.feeds;
  body.innerHTML = `
    <div class="nv-head">
      <h3>${esc(c.neuron)}</h3>
      ${c.unit ? `<button type="button" class="link" data-trace-unit="${c.unit.g}">in cluster ${esc(c.unit.name)} (Trace)</button>` : ""}
      <span class="muted small">‖W_down‖ ${n2(c.write_norm, 3)} · median residual ${n2(c.median_norm, 1)} → dose ${c.dose} writes ${n2(c.dose * c.median_norm, 1)}</span>
    </div>
    <section class="nv-sec"><h4>Logit lens <span class="dim">what its write pushes, read straight out (weights only)</span></h4>
      <div class="nv-row"><span class="nv-lbl">up</span>${chips(c.logit_lens.up, "up")}</div>
      <div class="nv-row"><span class="nv-lbl">down</span>${chips(c.logit_lens.down, "down")}</div></section>
    <section class="nv-sec"><h4>Steering effect <span class="dim">what steering it does to the predictions, averaged over the calibration texts</span></h4>
      ${[sl.plus, sl.minus].map((s) => `<div class="nv-steer"><b>dose ${sign(s.dose, 2)}</b><span class="muted small">KL ${n2(s.kl, 3)}</span></div>
        <div class="nv-row"><span class="nv-lbl">more</span>${chips(s.up, "up")}</div>
        <div class="nv-row"><span class="nv-lbl">less</span>${chips(s.down, "down")}</div>`).join("")}</section>
    ${c.corpus ? `<section class="nv-sec"><h4>Where it fires <span class="dim">its top activations in the neuron index (${c.corpus.tokens.toLocaleString()} tokens): mean ${n2(c.corpus.mean)}, spread ${n2(c.corpus.std)}</span></h4>
      <ul class="nv-ctx">${c.corpus.contexts.map(ctxRow).join("")}</ul></section>` : ""}
    <section class="nv-sec"><h4>${c.corpus ? "On the calibration texts" : "Where it fires"} <span class="dim">mean ${n2(a.mean)}, max ${n2(a.max)}, ${a.positions} tokens</span></h4>
      <ul class="nv-ctx">${a.contexts.map(ctxRow).join("")}</ul>
      ${c.corpus ? "" : `<p class="hint">A small, fixed set of texts. Build the neuron index (Find neurons, above) for its top activations across a corpus.</p>`}</section>
    <section class="nv-sec two"><div><h4>Feeds <span class="dim">later neurons it moves when steered (z)</span></h4>
        <div class="nv-sub">toward</div><div class="nv-links">${links(fd.toward, "z", (z) => sign(z, 1))}</div>
        <div class="nv-sub">away</div><div class="nv-links">${links(fd.away, "z", (z) => sign(z, 1))}</div></div>
      <div><h4>Fed by <span class="dim">earlier neurons driving it where it fires</span></h4>
        <div class="nv-sub">drive it up</div><div class="nv-links">${links(fb.drives_up, "c", (v) => sign(v, 3))}</div>
        <div class="nv-sub">drive it down</div><div class="nv-links">${links(fb.drives_down, "c", (v) => sign(v, 3))}</div></div></section>`;
}

const ctxRow = (x) => `<li><span class="v">${n2(x.act)}</span><span class="t">…${esc(x.before)}<mark>${esc(x.token)}</mark>${esc(x.after)}</span></li>`;
const snip = (c) => (c ? `<span class="t">…${esc(c.before.slice(-60))}<mark>${esc(c.token)}</mark>${esc(c.after)}</span>` : "");

// ---------------------------------------------------------------- the index and search
async function loadIndex() {
  try { N.index = await api("/brainscan/neurons/index"); } catch { N.index = null; }
  renderIndex();
  if (N.index?.status === "building") { clearTimeout(N.timer); N.timer = setTimeout(loadIndex, 1500); }
}

function renderIndex() {
  const el = $("#nv-index");
  if (!el) return;
  refreshBar("neurons");
  const x = N.index;
  if (!x) { el.textContent = ""; return; }
  const m = x.meta;
  const built = m ? `Index: ${m.tokens.toLocaleString()} tokens from ${m.sources.join(", ")}; built ${new Date(m.built * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })} in ${m.seconds}s.` : "";
  el.innerHTML = x.status === "building" ? `Building the neuron index: ${esc(x.step || "")} ${Math.round((x.progress || 0) * 100)}%. The model is busy until it's done.`
    : x.status === "error" ? `<span class="warn-text">The index build failed: ${esc(x.error)}</span> ${esc(built)}`
    : m ? `${esc(built)} <button type="button" class="link" data-build>Rebuild</button>`
    : `No neuron index for this model yet. <button type="button" class="link" data-build>Build it</button>: one read of the corpus (the starter texts, your recorded runs and any text in the data folder's corpus/, or BRAINSCAN_CORPUS) keeps every neuron's top contexts and lens.`;
}

async function buildIndex() {
  try { N.index = await post("/brainscan/neurons/index", {}); } catch (err) { message(err.message, "error"); }
  loadIndex();
}

async function search(q) {
  q = String(q || "").trim();
  if (q.length < 2) { message("Search for at least two characters.", "error"); return; }
  N.searching = q; N.results = null; renderResults();
  try { N.results = await api(`/brainscan/neurons/search?q=${encodeURIComponent(q)}&limit=12`); message(""); }
  catch (err) { message(err.message, "error"); }
  N.searching = null; renderResults();
}

function renderResults() {
  const box = $("#nv-results");
  if (!box) return;
  refreshBar("neurons");
  if (N.searching) { box.innerHTML = `<p class="muted small">Searching for “${esc(N.searching)}”: reading it where it occurs…</p>`; return; }
  const r = N.results;
  if (!r) { box.innerHTML = ""; return; }
  const col = (title, sub, rows, fmt) => `<div class="nv-col"><h4>${title} <span class="dim">${sub}</span></h4>
    <ul>${rows.map(fmt).join("") || `<li class="muted small">none</li>`}</ul></div>`;
  const link = (x, extra) => `<button type="button" class="nv-link" data-open-neuron="${x.neuron}">${x.neuron} <span>${extra}</span></button>`;
  box.innerHTML = `<p class="small">“${esc(r.query)}” ${r.in_corpus ? `occurs ${r.matches} time${r.matches === 1 ? "" : "s"} in the index's ${r.tokens.toLocaleString()} tokens` : "isn't in the corpus: neurons that fire on it are read on the query itself"}.</p>
    <div class="nv-cols">
      ${col("Fire on it", "z, where it occurs", r.fires_on, (x) => `<li>${link(x, `z ${n2(x.z, 1)}`)}</li>`)}
      ${col("Push it", "a logit-lens top token", r.pushes, (x) => `<li>${link(x, `#${x.rank + 1}`)}<span class="nv-toks">${x.tokens.map((t) => esc(vis(t))).join(" ")}</span></li>`)}
      ${col("Have it in their top contexts", "hits", r.contexts, (x) => `<li>${link(x, `${x.hits}× · ${n2(x.act)}`)}${snip(x.context)}</li>`)}
    </div>`;
}

// ---------------------------------------------------------------- steer and verify
const setKey = () => (N.model ? `${SET_KEY}:${N.model}` : null);
function saveSet() {
  const key = setKey();
  if (key) { try { localStorage.setItem(key, JSON.stringify(N.set)); } catch { /* storage unavailable */ } }
}

/** This model's steer set. The key used to be one for every model: the first model loaded takes that set. */
function loadSet() {
  N.set = [];
  const key = setKey();
  if (!key) return;
  try {
    let raw = localStorage.getItem(key);
    const old = localStorage.getItem(SET_KEY);
    if (old != null) {
      if (raw == null) { raw = old; localStorage.setItem(key, old); }
      localStorage.removeItem(SET_KEY);
    }
    const set = JSON.parse(raw || "[]");
    N.set = Array.isArray(set) ? set.filter((s) => /^L\d+ N\d+$/.test(s?.neuron) && Number.isFinite(Number(s.dose))).map((s) => ({ neuron: s.neuron, dose: Number(s.dose) })) : [];
  } catch { N.set = []; }
}

function renderSet() {
  const ul = $("#nv-set");
  if (!ul) return;
  ul.innerHTML = N.set.map((s, i) => `<li data-i="${i}"><button type="button" class="link mono" data-open-neuron="${s.neuron}">${s.neuron}</button>
      <label>dose <input type="number" step="0.25" min="-16" max="16" value="${s.dose}" data-dose></label>
      <button type="button" data-remove aria-label="Remove ${s.neuron} from the steer set">✕</button></li>`).join("")
    || `<li class="muted" style="border:none;background:none;padding:0">No neurons yet. Open one and press Add to steer set, or type several, like L5 N120 +1 and L9 N64 −1${N.knobs.length ? ". Or Load a saved knob below" : ""}.</li>`;
  const live = N.steering.length;
  $("#nv-live").textContent = live ? `Steering the chat with ${live} neuron${live === 1 ? "" : "s"}: ${N.steering.map((iv) => `${iv.knob ? `${iv.knob} · ` : ""}${iv.direction} ${sign(iv.strength, 2)}`).join(", ")}` : "";
}

function addToSet(neuron = N.name, dose = N.dose) {
  if (!neuron) return;
  const at = N.set.findIndex((s) => s.neuron === neuron);
  if (at >= 0) N.set[at].dose = dose; else N.set.push({ neuron, dose });
  saveSet(); render();
}

/** "L5 N120 +1, L9 N64 -0.5" → the steer set. */
function addTyped(text) {
  const found = [...String(text).matchAll(/L(\d+)\s*N(\d+)\s*([+−-]?\s*\d*\.?\d+)?/gi)];
  for (const m of found) addToSet(`L${m[1]} N${m[2]}`, m[3] ? Number(m[3].replace("−", "-").replace(/\s/g, "")) : 1);
  return found.length;
}

async function runSweep() {
  if (!N.set.length) { message("Add a neuron to the steer set first.", "error"); return; }
  const prompt = $("#nv-prompt").value.trim();
  if (!prompt) { message("Give a prompt to answer.", "error"); return; }
  const factors = $("#nv-factors").value.split(/[,\s]+/).map(Number).filter((x) => Number.isFinite(x)).slice(0, 8);
  N.busy = true; N.sweep = null; renderSweep(); refreshBar("neurons");
  try {
    N.sweep = await post("/brainscan/neurons/sweep", { prompt, steers: N.set, factors: factors.length ? factors : [0, 1], max_tokens: Number($("#nv-max").value) || 60 });
    message("");
  } catch (err) { message(err.message, "error"); }
  N.busy = false; renderSweep(); refreshBar("neurons");
}

function renderSweep() {
  const box = $("#nv-sweep");
  if (!box) return;
  if (N.busy) { box.innerHTML = `<p class="muted small">Answering at each dose level (greedy)…</p>`; return; }
  if (!N.sweep) { box.innerHTML = ""; return; }
  box.innerHTML = N.sweep.map((r) => `<div class="nv-out${r.factor === 0 ? " base" : ""}"><div class="nv-out-head"><b>× ${r.factor}</b>
      <span class="muted small">${r.factor === 0 ? "unsteered" : r.doses.map((d, i) => `${N.set[i]?.neuron ?? ""} ${sign(d, 2)}`).join(", ")}</span></div>
      <p>${esc(r.text)}</p></div>`).join("");
}

async function refreshSteering() {
  try { N.steering = (await api("/brainscan/neurons")).steering || []; } catch { N.steering = []; }
  renderSet(); refreshBar("neurons");
}

/** Steering changed here: the knob rows, this bar and Watch's steering panel (ui/app.js) catch up. */
async function steeringChanged() {
  await Promise.all([refreshSteering(), loadKnobs()]);
  document.dispatchEvent(new CustomEvent("brainscan:steering"));
}

async function steerChat() {
  if (!N.set.length) return;
  try {
    // Pressing it again replaces the set's steers rather than stacking them; a knob's steers stay.
    await refreshSteering();
    for (const iv of N.steering.filter((x) => !x.knob)) await api(`/brainscan/interventions/${iv.id}`, { method: "DELETE" }).catch(() => {});
    for (const s of N.set) await post("/brainscan/neurons/steer", { neuron: s.neuron, dose: s.dose, phase: "all" });
    await steeringChanged();
    noteBar("neurons", "Every new request is steered with the set, prompt and reply alike, and marked ▲steered");
  } catch (err) { message(err.message, "error"); }
}

async function unsteer() {
  for (const iv of N.steering) await api(`/brainscan/interventions/${iv.id}`, { method: "DELETE" }).catch(() => {});
  await steeringChanged();
  noteBar("neurons", "Neuron steering removed");
}

// ---------------------------------------------------------------- knobs
// A knob is the steer set with a name (brainscan/recorder/knobs.py). Steering one writes its neurons
// at their doses × a multiple, tagged with the knob, so steering it again replaces it.
const kurl = (name, tail = "") => `/brainscan/knobs/${encodeURIComponent(name)}${tail}`;
const knobOf = (name) => N.knobs.find((k) => k.name === name);
const dose = (x) => `${x >= 0 ? "+" : "−"}${+Math.abs(x).toFixed(2)}`;
const count = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

async function loadKnobs() {
  const seq = ++N.knobSeq;
  let r = null;
  try { r = await api("/brainscan/knobs"); } catch { /* an older recorder, or still loading: keep what is shown */ }
  if (seq !== N.knobSeq) return;
  if (r) { N.knobs = r.knobs || []; N.knobError = r.error || null; }
  renderKnobs(); renderSet(); refreshBar("neurons");
}

/** The Knobs panel's message line; `html` (already escaped) follows the text. */
function knobMsg(text, tone = "", html = "") {
  const el = $("#nk-msg");
  if (el) { el.className = `wb-msg ${tone}`; el.innerHTML = `${esc(text)}${html ? ` ${html}` : ""}`; }
}
function saveMsg(text, tone = "") {
  const el = $("#nv-ksave-msg");
  if (el) { el.className = `wb-msg ${tone}`; el.textContent = text || ""; }
}

function knobRow(k) {
  const live = (k.active || []).length, ask = N.confirm === k.name, busy = !!N.kbusy;
  const act = (id, label, title, off = false) => `<button type="button" class="btn sm" data-act="${id}" title="${esc(title)}"${busy || off ? " disabled" : ""}>${label}</button>`;
  return `<li class="nk-row${live ? " on" : ""}" data-name="${esc(k.name)}">
    <div class="nk-head"><b class="mono">${esc(k.name)}</b>${live ? `<span class="pill ok">steering now</span>` : ""}
      ${k.description ? `<span class="muted small">${esc(k.description)}</span>` : ""}</div>
    <div class="nv-links">${k.neurons.map((n) => `<button type="button" class="nv-link" data-open-neuron="${esc(n.neuron)}" title="Open ${esc(n.neuron)}">${esc(n.neuron)} <span>${dose(n.dose)}</span></button>`).join("")}</div>
    <div class="nk-acts">
      ${act("load", "Load", "Replace the steer set with this knob's neurons")}
      <span class="nk-steer"><label>× <input type="number" step="0.25" min="-16" max="16" value="${esc(N.factors[k.name] ?? 1)}" data-factor aria-label="Multiple of ${esc(k.name)}'s doses to steer with"></label>
        ${act("steer", "Steer chat", "Steer every new request with this knob, at this multiple of its doses")}
        ${act("stop", "Stop", "Stop steering the chat with this knob", !live)}</span>
      ${act("sweep", "Sweep", "Load it into the steer set and answer the prompt at each dose level")}
      ${act("monitor", "Monitor", "Watch each of its neurons as the model writes, and flag a run where one stays unusually active")}
      ${act("ablate", "Ablate", "Add a weight edit that zeroes these neurons' output weights. It isn't applied until you apply it")}
      ${ask ? `<span class="nk-ask">Delete? <button type="button" class="link danger" data-act="delete-yes">Yes</button> <button type="button" class="link" data-act="delete-no">No</button></span>`
        : `<button type="button" class="link danger" data-act="delete" ${live ? 'disabled title="Stop steering with it first"' : `title="Delete this knob"${busy ? " disabled" : ""}`}>Delete</button>`}
    </div></li>`;
}

function renderKnobs() {
  const ul = $("#nk-list");
  if (!ul) return;
  ul.innerHTML = (N.knobError ? `<li class="warn-text">${esc(N.knobError)}</li>` : "")
    + (N.knobs.map(knobRow).join("") || `<li class="muted nk-empty">Save the steer set as a knob to reuse it from Steer, Ablate and Monitor.</li>`);
}

function loadKnob(name) {
  const k = knobOf(name);
  if (!k) return;
  N.set = k.neurons.map((n) => ({ neuron: n.neuron, dose: n.dose }));
  N.from = name;
  saveSet(); render();
  knobMsg(`Loaded ${name} into the steer set.`, "ok");
}

async function steerKnob(name, factor = 1) {
  if (!Number.isFinite(factor)) throw new Error("Give the multiple as a number.");
  const ivs = await post(kurl(name, "/steer"), { factor });
  await steeringChanged();
  knobMsg(`Steering the chat with ${name}${factor === 1 ? "" : ` × ${factor}`}: ${count(Array.isArray(ivs) ? ivs.length : knobOf(name)?.neurons.length ?? 0, "neuron")}. Every new request is steered, prompt and reply alike, and marked ▲steered.`, "ok");
  noteBar("neurons", `Steering the chat with knob ${name}`);
}

async function stopKnob(name) {
  await api(kurl(name, "/steer"), { method: "DELETE" });
  await steeringChanged();
  knobMsg(`Stopped steering with ${name}.`, "ok");
  noteBar("neurons", `Stopped steering with knob ${name}`);
}

async function monitorKnob(name) {
  const rules = await post(kurl(name, "/monitor"), {});
  const n = Array.isArray(rules) ? rules.length : 0, first = rules?.[0];
  const rest = first?.rule ? String(first.rule).replace(first.direction || "", "").trim() : "";  // "z ≥ 4 for 2 tokens in a row, on …"
  document.dispatchEvent(new CustomEvent("brainscan:monitors"));
  knobMsg(`Watching ${count(n, "neuron")} of ${name}${rest ? `: a flag when ${rest}` : ""}.`, "ok", `<button type="button" class="link" data-pick="monitor">Open the Monitor action</button>`);
  noteBar("neurons", `Watching ${count(n, "neuron")} of knob ${name}`);
}

async function ablateKnob(name) {
  await post(kurl(name, "/ablate"), { factor: 0 });
  document.dispatchEvent(new CustomEvent("brainscan:workbench"));  // the edit stack (ui/edits.js) and Tune refresh
  knobMsg("Edit added (not applied): apply it in Ablate › Apply.", "ok",
    `<button type="button" class="link" data-pick="ablate">Open Ablate</button> <button type="button" class="link" data-show="workbench">Edit stack in Tune</button>`);
  noteBar("neurons", `Edit for knob ${name} added, not applied`);
}

async function deleteKnob(name) {
  await api(kurl(name), { method: "DELETE" });
  delete N.factors[name];
  if (N.from === name) N.from = null;
  await loadKnobs();
  knobMsg(`Deleted ${name}.`, "ok");
}

/** Run one knob action with the rows locked, and its result or error in the Knobs panel. */
async function knobAct(name, fn) {
  if (N.kbusy) return;
  N.kbusy = name; N.confirm = null; renderKnobs();
  try { await fn(); } catch (err) { knobMsg(err.message, "error"); }
  N.kbusy = null; renderKnobs();
}

function onKnobClick(ev) {
  const b = ev.target.closest("[data-act]"), li = ev.target.closest("li[data-name]");
  if (!b || !li || N.kbusy) return;
  const name = li.dataset.name, act = b.dataset.act;
  if (act === "load") loadKnob(name);
  else if (act === "sweep") { loadKnob(name); $("#h-nsteer")?.scrollIntoView({ behavior: "smooth", block: "start" }); runSweep(); }
  else if (act === "delete" || act === "delete-no") { N.confirm = act === "delete" ? name : null; renderKnobs(); }
  else if (act === "steer") {
    const raw = li.querySelector("[data-factor]").value.trim();
    knobAct(name, () => steerKnob(name, raw === "" ? NaN : Number(raw)));
  }
  else if (act === "stop") knobAct(name, () => stopKnob(name));
  else if (act === "monitor") knobAct(name, () => monitorKnob(name));
  else if (act === "ablate") knobAct(name, () => ablateKnob(name));
  else if (act === "delete-yes") knobAct(name, () => deleteKnob(name));
}

/** Save as knob: the form under the steer set. */
function openSave() {
  const k = knobOf(N.from);
  if (k && !$("#nv-kname").value) { $("#nv-kname").value = k.name; $("#nv-kdesc").value = k.description || ""; }
  $("#nv-save").hidden = false;
  saveMsg(""); saveNote();
  $("#nv-kname").focus(); $("#nv-kname").select();
  $("#nv-save").scrollIntoView({ behavior: "smooth", block: "center" });
}
function saveNote() {
  const old = knobOf($("#nv-kname").value.trim());
  $("#nv-ksave-note").textContent = old ? `${old.name} is already saved: saving replaces its neurons and doses.`
    : "Letters, digits, _ . - only. A name that's already saved is replaced.";
}
async function saveKnob(ev) {
  ev.preventDefault();
  if (!N.set.length) { saveMsg("Add a neuron to the steer set first.", "error"); return; }
  try {
    const k = await post("/brainscan/knobs", { name: $("#nv-kname").value.trim(), description: $("#nv-kdesc").value.trim(), neurons: N.set });
    N.from = k?.name || $("#nv-kname").value.trim();
    $("#nv-save").hidden = true; $("#nv-kname").value = ""; $("#nv-kdesc").value = "";
    saveMsg("");
    await loadKnobs();
    knobMsg(`Saved ${N.from}: ${count(N.set.length, "neuron")}.`, "ok");
    noteBar("neurons", `Saved knob ${N.from}`);
  } catch (err) { saveMsg(err.message, "error"); }
}

/** The tools that take a knob (a menu of the saved ones), here and in the Steer, Ablate and Monitor actions. */
const needKnob = () => (N.knobs.length ? null : "a saved knob");
const knobChoices = (fn) => () => N.knobs.map((k) => ({ label: `${k.name} · ${count(k.neurons.length, "neuron")}`,
  run: () => { $("#h-nknobs")?.scrollIntoView({ behavior: "smooth", block: "start" }); knobAct(k.name, () => fn(k.name)); } }));

/** The recorder switched models: a steer set and the knobs belong to one model. */
function adoptModel(name) {
  const first = !N.model;
  if (name) N.model = name;
  loadSet();
  N.from = null; N.confirm = null; N.factors = {};
  if (!first) { N.knobs = []; N.knobError = null; }
}
function onStatus(st) {
  const name = st?.model?.name;
  if (name && name !== N.model) { adoptModel(name); render(); renderKnobs(); loadKnobs(); return; }
  // Knobs saved or deleted elsewhere (the API, an agent): status lists their names.
  const names = (st?.knobs || []).slice().sort().join("\n");
  if (st?.knobs && names !== N.knobs.map((k) => k.name).sort().join("\n")) loadKnobs();
}

// ---------------------------------------------------------------- the bar
function setupBar() {
  const scroll = (sel) => $(sel)?.scrollIntoView({ behavior: "smooth", block: "start" });
  defineBar("neurons", {
    groups: ["Find", "Steer", "Verify", "Change", "Monitor"],
    context: () => (N.card ? `<b>${esc(N.card.neuron)}</b> · dose ${N.card.dose}` : N.loading ? `Reading <b>${esc(N.loading)}</b>…` : "No neuron open")
      + (N.set.length ? ` · steer set: ${N.set.length}` : "") + (N.knobs.length ? ` · knobs: ${N.knobs.length}` : "") + (N.steering.length ? ` · <b>steering the chat</b>` : ""),
    state: () => ({ neuron: N.card?.neuron || null, steer_set: N.set.length, neuron_steering: N.steering.length, knobs: N.knobs.length }),
    actions: [
      { id: "search", group: "Find", label: "Search", hint: "Find neurons by text: ones that fire on it, push it, or have it in their top contexts",
        need: () => (N.index?.meta ? null : "the neuron index: build it first"), run: () => { scroll("#h-nsearch"); $("#nv-s").focus(); } },
      { id: "build", group: "Find", label: () => (N.index?.meta ? "Rebuild index" : "Build index"), hint: "Read the corpus once and keep every neuron's top contexts and lens",
        need: () => (N.index?.status === "building" ? "the build running now to finish" : null), run: buildIndex },
      { id: "go", group: "Find", label: "Go to neuron", hint: "Open a neuron by name, like L5 N120", run: () => { $("#nv-q").focus(); $("#nv-q").select(); } },
      { id: "back", group: "Find", label: "Back", hint: "The neuron opened before this one",
        need: () => (N.history.length ? null : "a neuron opened before this one"), run: () => open(N.history.pop(), { push: false }) },
      { id: "add", group: "Steer", label: "Add to steer set", hint: "Steer this neuron, at the dose above, with the others in the set",
        need: () => (N.card ? null : "a neuron open"), run: () => addToSet() },
      { id: "chat", group: "Steer", label: "Steer the chat", hint: "Steer every new request with the set, prompt and reply alike",
        need: () => (N.set.length ? null : "a neuron in the steer set"), run: steerChat },
      { id: "unsteer", group: "Steer", label: "Remove neuron steering", hint: "Stop steering the chat with neurons, knobs included",
        need: () => (N.steering.length ? null : "neurons steering the chat"), run: unsteer },
      { id: "saveknob", group: "Steer", label: "Save as knob", hint: "Name the steer set and keep it for this model, to steer, ablate or watch again",
        need: () => (N.set.length ? null : "a neuron in the steer set"), run: openSave },
      { id: "knob", group: "Steer", label: "Steer a knob", hint: "Steer the chat with a saved knob, at its own doses",
        need: needKnob, choices: knobChoices((name) => steerKnob(name, 1)) },
      { id: "sweep", group: "Verify", label: "Run sweep", hint: "Answer the prompt at each dose level, side by side",
        need: () => (N.busy ? "the sweep running now to finish" : N.set.length ? null : "a neuron in the steer set"), run: () => { scroll("#h-nsteer"); runSweep(); } },
      { id: "knob-ablate", group: "Change", label: "Ablate a knob", hint: "Add a weight edit that zeroes a knob's neurons. It isn't applied until you apply it",
        need: needKnob, choices: knobChoices(ablateKnob) },
      { id: "knob-monitor", group: "Monitor", label: "Watch a knob", hint: "Flag a run where any of a knob's neurons stays unusually active",
        need: needKnob, choices: knobChoices(monitorKnob) },
    ],
  });
}

function boot() {
  if (!$("#view-neurons")) return;
  setupBar();  // the steer set loads once the model's name is known (onStatus)
  $("#nv-go").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const q = $("#nv-q").value;
    if (/L\d+\s*N\d+.*L\d+\s*N\d+/i.test(q) || /[+−-]\s*\d/.test(q)) { if (addTyped(q)) { message("Added to the steer set.", "ok"); return; } }
    open(q);
  });
  $("#nv-dose").addEventListener("change", () => { if (N.name) open(N.name, { push: false }); });
  $("#nv-sform").addEventListener("submit", (ev) => { ev.preventDefault(); search($("#nv-s").value); });
  $("#nv-index").addEventListener("click", (ev) => { if (ev.target.closest("[data-build]")) buildIndex(); });
  $("#nv-run").addEventListener("click", runSweep);
  $("#nv-set").addEventListener("change", (ev) => {
    const li = ev.target.closest("li[data-i]");
    if (li && ev.target.matches("[data-dose]")) { N.set[Number(li.dataset.i)].dose = Number(ev.target.value) || 0; saveSet(); }
  });
  $("#nv-set").addEventListener("click", (ev) => {
    const li = ev.target.closest("li[data-i]");
    if (li && ev.target.matches("[data-remove]")) { N.set.splice(Number(li.dataset.i), 1); saveSet(); render(); }
  });
  $("#nv-body").addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-trace-unit]");
    if (b) { document.dispatchEvent(new CustomEvent("brainscan:show", { detail: "weights" })); document.dispatchEvent(new CustomEvent("brainscan:trace", { detail: { unit: Number(b.dataset.traceUnit) } })); }
  });
  // A neuron named anywhere in the app opens here (capture phase: before a row's own click).
  document.addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-open-neuron]");
    if (!b) return;
    ev.preventDefault(); ev.stopPropagation();
    document.dispatchEvent(new CustomEvent("brainscan:neuron", { detail: b.dataset.openNeuron }));
  }, true);
  document.addEventListener("brainscan:neuron", (ev) => {
    if ($("#view-neurons").hidden) document.dispatchEvent(new CustomEvent("brainscan:show", { detail: "neurons" }));
    open(ev.detail);
  });
  $("#nv-save").addEventListener("submit", saveKnob);
  $("#nv-ksave-cancel").addEventListener("click", () => { $("#nv-save").hidden = true; saveMsg(""); });
  $("#nv-kname").addEventListener("input", saveNote);
  $("#nk-list").addEventListener("click", onKnobClick);
  $("#nk-list").addEventListener("input", (ev) => {
    const li = ev.target.closest("li[data-name]");
    if (li && ev.target.matches("[data-factor]")) N.factors[li.dataset.name] = ev.target.value;
  });
  $("#nk-msg").addEventListener("click", (ev) => {  // a link in a knob message: an action to pick, or a screen
    const b = ev.target.closest("[data-pick], [data-show]");
    if (b) document.dispatchEvent(new CustomEvent(b.dataset.pick ? "brainscan:pick" : "brainscan:show", { detail: b.dataset.pick || b.dataset.show }));
  });
  document.addEventListener("brainscan:status", (ev) => onStatus(ev.detail));  // every poll (ui/app.js)
  document.addEventListener("brainscan:view", (ev) => { if (ev.detail === "neurons") { refreshSteering(); loadIndex(); loadKnobs(); } });
  document.addEventListener("brainscan:model", (ev) => {
    N.card = null; N.history = []; N.sweep = null; N.results = null; N.steering = [];
    adoptModel(ev.detail?.name);
    render(); renderSweep(); renderResults(); renderKnobs(); loadIndex(); refreshSteering(); loadKnobs();
  });
  render();
  renderKnobs();
  refreshSteering();
  loadIndex();
  loadKnobs();
  api("/brainscan/status").then(onStatus).catch(() => { /* the recorder is loading: the next poll tells */ });
}
boot();
