// Brainscan workbench: find directions, edit weights, compare original vs edited, export variants.
import { defineBar, refreshBar } from "/ui/actionbar.js";

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
async function api(path, opts) {
  const res = await fetch(path, opts);
  const text = await res.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch { /* not JSON */ }
  if (!res.ok) throw new Error(body?.detail || text || `${res.status}`);
  return body;
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}) });
const lines = (sel) => $(sel).value.split("\n").map((s) => s.trim()).filter(Boolean);
const pct = (x) => `${Math.round(x * 100)}%`;

const W = { sweep: null, layer: null, status: null };

const EXAMPLE = {
  a: ["Où est la gare la plus proche ?", "Il fait très beau aujourd'hui.", "Fermez la fenêtre, s'il vous plaît.",
    "Je voudrais une tasse de café.", "Combien coûte ce livre ?", "Mon frère habite à Lyon.",
    "Nous partons en vacances demain.", "Pouvez-vous m'aider ?"],
  b: ["Where is the nearest train station?", "The weather is lovely today.", "Please close the window.",
    "I would like a cup of coffee.", "How much does this book cost?", "My brother lives in Leeds.",
    "We are going on holiday tomorrow.", "Can you help me?"],
};

function message(sel, text, tone = "") {
  const el = $(sel);
  el.className = `wb-msg ${tone}`;
  el.textContent = text || "";
}

async function busy(btn, label, fn) {
  const old = btn.textContent;
  btn.disabled = true; btn.textContent = label;
  try { return await fn(); } finally { btn.disabled = false; btn.textContent = old; }
}

// ---------------------------------------------------------------- tabs
// The screens. Actions move between them (ui/actionbar.js); "actions" is the forget-a-fact flow.
const VIEWS = ["actions", "scanner", "weights", "workbench", "neurons"];

function show(view) {
  for (const v of VIEWS) $(`#view-${v}`).hidden = v !== view;
  try { localStorage.setItem("brainscan-view", view); } catch { /* storage unavailable */ }
  if (view === "workbench") refresh();
  document.dispatchEvent(new CustomEvent("brainscan:view", { detail: view }));
}

// ---------------------------------------------------------------- state
async function refresh() {
  try { W.status = await api("/brainscan/status"); } catch (err) { message("#wb-edit-msg", err.message, "error"); return; }
  if (W.status.status !== "ready") { message("#wb-edit-msg", `Recorder ${W.status.status}: ${W.status.step || W.status.error || ""}`); return; }
  renderDirections(); renderEdits();
  refreshBar("workbench");
  if (!$("#wb-path").value) {
    const slug = (W.status.model.name || "model").split("/").pop().replace(/[^A-Za-z0-9._-]+/g, "-");
    $("#wb-path").value = `~/brainscan-exports/${slug}-edited-${new Date().toISOString().slice(0, 10)}`;
  }
}

// ---------------------------------------------------------------- 1. find a direction
async function analyze(ev) {
  const body = { positive: lines("#wb-a"), negative: lines("#wb-b"), position: $("#wb-pos").value, chat: $("#wb-chat").checked };
  if (body.positive.length < 2 || body.negative.length < 2) { message("#wb-find-msg", "Each set needs at least two examples.", "error"); return; }
  message("#wb-find-msg", `Reading ${body.positive.length + body.negative.length} examples at every layer…`);
  try {
    W.sweep = await busy(ev.target, "Analyzing…", () => post("/brainscan/directions/analyze", body));
    W.layer = W.sweep.best;
    $("#wb-save-row").hidden = false;
    $("#wb-layer").max = W.sweep.layers.length - 1;
    $("#wb-layer").value = W.layer;
    renderSweep();
    const best = W.sweep.layers[W.sweep.best];
    message("#wb-find-msg", `Best separation at layer ${best.layer}: d = ${best.d}${W.sweep.held_out ? ` ± ${best.d_sd} across ${W.sweep.folds} folds` : ""}, ${W.sweep.held_out ? "cross-validated" : "in-sample"} accuracy ${pct(best.acc)}.` +
      (W.sweep.held_out ? " With few examples the best layer can shift; 30 or more per set gives a steadier choice." : " Add four or more examples per set to cross-validate."));
  } catch (err) { message("#wb-find-msg", err.message, "error"); }
}

function renderSweep() {
  const s = W.sweep; if (!s) return;
  const n = s.layers.length, w = 640, h = 170, padL = 40, padR = 10, padT = 18, padB = 26;
  const ds = s.layers.map((x) => x.d), max = Math.max(...ds.map(Math.abs), 1e-6), neg = ds.some((d) => d < 0);
  const plotH = h - padT - padB, base = neg ? padT + plotH / 2 : padT + plotH, k = (neg ? plotH / 2 : plotH) / max;
  const bw = (w - padL - padR) / n;
  const c = { bar: cssVar("--series-1"), sel: cssVar("--focus"), axis: cssVar("--axis"), muted: cssVar("--muted"), ink: cssVar("--ink-2") };
  const bars = s.layers.map((x, i) => {
    const bh = Math.max(1, Math.abs(x.d) * k), y = x.d >= 0 ? base - bh : base, sel = i === W.layer;
    return `<g class="bar" data-layer="${i}"><rect x="${padL + i * bw}" y="${padT}" width="${bw}" height="${plotH}" fill="transparent"/>
      <rect x="${padL + i * bw + Math.min(1, bw * 0.15)}" y="${y}" width="${Math.max(1, bw - Math.min(2, bw * 0.3))}" height="${bh}" rx="${bw > 8 ? 2 : 0}"
        fill="${sel ? c.sel : c.bar}" opacity="${sel ? 1 : 0.8}"/>
      <title>Layer ${i}: separation d = ${x.d}${s.held_out ? ` ± ${x.d_sd}` : ""}, ${s.held_out ? "cross-validated" : "in-sample"} accuracy ${pct(x.acc)}</title></g>`;
  }).join("");
  const sel = s.layers[W.layer];
  const selX = padL + W.layer * bw + bw / 2, selY = sel.d >= 0 ? base - Math.abs(sel.d) * k - 4 : base + Math.abs(sel.d) * k + 12;
  $("#wb-sweep").innerHTML = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Separation by layer; best at layer ${s.best}">
    <line x1="${padL}" x2="${w - padR}" y1="${base}" y2="${base}" stroke="${c.axis}"/>
    ${bars}
    <text x="${selX}" y="${selY}" text-anchor="middle" font-size="11" fill="${c.ink}">L${W.layer} · d ${sel.d}</text>
    <text x="${padL}" y="${h - 6}" font-size="10" fill="${c.muted}">layer 0</text>
    <text x="${w - padR}" y="${h - 6}" font-size="10" fill="${c.muted}" text-anchor="end">layer ${n - 1}</text>
    <text x="4" y="${padT + 4}" font-size="10" fill="${c.muted}">d</text>
  </svg><div class="small muted">Separation d by layer (gap between set means in units of within-set spread). Click a bar to choose the layer.</div>`;
  $("#wb-sweep").querySelectorAll("g.bar").forEach((g) => {
    g.onclick = () => { W.layer = Number(g.dataset.layer); $("#wb-layer").value = W.layer; renderSweep(); };
  });
}

async function saveDirection(ev) {
  const name = $("#wb-name").value.trim();
  const layer = Number($("#wb-layer").value);
  if (!name) { message("#wb-find-msg", "Give the direction a name.", "error"); return; }
  try {
    await busy(ev.target, "Saving…", () => post("/brainscan/directions/save", { id: W.sweep.id, layer, name,
      description: `Set A vs set B (${W.sweep.n_positive} vs ${W.sweep.n_negative} examples), built in the workbench.` }));
    message("#wb-find-msg", `Saved "${name}" at layer ${layer}. It now shows in Watch and can be used for edits.`, "ok");
    await refresh();
    $("#wb-dir").value = name;
  } catch (err) { message("#wb-find-msg", err.message, "error"); }
}

function renderDirections() {
  const dirs = W.status.directions || [];
  const sel = $("#wb-dir"), prev = sel.value;
  sel.innerHTML = dirs.map((d) => `<option value="${esc(d.name)}">${esc(d.name)} · L${d.layer}</option>`).join("") +
    `<option value="random">random direction (control)</option>`;
  if ([...sel.options].some((o) => o.value === prev)) sel.value = prev;
  $("#wb-dirs").innerHTML = dirs.map((d) => `<li><div class="head"><span><span class="name">${esc(d.name)}</span>
      <span class="muted">· L${d.layer} · ${esc(d.kind)}</span></span><button class="link danger" data-del="${esc(d.name)}" aria-label="Delete direction ${esc(d.name)}">Delete</button></div>
      <div class="how">${esc(d.construction || "")}</div></li>`).join("") || `<li class="muted">No directions yet.</li>`;
  $("#wb-dirs").querySelectorAll("[data-del]").forEach((b) => {
    b.onclick = async () => {
      if (!confirm(`Delete direction "${b.dataset.del}"? Its files are removed from the bank.`)) return;
      try { await api(`/brainscan/directions/${encodeURIComponent(b.dataset.del)}`, { method: "DELETE" }); await refresh(); }
      catch (err) { message("#wb-find-msg", err.message, "error"); }
    };
  });
}

// ---------------------------------------------------------------- 2. edit weights
function syncForm() {
  const kind = $("#wb-kind").value;
  document.querySelectorAll("#wb-form [data-kind]").forEach((el) => { el.hidden = el.dataset.kind !== kind; });
  const comp = $("#wb-comp").value;
  $("#wb-index").closest("label").hidden = kind !== "scale" || !["head", "neuron"].includes(comp);
}

async function addEdit(ev) {
  ev.preventDefault();
  const kind = $("#wb-kind").value;
  const body = { kind, layers: $("#wb-layers").value.trim() || "all" };
  if (kind === "orthogonalize") {
    body.direction = $("#wb-dir").value;
    body.strength = Number($("#wb-strength").value);
    body.targets = [...document.querySelectorAll("#wb-form .targets input:checked")].map((i) => i.value);
    if (body.direction === "random") body.seed = Math.floor(Math.random() * 1e6);
  } else {
    const comp = $("#wb-comp").value;
    body.targets = [comp];
    body.factor = Number($("#wb-factor").value);
    if (["head", "neuron"].includes(comp)) body.index = Number($("#wb-index").value);
  }
  try {
    await post("/brainscan/workbench/edits", body);
    message("#wb-edit-msg", "Added. The weights change when you apply the stack.");
    await refresh();
  } catch (err) { message("#wb-edit-msg", err.message, "error"); }
}

function renderEdits() {
  const wb = W.status.workbench || { edits: [], applied: [] };
  const applied = new Set(wb.applied.map((e) => e.id));
  $("#wb-list").innerHTML = wb.edits.map((e) => `<li class="${e.enabled ? "" : "off"}" data-id="${e.id}">
      <input type="checkbox" ${e.enabled ? "checked" : ""} aria-label="Include in stack">
      <span class="desc">${esc(e.summary)}</span>
      ${applied.has(e.id) ? `<span class="badge good">applied</span>` : ""}
      <button aria-label="Remove edit" ${applied.has(e.id) ? "disabled title=\"Restore the original weights first\"" : ""}>✕</button></li>`).join("") ||
    `<li class="muted" style="border:none;background:none;padding:0">No edits yet.</li>`;
  $("#wb-list").querySelectorAll("li[data-id]").forEach((li) => {
    li.querySelector("input").onchange = async (ev) => {
      try { await api(`/brainscan/workbench/edits/${li.dataset.id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ enabled: ev.target.checked }) }); }
      catch (err) { message("#wb-edit-msg", err.message, "error"); }
      await refresh();
    };
    li.querySelector("button").onclick = async () => {
      try { await api(`/brainscan/workbench/edits/${li.dataset.id}`, { method: "DELETE" }); } catch (err) { message("#wb-edit-msg", err.message, "error"); }
      await refresh();
    };
  });
  const banner = $("#wb-banner");
  banner.hidden = !wb.applied.length;
  banner.textContent = wb.applied.length ? `Weights are edited (${wb.applied.length} edit${wb.applied.length > 1 ? "s" : ""}). Every recording made now is marked with these edits.` : "";
  const pending = wb.edits.filter((e) => e.enabled).map((e) => e.id).join() !== wb.applied.map((e) => e.id).join();
  $("#wb-state").textContent = wb.applied.length ? `${wb.applied.length} applied${pending ? " · stack changed, apply to update" : ""}` :
    (wb.edits.length ? "original weights · stack not applied" : "original weights");
}

async function applyStack(ev) {
  try {
    const res = await busy(ev.target, "Applying…", () => post("/brainscan/workbench/apply"));
    message("#wb-edit-msg", `Applied ${res.applied.length} edit${res.applied.length === 1 ? "" : "s"} in ${res.seconds} s. ` +
      `Originals of touched weights are kept in memory (${(res.snapshot_bytes / 1e9).toFixed(2)} GB) for exact restore.`, "ok");
  } catch (err) { message("#wb-edit-msg", err.message, "error"); }
  await refresh();
}

async function revert(ev) {
  try { await busy(ev.target, "Restoring…", () => post("/brainscan/workbench/revert")); message("#wb-edit-msg", "Original weights restored.", "ok"); }
  catch (err) { message("#wb-edit-msg", err.message, "error"); }
  await refresh();
}

// ---------------------------------------------------------------- 3. compare
async function runCompare(ev) {
  const prompts = lines("#wb-prompts");
  if (!prompts.length) { message("#wb-compare-msg", "Add at least one prompt.", "error"); return; }
  message("#wb-compare-msg", `Generating ${prompts.length * 2} answers and scoring reference text…`);
  try {
    const res = await busy(ev.target, "Running…", () => post("/brainscan/workbench/compare", { prompts, max_tokens: Number($("#wb-max").value) || 96 }));
    const m = res.metrics;
    $("#wb-metrics").innerHTML = [
      tile("Refusal-phrase rate", `${pct(m.refusal_rate_original)} → ${pct(m.refusal_rate_edited)}`, "substring match on the first 200 characters"),
      tile("KL on neutral text", m.kl_per_token.toFixed(4), `nats per token, original ‖ edited, ${m.reference_tokens} tokens`),
      tile("Perplexity on neutral text", `${m.perplexity_original.toFixed(2)} → ${m.perplexity_edited.toFixed(2)}`, "original → edited"),
    ].join("");
    $("#wb-results").innerHTML = `<table class="cmp"><thead><tr><th>Prompt</th><th>Original</th><th>Edited</th></tr></thead><tbody>${res.rows.map((r) =>
      `<tr><td>${esc(r.prompt)}</td><td>${flag(r.refusal_original)}<pre>${esc(r.original)}</pre></td><td>${flag(r.refusal_edited)}<pre>${esc(r.edited)}</pre></td></tr>`).join("")}</tbody></table>`;
    message("#wb-compare-msg", `${res.notes} Took ${m.seconds} s.`);
  } catch (err) { message("#wb-compare-msg", err.message, "error"); }
}
const tile = (label, value, sub) => `<div class="tile"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div><div class="sub">${esc(sub)}</div></div>`;
const flag = (on) => (on ? `<span class="badge serious">refusal phrase</span>` : "");

// ---------------------------------------------------------------- 4. export
async function exportModel(ev) {
  const path = $("#wb-path").value.trim();
  if (!path) { message("#wb-export-msg", "Choose a folder.", "error"); return; }
  message("#wb-export-msg", "Writing the checkpoint…");
  try {
    const res = await busy(ev.target, "Exporting…", () => post("/brainscan/workbench/export", { path }));
    message("#wb-export-msg", `Saved ${(res.bytes / 1e9).toFixed(2)} GB with ${res.edits} edit${res.edits === 1 ? "" : "s"} recorded to ${res.path}`, "ok");
  } catch (err) { message("#wb-export-msg", err.message, "error"); }
}

// ---------------------------------------------------------------- the action bar (ui/actionbar.js)
function setupBar() {
  const wb = () => W.status?.workbench || { edits: [], applied: [] };
  const pending = () => wb().edits.filter((e) => e.enabled).map((e) => e.id).join() !== wb().applied.map((e) => e.id).join();
  const to = (sel, focus) => () => { $(sel)?.scrollIntoView({ behavior: "smooth", block: "start" }); if (focus) setTimeout(() => $(focus)?.focus(), 350); };
  const pressWhenReady = (sel) => { let n = 0; const go = () => { const b = $(sel); if (b && !b.disabled) b.click(); else if (++n < 20) setTimeout(go, 100); }; setTimeout(go, 100); };
  defineBar("workbench", {
    groups: ["Find", "Change", "Verify", "Ship"],
    context: () => {
      const w = wb(), n = w.edits.length, a = w.applied.length;
      if (!W.status) return "Loading the edit stack…";
      return n ? `Your model: <b>${n} edit${n === 1 ? "" : "s"}</b> in the stack, ${a ? `<b>${a} applied</b>` : "original weights"}${a && pending() ? " · the stack changed, apply to update" : ""}`
        : "Your model: original weights, no edits yet";
    },
    actions: [
      { id: "find", group: "Find", label: "Find a direction", hint: "Two sets of examples give a direction in the residual stream, scored at every layer",
        run: to("#h-find", "#wb-a") },
      { id: "edit", group: "Change", label: "Edit weights", hint: "Scale or lesion an attention block, MLP block, head or neuron, or add another edit",
        run: to("#h-edit", "#wb-kind") },
      { id: "forget", group: "Change", label: "Forget a fact", hint: "Guided: name a fact, find the units that carry it, try, measure, keep",
        run: () => document.dispatchEvent(new CustomEvent("brainscan:action", { detail: { action: "forget" } })) },
      { id: "apply", group: "Change", label: "Apply stack", hint: "Apply every checked edit to the weights",
        need: () => (!wb().edits.some((e) => e.enabled) ? "an edit in the stack" : !pending() ? "a change to the stack: it's already applied" : null),
        run: () => $("#wb-apply").click() },
      { id: "restore", group: "Change", label: "Restore original", hint: "Undo every applied edit, exactly",
        need: () => (wb().applied.length ? null : "an applied edit"), run: () => $("#wb-revert").click() },
      { id: "rerun", group: "Verify", label: "Re-run the chat", hint: "Ask the open chat's question again on the original and the edited weights",
        need: () => (wb().applied.length ? null : "an applied edit"), run: () => { to("#h-compare")(); pressWhenReady(".wb-replay [data-replay]"); } },
      { id: "compare", group: "Verify", label: "Compare prompts", hint: "Your own prompts on both weights, with drift on neutral text",
        need: () => (wb().applied.length ? null : "an applied edit"), run: to("#h-compare", "#wb-prompts") },
      { id: "checks", group: "Verify", label: "Run checks", hint: "Every saved behavior check, original → now",
        run: () => { to("#h-checks")(); pressWhenReady(".wb-checks [data-run-checks]"); } },
      { id: "export", group: "Ship", label: "Export", hint: "Save the edited model as a Hugging Face checkpoint, with every edit recorded",
        need: () => (wb().applied.length ? null : "an applied edit"), run: to("#h-export", "#wb-path") },
    ],
  });
}

// ---------------------------------------------------------------- boot
function boot() {
  setupBar();
  document.addEventListener("brainscan:show", (ev) => { if (VIEWS.includes(ev.detail)) show(ev.detail); });  // from the guide
  // The stack changed elsewhere (a knockout made permanent in Trace, the Edits panel).
  document.addEventListener("brainscan:workbench", () => { if (!$("#view-workbench").hidden) refresh(); });
  // The edit panels report the stack from any tab; keep this tab's bar (and Do…) in step with it.
  document.addEventListener("brainscan:edit-state", (ev) => {
    const w = W.status?.workbench;
    if (!w || w.applied.length !== ev.detail.applied || w.edits.length !== ev.detail.edits) refresh();
  });
  document.addEventListener("brainscan:model", () => {  // layer sweeps, comparisons and export names belong to one model
    W.sweep = null;
    $("#wb-sweep").innerHTML = ""; $("#wb-save-row").hidden = true; $("#wb-path").value = "";
    $("#wb-metrics").innerHTML = ""; $("#wb-results").innerHTML = "";
    for (const id of ["#wb-find-msg", "#wb-edit-msg", "#wb-compare-msg", "#wb-export-msg"]) message(id, "");
    refresh();
  });
  $("#wb-example").onclick = () => { $("#wb-a").value = EXAMPLE.a.join("\n"); $("#wb-b").value = EXAMPLE.b.join("\n"); $("#wb-name").value = "language_fr"; };
  $("#wb-analyze").onclick = analyze;
  $("#wb-save").onclick = saveDirection;
  $("#wb-kind").onchange = syncForm;
  $("#wb-comp").onchange = syncForm;
  $("#wb-form").addEventListener("submit", addEdit);
  $("#wb-apply").onclick = applyStack;
  $("#wb-revert").onclick = revert;
  $("#wb-run").onclick = runCompare;
  $("#wb-export").onclick = exportModel;
  syncForm();
  // The forget flow is a step inside a tab's work, never a place to land on reload.
  const saved = (() => { try { return localStorage.getItem("brainscan-view"); } catch { return null; } })();
  show(VIEWS.includes(saved) && saved !== "actions" ? saved : "scanner");
}
boot();
