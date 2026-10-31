// Vet: check a model you didn't train before trusting it (llminspect/check.py). The hero is every model on this
// machine at once, one cell each, sized by how much disk it takes and lit by its latest check: critical for findings
// to look at first, warning for findings worth a look, quiet for a check that found nothing, an outline for none
// yet. Pick a cell, check it, and the report opens below, drawn by the same renderer as the page agents link to
// (check_html), in this app's theme. The loaded model is the base by default; the weight checks need it.
//
// API: GET /llminspect/check/models; POST /llminspect/check {candidate, base, deep} starts a job;
//   GET /llminspect/check/jobs/{job}; GET /llminspect/check/reports/{id}/html?theme=&embed=true.
import { defineBar, noteBar, refreshBar } from "/ui/actionbar.js";

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

const V = { models: [], base: null, sel: null, job: null, report: null, timer: 0, ticks: 0, loaded: false };

/** A model's state on the shelf, from its latest check. */
function tone(m) {
  const c = m.last?.counts;
  if (!c) return "none";
  return c.high ? "high" : c.medium ? "medium" : "clear";
}
const RANK = { high: 0, medium: 1, clear: 2, none: 3 };
const gbText = (gb) => (gb >= 100 ? `${Math.round(gb)} GB` : gb >= 1 ? `${gb.toFixed(1)} GB` : `${Math.round(gb * 1000)} MB`);

function theme() {
  const set = document.documentElement.dataset.theme;
  return set === "dark" || set === "light" ? set : matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

// ---------------------------------------------------------------- the shelf

function renderShelf() {
  const el = $("#vt-cells");
  if (!el) return;
  if (!V.loaded) { el.innerHTML = '<p class="vt-calm">Reading the model folders…</p>'; return; }
  if (!V.models.length) {
    el.innerHTML = '<p class="vt-calm">No models found. Add folders with LLMINSPECT_MODEL_DIRS.</p>';
    return;
  }
  const max = Math.max(...V.models.map((m) => m.gb), 1);
  const sorted = [...V.models].sort((a, b) => RANK[tone(a)] - RANK[tone(b)] || b.gb - a.gb);
  el.innerHTML = sorted.map((m) => {
    const w = 14 + 120 * Math.sqrt(m.gb / max);  // area-ish: a 60 GB model reads bigger than a 1 GB one, not 60x
    const t = tone(m), on = V.sel === m.path, isBase = V.base?.path === m.path;
    const label = `${m.name}, ${gbText(m.gb)}${isBase ? ", the loaded model" : ""}: ${m.last ? m.last.headline : "not checked yet"}`;
    return `<button type="button" class="vt-cell ${t}${on ? " on" : ""}${isBase ? " base" : ""}" style="--w:${w.toFixed(1)}px"
      data-path="${esc(m.path)}" role="option" aria-selected="${on}" aria-label="${esc(label)}" title="${esc(label)}"></button>`;
  }).join("");
  const checked = V.models.filter((m) => m.last).length, flagged = V.models.filter((m) => tone(m) === "high").length;
  const tb = V.models.reduce((s, m) => s + m.gb, 0) / 1000;
  $("#vt-sub").innerHTML = `${V.models.length} models · ${tb >= 1 ? `${tb.toFixed(2)} TB` : `${Math.round(tb * 1000)} GB`} · `
    + `${checked} checked${flagged ? ` · <b class="hi">${flagged} to look at first</b>` : ""}`;
  renderPick();
}

/** The line under the shelf: the model under the mouse, else the one picked. */
function readout(m) {
  const read = $("#vt-read");
  if (!m) {
    read.innerHTML = '<span class="vt-calm">Pick a model on the shelf, or paste a model folder below.</span>';
    return;
  }
  read.innerHTML = `<b>${esc(m.name)}</b><span>${gbText(m.gb)} · ${esc(m.formats.join(" ") || "no weights")}`
    + `${V.base?.path === m.path ? " · the loaded model" : ""}</span>`
    + (m.last ? `<span class="vt-last ${tone(m)}">${esc(m.last.headline)}</span>` : '<span class="vt-calm">Not checked yet</span>');
}

function renderPick() {
  const m = V.models.find((x) => x.path === V.sel);
  readout(m);
  if (m) $("#vt-path").value = m.path;
  const deep = $("#vt-deep");
  const baseOk = !!V.base && $("#vt-base").value === "loaded";
  deep.disabled = !baseOk;
  if (!baseOk) deep.checked = false;
  $("#vt-deep-why").textContent = !V.base ? "load a model first: it is the base the weights are compared with"
    : $("#vt-base").value !== "loaded" ? "the weights are compared with the loaded model: pick it as the base" : "";
  $("#vt-base-loaded").textContent = V.base ? `the loaded model, ${V.base.name}` : "the loaded model (none loaded)";
  $("#vt-base-loaded").disabled = !V.base;
  refreshBar("vet");
}

function pick(path) {
  V.sel = path;
  remember(KEEP, path);
  renderShelf();
  const m = V.models.find((x) => x.path === path);
  if (m?.last) showReport(m.last.id);
}

// ---------------------------------------------------------------- checking

async function start(deep = $("#vt-deep").checked) {
  const candidate = $("#vt-path").value.trim();
  if (!candidate) { message("Pick a model or paste its folder first.", "error"); return; }
  if (V.job) return;
  message("");
  try {
    const job = await post("/llminspect/check", { candidate, base: $("#vt-base").value, deep });
    V.job = job.job; remember(KEEP_JOB, V.job);
    progress(job.step);
    poll();
  } catch (err) { message(err.message, "error"); }
  refreshBar("vet");
}

async function startAll() {
  if (V.job) return;
  message("");
  try {
    const job = await post("/llminspect/check/all", {});
    V.job = job.job; V.ticks = 0; remember(KEEP_JOB, V.job);
    progress(job.step);
    poll();
  } catch (err) { message(err.message, "error"); }
  refreshBar("vet");
}

async function poll() {
  clearTimeout(V.timer);
  if (!V.job) return;
  let job;
  try { job = await api(`/llminspect/check/jobs/${V.job}`); } catch (err) {  // gone (the server restarted): forget it
    V.job = null; remember(KEEP_JOB, null); progress(null); refreshBar("vet"); return;
  }
  const all = Array.isArray(job.done);
  if (job.status === "running") {
    progress(job.step);
    if (all && ++V.ticks % 4 === 0) loadModels();  // light the shelf as reports land
    V.timer = setTimeout(poll, 700);
    return;
  }
  V.job = null; remember(KEEP_JOB, null);
  progress(null);
  if (job.status === "error") { message(job.error, "error"); refreshBar("vet"); return; }
  await loadModels();
  if (all) {
    const high = job.done.filter((d) => d.counts.high).length;
    message(`Checked ${job.done.length} models${high ? `: ${high} to look at first` : ""}${job.failed.length ? `; ${job.failed.length} couldn't be read` : ""}.`, "ok");
    return;
  }
  noteBar("vet", "Check done");
  showReport(job.report);
}

function progress(step) {
  const el = $("#vt-progress");
  el.hidden = !step;
  el.innerHTML = step ? `<span class="vt-pulse" aria-hidden="true"></span>${esc(step)}` : "";
}

function message(text, toneName = "") {
  const el = $("#vt-msg");
  el.className = `wb-msg ${toneName}`;
  el.textContent = text || "";
}

function showReport(id) {
  V.report = id;
  const frame = $("#vt-frame");
  frame.hidden = false;
  frame.src = `/llminspect/check/reports/${encodeURIComponent(id)}/html?theme=${theme()}&embed=true`;
  $("#vt-open").href = `/llminspect/check/reports/${encodeURIComponent(id)}/html`;
  $("#vt-open").hidden = false;
  refreshBar("vet");
}

function fitFrame() {  // the report is one long page: let it set its own height, no inner scrollbar
  const frame = $("#vt-frame");
  try { frame.style.height = `${frame.contentDocument.documentElement.scrollHeight + 8}px`; } catch { /* not loaded */ }
}

async function loadModels() {
  try {
    const d = await api("/llminspect/check/models");
    V.models = d.models; V.base = d.base; V.loaded = true;
    $("#vt-models").innerHTML = V.models.map((m) => `<option value="${esc(m.path)}">${esc(m.name)}</option>`).join("");
    const picked = V.models.find((m) => m.path === V.sel);
    if (picked?.last && !V.report) showReport(picked.last.id);  // the model picked last time, with its report
  } catch (err) { V.loaded = true; message(err.message, "error"); }
  renderShelf();
}

// ---------------------------------------------------------------- boot

function enter() {
  if (!V.loaded) loadModels();
}

function boot() {
  if (!$("#view-vet")) return;
  try { V.sel = localStorage.getItem(KEEP); V.job = localStorage.getItem(KEEP_JOB); } catch { /* storage unavailable */ }
  if (V.job) poll();  // a check still running from before a reload
  $("#vt-cells").addEventListener("click", (ev) => { const b = ev.target.closest(".vt-cell"); if (b) pick(b.dataset.path); });
  const hover = (ev) => { const b = ev.target.closest(".vt-cell"); readout(V.models.find((x) => x.path === (b ? b.dataset.path : V.sel))); };
  $("#vt-cells").addEventListener("mouseover", hover);
  $("#vt-cells").addEventListener("focusin", hover);
  $("#vt-cells").addEventListener("mouseleave", () => readout(V.models.find((x) => x.path === V.sel)));
  $("#vt-cells").addEventListener("keydown", (ev) => {  // arrows walk the shelf
    if (!["ArrowRight", "ArrowLeft"].includes(ev.key)) return;
    const cells = [...$("#vt-cells").querySelectorAll(".vt-cell")];
    const i = cells.indexOf(document.activeElement);
    const next = cells[Math.max(0, Math.min(cells.length - 1, i + (ev.key === "ArrowRight" ? 1 : -1)))];
    if (next) { next.focus(); ev.preventDefault(); }
  });
  $("#vt-form").addEventListener("submit", (ev) => { ev.preventDefault(); start(); });
  $("#vt-base").addEventListener("change", renderPick);
  $("#vt-path").addEventListener("change", () => { const m = V.models.find((x) => x.path === $("#vt-path").value.trim()); V.sel = m ? m.path : null; renderShelf(); });
  $("#vt-frame").addEventListener("load", fitFrame);
  // The report follows the app's theme: the toggle (data-theme) or, with none set, the system's.
  const retheme = () => { if (V.report) showReport(V.report); };
  new MutationObserver(retheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", retheme);
  document.addEventListener("llminspect:view", (ev) => { if (ev.detail === "vet") enter(); });
  document.addEventListener("llminspect:model", () => { V.loaded = false; if (!$("#view-vet").hidden) loadModels(); });

  defineBar("vet", {
    groups: ["Check", "Read"],
    context: () => (V.job ? "Checking…" : V.sel ? `<b>${esc(V.models.find((m) => m.path === V.sel)?.name || V.sel)}</b>` : "No model picked"),
    state: () => ({ vet_picked: !!V.sel, vet_checked: !!V.report }),
    actions: [
      { id: "check", group: "Check", label: "Check", hint: "Code that runs on load, where it comes from, what changed in its files",
        need: () => (V.job ? "the check running now to finish" : V.sel || $("#vt-path").value.trim() ? null : "a model picked on the shelf"),
        run: () => start(false) },
      { id: "all", group: "Check", label: "Check every model", hint: "Check every model on the shelf, files only, and light the shelf as it goes",
        need: () => (V.job ? "the check running now to finish" : V.models.length ? null : "models on the shelf"), run: startAll },
      { id: "deep", group: "Check", label: "Check with weights", hint: "Also what changed inside the weights, against the loaded model: triggers, refusals, behavior",
        need: () => (V.job ? "the check running now to finish" : !V.base ? "a model loaded, as the base"
          : !(V.sel || $("#vt-path").value.trim()) ? "a model picked on the shelf" : null),
        run: () => { $("#vt-base").value = "loaded"; start(true); } },
      { id: "open", group: "Read", label: "Open the page", hint: "The report as a page of its own, to save or share",
        need: () => (V.report ? null : "a check done"), run: () => window.open($("#vt-open").href, "_blank") },
      { id: "refresh", group: "Read", label: "Refresh shelf", hint: "Read the model folders again", run: () => { V.loaded = false; renderShelf(); loadModels(); } },
    ],
  });
  if (!$("#view-vet").hidden) enter();
}
boot();
