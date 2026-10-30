// The deploy scan (Tune): check a candidate checkpoint against the loaded model before deploying it, in
// one report (llminspect/recorder/scan.py). The form takes the candidate, optional canary secrets, an
// optional manifest and the sections to run; the scan runs in the background with its step shown here;
// the result leads with what stands out, then one line per section, and links the full printable report.
// Past reports for this model are listed under it. A report is evidence with its limits, never a verdict.
//
// API: POST /llminspect/scan {spec, canaries, manifest, sections} starts it; GET /llminspect/scan is the job
//   { status: none|running|ready|error, step, progress, error, report: { id, signals } };
//   GET /llminspect/scan/reports lists saved ones; GET .../reports/{id} and .../{id}/html read one.
import { extendBar, noteBar, refreshBar } from "/ui/actionbar.js";

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
async function api(path, opts) {
  const res = await fetch(path, opts);
  const text = await res.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch { /* not JSON */ }
  if (!res.ok) { const err = new Error(body?.detail || text || `${res.status}`); err.status = res.status; throw err; }
  return body;
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}) });
const JOB = "/llminspect/scan";
const KEEP = "llminspect-scan-spec";

const SECTIONS = [
  { id: "integrity", label: "Integrity", hint: "File hashes against a manifest, and code files shipped with the weights" },
  { id: "weights", label: "Weights", hint: "Where the candidate's weights differ from the base" },
  { id: "behavior", label: "Behavior", hint: "How far its predictions drift on neutral prompts" },
  { id: "refusals", label: "Refusals", hint: "Whether it still refuses the harmful requests the base refuses" },
  { id: "triage", label: "Triage", hint: "Which neurons the difference runs through, and single tokens that act like triggers" },
  { id: "memorization", label: "Memorization", hint: "Whether it ranks or completes the canary secrets you give" },
];
const LEVELS = { high: "Look first", medium: "Worth a look", note: "Note" };

const S = { job: null, report: null, reports: [], timer: 0, busy: false, fail: 0, shown: null, supported: true };

function message(text, tone = "") {
  const el = $("#sc-msg");
  if (el) { el.className = `wb-msg ${tone}`; el.textContent = text || ""; }
}

// ---------------------------------------------------------------- the job
async function fetchJob() {
  try {
    S.job = await api(JOB);
    S.supported = true; S.fail = 0;
  } catch (err) {
    if (err.status === 404) { S.supported = false; renderAll(); return; }
    if (++S.fail > 3) { message(err.message, "error"); return; }
  }
  const j = S.job;
  if (j?.status === "running") { clearTimeout(S.timer); S.timer = setTimeout(fetchJob, 1200); }
  if (j?.status === "ready" && j.report?.id && j.report.id !== S.shown) { await openReport(j.report.id); loadReports(); }
  renderProgress(); refreshBar("workbench");
}

async function start(ev) {
  ev?.preventDefault();
  const spec = $("#sc-spec").value.trim();
  if (!spec) { message("Name the candidate to scan.", "error"); $("#sc-spec").focus(); return; }
  const sections = SECTIONS.map((x) => x.id).filter((id) => $(`#sc-sections input[value="${id}"]`)?.checked);
  if (!sections.length) { message("Pick at least one section.", "error"); return; }
  const canaries = $("#sc-canaries").value.split("\n").map((s) => s.trim()).filter(Boolean);
  const manifest = $("#sc-manifest").value.trim() || undefined;
  S.busy = true; renderProgress();
  try {
    S.job = await post(JOB, { spec, canaries, manifest, sections });
    try { localStorage.setItem(KEEP, spec); } catch { /* storage unavailable */ }
    S.report = null; S.shown = null; renderReport();
    message("");
    noteBar("workbench", `Scanning ${spec}`);
  } catch (err) { message(err.message, "error"); }
  S.busy = false;
  fetchJob();
}

// ---------------------------------------------------------------- reports
async function loadReports() {
  try { S.reports = await api(`${JOB}/reports`); } catch { S.reports = []; }
  renderHistory();
}

async function openReport(id) {
  try {
    S.report = await api(`${JOB}/reports/${encodeURIComponent(id)}`);
    S.shown = id;
  } catch (err) { message(err.message, "error"); }
  renderReport(); renderHistory(); refreshBar("workbench");
}

// ---------------------------------------------------------------- drawing
function renderProgress() {
  const box = $("#sc-progress"), btn = $("#sc-start");
  if (!box) return;
  const j = S.job, running = j?.status === "running";
  btn.disabled = S.busy || running || !S.supported;
  btn.textContent = running ? "Scanning…" : "Start scan";
  if (!S.supported) { box.hidden = false; box.innerHTML = `<p class="muted small">This recorder has no deploy scan: restart it on the current version.</p>`; return; }
  if (running) {
    const p = Math.round((j.progress || 0) * 100);
    box.hidden = false;
    box.innerHTML = `<div class="sc-bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${p}"><i style="width:${p}%"></i></div>
      <span class="small">${esc(j.step || "working")} · ${p}%</span>
      <span class="muted small">The model is busy until the scan finishes.</span>`;
  } else if (j?.status === "error") {
    box.hidden = false;
    box.innerHTML = `<p class="small warn-text">The scan failed: ${esc(j.error)}</p>`;
  } else { box.hidden = true; box.innerHTML = ""; }
}

const n = (x, d = 3) => (x == null || !Number.isFinite(Number(x)) ? "–" : x > 0 && x < 10 ** -d ? Number(x).toExponential(1) : Number(x).toFixed(d));
const count = (k, word) => `${k} ${word}${k === 1 ? "" : "s"}`;

/** One line per section that ran: its headline number and what it measured. */
function sectionRows(r) {
  const rows = [];
  const integ = r.integrity;
  if (integ) {
    const m = integ.manifest;
    rows.push(["integrity", m ? (m.mismatch?.length ? `${m.mismatch.length} mismatched` : "matches the manifest") : "no manifest given",
      `${count(integ.files?.length || 0, "file")} hashed${(integ.files || []).some((f) => f.kind === "code") ? ", code files present" : ""}`]);
  }
  const w = r.weights;
  if (r.identical) rows.push(["weights", "identical", "the candidate's weights are the base's"]);
  else if (w) rows.push(["weights", w.summary?.pattern || "–", `${count(w.counts?.changed ?? 0, "tensor")} changed of ${(w.counts?.changed ?? 0) + (w.counts?.identical ?? 0)}`]);
  const b = r.behavior;
  if (b) rows.push(["behavior", `KL ${n(b.metrics?.kl_per_token)}`, `per token on neutral prompts; perplexity ${n(b.metrics?.perplexity_original, 2)} → ${n(b.metrics?.perplexity_edited, 2)}`]);
  const f = r.refusals;
  if (f) rows.push(["refusals", f.base_refused ? `${f.still_refused} of ${f.base_refused} kept` : "base refuses none", `the base refuses ${f.base_refused} of ${f.probes} harmful requests; ${f.measure || ""}`]);
  const t = r.triage;
  if (t) {
    const conf = (t.neurons || []).filter((x) => x.confirmed).length, groups = (t.groups || []).filter((g) => g.confirmed).length;
    const z = Math.max(0, ...((t.triggers?.tokens) || []).map((x) => Math.max(x.z, x.z_after ?? x.z)));
    rows.push(["triage", `${count(conf, "neuron")} · ${count(groups, "group")} confirmed`, `drift ${n(t.drift)} on neutral text${t.triggers ? `; top token z ${n(z, 1)}` : ""}`]);
  }
  const m = r.memorization;
  if (m) {
    const cs = m.canaries || [];
    const hit = cs.filter((c) => (c.candidate?.rank === 1 && c.base?.rank > 1) || (c.candidate?.completes && !c.base?.completes)).length;
    rows.push(["memorization", cs.length ? `${hit} of ${cs.length} flagged` : "no canaries", cs.length ? "rank among strings of the same shape, and greedy completion" : m.measure || ""]);
  }
  return rows;
}

function renderReport() {
  const box = $("#sc-report");
  if (!box) return;
  const r = S.report;
  if (!r) { box.innerHTML = ""; box.hidden = true; return; }
  const sig = r.signals || [];
  const by = (lvl) => sig.filter((s) => s.level === lvl).length;
  const order = { high: 0, medium: 1, note: 2 };
  const sorted = [...sig].sort((a, b) => (order[a.level] ?? 3) - (order[b.level] ?? 3));
  const html = `${JOB}/reports/${encodeURIComponent(r.id)}/html`;
  box.hidden = false;
  box.innerHTML = `
    <header class="sc-head">
      <div><span class="eyebrow">Scan report · ${esc(new Date(r.created).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }))}${r.seconds != null ? ` · ${r.seconds}s` : ""}</span>
        <h3>${esc(r.candidate?.name)} <span class="dim">against ${esc(r.base?.name)}</span></h3></div>
      <div class="sc-counts">
        <span class="sc-count high"><b>${by("high")}</b> look first</span>
        <span class="sc-count medium"><b>${by("medium")}</b> worth a look</span>
        <span class="sc-count note"><b>${by("note")}</b> notes</span>
      </div>
    </header>
    ${sorted.length ? `<ul class="sc-signals">${sorted.map((s) => `<li class="${esc(s.level)}"><span class="lvl">${LEVELS[s.level] || esc(s.level)}</span><span>${esc(s.text)}</span><span class="sec">${esc(s.section)}</span></li>`).join("")}</ul>`
      : `<p class="sc-quiet">Nothing in the sections run crossed the report's thresholds. That is not evidence of safety: see the limits in the full report.</p>`}
    <table class="sc-sections"><tbody>${sectionRows(r).map(([sec, head, sub]) => `<tr><th>${esc(sec)}</th><td class="v">${esc(head)}</td><td class="muted small">${esc(sub)}</td></tr>`).join("")}</tbody></table>
    <div class="sc-links">
      <a class="btn sm" href="${html}" target="_blank" rel="noopener">Open the full report</a>
      <a class="link" href="${JOB}/reports/${encodeURIComponent(r.id)}" target="_blank" rel="noopener">JSON</a>
      <span class="muted small">Evidence, not a verdict. The full report lists what this scan cannot see.</span>
    </div>`;
}

function renderHistory() {
  const box = $("#sc-history");
  if (!box) return;
  const rows = S.reports.slice(0, 8);
  box.hidden = !rows.length;
  box.innerHTML = rows.length ? `<span class="muted small">Earlier scans</span>${rows.map((x) => `<button type="button" class="nv-link${x.id === S.shown ? " on" : ""}" data-report="${esc(x.id)}" title="${esc(x.created)}">${esc(x.candidate)} <span>${count(x.signals, "signal")}</span></button>`).join("")}` : "";
}

function renderAll() { renderProgress(); renderReport(); renderHistory(); }

// ---------------------------------------------------------------- wiring
function boot() {
  if (!$("#sc-panel")) return;
  $("#sc-sections").innerHTML = SECTIONS.map((x) => `<label title="${esc(x.hint)}"><input type="checkbox" value="${x.id}" checked> ${x.label}</label>`).join("");
  extendBar("workbench", {
    groups: [],
    actions: [{ id: "scan", group: "Verify", label: "Scan a checkpoint",
      hint: "Check a candidate checkpoint against this model before deploying it: integrity, weights, behavior, refusals, triage, memorization",
      need: () => (!S.supported ? "a recorder with the deploy scan: restart it" : S.job?.status === "running" ? "the scan running now to finish" : null),
      run: () => { $("#h-scan")?.scrollIntoView({ behavior: "smooth", block: "start" }); setTimeout(() => $("#sc-spec")?.focus({ preventScroll: true }), 350); } }],
  });
  const panel = $("#sc-panel");
  panel.addEventListener("keydown", (ev) => ev.stopPropagation());  // typing a path never drives the graph
  $("#sc-form").addEventListener("submit", start);
  $("#sc-history").addEventListener("click", (ev) => { const b = ev.target.closest("[data-report]"); if (b) openReport(b.dataset.report); });
  try { $("#sc-spec").value = localStorage.getItem(KEEP) || ""; } catch { /* storage unavailable */ }
  const enter = () => { fetchJob(); loadReports(); };
  document.addEventListener("llminspect:view", (ev) => { if (ev.detail === "workbench") enter(); });
  document.addEventListener("llminspect:model", () => { S.job = null; S.report = null; S.shown = null; S.reports = []; renderAll(); if (!$("#view-workbench").hidden) enter(); });
  renderAll();
  if (!$("#view-workbench").hidden) enter();
}
boot();
