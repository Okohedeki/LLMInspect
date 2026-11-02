// Sweep: check many Hugging Face Hub models at once, without downloading any (llminspect/sweep.py).
// The hero is a field of tiles, one per model in the order they were checked, lit by the most serious risk each
// carries. Most tiles stay faint; what needs a look glows. Counts per risk sit above, the flagged models below,
// and any tile opens its model in Unmask.
//
// Colors mean one thing each, as on the Unmask card: red, code that runs (on load, or if trusted, or a template
// escape); amber, what was edited (targeted weight edits, template additions); violet, a false parent claim.
//
// API: POST /llminspect/sweep, GET /llminspect/sweep/{id}, GET /llminspect/sweeps
import { defineBar, refreshBar, noteBar } from "/ui/actionbar.js";

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const RISK = {
  "runs code on load": { tone: "red", short: "code on load" },
  "imports other code on load": { tone: "red", short: "imports code" },
  "runs code if trusted": { tone: "red", short: "code if trusted" },
  "template": { tone: "red", short: "template escape" },
  "claims a false parent": { tone: "violet", short: "false parent" },
  "template adds text": { tone: "amber", short: "template adds text" },
  "targeted weight edit": { tone: "amber", short: "weight edit" },
  "format": { tone: "muted", short: "malformed file" },
};
const TONE = { red: "#f05252", amber: "#f5a524", violet: "#8b7dff", muted: "#6b7385", none: "#262b38", error: "#3a2a2a" };
const S = { id: null, data: null, job: null, timer: 0, hover: -1, tiles: [], history: [], t0: 0 };
const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

async function api(path, opts) {
  const r = await fetch(path, opts);
  const text = await r.text();
  const body = text ? JSON.parse(text) : null;
  if (!r.ok) throw new Error(body?.detail || `${r.status}`);
  return body;
}

// ---------------------------------------------------------------- running

async function start(ev) {
  ev?.preventDefault();
  const tags = $("#sw-tag").value.trim();
  const body = { search: $("#sw-search").value.trim() || null, author: $("#sw-author").value.trim() || null,
    tags: tags ? tags.split(/[\s,]+/) : null, sort: $("#sw-sort").value, limit: Number($("#sw-limit").value) || 50,
    weights: $("#sw-weights").checked };
  try {
    const job = await api("/llminspect/sweep", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    S.id = job.id; S.job = job; S.data = null; S.t0 = performance.now();
    poll();
  } catch (err) { noteBar("sweep", `Couldn't start: ${err.message}`); }
  refreshBar("sweep");
}

async function poll() {
  clearTimeout(S.timer);
  if (!S.id) return;
  try {
    const got = await api(`/llminspect/sweep/${encodeURIComponent(S.id)}`);
    S.data = got; S.job = got.job || S.job;
  } catch (err) { S.job = { ...(S.job || {}), status: "error", error: err.message }; }
  render();
  if (S.job?.status === "running") S.timer = setTimeout(poll, 1200);
  else loadHistory();
}

async function open(id) { S.id = id; S.job = null; await poll(); }

async function loadHistory() {
  try { S.history = await api("/llminspect/sweeps"); } catch { S.history = []; }
  $("#sw-history").innerHTML = S.history.slice(0, 8).map((h) => {
    const q = h.query || {};
    const what = q.search || q.author || (q.tags || []).join(" ") || (q.models || []).slice(0, 1).join("") || "top models";
    return `<button type="button" class="um-chip" data-sweep="${esc(h.id)}" title="${esc(h.id)}">${esc(what)} · ${h.models}${h.flagged ? ` · <b>${h.flagged} flagged</b>` : ""}</button>`;
  }).join("");
}

// ---------------------------------------------------------------- drawing

function rows() { return S.data?.rows || []; }
function toneOf(r) { return r.error ? "error" : r.top ? RISK[r.top]?.tone || "muted" : "none"; }

function draw() {
  const cv = $("#sw-field");
  cv.hidden = !S.id;
  if (!S.id) return;
  const box = cv.getBoundingClientRect();
  const dpr = devicePixelRatio || 1;
  if (cv.width !== Math.round(box.width * dpr) || cv.height !== Math.round(box.height * dpr)) {
    cv.width = Math.round(box.width * dpr); cv.height = Math.round(box.height * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, box.width, box.height);
  const list = rows();
  const total = Math.max(list.length, S.job?.total || 0, 1);
  const pad = 4, gap = 4;
  // as large as the width allows while every model fits in at most ~ the height of a screen's half
  let size = 64;
  while (size > 6 && Math.ceil(total / Math.floor((box.width - 2 * pad + gap) / (size + gap))) * (size + gap) > Math.max(240, innerHeight * 0.5)) size -= 2;
  const cols = Math.max(1, Math.floor((box.width - 2 * pad + gap) / (size + gap)));
  const height = 2 * pad + Math.ceil(total / cols) * (size + gap);
  if (Math.abs(box.height - height) > 1) { cv.style.height = `${height}px`; requestAnimationFrame(draw); return; }
  const t = (performance.now() - S.t0) / 1000;
  S.tiles = [];
  for (let i = 0; i < total; i++) {
    const x = pad + (i % cols) * (size + gap), y = pad + Math.floor(i / cols) * (size + gap);
    const r = list[i];
    S.tiles.push({ x, y, size, r });
    if (!r) {  // waiting to be checked: a faint outline that breathes while the sweep runs
      const a = S.job?.status === "running" && !reduced() ? 0.12 + 0.06 * Math.sin(t * 2.4 - i * 0.15) : 0.1;
      ctx.strokeStyle = `rgba(160,170,190,${a})`; ctx.lineWidth = 1;
      ctx.strokeRect(x + 0.5, y + 0.5, size - 1, size - 1);
      continue;
    }
    const tone = toneOf(r), col = TONE[tone];
    const lit = tone === "red" || tone === "amber" || tone === "violet";
    if (lit) { ctx.shadowColor = col; ctx.shadowBlur = Math.min(18, size * 0.8); }
    ctx.fillStyle = col;
    ctx.globalAlpha = lit ? 1 : 0.85;
    ctx.beginPath(); ctx.roundRect(x, y, size, size, Math.min(4, size / 4)); ctx.fill();
    ctx.shadowBlur = 0; ctx.globalAlpha = 1;
    if (i === S.hover) { ctx.strokeStyle = "#f4f5f8"; ctx.lineWidth = 1.5; ctx.strokeRect(x - 1.5, y - 1.5, size + 3, size + 3); }
  }
  if (S.job?.status === "running") requestAnimationFrame(draw);
}

function render() {
  const s = S.data?.summary, job = S.job, list = rows();
  const running = job?.status === "running";
  const flagged = s ? s.flagged.length : list.filter((r) => r.risks?.length).length;
  $("#sw-head").innerHTML = !S.id ? `<h2 class="sw-title">Sweep the Hub</h2><p class="sw-sub">Pick models by search, author or tag. Each one's pickles, configs and templates are read by byte range; nothing is downloaded or run.</p>`
    : `<h2 class="sw-title">${running ? `Checking ${job.done || 0} of ${job.total ?? "…"}` : `${list.length} models checked`}${flagged ? `, <span class="sw-hot">${flagged} flagged</span>` : ""}</h2>
       <p class="sw-sub">${job?.error ? `Stopped: ${esc(job.error)}` : running && job.last ? `Last: ${esc(job.last)}` : s ? `Not checked: ${esc(s.not_checked.join("; "))}.` : ""}</p>`;
  const counts = s?.counts || {};
  $("#sw-counts").innerHTML = Object.entries(RISK).map(([k, v]) => {
    const n = counts[k] ?? list.filter((r) => (r.risks || []).some((x) => x.risk === k)).length;
    return n ? `<div class="sw-count ${v.tone}"><b>${n}</b><span>${esc(v.short)}</span></div>` : "";
  }).join("") + (list.length ? `<div class="sw-count none"><b>${list.filter((r) => !r.risks?.length && !r.error).length}</b><span>no risk found</span></div>` : "");
  const shown = list.filter((r) => r.risks?.length).sort((a, b) => Object.keys(RISK).indexOf(a.top) - Object.keys(RISK).indexOf(b.top));
  $("#sw-flagged").innerHTML = shown.length ? `<h3>Flagged</h3><ol class="sw-list">${shown.map((r) => `
    <li><button type="button" class="sw-row" data-model="${esc(r.model)}" title="Open in Unmask">
      <span class="sw-dot ${RISK[r.top]?.tone || "muted"}" aria-hidden="true"></span>
      <span class="sw-model">${esc(r.model)}</span>
      <span class="sw-risk">${esc(RISK[r.top]?.short || r.top)}</span>
      <span class="sw-what">${esc((r.risks[0] || {}).what || "")}</span></button></li>`).join("")}</ol>` : list.length && !running ? `<p class="sw-sub">No model here carries a risk these checks look for. That isn't a clean bill of health: behavior wasn't run.</p>` : "";
  draw();
  refreshBar("sweep");
}

function tileAt(ev) {
  const box = $("#sw-field").getBoundingClientRect();
  const x = ev.clientX - box.left, y = ev.clientY - box.top;
  return S.tiles.findIndex((t) => t.r && x >= t.x && x <= t.x + t.size && y >= t.y && y <= t.y + t.size);
}

function hover(ev) {
  const i = tileAt(ev);
  const tip = $("#sw-tip");
  if (i !== S.hover) { S.hover = i; draw(); }
  if (i < 0) { tip.hidden = true; return; }
  const r = S.tiles[i].r;
  tip.innerHTML = `<b>${esc(r.model)}</b><br>${r.error ? esc(r.error) : r.risks?.length
    ? r.risks.slice(0, 3).map((x) => `<span class="sw-dot ${RISK[x.risk]?.tone || "muted"}"></span>${esc(RISK[x.risk]?.short || x.risk)}: ${esc(x.what)}`).join("<br>")
    : "no risk found by these checks"}${r.headline ? `<br><span class="dim">${esc(r.headline)}</span>` : ""}`;
  const box = $("#sw-stage").getBoundingClientRect();
  tip.style.left = `${Math.min(ev.clientX - box.left + 14, box.width - 340)}px`;
  tip.style.top = `${ev.clientY - box.top + 14}px`;
  tip.hidden = false;
}

function unmask(model) {
  const input = $("#um-input");
  if (!input) return;
  input.value = model;
  document.dispatchEvent(new CustomEvent("llminspect:show", { detail: "unmask" }));
  setTimeout(() => $("#um-form").requestSubmit(), 60);
}

// ---------------------------------------------------------------- boot

function boot() {
  $("#sw-form").addEventListener("submit", start);
  $("#sw-field").addEventListener("mousemove", hover);
  $("#sw-field").addEventListener("mouseleave", () => { S.hover = -1; $("#sw-tip").hidden = true; draw(); });
  $("#sw-field").addEventListener("click", (ev) => { const i = tileAt(ev); if (i >= 0) unmask(S.tiles[i].r.model); });
  $("#sw-flagged").addEventListener("click", (ev) => { const b = ev.target.closest("[data-model]"); if (b) unmask(b.dataset.model); });
  $("#sw-history").addEventListener("click", (ev) => { const b = ev.target.closest("[data-sweep]"); if (b) open(b.dataset.sweep); });
  new ResizeObserver(() => draw()).observe($("#sw-field"));
  document.addEventListener("llminspect:view", (ev) => { if (ev.detail === "sweep") { loadHistory(); render(); } });
  defineBar("sweep", {
    groups: ["Sweep"],
    context: () => (S.job?.status === "running" ? `Sweeping: <b>${S.job.done || 0}/${S.job.total ?? "…"}</b>` : S.data ? `<b>${rows().length}</b> models` : "No sweep yet"),
    state: () => ({ swept: !!S.data && S.job?.status !== "running" }),
    actions: [
      { id: "run", group: "Sweep", label: "Sweep", hint: "Check the models the search picks, by byte range",
        need: () => (S.job?.status === "running" ? "the sweep running now to finish" : null), run: () => start() },
    ],
  });
  render();
}
boot();
