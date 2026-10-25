// Brainscan scanner: a live view of a model's internals, time-locked to what the agent does.
import { defineBar, noteBar, refreshBar } from "/ui/actionbar.js";
//
// One shared x axis (tokens on the run's tape) carries every track:
//   events   tool calls, results, errors, test outcomes, edits, verdicts
//   layers   logit-lens probability of each token at every layer ("when did it know")
//   bits     surprisal of each token and entropy of the distribution it came from
//   z        projections onto the direction bank, beside a matched random control

const $ = (sel) => document.querySelector(sel);
const api = async (path, opts) => {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
};
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const visTok = (s) => (s ?? "").replace(/\n/g, "⏎").replace(/\t/g, "⇥").replace(/ /g, "·") || "∅";
const surprisal = (p) => (p == null ? null : Math.min(16, -Math.log2(Math.max(p, 1e-12))));
const fmt = (x, d = 2) => (x == null || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
const bytes = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB` : n >= 1e6 ? `${(n / 1e6).toFixed(0)} MB` : `${(n / 1e3).toFixed(0)} KB`);

const S = {
  runs: [], runId: null, meta: null,
  tape: [], view: [], events: [], messages: [],
  dirs: [], nLayers: 32, status: null,
  colW: 6, scrollX: 0, plotW: 600, follow: true, showReplay: false,
  hover: null, selected: null, liveCard: null, ws: null, wsOpen: false, raf: 0,
  generating: null, genAt: 0,
};

// ---------------------------------------------------------------- colors
let C = {};
function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function hexToRgb(h) { h = h.replace("#", ""); return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)); }
function buildLut(stops) {
  const rgb = stops.map(hexToRgb), lut = [];
  for (let i = 0; i < 256; i++) {
    const x = (i / 255) * (rgb.length - 1), k = Math.min(Math.floor(x), rgb.length - 2), f = x - k;
    const c = rgb[k].map((v, j) => Math.round(v + (rgb[k + 1][j] - v) * f));
    lut.push(`rgb(${c[0]},${c[1]},${c[2]})`);
  }
  return lut;
}
function readColors() {
  const v = (n) => cssVar(n);
  C = {
    surface: v("--surface"), ink: v("--ink"), ink2: v("--ink-2"), muted: v("--muted"), grid: v("--grid"), axis: v("--axis"),
    series: [v("--series-1"), v("--series-2"), v("--series-3")], control: v("--control"),
    good: v("--good"), critical: v("--critical"), serious: v("--serious"), noData: v("--no-data"), focus: v("--focus"),
    roles: { system: v("--role-system"), user: v("--role-user"), assistant: v("--role-assistant"), reasoning: v("--role-reasoning"), tool: v("--role-tool"), template: v("--role-template") },
    ramp: v("--ramp").split(",").map((s) => s.trim()),
  };
  C.lut = buildLut(C.ramp);
}
// Lens bytes from the server are log10 p mapped from [1e-6, 1] onto [0, 255].
const lensP = (b) => 10 ** ((b / 255) * 6 - 6);
const fmtP = (p) => (p < 1.5e-6 ? "< 10⁻⁶" : p < 0.001 ? p.toExponential(1) : fmt(p, 3));

// ---------------------------------------------------------------- events vocabulary
function eventLook(e) {
  const d = e.data || {};
  switch (e.type) {
    case "system_prompt": return { label: "system", tone: "muted", glyph: "S" };
    case "user_message": return { label: "user", tone: "ink2", glyph: "U" };
    case "tool_call": return { label: `→ ${d.name}`, tone: "ink", glyph: "→", line: true };
    case "tool_result": return { label: `← ${d.name || "result"}`, tone: "ink2", glyph: "←" };
    case "error": return { label: "✕ error", tone: "critical", glyph: "✕", line: true, status: "critical" };
    case "test_result": {
      const bad = (d.failed || 0) + (d.errors || 0);
      return bad ? { label: `✕ ${bad} failing`, tone: "critical", glyph: "✕", line: true, status: "critical" }
        : { label: `✓ ${d.passed} passed`, tone: "good", glyph: "✓", line: true, status: "good" };
    }
    case "file_edit": return d.is_test ? { label: `▲ edited ${d.path}`, tone: "serious", glyph: "▲", line: true, status: "serious" }
      : { label: `✎ ${d.path}`, tone: "ink2", glyph: "✎" };
    case "final_answer": return { label: "■ answer", tone: "ink", glyph: "■" };
    case "think_end": return { label: "◆ end of thinking", tone: "ink2", glyph: "◆" };
    case "cutoff": return { label: d.while_reasoning ? "▲ cut off while thinking" : "▲ cut off", tone: "serious", glyph: "▲", line: true, status: "serious" };
    case "cancelled": return { label: d.in_prompt ? "■ stopped while reading" : d.while_reasoning ? "■ stopped while thinking" : "■ stopped", tone: "serious", glyph: "■", line: true, status: "serious" };
    case "monitor": return { label: `⚑ ${d.name}`, tone: "serious", glyph: "⚑", line: true, status: "serious" };
    case "verdict": return d.success ? { label: "✓ PASS", tone: "good", glyph: "✓", line: true, status: "good" }
      : { label: "✕ FAIL", tone: "critical", glyph: "✕", line: true, status: "critical" };
    default: return null;
  }
}
const toneColor = (tone) => ({ muted: C.muted, ink: C.ink, ink2: C.ink2, critical: C.critical, good: C.good, serious: C.serious }[tone] || C.ink2);

// ---------------------------------------------------------------- view model
function rebuildView() {
  S.view = [];
  for (const r of S.tape) if (r && (S.showReplay || r.ph !== "replay")) S.view.push(r.t);
}
function viewIndexOfT(t) {
  let lo = 0, hi = S.view.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (S.view[m] < t) lo = m + 1; else hi = m; }
  return Math.min(lo, Math.max(0, S.view.length - 1));
}
function clampScroll() {
  const max = Math.max(0, S.view.length * S.colW - S.plotW);
  if (S.follow) S.scrollX = max;
  S.scrollX = Math.min(Math.max(0, S.scrollX), max);
}
function schedule() { if (!S.raf) S.raf = requestAnimationFrame(draw); }

// ---------------------------------------------------------------- scope drawing
const GUTTER = 76, PAD_R = 92, MIN_COL = 0.2;  // columns may be sub-pixel when a whole run is fitted
function layout() {
  const L = S.nLayers, rowH = Math.max(3, Math.min(8, Math.floor(208 / L)));
  const y = {}; let cur = 6;
  y.events = [cur, 24]; cur += 24 + 6;
  y.heat = [cur, L * rowH]; cur += L * rowH;
  y.roles = [cur + 3, 4]; cur += 7 + 18;
  y.bits = [cur, 88]; cur += 88 + 20;
  y.z = [cur, 88]; cur += 88 + 10;
  y.ruler = [cur, 18]; cur += 18 + 4;
  return { y, rowH, height: cur };
}

function draw() {
  S.raf = 0;
  const cv = $("#scope"), ctx = cv.getContext("2d"), dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, lay = layout(), H = lay.height;
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); cv.style.height = `${H}px`;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = C.surface; ctx.fillRect(0, 0, W, H);
  S.plotW = Math.max(50, W - GUTTER - PAD_R);
  S.lay = lay;
  clampScroll();
  const n = S.view.length, cw = S.colW;
  const i0 = Math.max(0, Math.floor(S.scrollX / cw)), i1 = Math.min(n, Math.ceil((S.scrollX + S.plotW) / cw) + 1);
  const X = (i) => GUTTER + i * cw - S.scrollX;
  const rec = (i) => S.tape[S.view[i]];

  drawGutter(ctx, lay, W);
  ctx.save();
  ctx.beginPath(); ctx.rect(GUTTER, 0, S.plotW, H); ctx.clip();

  // event guide lines behind everything
  for (const e of S.events) {
    const look = eventLook(e); if (!look || !look.line || !n) continue;
    const x = X(viewIndexOfT(e.t)) + cw / 2;
    if (x < GUTTER - 2 || x > GUTTER + S.plotW + 2) continue;
    ctx.strokeStyle = toneColor(look.tone); ctx.globalAlpha = 0.35; ctx.setLineDash([2, 3]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, lay.y.heat[0]); ctx.lineTo(x, lay.y.z[0] + lay.y.z[1]); ctx.stroke();
    ctx.setLineDash([]); ctx.globalAlpha = 1;
  }

  // heatmap: logit-lens probability of the actual token, by layer (layer 0 at the bottom)
  const [hy, hh] = lay.y.heat, L = S.nLayers, rh = lay.rowH, gap = cw >= 4 ? 1 : 0;
  for (let i = i0; i < i1; i++) {
    const r = rec(i), x = X(i);
    if (!r.lens) { ctx.fillStyle = C.noData; ctx.fillRect(x, hy, cw - gap, hh); continue; }
    for (let l = 0; l < L; l++) {
      ctx.fillStyle = C.lut[r.lens[l]];
      ctx.fillRect(x, hy + (L - 1 - l) * rh, cw - gap, rh - (rh >= 5 ? 1 : 0));
    }
  }
  // role strip
  const [ry, rhh] = lay.y.roles;
  for (let i = i0; i < i1; i++) {
    const r = rec(i);
    ctx.fillStyle = C.roles[r.r] || C.roles.template;
    ctx.globalAlpha = r.ph === "replay" ? 0.4 : 1;
    ctx.fillRect(X(i), ry, cw - gap, rhh);
  }
  ctx.globalAlpha = 1;

  // bits: surprisal and entropy share one axis (both in bits)
  const [by, bh] = lay.y.bits;
  let bmax = 8;
  for (let i = i0; i < i1; i++) { const r = rec(i); bmax = Math.max(bmax, surprisal(r.p) ?? 0, r.e ?? 0); }
  bmax = Math.min(16, Math.ceil(bmax));
  S.bmax = bmax;
  gridLines(ctx, by, bh, [0, bmax / 2, bmax], bmax, 0);
  const Yb = (v) => by + bh - (v / bmax) * bh;
  line(ctx, i0, i1, X, (i) => surprisal(rec(i).p), Yb, C.series[0], 2);
  line(ctx, i0, i1, X, (i) => rec(i).e, Yb, C.series[1], 2);

  // z: projections with a band at |z| <= 2 and the random control
  const [zy, zh] = lay.y.z;
  let zmax = 3;
  for (let i = i0; i < i1; i++) { const pj = rec(i).pj || {}; for (const k in pj) zmax = Math.max(zmax, Math.abs(pj[k])); }
  zmax = Math.min(12, Math.ceil(zmax));
  S.zmax = zmax;
  const Yz = (v) => zy + zh / 2 - (v / zmax) * (zh / 2);
  ctx.fillStyle = C.grid; ctx.globalAlpha = 0.45; ctx.fillRect(GUTTER, Yz(2), S.plotW, Yz(-2) - Yz(2)); ctx.globalAlpha = 1;
  gridLines(ctx, zy, zh, [-zmax, 0, zmax], zmax, -zmax);
  dirSeries().forEach((d) => {
    line(ctx, i0, i1, X, (i) => (rec(i).pj || {})[d.name], Yz, d.color, d.control ? 1.5 : 2, d.control ? [4, 3] : null);
  });

  // ruler: tokens when zoomed in, tape indices otherwise
  const [uy] = lay.y.ruler;
  ctx.fillStyle = C.muted; ctx.font = "10px ui-monospace, Consolas, monospace"; ctx.textBaseline = "top";
  if (cw >= 22) {
    for (let i = i0; i < i1; i++) {
      const r = rec(i), s = visTok(r.tx);
      ctx.fillStyle = r.ph === "gen" ? C.ink : C.muted;
      ctx.save(); ctx.beginPath(); ctx.rect(X(i), uy, cw - 2, 14); ctx.clip(); ctx.fillText(s, X(i) + 1, uy + 2); ctx.restore();
    }
  } else if (n) {
    const step = niceStep(Math.ceil(70 / cw));
    ctx.strokeStyle = C.axis; ctx.lineWidth = 1;
    for (let i = Math.ceil(i0 / step) * step; i < i1; i += step) {
      const x = X(i);
      ctx.beginPath(); ctx.moveTo(x + 0.5, uy); ctx.lineTo(x + 0.5, uy + 4); ctx.stroke();
      ctx.fillText(String(S.view[i]), x + 3, uy + 3);
    }
  }

  // events lane on top
  const [ey, eh] = lay.y.events;
  let lastLabelEnd = -Infinity;
  ctx.textBaseline = "middle"; ctx.font = "11px system-ui, sans-serif";
  for (const e of S.events) {
    const look = eventLook(e); if (!look || !n) continue;
    const x = X(viewIndexOfT(e.t)) + cw / 2;
    if (x < GUTTER - 120 || x > GUTTER + S.plotW + 4) continue;
    ctx.fillStyle = toneColor(look.tone);
    ctx.beginPath(); ctx.arc(x, ey + eh / 2, 4, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = C.surface; ctx.lineWidth = 2; ctx.stroke();
    const w = ctx.measureText(look.label).width;
    if (x + 7 > lastLabelEnd + 6) {
      ctx.fillStyle = look.status ? toneColor(look.tone) : C.ink2;
      ctx.fillText(look.label, x + 7, ey + eh / 2);
      lastLabelEnd = x + 7 + w;
    }
  }

  // selection and hover columns
  const mark = (t, color, width) => {
    if (t == null || !n) return;
    const i = viewIndexOfT(t); if (S.view[i] !== t) return;
    ctx.strokeStyle = color; ctx.lineWidth = width;
    ctx.strokeRect(X(i) - 0.5, lay.y.heat[0] - 1, Math.max(cw, 2) + 1, lay.y.z[0] + lay.y.z[1] - lay.y.heat[0] + 2);
  };
  mark(S.selected, C.focus, 2);
  if (S.hover && S.hover.t != null) mark(S.hover.t, C.ink2, 1);
  ctx.restore();

  drawDirectLabels(ctx, lay, i1, X, rec, Yb, Yz);
  const ov = $("#overlay");
  if (!n && S.loadingRun) {
    ov.hidden = false;
    ov.innerHTML = `<b>Loading recording…</b>`;
  } else if (!n && S.status?.status === "ready") {
    ov.hidden = false;
    ov.innerHTML = `<div><b>No tokens recorded in this run yet.</b></div><div class="empty">Point any OpenAI-compatible harness at <code>${location.origin}/v1</code>, or run<br><code>scripts/run.sh -m brainscan.harness.mini_agent --task fix_fizzbuzz</code></div>`;
  } else if (S.status && S.status.status !== "ready") {
    ov.hidden = false;
    ov.innerHTML = S.status.status === "error" ? `<b>Recorder failed to start</b><div class="empty">${esc(S.status.error)}</div>`
      : `<b>Loading the subject model…</b><div class="empty">${esc(S.status.step)}</div>`;
  } else ov.hidden = true;
}

function niceStep(raw) { const p = 10 ** Math.floor(Math.log10(raw)); return [1, 2, 5, 10].map((m) => m * p).find((s) => s >= raw); }

function gridLines(ctx, y0, h, ticks, max, min) {
  ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
  for (const t of ticks) {
    const y = Math.round(y0 + h - ((t - min) / (max - min)) * h) + 0.5;
    ctx.beginPath(); ctx.moveTo(GUTTER, y); ctx.lineTo(GUTTER + S.plotW, y); ctx.stroke();
  }
}

function line(ctx, i0, i1, X, val, Y, color, width, dash) {
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = "round"; ctx.lineCap = "round";
  ctx.setLineDash(dash || []);
  ctx.beginPath();
  let pen = false;
  for (let i = i0; i < i1; i++) {
    const v = val(i);
    if (v == null || Number.isNaN(v)) { pen = false; continue; }
    const x = X(i) + S.colW / 2, y = Y(v);
    if (pen) ctx.lineTo(x, y); else { ctx.moveTo(x, y); pen = true; }
  }
  ctx.stroke(); ctx.setLineDash([]);
}

function dirSeries() {
  let k = 0;
  return S.dirs.map((d) => {
    const control = d.kind === "random_control";
    return { name: d.name, control, color: control ? C.control : C.series[k++ % C.series.length], meta: d };
  });
}

function drawGutter(ctx, lay, W) {
  ctx.fillStyle = C.ink2; ctx.font = "11px system-ui, sans-serif"; ctx.textBaseline = "middle"; ctx.textAlign = "left";
  const label = (txt, y) => ctx.fillText(txt, 8, y);
  label("events", lay.y.events[0] + lay.y.events[1] / 2);
  label("logit lens", lay.y.heat[0] + 8);
  ctx.fillStyle = C.muted; ctx.font = "10px system-ui, sans-serif";
  label("p(token)", lay.y.heat[0] + 22);
  ctx.textAlign = "right";
  ctx.fillText(`L${S.nLayers - 1}`, GUTTER - 6, lay.y.heat[0] + 4);
  ctx.fillText("L0", GUTTER - 6, lay.y.heat[0] + lay.y.heat[1] - 4);
  ctx.textAlign = "left"; ctx.fillStyle = C.ink2; ctx.font = "11px system-ui, sans-serif";
  label("bits", lay.y.bits[0] + 8);
  label("z", lay.y.z[0] + 8);
  ctx.fillStyle = C.muted; ctx.font = "10px system-ui, sans-serif"; ctx.textAlign = "right";
  if (S.bmax) { ctx.fillText(String(S.bmax), GUTTER - 6, lay.y.bits[0] + 4); ctx.fillText("0", GUTTER - 6, lay.y.bits[0] + lay.y.bits[1] - 2); }
  if (S.zmax) {
    ctx.fillText(`+${S.zmax}`, GUTTER - 6, lay.y.z[0] + 4); ctx.fillText("0", GUTTER - 6, lay.y.z[0] + lay.y.z[1] / 2);
    ctx.fillText(`−${S.zmax}`, GUTTER - 6, lay.y.z[0] + lay.y.z[1] - 4);
  }
  ctx.textAlign = "left";
  label(S.colW >= 22 ? "tokens" : "tape", lay.y.ruler[0] + 8);
}

function drawDirectLabels(ctx, lay, i1, X, rec, Yb, Yz) {
  if (!S.view.length) return;
  const last = Math.min(i1, S.view.length) - 1, r = rec(last), xL = GUTTER + S.plotW + 8;
  ctx.font = "11px system-ui, sans-serif"; ctx.textBaseline = "middle"; ctx.textAlign = "left";
  const placed = [];
  const put = (y, color, text, dashed) => {
    while (placed.some((p) => Math.abs(p - y) < 12)) y += 12;
    placed.push(y);
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash(dashed ? [3, 2] : []);
    ctx.beginPath(); ctx.moveTo(xL, y); ctx.lineTo(xL + 10, y); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = C.ink2; ctx.fillText(text, xL + 14, y);
  };
  if (r.p != null) put(Yb(surprisal(r.p)), C.series[0], "surprisal");
  if (r.e != null) put(Yb(r.e), C.series[1], "entropy");
  placed.length = 0;
  dirSeries().forEach((d) => {
    const v = (r.pj || {})[d.name];
    if (v != null) put(Yz(Math.max(-S.zmax, Math.min(S.zmax, v))), d.color, shortName(d.name), d.control);
  });
}
const shortName = (name) => (name.length > 12 ? `${name.slice(0, 11)}…` : name);

function renderLegend() {
  const ds = dirSeries();
  $("#legend").innerHTML = [
    `<span class="item"><span class="grad" style="background:linear-gradient(90deg, ${C.ramp.join(",")})"></span>logit-lens p(token), log scale 10⁻⁶ → 1</span>`,
    ...["system", "user", "reasoning", "assistant", "tool"].map((r) => `<span class="item"><span class="box" style="background:${C.roles[r]}"></span>${r}</span>`),
    `<span class="item"><span class="line" style="background:${C.series[0]}"></span>surprisal −log₂ p</span>`,
    `<span class="item"><span class="line" style="background:${C.series[1]}"></span>entropy</span>`,
    ...ds.map((d) => `<span class="item" title="${esc(d.meta.construction || "")}"><span class="line${d.control ? " dashed" : ""}" style="background:${d.color}"></span>${esc(d.name)} · L${d.meta.layer}</span>`),
    `<span class="item"><span class="box" style="background:${C.grid}"></span>|z| ≤ 2</span>`,
  ].join("");
}

// ---------------------------------------------------------------- hover, tooltip, pan, zoom
function hitTest(mx, my) {
  const lay = S.lay; if (!lay || !S.view.length || mx < GUTTER || mx > GUTTER + S.plotW) return null;
  const i = Math.floor((mx - GUTTER + S.scrollX) / S.colW);
  if (i < 0 || i >= S.view.length) return null;
  const hit = { i, t: S.view[i], track: null, layer: null };
  const inY = ([y, h]) => my >= y && my <= y + h;
  if (inY(lay.y.events)) hit.track = "events";
  else if (inY(lay.y.heat)) { hit.track = "heat"; hit.layer = S.nLayers - 1 - Math.floor((my - lay.y.heat[0]) / lay.rowH); }
  else hit.track = "token";
  return hit;
}

function showTooltip(hit, mx, my) {
  const tip = $("#tooltip");
  if (!hit) { tip.hidden = true; return; }
  const r = S.tape[hit.t];
  let html = `<div><span class="tok">${esc(visTok(r.tx))}</span> <span class="muted">${r.r} · ${r.ph} · t${r.t}</span></div>`;
  if (hit.track === "events") {
    const near = S.events.filter((e) => eventLook(e) && Math.abs(viewIndexOfT(e.t) - hit.i) * S.colW < 10);
    if (near.length) html = near.map((e) => `<div><b>${esc(eventLook(e).label)}</b> <span class="muted">t${e.t} · ${e.source}</span></div>${e.data?.excerpt ? `<div class="muted">${esc(e.data.excerpt)}</div>` : ""}`).join("");
  } else if (hit.track === "heat" && r.lens) {
    html += `<div class="row"><span>layer ${hit.layer}: p(this token)</span><span>${fmtP(lensP(r.lens[hit.layer]))}</span></div>`;
  }
  if (hit.track !== "events") {
    html += `<div class="row"><span><i class="sw" style="background:${C.series[0]}"></i>surprisal</span><span>${fmt(surprisal(r.p))} bits</span></div>`;
    html += `<div class="row"><span><i class="sw" style="background:${C.series[1]}"></i>entropy</span><span>${fmt(r.e)} bits</span></div>`;
    for (const d of dirSeries()) {
      const v = (r.pj || {})[d.name];
      if (v != null) html += `<div class="row"><span><i class="sw" style="background:${d.color}"></i>${esc(d.name)}</span><span>${v >= 0 ? "+" : ""}${fmt(v)} z</span></div>`;
    }
  }
  tip.innerHTML = html; tip.hidden = false;
  const wrap = $("#scope-wrap").getBoundingClientRect();
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = `${Math.min(mx + 14, wrap.width - w - 4)}px`;
  tip.style.top = `${Math.max(4, Math.min(my + 14, wrap.height - h - 4))}px`;
}

function setupScopeInput() {
  const cv = $("#scope");
  let drag = null;
  cv.addEventListener("pointermove", (ev) => {
    const r = cv.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
    if (drag) {
      const dx = ev.clientX - drag.x;
      if (Math.abs(dx) > 3) drag.moved = true;
      S.scrollX = drag.scroll - dx; S.follow = false; $("#follow").checked = false;
      schedule(); return;
    }
    S.hover = hitTest(mx, my); showTooltip(S.hover, mx, my); schedule();
  });
  cv.addEventListener("pointerleave", () => { S.hover = null; $("#tooltip").hidden = true; schedule(); });
  cv.addEventListener("pointerdown", (ev) => { drag = { x: ev.clientX, scroll: S.scrollX, moved: false }; cv.setPointerCapture(ev.pointerId); });
  cv.addEventListener("pointerup", (ev) => {
    const wasDrag = drag?.moved; drag = null;
    if (wasDrag) return;
    const r = cv.getBoundingClientRect(), hit = hitTest(ev.clientX - r.left, ev.clientY - r.top);
    if (hit) selectToken(hit.t);
  });
  cv.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    const r = cv.getBoundingClientRect(), mx = ev.clientX - r.left;
    if (ev.ctrlKey || ev.metaKey) {
      const anchor = (mx - GUTTER + S.scrollX) / S.colW;
      S.colW = Math.max(MIN_COL, Math.min(40, S.colW * (ev.deltaY < 0 ? 1.2 : 1 / 1.2)));
      S.scrollX = anchor * S.colW - (mx - GUTTER);
    } else {
      S.scrollX += ev.deltaX || ev.deltaY;
    }
    S.follow = false; $("#follow").checked = false; schedule();
  }, { passive: false });
  cv.tabIndex = 0;
  cv.addEventListener("keydown", (ev) => {
    if (S.selected == null) return;
    const i = viewIndexOfT(S.selected);
    if (ev.key === "ArrowRight" && i + 1 < S.view.length) selectToken(S.view[i + 1], true);
    if (ev.key === "ArrowLeft" && i > 0) selectToken(S.view[i - 1], true);
  });
  const zoom = (f) => {
    const center = (S.scrollX + S.plotW / 2) / S.colW;
    S.colW = Math.max(MIN_COL, Math.min(40, S.colW * f));
    S.scrollX = center * S.colW - S.plotW / 2; schedule();
  };
  $("#zoom-in").onclick = () => zoom(1.5);
  $("#zoom-out").onclick = () => zoom(1 / 1.5);
  $("#fit").onclick = () => {
    S.colW = Math.max(MIN_COL, Math.min(40, S.plotW / Math.max(1, S.view.length)));
    S.scrollX = 0; S.follow = false; $("#follow").checked = false; schedule();
  };
  $("#follow").onchange = (ev) => { S.follow = ev.target.checked; schedule(); };
  $("#show-replay").onchange = (ev) => { S.showReplay = ev.target.checked; rebuildView(); schedule(); };
  new ResizeObserver(schedule).observe($("#scope-wrap"));
}

// ---------------------------------------------------------------- transcript
function roleColor(role) { return C.roles[role] || C.muted; }

function messageBadges(m) {
  return S.events.filter((e) => e.t === m.t && ["error", "test_result"].includes(e.type)).map((e) => {
    const look = eventLook(e);
    const text = e.type === "error" ? "error" : look.label.replace(/^[✓✕]\s*/, "");
    return `<span class="badge ${look.status}">${esc(text)}</span>`;
  }).join("");
}

function messageHtml(m) {
  const calls = (m.tool_calls || []).map((c) => {
    let args = c.function?.arguments ?? "";
    if (typeof args !== "string") args = JSON.stringify(args);
    return `<div class="call">${esc(c.function?.name)}(${esc(args.length > 300 ? args.slice(0, 300) + "…" : args)})</div>`;
  }).join("");
  const label = m.role === "tool" ? `tool · ${esc(m.name || "")}` : m.role;
  const content = m.content ? `<pre>${esc(m.content)}</pre>` : "";
  const long = (m.content || "").split("\n").length > 14 || (m.content || "").length > 900;
  const words = (m.reasoning || "").split(/\s+/).filter(Boolean).length;
  const thinking = m.reasoning ? `<details class="thinking"><summary>Thinking · ${words} words</summary><pre>${esc(m.reasoning)}</pre></details>` : "";
  return `<div class="msg-head"><span class="role">${label}</span><span>t${m.t}</span>${m.generated ? `<span>generated · ${esc(m.finish || "")}</span>` : ""}</div>
    ${thinking}${content}${long ? `<button class="more">Show all</button>` : ""}${calls}<div class="badges">${messageBadges(m)}</div>`;
}

function addMessageCard(m) {
  const el = document.createElement("div");
  el.className = "msg"; el.dataset.t = m.t; el.style.setProperty("--role-color", roleColor(m.role));
  el.innerHTML = messageHtml(m);
  el.addEventListener("click", (ev) => {
    if (ev.target.classList.contains("more")) { el.classList.toggle("expanded"); ev.target.textContent = el.classList.contains("expanded") ? "Show less" : "Show all"; return; }
    selectToken(m.t, true);
  });
  const box = $("#messages");
  const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 80;
  box.appendChild(el);
  if (nearBottom) box.scrollTop = box.scrollHeight;
  return el;
}

function renderTranscript() {
  const box = $("#messages"); box.innerHTML = ""; S.liveCard = null;
  if (!S.messages.length) {
    box.innerHTML = `<p class="empty">No messages yet.</p>`;
    return;
  }
  [...S.messages].sort((a, b) => a.t - b.t || a.i - b.i).forEach(addMessageCard);
}

function liveToken(rec) {
  if (rec.ph !== "gen") return;
  if (!S.liveCard) {
    const empty = $("#messages .empty"); if (empty) empty.remove();
    S.liveCard = addMessageCard({ role: "assistant", t: rec.t, content: "", generated: true, finish: "generating…" });
    S.liveCard.querySelector(".msg-head").insertAdjacentHTML("afterend",
      `<details class="thinking live" open hidden><summary>Thinking…</summary><pre></pre></details><pre class="caret"></pre>`);
  }
  if (/^<\|.*\|>$/.test(rec.tx) || /^<\/?think>$/.test(rec.tx)) return;
  if (rec.r === "reasoning") {
    const think = S.liveCard.querySelector("details.thinking");
    think.hidden = false;
    think.querySelector("pre").textContent += rec.tx;
  } else {
    S.liveCard.querySelector("pre.caret").textContent += rec.tx;
  }
  const box = $("#messages"); box.scrollTop = box.scrollHeight;
}

function refreshBadges() {
  document.querySelectorAll("#messages .msg").forEach((el) => {
    const m = S.messages.find((x) => String(x.t) === el.dataset.t);
    if (m) el.querySelector(".badges").innerHTML = messageBadges(m);
  });
}

// ---------------------------------------------------------------- inspector
async function selectToken(t, scrollIntoView = false) {
  S.selected = t;
  if (scrollIntoView && S.view.length) {
    const i = viewIndexOfT(t), x = i * S.colW;
    if (x < S.scrollX || x > S.scrollX + S.plotW - S.colW) { S.scrollX = x - S.plotW / 2; S.follow = false; $("#follow").checked = false; }
  }
  document.querySelectorAll("#messages .msg").forEach((el) => {
    const m = S.messages.find((x) => String(x.t) === el.dataset.t);
    el.classList.toggle("selected", !!m && m.t === t);
  });
  schedule();
  try {
    const d = await api(`/brainscan/runs/${encodeURIComponent(S.runId)}/tokens/${t}`);
    if (S.selected === t) renderInspector(d);
  } catch (err) { $("#inspector").innerHTML = `<p class="empty">${esc(err.message)}</p>`; }
}

function renderInspector(d) {
  S.inspected = d;
  refreshBar("scanner");
  $("#insp-where").textContent = `t${d.t} · position ${d.pos} · request ${d.req}`;
  const s = surprisal(d.p);
  const facts = [
    `<span>role <b>${esc(d.role)}</b></span>`, `<span>phase <b>${esc(d.ph)}</b></span>`,
    `<span>p <b>${fmt(d.p, 4)}</b></span>`, `<span>surprisal <b>${fmt(s)} bits</b></span>`, `<span>entropy before <b>${fmt(d.ent)} bits</b></span>`,
  ];
  if (d.has_residual) facts.push(`<span class="badge">full residual saved</span>`);
  if (d.model_mismatch) facts.push(`<span class="badge serious">recorded with a different model</span>`);
  let html = `<div class="tok-hero"><span class="tok">${esc(visTok(d.text))}</span><div class="facts">${facts.join("")}</div>
    ${d.p != null && !d.model_mismatch ? `<button type="button" class="btn sm" id="insp-trace" title="Open the Trace tab on this token: the layers, units and knockouts behind it">Trace this token →</button>` : ""}</div>`;
  if (d.p == null) html += `<p class="empty">No prediction is stored for this slot (first token of the context or of a resumed segment).</p>`;
  const alts = (d.top || []).map(([tok, p]) => `<li class="${tok === d.text ? "actual" : ""}"><span class="t">${esc(visTok(tok))}</span><span class="track"><span class="fill" style="width:${(p * 100).toFixed(1)}%;display:block"></span></span><span class="v">${fmt(p, 3)}</span></li>`).join("");
  let lens = "";
  if (d.lens_p) {
    const rows = d.lens_p.map((p, l) => ({ l, p, top: d.lens_top_text[l], hit: d.lens_top[l] === d.tok })).reverse();
    const logW = (p) => Math.max(0, Math.min(100, ((Math.log10(Math.max(p, 1e-6)) + 6) / 6) * 100)).toFixed(1);
    lens = `<div class="lens-wrap"><table class="lens-table"><thead><tr><th>layer</th><th>top guess</th><th>p(this token), log bar</th></tr></thead><tbody>${rows.map((r) =>
      `<tr class="${r.hit ? "hit" : ""}"><td>${r.l}</td><td class="t">${esc(visTok(r.top))}${r.hit ? " ✓" : ""}</td><td><div class="cell-bar"><span class="track"><span class="fill" style="width:${logW(r.p)}%"></span></span>${fmtP(r.p)}</div></td></tr>`).join("")}</tbody></table></div>`;
    const firstHit = d.lens_top.findIndex((x) => x === d.tok);
    lens = `<p class="small muted">${firstHit >= 0 ? `This token first becomes the top guess at layer ${firstHit}.` : "No layer has this token as its top guess."}</p>` + lens;
  }
  const projs = dirSeries().map((ds) => {
    const v = (d.proj || {})[ds.name]; if (v == null) return "";
    const c = Math.max(-6, Math.min(6, v)), left = c >= 0 ? 50 : 50 + (c / 6) * 50, width = Math.abs(c / 6) * 50;
    return `<li><div class="head"><span>${esc(ds.name)} · L${ds.meta.layer}</span><span>${v >= 0 ? "+" : ""}${fmt(v)} z</span></div>
      <div class="zbar"><span class="zero"></span><span class="fill" style="left:${left}%;width:${width}%;background:${ds.color}"></span></div>
      <div class="how">${esc(ds.meta.construction || "")}</div></li>`;
  }).join("");
  html += `<div class="cols"><div><h3>Before this token, the model predicted</h3><ul class="bars">${alts}</ul>
    <h3>Projections at this position</h3><ul class="proj-list">${projs || '<li class="muted">No residual (stop token).</li>'}</ul>
    ${d.norm ? `<h3>Residual norm by layer</h3>${sparkline(d.norm)}` : ""}</div>
    <div><h3>Logit lens: each layer's guess for this slot</h3>${lens || '<p class="empty">No lens for this token.</p>'}</div></div>`;
  $("#inspector").innerHTML = html;
  $("#insp-trace")?.addEventListener("click", () => {
    document.dispatchEvent(new CustomEvent("brainscan:show", { detail: "weights" }));
    document.dispatchEvent(new CustomEvent("brainscan:trace", { detail: { word: d.t } }));
  });
}

function sparkline(vals) {
  const w = 260, h = 48, max = Math.max(...vals), min = 0;
  const pts = vals.map((v, i) => `${(i / (vals.length - 1)) * (w - 4) + 2},${h - 4 - ((v - min) / (max - min || 1)) * (h - 10)}`).join(" ");
  return `<svg width="${w}" height="${h}" role="img" aria-label="Residual norm from layer 0 to ${vals.length - 1}, max ${fmt(max, 1)}">
    <line x1="2" y1="${h - 4}" x2="${w - 2}" y2="${h - 4}" stroke="${C.axis}" stroke-width="1"/>
    <polyline points="${pts}" fill="none" stroke="${C.series[0]}" stroke-width="2" stroke-linejoin="round"/>
    <text x="${w - 2}" y="10" text-anchor="end" font-size="10" fill="${C.muted}">max ${fmt(max, 0)}</text></svg>`;
}

// ---------------------------------------------------------------- steering panel
function renderSteering() {
  const st = S.status || {};
  const ivs = st.interventions || [];
  const on = ivs.filter((iv) => iv.enabled);
  const banner = $("#steer-banner");
  banner.hidden = !on.length;
  banner.textContent = on.length ? `Steering is on (${on.length}). New requests are recorded with these edits and marked as intervened.` : "";
  $("#iv-list").innerHTML = ivs.map((iv) => {
    const what = iv.mode === "add" ? `add ${iv.strength >= 0 ? "+" : ""}${iv.strength}×|h|` : iv.mode === "clamp" ? `clamp to ${iv.value}`
      : iv.mode === "write" ? `${iv.knob ? `knob ${iv.knob}` : "neuron"}, dose ${iv.strength >= 0 ? "+" : ""}${iv.strength}` : "ablate";
    return `<li class="${iv.enabled ? "" : "off"}" data-id="${iv.id}"><input type="checkbox" ${iv.enabled ? "checked" : ""} aria-label="Enable">
      <span class="desc">${esc(what)} · ${esc(iv.direction)} @ L${iv.layer} · ${esc(iv.phase)}</span><button aria-label="Remove">✕</button></li>`;
  }).join("") || `<li class="muted" style="border:none;background:none;padding:0">No interventions. Readouts are passive.</li>`;
  $("#iv-list").querySelectorAll("li[data-id]").forEach((li) => {
    li.querySelector("input").onchange = async (ev) => { await api(`/brainscan/interventions/${li.dataset.id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ enabled: ev.target.checked }) }); pollStatus(); };
    li.querySelector("button").onclick = async () => { await api(`/brainscan/interventions/${li.dataset.id}`, { method: "DELETE" }); pollStatus(); };
  });
  $("#dir-list").innerHTML = (st.directions || []).map((d) => `<li><div><span class="name">${esc(d.name)}</span> <span class="muted">· L${d.layer} · ${esc(d.kind)}</span></div><div class="how">${esc(d.description || "")}</div><div class="how">${esc(d.construction || "")}</div></li>`).join("");
  const sel = $("#iv-direction"), prev = sel.value;
  const options = [...(st.directions || []).map((d) => d.name), "random"];
  if (sel.options.length !== options.length) {
    sel.innerHTML = options.map((n) => `<option value="${esc(n)}">${n === "random" ? "random vector (control)" : esc(n)}</option>`).join("");
    if (options.includes(prev)) sel.value = prev;
    syncLayerDefault();
  }
}
function syncLayerDefault() {
  const d = (S.status?.directions || []).find((x) => x.name === $("#iv-direction").value);
  $("#iv-layer").value = d ? d.layer : Math.round((S.nLayers - 1) * 0.66);
  $("#iv-layer").max = S.nLayers - 1;
}
function setupSteering() {
  $("#iv-direction").onchange = syncLayerDefault;
  $("#iv-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const mode = $("#iv-mode").value, num = Number($("#iv-strength").value);
    const body = { direction: $("#iv-direction").value, layer: Number($("#iv-layer").value), mode, phase: $("#iv-phase").value };
    if (mode === "add") body.strength = num; if (mode === "clamp") body.value = num;
    try { await api("/brainscan/interventions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }); }
    catch (err) { alert(err.message); }
    pollStatus();
  });
}

// ---------------------------------------------------------------- runs and live feed
function runLabel(m) {
  const when = new Date((m.created || 0) * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const out = m.outcome ? (m.outcome.success ? " ✓" : " ✕") : "";
  const iv = (m.interventions || []).length ? " ▲steered" : "";
  const fl = m.flags ? ` ⚑${m.flags}` : "";  // flags monitors raised while it was recorded
  return `${m.label || m.id} · ${when} · ${m.tokens || 0} tok${out}${iv}${fl}`;
}
function renderRunSelect() {
  const sel = $("#run-select");
  sel.innerHTML = S.runs.length ? S.runs.map((m) => `<option value="${esc(m.id)}">${esc(runLabel(m))}</option>`).join("") : `<option value="">no runs yet</option>`;
  if (S.runId) sel.value = S.runId;
}
async function loadRuns() {
  try { S.runs = await api("/brainscan/runs"); } catch { S.runs = []; }
  renderRunSelect();
}
async function openRun(id) {
  S.runId = id; S.selected = null; S.tape = []; S.view = []; S.events = []; S.messages = [];
  $("#inspector").innerHTML = `<p class="empty">Click any token in the scope or transcript to see what the model predicted there, layer by layer.</p>`;
  $("#insp-where").textContent = "";
  renderRunSelect();
  document.dispatchEvent(new CustomEvent("brainscan:run", { detail: id }));  // the Weights tab follows the same run
  if (!id) { renderTranscript(); schedule(); return; }
  // Live messages for this run that arrive while the snapshot loads are buffered, then replayed.
  S.loadingRun = id; S.buffer = [];
  try {
    const [run, tape] = await Promise.all([api(`/brainscan/runs/${encodeURIComponent(id)}`), api(`/brainscan/runs/${encodeURIComponent(id)}/tape`)]);
    if (S.runId !== id) return;
    S.meta = run.meta; S.events = run.events; S.messages = run.messages;
    for (const r of tape) S.tape[r.t] = r;
    const L = run.meta?.model?.n_layers; if (L) S.nLayers = L;
    rebuildView(); renderTranscript(); renderMeta();
  } finally {
    if (S.loadingRun === id) {
      const buffered = S.buffer;
      S.loadingRun = null; S.buffer = [];
      buffered.forEach(handle);
    }
  }
  schedule();
}
function renderMeta() {
  refreshBar("scanner");
  const m = S.meta; if (!m) { $("#run-meta").textContent = ""; return; }
  const out = m.outcome ? (m.outcome.success ? "✓ pass" : "✕ fail") : "in progress";
  $("#run-meta").textContent = `${m.requests} requests · ${m.tokens} tokens · ${out}`;
}

/** Which request is generating, for the Stop buttons (ui/chat.js): live from "req"/"done", else polled. */
function setGenerating(g) {
  S.genAt = performance.now();
  S.generating = g;
  refreshBar("scanner");
  document.dispatchEvent(new CustomEvent("brainscan:generating", { detail: g }));
}

function handle(msg) {
  // A monitor fired, in any run (ui/monitors.js shows it).
  if (msg.k === "ev" && msg.ev?.type === "monitor") document.dispatchEvent(new CustomEvent("brainscan:flag", { detail: { run: msg.run, ev: msg.ev } }));
  if (msg.k === "req") setGenerating({ run: msg.run, req: msg.req });
  else if (msg.k === "done") { setGenerating(null); document.dispatchEvent(new CustomEvent("brainscan:done", { detail: msg })); }
  if (S.loadingRun && msg.run === S.loadingRun && ["toks", "ev", "msg", "done"].includes(msg.k)) {
    S.buffer.push(msg);
    return;
  }
  switch (msg.k) {
    case "run": {
      const i = S.runs.findIndex((r) => r.id === msg.run);
      if (i >= 0) S.runs[i] = msg.meta; else S.runs.unshift(msg.meta);
      renderRunSelect();
      if (msg.run === S.runId) { S.meta = msg.meta; renderMeta(); }
      else if ((i < 0 && S.follow) || msg.run === S.expectRun) openRun(msg.run);
      if (msg.run === S.expectRun) S.expectRun = null;
      break;
    }
    case "toks":
      if (msg.run !== S.runId) break;
      for (const r of msg.items) {
        if (S.tape[r.t]) continue;  // already in the snapshot
        S.tape[r.t] = r;
        if (S.showReplay || r.ph !== "replay") {
          if (!S.view.length || r.t > S.view[S.view.length - 1]) S.view.push(r.t); else rebuildView();
        }
        liveToken(r);
      }
      schedule();
      break;
    case "ev":
      if (msg.run !== S.runId || S.events.some((e) => e.ts === msg.ev.ts && e.type === msg.ev.type)) break;
      S.events.push(msg.ev); refreshBadges(); schedule();
      break;
    case "msg":
      if (msg.run !== S.runId || S.messages.some((m) => m.req === msg.msg.req && m.i === msg.msg.i && m.t === msg.msg.t)) break;
      S.messages.push(msg.msg);
      if (msg.msg.generated && S.liveCard) { S.liveCard.remove(); S.liveCard = null; }
      { const empty = $("#messages .empty"); if (empty) empty.remove(); }
      addMessageCard(msg.msg); refreshBadges();
      break;
    case "done": {
      const i = S.runs.findIndex((r) => r.id === msg.run);
      if (i >= 0) { S.runs[i] = msg.meta; renderRunSelect(); }
      if (msg.run === S.runId) { S.meta = msg.meta; renderMeta(); }
      if (msg.perf?.decode_tok_s) $("#stat-speed").textContent = `${msg.perf.decode_tok_s} tok/s`;
      break;
    }
    case "status": pollStatus(); loadRuns(); break;
    case "model": pollStatus(); loadRuns(); break;  // pollStatus tells the other views
  }
}

function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/brainscan/live`);
  S.ws = ws;
  ws.onopen = () => { S.wsOpen = true; $("#live-dot").classList.add("on"); };
  ws.onclose = () => { S.wsOpen = false; $("#live-dot").classList.remove("on"); setTimeout(connect, 1500); };
  ws.onmessage = (ev) => { for (const msg of JSON.parse(ev.data)) handle(msg); };
}

async function pollStatus() {
  const asked = performance.now();
  try { S.status = await api("/brainscan/status"); } catch (err) { S.status = { status: "error", error: err.message }; }
  const st = S.status;
  // A live "req" or "done" that arrived while this poll was out is newer than the poll.
  const gen = st.generating ? { run: st.generating.run, req: st.generating.req } : null;
  if (S.genAt < asked && JSON.stringify(gen) !== JSON.stringify(S.generating)) setGenerating(gen);
  if (st.model && st.model.spec !== S.modelSpec) {
    // The recorder switched models: the Trace and Tune views rebuild for it, the run list refreshes.
    if (S.modelSpec !== undefined) { document.dispatchEvent(new CustomEvent("brainscan:model", { detail: st.model })); loadRuns(); }
    S.modelSpec = st.model.spec;
  }
  if (st.model) {
    S.nLayers = S.meta?.model?.n_layers || st.model.n_layers;
    const prov = st.model.provenance || {};
    const q = prov.quantization ? `<span class="warn" title="${esc(prov.note || "")}">${esc(prov.quantization)} → ${esc(st.model.dtype)} · exploratory</span>` : `<span>${esc(st.model.dtype)}</span>`;
    const edited = st.workbench?.applied?.length;
    const ed = edited ? ` · <span class="warn" title="${esc(st.workbench.applied.map((e) => e.summary).join("\n"))}">weights edited (${edited})</span>` : "";
    $("#model-badge").innerHTML = `${esc(st.model.name)}<span class="mb-more"> · ${st.model.n_layers} layers · ${q}</span>${ed}`;
  } else {
    $("#model-badge").textContent = st.status === "error" ? "Recorder error" : `Loading: ${st.step || ""}`;
  }
  if (st.gpu) $("#stat-vram").textContent = `VRAM ${(st.gpu.used / 2 ** 30).toFixed(1)} / ${(st.gpu.total / 2 ** 30).toFixed(0)} GB`;
  if (st.disk) $("#stat-disk").textContent = `Disk ${bytes(st.disk.used)} / ${bytes(st.disk.budget)}`;
  if (st.perf?.decode_tok_s) $("#stat-speed").textContent = `${st.perf.decode_tok_s} tok/s`;
  const dirsChanged = JSON.stringify((st.directions || []).map((d) => d.name)) !== JSON.stringify(S.dirs.map((d) => d.name));
  if (st.directions) S.dirs = st.directions;
  if (dirsChanged) renderLegend();
  renderSteering();
  refreshBar("scanner");
  document.dispatchEvent(new CustomEvent("brainscan:status", { detail: st }));  // the Neurons view reads the model's name and its knobs
  schedule();
}

// ---------------------------------------------------------------- theme and boot
function setupTheme() {
  const saved = (() => { try { return localStorage.getItem("brainscan-theme"); } catch { return null; } })();
  if (saved) document.documentElement.dataset.theme = saved;
  $("#theme-toggle").onclick = () => {
    const dark = matchMedia("(prefers-color-scheme: dark)").matches;
    const cur = document.documentElement.dataset.theme || (dark ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("brainscan-theme", next); } catch { /* storage unavailable */ }
    readColors(); renderLegend(); renderTranscript(); schedule();
  };
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { readColors(); renderLegend(); schedule(); });
}

// ---------------------------------------------------------------- the action bar (ui/actionbar.js)
function setupBar() {
  const composer = () => $("#view-scanner .composer");
  const steering = () => S.status?.interventions || [];
  const traceable = () => (S.selected == null ? "a token: click one on the timeline or in the transcript"
    : S.inspected?.t === S.selected && (S.inspected.p == null || S.inspected.model_mismatch) ? "a token the model predicted (with the loaded model)" : null);
  const openRun = () => S.runs?.find((r) => r.id === S.runId);
  defineBar("scanner", {
    groups: ["Run", "Read", "Steer", "Hand off"],
    // What the action steps read (ui/verbs.js).
    state: () => ({ run: !!(S.runId && S.view.length), steered: !!(openRun()?.interventions || []).length,
      steering: steering().filter((iv) => iv.enabled).length, directions: S.dirs.length }),
    context: () => {
      const run = S.runs?.find((r) => r.id === S.runId);
      const where = run ? `<b>${esc(run.label || run.id)}</b> · ${run.tokens || 0} tokens` : "No run open · ask the model below";
      const tok = S.inspected && S.inspected.t === S.selected ? ` · token <b>“${esc(visTok(S.inspected.text))}”</b> (t${S.selected}) selected` : "";
      const st = steering().filter((iv) => iv.enabled).length;
      return `${where}${tok}${S.generating ? " · <b>generating</b>" : ""}${st ? ` · steering on (${st})` : ""}`;
    },
    actions: [
      { id: "ask", group: "Run", label: "Ask the model", hint: "Type a message under the transcript; it continues the open chat",
        run: () => composer()?.querySelector("textarea")?.focus() },
      { id: "new", group: "Run", label: "New chat", hint: "The next message starts a new conversation",
        run: () => { composer()?.querySelector(".composer-new")?.click(); composer()?.querySelector("textarea")?.focus(); } },
      { id: "stop", group: "Run", label: "Stop", hint: "End the answer at the next token, keeping what it wrote",
        need: () => (S.generating ? null : "a request that is generating"),
        run: async () => {
          try { await api("/brainscan/cancel", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }); noteBar("scanner", "Stopping at the next token"); }
          catch (err) { noteBar("scanner", `Couldn't stop: ${err.message}`); }
        } },
      { id: "model", group: "Run", label: "Switch model", hint: "Any open-weights model already on this machine",
        run: () => document.dispatchEvent(new CustomEvent("brainscan:models")) },
      { id: "fit", group: "Read", label: "Fit timeline", hint: "Show the whole run on the timeline",
        need: () => (S.view.length ? null : "a run with tokens"), run: () => $("#fit").click() },
      { id: "live", group: "Read", label: () => (S.follow ? "Following live" : "Follow live"), hint: "Keep the newest token in view",
        run: () => { const f = $("#follow"); f.checked = !f.checked; f.dispatchEvent(new Event("change")); refreshBar("scanner"); } },
      { id: "steer", group: "Steer", label: "Add steering", hint: "Add, ablate or clamp a saved direction at a layer while the model writes",
        run: () => { $("#h-steer")?.scrollIntoView({ behavior: "smooth", block: "start" }); $("#iv-direction")?.focus(); } },
      { id: "unsteer", group: "Steer", label: "Remove steering", hint: "Turn every intervention off and remove it",
        need: () => (steering().length ? null : "steering to remove"),
        run: async () => {
          for (const iv of steering()) await api(`/brainscan/interventions/${iv.id}`, { method: "DELETE" });
          await pollStatus(); noteBar("scanner", "Steering removed");
        } },
      { id: "trace", group: "Hand off", label: "Trace this token →", hint: "Open it in Trace: the layers, units and knockouts behind it",
        need: traceable,
        run: () => {
          document.dispatchEvent(new CustomEvent("brainscan:show", { detail: "weights" }));
          document.dispatchEvent(new CustomEvent("brainscan:trace", { detail: { word: S.selected } }));
        } },
    ],
  });
}

async function boot() {
  setupTheme(); readColors(); setupScopeInput(); setupSteering(); setupBar();
  document.addEventListener("brainscan:select", (ev) => { if (S.runId && ev.detail?.t != null) selectToken(ev.detail.t, true); });  // a flag picked
  document.addEventListener("brainscan:steering", pollStatus);  // steering changed elsewhere (a knob in Neurons)
  $("#run-select").onchange = (ev) => openRun(ev.target.value);
  // A message sent from a composer: open its run in every view as soon as it appears (ui/chat.js).
  document.addEventListener("brainscan:expect-run", (ev) => {
    if (ev.detail === S.runId) return;  // continuing the open conversation
    S.expectRun = ev.detail;
    if (S.runs.some((r) => r.id === ev.detail)) { S.expectRun = null; openRun(ev.detail); }
  });
  connect();  // subscribe before loading snapshots so nothing recorded in between is lost
  await pollStatus();
  await loadRuns();
  renderLegend();
  if (S.runs.length) await openRun(S.runs[0].id); else { renderTranscript(); schedule(); }
  setInterval(pollStatus, 2500);
}
boot();
