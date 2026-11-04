// Unmask: what a model on the Hugging Face Hub really is, read without downloading it (llminspect/hub.py).
// The hero is a card, drawn on a canvas at 1200×630 so the picture saved is the picture shown: the headline,
// the model's lineage (each parent its card claims, and whether the weights agree), and one bar per layer lit
// by how far that layer moved from the parent, with the sampled layers whose change lies in a few directions
// marked, and threaded together when they share them. The card stays dark in both themes, like a print.
//
// Colors mean one thing each: teal, a parent the weights confirm; amber, what was edited; red, a false claim
// or code that runs on load.
//
// API: GET /llminspect/unmask?model=ORG/NAME
import { defineBar, refreshBar, noteBar } from "/ui/actionbar.js";

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const KEEP = "llminspect-unmask";
const W = 1200, H = 630, PAD = 56;
const C = { bg: "#07080d", ink: "#f4f5f8", ink2: "#a9afbf", muted: "#666d80", faint: "#1b1f2b", line: "#2a2f3d",
  teal: "#22c7d6", amber: "#f5a524", red: "#f05252", violet: "#8b7dff" };
const U = { model: "", r: null, state: "empty", error: "", t0: 0, raf: 0, recent: [] };
const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const PUBLIC = !!document.body.dataset.public;  // the public page (llminspect/public.py)

try { U.recent = JSON.parse(localStorage.getItem(KEEP) || "[]"); } catch { U.recent = []; }
const keep = () => { try { localStorage.setItem(KEEP, JSON.stringify(U.recent.slice(0, 6))); } catch { /* storage unavailable */ } };

// ---------------------------------------------------------------- reading the result

/** How far a layer moved, 0..1, from its correlation with the parent (1 − corr on a log scale). */
function moved(corr) {
  if (corr == null || corr >= 0.99999) return 0;
  return Math.max(0.3, Math.min(1, (Math.log10(1 - corr) + 5) / 4.5));
}

/** In a quantized model rounding moves every layer's fingerprint, so a layer's height comes from the change
 *  measured above the rounding (its size relative to the parent's rows, on a log scale); unmeasured layers are 0. */
function changed(p) {
  if (!p || !(p.rel > 0)) return 0;
  return Math.max(0.3, Math.min(1, (Math.log10(p.rel) + 3.5) / 3));
}

/** The parents to draw, oldest first, ending with the parent the edge into the model comes from. */
function chain(r) {
  const firsts = [];
  for (const row of r.lineage) if (!firsts.some((x) => x.depth === row.depth)) firsts.push(row);
  firsts.sort((a, b) => b.depth - a.depth);
  const p = r.parent, copy = r.copy_of?.model;
  const claimRow = (name) => (r.claims || []).find((c) => c.model === name);
  const nodes = firsts.map((row) => ({ name: row.model, depth: row.depth, claim: claimRow(row.model), error: row.error, copy: row.model === copy }));
  if (p && !p.claimed) {  // the weights point elsewhere: the real parent sits next to the model, the claim off to the side
    const false_ = nodes.filter((n) => n.depth === 1);
    return { nodes: [{ name: p.model, depth: 1, real: true }], aside: false_ };
  }
  return { nodes, aside: [] };
}

const short = (name, n = 26) => (name.length > n ? `${name.slice(0, n - 1)}…` : name);
const split = (name) => { const i = name.indexOf("/"); return i < 0 ? ["", name] : [name.slice(0, i + 1), name.slice(i + 1)]; };
const mb = (b) => (b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : `${(b / 1e6).toFixed(1)} MB`);

function risk(r) {
  const top = r.files?.findings?.[0]?.level;
  return ["runs code on load", "imports other code on load", "runs code if trusted"].includes(top) ? top : null;
}
const addsText = (r) => (r.files?.template || []).some((t) => t.added_text?.length);

// ---------------------------------------------------------------- the card

function wrap(ctx, text, width) {
  const words = text.split(" "), lines = [];
  let line = "";
  for (const w of words) {
    const t = line ? `${line} ${w}` : w;
    if (ctx.measureText(t).width > width && line) { lines.push(line); line = w; } else line = t;
  }
  if (line) lines.push(line);
  return lines;
}

function backdrop(ctx, r) {
  ctx.fillStyle = C.bg;
  ctx.fillRect(0, 0, W, H);
  const glow = (x, y, rad, color, a) => {
    const g = ctx.createRadialGradient(x, y, 0, x, y, rad);
    g.addColorStop(0, color + a); g.addColorStop(1, color + "00");
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
  };
  glow(180, 40, 520, C.violet, "26");
  glow(W - 160, H - 60, 560, r?.edit && r.edit.kind !== "same" ? C.amber : C.teal, "1c");
  ctx.fillStyle = "rgba(255,255,255,0.045)";
  for (let x = 24; x < W; x += 24) for (let y = 24; y < H; y += 24) ctx.fillRect(x, y, 1, 1);
}

function chip(ctx, x, y, text, color, right = false) {
  ctx.font = `600 12px ${fonts().mono}`;
  const w = ctx.measureText(text).width + 22, x0 = right ? x - w : x;
  ctx.fillStyle = color + "1f"; ctx.strokeStyle = color + "66"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.roundRect(x0, y - 15, w, 24, 12); ctx.fill(); ctx.stroke();
  ctx.fillStyle = color; ctx.textAlign = "left"; ctx.fillText(text, x0 + 11, y + 1);
  return right ? x0 - 8 : x0 + w + 8;
}

let FONTS = null;
function fonts() {
  if (!FONTS) {
    const cs = getComputedStyle(document.documentElement);
    FONTS = { sans: cs.getPropertyValue("--sans").trim() || "system-ui, sans-serif", mono: cs.getPropertyValue("--mono").trim() || "monospace" };
  }
  return FONTS;
}

function header(ctx, r, title, sub) {
  const f = fonts();
  ctx.textBaseline = "alphabetic"; ctx.textAlign = "left";
  ctx.font = `600 12px ${f.mono}`; ctx.fillStyle = C.muted;
  ctx.fillText("U N M A S K E D", PAD, 62);
  ctx.textAlign = "right"; ctx.fillText("L L M I N S P E C T", W - PAD, 62);
  ctx.textAlign = "left";
  ctx.font = `500 21px ${f.mono}`; ctx.fillStyle = C.ink2;
  ctx.fillText(short(sub, 64), PAD, 106);
  // the largest size that fits in three lines
  let size = 36, lines;
  for (; size >= 24; size -= 3) {
    ctx.font = `650 ${size}px ${f.sans}`;
    lines = wrap(ctx, title, W - 2 * PAD);
    if (lines.length <= 3) break;
  }
  lines = lines.slice(0, 3);
  const lh = Math.round(size * 1.22);
  // what was done to the model in amber, the rest in white
  const lead = r?.edit && r.edit.kind !== "same" ? title.indexOf(r.edit.headline) : -1;
  const end = lead + (r?.edit?.headline.length ?? 0);
  let at = 0;
  lines.forEach((l, i) => {
    let x = PAD;
    for (const word of l.split(" ")) {
      // the part of the word inside the edit's span in amber (so a trailing ";" stays white)
      const hot = lead >= 0 && at >= lead ? Math.max(0, Math.min(word.length, end - at)) : 0;
      ctx.fillStyle = C.amber; ctx.fillText(word.slice(0, hot), x, 152 + i * lh);
      ctx.fillStyle = C.ink; ctx.fillText(word.slice(hot), x + ctx.measureText(word.slice(0, hot)).width, 152 + i * lh);
      x += ctx.measureText(`${word} `).width;
      at += word.length + 1;
    }
  });
  // what else the files say, as chips under the headline
  let x = PAD;
  const y = 152 + (lines.length - 1) * lh + 36;
  if (r && risk(r)) x = chip(ctx, x, y, `files: ${risk(r)}`, C.red);
  if (r && addsText(r)) x = chip(ctx, x, y, "chat template adds text", C.amber);
  return y + 18;
}

function node(ctx, x, y, rad, fill, ring, glowColor) {
  if (glowColor) { ctx.shadowColor = glowColor; ctx.shadowBlur = 24; }
  ctx.fillStyle = fill; ctx.beginPath(); ctx.arc(x, y, rad, 0, Math.PI * 2); ctx.fill();
  ctx.shadowBlur = 0;
  if (ring) { ctx.strokeStyle = ring; ctx.lineWidth = 2; ctx.stroke(); }
}

function label(ctx, x, y, name, color, align = "center") {
  const f = fonts(), [org, base] = split(name);
  ctx.textAlign = align;
  ctx.font = `500 11.5px ${f.mono}`; ctx.fillStyle = C.muted;
  ctx.fillText(short(org, 24), x, y);
  ctx.font = `600 13.5px ${f.mono}`; ctx.fillStyle = color;
  ctx.fillText(short(base, 28), x, y + 18);
}

function edge(ctx, x0, y0, x1, y1, color, dashed, text) {
  ctx.strokeStyle = color; ctx.lineWidth = dashed ? 1.5 : 2.5;
  ctx.setLineDash(dashed ? [6, 6] : []);
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  ctx.setLineDash([]);
  if (text) {
    ctx.font = `500 11px ${fonts().mono}`; ctx.fillStyle = color; ctx.textAlign = "center";
    ctx.fillText(text, (x0 + x1) / 2, (y0 + y1) / 2 - 10);
  }
}

function lineage(ctx, r, y) {
  const { nodes, aside } = chain(r);
  const model = { x: W - PAD - 150, y };
  const gap = Math.min(250, (model.x - PAD - 90) / Math.max(1, nodes.length));
  nodes.forEach((n, i) => { n.x = model.x - gap * (nodes.length - i); n.y = y; });
  // the claimed chain above the parent: grey, what cards say
  for (let i = 0; i < nodes.length - 1; i++) edge(ctx, nodes[i].x + 10, y, nodes[i + 1].x - 10, y, C.line, true, "card says");
  const p = r.parent, last = nodes[nodes.length - 1];
  if (last) {
    const ok = (p && p.model === last.name && p.median > 0.9) || last.copy;
    const unread = last.claim?.unreadable || last.error;
    const color = ok ? C.teal : unread ? C.muted : C.red;
    const text = last.copy ? `same weights${r.quant ? ` · ${r.quant.type}` : ""}`
      : ok ? `weights agree · ${p.median.toFixed(3)}` : unread ? "couldn't be read" : p ? "weights disagree" : "no match";
    edge(ctx, last.x + 12, y, model.x - 18, y, color, !ok, text);
  }
  for (const n of nodes) {
    const isParent = p && n.name === p.model && p.median > 0.9;
    node(ctx, n.x, y, isParent ? 9 : 6, isParent ? C.teal : "#2c3242", isParent ? null : n.copy ? C.teal : "#3b4256", isParent ? C.teal : null);
    label(ctx, n.x, y + 30, n.name, isParent || n.copy ? C.ink : C.ink2);
  }
  // a claim the weights disprove: off to the side, struck through in red
  aside.forEach((n, i) => {
    const ax = PAD + 70 + i * 200, ay = y - 62;
    edge(ctx, ax + 8, ay + 6, model.x - 16, y - 8, C.red, true, null);
    node(ctx, ax, ay, 6, C.red + "55", C.red, null);
    ctx.font = `600 11px ${fonts().mono}`; ctx.fillStyle = C.red; ctx.textAlign = "left";
    ctx.fillText(`claimed ${short(n.name, 30)} ✕`, ax + 14, ay + 4);
  });
  node(ctx, model.x, y, 13, C.violet, "#ffffff", C.violet);
  label(ctx, model.x, y + 34, r.model, C.ink);
  if (!nodes.length && !aside.length) {
    ctx.font = `500 13px ${fonts().mono}`; ctx.fillStyle = C.muted; ctx.textAlign = "right";
    ctx.fillText("its card names no parent", model.x - 34, y + 4);
  }
}

function layers(ctx, r, top, phase) {
  const f = fonts(), x0 = PAD, x1 = W - PAD, base = top + 106;
  const corr = r?.parent?.layers || {};
  const ids = Object.keys(corr).map(Number).filter((l) => l >= 0).sort((a, b) => a - b);
  const n = ids.length || 32;
  const step = (x1 - x0) / n, bw = Math.max(3, Math.min(16, step * 0.62));
  const rank = r?.edit?.rank || { layers: {}, few: [] };
  ctx.textAlign = "left"; ctx.font = `600 11px ${f.mono}`; ctx.fillStyle = C.muted;
  const above = r?.quant ? ` ABOVE ${r.quant.type} ROUNDING` : "";
  ctx.fillText(r?.parent ? `${n} LAYERS · HOW FAR EACH MOVED FROM ${short(r.parent.model, 40).toUpperCase()}${above}` : `LAYERS`, x0, top);
  if (r?.edit) chip(ctx, x1, top - 4, r.edit.kind.replace("_", " "), r.edit.kind === "same" ? C.teal : C.amber, true);
  const dots = [];
  for (let i = 0; i < n; i++) {
    const l = ids[i], m = r ? (r.quant ? changed(rank.layers[l]) : moved(corr[l])) : 0.08 + 0.5 * Math.abs(Math.sin(i * 1.7 + phase));
    const cx = x0 + step * (i + 0.5), h = 6 + m * 66;
    if (r && m > 0) {
      ctx.shadowColor = C.amber; ctx.shadowBlur = 14 * m;
      const g = ctx.createLinearGradient(0, base - h, 0, base);
      g.addColorStop(0, C.amber); g.addColorStop(1, C.amber + "33");
      ctx.fillStyle = g;
    } else ctx.fillStyle = r ? "#2a2f3f" : `rgba(255,255,255,${0.05 + 0.08 * m})`;
    ctx.beginPath(); ctx.roundRect(cx - bw / 2, base - h, bw, h, Math.min(4, bw / 2)); ctx.fill();
    ctx.shadowBlur = 0;
    if (r && rank.few.includes(l)) dots.push([cx, base - h - 12]);
  }
  if (dots.length) {
    if (r.edit.kind === "direction_removed" && dots.length > 1) {  // the same directions through every edited layer
      ctx.strokeStyle = "#fff4d6"; ctx.lineWidth = 1.5; ctx.shadowColor = C.amber; ctx.shadowBlur = 10;
      ctx.beginPath(); ctx.moveTo(...dots[0]);
      for (let i = 1; i < dots.length; i++) {
        const [ax, ay] = dots[i - 1], [bx, by] = dots[i];
        ctx.bezierCurveTo(ax + (bx - ax) / 2, ay - 18, ax + (bx - ax) / 2, by - 18, bx, by);
      }
      ctx.stroke(); ctx.shadowBlur = 0;
    }
    for (const [x, y] of dots) node(ctx, x, y, 3.5, "#fff4d6", null, C.amber);
  }
  if (r) {
    ctx.font = `500 11px ${f.mono}`; ctx.fillStyle = C.muted; ctx.textAlign = "left";
    ctx.fillText("L0", x0, base + 18); ctx.textAlign = "right"; ctx.fillText(`L${ids[n - 1] ?? n - 1}`, x1, base + 18);
    if (dots.length) {
      ctx.textAlign = "center"; ctx.fillStyle = C.ink2;
      const shared = r.edit.kind === "direction_removed" ? `, the same in each (overlap ${(rank.shared ?? 0).toFixed(2)})` : "";
      const ks = rank.few.map((l) => rank.layers[l]?.k || 1);
      const howMany = Math.max(...ks) === 1 ? "one direction" : `${Math.min(...ks)}–${Math.max(...ks)} directions`.replace(/^(\d)–\1 /, "$1 ");
      ctx.fillText(`● sampled layers whose change lies in ${howMany}${shared}`, W / 2, base + 18);
    }
  }
  return base + 18;
}

function footer(ctx, r) {
  ctx.font = `500 12px ${fonts().mono}`; ctx.fillStyle = C.muted; ctx.textAlign = "left";
  const size = r.files?.bytes ? ` of ${mb(r.files.bytes)}` : "";
  ctx.fillText(`read ${mb(r.bytes_read)}${size} · ${r.seconds} s · nothing downloaded, loaded or run`, PAD, H - 30);
  ctx.textAlign = "right";
  ctx.fillText(r.sha ? `revision ${r.sha.slice(0, 10)}` : "local folder", W - PAD, H - 30);
}

function draw(phase = 0) {
  const cv = $("#um-card");
  if (!cv) return;
  const dpr = 2;
  if (cv.width !== W * dpr) { cv.width = W * dpr; cv.height = H * dpr; }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const r = U.state === "done" ? U.r : null;
  backdrop(ctx, r);
  if (r) {
    const y = header(ctx, r, r.headline, r.quant ? `${r.model} · ${r.quant.type}` : r.model);
    lineage(ctx, r, Math.max(y + 52, 322));
    layers(ctx, r, 432, 0);
    footer(ctx, r);
  } else {
    const title = U.state === "loading" ? "Reading its weights by byte range…"
      : U.state === "error" ? `Couldn't read ${U.model}` : "Who made this model, and what did they change?";
    const sub = U.state === "loading" ? U.model : U.state === "error" ? U.error : "a model id from the Hugging Face Hub";
    header(ctx, null, title, sub);
    ctx.globalAlpha = 0.5;
    [PAD + 120, PAD + 400, W - PAD - 150].forEach((x, i) => node(ctx, x, 360, i === 2 ? 12 : 6, i === 2 ? C.violet + "88" : "#262b39", null, null));
    edge(ctx, PAD + 130, 360, PAD + 390, 360, C.line, true, null);
    edge(ctx, PAD + 412, 360, W - PAD - 168, 360, C.line, true, null);
    ctx.globalAlpha = 1;
    layers(ctx, null, 432, phase);
    if (U.state === "loading") {
      ctx.font = `500 12px ${fonts().mono}`; ctx.fillStyle = C.muted; ctx.textAlign = "left";
      ctx.fillText(`${((performance.now() - U.t0) / 1000).toFixed(1)} s · reading by byte range, nothing downloaded`, PAD, H - 30);
    }
  }
  cv.setAttribute("aria-label", r ? `${r.model}: ${r.headline}` : "Unmask card: no model read yet");
}

function animate() {
  cancelAnimationFrame(U.raf);
  if (U.state !== "loading") { draw(); return; }
  const tick = (t) => { draw(reduced() ? 0 : t / 600); if (U.state === "loading") U.raf = requestAnimationFrame(tick); };
  U.raf = requestAnimationFrame(tick);
}

// ---------------------------------------------------------------- the details under the card

function details() {
  const el = $("#um-details"), r = U.state === "done" ? U.r : null;
  $("#um-share").hidden = !r;
  if (!r) { el.innerHTML = ""; return; }
  const claims = (r.claims || []).map((c) => {
    const tone = /^(derived from it|the same weights|close:)/.test(c.reading) ? "ok" : c.unreadable ? "" : "bad";
    return `<li><span class="um-k">${esc(c.claimed)}</span><code>${esc(c.model)}</code><span class="um-v ${tone}">${esc(c.unreadable ? "couldn't be read" : c.reading)}${c.median != null ? ` · ${c.median.toFixed(4)}` : ""}</span></li>`;
  }).join("") || `<li class="um-none">Its card names no parent.</li>`;
  const copy = r.copy_of ? `<p class="um-line"><code>${esc(r.copy_of.model)}</code> holds the same weights${r.quant ? ` (within ${esc(r.quant.type)} rounding)` : ""}, so the edit is measured against the model it was made from.</p>` : "";
  const parent = copy + (r.parent ? `<p class="um-line">Compared with <code>${esc(r.parent.model)}</code>${r.parent.claimed ? "" : ` (found ${r.parent.found === "the Hub" ? "on the Hub by name" : "on this machine"})`}: median correlation ${r.parent.median.toFixed(4)}.</p>` : "");
  const rank = r.edit ? Object.entries(r.edit.rank.layers).map(([l, p]) =>
    `<tr><td>L${l}</td><td>${(p.rel * 100).toFixed(2)}%</td><td>${p.top1 == null ? "–" : `${(p.top1 * 100).toFixed(1)}%`}</td><td>${p.top8 == null ? "–" : `${(p.top8 * 100).toFixed(1)}%`}</td><td>${p.k || "–"}</td></tr>`).join("") : "";
  const files = (r.files.findings || []).slice(0, 6).map((f) =>
    `<li><span class="um-v ${f.level === "info" ? "" : "bad"}">${esc(f.level)}</span> <code>${esc(f.file)}</code> ${esc(f.detail)}</li>`).join("")
    || `<li class="um-none">Nothing in its configs, code or templates runs on load.</li>`;
  const tmpl = (r.files.template || []).filter((t) => t.added_text?.length).map((t) =>
    `<li>Template <code>${esc(t.name)}</code> adds: ${t.added_text.slice(0, 4).map((s) => `<q>${esc(s.slice(0, 160))}</q>`).join(" ")}</li>`).join("");
  el.innerHTML = `
    <section><h3>Lineage</h3><ul class="um-list">${claims}</ul>${parent}</section>
    <section><h3>The edit</h3>${r.edit ? `<p class="um-line">${esc(r.edit.headline)}.</p>
      <table class="um-table"><thead><tr><th>Layer</th><th title="Size of the change, relative to the parent's rows">Change</th><th title="Share of the change in its top direction">Top 1</th><th title="Share of the change in its top 8 directions">Top 8</th><th title="Directions holding 90% of the change">Directions</th></tr></thead><tbody>${rank}</tbody></table>`
      : `<p class="um-line um-none">No parent close enough to measure the edit against.</p>`}</section>
    <section><h3>Files</h3><p class="um-line">${r.files.count} files, ${mb(r.files.bytes)}, ${esc(r.files.formats.join(" ") || "no weights")}.</p><ul class="um-list">${files}${tmpl}</ul></section>
    <section><h3>Not checked</h3><ul class="um-list">${(r.not_checked || []).map((x) => `<li class="um-none">${esc(x)}</li>`).join("")}</ul></section>`;
}

function recent() {
  $("#um-recent").innerHTML = U.recent.map((m) => `<button type="button" class="um-chip" data-m="${esc(m)}">${esc(short(m, 40))}</button>`).join("");
}

// ---------------------------------------------------------------- running it

async function run(model) {
  model = (model ?? $("#um-input").value).trim().replace(/^https?:\/\/huggingface\.co\//, "").replace(/\/+$/, "");
  if (!model || U.state === "loading") return;
  $("#um-input").value = model;
  Object.assign(U, { model, state: "loading", r: null, error: "", t0: performance.now() });
  $("#um-go").disabled = true;
  details(); animate(); refreshBar("unmask");
  try {
    const res = await fetch(`/llminspect/unmask?model=${encodeURIComponent(model)}`);
    const body = await res.json().catch(() => null);
    if (!res.ok) throw new Error(body?.detail || `${res.status}`);
    Object.assign(U, { r: body, state: "done" });
    if (PUBLIC) history.replaceState(null, "", `/m/${model}`);  // the address is the link to share
    U.recent = [model, ...U.recent.filter((m) => m !== model)];
    keep(); recent();
  } catch (err) {
    Object.assign(U, { state: "error", error: String(err.message || err).slice(0, 120) });
  } finally {
    $("#um-go").disabled = false;
    animate(); details(); refreshBar("unmask");
  }
}

function save() {
  if (!U.r) return;
  $("#um-card").toBlob((blob) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `unmask-${U.r.model.replace(/[^A-Za-z0-9._-]+/g, "-")}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }, "image/png");
}

async function copy() {
  if (!U.r) return;
  const link = PUBLIC ? `${location.origin}/m/${U.r.model}` : U.r.url;
  const text = `${U.r.model}: ${U.r.headline}${link ? `\n${link}` : ""}`;
  try { await navigator.clipboard.writeText(text); noteBar("unmask", "Copied"); } catch { noteBar("unmask", "Couldn't copy"); }
}

function boot() {
  $("#um-form").addEventListener("submit", (ev) => { ev.preventDefault(); run(); });
  $("#um-recent").addEventListener("click", (ev) => { const b = ev.target.closest("[data-m]"); if (b) run(b.dataset.m); });
  $("#um-save").addEventListener("click", save);
  $("#um-copy").addEventListener("click", copy);
  recent(); draw(); details();
  document.addEventListener("llminspect:view", (ev) => { if (ev.detail === "unmask") draw(); });
  defineBar("unmask", {
    groups: ["Read", "Share"],
    context: () => (U.state === "loading" ? `Reading <b>${esc(U.model)}</b>…` : U.r ? `<b>${esc(U.r.model)}</b>` : "No model read yet"),
    state: () => ({ unmasked: U.state === "done" }),
    actions: [
      { id: "run", group: "Read", label: "Unmask", hint: "Read a Hub model's parents and edit by byte range. Nothing is downloaded.",
        need: () => (U.state === "loading" ? "the read running now to finish" : $("#um-input").value.trim() ? null : "a model id typed at the top"),
        run: () => run() },
      { id: "save", group: "Share", label: "Save image", hint: "The card as a 2400×1260 PNG", need: () => (U.r ? null : "a model unmasked"), run: save },
      { id: "copy", group: "Share", label: "Copy summary", hint: "The headline and the model's link", need: () => (U.r ? null : "a model unmasked"), run: copy },
    ],
  });
  $("#um-input").addEventListener("input", () => refreshBar("unmask"));
  if (document.body.dataset.unmask) run(document.body.dataset.unmask);  // a shared /m/ORG/NAME link
}
boot();
