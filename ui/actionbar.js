// Actions and action bars. An action is something you do to the model (ablate, scale, steer, probe,
// compare, check: ui/verbs.js), picked in the top bar. It runs as steps, each on one screen (Watch,
// Trace, Tune), and the bar at the top of the screen on show draws the action's steps, what to do in
// this step, the tools it needs and the next step. Explore is no action: each screen's bar shows all
// of its tools, grouped. A tool that needs something first stays visible, greyed, and says what it
// needs when hovered or clicked.
//
// Each screen's module registers its tools here (defineBar), with a state() the steps read to mark
// themselves done, and calls refreshBar when its state changes. The same tools feed the Do… palette
// (⌘K or Ctrl+K), which searches every tool and action and runs one, switching screens if needed.

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const TAB = { scanner: "Watch", weights: "Trace", workbench: "Tune", neurons: "Neurons" };
const TONE = { Test: "ex", Steer: "ex", Change: "ch", Verify: "vf", Ship: "vf", Monitor: "vf" };
const SAVED = "brainscan-action";

const BARS = new Map();   // tab -> { groups, context, actions, states, el, sig, note }
const PENDING = new Map(); // tab -> tools other modules added before the screen defined its bar
let view = null;          // the screen on show
// What the edit panels report from any screen: the edit stack and the behavior checks (ui/edits.js).
const SHARED = { applied: 0, edits: 0, checks: 0, checked: false };
const V = { list: [], planned: [], cur: null, step: 0 };  // the actions, the one picked, its step

/** A tool's state: null when it can run now, or what it needs first. */
const needs = (a) => { try { return a.need ? a.need() : null; } catch { return "Not available yet"; } };
const labelOf = (a) => (typeof a.label === "function" ? a.label() : a.label);
const textOf = (x) => { try { return typeof x === "function" ? x() : x || ""; } catch { return ""; } };
const isDone = (s) => { try { return !!s.done?.(); } catch { return false; } };
/** "tab:id" → the tool and the screen it belongs to. */
const toolOf = (ref) => { const [tab, id] = ref.split(":"); return { tab, a: BARS.get(tab)?.actions.find((x) => x.id === id) }; };
const show = (tab) => document.dispatchEvent(new CustomEvent("brainscan:show", { detail: tab }));
const runOn = (tab, run) => { if (tab !== view) { show(tab); setTimeout(run, 120); } else run(); };

/** What a screen reports through its bar's state(), or the shared edit and check state. */
export function st(key) {
  if (key in SHARED) return SHARED[key];
  for (const bar of BARS.values()) {
    for (const state of bar.states) {
      let s = null;
      try { s = state(); } catch { /* not ready */ }
      if (s && key in s) return s[key];
    }
  }
  return undefined;
}

/** spec: { groups: [names in order], context: () => html, state: () => ({...}),
 *          actions: [{ id, group, label, hint, need, run, choices }] }
 *  `choices` (optional) returns [{ label, run }]: the tool opens a menu of them. */
export function defineBar(tab, spec) {
  const bar = { ...spec, states: spec.state ? [spec.state] : [], el: document.querySelector(`[data-bar="${tab}"]`), sig: "", note: null };
  for (const extra of PENDING.get(tab) || []) merge(bar, extra);
  PENDING.delete(tab);
  BARS.set(tab, bar);
  if (bar.el) {
    bar.el.addEventListener("click", (ev) => onBarClick(tab, ev));
    // The bar on show is sticky; what sits or scrolls below it reads its height.
    new ResizeObserver(() => {
      if (bar.el.offsetParent) document.documentElement.style.setProperty("--actbar-h", `${bar.el.offsetHeight}px`);
    }).observe(bar.el);
  }
  refreshBar(tab);
}

/** Another module's tools on a screen's bar (spec: { groups, before, state, actions }, `before`
 * naming the group they go in front of), as if the screen had defined them. */
export function extendBar(tab, spec) {
  const bar = BARS.get(tab);
  if (!bar) { PENDING.set(tab, [...(PENDING.get(tab) || []), spec]); return; }
  merge(bar, spec);
  refreshBar(tab);
}
function merge(bar, spec) {
  const at = spec.before ? bar.groups.indexOf(spec.before) : -1;
  bar.groups = at < 0 ? [...bar.groups, ...spec.groups] : [...bar.groups.slice(0, at), ...spec.groups, ...bar.groups.slice(at)];
  bar.actions = [...bar.actions, ...spec.actions];
  if (spec.state) bar.states.push(spec.state);
}

/** Re-render a screen's bar if what it shows changed (cheap to call often). The bar on show is
 * redrawn too: its steps read every screen's state. */
export function refreshBar(tab) {
  render(tab);
  if (view && view !== tab) render(view);
}

/** A short message in a screen's bar, in place of its context line for a few seconds. */
export function noteBar(tab, text) { if (BARS.has(tab)) flashNote(tab, text); }

// ---------------------------------------------------------------- drawing a bar
/** How a screen's bar shows the picked action: every tool (explore), one step's tools (step), or
 * every tool under the action's steps when this screen isn't one of them (off). */
function modeOf(tab) {
  const verb = V.cur;
  if (!verb || verb.explore) return "explore";
  return verb.steps[V.step]?.view === tab ? "step" : "off";
}

function render(tab) {
  const bar = BARS.get(tab);
  if (!bar?.el) return;
  let ctx = "";
  try { ctx = bar.context(); } catch { /* not ready */ }
  const verb = V.cur, mode = modeOf(tab), step = mode === "step" ? verb.steps[V.step] : null;
  const refs = step ? [...(step.tools || []), ...(step.also || [])] : [];
  const tools = step ? refs.map((r) => { const { a } = toolOf(r); return a ? [r, needs(a), labelOf(a)] : [r]; })
    : bar.actions.map((a) => [a.id, needs(a), labelOf(a)]);
  const rail = verb ? verb.steps.map((s) => (verb.explore ? s.view : isDone(s))) : [];
  const sig = JSON.stringify([ctx, bar.note, verb?.id, V.step, mode, rail, step ? textOf(step.what) : "", tools]);
  if (sig === bar.sig) return;
  bar.sig = sig;
  const line = bar.note ? `<span class="ab-note">${esc(bar.note)}</span>` : ctx;
  if (!verb) { bar.el.innerHTML = `<p class="ab-ctx">${line}</p>${groupsHtml(bar)}`; return; }
  let body;
  if (mode === "explore") body = groupsHtml(bar);
  else if (mode === "step") body = stepHtml(tab, verb, step);
  else {
    const back = verb.steps[V.step];
    body = `<p class="ab-what">${esc(TAB[tab])} isn't a step of ${esc(verb.label)}: its tools are below.
      <button type="button" class="link" data-step="${V.step}">Back to ${V.step + 1} · ${esc(back.label)} →</button></p>${groupsHtml(bar)}`;
  }
  bar.el.classList.toggle("in-step", mode === "step");
  bar.el.innerHTML = `${railHtml(tab, verb, mode, line)}${body}`;
}

function railHtml(tab, verb, mode, line) {
  const active = verb.explore ? verb.steps.findIndex((s) => s.view === tab) : mode === "step" ? V.step : -1;
  const steps = verb.steps.map((s, i) => {
    const done = !verb.explore && isDone(s), on = i === active;
    const title = verb.explore ? `Open ${s.label}` : `${s.label}, in ${TAB[s.view]}: ${textOf(s.what)}`;
    return `${i ? '<li class="ab-sep" aria-hidden="true">›</li>' : ""}<li><button type="button" class="ab-step${on ? " on" : ""}${done ? " done" : ""}"
      data-step="${i}" ${on ? 'aria-current="step"' : ""} title="${esc(title)}"><span class="n">${done ? "✓" : i + 1}</span>${esc(s.label)}${
      verb.explore ? "" : `<span class="ab-where">${TAB[s.view]}</span>`}</button></li>`;
  }).join("");
  return `<div class="ab-rail"><span class="ab-verb${verb.explore ? " explore" : ""}" title="${esc(verb.lead)}">${esc(verb.label)}</span>
    <ol class="ab-steps" aria-label="${esc(verb.label)} steps">${steps}</ol><p class="ab-ctx">${line}</p></div>`;
}

function stepHtml(tab, verb, step) {
  const next = verb.steps[V.step + 1];
  const btns = (refs) => refs.map((r) => toolBtn(tab, r)).join("");
  return `<p class="ab-what"><b>${V.step + 1} · ${esc(step.label)}.</b> ${esc(textOf(step.what))}</p>
    <div class="ab-groups"><div class="ab-group">${btns(step.tools || [])}</div>
      ${step.also?.length ? `<div class="ab-group"><span class="ab-gl">Also</span>${btns(step.also)}</div>` : ""}
      ${next ? `<button type="button" class="ab-next" data-next title="${esc(`${next.label}, in ${TAB[next.view]}: ${textOf(next.what)}`)}">Next: ${esc(next.label)} →</button>` : ""}</div>`;
}

/** A step's tool, which may belong to another screen (it says which, and switches there to run). */
function toolBtn(tab, ref) {
  const { tab: t, a } = toolOf(ref);
  if (!a) return "";
  const why = needs(a);
  return `<button type="button" class="ab-btn ${TONE[a.group] || ""}${why ? " off" : ""}" data-ref="${ref}" ${why ? 'aria-disabled="true"' : ""}
    title="${esc(why ? `Needs: ${why}` : a.hint || "")}">${esc(labelOf(a))}${a.choices ? " ▾" : ""}${t !== tab ? `<span class="ab-where">${TAB[t]}</span>` : ""}</button>`;
}

function groupsHtml(bar) {
  return `<div class="ab-groups">${bar.groups.map((g) => {
    const acts = bar.actions.filter((a) => a.group === g);
    return acts.length ? `<div class="ab-group"><span class="ab-gl">${esc(g)}</span>${acts.map((a) => {
      const why = needs(a);
      return `<button type="button" class="ab-btn ${TONE[g] || ""}${why ? " off" : ""}" data-id="${a.id}" ${why ? 'aria-disabled="true"' : ""}
        title="${esc(why ? `Needs: ${why}` : a.hint || "")}">${esc(labelOf(a))}${a.choices ? " ▾" : ""}</button>`;
    }).join("")}</div>` : "";
  }).join("")}</div>`;
}

function flashNote(tab, text) {
  const bar = BARS.get(tab);
  bar.note = text;
  render(tab);
  clearTimeout(bar.noteTimer);
  bar.noteTimer = setTimeout(() => { bar.note = null; render(tab); }, 3500);
}

function onBarClick(tab, ev) {
  const step = ev.target.closest("[data-step]");
  if (step) { goStep(Number(step.dataset.step)); return; }
  if (ev.target.closest("[data-next]")) { goStep(V.step + 1); return; }
  const btn = ev.target.closest(".ab-btn[data-ref], .ab-btn[data-id]");
  if (!btn) return;
  const { tab: t, a } = btn.dataset.ref ? toolOf(btn.dataset.ref) : { tab, a: BARS.get(tab).actions.find((x) => x.id === btn.dataset.id) };
  if (!a) return;
  const why = needs(a);
  if (why) { flashNote(tab, `Needs: ${why}`); return; }
  if (a.choices) { openMenu(btn, a.choices().map((c) => ({ label: c.label, run: () => runOn(t, c.run) }))); return; }
  runOn(t, a.run);
}

// ---------------------------------------------------------------- the actions and their steps
/** ui/verbs.js: the actions (in order; `group` sorts them in the top bar), and the planned ones. */
export function defineVerbs(list, planned = []) {
  V.list = list; V.planned = planned;
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(SAVED) || "null"); } catch { /* storage unavailable */ }
  V.cur = list.find((v) => v.id === saved?.verb) || list[0];
  V.step = Math.min(Math.max(0, Number(saved?.step) || 0), V.cur.steps.length - 1);
  if (view) alignStep();
  const nav = $("#verbs");
  if (nav) nav.addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-verb]");
    if (b) pick(b.dataset.verb);
    else if (ev.target.closest("[data-more]")) openPlanned(ev.target.closest("[data-more]"));
  });
  renderPicker();
  if (BARS.has(view)) render(view);
}

/** The actions and the planned ones, for the guide. */
export const actionList = () => ({ list: V.list, planned: V.planned });

function save() { try { localStorage.setItem(SAVED, JSON.stringify({ verb: V.cur?.id, step: V.step })); } catch { /* storage unavailable */ } }

/** Pick an action: start at its first step that isn't done yet (Explore stays on this screen). */
function pick(id) {
  const verb = V.list.find((v) => v.id === id);
  if (!verb) return;
  V.cur = verb;
  if (verb.explore) V.step = Math.max(0, verb.steps.findIndex((s) => s.view === view));
  else { const i = verb.steps.findIndex((s) => !isDone(s)); V.step = i < 0 ? verb.steps.length - 1 : i; }
  renderPicker();
  goStep(V.step);
}

function goStep(i) {
  const verb = V.cur;
  if (!verb || i < 0 || i >= verb.steps.length) return;
  V.step = i;
  save();
  const s = verb.steps[i];
  if (s.view !== view) show(s.view);
  else if (BARS.has(view)) render(view);
  // A step on a long screen (Tune, Watch's steering) brings its panel into view.
  if (s.focus) setTimeout(() => $(s.focus)?.scrollIntoView({ behavior: "smooth", block: "start" }), 140);
}

/** The screen changed some other way (a hand-off, Do…, the guide): follow it to a step on that
 * screen, the first not done yet. */
function alignStep() {
  const verb = V.cur;
  if (!verb || verb.steps[V.step]?.view === view) return;
  const here = verb.steps.map((s, i) => [s, i]).filter(([s]) => s.view === view);
  if (!here.length) return;  // not a step of this action: the bar says so
  V.step = (here.find(([s]) => !isDone(s)) || here[0])[1];
  save();
}

function renderPicker() {
  const nav = $("#verbs");
  if (!nav) return;
  const btn = (v) => `<button type="button" class="verb${v === V.cur ? " on" : ""}${v.explore ? " explore" : ""}" data-verb="${v.id}"
    aria-pressed="${v === V.cur}" title="${esc(v.lead)}">${esc(v.label)}</button>`;
  const groups = [...new Set(V.list.filter((v) => v.group).map((v) => v.group))];
  nav.innerHTML = V.list.filter((v) => !v.group).map(btn).join("")
    + groups.map((g) => `<span class="verbs-g">${esc(g)}</span>${V.list.filter((v) => v.group === g).map(btn).join("")}`).join("")
    + (V.planned.length ? `<button type="button" class="verb more" data-more aria-haspopup="true" title="Actions that aren't built yet">More ▾</button>` : "");
}

function openPlanned(anchor) {
  openPopover(anchor, `<p class="vm-head">Planned, not built yet</p>${V.planned.map((p) => `<div class="vm-row"><b>${esc(p.label)}</b><span>${esc(p.what)}</span></div>`).join("")}`,
    "verbs-more", () => closeMenu());
}

// ---------------------------------------------------------------- a small menu for tools with choices
function openMenu(anchor, items) {
  if (!items.length) return;
  openPopover(anchor, items.map((it, i) => `<button type="button" data-i="${i}">${esc(it.label)}</button>`).join(""), "", (ev) => {
    const b = ev.target.closest("[data-i]");
    if (b) { closeMenu(); items[Number(b.dataset.i)].run(); }
  });
}
function openPopover(anchor, html, cls, onClick) {
  closeMenu();
  const r = anchor.getBoundingClientRect();
  const menu = document.createElement("div");
  menu.className = `ab-menu glass ${cls}`;
  menu.id = "ab-menu";
  menu.innerHTML = html;
  menu.onclick = onClick;
  document.body.appendChild(menu);
  menu.style.left = `${Math.max(8, Math.min(r.left, innerWidth - menu.offsetWidth - 8))}px`;
  menu.style.top = `${r.bottom + 4}px`;
  setTimeout(() => document.addEventListener("pointerdown", outside, { once: true }), 0);
}
const outside = (ev) => { if (!ev.target.closest("#ab-menu")) closeMenu(); else document.addEventListener("pointerdown", outside, { once: true }); };
function closeMenu() { $("#ab-menu")?.remove(); }

// ---------------------------------------------------------------- the Do… palette
const D = { items: [], active: 0 };

function allItems() {
  const out = V.list.map((v) => ({ verb: v.id, group: "action", label: v.explore ? "Explore (no action)" : `${v.label} (action)`, why: null, hint: v.lead }));
  for (const [tab, bar] of BARS) {
    for (const a of bar.actions) {
      const why = needs(a), label = labelOf(a);
      if (a.choices && !why) {
        for (const c of a.choices()) out.push({ tab, group: a.group, label: `${label}: ${c.label}`, why: null, run: c.run });
      } else {
        out.push({ tab, group: a.group, label, why, run: a.run, hint: a.hint });
      }
    }
  }
  return out;
}

function openDo() {
  if (!$("#do-palette")) return;
  closeMenu();
  $("#do-palette").hidden = false;
  const q = $("#do-q");
  q.value = "";
  q.focus();
  searchDo();
}
function closeDo() { const p = $("#do-palette"); if (p) p.hidden = true; }

function searchDo() {
  const q = $("#do-q").value.trim().toLowerCase();
  const words = q.split(/\s+/).filter(Boolean);
  const items = allItems().filter((it) => {
    const hay = `${it.label} ${it.group} ${TAB[it.tab] || ""}`.toLowerCase();
    return words.every((w) => hay.includes(w));
  });
  // The screen on show first, then what can run now.
  items.sort((a, b) => (b.tab === view) - (a.tab === view) || (!!a.why) - (!!b.why));
  // A tool two screens share (Restore original, Re-run the chat) is listed once, from the nearest screen.
  const seen = new Set();
  D.items = items.filter((it) => !seen.has(it.label) && seen.add(it.label));
  D.active = Math.min(D.active, Math.max(0, D.items.length - 1));
  if (!q) D.active = 0;
  renderDo();
}

function renderDo() {
  const list = $("#do-list");
  list.innerHTML = D.items.map((it, i) => `<li role="option" data-i="${i}" class="${i === D.active ? "active" : ""}${it.why ? " off" : ""}" aria-selected="${i === D.active}">
      <span class="do-label">${esc(it.label)}</span>
      <span class="do-meta">${it.why ? `<span class="do-why">needs: ${esc(it.why)}</span>` : ""}<span class="do-where">${it.verb ? "pick" : `${TAB[it.tab]} · ${esc(it.group.toLowerCase())}`}</span></span></li>`).join("")
    || `<li class="empty">Nothing matches. Try “ablate”, “knock”, “check” or “export”.</li>`;
  list.querySelector("li.active")?.scrollIntoView({ block: "nearest" });
}

function runDo(i) {
  const it = D.items[i];
  if (!it) return;
  if (it.why) { $("#do-note").textContent = `${it.label} needs: ${it.why}`; return; }
  closeDo();
  if (it.verb) pick(it.verb);
  else runOn(it.tab, it.run);
}

function boot() {
  document.addEventListener("brainscan:pick", (ev) => pick(ev.detail));  // from the guide
  document.addEventListener("brainscan:view", (ev) => {
    view = ev.detail;
    // The forget-a-fact flow is ablation, guided: it has its own steps.
    if (view === "actions") {
      const ablate = V.list.find((v) => v.id === "ablate");
      if (ablate && V.cur !== ablate) { V.cur = ablate; V.step = 1; save(); renderPicker(); }
      return;
    }
    alignStep();
    if (BARS.has(view)) render(view);
  });
  document.addEventListener("brainscan:edit-state", (ev) => {
    const d = ev.detail || {};
    for (const k of Object.keys(SHARED)) if (k in d) SHARED[k] = d[k];
    if (BARS.has(view)) render(view);
  });
  $("#do-btn")?.addEventListener("click", openDo);
  const q = $("#do-q");
  if (q) {
    q.addEventListener("input", () => { $("#do-note").textContent = ""; searchDo(); });
    q.addEventListener("keydown", (ev) => {
      if (ev.key === "ArrowDown") { D.active = Math.min(D.items.length - 1, D.active + 1); renderDo(); ev.preventDefault(); }
      else if (ev.key === "ArrowUp") { D.active = Math.max(0, D.active - 1); renderDo(); ev.preventDefault(); }
      else if (ev.key === "Enter") { runDo(D.active); ev.preventDefault(); }
      else if (ev.key === "Escape") { closeDo(); ev.preventDefault(); }
      ev.stopPropagation();
    });
    $("#do-list").addEventListener("click", (ev) => { const li = ev.target.closest("li[data-i]"); if (li) runDo(Number(li.dataset.i)); });
    $("#do-palette").addEventListener("pointerdown", (ev) => { if (ev.target.id === "do-palette") closeDo(); });
  }
  // ⌘K / Ctrl+K anywhere opens Do…
  document.addEventListener("keydown", (ev) => {
    if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "k") {
      ev.preventDefault(); ev.stopImmediatePropagation();
      if ($("#do-palette")?.hidden) openDo(); else closeDo();
    }
  }, true);
}
boot();
