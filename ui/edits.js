// Weight edits next to the trace. One module renders, wherever it is mounted (the Trace tab's Edits
// panel, and Tune): the edit stack (apply, restore, toggle, remove), the open chat asked again on the
// edited weights, and behavior checks measured before → after. Mount points carry data-edits with
// the sections to show: "stack", "replay", "checks", or "all".

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const visTok = (s) => (s ?? "").replace(/\n/g, "⏎").replace(/\t/g, "⇥") || "∅";
const fmtP = (p) => (p == null ? "–" : p >= 0.001 || p === 0 ? p.toFixed(3) : p.toExponential(1));
async function api(path, opts) {
  const res = await fetch(path, opts);
  const text = await res.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch { /* not JSON */ }
  if (!res.ok) { const err = new Error(body?.detail || text || `${res.status}`); err.status = res.status; throw err; }
  return body;
}
const send = (method, path, body) => api(path, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}) });

const E = { stack: null, checks: [], results: null, replay: null, busy: null, msg: null, roots: [] };
const currentRun = () => $("#run-select")?.value || null;
const chatSettings = () => { try { return JSON.parse(localStorage.getItem("brainscan-chat") || "{}"); } catch { return {}; } };
const applied = () => (E.stack?.applied || []).length;
const appliedIds = () => (E.stack?.applied || []).map((e) => e.id).join();

async function refresh() {
  try {
    [E.stack, E.checks] = await Promise.all([api("/brainscan/workbench"), api("/brainscan/checks")]);
  } catch (err) { if (err.status !== 503) E.msg = { text: err.message, tone: "error" }; }
  render();
}

async function act(label, fn, after) {
  E.busy = label; E.msg = null; render();
  try { await fn(); if (after) await after(); } catch (err) { E.msg = { text: err.message, tone: "error" }; }
  E.busy = null;
  await refresh();
}

// ---------------------------------------------------------------- sections
function stackHtml() {
  const s = E.stack;
  if (!s) return `<section class="ed-sec"><p class="muted">Loading the edit stack…</p></section>`;
  const n = applied();
  const rows = s.edits.map((e) => `<li data-id="${e.id}" class="${e.applied ? "on" : ""}">
      <label class="check"><input type="checkbox" data-toggle ${e.enabled ? "checked" : ""} ${E.busy ? "disabled" : ""}><span>${esc(e.summary)}</span></label>
      ${e.applied ? `<span class="pill ok">applied</span>` : ""}
      <button type="button" class="tbtn icon" data-remove aria-label="Remove this edit" ${e.applied ? "disabled title=\"Restore the original weights first\"" : "title=\"Remove\""}>✕</button></li>`).join("");
  return `<section class="ed-sec">
    <div class="ed-head"><b>Edit stack</b><span class="pill ${n ? "warn" : ""}">${n ? `${n} edit${n === 1 ? "" : "s"} applied` : "original weights"}</span></div>
    <ul class="ed-list">${rows || `<li class="muted">No edits yet. Knock out units on a token and press <b>Make permanent</b>, lesion a unit from its panel, or build an edit in Tune.</li>`}</ul>
    <div class="ed-row">
      <button type="button" class="primary sm" data-apply ${E.busy || !s.edits.some((e) => e.enabled) ? "disabled" : ""}>Apply stack</button>
      <button type="button" class="btn sm" data-revert ${E.busy || !n ? "disabled" : ""}>Restore original</button>
      ${E.busy ? `<span class="muted small">${esc(E.busy)}…</span>` : ""}
    </div>
    <p class="hint">Applying replays every checked edit on the original weights; restoring is exact. Requests made while edits are applied are tagged with them.</p>
  </section>`;
}

function replayHtml() {
  const run = currentRun(), r = E.replay, n = applied();
  let card = "";
  if (r?.error) card = `<p class="warn-text">${esc(r.error)}</p>`;
  else if (r) {
    const fd = r.first_diff;
    card = `<div class="rp-card">
      <div class="rp-q"><span class="lbl">Asked again</span>${esc(r.prompt)}${r.turns > 1 ? ` <span class="dim">(with the ${r.turns - 1} earlier messages)</span>` : ""}</div>
      <div class="rp-cols">
        <div><span class="lbl">Original weights</span><p>${esc(r.original) || "<i>(empty)</i>"}</p></div>
        <div><span class="lbl">Edited weights</span><p>${esc(r.edited) || "<i>(empty)</i>"}</p></div>
      </div>
      ${fd ? `<p class="hint">They part at token ${fd.index + 1}${fd.before ? `, after “…${esc(fd.before.slice(-40))}”` : ""}:
          original <code>${esc(visTok(fd.original_token))}</code>, edited <code>${esc(visTok(fd.edited_token))}</code>.</p>`
        : `<p class="hint">The same answer, token for token: the edits didn't change it.</p>`}
      <div class="ed-row">
        ${fd?.t != null ? `<button type="button" class="btn sm" data-trace-diff title="Open the edited answer in Trace on the first different word, compared with the original weights">Trace the difference →</button>` : ""}
        <button type="button" class="link" data-open-edited>Open the edited answer in Trace</button>
        <span class="dim small">${r.seconds}s</span>
      </div></div>`;
  }
  const why = !n ? "Apply an edit first." : !run ? "Open a chat first (ask the model, or pick a run)." : "";
  return `<section class="ed-sec">
    <div class="ed-head"><b>This chat on the edited weights</b></div>
    <p class="hint">Asks the open chat's last question again, greedy, on the original and the edited weights, and finds the first word where the answers part. The edited answer is recorded as its own run, so you can trace it.</p>
    <div class="ed-row"><button type="button" class="btn sm" data-replay ${why || E.busy ? "disabled" : ""} title="${esc(why)}">Re-run this chat</button>
      ${why ? `<span class="muted small">${why}</span>` : ""}</div>
    ${card}
  </section>`;
}

function checksHtml() {
  const res = new Map((E.results?.checks || []).map((c) => [c.id, c]));
  const rows = E.checks.map((c) => {
    const r = res.get(c.id), now = r?.now, orig = r?.original;
    let val = `<span class="muted small">not run</span>`;
    if (r?.error) val = `<span class="warn-text small">${esc(r.error)}</span>`;
    else if (now) {
      const moved = orig ? now.prob - orig.prob : 0;
      val = orig
        ? `<span class="ck-val ${moved < -0.01 ? "down" : moved > 0.01 ? "up" : ""}">p ${fmtP(orig.prob)} → <b>${fmtP(now.prob)}</b></span><span class="dim small">#${orig.rank} → #${now.rank}${now.rank > 1 ? ` · now ${esc(visTok(now.top[0]?.text))}` : ""}</span>`
        : `<span class="ck-val">p <b>${fmtP(now.prob)}</b></span><span class="dim small">#${now.rank}${now.rank > 1 ? ` · top ${esc(visTok(now.top[0]?.text))}` : ""}</span>`;
    }
    return `<li data-id="${c.id}"><span class="ck-label" title="${esc(c.kind === "token" ? `From run ${c.run}, token ${c.t}` : c.chat ? "Asked as a chat question" : "Text continuation")}">
        <span class="ck-kind">${c.kind === "token" ? "trace" : c.chat ? "chat" : "text"}</span>${esc(visTok(c.label))}</span>
      <span class="ck-res">${val}</span>
      <button type="button" class="tbtn icon" data-remove-check aria-label="Remove this check">✕</button></li>`;
  }).join("");
  return `<section class="ed-sec">
    <div class="ed-head"><b>Behavior checks</b>
      <button type="button" class="btn sm" data-run-checks ${E.checks.length && !E.busy ? "" : "disabled"}>Run checks</button></div>
    <ul class="ck-list">${rows || `<li class="muted">No checks yet. Add one from any traced word (<b>Add as a check</b> in its panel), or type one below.</li>`}</ul>
    <form class="ck-add" data-add-check>
      <input name="prompt" type="text" placeholder="After this… e.g. The capital of Australia is" aria-label="Text before the expected word" spellcheck="false">
      <input name="expect" type="text" placeholder="expect… e.g. Canberra" aria-label="The word you expect next" spellcheck="false">
      <label class="check small"><input type="checkbox" name="chat"> ask as a chat question</label>
      <button type="submit" class="btn sm">Add check</button>
    </form>
    <p class="hint">A check is what the model should say next. With edits applied, each shows its probability on the original weights → now, so you can see what an edit fixed and what it broke.</p>
  </section>`;
}

function render() {
  for (const root of E.roots) {
    const want = (root.dataset.edits || "all").split(/\s+/);
    const has = (k) => want.includes("all") || want.includes(k);
    root.innerHTML = `${E.msg ? `<p class="${E.msg.tone === "error" ? "warn-text" : "ok-text"} ed-msg">${esc(E.msg.text)}</p>` : ""}
      ${has("stack") ? stackHtml() : ""}${has("replay") ? replayHtml() : ""}${has("checks") ? checksHtml() : ""}`;
  }
  const btn = $("#gv-edits-btn .count"), n = applied();
  if (btn) { btn.textContent = n ? String(n) : ""; btn.hidden = !n; }
  // checked: the checks were run on the weights as they are now
  document.dispatchEvent(new CustomEvent("brainscan:edit-state", { detail: { applied: n, edits: (E.stack?.edits || []).length,
    checks: E.checks.length, checked: !!E.results && E.resultsFor === appliedIds() } }));
}

// ---------------------------------------------------------------- actions
async function runChecks() {
  if (!E.checks.length) return;
  E.busy = "Running checks"; render();
  const on = appliedIds();
  try { E.results = await send("POST", "/brainscan/checks/run"); E.resultsFor = on; } catch (err) { E.msg = { text: err.message, tone: "error" }; }
  E.busy = null; render();
}

async function replay() {
  const run = currentRun(), cs = chatSettings();
  const kwargs = cs.think === "on" ? { enable_thinking: true } : cs.think === "off" ? { enable_thinking: false } : undefined;
  E.replay = null;
  await act("Generating both answers", async () => {
    try {
      E.replay = await send("POST", "/brainscan/workbench/replay", { run, max_tokens: Math.min(Number(cs.max) || 256, 1024), template_kwargs: kwargs });
    } catch (err) { E.replay = { error: err.status === 409 ? "The model is busy; try again when it finishes." : err.message }; }
  });
}

function openInTrace(run, word) {
  const sel = $("#run-select");
  document.dispatchEvent(new CustomEvent("brainscan:show", { detail: "weights" }));
  if (sel && sel.value !== run) {
    if (![...sel.options].some((o) => o.value === run)) sel.insertAdjacentHTML("beforeend", `<option value="${esc(run)}">${esc(run)}</option>`);
    sel.value = run;
    sel.dispatchEvent(new Event("change"));
  }
  if (word != null) document.dispatchEvent(new CustomEvent("brainscan:trace", { detail: { word, cmp: true } }));
}

function onClick(ev) {
  const t = ev.target;
  const id = t.closest("li[data-id]")?.dataset.id;
  if (t.matches("[data-apply]")) act("Applying", () => send("POST", "/brainscan/workbench/apply"), runChecks);
  else if (t.matches("[data-revert]")) act("Restoring", () => send("POST", "/brainscan/workbench/revert"), runChecks);
  else if (t.matches("[data-remove]") && id) act("Removing", () => send("DELETE", `/brainscan/workbench/edits/${id}`));
  else if (t.matches("[data-replay]")) replay();
  else if (t.matches("[data-trace-diff]") && E.replay?.first_diff) openInTrace(E.replay.edited_run, E.replay.first_diff.t);
  else if (t.matches("[data-open-edited]") && E.replay) openInTrace(E.replay.edited_run, null);
  else if (t.matches("[data-run-checks]")) runChecks();
  else if (t.matches("[data-remove-check]") && id) act("Removing", () => send("DELETE", `/brainscan/checks/${id}`));
}

function onChange(ev) {
  const id = ev.target.closest("li[data-id]")?.dataset.id;
  if (ev.target.matches("[data-toggle]") && id) act("Updating", () => send("PATCH", `/brainscan/workbench/edits/${id}`, { enabled: ev.target.checked }));
}

function onSubmit(ev) {
  if (!ev.target.matches("[data-add-check]")) return;
  ev.preventDefault();
  const f = ev.target, body = { prompt: f.prompt.value, expect: f.expect.value, chat: f.chat.checked };
  act("Adding the check", () => send("POST", "/brainscan/checks", body), runChecks);
}

function boot() {
  E.roots = [...document.querySelectorAll("[data-edits]")];
  for (const root of E.roots) {
    root.addEventListener("click", onClick);
    root.addEventListener("change", onChange);
    root.addEventListener("submit", onSubmit);
    root.addEventListener("keydown", (ev) => ev.stopPropagation());  // typing a check never drives the graph
  }
  // The stack or the checks changed elsewhere (the weights view relays the live "workbench" message).
  document.addEventListener("brainscan:workbench", refresh);
  document.addEventListener("brainscan:checks", async (ev) => { await refresh(); if (ev.detail?.run) runChecks(); });
  document.addEventListener("brainscan:model", () => { E.results = null; E.replay = null; refresh(); });
  document.querySelector("#run-select")?.addEventListener("change", render);
  document.addEventListener("brainscan:run", render);
  refresh();
}
boot();
