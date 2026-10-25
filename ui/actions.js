// "Forget this fact": a guided knockout made permanent, as a five-step flow (Goal, Target, Try,
// Measure, Keep) on /brainscan/actions/forget/* (see brainscan/recorder/actions.py). It opens from
// the action bars (Trace hands over the traced word through brainscan:action; Tune and Do… open it
// with a typed fact), and its back link returns to the tab it was opened from.

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtP = (p) => (p == null ? "–" : p >= 0.001 || p === 0 ? p.toFixed(3) : p.toExponential(1));
const vis = (s) => (s ?? "").replace(/\n/g, "⏎").replace(/\t/g, "⇥") || "∅";
const cleanText = (s) => (s || "").replace(/<\|[^|>]*\|>\s*(?:system|user|assistant)?|<\/?think>/g, " ").replace(/\s+/g, " ").trim();

async function api(method, path, body) {
  const res = await fetch(path, { method, headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { /* not JSON */ }
  if (!res.ok) {
    const err = new Error(res.status === 409 ? "The model is busy answering a request. Try again when it finishes." : data?.detail || text || `${res.status}`);
    err.status = res.status;
    throw err;
  }
  return data;
}

const STEPS = [["goal", "Goal"], ["target", "Target"], ["try", "Try"], ["measure", "Measure"], ["keep", "Keep"]];
const EXAMPLE = {
  prompt: "The capital city of Australia is", answer: "Canberra",
  leads: "Australia's capital city is",
  questions: "What is the capital of Australia? Answer in one sentence.",
  keep: "The capital of France is → Paris\nThe largest city in Australia is → Sydney\nThe chemical symbol for gold is → Au",
};

const F = {
  step: "goal", source: null, prompt: "", answer: "", leads: "", questions: "", keep: "", checks: true,
  target: null, ranking: "direct", k: 1, tried: null, measured: null, kept: null, busy: null, msg: null,
};

function reset(source = null) {
  Object.assign(F, { step: "goal", source, prompt: "", answer: "", leads: "", questions: "", keep: "", checks: true,
    target: null, ranking: "direct", k: 1, tried: null, measured: null, kept: null, busy: null, msg: null });
  if (source?.question) F.questions = source.question;
}

// Target ranks the units two ways: "direct" (what they wrote toward the answer) and, with facts to
// keep, "specific" (toward the answer more than toward any fact to keep, added only while every fact
// to keep is spared). The chosen ranking and size are what Try, Measure and Keep use.
const RANKINGS = { direct: "Most direct", specific: "Most specific" };
const ranking = () => F.target?.rankings[F.ranking] || F.target?.rankings.direct;
const units = () => (ranking()?.units || []).slice(0, F.k).map((u) => u.g);
const pick = () => `${F.ranking}:${F.k}`;
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
// The same test as Measure's verdict: lost first place, or more than 0.2 of its probability.
const broke = (before, after) => (before.rank === 1 && after.rank > 1) || before.prob - after.prob > 0.2;
const shortLabel = (label) => cleanText(label).split("→").pop().trim() || cleanText(label);
const answerText = () => F.target?.fact.answer || (F.source ? F.source.word?.trim() : F.answer.trim()) || "the answer";
function factBody() {
  const fact = F.source ? { run: F.source.run, t: F.source.t, target: F.source.target } : { prompt: F.prompt, answer: F.answer };
  return { ...fact, leads: F.leads, questions: F.questions, keep: F.keep, checks: F.checks };
}
const parseKeep = (text) => text.split("\n").map((l) => l.trim()).filter(Boolean).map((l) => {
  const sep = ["→", "->", "=>", "|"].find((x) => l.includes(x));
  if (!sep) return null;
  const [a, ...b] = l.split(sep);
  return a.trim() && b.join(sep).trim() ? { prompt: a.trim(), expect: b.join(sep).trim() } : null;
}).filter(Boolean);

// ---------------------------------------------------------------- views
let current = "scanner", from = "weights";  // the tab on screen, and the one the flow was opened from
function close() { go(from); }
function openForget(source = null) {
  if (current !== "actions") from = current;
  reset(source && { ...source, lead: cleanText(source.lead) });
  document.dispatchEvent(new CustomEvent("brainscan:show", { detail: "actions" }));
  render();
  if (F.source && !F.questions) prefillQuestion(F.source);
}

/** A word from a chat: ask the chat's own question again as one of the tests. */
async function prefillQuestion(src) {
  try {
    const run = await api("GET", `/brainscan/runs/${encodeURIComponent(src.run)}`);
    const asked = (run.messages || []).filter((m) => m.role === "user" && m.t <= src.t && (m.content || "").trim())
      .sort((a, b) => a.t - b.t).pop();
    if (asked && F.source === src && !F.questions) {
      F.questions = asked.content.trim();
      if (F.step === "goal" && !F.busy) render();
    }
  } catch { /* the question stays empty */ }
}
function go(view, then) {
  document.dispatchEvent(new CustomEvent("brainscan:show", { detail: view }));
  if (then) requestAnimationFrame(() => setTimeout(then, 60));
}
const scrollTo = (sel) => () => $(sel)?.scrollIntoView({ behavior: "smooth", block: "start" });

function openRunInTrace(run, word) {
  const sel = $("#run-select");
  go("weights");
  if (sel && sel.value !== run) {
    if (![...sel.options].some((o) => o.value === run)) sel.insertAdjacentHTML("beforeend", `<option value="${esc(run)}">${esc(run)}</option>`);
    sel.value = run;
    sel.dispatchEvent(new Event("change"));
  }
  if (word != null) document.dispatchEvent(new CustomEvent("brainscan:trace", { detail: { word } }));
}

// ---------------------------------------------------------------- the flow
const reachable = (step) => step === "goal" || !!F.target;

function render() {
  const i = STEPS.findIndex(([k]) => k === F.step);
  $("#fl-steps").innerHTML = STEPS.map(([k, label], j) => `<li class="${j < i ? "done" : j === i ? "now" : ""}">
    <button type="button" data-step="${k}" ${reachable(k) && !F.busy ? "" : "disabled"} ${j === i ? 'aria-current="step"' : ""}>
      <span class="n">${j + 1}</span>${label}</button></li>`).join("");
  $("#fl-steps").querySelectorAll("[data-step]").forEach((b) => { b.onclick = () => goto(b.dataset.step); });
  $("#fl-summary").innerHTML = summaryHtml();
  const body = $("#fl-body");
  body.innerHTML = ({ goal: goalHtml, target: targetHtml, try: tryHtml, measure: measureHtml, keep: keepHtml })[F.step]() +
    `<div class="fl-msg ${F.msg?.tone || ""}" role="status">${F.busy ? `<span class="spinner sm"></span>${esc(F.busy)}…` : esc(F.msg?.text || "")}</div>`;
  bind(body);
}

function summaryHtml() {
  if (!F.target) return "";
  const f = F.target.fact;
  return `<span class="sum-chip"><span class="lbl">Fact</span>…${esc(f.lead.slice(-50))} → <b>${esc(f.answer)}</b></span>
    <span class="sum-chip"><span class="lbl">Units</span>${F.k} of ${ranking().units.length} ${F.target.rankings.specific ? esc(RANKINGS[F.ranking].toLowerCase()) : "ranked"}</span>
    ${F.target.weights_edited ? `<span class="sum-chip warn"><span class="lbl">Weights</span>already edited: measured on top of your applied edits</span>` : ""}`;
}

function goalHtml() {
  const fact = F.source
    ? `<div class="fact-card"><span class="lbl">From your chat</span>
        <p class="mono">…${esc(F.source.lead || "")}<mark>${esc(vis(F.source.word))}</mark></p>
        <button type="button" class="link" id="fl-typed">Type a fact instead</button></div>`
    : `<label>Lead-in<input id="fl-prompt" type="text" spellcheck="false" placeholder="The capital of Australia is" value="${esc(F.prompt)}"></label>
       <label>The answer it gives next<input id="fl-answer" type="text" spellcheck="false" placeholder="Canberra" value="${esc(F.answer)}"></label>
       <button type="button" class="link" id="fl-example">Load an example</button>`;
  return `<div class="fl-grid">
    <div class="fl-col"><h3>The fact to remove</h3>${fact}
      <p class="hint">The answer is the word the model writes right after the lead-in. You can also start from any word in a chat:
        click it in Trace and press <b>Forget this fact</b>.</p></div>
    <div class="fl-col"><h3>How to test it <span class="dim">optional</span></h3>
      <label>Other lead-ins to the same answer, one per line<textarea id="fl-leads" rows="3" spellcheck="false" placeholder="Australia's capital city is">${esc(F.leads)}</textarea></label>
      <label>Questions to ask it, one per line<textarea id="fl-questions" rows="2" spellcheck="false" placeholder="What is the capital of Australia?">${esc(F.questions)}</textarea></label>
      <h3>Facts it should keep <span class="dim">optional</span></h3>
      <label>One per line, as lead-in → answer<textarea id="fl-keep" rows="3" spellcheck="false" placeholder="The capital of France is → Paris">${esc(F.keep)}</textarea></label>
      <label class="check"><input type="checkbox" id="fl-checks" ${F.checks ? "checked" : ""}> Also measure my saved behavior checks</label></div>
  </div>
  <div class="fl-actions"><button type="button" class="primary" id="fl-find" ${F.busy ? "disabled" : ""}>Find where it lives →</button></div>`;
}

/** The facts to keep that broke with a curve point's units knocked out. */
const lostAt = (p) => F.target.keep.filter((k, i) => p.keep[i] && broke(k, p.keep[i]));

function keepCell(p) {
  const keep = F.target.keep;
  if (!keep.length) return "";
  const lost = lostAt(p);
  const tip = keep.map((k, i) => `${shortLabel(k.label)}: ${fmtP(k.prob)} → ${fmtP(p.keep[i]?.prob)}`).join("\n");
  return `<span class="kc ${lost.length ? "bad" : "ok"}" title="${esc(tip)}">${lost.length
    ? `breaks ${esc(lost.map((k) => shortLabel(k.label)).join(", "))}` : `spares ${keep.length === 1 ? "the fact" : `all ${keep.length}`} to keep`}</span>`;
}

function curveHtml(name) {
  const r = F.target.rankings[name], on = name === F.ranking;
  return `<ul class="curve">${r.curve.map((p) => `<li data-rank="${name}" data-k="${p.k}" class="${p.k === r.suggested ? "sugg" : ""} ${on && p.k === F.k ? "sel" : ""}" title="Knock out the top ${p.k}">
      <span class="k">${plural(p.k, "unit")}</span>
      <span class="pbar"><i style="width:${Math.max(1, p.prob * 100).toFixed(1)}%"></i></span>
      <span class="v">p ${fmtP(p.prob)}</span>
      <span class="st">${leadCell(p)}${keepCell(p)}</span></li>`).join("")}</ul>`;
}

/** Whether the answer is gone at a curve point, after the fact's lead-in or after every way of asking. */
function leadCell(p) {
  const leads = F.target.leads, n = leads.length;
  if (n < 2) return `<span class="dim">${p.gone ? `gone: now answers <b>${esc(vis(p.top))}</b>` : p.rank === 1 ? "still its first answer" : `still #${p.rank}`}</span>`;
  const tip = leads.map((ld, i) => `${cleanText(ld.text).slice(-50)}: ${fmtP(ld.prob)} → ${fmtP(p.leads[i]?.prob)} (#${p.leads[i]?.rank})`).join("\n");
  return `<span class="dim" title="${esc(tip)}">${p.gone ? `gone after all ${n} ways of asking: now answers <b>${esc(vis(p.top))}</b>`
    : `gone after ${p.leads_gone} of ${n} ways of asking`}</span>`;
}

/** "after every way of asking" when the fact is asked for more than one way. */
const everyWay = () => (F.target.leads.length > 1 ? ` after all ${F.target.leads.length} ways of asking` : "");

/** One line on a ranking: how many units remove the answer, and what that costs the facts to keep. */
function rankLine(name) {
  const r = F.target.rankings[name];
  if (!r.units.length) return "Every unit that pushes the answer pushes a fact to keep as much, or hurts one.";
  const p = r.curve.find((x) => x.k === r.suggested) || r.curve[r.curve.length - 1];
  const lost = lostAt(p), n = F.target.leads.length;
  const removes = r.flipped ? `${plural(r.suggested, "unit")} remove it${everyWay()}`
    : n > 1 && p.leads_gone ? `${plural(p.k, "unit")} remove it after ${p.leads_gone} of ${n} ways of asking` : `${plural(p.k, "unit")} don't remove it`;
  return `${removes}${F.target.keep.length
    ? (lost.length ? `, and ${esc(lost.map((k) => shortLabel(k.label)).join(", "))} break${lost.length === 1 ? "s" : ""}` : ", and every fact to keep is spared") : ""}.`;
}

/** How a unit's effects were combined over the ways of asking ("" for one lead-in). */
function combinedHow() {
  const n = F.target.leads.length;
  if (n < 2) return "";
  return F.target.combine === "min" ? `at the one of the ${n} ways of asking where it wrote least` : `averaged over the ${n} ways of asking`;
}

/** Every way of asking for the fact that Target ranks against: the lead-ins and saved checks for the same answer. */
function leadsHtml() {
  const leads = F.target.leads;
  if (leads.length < 2) return "";
  return `<div class="chips"><span class="lbl">ranked against ${leads.length} ways of asking</span>${leads.map((ld) => `<span class="chip" title="${esc(cleanText(ld.text))}">
    ${ld.source === "check" ? `<span class="dim">check</span> ` : ""}…${esc(cleanText(ld.text).replace(/\s*→.*$/, "").slice(-40))}
    <span class="dim">${fmtP(ld.prob)}${ld.first ? "" : ` #${ld.rank}`}</span></span>`).join("")}</div>`;
}

function rankingsHtml() {
  const t = F.target, spec = t.rankings.specific;
  if (!spec) {
    return `<h3>How many units it takes</h3>${curveHtml("direct")}
      ${t.keep.length ? "" : `<p class="hint">Add facts to keep in Goal to also rank the units by how specific they are to this answer.</p>`}`;
  }
  const how = combinedHow();
  const skipped = spec.skipped.length
    ? ` <span title="${esc(spec.skipped.map((u) => u.name).join(", "))}">Skipped ${plural(spec.skipped.length, "unit")} that hurt a fact to keep.</span>` : "";
  const col = (name, how) => {
    const r = t.rankings[name], on = name === F.ranking;
    return `<div class="rank-col ${on ? "on" : ""}">
      <button type="button" class="rank-head" data-rank="${name}" aria-pressed="${on}" ${r.units.length ? "" : "disabled"}>
        <b>${RANKINGS[name]}</b><span class="rank-line">${rankLine(name)}</span><span class="dim">${how}</span></button>
      ${r.units.length ? curveHtml(name) : ""}</div>`;
  };
  return `<h3>How many units it takes, ranked two ways</h3>
    <div class="rank-cols">
      ${col("direct", `By how much each unit wrote toward the answer${how ? `, ${how}` : ""}.`)}
      ${col("specific", `By how much more it wrote toward the answer than toward any fact to keep${how ? ` (${how})` : ""}, each added only if every fact to keep keeps first place and ${Math.round(spec.guard * 100)}% of its probability.${skipped}`)}
    </div>`;
}

function kSlider() {
  const n = ranking().units.length;
  return `<div class="fl-k"><label for="fl-k">Units to remove</label>
    <input type="range" id="fl-k" min="1" max="${n}" step="1" value="${F.k}"><b id="fl-kv">${F.k}</b>
    <span class="dim">of ${n} ranked</span></div>`;
}

function targetHtml() {
  const t = F.target, f = t.fact;
  const alts = f.alternatives.filter((a) => a.text !== f.token).slice(0, 4)
    .map((a) => `<span class="chip">${esc(vis(a.text))} <span class="dim">${fmtP(a.prob)}</span></span>`).join("");
  const r = ranking(), last = r.curve[r.curve.length - 1];
  return `<p class="fl-lead">After <q>…${esc(f.lead.slice(-80))}</q> the model answers <span class="tok-chip target">${esc(vis(f.token))}</span>
      with probability <b>${fmtP(f.prob)}</b>${f.rank > 1 ? ` (ranked #${f.rank})` : ""}.</p>
    ${f.first ? "" : `<div class="verdict partial">After this lead-in the model's first answer is “${esc(vis(f.alternatives[0]?.text))}”, not “${esc(f.answer)}”:
      it isn't stating this fact here. Pick a lead-in it answers with “${esc(f.answer)}” first, or start from a word in a chat. Here, “gone” means under a tenth of its probability.</div>`}
    ${alts ? `<div class="chips"><span class="lbl">also considered</span>${alts}</div>` : ""}
    ${leadsHtml()}
    ${rankingsHtml()}
    <p class="hint">${r.flipped
      ? `Suggested${t.rankings.specific ? ` (${esc(RANKINGS[F.ranking].toLowerCase())})` : ""}: <b>${plural(r.suggested, "unit")}</b>, the smallest set here after which the model no longer gives “${esc(f.answer)}”${everyWay()}.`
      : `Even ${last.k} units don't remove it${everyWay()}: the answer is spread widely, or other units take over. You can still try.`}
      Each row removes the top units together and reads ${t.leads.length > 1 ? "every way of asking" : "the lead-in"}${t.keep.length ? " and the facts to keep" : ""} with them removed. Measure judges the set you pick.</p>
    ${kSlider()}
    <div class="unit-chips">${r.units.map((u, i) => `<span class="uchip ${u.kind} ${i < F.k ? "on" : ""}" title="${u.kind === "head" ? "attention head" : "MLP neuron cluster"} · wrote ${u.direct >= 0 ? "+" : ""}${u.direct} toward the answer${u.by_lead ? `, ${combinedHow()} (${u.by_lead.map((x) => (x >= 0 ? "+" : "") + x).join(", ")})` : ""}${u.keep != null ? `, up to +${u.keep} toward a fact to keep` : ""}">${esc(u.name)}</span>`).join("")}</div>
    ${F.source ? `<button type="button" class="link" id="fl-why">See the trace behind this word →</button>` : ""}
    <div class="fl-actions"><button type="button" class="btn" data-goto="goal">← Goal</button>
      <button type="button" class="primary" data-goto="try">Try it →</button></div>`;
}

function highlight(text) {
  const a = answerText();
  const safe = esc(text || "(nothing)");
  if (!a) return safe;
  const re = new RegExp(esc(a).replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi");
  return safe.replace(re, (m) => `<mark>${m}</mark>`);
}

function tryHtml() {
  const r = F.tried;
  const stale = r && r.pick !== pick();
  const rows = r ? r.rows.map((x) => `<tr><td><span class="kind">${x.kind === "lead" ? "lead-in" : "question"}</span>${esc(x.prompt.slice(-120))}</td>
      <td>${highlight(x.original)}${x.said_before ? "" : ` <span class="dim small">(didn't say it)</span>`}</td>
      <td class="${x.says_now ? "still" : "gone"}">${highlight(x.edited)}
        <span class="badge ${x.says_now ? "serious" : "good"}">${x.says_now ? `still says ${esc(answerText())}` : "doesn't say it"}</span></td></tr>`).join("") : "";
  return `${kSlider()}
    ${r ? `<table class="try-table${stale ? " stale" : ""}"><thead><tr><th>Asked</th><th>Original</th><th>Without the ${r.units} unit${r.units === 1 ? "" : "s"}</th></tr></thead><tbody>${rows}</tbody></table>`
      : `<p class="empty">Running both versions…</p>`}
    <p class="hint">The same prompts, greedy, with and without the units. Nothing is changed in the weights yet.
      ${stale ? "<b>You changed the units: run it again.</b>" : ""}</p>
    <div class="fl-actions"><button type="button" class="btn" data-goto="target">← Target</button>
      <button type="button" class="btn" id="fl-rerun" ${F.busy ? "disabled" : ""}>Run again</button>
      <button type="button" class="primary" data-goto="measure">Measure it →</button></div>`;
}

const tile = (label, value, sub, tone = "") => `<div class="tile ${tone}"><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub}</div></div>`;
const pr = (x) => (x ? `${fmtP(x.prob)} <span class="dim">#${x.rank}</span>` : "–");

function measureHtml() {
  const m = F.measured;
  if (!m) return `<p class="empty">Measuring…</p><div class="skeleton"></div><div class="skeleton tall"></div>`;
  const stale = m.pick !== pick();
  const n = m.leads.length, gone = m.leads.filter((r) => r.gone).length;
  const ctrlGone = m.leads.filter((r) => r.control_gone >= 0.5).length;
  const main = m.leads[0];
  const kept = m.keep.filter((k) => k.before);
  const broken = kept.filter((k) => broke(k.before, k.after));
  const d = m.drift;
  return `<div class="verdict ${m.verdict.status}">${esc(m.verdict.text)}${stale ? " <b>(measured with different units: measure again.)</b>" : ""}</div>
    <div class="tiles">
      ${tile("The fact", `gone after ${gone} of ${n}`, `main lead-in p ${fmtP(main.before.prob)} → ${fmtP(main.after.prob)}${main.gone ? `, now “${esc(vis(main.after.top))}”` : ""}`, gone === n ? "good" : gone ? "warn" : "bad")}
      ${tile(`Random units (${m.control.units}, ${m.control.seeds} sets)`, ctrlGone ? `removed it on ${ctrlGone} of ${n}` : "left it alone", `main lead-in p ${fmtP(main.before.prob)} → ${fmtP(main.control.prob)}`, ctrlGone ? "bad" : "good")}
      ${tile("Facts to keep", kept.length ? `${kept.length - broken.length} of ${kept.length} intact` : "none given", kept.length ? (broken.length ? `broke: ${broken.map((k) => esc(k.label.slice(-40))).join("; ")}` : "none lost first place") : "add some in Goal to catch side effects", !kept.length ? "" : broken.length ? "bad" : "good")}
      ${tile("Neutral text", `KL ${d.kl_per_token.toFixed(3)}`, `perplexity ${d.perplexity_before} → ${d.perplexity_after} over ${d.tokens} tokens`, d.kl_per_token > 0.05 ? "warn" : "good")}
    </div>
    <details class="fl-details" open><summary>Every lead-in</summary>
      <table class="m-table"><thead><tr><th>Lead-in</th><th>Before</th><th>After</th><th>Random units</th></tr></thead><tbody>
      ${m.leads.map((r) => `<tr><td>${r.source === "check" ? `<span class="kind">check</span>` : ""}${esc(r.prompt.slice(-90))} <span class="dim">→ ${esc(vis(r.token))}</span></td><td>${pr(r.before)}</td>
        <td class="${r.gone ? "gone" : "still"}">${pr(r.after)}${r.gone ? ` <span class="dim">now “${esc(vis(r.after.top))}”</span>` : ""}</td><td>${pr(r.control)}</td></tr>`).join("")}
      </tbody></table></details>
    ${m.keep.length ? `<details class="fl-details" open><summary>Facts to keep and your checks</summary>
      <table class="m-table"><thead><tr><th>Fact</th><th>Before</th><th>After</th></tr></thead><tbody>
      ${m.keep.map((k) => `<tr><td><span class="kind">${k.source === "check" ? "check" : "typed"}</span>${esc(vis(k.label).slice(-90))}</td>
        ${k.error ? `<td colspan="2" class="dim">${esc(k.error)}</td>` : `<td>${pr(k.before)}</td><td class="${broken.includes(k) ? "still" : ""}">${pr(k.after)}</td>`}</tr>`).join("")}
      </tbody></table></details>` : ""}
    <p class="hint">Measured with the units knocked out at run time, which is exactly what keeping them as a weight edit does.
      Random units: the same number of heads and clusters drawn at random, three times; if they remove the fact too, it wasn't found in these units.
      Neutral text: KL between the next-token distributions with and without the units, on five reference paragraphs.</p>
    <div class="fl-actions"><button type="button" class="btn" data-goto="try">← Try</button>
      <button type="button" class="btn" id="fl-remeasure" ${F.busy ? "disabled" : ""}>Measure again</button>
      <button type="button" class="primary" data-goto="keep">Keep it →</button></div>`;
}

function keepHtml() {
  const names = (ranking()?.units || []).slice(0, F.k).map((u) => u.name);
  if (F.kept) {
    return `<div class="verdict works">Done: the ${names.length} unit${names.length === 1 ? "" : "s"} behind “${esc(answerText())}” are removed from the weights, and the edit is applied now.</div>
      <p class="hint">${F.kept.checks ? `Saved ${F.kept.checks} behavior check${F.kept.checks === 1 ? "" : "s"}; they show before → after on every edit from now on. ` : ""}
        The edit is listed under Your model. Restoring the original weights undoes every applied edit; the edits stay listed so you can apply them again.</p>
      <div class="fl-actions"><button type="button" class="btn" id="fl-ask">Ask the edited model</button>
        <button type="button" class="btn" id="fl-export">Export the edited model</button>
        <button type="button" class="btn" id="fl-revert">Restore original weights</button>
        <button type="button" class="link" id="fl-home">Back to all actions</button></div>`;
  }
  const m = F.measured, measuredHere = m && m.pick === pick();
  return `<p class="fl-lead">Remove these ${names.length} unit${names.length === 1 ? "" : "s"} from the weights:</p>
    <div class="unit-chips">${names.map((n) => `<span class="uchip on">${esc(n)}</span>`).join("")}</div>
    ${measuredHere ? `<div class="verdict ${m.verdict.status}">${esc(m.verdict.text)}</div>`
      : `<div class="verdict kept">Not measured with these ${plural(names.length, "unit")} yet. Measure before keeping it.</div>`}
    <p class="hint">Their output weights are set to zero, as one edit in your model's stack, applied now. Nothing on disk changes until you export,
      and the original weights can be restored exactly at any time.</p>
    <label class="check"><input type="checkbox" id="fl-save-keep" ${parseKeep(F.keep).length ? "checked" : "disabled"}> Save the facts to keep as behavior checks</label>
    <label class="check"><input type="checkbox" id="fl-save-fact" checked> Track the forgotten fact as a check too</label>
    <div class="fl-actions"><button type="button" class="btn" data-goto="measure">← Measure</button>
      <button type="button" class="btn" id="fl-discard">Discard</button>
      <button type="button" class="primary" id="fl-apply" ${F.busy ? "disabled" : ""}>Remove from the weights</button></div>`;
}

// ---------------------------------------------------------------- behavior
function bind(body) {
  const val = (id, key) => { const el = body.querySelector(id); if (el) el.oninput = () => { F[key] = el.value; }; };
  val("#fl-prompt", "prompt"); val("#fl-answer", "answer"); val("#fl-leads", "leads"); val("#fl-questions", "questions"); val("#fl-keep", "keep");
  const checks = body.querySelector("#fl-checks");
  if (checks) checks.onchange = () => { F.checks = checks.checked; };
  body.querySelector("#fl-example")?.addEventListener("click", () => { Object.assign(F, EXAMPLE); render(); });
  body.querySelector("#fl-typed")?.addEventListener("click", () => { F.source = null; F.target = null; render(); });
  body.querySelector("#fl-find")?.addEventListener("click", find);
  body.querySelectorAll("[data-goto]").forEach((b) => { b.onclick = () => goto(b.dataset.goto); b.disabled = !!F.busy; });
  const k = body.querySelector("#fl-k");
  if (k) {
    k.oninput = () => {
      F.k = Number(k.value);
      body.querySelector("#fl-kv").textContent = F.k;
      body.querySelectorAll(".unit-chips .uchip").forEach((c, i) => c.classList.toggle("on", i < F.k));
      body.querySelectorAll(".curve li").forEach((li) => li.classList.toggle("sel", li.dataset.rank === F.ranking && Number(li.dataset.k) === F.k));
      $("#fl-summary").innerHTML = summaryHtml();
    };
    k.onchange = () => { if (F.step === "try") runTry(); };
  }
  body.querySelectorAll(".curve li[data-k]").forEach((li) => { li.onclick = () => { F.ranking = li.dataset.rank; F.k = Number(li.dataset.k); render(); }; });
  body.querySelectorAll(".rank-head[data-rank]").forEach((b) => {
    b.onclick = () => { if (F.ranking !== b.dataset.rank) { F.ranking = b.dataset.rank; F.k = ranking().suggested || 1; render(); } };
  });
  body.querySelector("#fl-why")?.addEventListener("click", () => openRunInTrace(F.source.run, F.source.of));
  body.querySelector("#fl-rerun")?.addEventListener("click", runTry);
  body.querySelector("#fl-remeasure")?.addEventListener("click", runMeasure);
  body.querySelector("#fl-apply")?.addEventListener("click", keep);
  body.querySelector("#fl-discard")?.addEventListener("click", () => { reset(); close(); });
  body.querySelector("#fl-home")?.addEventListener("click", () => { reset(); close(); });
  body.querySelector("#fl-ask")?.addEventListener("click", () => go("weights", () => $("#tv-convo textarea")?.focus()));
  body.querySelector("#fl-export")?.addEventListener("click", () => go("workbench", scrollTo("#h-export")));
  body.querySelector("#fl-revert")?.addEventListener("click", revert);
}

function goto(step) {
  if (!reachable(step) || F.busy) return;
  F.step = step; F.msg = null;
  render();
  if (step === "try" && (!F.tried || F.tried.pick !== pick())) runTry();
  if (step === "measure" && (!F.measured || F.measured.pick !== pick())) runMeasure();
}

async function work(label, fn) {
  F.busy = label; F.msg = null; render();
  try { await fn(); } catch (err) { F.msg = { text: err.message, tone: "error" }; }
  F.busy = null; render();
}

function find() {
  if (!F.source && (!F.prompt.trim() || !F.answer.trim())) {
    F.msg = { text: "Write the lead-in and the answer the model gives after it.", tone: "error" }; render(); return;
  }
  work("Finding the units behind the answer", async () => {
    F.target = await api("POST", "/brainscan/actions/forget/target", factBody());
    F.ranking = F.target.ranking; F.k = ranking().suggested || 1; F.tried = null; F.measured = null; F.kept = null;
    F.step = "target";
  });
}

function runTry() {
  const k = F.k, p = pick();
  return work(`Answering with and without ${plural(k, "unit")}`, async () => {
    F.tried = { ...(await api("POST", "/brainscan/actions/forget/try", { ...factBody(), units: units() })), k, pick: p };
  });
}

function runMeasure() {
  const k = F.k, p = pick();
  return work("Measuring: the fact, random units, facts to keep, neutral text", async () => {
    F.measured = { ...(await api("POST", "/brainscan/actions/forget/measure", { ...factBody(), units: units() })), k, pick: p };
  });
}

function keep() {
  const saveKeep = $("#fl-save-keep")?.checked, saveFact = $("#fl-save-fact")?.checked;
  const f = F.target.fact;
  work("Removing the units from the weights", async () => {
    await api("POST", "/brainscan/anatomy/lesion", { units: units(), apply: true, note: `forget “${f.answer}” after “${f.lead.slice(-40)}”` });
    let saved = 0;
    if (saveKeep) for (const c of parseKeep(F.keep)) { await api("POST", "/brainscan/checks", { ...c, chat: false }); saved++; }
    if (saveFact) {
      await api("POST", "/brainscan/checks", F.source ? { run: F.source.run, t: F.source.t, target: F.source.target } : { prompt: F.prompt, expect: F.answer });
      saved++;
    }
    F.kept = { checks: saved };
    document.dispatchEvent(new CustomEvent("brainscan:workbench"));
    document.dispatchEvent(new CustomEvent("brainscan:checks", { detail: { run: true } }));
  });
}

function revert() {
  work("Restoring the original weights", async () => {
    await api("POST", "/brainscan/workbench/revert", {});
    document.dispatchEvent(new CustomEvent("brainscan:workbench"));
    F.msg = { text: "Original weights restored.", tone: "ok" };
  });
}

// ---------------------------------------------------------------- boot
function boot() {
  document.addEventListener("brainscan:view", (ev) => { current = ev.detail; });
  $("#fl-back").onclick = () => { if (!F.busy) close(); };
  // From an action bar or Do…: a traced word from Trace, or nothing (type a fact).
  document.addEventListener("brainscan:action", (ev) => {
    const d = ev.detail || {};
    if (d.action !== "forget") return;
    openForget(d.run ? { run: d.run, t: d.t, target: d.target, of: d.of, word: d.word, lead: d.lead, question: d.question } : null);
  });
  // Units belong to one model: a switch starts over.
  document.addEventListener("brainscan:model", () => { if (current === "actions" && !F.busy) { reset(); close(); } });
}
boot();
