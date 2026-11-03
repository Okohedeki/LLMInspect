// Verbs: the things to do to the loaded model beyond Ablate and Steer (llminspect/verbs). One screen for all of
// them: the verb's steps down the left, each a short form, and the hero on the right, the step's result read in
// plain words, with its numbers large and its tables in full. A step that changes the weights keeps its change as
// an edit in Tune's stack, unapplied, and says so; Apply bakes it in.
//
// What a step returns that the next one needs (a probe, a head, a concept vector, a key's bounds) is carried
// into the next form for you.
//
// API: GET /llminspect/verbs, POST /llminspect/verbs/{verb}/{step}
import { defineBar, refreshBar, noteBar } from "/ui/actionbar.js";

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// Fields: [name, kind, label, extra]. kinds: text, area (one item per line → list), paras (blank-line separated →
// list), num, nums (comma list), json, bool, pick (choices). `from: "step.key"` fills the field from a result.
const F = (name, kind, label, extra = {}) => ({ name, kind, label, ...extra });
export const LAB = {
  sign: { label: "Sign", group: "Protect", lead: "Hide an ownership mark in the weights that survives fine-tuning, and check for it later.",
    steps: {
      embed: { label: "Mark", what: "Add a mark derived from your key. The key isn't stored; keep it.", fields: [F("key", "text", "Key (a secret)"),
        F("strength", "num", "Strength (optional)"), F("targets", "pick", "Matrices", { choices: ["", "attn", "mlp"], list: true })] },
      verify: { label: "Verify", what: "How far this key's pattern stands out against 100 random keys.", fields: [F("key", "text", "Key"),
        F("against", "area", "Other keys to compare (optional)"), F("targets", "pick", "Matrices", { choices: ["", "attn", "mlp"], list: true })] } } },
  harden: { label: "Harden", group: "Protect", lead: "Measure how easily refusals are removed, then make them survive direction removal and steering.",
    steps: {
      attack: { label: "Attack", what: "Remove 1, 2 and 4 refusal directions and steer against them, beside a random direction. Blank uses the built-in sets.",
        fields: [F("refuse", "area", "Prompts it should refuse (optional)"), F("comply", "area", "Harmless prompts (optional)"), F("n", "num", "Prompts per set", { value: 16 })] },
      harden: { label: "Harden", what: "Train a small change so refusals hold with the directions removed, keeping everything else close.",
        fields: [F("refuse", "area", "Prompts it should refuse (optional)"), F("comply", "area", "Harmless prompts (optional)"),
          F("steps", "num", "Training steps", { value: 100 }), F("rank", "num", "Rank", { value: 8 })] } } },
  scrub: { label: "Scrub", group: "Protect", lead: "Find text the model memorized word for word, remove it, and show the rest still works.",
    steps: {
      find: { label: "Find", what: "Each passage's start, continued greedily, against the real rest of it.", fields: [F("texts", "paras", "Passages (blank line between)")] },
      remove: { label: "Remove", what: "A low-rank change that unlearns one passage, with drift on neutral text measured.", fields: [F("text", "area1", "The memorized passage")] } } },
  audit: { label: "Audit", group: "Inspect", lead: "Search, blind, for what an edit or fine-tune trained the model to do, against its base.",
    steps: {
      diff: { label: "Weights", what: "Which tensors moved, and what their main change pushes the model to say.", fields: [F("base", "text", "Base model folder")] },
      behave: { label: "Behavior", what: "The prompts where the two models part most, answers side by side.", fields: [F("base", "text", "Base model folder", { from: "diff.base" }), F("prompts", "area", "Prompts (optional)")] } } },
  persona: { label: "Persona", group: "Inspect", lead: "Where the model's character sits, and whether a fine-tune moved it toward harmful traits before its behavior shows it.",
    steps: {
      vectors: { label: "Traits", what: "A vector per trait (evil, sycophancy, hallucination, deception, power-seeking, the default assistant) from contrasting system prompts.",
        fields: [F("traits", "area", "Traits (optional, one per line)"), F("layer", "num", "Layer (optional)")] },
      scan: { label: "Scan", what: "How far a fine-tune moved the model along each trait, against its base, beside random directions.",
        fields: [F("base", "text", "Base model folder"), F("traits", "area", "Traits (optional)")] },
      monitor: { label: "Monitor", what: "Each trait read token by token across a conversation, so drift shows.",
        fields: [F("text", "area1", "Conversation or text"), F("vectors", "json", "Vectors", { from: "vectors.traits" }), F("layer", "num", "Layer (optional)")] } } },
  traces: { label: "Traces", group: "Inspect", lead: "What a narrow fine-tune trained in, read from how its activations differ from its base on unrelated text.",
    steps: {
      diff: { label: "Diff", what: "The fine-tune's consistent activation difference on the first tokens of neutral text, read as tokens.",
        fields: [F("base", "text", "Base model folder"), F("positions", "num", "First tokens", { value: 5 })] },
      steer: { label: "Steer", what: "The base model with the difference added, so the fine-tune's topic surfaces in plain text, beside a random vector.",
        fields: [F("base", "text", "Base model folder", { from: "diff.base" }), F("vector", "json", "Difference", { from: "diff.vector" }),
          F("layer", "num", "Layer", { from: "diff.best.layer" }), F("strengths", "nums", "Strengths", { value: "0, 4, 8, 16" })] },
      tokens: { label: "Tokens", what: "The tokens whose embeddings line up with the difference.",
        fields: [F("base", "text", "Base model folder", { from: "diff.base" }), F("vector", "json", "Difference", { from: "diff.vector" }), F("layer", "num", "Layer", { from: "diff.best.layer" })] } } },
  backdoor: { label: "Backdoor", group: "Protect", lead: "Hidden triggers: a probe for 'about to do the hidden thing', then the strings that set it off.",
    steps: {
      probe: { label: "Probe", what: "A probe from generic yes/no contrast questions, scored held out beside shuffled labels.", fields: [F("layer", "num", "Layer (optional)")] },
      scan: { label: "Scan", what: "Ordinary prompts with candidate triggers added, each trigger's lift against random strings.",
        fields: [F("probe", "json", "Probe", { from: "probe.probe" }), F("layer", "num", "Layer", { from: "probe.layer" }),
          F("triggers", "area", "Candidate triggers (optional)"), F("prompts", "area", "Prompts (optional)")] },
      search: { label: "Search", what: "Short token sequences that raise the probe most, beside random sequences.",
        fields: [F("probe", "json", "Probe", { from: "probe.probe" }), F("layer", "num", "Layer", { from: "probe.layer" }), F("top", "num", "How many", { value: 20 })] },
      compare: { label: "Compare", what: "The same triggers on the base model, so one that only works on the fine-tune stands out.",
        fields: [F("base", "text", "Base model folder"), F("probe", "json", "Probe", { from: "probe.probe" }), F("layer", "num", "Layer", { from: "probe.layer" }),
          F("triggers", "area", "Triggers (optional)")] } } },
  features: { label: "Features", group: "Inspect", lead: "Read, compare and steer the features of a sparse dictionary of the residual stream, loaded from a folder.",
    steps: {
      info: { label: "Load", what: "The dictionary's layer, size and how well it reconstructs the model's activations. Download one for the loaded model (safetensors) into a folder first.", fields: [F("path", "text", "Dictionary folder (one layer)")] },
      read: { label: "Read", what: "The features active on each token of your texts.", fields: [F("path", "text", "Dictionary folder", { from: "info.path" }), F("texts", "area", "Texts")] },
      diff: { label: "Diff", what: "Features that fire more or less than in the base model on the same texts.",
        fields: [F("path", "text", "Dictionary folder", { from: "info.path" }), F("base", "text", "Base model folder"), F("texts", "area", "Texts")] },
      steer: { label: "Steer", what: "Answers with one feature pushed, beside a random feature.", fields: [F("path", "text", "Dictionary folder", { from: "info.path" }),
        F("feature", "num", "Feature"), F("strength", "nums", "Strengths (share of the residual norm)", { value: "0, 0.1, 0.2, 0.4" }), F("prompts", "area", "Prompts")] } } },
  saysknows: { label: "Says vs. knows", group: "Inspect", lead: "Put what the model answered beside what its hidden states say it believes.",
    steps: {
      train: { label: "Probe", what: "A truth probe per layer from true and false statements, with shuffled labels as the control.",
        fields: [F("true", "area", "True statements"), F("false", "area", "False statements"), F("layer", "num", "Layer (optional)")] },
      ask: { label: "Ask", what: "The model's answer beside the probe's reading; disagreements are flagged.",
        fields: [F("questions", "json", "Questions [{question, answer_a, answer_b}]", { value: "[]" }), F("probe", "json", "Probe", { from: "train.probe" })] } } },
  faithfulness: { label: "Faithfulness", group: "Inspect", lead: "Did the answer use a hint the reasoning never mentions?",
    steps: {
      run: { label: "Run", what: "Each question with and without a planted hint: whether the answer moved to it, and whether the reasoning says so.",
        fields: [F("items", "json", "Items [{question, choices, hint_answer}]", { value: "[]" })] } } },
  inject: { label: "Inject and ask", group: "Inspect", lead: "Plant a concept in the residual stream and ask whether the model notices it.",
    steps: {
      concept: { label: "Concept", what: "The concept's vector: prompts about it against other words.", fields: [F("concept", "text", "Concept (a word)"),
        F("contrast", "area", "Contrast words (optional)"), F("layer", "num", "Layer (optional)")] },
      ask: { label: "Ask", what: "The model asked about its thoughts with the vector added, beside a random vector of the same size.",
        fields: [F("vector", "json", "Vector", { from: "concept.vector" }), F("layer", "num", "Layer", { from: "concept.layer" }),
          F("concept", "text", "Concept", { from: "concept.concept" }), F("strength", "nums", "Strengths", { value: "0, 2, 4, 8" }), F("trials", "num", "Trials", { value: 3 })] } } },
  classify: { label: "Classify", group: "Build", lead: "Train a tiny probe on the model's own activations and ship it as a head that runs for free.",
    steps: {
      train: { label: "Train", what: "A head per layer on labelled examples, scored held out beside shuffled labels.",
        fields: [F("examples", "json", "Examples {label: [texts]}", { value: '{"positive": [], "negative": []}' }), F("layer", "num", "Layer (optional)"),
          F("pooling", "pick", "Pooling", { choices: ["last", "mean"] })] },
      run: { label: "Run", what: "Label new texts with one partial forward pass.", fields: [F("head", "json", "Head", { from: "train.head" }), F("texts", "area", "Texts")] },
      export: { label: "Export", what: "Write the head as a small file to use without LLMInspect.", fields: [F("head", "json", "Head", { from: "train.head" }), F("path", "text", "Path (no extension)")] } } },
  rewrite: { label: "Rewrite a fact", group: "Build", lead: "Change an answer instead of removing it: one rank-one edit at one MLP layer.",
    steps: {
      locate: { label: "Locate", what: "Where the subject's state carries the answer, layer by layer.", fields: [F("prompt", "text", "Lead-in"), F("subject", "text", "Subject (in the lead-in)")] },
      edit: { label: "Edit", what: "The rank-one change that makes the new answer likely, with paraphrases and drift checked.",
        fields: [F("prompt", "text", "Lead-in", { from: "locate.prompt" }), F("subject", "text", "Subject", { from: "locate.subject" }), F("new", "text", "New answer"),
          F("layer", "num", "Layer (optional)", { from: "locate.best_layer" }), F("paraphrases", "area", "Paraphrases (optional)")] } } },
  transplant: { label: "Transplant", group: "Build", lead: "Carry a fine-tune or a direction from one model into another, such as a newer release.",
    steps: {
      plan: { label: "Plan", what: "What the fine-tune changed and whether it fits the loaded model.", fields: [F("base", "text", "Original base folder"), F("tuned", "text", "Its fine-tune's folder")] },
      carry: { label: "Carry", what: "The fine-tune's difference, kept as an edit on the loaded model.", fields: [F("base", "text", "Original base", { from: "plan.base" }),
        F("tuned", "text", "Fine-tune", { from: "plan.tuned" }), F("scale", "num", "Scale", { value: 1 }), F("low_rank", "num", "Low rank (optional)")] },
      direction: { label: "Direction", what: "A residual direction from another model, mapped through the shared vocabulary.", fields: [F("vector", "json", "Vector"),
        F("layer", "num", "Layer"), F("from_model", "text", "Its model's folder")] } } },
  shrink: { label: "Shrink", group: "Build", lead: "Drop the heads, neurons and experts your own prompts never use, and measure the cost.",
    steps: {
      profile: { label: "Profile", what: "How much each unit is used on your prompts.", fields: [F("texts", "area", "Your prompts"), F("chat", "bool", "As chat")] },
      prune: { label: "Prune", what: "Zero the least-used units, beside the same number at random.", fields: [F("texts", "area", "Your prompts"),
        F("fraction", "num", "Fraction to drop", { value: 0.1 }), F("heldout", "area", "Held-out prompts (optional)")] } } },
  cap: { label: "Cap", group: "Build", lead: "Keep a direction within the range it has on ordinary text, so a persona can't drift.",
    steps: {
      calibrate: { label: "Calibrate", what: "The direction's range on ordinary text.", fields: [F("vector", "json", "Vector"), F("layer", "num", "Layer")] },
      test: { label: "Test", what: "A push along the direction, with and without the cap.", fields: [F("vector", "json", "Vector", { from: "calibrate.vector" }),
        F("layer", "num", "Layer", { from: "calibrate.layer" }), F("low", "num", "Low", { from: "calibrate.low" }), F("high", "num", "High", { from: "calibrate.high" }),
        F("prompts", "area", "Prompts"), F("push", "num", "Push", { value: 8 })] },
      keep: { label: "Keep", what: "Cap the live model: every reply from now on keeps the direction within these bounds. Remove it under Steering in Watch.",
        fields: [F("name", "text", "Name", { value: "cap" }), F("vector", "json", "Vector", { from: "calibrate.vector" }), F("layer", "num", "Layer", { from: "calibrate.layer" }),
          F("low", "num", "Low", { from: "calibrate.low" }), F("high", "num", "High", { from: "calibrate.high" })],
        post: (b) => ({ url: "/llminspect/interventions", body: { direction: b.name || "cap", mode: "cap", vector: b.vector, layer: b.layer, value: b.low, high: b.high } }) } } },
};

const L = { verb: "sign", step: null, results: {}, inputs: {}, busy: false, error: "", catalog: null };

const get = (obj, path) => path.split(".").reduce((o, k) => (o == null ? o : o[k]), obj);

/** The value a field shows: what was typed, else what an earlier step returned, else its default. */
function valueOf(verb, step, f) {
  const typed = L.inputs[`${verb}.${step}.${f.name}`];
  if (typed != null) return typed;
  if (f.from) {
    const [s, ...rest] = f.from.split(".");
    const v = rest.length ? get(L.results[`${verb}.${s}`], rest.join(".")) ?? get(L.results[`${verb}.${s}`]?._body, rest.join(".")) : undefined;
    if (v != null) return typeof v === "object" ? JSON.stringify(v) : String(v);
  }
  return f.value != null ? String(f.value) : "";
}

function fieldHtml(verb, step, f) {
  const id = `lab-${f.name}`, v = valueOf(verb, step, f);
  const attrs = `id="${id}" data-field="${esc(f.name)}" aria-label="${esc(f.label)}"`;
  let input;
  if (f.kind === "area" || f.kind === "paras" || f.kind === "area1") input = `<textarea ${attrs} rows="${f.kind === "paras" ? 6 : 4}" spellcheck="false">${esc(v)}</textarea>`;
  else if (f.kind === "json") input = `<textarea ${attrs} rows="3" class="mono" spellcheck="false">${esc(v)}</textarea>`;
  else if (f.kind === "bool") input = `<input type="checkbox" ${attrs} ${v === "true" ? "checked" : ""}>`;
  else if (f.kind === "pick") input = `<select ${attrs}>${f.choices.map((c) => `<option value="${esc(c)}" ${c === v ? "selected" : ""}>${esc(c || "all")}</option>`).join("")}</select>`;
  else input = `<input ${attrs} ${f.kind === "num" ? 'inputmode="decimal"' : ""} value="${esc(v)}" autocomplete="off" spellcheck="false">`;
  return `<label class="lab-field ${f.kind === "bool" ? "inline" : ""}" for="${id}"><span>${esc(f.label)}</span>${input}</label>`;
}

/** A form's values as the step's JSON body; blank optional fields are left out so the step's defaults apply. */
function bodyOf(form, spec) {
  const body = {};
  for (const f of spec.fields) {
    const el = form.querySelector(`[data-field="${f.name}"]`);
    const raw = f.kind === "bool" ? el.checked : el.value.trim();
    if (raw === "" || raw == null) continue;
    if (f.kind === "num") body[f.name] = Number(raw);
    else if (f.kind === "nums") body[f.name] = raw.split(/[\s,]+/).filter(Boolean).map(Number);
    else if (f.kind === "area") body[f.name] = raw.split("\n").map((s) => s.trim()).filter(Boolean);
    else if (f.kind === "paras") body[f.name] = raw.split(/\n\s*\n/).map((s) => s.trim()).filter(Boolean);
    else if (f.kind === "json") body[f.name] = JSON.parse(raw);
    else if (f.kind === "pick") body[f.name] = f.list ? [raw] : raw;
    else body[f.name] = raw;
  }
  return body;
}

async function run(ev) {
  ev?.preventDefault();
  const spec = LAB[L.verb].steps[L.step];
  let body;
  try { body = bodyOf($("#lab-form"), spec); } catch (err) { L.error = `A field isn't valid JSON: ${err.message}`; render(); return; }
  L.busy = true; L.error = ""; render();
  try {
    const call = spec.post ? spec.post(body) : { url: `/llminspect/verbs/${L.verb}/${L.step}`, body };
    const r = await fetch(call.url, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(call.body) });
    const out = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(out.detail || `${r.status}`);
    L.results[`${L.verb}.${L.step}`] = { ...out, _body: body };
    if (out.edit) document.dispatchEvent(new CustomEvent("llminspect:workbench"));
  } catch (err) { L.error = err.message; }
  L.busy = false; render();
}

async function applyEdits() {
  try {
    const r = await fetch("/llminspect/workbench/apply", { method: "POST" });
    if (!r.ok) throw new Error((await r.json()).detail);
    noteBar("lab", "Applied: the edit is in the weights now. Restore original undoes it, in Tune.");
    document.dispatchEvent(new CustomEvent("llminspect:workbench"));
  } catch (err) { noteBar("lab", `Couldn't apply: ${err.message}`); }
}

// ---------------------------------------------------------------- the result, read

const HEAD_KEYS = ["reading", "verdict", "note", "summary_text"];
const fmt = (v) => (typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString() : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(3)) : String(v));
const label = (k) => k.replace(/_/g, " ");

function tableHtml(rowsIn) {
  const rows = rowsIn.slice(0, 60);
  const cols = [...new Set(rows.flatMap((r) => Object.keys(r)))].filter((c) => !Array.isArray(rows.find((r) => r[c] != null)?.[c]) || rows.every((r) => (r[c] || []).length < 6)).slice(0, 8);
  const cell = (v) => (v == null ? "" : typeof v === "object" ? `<code>${esc(JSON.stringify(v).slice(0, 120))}</code>` : typeof v === "boolean" ? (v ? "yes" : "no") : esc(fmt(v)));
  return `<div class="lab-table-wrap"><table class="lab-table"><thead><tr>${cols.map((c) => `<th>${esc(label(c))}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((r) => `<tr>${cols.map((c) => `<td>${cell(r[c])}</td>`).join("")}</tr>`).join("")}</tbody></table>
    ${rowsIn.length > rows.length ? `<p class="dim">${rowsIn.length - rows.length} more rows in the raw result.</p>` : ""}</div>`;
}

function resultHtml(out) {
  if (!out) return "";
  const { _body, edit, ...r } = out;
  const head = HEAD_KEYS.map((k) => r[k]).find((v) => typeof v === "string");
  const nums = Object.entries(r).filter(([k, v]) => typeof v === "number" && !k.startsWith("_")).slice(0, 8);
  const flat = (obj, pre = "") => Object.entries(obj || {}).flatMap(([k, v]) => (v && typeof v === "object" && !Array.isArray(v) && pre === ""
    ? flat(v, `${k} `) : typeof v === "number" ? [[`${pre}${k}`, v]] : []));
  const subNums = Object.entries(r).filter(([, v]) => v && typeof v === "object" && !Array.isArray(v)).flatMap(([k, v]) => flat({ [k]: v })).slice(0, 12);
  const tables = Object.entries(r).filter(([, v]) => Array.isArray(v) && v.length && typeof v[0] === "object" && !Array.isArray(v[0]));
  const lists = Object.entries(r).filter(([, v]) => Array.isArray(v) && v.length && typeof v[0] === "string");
  return `${head ? `<p class="lab-reading">${esc(head)}</p>` : ""}
    ${edit ? `<div class="lab-edit"><span>Kept as an edit in Tune's stack, not applied yet: <b>${esc(edit.summary)}</b></span><button type="button" class="tbtn" id="lab-apply">Apply now</button></div>` : ""}
    ${nums.length ? `<div class="lab-nums">${nums.map(([k, v]) => `<div><b>${esc(fmt(v))}</b><span>${esc(label(k))}</span></div>`).join("")}</div>` : ""}
    ${subNums.length ? `<dl class="lab-kv">${subNums.map(([k, v]) => `<dt>${esc(label(k))}</dt><dd>${esc(fmt(v))}</dd>`).join("")}</dl>` : ""}
    ${tables.map(([k, v]) => `<h3>${esc(label(k))}</h3>${tableHtml(v)}`).join("")}
    ${lists.map(([k, v]) => `<h3>${esc(label(k))}</h3><ul class="lab-list">${v.slice(0, 20).map((s) => `<li>${esc(s)}</li>`).join("")}</ul>`).join("")}
    <details class="lab-raw"><summary>Raw result</summary><pre>${esc(JSON.stringify(r, null, 1).slice(0, 60000))}</pre></details>`;
}

// ---------------------------------------------------------------- rendering

function render() {
  const verb = LAB[L.verb];
  if (!L.step || !verb.steps[L.step]) L.step = Object.keys(verb.steps)[0];
  const spec = verb.steps[L.step];
  const groups = [...new Set(Object.values(LAB).map((v) => v.group))];
  $("#lab-verbs").innerHTML = groups.map((g) => `<div class="lab-group"><span class="eyebrow">${esc(g)}</span>${Object.entries(LAB).filter(([, v]) => v.group === g)
    .map(([id, v]) => `<button type="button" class="lab-verb ${id === L.verb ? "on" : ""}" data-verb="${id}" aria-pressed="${id === L.verb}">${esc(v.label)}</button>`).join("")}</div>`).join("");
  $("#lab-title").textContent = verb.label;
  $("#lab-lead").textContent = verb.lead;
  $("#lab-steps").innerHTML = Object.entries(verb.steps).map(([id, s], i) => `<button type="button" role="tab" class="lab-step ${id === L.step ? "on" : ""}" data-step="${id}"
    aria-selected="${id === L.step}"><span class="lab-n">${i + 1}</span>${esc(s.label)}${L.results[`${L.verb}.${id}`] ? ' <span class="lab-done" aria-label="done">●</span>' : ""}</button>`).join("");
  $("#lab-form").innerHTML = `<p class="lab-what">${esc(spec.what)}</p>${spec.fields.map((f) => fieldHtml(L.verb, L.step, f)).join("")}
    <button type="submit" class="um-go" ${L.busy ? "disabled" : ""}>${L.busy ? "Running…" : `Run ${esc(spec.label)}`}</button>`;
  const out = L.results[`${L.verb}.${L.step}`];
  $("#lab-result").innerHTML = L.error ? `<p class="lab-error">${esc(L.error)}</p>` : L.busy ? `<div class="lab-wait"><span class="lab-pulse"></span>Running ${esc(spec.label)} on the loaded model…</div>`
    : out ? resultHtml(out) : `<div class="lab-empty"><span class="lab-big">${esc(verb.label)}</span><p>${esc(spec.what)}</p><p class="dim">Runs on the model loaded in Watch. Nothing changes in the weights until you apply an edit.</p></div>`;
  $("#lab-apply")?.addEventListener("click", applyEdits);
  refreshBar("lab");
}

function remember(ev) {
  const el = ev.target.closest("[data-field]");
  if (el) L.inputs[`${L.verb}.${L.step}.${el.dataset.field}`] = el.type === "checkbox" ? String(el.checked) : el.value;
}

export function openVerb(verb, step) {
  if (!LAB[verb]) return;
  L.verb = verb; L.step = step && LAB[verb].steps[step] ? step : null; L.error = "";
  render();
}

function boot() {
  $("#lab-verbs").addEventListener("click", (ev) => { const b = ev.target.closest("[data-verb]"); if (b) openVerb(b.dataset.verb); });
  $("#lab-steps").addEventListener("click", (ev) => { const b = ev.target.closest("[data-step]"); if (b) { L.step = b.dataset.step; L.error = ""; render(); } });
  $("#lab-form").addEventListener("input", remember);
  $("#lab-form").addEventListener("change", remember);
  $("#lab-form").addEventListener("submit", run);
  document.addEventListener("llminspect:step", (ev) => { if (LAB[ev.detail.verb]) openVerb(ev.detail.verb, ev.detail.step); });
  defineBar("lab", {
    groups: ["Run"],
    context: () => `<b>${esc(LAB[L.verb].label)}</b> · ${esc(LAB[L.verb].steps[L.step || Object.keys(LAB[L.verb].steps)[0]].label)}`,
    state: () => Object.fromEntries(Object.keys(L.results).map((k) => [`lab_${k.replace(".", "_")}`, true])),
    actions: [{ id: "run", group: "Run", label: "Run step", hint: "Run the step on the loaded model", need: () => (L.busy ? "the step running now to finish" : null), run: () => run() }],
  });
  render();
}
boot();
