// The guide: the actions first (what you can do to the model, each with its steps and a Start), then
// every tool on the three screens they run on (watch a run, trace it, tune the weights), each with a
// "Show me" that takes you there. Opens from the ? button or the ? key, and on the first visit.
import { actionList } from "/ui/actionbar.js";

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const SEEN = "brainscan-guide-seen";
const SCREEN = { scanner: "Watch", weights: "Trace", workbench: "Tune" };

const go = (view, then) => () => {
  document.dispatchEvent(new CustomEvent("brainscan:show", { detail: view }));
  if (then) requestAnimationFrame(() => setTimeout(then, 60));
};
const trace = (detail) => document.dispatchEvent(new CustomEvent("brainscan:trace", { detail }));
const scrollTo = (sel) => () => $(sel)?.scrollIntoView({ behavior: "smooth", block: "start" });

const STEPS = [
  {
    view: "scanner", n: 1, title: "Watch", lead: "Run any agent on the model and see every token's internals as it works.",
    tools: [
      { name: "The bar and Do…", key: "⌘K", what: "Each screen has a bar above the work. With an action picked it shows the action's steps, the step you're on, its tools and the next step; in Explore it shows every tool on the screen. Greyed tools say what they need first. Do… (⌘K or Ctrl+K) searches every action and tool and runs it.", act: () => document.querySelector("#do-btn")?.click() },
      { name: "Ask the model", what: "Type a message under the transcript (or in the Trace tab's conversation). Its tokens stream into the transcript, the timeline and the unit graph as they are generated; the chat continues until \"New chat\".", act: go("weights", () => document.querySelector("#tv-convo textarea")?.focus()) },
      { name: "Stop", key: "Esc", what: "While the model answers, Send becomes Stop (or press Esc in the box): the reply ends at the next token, and what it generated so far is kept and recorded. The Trace dock shows a Stop whenever any request is generating, an agent harness's included.", act: go("weights", () => ($("#an-stop:not([hidden])") || $("#tv-convo textarea"))?.focus()) },
      { name: "Choose a model", what: "Click the model name in the top bar to switch to any open-weights model already on this machine: Hugging Face checkpoints and Ollama tags. Nothing is downloaded.", act: () => document.dispatchEvent(new CustomEvent("brainscan:models")) },
      { name: "Connect an agent", what: "Point any OpenAI-compatible harness at the endpoint below. Each request becomes a recorded run.", endpoint: true },
      { name: "Transcript and timeline", what: "Messages, tool calls and results lined up with a per-token timeline: events, logit lens by layer, surprisal, projections onto directions.", act: go("scanner") },
      { name: "Token inspector", what: "Click any token to see what the model predicted there, layer by layer.", act: go("scanner", scrollTo("#h-inspector")) },
      { name: "Monitors", what: "A rule on a direction: its z-score past a threshold for some tokens in a row. Runs are checked as they are recorded (a ⚑ on the timeline, in the bar and in the run list) and older runs on a scan. Each run's flags show beside how often the random direction trips the same rule.", act: go("scanner", scrollTo("#h-monitors")) },
      { name: "Steering", what: "Add, ablate or clamp a direction at a layer while the model generates. Requests made with steering on are marked.", act: go("scanner", scrollTo("#h-steer")) },
    ],
  },
  {
    view: "weights", n: 2, title: "Trace", lead: "Open the model as a graph of attention heads and MLP neuron clusters, and follow a token through it.",
    tools: [
      { name: "Why did it write that?", key: "W", what: "Click any word in the conversation: the graph colors by what each unit did to it, and a panel shows where its logit came from layer by layer, the layer where the model first had it as its top guess, the units that pushed it up or down, and the runners-up.", act: go("weights", () => trace({ mode: "attr" })) },
      { name: "Forget this fact", what: "A knockout made permanent, guided. Start from a traced word, or type a fact (a lead-in and the answer) from Tune or Do…. Brainscan finds the units that carry it after every way of asking for it, ranked two ways: the most direct, and, given facts it should keep, the most specific (units that push the answer more than any fact to keep, added only while every fact to keep survives). It shows answers with and without them, measures the fact against random units, facts to keep and neutral text, then removes the units from the weights if you keep it.", act: go("weights", () => trace({ mode: "attr" })) },
      { name: "Knockout and scale tests", what: "Remove units (the top ones, your selection, or the units on shown paths), or scale the selection (×0.5 dampens, ×2 boosts), and re-run: the token's probability before → after, its new rank and the new top guesses, with the units' total effect next to their direct effect. \"Make permanent\" turns a test into one weight edit, applied at once.", act: go("weights", () => trace({ mode: "attr" })) },
      { name: "Edits beside the trace", key: "E", what: "The Edits panel in Trace: apply or restore the edit stack, re-run the open chat on the edited weights, and run behavior checks, without leaving the graph.", act: go("weights", () => document.querySelector("#gv-edits-btn")?.click()) },
      { name: "Live unit graph", what: "Every head and neuron cluster, lit by how much it wrote at the current token. Scroll to zoom, drag to pan, replay the run on the timeline.", act: go("weights") },
      { name: "Neurons", what: "A secondary view for single MLP neurons (L5 N120): what its write pushes (logit lens), what steering it does to the predictions (steering effect at ±dose), where it fires, and the neurons it feeds and is fed by. Steer one or several with dose scaling and answer a prompt at several dose levels, or steer the chat. Open one from a cluster's neurons in Trace.", act: go("neurons", () => document.querySelector("#nv-q")?.focus()) },
      { name: "Explain the next token", key: "A", what: "Color each unit by how much it pushed the next token up or down. Click a runner-up token to explain it instead.", act: go("weights", () => trace({ mode: "attr" })) },
      { name: "Before and after an edit", what: "With weight edits applied, turn on \"vs original\" in Explain: each unit is colored by how the edit changed its effect on the token, the dock shows the probability before → after, and Paths traces the circuit behind the change.", act: go("weights", () => trace({ mode: "attr" })) },
      { name: "Path explorer", key: "P", what: "Strongest routes out of or into a selection, between selected units, or the circuit behind the next token. Select all, frame, copy as JSON.", act: go("weights", () => trace({ paths: "explain" })) },
      { name: "Select many units", key: "⇧", what: "Shift-click to add a unit, shift-drag a box to select a region, alt-drag to remove. Esc clears.", act: go("weights") },
      { name: "Inside a unit", what: "Open a unit for its weights, what it writes, its connections, and at the current token: a head's attention pattern or a cluster's neurons, expandable in the graph.", act: go("weights", () => trace({ palette: true })) },
    ],
  },
  {
    view: "workbench", n: 3, title: "Tune", lead: "Change the weights, measure what changed, and export the variant.",
    tools: [
      { name: "Find a direction", what: "Two sets of examples give a direction in the residual stream, scored at every layer on held-out examples.", act: go("workbench", scrollTo("#h-find")) },
      { name: "Edit weights", what: "Remove a direction from the weights, or scale a block, head or neuron (a lesion at 0). Lesion any unit straight from the Trace graph, then compare its circuit with the original there.", act: go("workbench", scrollTo("#h-edit")) },
      { name: "Compare", what: "Re-run the open chat on the original and edited weights and trace the first word that changed, or run your own prompts side by side with side-effect checks on neutral text.", act: go("workbench", scrollTo("#h-compare")) },
      { name: "Check behavior", what: "Save what the model should say next, from any traced word (\"Add as a check\") or typed. With edits applied, each check shows its probability on the original weights → now: what an edit fixed and what it broke.", act: go("workbench", scrollTo("#h-checks")) },
      { name: "Export", what: "Save the edited model as a Hugging Face checkpoint with every edit recorded.", act: go("workbench", scrollTo("#h-export")) },
    ],
  },
];

const KEYS = [
  ["?", "this guide"], ["⌘K", "do… any action"], ["/", "jump to a unit"], ["A", "explain next token"], ["P", "paths"],
  ["W", "why this token"], ["E", "weight edits"], ["T", "show/hide text"], ["Space", "replay"], ["← →", "step tokens"], ["F", "fit graph"], ["Esc", "clear / close"],
];

function render() {
  const endpoint = `${location.origin}/v1`;
  const { list, planned } = actionList();
  const acts = list.filter((v) => !v.explore);
  $("#guide-body").innerHTML = `
    <h3 class="guide-sec">Actions: what you can do to the model</h3>
    <div class="guide-acts">${acts.map((v) => `
      <section class="guide-act">
        <header><span class="eyebrow">${esc(v.group || "")}</span><h3>${esc(v.label)}</h3></header>
        <p>${esc(v.lead)}</p>
        <p class="ga-steps">${v.steps.map((s, i) => `<b>${i + 1} ${esc(s.label)}</b> <span>${SCREEN[s.view]}</span>`).join(" › ")}</p>
        <button type="button" class="link" data-pick="${esc(v.id)}">Start ${esc(v.label.toLowerCase())} →</button>
      </section>`).join("")}
    </div>
    ${planned.length ? `<p class="guide-planned"><b>Planned:</b> ${planned.map((p) => `<span title="${esc(p.what)}">${esc(p.label)}</span>`).join(" · ")}</p>` : ""}
    <h3 class="guide-sec">The screens the steps run on</h3>
    <div class="guide-steps">${STEPS.map((s, si) => `
      <section class="guide-step">
        <header><span class="step-n">${s.n}</span><div><h3>${s.title}</h3><p>${esc(s.lead)}</p></div></header>
        <ul>${s.tools.map((t, ti) => `<li>
          <div class="tool-head"><b>${esc(t.name)}</b>${t.key ? `<kbd>${esc(t.key)}</kbd>` : ""}</div>
          <p>${esc(t.what)}</p>
          ${t.endpoint ? `<div class="endpoint"><code>${esc(endpoint)}</code><button type="button" class="link" data-copy="${esc(endpoint)}">Copy</button></div>`
            : `<button type="button" class="link" data-step="${si}" data-tool="${ti}">Show me →</button>`}
        </li>`).join("")}</ul>
      </section>`).join("")}
    </div>
    <div class="guide-keys">${KEYS.map(([k, v]) => `<span><kbd>${esc(k)}</kbd> ${esc(v)}</span>`).join("")}</div>`;
  $("#guide-body").querySelectorAll("[data-pick]").forEach((b) => {
    b.onclick = () => { close(); document.dispatchEvent(new CustomEvent("brainscan:pick", { detail: b.dataset.pick })); };
  });
  $("#guide-body").querySelectorAll("[data-step]").forEach((b) => {
    b.onclick = () => { close(); STEPS[Number(b.dataset.step)].tools[Number(b.dataset.tool)].act(); };
  });
  $("#guide-body").querySelectorAll("[data-copy]").forEach((b) => {
    b.onclick = async () => {
      try { await navigator.clipboard.writeText(b.dataset.copy); b.textContent = "Copied"; } catch { b.textContent = "Select and copy"; }
      setTimeout(() => { b.textContent = "Copy"; }, 1600);
    };
  });
}

let lastFocus = null;
function open() {
  render();
  lastFocus = document.activeElement;
  $("#guide").hidden = false;
  $("#guide-close").focus();
  try { localStorage.setItem(SEEN, "1"); } catch { /* storage unavailable */ }
}
function close() {
  if ($("#guide").hidden) return;
  $("#guide").hidden = true;
  lastFocus?.focus?.();
}

function boot() {
  $("#help-btn").onclick = open;
  $("#guide-close").onclick = close;
  $("#guide").addEventListener("pointerdown", (ev) => { if (ev.target.id === "guide") close(); });
  document.addEventListener("keydown", (ev) => {
    if (!$("#guide").hidden && ev.key === "Escape") { ev.stopImmediatePropagation(); close(); return; }
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || "") || document.activeElement?.isContentEditable;
    if (!typing && (ev.key === "?" || (ev.key === "/" && ev.shiftKey))) { ev.preventDefault(); if ($("#guide").hidden) open(); else close(); }
  }, true);
  const seen = (() => { try { return localStorage.getItem(SEEN); } catch { return "1"; } })();
  if (!seen) open();
}
boot();
