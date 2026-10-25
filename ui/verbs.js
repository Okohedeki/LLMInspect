// The actions: what you can do to the model. Each one runs as steps, and each step happens on one
// of the screens (Watch records, Trace locates and tests, Tune changes and ships) with the tools that
// step needs, named "tab:id" from the screens' bars. A step is marked done from what the screens
// report (st). The top bar picks the action; the bar of the screen on show draws its steps
// (ui/actionbar.js), and `focus` names the panel a step scrolls to. Explore is no action: each
// screen's full bar, as before.
import { defineVerbs, st } from "/ui/actionbar.js";

const recorded = () => !!st("run");
const applied = () => st("applied") > 0;
const word = () => st("word");
const quote = (w) => `“${w}”`;

const record = (what) => ({ id: "record", label: "Record", view: "scanner", done: recorded, what,
  tools: ["scanner:ask", "scanner:new", "scanner:model"] });

const measure = {
  id: "measure", label: "Measure", view: "weights", done: () => applied() && st("checked"),
  what: "See what else the edit changed: every saved check before → now, the chat asked again on both weights, the circuit beside the original, your own prompts with drift on neutral text.",
  tools: ["weights:checks", "weights:rerun", "weights:cmp", "workbench:compare"], also: ["weights:check"],
};

const exportStep = {
  id: "export", label: "Export", view: "workbench", focus: "#h-export",
  what: "Save the edited model as a Hugging Face checkpoint with every edit recorded, or restore the original weights exactly.",
  tools: ["workbench:export", "workbench:restore"],
};

const findDirection = (what) => ({ id: "direction", label: "Direction", view: "workbench", focus: "#h-find", done: () => st("directions") > 0, what,
  tools: ["workbench:find"] });

defineVerbs([
  {
    id: "explore", label: "Explore", explore: true,
    lead: "No action: the three screens with every tool they have.",
    steps: [
      { id: "watch", label: "Watch", view: "scanner" },
      { id: "trace", label: "Trace", view: "weights" },
      { id: "tune", label: "Tune", view: "workbench" },
      { id: "neurons", label: "Neurons", view: "neurons" },
    ],
  },
  {
    id: "ablate", group: "Adjust", label: "Ablate",
    lead: "Remove the units a word depends on (a fact, a habit, a phrasing), test it, keep it as a weight edit.",
    steps: [
      record("Get the model to say what you want gone: ask it below, or pick a run at the top."),
      { id: "locate", label: "Locate", view: "weights", done: () => !!word(),
        what: () => (word() ? `Tracing ${quote(word())}: the graph colors every unit by what it did to it. Select units to remove (shift-drag), or test the top 5.`
          : "Click the word you want gone in the conversation. The graph then colors every unit by what it did to it."),
        tools: ["weights:why", "weights:runner", "weights:paths", "weights:jump"], also: ["weights:forget", "neurons:knob-ablate"] },
      { id: "test", label: "Test", view: "weights", done: () => !!st("tested"),
        what: "Knock the units out and read the word again: its probability before → after, and what the model says instead. Nothing changes in the weights yet.",
        tools: ["weights:ko5", "weights:kosel", "weights:kopaths"] },
      { id: "apply", label: "Apply", view: "weights", done: applied,
        what: "Keep the test as a weight edit, applied now. Nothing is lost: Restore original undoes every edit exactly.",
        tools: ["weights:perm", "weights:lesion", "weights:restore"] },
      measure,
      exportStep,
    ],
  },
  {
    id: "scale", group: "Adjust", label: "Scale",
    lead: "Dampen or boost chosen heads and neuron clusters, test it on a word, keep it as a weight edit.",
    steps: [
      record("Get the model to write something the units you want to scale take part in: ask it below, or pick a run."),
      { id: "locate", label: "Locate", view: "weights", done: () => !!word() && st("selection") > 0,
        what: () => (!word() ? "Click a word in the conversation, then select the units to scale on the graph (shift-click or shift-drag)."
          : st("selection") ? `${st("selection")} unit${st("selection") === 1 ? "" : "s"} selected on ${quote(word())}. Test a scale next, or add more units.`
          : `Tracing ${quote(word())}. Select the units to scale on the graph: shift-click or shift-drag.`),
        tools: ["weights:why", "weights:paths", "weights:jump"] },
      { id: "test", label: "Test", view: "weights", done: () => !!st("tested"),
        what: "Scale the selection and read the word again: ×0.5 dampens, ×2 boosts, ×0 removes. Nothing changes in the weights yet.",
        tools: ["weights:scale", "weights:kosel"] },
      { id: "apply", label: "Apply", view: "weights", done: applied,
        what: "Keep the test as a weight edit (Make permanent), or scale a whole attention block, MLP block, head or neuron by number in Tune.",
        tools: ["weights:perm", "workbench:edit", "weights:restore"] },
      measure,
      exportStep,
    ],
  },
  {
    id: "steer", group: "Adjust", label: "Steer",
    lead: "Push a direction (a tone, a language, a trait) into the model while it writes. The weights don't change.",
    steps: [
      findDirection("Give two sets of examples, one per line, that differ only in the trait. Brainscan finds the direction between them at every layer; save the best layer to the direction bank."),
      { id: "steer", label: "Steer", view: "scanner", focus: "#h-steer", done: () => st("steering") > 0,
        what: "Pick the direction, a layer and a strength under Steering, then Add. Start small: 0.05 of the residual's size. Or steer a saved knob (a named set of neurons), or single neurons, in the Neurons view.",
        tools: ["scanner:steer", "scanner:unsteer"], also: ["neurons:knob", "neurons:go"] },
      { id: "run", label: "Run", view: "scanner", done: () => !!st("steered"),
        what: "Ask the model. Requests made with steering on are marked ▲steered in the run list.",
        tools: ["scanner:ask", "scanner:new"] },
      { id: "compare", label: "Compare", view: "scanner",
        what: "Read the direction's lane on the timeline, then remove the steering and ask the same thing again to see the difference. Steering isn't saved in the weights or in exports.",
        tools: ["scanner:fit", "scanner:unsteer", "scanner:ask"] },
    ],
  },
  {
    id: "probe", group: "Analyze", label: "Probe",
    lead: "Read a state on every token (formal or casual, sure or hedging, any contrast you can give examples of) beside a random direction as the control.",
    steps: [
      findDirection("Give two sets of examples, one per line, for the state you want to read. Brainscan scores the direction between them on held-out examples at every layer; save it."),
      record("Ask the model, or point an agent at the endpoint (see the guide). Every new token is read against every saved direction as it is recorded."),
      { id: "read", label: "Read", view: "scanner",
        what: "The z lane of the timeline shows each direction per token, with a dashed random direction as the control; the grey band is ±2. Click a token for its projections.",
        tools: ["scanner:fit", "scanner:live"] },
    ],
  },
  {
    id: "monitor", group: "Analyze", label: "Monitor",
    lead: "Flag an agent run when a probe fires: a direction's z-score past a threshold for some tokens in a row, with the random direction's count on the same rule beside it.",
    steps: [
      findDirection("The state to watch for, as a direction: two sets of examples, one per line, or one already in the bank. Save it."),
      { id: "rule", label: "Rule", view: "scanner", focus: "#h-monitors", done: () => st("monitors") > 0,
        what: "Pick the direction (or a neuron of a saved knob), a threshold and how many tokens in a row, then Add monitor. Or watch a whole knob. Every run recorded from now on is checked as it goes.",
        tools: ["scanner:monitor"], also: ["neurons:knob-monitor"] },
      record("Run the agent or ask the model. A flag shows on the timeline (⚑), in this bar and in the run list the moment the rule is met."),
      { id: "review", label: "Review", view: "scanner", focus: "#h-flags",
        what: "Each flag: where it began, its peak, and how often the random direction trips the same rule on this run. Jump to a flag, then trace its peak token to see what drove it.",
        tools: ["scanner:flags", "scanner:nextflag", "scanner:trace", "scanner:scanruns"] },
      { id: "explain", label: "Explain", view: "weights", done: () => !!st("word"),
        what: () => (st("word") ? `Tracing “${st("word")}”: where its logit came from, layer by layer, and the units behind it. If a few units carry it, test removing them (Ablate).`
          : "Trace a flagged token (Trace → on a flag, or click a word in the conversation): where its logit came from and the units behind it."),
        tools: ["weights:why", "weights:follow", "weights:paths"] },
    ],
  },
  {
    id: "compare", group: "Analyze", label: "Compare",
    lead: "What the weight edits did: the chat asked again on both weights, your own prompts with drift on neutral text, the circuit behind a word before and after.",
    steps: [
      { id: "change", label: "Change", view: "workbench", focus: "#h-edit", done: applied,
        what: "Apply at least one weight edit: from Ablate or Scale, or the edit form here.",
        tools: ["workbench:edit", "workbench:apply"] },
      { id: "rerun", label: "Re-run", view: "workbench", focus: "#h-compare",
        what: "Ask the open chat's question again on the original and the edited weights, side by side.",
        tools: ["workbench:rerun"] },
      { id: "prompts", label: "Prompts", view: "workbench", focus: "#h-compare",
        what: "Your own prompts on both weights (greedy, steering off), with KL and perplexity drift on neutral reference text.",
        tools: ["workbench:compare"] },
      { id: "inside", label: "Inside", view: "weights",
        what: "Click a word: with vs original on, every unit is colored by how the edits changed its effect on that word.",
        tools: ["weights:cmp", "weights:why"] },
    ],
  },
  {
    id: "check", group: "Analyze", label: "Check",
    lead: "Behavior tests for every change: after this context, the model should say this word. Each check reads original → now.",
    steps: [
      { id: "save", label: "Save", view: "weights", done: () => st("checks") > 0,
        what: "Click a word the model should keep saying, then Add as a check. Or type one in Tune under Check behavior.",
        tools: ["weights:check"], also: ["workbench:checks"] },
      { id: "run", label: "Run", view: "workbench", focus: "#h-checks", done: () => !!st("checked"),
        what: "Run every check on the original and the current weights: what an edit fixed and what it broke.",
        tools: ["workbench:checks"] },
    ],
  },
], [
  { label: "Cap", what: "Keep a direction within the range it has on ordinary text, so a persona can't drift." },
  { label: "Rewrite a fact", what: "Change an answer instead of removing it: one rank-one edit at one MLP layer." },
  { label: "Inject and ask", what: "Plant a concept in the residual stream and ask whether the model notices it." },
  { label: "Faithfulness", what: "Did the answer use a hint the reasoning never mentions?" },
  { label: "Audit", what: "Search, blind, for an objective a model was trained to pursue." },
]);
