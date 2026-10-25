// Ask the model from the UI: a composer in the Watch and Trace tabs that sends a chat request to the
// recorder's own /v1 endpoint. Every request is recorded like any other run, so its tokens stream into
// the transcript, the timeline and the unit graph live. A conversation started here continues in the
// same run until "New chat"; a run recorded by an agent harness is never appended to.
//
// Stop: while the composer's request runs its Send button is a Stop button, and the Trace dock shows
// a Stop whenever any request is generating (an agent harness's too). Both call /brainscan/cancel: the
// request finishes between two tokens with finish_reason "cancelled" and keeps what it generated.

const CHAT_PREFIX = "chat-";
const K = {
  busy: false, fresh: false, roots: [], settings: { think: "default", max: 512, temperature: 0 },
  runId: null, ctrl: null, stopping: false,  // the composer's own request
  generating: null, dockStopping: false, noteTimer: 0,  // any request (live from app.js)
};

try { Object.assign(K.settings, JSON.parse(localStorage.getItem("brainscan-chat") || "{}")); } catch { /* storage unavailable */ }
const save = () => { try { localStorage.setItem("brainscan-chat", JSON.stringify(K.settings)); } catch { /* storage unavailable */ } };

async function api(path, opts) {
  const res = await fetch(path, opts);
  const text = await res.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch { /* not JSON */ }
  if (!res.ok) { const err = new Error(body?.detail || body?.error?.message || text || `${res.status}`); err.status = res.status; throw err; }
  return body;
}

const currentRun = () => document.querySelector("#run-select")?.value || null;
const isChatRun = (id) => !!id && id.startsWith(CHAT_PREFIX);
const newRunId = () => {
  const d = new Date(), p = (n) => String(n).padStart(2, "0");
  return `${CHAT_PREFIX}${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
};

/** The conversation so far, from the run's recorded messages (user, system and assistant text). A
 * reply with no text (stopped, or cut off while thinking) stays in as an empty turn, so the recorder
 * lines the next request's messages up with the ones it already recorded. */
async function history(runId) {
  const run = await api(`/brainscan/runs/${encodeURIComponent(runId)}`);
  return [...(run.messages || [])]
    .sort((a, b) => a.t - b.t || a.i - b.i)
    .filter((m) => ["system", "user", "assistant"].includes(m.role) && ((m.content || "").length || m.generated))
    .map((m) => ({ role: m.role, content: m.content || "" }));
}

function setStatus(text, tone = "") {
  for (const root of K.roots) {
    const el = root.querySelector(".composer-status");
    el.textContent = text; el.className = `composer-status ${tone}`;
  }
}

function sync() {
  const where = K.fresh || !isChatRun(currentRun()) ? "starts a new chat" : "continues this chat";
  for (const root of K.roots) {
    const btn = root.querySelector(".composer-send");
    btn.className = `composer-send ${K.busy ? "btn stop" : "primary"}`;
    btn.textContent = K.busy ? (K.stopping ? "Stopping…" : "■ Stop") : "Send";
    btn.title = K.busy ? "Stop the answer here (Esc). The tokens generated so far are kept and recorded." : "Send (Enter)";
    btn.disabled = K.stopping;
    root.querySelector("textarea").placeholder = K.busy ? "The model is answering… Esc stops it" : `Ask the model (${where}) · Enter to send, Shift+Enter for a new line`;
    root.querySelector(".composer-think").value = K.settings.think;
    root.querySelector(".composer-max").value = K.settings.max;
    root.querySelector(".composer-temp").value = K.settings.temperature;
  }
  syncDock();
}

function syncDock() {
  const dock = document.querySelector("#an-stop");
  if (dock) {
    dock.hidden = !K.generating;
    dock.disabled = K.dockStopping;
    dock.textContent = K.dockStopping ? "Stopping…" : "■ Stop";
  }
}

const tokens = (n) => `${n} token${n === 1 ? "" : "s"}`;
const GENERATING = "Generating: the tokens stream into the transcript and the graph…";

function dockNote(text) {
  const el = document.querySelector("#an-stop-note");
  if (!el) return;
  el.textContent = text;
  clearTimeout(K.noteTimer);
  if (text) K.noteTimer = setTimeout(() => { el.textContent = ""; }, 8000);
}

/** Stop the composer's own request. If it hasn't reached the model yet, the fetch is dropped too. */
async function stop() {
  if (!K.busy || K.stopping) return;
  K.stopping = true; sync();
  setStatus("Stopping…");
  try {
    const got = await api("/brainscan/cancel", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ run: K.runId }) });
    if (!got.cancelled && !got.queued) K.ctrl?.abort();  // still on its way to the recorder
  } catch {
    K.ctrl?.abort();  // the server stops a request whose client went away
  }
}

/** Stop whatever is generating: the Trace dock's Stop, for agent-harness runs as well as chats. */
async function stopAny() {
  if (K.dockStopping) return;
  K.dockStopping = true; syncDock();
  try {
    const got = await api("/brainscan/cancel", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    if (!got.cancelled) { K.dockStopping = false; syncDock(); dockNote("Nothing was generating."); }
  } catch (err) {
    K.dockStopping = false; syncDock();
    dockNote(`Couldn't stop: ${err.message}`);
  }
}

async function send(root) {
  const box = root.querySelector("textarea");
  const text = box.value.trim();
  if (!text || K.busy) return;
  K.busy = true; sync();
  const continuing = !K.fresh && isChatRun(currentRun());
  const runId = continuing ? currentRun() : newRunId();
  K.runId = runId; K.ctrl = new AbortController();
  setStatus(continuing ? "Sending…" : "Starting a new chat…");
  try {
    const messages = [...(continuing ? await history(runId) : []), { role: "user", content: text }];
    // Open the run in every view as soon as it appears, and follow its tokens live.
    document.dispatchEvent(new CustomEvent("brainscan:expect-run", { detail: runId }));
    box.value = "";
    for (const r of K.roots) if (r !== root) r.querySelector("textarea").value = "";
    K.fresh = false;
    if (K.stopping) throw new DOMException("stopped", "AbortError");  // Stop while the history loaded
    setStatus(K.generating && K.generating.run !== runId ? "Waiting for the model: another request is generating…" : GENERATING);
    const kwargs = K.settings.think === "on" ? { enable_thinking: true } : K.settings.think === "off" ? { enable_thinking: false } : undefined;
    const res = await api("/v1/chat/completions", {
      method: "POST", signal: K.ctrl.signal,
      headers: { "Content-Type": "application/json", "X-Brainscan-Run": runId, "X-Brainscan-Label": "chat" },
      body: JSON.stringify({ messages, max_tokens: Number(K.settings.max) || 512, temperature: Number(K.settings.temperature) || 0,
        ...(kwargs ? { chat_template_kwargs: kwargs } : {}) }),
    });
    const n = res.usage?.completion_tokens ?? 0;
    const finish = res.choices?.[0]?.finish_reason || "stop";
    if (finish !== "cancelled") setStatus(`Done: ${tokens(n)} (${finish}). Click any word to see which units produced it.`, "ok");
    else if (n) setStatus(`Stopped: kept ${tokens(n)}, recorded with the run. Click any word to see which units produced it.`, "ok");
    else if (res.brainscan?.run) setStatus("Stopped before the model wrote anything; the part of the prompt it read is recorded with the run.", "ok");
    else {
      if (!box.value) box.value = text;  // it never reached the model: keep what was typed
      setStatus("Stopped before the model started on it; nothing was recorded.", "ok");
    }
  } catch (err) {
    if (!box.value) box.value = text;  // keep what was typed
    if (err.name === "AbortError") setStatus("Stopped before the model started on it.", "ok");
    else setStatus(err.status === 503 ? "The model is loading or switching; try again in a moment." : err.message, "error");
  } finally {
    K.busy = false; K.stopping = false; K.ctrl = null; sync();
  }
}

/** Mount a composer into `root` (an element with the data-composer attribute). */
function mount(root) {
  root.innerHTML = `
    <textarea rows="2" spellcheck="true" aria-label="Message to the model"></textarea>
    <div class="composer-row">
      <label title="Reasoning models: think before answering">Think
        <select class="composer-think"><option value="default">model default</option><option value="on">on</option><option value="off">off</option></select></label>
      <label title="Most tokens to generate">Max<input class="composer-max" type="number" min="1" max="4096" step="1"></label>
      <label title="0 is greedy (the same answer every time)">Temp<input class="composer-temp" type="number" min="0" max="2" step="0.1"></label>
      <button type="button" class="btn sm composer-new" title="Start a new conversation on the next message">New chat</button>
      <button type="button" class="primary composer-send">Send</button>
    </div>
    <div class="composer-status" role="status"></div>`;
  const box = root.querySelector("textarea");
  box.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey && !ev.isComposing) { ev.preventDefault(); send(root); }
    else if (ev.key === "Escape" && K.busy) { ev.preventDefault(); stop(); }
    ev.stopPropagation();  // typing here never triggers the graph's shortcuts
  });
  root.querySelector(".composer-send").onclick = () => (K.busy ? stop() : send(root));
  root.querySelector(".composer-new").onclick = () => { K.fresh = true; setStatus("The next message starts a new chat."); sync(); box.focus(); };
  root.querySelector(".composer-think").onchange = (ev) => { K.settings.think = ev.target.value; save(); sync(); };
  root.querySelector(".composer-max").onchange = (ev) => { K.settings.max = Number(ev.target.value) || 512; save(); sync(); };
  root.querySelector(".composer-temp").onchange = (ev) => { K.settings.temperature = Math.max(0, Number(ev.target.value) || 0); save(); sync(); };
  K.roots.push(root);
}

function boot() {
  document.querySelectorAll("[data-composer]").forEach(mount);
  document.querySelector("#run-select")?.addEventListener("change", sync);
  document.addEventListener("brainscan:run", sync);
  // Which request is generating comes from the live feed app.js reads.
  document.addEventListener("brainscan:generating", (ev) => {
    K.generating = ev.detail;
    if (!ev.detail) K.dockStopping = false;
    if (K.busy && !K.stopping && ev.detail?.run === K.runId) setStatus(GENERATING);  // its turn came
    syncDock();
  });
  document.addEventListener("brainscan:done", (ev) => {
    const d = ev.detail || {};
    if (d.finish === "cancelled") dockNote(`Stopped · kept ${tokens(d.generated ?? 0)}`);
  });
  const dock = document.querySelector("#an-stop");
  if (dock) dock.onclick = stopAny;
  sync();
}
boot();
