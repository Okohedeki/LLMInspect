// The model picker: open-weights models already on this machine, from the model badge in the top
// bar. Loading one unloads the current model; Brainscan never downloads anything.

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
async function api(path, opts) {
  const res = await fetch(path, opts);
  const text = await res.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch { /* not JSON */ }
  if (!res.ok) { const err = new Error(body?.detail || text || `${res.status}`); err.status = res.status; throw err; }
  return body;
}

const M = { open: false, data: null, error: null, confirm: null, sending: false, timer: 0, seq: 0 };

const params = (b) => (b == null ? null : b >= 1 ? `${b.toFixed(b >= 10 ? 0 : 1)}B` : `${Math.round(b * 1000)}M`);
const nameOf = (spec) => M.data?.models.find((m) => m.spec === spec)?.name || spec;

async function load() {
  const seq = ++M.seq;
  try { M.data = await api("/brainscan/models"); M.error = null; } catch (err) { M.error = err.message; }
  if (seq !== M.seq) return;
  render();
  clearTimeout(M.timer);
  if (M.open && M.data?.status === "loading") M.timer = setTimeout(load, 1200);  // follow the switch
}

function render() {
  const body = $("#mp-body"), d = M.data;
  if (!d) { body.innerHTML = M.error ? `<p class="empty">${esc(M.error)}</p>` : `<div class="skeleton short"></div><div class="skeleton"></div>`; return; }
  const loading = d.status === "loading", cur = d.models.find((m) => m.current);
  const budget = d.budget_gb != null ? `${d.budget_gb} GB of GPU memory free once the current model is unloaded.` : "Models load on the CPU.";
  let status = "";
  if (loading) status = `<div class="mp-status"><span class="spinner sm"></span><span>${esc(d.step || "Loading…")}</span></div>`;
  else if (d.switch?.error) {
    status = `<div class="mp-status err"><b>Couldn't load ${esc(nameOf(d.switch.to))}.</b> ${esc(d.switch.error)}
      ${d.switch.restored ? ` Reloaded ${esc(nameOf(d.switch.restored))}.` : ""}</div>`;
  } else if (d.status === "error") status = `<div class="mp-status err">${esc(d.error || "The recorder failed to load a model.")}</div>`;
  if (M.error) status += `<div class="mp-status err">${esc(M.error)}</div>`;

  const rows = d.models.map((m, i) => {
    const tags = [m.source === "ollama" ? "Ollama" : "checkpoint", params(m.params_b), m.n_layers ? `${m.n_layers} layers` : null,
      m.need_gb && d.budget_gb != null ? `needs ~${m.need_gb} GB` : null].filter(Boolean).map((t) => `<span class="chip">${esc(t)}</span>`).join("");
    const badge = m.current ? `<span class="pill ok">loaded</span>`
      : m.supported === false ? `<span class="pill warn">can't load</span>`
        : m.fits === false ? `<span class="pill warn" title="${esc(budget)}">too big for this GPU</span>` : "";
    const blocked = loading || M.sending || m.supported === false || m.fits === false;
    const action = m.current ? "" : `<button type="button" class="btn sm" data-i="${i}" ${blocked ? "disabled" : ""}>Load</button>`;
    const confirm = M.confirm === m.spec ? `<div class="mp-confirm" role="alert">
        <p>Unload <b>${esc(cur?.name || "the current model")}</b> and load <b>${esc(m.name)}</b>? Steering and any applied
        weight edits are dropped (the files on disk don't change), and agent requests get a 503 until it has loaded.
        ${m.converted ? "" : "This Ollama tag is converted to bf16 first, which takes several minutes the first time."}</p>
        <div class="row"><button type="button" class="primary" data-go="${i}">Load ${esc(m.name)}</button>
        <button type="button" class="btn" data-cancel>Cancel</button></div></div>` : "";
    return `<li class="${m.current ? "current" : ""}">
      <div class="mp-row"><div class="mp-name"><b>${esc(m.name)}</b>${badge}</div>${action}</div>
      <div class="chips">${tags}</div>
      ${m.notes?.length ? `<ul class="mp-notes">${m.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>` : ""}
      ${m.path ? `<div class="mp-path mono" title="${esc(m.path)}">${esc(m.path)}</div>` : ""}
      ${confirm}</li>`;
  }).join("");

  body.innerHTML = `
    <div class="mp-head"><div><b>${esc(cur?.name || (loading ? "Loading…" : "No model loaded"))}</b>${cur ? ` <span class="dim">is loaded</span>` : ""}</div>
      <p class="hint">${esc(budget)}</p></div>
    ${status}
    <ul class="mp-list">${rows || `<li class="empty">No models found in the folders below.</li>`}</ul>
    <p class="hint mp-foot">Searched ${d.roots.map((r) => `<code>${esc(r)}</code>`).join(", ")} and the local Ollama store.
      Add folders with <code>BRAINSCAN_MODEL_DIRS</code> in brainscan.env. Brainscan never downloads models: to add
      one from Hugging Face, run <code>hf download Org/Name --local-dir ~/.cache/brainscan/models/Org--Name</code> in WSL,
      then reopen this list.</p>`;

  body.querySelectorAll("button[data-i]").forEach((b) => { b.onclick = () => { M.confirm = d.models[Number(b.dataset.i)].spec; render(); }; });
  body.querySelector("[data-cancel]")?.addEventListener("click", () => { M.confirm = null; render(); });
  body.querySelector("[data-go]")?.addEventListener("click", (ev) => switchTo(d.models[Number(ev.target.dataset.go)].spec));
}

async function switchTo(spec) {
  M.sending = true; M.confirm = null; M.error = null; render();
  try {
    await api("/brainscan/model", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ spec }) });
  } catch (err) { M.error = err.message; }
  M.sending = false;
  load();
}

function place() {
  const r = $("#model-badge").getBoundingClientRect(), box = $("#model-picker");
  box.style.top = `${r.bottom + 8}px`;
  box.style.left = `${Math.max(12, Math.min(r.left, innerWidth - box.offsetWidth - 12))}px`;
}

function open() {
  M.open = true;
  $("#model-picker").hidden = false;
  $("#model-badge").setAttribute("aria-expanded", "true");
  place();
  render();
  load();
}
function close() {
  M.open = false; M.confirm = null;
  clearTimeout(M.timer);
  $("#model-picker").hidden = true;
  $("#model-badge").setAttribute("aria-expanded", "false");
}

function boot() {
  $("#model-badge").addEventListener("click", () => (M.open ? close() : open()));
  $("#mp-close").onclick = close;
  document.addEventListener("brainscan:models", open);  // from the guide
  document.addEventListener("pointerdown", (ev) => {
    if (M.open && !ev.target.closest("#model-picker, #model-badge")) close();
  });
  document.addEventListener("keydown", (ev) => {
    if (M.open && ev.key === "Escape") { ev.stopImmediatePropagation(); close(); }
  }, true);
  addEventListener("resize", () => { if (M.open) place(); });
}
boot();
