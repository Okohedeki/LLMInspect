// The public page (llminspect/public.py): two screens, Unmask and Sweep, and nothing that needs a loaded model.
// A link to /m/ORG/NAME opens that model's card straight away (body data-unmask); the address follows the model
// on show, so the address bar is always a link to share.
const $ = (sel) => document.querySelector(sel);
const VIEWS = ["unmask", "sweep"];

function show(view) {
  if (!VIEWS.includes(view)) return;
  for (const v of VIEWS) $(`#view-${v}`).hidden = v !== view;
  for (const b of document.querySelectorAll("[data-show]")) {
    const on = b.dataset.show === view;
    b.classList.toggle("on", on);
    b.setAttribute("aria-pressed", String(on));
  }
  document.dispatchEvent(new CustomEvent("llminspect:view", { detail: view }));
}

function theme() {
  try { const saved = localStorage.getItem("llminspect-theme"); if (saved) document.documentElement.dataset.theme = saved; } catch { /* storage unavailable */ }
  $("#theme-toggle").addEventListener("click", () => {
    const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("llminspect-theme", next); } catch { /* storage unavailable */ }
  });
}

function boot() {
  theme();
  document.addEventListener("click", (ev) => { const b = ev.target.closest("[data-show]"); if (b) show(b.dataset.show); });
  document.addEventListener("llminspect:show", (ev) => show(ev.detail));
  show("unmask");
}
boot();
