// Demo scenes for recorder.mjs: one video per action, after an overview, and a last one on the
// screens and Do…. Every function here is serialized and evaluated in the page, so it can't close
// over anything in this module: HELPER installs window.demo (a visible cursor, captions, title
// cards, real mouse input, app helpers), and each scene's setup and script use it. Captions quote
// the numbers the app shows, so they hold for whichever model the recorder serves.

export function HELPER() {
  if (window.demo) return true;
  const style = document.createElement("style");
  style.textContent = `
    #demo-cursor { position: fixed; left: 0; top: 0; width: 26px; height: 26px; margin: -13px 0 0 -13px; border-radius: 50%;
      background: rgba(255,255,255,0.22); border: 2.5px solid #fff; z-index: 2147483647; pointer-events: none;
      box-shadow: 0 0 0 4px rgba(139,124,255,0.55), 0 6px 18px rgba(0,0,0,0.55); transform: translate(960px, 540px); }
    #demo-cursor.press { animation: demo-press 0.45s ease-out; }
    @keyframes demo-press { 0% { box-shadow: 0 0 0 4px rgba(139,124,255,0.6), 0 6px 18px rgba(0,0,0,0.55); }
      45% { box-shadow: 0 0 0 18px rgba(139,124,255,0.25), 0 6px 18px rgba(0,0,0,0.55); }
      100% { box-shadow: 0 0 0 4px rgba(139,124,255,0.6), 0 6px 18px rgba(0,0,0,0.55); } }
    #demo-cap { position: fixed; left: 50%; bottom: 150px; transform: translate(-50%, 12px); max-width: 1120px; width: max-content;
      padding: 16px 26px; border-radius: 16px; background: rgba(9, 10, 16, 0.93); color: #fff; text-align: center;
      font: 600 24px/1.38 Inter, system-ui, sans-serif; border: 1px solid rgba(255,255,255,0.16);
      box-shadow: 0 16px 48px rgba(0,0,0,0.6); z-index: 2147483646; pointer-events: none; opacity: 0;
      transition: opacity 0.35s ease, transform 0.35s ease; }
    #demo-cap.top { bottom: auto; top: 196px; }  /* below the top bar and the action's step bar */
    #demo-cap.on { opacity: 1; transform: translate(-50%, 0); }
    #demo-cap small { display: block; margin-top: 6px; font: 500 18px/1.42 Inter, system-ui, sans-serif; color: #c9c9da; }
    #demo-title { position: fixed; inset: 0; z-index: 2147483645; display: flex; flex-direction: column; align-items: center;
      justify-content: center; gap: 16px; background: radial-gradient(900px 520px at 50% 45%, #1c1745, #07080b 70%);
      color: #fff; opacity: 0; transition: opacity 0.6s ease; pointer-events: none; text-align: center; }
    #demo-title.on { opacity: 1; }
    #demo-title .k { font: 700 16px Inter, system-ui, sans-serif; letter-spacing: 0.24em; text-transform: uppercase; color: #a99cff; }
    #demo-title .t { font: 800 68px/1.1 Inter, system-ui, sans-serif; letter-spacing: -0.02em; max-width: 1500px; }
    #demo-title .s { font: 500 28px/1.4 Inter, system-ui, sans-serif; color: #c9c9da; max-width: 1300px; }
  `;
  document.head.appendChild(style);
  const cursor = Object.assign(document.createElement("div"), { id: "demo-cursor" });
  const cap = Object.assign(document.createElement("div"), { id: "demo-cap" });
  const title = Object.assign(document.createElement("div"), { id: "demo-title" });
  document.body.append(cursor, cap, title);

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const ease = (k) => (k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2);
  const q = (sel) => document.querySelector(sel);
  const pos = { x: 960, y: 540 };
  const place = (x, y) => { pos.x = x; pos.y = y; cursor.style.transform = `translate(${x}px, ${y}px)`; };
  const center = (el) => { const r = el.getBoundingClientRect(); return { x: r.left + Math.min(r.width / 2, 150), y: r.top + Math.min(r.height / 2, 60) }; };
  const words = (s) => String(s).replace(/<[^>]+>/g, " ").split(/\s+/).filter(Boolean).length;

  // Real mouse input goes through the recorder (Input.dispatchMouseEvent), which acks when done.
  let seq = 0;
  const acks = new Map();
  window.__demoAck = (id) => { acks.get(id)?.(); acks.delete(id); };
  const input = (events) => new Promise((res) => {
    const id = ++seq;
    acks.set(id, res);
    window.__demoInput(JSON.stringify({ id, events }));
    setTimeout(() => { if (acks.has(id)) { acks.delete(id); res(); } }, 3000);
  });

  const d = window.demo = {
    sleep, q,
    async moveXY(x, y, ms = 750) {
      cursor.style.transition = `transform ${ms}ms cubic-bezier(0.22, 0.8, 0.24, 1)`;
      place(x, y);
      await sleep(ms + 60);
    },
    /** Scroll el into view in its own scroll container if it isn't visible. */
    async reveal(el) {
      let p = el.parentElement;
      while (p && p !== document.body) {
        const st = getComputedStyle(p);
        if (/(auto|scroll)/.test(st.overflowY) && p.scrollHeight > p.clientHeight) break;
        p = p.parentElement;
      }
      const r = el.getBoundingClientRect();
      const box = p && p !== document.body ? p.getBoundingClientRect() : { top: 0, bottom: innerHeight };
      if (r.top < box.top + 8 || r.bottom > box.bottom - 8) {
        el.scrollIntoView({ behavior: "smooth", block: r.height > (box.bottom - box.top) * 0.6 ? "start" : "center" });
        await sleep(750);
      }
    },
    async move(el, ms = 750) {
      if (!el) throw new Error("demo.move: missing element");
      await d.reveal(el);
      const p = center(el);
      await d.moveXY(p.x, p.y, ms);
    },
    press() { cursor.classList.remove("press"); void cursor.offsetWidth; cursor.classList.add("press"); },
    async click(el, after = 450) {
      if (!el) throw new Error("demo.click: missing element");
      await d.move(el);
      d.press();
      if (el.focus) el.focus({ preventScroll: true });
      if (typeof el.click === "function") el.click(); else el.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await sleep(after);
    },
    async hover(el, ms = 650) {
      await d.move(el, ms);
      el.dispatchEvent(new MouseEvent("mouseenter"));
      el.dispatchEvent(new MouseEvent("mouseover", { bubbles: true }));
    },
    unhover(el) { el.dispatchEvent(new MouseEvent("mouseleave")); el.dispatchEvent(new MouseEvent("mouseout", { bubbles: true })); },
    async type(el, text, perChar = 34) {
      el.focus({ preventScroll: true });
      for (const ch of text) {
        el.value += ch;
        el.dispatchEvent(new Event("input", { bubbles: true }));
        await sleep(perChar + Math.random() * 28);
      }
    },
    set(el, value) { el.value = value; el.dispatchEvent(new Event("input", { bubbles: true })); el.dispatchEvent(new Event("change", { bubbles: true })); },
    key(key, opts = {}, target = document) { target.dispatchEvent(new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true, ...opts })); },

    // real mouse input (canvas hover, drag, wheel)
    async realMove(x, y, ms = 700, extra = {}) {
      cursor.style.transition = "none";
      const from = { ...pos }, steps = Math.max(6, Math.round(ms / 22));
      for (let i = 1; i <= steps; i++) {
        const k = ease(i / steps), px = from.x + (x - from.x) * k, py = from.y + (y - from.y) * k;
        place(px, py);
        await input([{ type: "mouseMoved", x: px, y: py, ...extra }]);
        await sleep(10);
      }
    },
    async realClick(x, y, modifiers = 0) {
      await d.realMove(x, y);
      d.press();
      await input([{ type: "mousePressed", x, y, button: "left", buttons: 1, clickCount: 1, modifiers },
        { type: "mouseReleased", x, y, button: "left", buttons: 0, clickCount: 1, modifiers }]);
      await sleep(400);
    },
    async realDrag(x0, y0, x1, y1, modifiers = 0, ms = 1100) {
      await d.realMove(x0, y0);
      await input([{ type: "mousePressed", x: x0, y: y0, button: "left", buttons: 1, clickCount: 1, modifiers }]);
      await d.realMove(x1, y1, ms, { button: "left", buttons: 1, modifiers });
      await sleep(150);
      await input([{ type: "mouseReleased", x: x1, y: y1, button: "left", buttons: 0, clickCount: 1, modifiers }]);
      await sleep(300);
    },
    async wheel(x, y, dy, times = 1, gap = 120) {
      await d.realMove(x, y, 450);
      for (let i = 0; i < times; i++) { await input([{ type: "mouseWheel", x, y, deltaX: 0, deltaY: dy }]); await sleep(gap); }
    },

    // captions and title cards
    async cap(text, sub = "", where = "bottom") {
      if (cap.classList.contains("on")) { cap.classList.remove("on"); await sleep(300); }
      cap.classList.toggle("top", where === "top");
      cap.innerHTML = text + (sub ? `<small>${sub}</small>` : "");
      cap.classList.add("on");
      await sleep(400);
    },
    /** A caption held long enough to read. */
    async say(text, sub = "", where = "bottom") {
      await d.cap(text, sub, where);
      await sleep(1100 + (words(text) + words(sub)) * 260);
    },
    async hide() { cap.classList.remove("on"); await sleep(450); },
    async title(text, sub = "", ms = 2800) {
      await d.hide();
      title.innerHTML = `<div class="k">Brainscan</div><div class="t">${text}</div>${sub ? `<div class="s">${sub}</div>` : ""}`;
      title.classList.add("on");
      await sleep(ms);
      title.classList.remove("on");
      await sleep(700);
    },
    async until(fn, timeout = 60000, what = "") {
      const t0 = performance.now();
      while (performance.now() - t0 < timeout) {
        try { if (await fn()) return true; } catch { /* not yet */ }
        await sleep(200);
      }
      throw new Error(`demo.until timed out: ${what || fn.toString().slice(0, 120)}`);
    },

    // app helpers
    CAPITAL: "What is the capital of Australia? Answer in one sentence.",
    show(view) { document.dispatchEvent(new CustomEvent("brainscan:show", { detail: view })); },
    /** The bar of the screen on show, and its parts: a tool by its label, a step chip, Next. */
    bar() { return [...document.querySelectorAll(".act-bar")].find((b) => b.offsetParent); },
    tool(label) { return [...d.bar().querySelectorAll(".ab-btn")].find((b) => b.textContent.trim().startsWith(label)) || null; },
    stepChip(label) { return [...d.bar().querySelectorAll(".ab-step")].find((b) => b.textContent.includes(label)) || null; },
    next() { return d.bar().querySelector("[data-next]"); },
    async pick(id) { await d.click(q(`#verbs [data-verb="${id}"]`), 900); },
    /** Long model output, cut for a caption: its first sentence, or the first n characters. */
    short(text, n = 110) {
      const t = String(text).replace(/\s+/g, " ").trim();
      const end = t.search(/[.!?](\s|$)/);
      const cut = end > 0 && end < n ? t.slice(0, end + 1) : t.slice(0, n);
      return cut.length < t.length ? `${cut.replace(/[.!?]$/, "")}…` : cut;
    },
    /** A knockout or scale test's numbers, and what the model would say instead. */
    koNums(li) { return li.querySelector(".ko-nums").innerText.replace(/\s+/g, " ").split(" logit")[0]; },
    mostLikely(li) { return ((li.querySelector(".hint").innerText.match(/Most likely now: (.*)\./) || [])[1] || "").replace(/·(?=\S)/g, ""); },
    /** Ask in a composer (a new chat, greedy, thinking off) and return the answer. */
    async ask(c, text) {
      await d.click(c.querySelector(".composer-new"));
      d.set(c.querySelector(".composer-think"), "off");
      d.set(c.querySelector(".composer-max"), "80");
      d.set(c.querySelector(".composer-temp"), "0");
      const ta = c.querySelector("textarea");
      await d.click(ta);
      await d.type(ta, text);
      await sleep(400);
      d.key("Enter", {}, ta);
      await d.until(() => /^Done/.test(c.querySelector(".composer-status").innerText), 180000, "the answer");
      await sleep(1500);
      const run = await d.api("GET", `/brainscan/runs/${q("#run-select").value}`);
      return ([...run.messages].reverse().find((m) => m.role === "assistant")?.content || "").trim();
    },
    /** The latest chat that asked CAPITAL (the ablate scene makes one). */
    async capitalChat() {
      const runs = await (await fetch("/brainscan/runs")).json();
      for (const r of runs.filter((x) => x.id.startsWith("chat-") && !x.id.includes("-edited") && (x.tokens || 0) > 0).slice(0, 12)) {
        const run = await d.api("GET", `/brainscan/runs/${r.id}`);
        if (run.messages.some((m) => m.role === "user" && m.content === d.CAPITAL)) return r.id;
      }
      return d.latestChat();
    },
    /** The last token in the Trace conversation whose text is `text`. */
    word(text) { return [...document.querySelectorAll("#tv-convo-body .ct")].reverse().find((e) => e.textContent.trim() === text) || null; },
    async latestChat() {
      const runs = await (await fetch("/brainscan/runs")).json();
      return runs.find((r) => r.id.startsWith("chat-") && !r.id.endsWith("-edited") && (r.tokens || 0) > 0)?.id || null;
    },
    async openRun(id) {
      const sel = q("#run-select");
      if (![...sel.options].some((o) => o.value === (id || ""))) sel.insertAdjacentHTML("beforeend", `<option value="${id || ""}">${id || "(none)"}</option>`);
      sel.value = id || "";
      sel.dispatchEvent(new Event("change"));
    },
    async api(method, path, body) {
      const res = await fetch(path, { method, headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
      const text = await res.text();
      return text ? JSON.parse(text) : null;
    },
    /** Put the app in a known state before a scene is recorded. */
    async reset(o = {}) {
      document.activeElement?.blur?.();
      const status = await d.api("GET", "/brainscan/status");
      for (const iv of status.interventions || []) await d.api("DELETE", `/brainscan/interventions/${iv.id}`);
      if (o.revert) await d.api("POST", "/brainscan/workbench/revert", {});
      if (o.clearEdits) for (const e of (await d.api("GET", "/brainscan/workbench")).edits) await d.api("DELETE", `/brainscan/workbench/edits/${e.id}`);
      if (o.clearChecks) for (const c of await d.api("GET", "/brainscan/checks")) await d.api("DELETE", `/brainscan/checks/${c.id}`);
      if (o.apply) {
        const wb = await d.api("GET", "/brainscan/workbench");
        if (wb.edits.length && !wb.applied.length) await d.api("POST", "/brainscan/workbench/apply", {});
      }
      document.dispatchEvent(new CustomEvent("brainscan:workbench"));
      if (!q("#guide").hidden) q("#guide-close").click();
      if (!q("#model-picker").hidden) q("#mp-close").click();
      if (!q("#gv-palette").hidden) d.key("Escape", {}, q("#gv-q"));
      if (q("#wb-kind").value !== "scale") d.set(q("#wb-kind"), "scale");  // the lesion form, the one the videos use
      if (o.view) d.show(o.view);
      await sleep(1500);
      if (o.run !== undefined) {
        await d.openRun(o.run === "latest-chat" ? await d.latestChat() : o.run === "capital-chat" ? await d.capitalChat() : o.run);
        await sleep(2500);
        if (!o.run) {  // the picker re-renders with the newest run showing; say none is open
          const sel = q("#run-select");
          sel.insertAdjacentHTML("afterbegin", `<option value="">No run open: ask the model below</option>`);
          sel.value = "";
        }
      }
      if (o.view === "weights") {
        for (let i = 0; i < 6; i++) { d.key("Escape"); await sleep(60); }
        q("#gv-fit").click();
      }
      // The action the scene is about, at a given step (or its first step not done yet).
      if (o.action) {
        document.dispatchEvent(new CustomEvent("brainscan:pick", { detail: o.action }));
        await sleep(1200);
        if (o.step) { d.stepChip(o.step)?.click(); await sleep(1000); }
      }
      window.scrollTo(0, 0);
      for (const el of document.querySelectorAll("#view-workbench, .workbench, #messages, #an-detail, #tv-convo-body")) el.scrollTop = 0;
      d.hide();
      await sleep(1500);
      return true;
    },
    /** For setups: trace a word in the conversation and wait for its routes. */
    async traceQuick(text) {
      await d.until(() => d.word(text), 30000, `the word ${text}`);
      d.word(text).click();
      await d.until(() => q("#an-detail .routes li[data-i]"), 120000, "routes");
      await sleep(1200);
    },
  };
  return true;
}

export const SCENES = [
  // ------------------------------------------------------------------ 1. actions
  {
    name: "01-actions",
    setup: async () => { await window.demo.reset({ view: "scanner", run: "latest-chat", revert: true, action: "explore" }); return true; },
    script: async () => {
      const d = window.demo, q = d.q;
      await d.title("Adjust open-weights models, and see what changed", "Brainscan runs the model on your GPU and records what happens inside it", 3400);
      await d.say("Brainscan runs an open-weights model on your machine and records its internals, token by token.",
        `Here: ${q("#model-badge").innerText.split(" · ")[0]}, within a GPU memory cap.`);
      await d.move(q("#verbs"));
      await d.say("Start with what you want to do to the model.",
        "Adjust it: ablate, scale or steer. Analyze it: probe, monitor, compare or check.", "top");
      await d.click(q("#verbs [data-more]"));
      await d.cap("More lists what's planned next.", "", "top");
      await d.sleep(2600);
      document.body.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true }));
      await d.pick("ablate");
      await d.move(d.bar().querySelector(".ab-steps"));
      await d.say("Each action runs as steps, on three screens.",
        "Ablate: record in Watch, locate, test, apply and measure in Trace, export from Tune.", "top");
      for (const chip of [...d.bar().querySelectorAll(".ab-step")].slice(0, 4)) { await d.move(chip, 500); await d.sleep(500); }
      await d.move(d.bar().querySelector(".ab-what"));
      await d.say("The bar says what to do in this step, offers its tools and the next step.",
        "Done steps are ticked. Greyed tools say what they need first.", "top");
      await d.click(q("#help-btn"));
      await d.sleep(600);
      await d.cap("The guide starts with the actions, each with its steps and a Start link.", "Press ? to open it from anywhere.", "top");
      await d.sleep(3600);
      await d.click(q("#guide-close"));
      await d.title("Next: ablate a fact", "", 2400);
      return true;
    },
  },

  // ------------------------------------------------------------------ 2. ablate
  {
    name: "02-ablate",
    setup: async () => { await window.demo.reset({ view: "scanner", run: "", revert: true, clearEdits: true, clearChecks: true, action: "ablate", step: "Record" }); return true; },
    script: async () => {
      const d = window.demo, q = d.q;
      await d.title("Ablate", "Remove what a word depends on, test it, keep it as a weight edit", 3000);
      await d.move(d.bar().querySelector(".ab-what"));
      await d.say("Step 1, Record: get the model to say it.", "", "top");
      const answer = await d.ask(q("#view-scanner .composer"), d.CAPITAL);
      await d.say(`“${answer}”`, "Next: find what in the model wrote “Canberra”.", "top");
      await d.click(d.next());
      await d.until(() => d.word("Canberra"), 30000, "Canberra in Trace");
      await d.cap("Step 2, Locate: click the word.", "", "top");
      await d.click(d.word("Canberra"));
      await d.until(() => q("#an-detail .routes li[data-i]"), 120000, "routes");
      await d.sleep(1000);
      const stats = [...document.querySelectorAll("#an-detail .d-stats b")].map((b) => b.textContent);
      await d.say(`“Canberra”: probability ${stats[0]}. Every unit is colored by what it did to this word.`,
        "Find paths shows the strongest routes into it.", "top");
      await d.click(d.tool("Find paths"));
      await d.until(() => q("#gv-path-list li"), 30000, "paths");
      await d.sleep(1800);
      await d.click(d.next());
      await d.cap("Step 3, Test: knock out the five units that pushed it up most.", "Nothing changes in the weights yet.", "top");
      await d.click(d.tool("Knock out top 5"));
      await d.until(() => q("#tp-ko-list li .ko-nums"), 120000, "knockout");
      await d.sleep(800);
      let li = q("#tp-ko-list li");
      await d.say(`Top 5 removed: ${d.koNums(li)}.`, li.querySelector(".hint").innerText.split(" Most likely")[0], "top");
      await d.cap("Not enough. Knock out every unit on the paths instead.", "", "top");
      await d.click(d.tool("Knock out path units"));
      await d.until(() => document.querySelectorAll("#tp-ko-list li .ko-nums").length >= 2, 120000, "path knockout");
      await d.sleep(800);
      li = q("#tp-ko-list li");
      await d.say(`${li.querySelector(".ko-head b").innerText}: ${d.koNums(li)}.`, `Most likely now: ${d.mostLikely(li)}.`, "top");
      await d.click(d.next());
      await d.cap("Step 4, Apply: keep that test as a weight edit.", "", "top");
      await d.click(d.tool("Make permanent"));
      await d.until(() => q("#tp-ko-list .pill.ok"), 120000, "permanent");
      await d.sleep(1200);
      await d.say("One edit, applied now.", "Restore original undoes every edit exactly.", "top");
      await d.click(d.next());
      await d.cap("Step 5, Measure: ask the chat's question again on both weights.", "", "top");
      await d.click(d.tool("Re-run the chat"));
      await d.until(() => q("#gv-edits .rp-card"), 180000, "replay");
      await d.sleep(800);
      await d.move(q("#gv-edits .rp-card"));
      const cols = [...document.querySelectorAll("#gv-edits .rp-cols p")].map((p) => p.innerText.trim());
      await d.say(`Original: “${d.short(cols[0])}”`, `Edited: “${d.short(cols[1])}”`, "top");
      await d.click(d.next());
      await d.say("Step 6, Export: save the edited model as a Hugging Face checkpoint, every edit recorded.", "", "top");
      await d.click(d.tool("Restore original"));
      await d.cap("Or restore the original weights.", "", "top");
      await d.until(async () => !(await d.api("GET", "/brainscan/workbench")).applied.length, 60000, "restored");
      await d.sleep(1800);
      await d.title("Next: scale instead of remove", "", 2400);
      return true;
    },
  },

  // ------------------------------------------------------------------ 3. scale
  {
    name: "03-scale",
    setup: async () => {
      const d = window.demo;
      await d.reset({ view: "weights", run: "capital-chat", revert: true, clearEdits: true, action: "scale" });
      await d.traceQuick("Canberra");
      d.stepChip("Locate")?.click();
      await d.sleep(800);
      return true;
    },
    script: async () => {
      const d = window.demo, q = d.q;
      await d.title("Scale", "Dampen or boost the units behind a word", 2800);
      await d.say("Locate: the same word, “Canberra”. Pick units to scale: here, every unit on its strongest paths.", "", "top");
      await d.click(d.tool("Find paths"));
      await d.until(() => q("#gv-path-list li"), 30000, "paths");
      await d.sleep(1200);
      const sel = [...document.querySelectorAll("#gv-path-head button")].find((b) => /select units/i.test(b.textContent));
      await d.click(sel);
      await d.say(d.bar().querySelector(".ab-what").innerText.replace(/^\d+ · Locate\.\s*/, "").replace(/·/g, ""), "", "top");
      await d.click(d.next());
      await d.click(d.tool("Scale selection"));
      await d.cap("Test: scale their writes. ×0.5 dampens, ×2 boosts.", "", "top");
      await d.sleep(1200);
      await d.click([...document.querySelectorAll("#ab-menu button")].find((b) => b.textContent.startsWith("×0.25")));
      await d.until(() => q("#tp-ko-list li .ko-nums"), 120000, "scale test");
      await d.sleep(800);
      const li = q("#tp-ko-list li");
      await d.say(`${li.querySelector(".ko-head b").innerText}: ${d.koNums(li)}.`, li.querySelector(".hint").innerText.split(" Most likely")[0], "top");
      await d.click(d.next());
      await d.click(d.tool("Make permanent"));
      await d.until(() => q("#tp-ko-list .pill.ok"), 120000, "permanent");
      await d.sleep(800);
      await d.say("Apply: kept as a scale edit. Read on the edited weights, the word matches the test.", "", "top");
      await d.click(d.tool("Restore original"));
      await d.until(async () => !(await d.api("GET", "/brainscan/workbench")).applied.length, 60000, "restored");
      await d.cap("And back to the original weights.", "", "top");
      await d.sleep(2000);
      await d.title("Next: steer", "", 2400);
      return true;
    },
  },

  // ------------------------------------------------------------------ 4. steer
  {
    name: "04-steer",
    setup: async () => {
      const d = window.demo, q = d.q;
      await d.api("DELETE", "/brainscan/directions/language_fr").catch(() => {});
      await d.reset({ view: "workbench", revert: true, action: "steer", step: "Direction" });
      q("#wb-a").value = ""; q("#wb-b").value = "";
      q("#wb-sweep").innerHTML = ""; q("#wb-save-row").hidden = true; q("#wb-find-msg").textContent = "";
      window.scrollTo(0, 0);
      await d.sleep(800);
      return true;
    },
    script: async () => {
      const d = window.demo, q = d.q;
      await d.title("Steer", "Push a direction into the model while it writes", 2800);
      await d.click(q("#wb-example"));
      await d.say("Step 1, Direction: two sets of examples, one per line.", "Here: French sentences against their English translations.", "top");
      await d.click(q("#wb-analyze"));
      await d.cap("Brainscan reads every example at every layer and scores how well the difference separates held-out ones.", "", "top");
      await d.until(() => q("#wb-sweep svg"), 180000, "sweep");
      await d.sleep(800);
      await d.move(q("#wb-sweep"));
      await d.say(q("#wb-find-msg").innerText.split(". ")[0] + ".", "Save it at a middle layer, where steering works best.", "top");
      d.set(q("#wb-layer"), "12");
      d.set(q("#wb-name"), "language_fr");
      await d.click(q("#wb-save"));
      await d.sleep(1200);
      await d.click(d.next());
      await d.until(() => [...q("#iv-direction").options].some((o) => o.value === "language_fr"), 20000, "the new direction");
      await d.sleep(800);
      d.set(q("#iv-direction"), "language_fr");
      await d.sleep(300);
      d.set(q("#iv-layer"), "12");
      d.set(q("#iv-mode"), "add");
      d.set(q("#iv-strength"), "0.5");
      d.set(q("#iv-phase"), "gen");
      await d.move(q("#iv-form"));
      await d.say("Step 2, Steer: add the direction at layer 12 while the model writes.", "Half the residual's size, on generated tokens only.", "top");
      await d.click(q('#iv-form button[type="submit"]'));
      await d.until(() => !q("#steer-banner").hidden, 10000, "steering on");
      await d.click(d.next());
      const answer = await d.ask(q("#view-scanner .composer"), "Describe a sunny morning in two sentences.");
      await d.say(`Step 3, Run: “${answer}”`, "On this small model the direction reads as “not English”: it pushes toward Romance languages.", "top");
      await d.click(d.next());
      await d.click(d.tool("Remove steering"));
      await d.cap("Step 4, Compare: remove the steering and the model is back to normal.", "Steered runs are marked ▲steered in the run list.", "top");
      await d.sleep(2800);
      await d.title("Next: probe", "", 2400);
      return true;
    },
  },

  // ------------------------------------------------------------------ 5. probe
  {
    name: "05-probe",
    setup: async () => { await window.demo.reset({ view: "scanner", run: "latest-chat", action: "probe", step: "Read" }); return true; },
    script: async () => {
      const d = window.demo, q = d.q;
      await d.title("Probe", "Read a state on every token, beside a random control", 2800);
      await d.move(d.bar().querySelector(".ab-steps"));
      await d.say("Probe: a saved direction, read on every token as it is recorded.", "Any contrast you can give examples of: formal or casual, sure or hedging.", "top");
      const r = q("#scope").getBoundingClientRect();
      await d.realMove(r.left + r.width * 0.5, r.top + r.height * 0.78, 900);
      await d.say("The z lane: each direction per token, with a random direction as the control (dashed).",
        "The grey band is ±2. Outside it, the probe reads something the random direction doesn't.", "top");
      // A token the model wrote, from the middle of its reply.
      const tape = await d.api("GET", `/brainscan/runs/${q("#run-select").value}/tape`);
      const gen = tape.filter((x) => x.ph === "gen");
      await d.realMove(r.left + r.width * 0.45, r.top + r.height * 0.5, 800);
      d.press();
      document.dispatchEvent(new CustomEvent("brainscan:select", { detail: { t: (gen[gen.length >> 1] || tape[tape.length - 1]).t } }));
      await d.until(() => q("#inspector .proj-list"), 30000, "projections");
      await d.sleep(600);
      await d.move(q("#inspector .proj-list"));
      await d.say("Click a token: its projection on every direction at that position.", "", "top");
      await d.title("Next: monitor a run", "", 2400);
      return true;
    },
  },

  // ------------------------------------------------------------------ 6. monitor
  {
    name: "06-monitor",
    setup: async () => {
      const d = window.demo;
      for (const m of await d.api("GET", "/brainscan/monitors")) await d.api("DELETE", `/brainscan/monitors/${m.id}`);
      await d.reset({ view: "scanner", run: "", action: "monitor", step: "Rule" });
      return true;
    },
    script: async () => {
      const d = window.demo, q = d.q;
      await d.title("Monitor", "Flag a run the moment a probe fires", 2800);
      d.set(q("#mon-dir"), "error_vs_success");
      d.set(q("#mon-sign"), "either");
      d.set(q("#mon-z"), "1");
      d.set(q("#mon-n"), "3");
      await d.move(q("#mon-form"));
      await d.say("Step 2, Rule: a direction past a threshold for some tokens in a row.", "Here: error_vs_success beyond ±1 for 3 tokens.", "top");
      await d.click(q('#mon-form button[type="submit"]'));
      await d.sleep(1000);
      await d.click(d.next());
      await d.ask(q("#view-scanner .composer"), "Run the tests again. They fail: AssertionError: expected 5, got 9. What went wrong?");
      await d.until(() => q("#mon-run .mon-flags li"), 30000, "a flag");
      await d.say("Step 3, Record: the flag is raised while the model writes.", "On the timeline (⚑), in the bar and in the run list.", "top");
      await d.click(d.next());
      await d.sleep(1000);
      await d.move(q("#mon-run"));
      const row = q("#mon-run .mon-row");
      await d.say(row.querySelector("p").innerText, row.querySelector(".warn-text")?.innerText || "", "top");
      await d.click(d.tool("Next flag"));
      await d.sleep(1500);
      await d.cap("Next flag selects the token where the rule was met. Trace → explains it.", "", "top");
      await d.click(q("#mon-run [data-trace]"));
      await d.until(() => q("#an-detail .routes li[data-i]"), 120000, "trace");
      await d.sleep(1200);
      await d.say("Step 5, Explain: the units and layers behind the flagged token.", "", "top");
      await d.title("Next: check and compare", "", 2400);
      return true;
    },
  },

  // ------------------------------------------------------------------ 7. check
  {
    name: "07-check",
    setup: async () => {
      const d = window.demo;
      await d.reset({ view: "weights", run: "capital-chat", revert: true, clearEdits: true, clearChecks: true, action: "check" });
      await d.traceQuick("Canberra");
      d.stepChip("Save")?.click();
      await d.sleep(800);
      return true;
    },
    script: async () => {
      const d = window.demo, q = d.q;
      await d.title("Check", "Behavior tests for every change", 2800);
      await d.say("Step 1, Save: after this context, the model should say this word.", "", "top");
      await d.click(d.tool("Add as a check"));
      await d.sleep(1200);
      await d.click(d.next());
      await d.sleep(1200);
      const form = q(".wb-checks form[data-add-check]");
      await d.cap("Or type one, for behavior an edit shouldn't touch.", "", "top");
      await d.click(form.querySelector('[name="prompt"]'));
      await d.type(form.querySelector('[name="prompt"]'), "The capital of France is");
      await d.click(form.querySelector('[name="expect"]'));
      await d.type(form.querySelector('[name="expect"]'), "Paris");
      await d.click(form.querySelector('button[type="submit"]'));
      await d.until(() => document.querySelectorAll(".wb-checks .ck-list .ck-val").length >= 2, 120000, "checks");
      await d.sleep(800);
      await d.move(q(".wb-checks .ck-list"));
      await d.say("Step 2, Run: each check's probability on the original weights → now.", "With an edit applied, what it fixed and what it broke.", "top");
      await d.title("Next: compare", "", 2400);
      return true;
    },
  },

  // ------------------------------------------------------------------ 8. compare
  {
    name: "08-compare",
    setup: async () => {
      const d = window.demo;
      await d.reset({ view: "workbench", run: "capital-chat", revert: true, clearEdits: true, action: "compare", step: "Change" });
      // An edit to compare: the units that push "Canberra" up most, in the stack but not applied.
      const run = d.q("#run-select").value;
      const tape = await d.api("GET", `/brainscan/runs/${run}/tape`);
      const t = tape.find((r) => r.tx === " Canberra").t - 1;
      const ex = await d.api("GET", `/brainscan/runs/${run}/tokens/${t}/explain`);
      const units = ex.top.filter(([, c]) => c > 0).slice(0, 12).map(([g]) => g);
      await d.api("POST", "/brainscan/anatomy/lesion", { units, apply: false, note: "units behind Canberra" });
      document.dispatchEvent(new CustomEvent("brainscan:workbench"));
      await d.sleep(1500);
      d.stepChip("Change")?.click();
      await d.sleep(800);
      return true;
    },
    script: async () => {
      const d = window.demo, q = d.q;
      await d.title("Compare", "What an edit did, outside and inside", 2800);
      await d.move(q("#wb-list"));
      await d.say("Step 1, Change: an edit in the stack, removing the units behind “Canberra”.", "", "top");
      await d.click(d.tool("Apply stack"));
      await d.until(async () => (await d.api("GET", "/brainscan/workbench")).applied.length, 60000, "applied");
      await d.sleep(1200);
      await d.click(d.next());
      await d.click(d.tool("Re-run the chat"));
      await d.cap("Step 2, Re-run: the chat's question on both weights.", "", "top");
      await d.until(() => q(".wb-replay .rp-card"), 180000, "replay");
      await d.sleep(800);
      const cols = [...document.querySelectorAll(".wb-replay .rp-cols p")].map((p) => p.innerText.trim());
      await d.say(`Original: “${d.short(cols[0])}”`, `Edited: “${d.short(cols[1])}”`, "top");
      await d.click(d.next());
      const ta = q("#wb-prompts");
      await d.click(ta);
      await d.type(ta, "Write one sentence about rain.", 30);
      d.set(q("#wb-max"), "40");
      await d.click(q("#wb-run"));
      await d.cap("Step 3, Prompts: your own prompts on both weights, and drift on neutral text.", "", "top");
      await d.until(() => q("#wb-metrics .tile"), 180000, "comparison");
      await d.sleep(800);
      await d.move(q("#wb-metrics"));
      const kl = [...document.querySelectorAll("#wb-metrics .tile")].find((t) => /KL/.test(t.innerText));
      await d.say("How far the edited model drifted on text the edit had nothing to do with.",
        kl ? `KL ${kl.querySelector(".value").innerText} nats per token.` : "", "top");
      await d.click(d.next());
      await d.until(() => d.word("Canberra"), 30000, "Canberra in Trace");
      await d.click(d.word("Canberra"));
      await d.until(() => q("#an-detail .routes li[data-i]"), 120000, "routes");
      const vs = d.tool("vs original");  // it stays on once turned on; clicking it then would turn it off
      if (/: on/.test(vs.textContent)) await d.move(vs); else await d.click(vs);
      await d.until(() => q("#an-detail .cmp-line"), 90000, "before/after");
      await d.sleep(1000);
      await d.say(`Step 4, Inside: ${q("#an-detail .cmp-line").innerText.replace(/\s+/g, " ").trim()}.`,
        "Every unit is colored by how the edit changed its effect on the word.", "top");
      await d.api("POST", "/brainscan/workbench/revert", {});
      document.dispatchEvent(new CustomEvent("brainscan:workbench"));
      await d.title("Next: the screens, and Do…", "", 2400);
      return true;
    },
  },

  // ------------------------------------------------------------------ 9. the screens and Do…
  {
    name: "09-screens-and-do",
    setup: async () => { await window.demo.reset({ view: "scanner", run: "capital-chat", revert: true, action: "explore" }); return true; },
    script: async () => {
      const d = window.demo, q = d.q;
      await d.title("The screens, and Do…", "Every tool, one search away", 2800);
      await d.say("Explore is no action: each screen with every tool it has.", "", "top");
      for (const label of ["Watch", "Trace", "Tune"]) {
        await d.click(d.stepChip(label));
        await d.sleep(1600);
        await d.move(d.bar().querySelector(".ab-groups"));
        await d.sleep(1400);
      }
      await d.click(q("#do-btn"));
      const inp = q("#do-q");
      await d.type(inp, "knock", 90);
      await d.say("Do… (⌘K) searches every action and tool, and runs it from any screen.", "Greyed ones say what they need first.", "top");
      d.key("Escape", {}, inp);
      await d.title("Brainscan", "Adjust open-weights models, and see what changed", 3400);
      return true;
    },
  },
];
