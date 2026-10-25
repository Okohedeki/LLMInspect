// Records demo videos of the Brainscan UI: drives a headless Chrome over the DevTools protocol,
// captures screencast frames, and stitches them into MP4s with ffmpeg. Scenes live in scenes.mjs.
//
//   node scripts/demo/recorder.mjs <out-dir> [scene names...]
//
// Needs a running recorder (BRAINSCAN_URL, default http://localhost:8765/), Chrome (CHROME) and ffmpeg
// (FFMPEG) on this machine, and Node 22+ for the built-in WebSocket. Scenes run in order and build on
// each other (the trace scenes use the chat the "ask" scene creates), and they change the recorder's
// state: weight edits, checks and steering are reset as each scene needs. Keep <out-dir> short on
// Windows; the frames are written there before ffmpeg joins them.
//
// Scene scripts run in the page. They can send real mouse input (hover, drag, wheel on the graph's
// canvas) through the __demoInput binding, which this script replays with Input.dispatchMouseEvent.
import { spawn, spawnSync } from "node:child_process";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { SCENES, HELPER } from "./scenes.mjs";

const CHROME = process.env.CHROME || "C:/Program Files/Google/Chrome/Application/chrome.exe";
const FFMPEG = process.env.FFMPEG || "ffmpeg";
const APP = process.env.BRAINSCAN_URL || "http://localhost:8765/";
const W = 1920, H = 1080, PORT = 9333;
const ROOT = new URL(".", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const OUT = process.argv[2] || join(ROOT, "out");
const ONLY = process.argv.slice(3);
const KEEP_FRAMES = !!process.env.KEEP_FRAMES;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const src = (s) => (typeof s === "function" ? `(${s.toString()})()` : s);

const profile = join(tmpdir(), "bs-demo-chrome");  // short: Chrome fails on profile paths near MAX_PATH
rmSync(profile, { recursive: true, force: true });
const chrome = spawn(CHROME, [`--headless=new`, `--remote-debugging-port=${PORT}`, `--window-size=${W},${H}`,
  `--user-data-dir=${profile}`, "--hide-scrollbars", "--force-device-scale-factor=1", "--no-first-run",
  "--no-default-browser-check", "--disable-extensions", "about:blank"], { stdio: ["ignore", "ignore", "pipe"] });
let chromeErr = "";
chrome.stderr.on("data", (b) => { chromeErr = (chromeErr + b.toString()).slice(-2000); });
chrome.on("exit", (code) => { if (code) console.error(`chrome exited ${code}: ${chromeErr}`); });

async function target() {
  for (let i = 0; i < 160; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      const page = list.find((t) => t.type === "page");
      if (page) return page.webSocketDebuggerUrl;
    } catch { /* not up yet */ }
    await sleep(250);
  }
  throw new Error(`chrome did not start: ${chromeErr}`);
}

const ws = new WebSocket(await target());
await new Promise((r) => ws.addEventListener("open", r, { once: true }));
let nextId = 1;
const pending = new Map(), listeners = [];
ws.addEventListener("message", (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.id && pending.has(msg.id)) { const { res, rej } = pending.get(msg.id); pending.delete(msg.id); msg.error ? rej(new Error(msg.error.message)) : res(msg.result); }
  else if (msg.method) for (const fn of listeners) fn(msg);
});
const cdp = (method, params = {}) => new Promise((res, rej) => { const id = nextId++; pending.set(id, { res, rej }); ws.send(JSON.stringify({ id, method, params })); });

async function evaluate(expression, timeout = 600000) {
  const r = await cdp("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true, timeout });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
  return r.result?.value;
}

// real mouse input asked for by the page
listeners.push(async (msg) => {
  if (msg.method !== "Runtime.bindingCalled" || msg.params.name !== "__demoInput") return;
  const { id, events } = JSON.parse(msg.params.payload);
  try {
    for (const e of events) await cdp("Input.dispatchMouseEvent", e);
  } finally {
    cdp("Runtime.evaluate", { expression: `window.__demoAck && window.__demoAck(${id})` }).catch(() => {});
  }
});

// page console, for debugging scenes
listeners.push((msg) => {
  if (msg.method === "Runtime.consoleAPICalled" && ["error", "warning"].includes(msg.params.type)) {
    console.log(`  [page ${msg.params.type}]`, msg.params.args.map((a) => a.value ?? a.description ?? "").join(" ").slice(0, 300));
  }
  if (msg.method === "Runtime.exceptionThrown") console.log("  [page exception]", msg.params.exceptionDetails?.exception?.description?.slice(0, 300));
});

// frames
let rec = null;
listeners.push((msg) => {
  if (msg.method !== "Page.screencastFrame") return;
  const { data, metadata, sessionId } = msg.params;
  cdp("Page.screencastFrameAck", { sessionId }).catch(() => {});
  if (!rec) return;
  const file = join(rec.dir, `f${String(rec.frames.length).padStart(6, "0")}.jpg`);
  writeFileSync(file, Buffer.from(data, "base64"));
  rec.frames.push({ file, t: metadata.timestamp });
});

async function startRecording(name) {
  const dir = join(OUT, `${name}-frames`);
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });
  rec = { name, dir, frames: [] };
  await cdp("Page.startScreencast", { format: "jpeg", quality: 85, maxWidth: W, maxHeight: H, everyNthFrame: 1 });
}

async function stopRecording() {
  await sleep(600);
  await cdp("Page.stopScreencast");
  const { name, dir, frames } = rec;
  rec = null;
  if (frames.length < 2) throw new Error(`${name}: no frames`);
  const lines = [];
  frames.forEach((f, i) => {
    const d = i + 1 < frames.length ? Math.max(frames[i + 1].t - f.t, 0.001) : 1.0;
    lines.push(`file '${f.file.replace(/\\/g, "/")}'`, `duration ${d.toFixed(4)}`);
  });
  lines.push(`file '${frames[frames.length - 1].file.replace(/\\/g, "/")}'`);
  const list = join(dir, "list.txt");
  writeFileSync(list, lines.join("\n"));
  const out = join(OUT, `${name}.mp4`);
  const r = spawnSync(FFMPEG, ["-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", list,
    "-vf", `fps=30,scale=${W}:${H}:flags=lanczos,format=yuv420p`, "-c:v", "libx264", "-preset", "medium", "-crf", "20",
    "-movflags", "+faststart", out], { encoding: "utf8", maxBuffer: 1 << 26 });
  if (r.status !== 0) throw new Error(`ffmpeg failed: ${r.stderr}`);
  const secs = frames[frames.length - 1].t - frames[0].t + 1.0;
  console.log(`${name}: ${frames.length} frames, ${secs.toFixed(1)}s -> ${out}`);
  if (!KEEP_FRAMES) rmSync(dir, { recursive: true, force: true });
  return out;
}

try {
  mkdirSync(OUT, { recursive: true });
  await cdp("Page.enable");
  await cdp("Runtime.enable");
  await cdp("Runtime.addBinding", { name: "__demoInput" });
  await cdp("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: 1, mobile: false });
  await cdp("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: "dark" }] });
  await cdp("Page.navigate", { url: APP });
  await sleep(2500);
  await evaluate(`localStorage.setItem("brainscan-guide-seen", "1"); localStorage.setItem("brainscan-convo", "1"); localStorage.setItem("brainscan-view", "weights"); localStorage.setItem("brainscan-action", JSON.stringify({ verb: "explore", step: 1 })); true`);
  await cdp("Page.reload", {});
  await sleep(5000);
  for (const scene of SCENES) {
    if (ONLY.length && !ONLY.includes(scene.name)) continue;
    console.log(`scene ${scene.name}`);
    await evaluate(src(HELPER));
    if (scene.setup) await evaluate(src(scene.setup));
    await startRecording(scene.name);
    try {
      await evaluate(src(scene.script));
    } catch (err) {
      console.error(`  ${scene.name} failed: ${err.message}`);
      process.exitCode = 1;
    } finally {
      await stopRecording();
    }
  }
} catch (err) {
  console.error("FAILED:", err.message);
  process.exitCode = 1;
} finally {
  try { ws.close(); } catch { /* closing */ }
  chrome.kill();
}
