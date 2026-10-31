"""The page people read for a model check (check.py): the same report agents get, drawn.

One headline says what was found. Under it, the hero: the model's layers side by side, dim where they are
the base's and lit where they moved, so a targeted edit shows as a few bright columns and a full fine-tune as
an even glow. Findings follow, most serious first, then each check's evidence. Three accents, each with one
meaning everywhere: coral for findings to look at first, amber for findings worth a look, violet for what
differs from the base. Self-contained: no scripts, fonts or outside requests, so it opens anywhere and prints.
"""

from __future__ import annotations

import html
import math

SEV_WORD = {"high": "Look first", "medium": "Worth a look", "note": "Note"}
AREA_WORD = {"files": "Files", "origin": "Origin", "outside_weights": "Outside the weights",
             "inside_weights": "Inside the weights"}


def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e8 else f"{n / 1e6:.1f} MB" if n >= 1e5 else f"{n / 1e3:.0f} kB"


def _moved(corr: float) -> float:
    """0 for a layer identical to the base, up to 1 for one that has nothing of it left (log scale, so a light
    fine-tune still shows)."""
    gap = max(0.0, 1.0 - corr)
    return 0.0 if gap < 5e-4 else min(1.0, 0.25 + 0.75 * (math.log10(gap) + 3.3) / 3.3)


def _strip(layers: dict[str, float]) -> str:
    cols = []
    for k in sorted(layers, key=int):
        m = _moved(layers[k])
        label = "embedding" if k == "-1" else f"layer {k}"
        state = "unchanged" if m == 0 else f"moved (correlation {layers[k]:.4f})"
        cols.append(f'<i class="{"lit" if m else ""}" style="--m:{m:.3f}" title="{_e(label)}: {_e(state)}"></i>')
    keys = sorted(layers, key=int)
    every = max(1, len(keys) // 8)
    tick = lambda k: "emb" if k == "-1" else f"L{k}" if k == keys[-1] or (int(k) % every == 0 and int(k) > 0) else ""
    ticks = "".join(f"<span>{tick(k)}</span>" for k in keys)  # L0 sits next to emb, which already marks the start
    return (f'<div class="strip" role="img" aria-label="Each column is a layer; lit columns moved away from the base">'
            f'{"".join(cols)}</div><div class="axis">{ticks}</div>')


def _finding(f: dict) -> str:
    ev = f.get("evidence") or []
    evid = (f'<details><summary>{len(ev)} piece{"s" if len(ev) != 1 else ""} of evidence</summary>'
            f'<pre>{_e(chr(10).join(map(str, ev)))}</pre></details>') if ev else ""
    nxt = f.get("next") or {}
    step = ""
    if nxt:
        tool = f' <code>{_e(nxt["tool"])}</code>' if nxt.get("tool") else ""
        step = f'<p class="next">→ {_e(nxt.get("do", ""))}{tool}</p>'
    detail = f'<p class="detail">{_e(f["detail"])}</p>' if f.get("detail") and f["detail"] != f["title"] else ""
    where = f'<span class="where">{_e(f["where"])}</span>' if f.get("where") else ""
    return (f'<li class="f {f["severity"]}"><span class="sev">{SEV_WORD[f["severity"]]}</span><div>'
            f'<p class="title">{_e(f["title"])}</p>{detail}{evid}{step}</div>'
            f'<span class="meta">{_e(f["id"])} · {_e(AREA_WORD.get(f["area"], f["area"]))}{" · " if where else ""}{where}</span></li>')


def _files(sec: dict, flagged: dict[str, str], hero: bool = False) -> str:
    cells = []
    for name, size, fmt in sec.get("list", []):
        sev = flagged.get(name, "")
        big = "w" if size > 1e8 else ""
        cells.append(f'<i class="{sev} {big}" title="{_e(name)} · {_e(fmt)} · {_e(_gb(size))}"></i>')
    fmts = " · ".join(f"{_e(k)} {v}" for k, v in sec.get("formats", {}).items())
    return (f'<section id="files"><h2>Files</h2><p class="lede">{sec["files"]} files, {_gb(sec["bytes"])} · weights as '
            f'{_e(", ".join(sec["weight_formats"]) or "none")} · {sec["pickles"]} pickle-based</p>'
            f'<div class="cells{" hero" if hero else ""}" aria-label="Every file; lit cells have findings">{"".join(cells)}</div>'
            f'<p class="small">{fmts}</p></section>')


def _origin(sec: dict) -> str:
    best = (sec.get("matches") or [None])[0]
    if not best:
        return ('<section id="origin"><h2>Origin</h2><p class="lede">No local model of the same architecture to compare '
                'with.</p></section>')
    claim = sec.get("claimed_base") or []
    rows = "".join(f'<tr><td>{_e(m["name"])}</td><td class="n">{m["median"]:.4f}</td><td class="n">{m["min"]:.4f}</td>'
                   f'<td class="n">{m["identical"]}/{m["tensors"]}</td></tr>' for m in sec["matches"][:6])
    return (f'<section id="origin"><h2>Origin</h2><p class="lede">Closest local model: <b>{_e(best["name"])}</b> · '
            f'{_e(sec.get("reading", ""))}' + (f' · the card claims <b>{_e(", ".join(claim))}</b>' if claim else "") + '</p>'
            f'<table><thead><tr><th>Compared with</th><th class="n">Median correlation</th><th class="n">Lowest</th>'
            f'<th class="n">Identical tensors</th></tr></thead><tbody>{rows}</tbody></table></section>')


def _outside(sec: dict) -> str:
    parts = []
    for t in sec.get("templates", []):
        diff = "".join(f'<span class="{"add" if l.startswith("+") and not l.startswith("+++") else "del" if l.startswith("-") and not l.startswith("---") else ""}">'
                       f'{_e(l)}</span>\n' for l in t.get("diff", [])[:120])
        parts.append(f'<h3>Chat template · {_e(t["name"])}</h3><pre class="diff">{diff or _e("present on one side only")}</pre>')
    tok = sec.get("tokenizer", {})
    v = tok.get("vocab")
    if v:
        parts.append(f'<h3>Tokenizer</h3><p class="small">{v["base"]:,} → {v["candidate"]:,} tokens · {v["added"]} added · '
                     f'{v["removed"]} removed · {v["renumbered"]} renumbered · {len(tok.get("encodes_differently", []))} probe '
                     f'texts encode differently</p>')
    for key, label in (("config", "Config"), ("generation", "Generation defaults")):
        rows = "".join(f'<tr><td><code>{_e(c["key"])}</code></td><td><code>{_e(c["base"])}</code></td>'
                       f'<td class="changed"><code>{_e(c["candidate"])}</code></td></tr>' for c in sec.get(key, [])[:30])
        if rows:
            parts.append(f'<h3>{label}</h3><table><thead><tr><th>Field</th><th>Base</th><th>Candidate</th></tr></thead>'
                         f'<tbody>{rows}</tbody></table>')
    only = sec.get("files", {}).get("only_candidate") or []
    if only:
        parts.append(f'<h3>Files only the candidate has</h3><p class="small mono">{_e(", ".join(only[:30]))}</p>')
    raw = [k for k, v in sec.get("read", {}).items() if v == "raw files"]
    how = f'<p class="small">Compared as raw files, which transformers could not read: {_e(", ".join(raw))}</p>' if raw else ""
    body = "".join(parts) or '<p class="lede">Nothing differs outside the weights.</p>'
    return f'<section id="outside_weights"><h2>Outside the weights</h2>{body}{how}</section>'


def _inside(sec: dict) -> str:
    w = sec.get("weights") or {}
    parts = []
    if w:
        sm = w["summary"]
        layers = sorted(sm.get("layers", []), key=lambda x: x["layer"])
        top = max((x["mean_rel"] for x in layers), default=0) or 1
        bars = "".join(f'<i class="{"lit" if x["mean_rel"] > 0 else ""}" style="--m:{x["mean_rel"] / top:.3f}" '
                       f'title="layer {x["layer"]}: relative change {x["mean_rel"]:.3g}"></i>' for x in layers)
        parts.append(f'<p class="lede"><b>{_e(sm["pattern"])}</b> · {w["counts"]["changed"]} of {w["counts"]["compared"]} '
                     f'tensors differ</p><div class="strip bars">{bars}</div>')
    tg = ((sec.get("triage") or {}).get("triggers") or {}).get("tokens", [])
    if tg:
        rows = "".join(f'<tr><td><code>{_e(t["text"])}</code></td><td class="n {"hot" if t["z"] >= 4 else ""}">{t["z"]:.1f}</td>'
                       f'<td class="n">{t.get("z_after", 0):.1f}</td></tr>' for t in tg)
        parts.append(f'<h3>Possible trigger tokens</h3><table><thead><tr><th>Token</th><th class="n">z right after it</th>'
                     f'<th class="n">z over the rest</th></tr></thead><tbody>{rows}</tbody></table>')
    ref = sec.get("refusals")
    if ref:
        parts.append(f'<h3>Refusals</h3><p class="small">The base refused {ref["base_refused"]} of {ref["probes"]} harmful '
                     f'requests; the candidate still refuses {ref["still_refused"]} of those.</p>')
    b = (sec.get("behavior") or {}).get("metrics")
    if b:
        parts.append(f'<h3>Behavior</h3><p class="small">KL {b["kl_per_token"]:.3f} per token on neutral text · perplexity '
                     f'{b["perplexity_original"]:.1f} → {b["perplexity_edited"]:.1f}</p>')
    return f'<section id="inside_weights"><h2>Inside the weights</h2>{"".join(parts)}</section>'


CSS = """
:root { --canvas:#0a0b0f; --surface:rgba(255,255,255,.035); --line:rgba(255,255,255,.08); --t1:#eceef3; --t2:#a7adbb;
  --t3:#6e7482; --high:#ff6b4a; --medium:#f2b84b; --moved:#8b7bff; --mono:ui-monospace,"JetBrains Mono","SF Mono",Consolas,monospace; }
@media (prefers-color-scheme: light) { :root { --canvas:#fbfbfc; --surface:rgba(0,0,0,.03); --line:rgba(0,0,0,.09);
  --t1:#101218; --t2:#4b5160; --t3:#7d8392; --high:#d8431f; --medium:#b07c0c; --moved:#5b48e8; } }
* { box-sizing:border-box } html { background:var(--canvas) }
body { margin:0; color:var(--t1); font:15px/1.55 Inter,"IBM Plex Sans",ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  -webkit-font-smoothing:antialiased }
main { max-width:1120px; margin:0 auto; padding:56px 32px 96px }
.eyebrow,.sev,h2,h3,.axis,.meta { font-family:var(--mono); font-size:11px; letter-spacing:.12em; text-transform:uppercase }
.eyebrow { color:var(--t3) } .eyebrow b { color:var(--t2); font-weight:500 }
h1 { font-size:clamp(26px,4vw,42px); line-height:1.12; letter-spacing:-.025em; font-weight:600; margin:14px 0 18px; max-width:30ch }
h1.high { color:var(--t1) } h1 .mark { color:var(--high) }
.counts { display:flex; gap:28px; font-family:var(--mono); font-size:13px; color:var(--t2); margin:0 0 40px; flex-wrap:wrap }
.counts b { font-size:22px; font-weight:500; color:var(--t1); margin-right:6px; font-variant-numeric:tabular-nums }
.counts .high b { color:var(--high) } .counts .medium b { color:var(--medium) }
.strip { display:flex; align-items:flex-end; gap:3px; height:140px; padding:0 0 1px; border-bottom:1px solid var(--line) }
.strip i { flex:1; min-width:3px; height:calc(10% + 90% * var(--m)); background:var(--t3); opacity:.22; border-radius:2px 2px 0 0;
  transform-origin:bottom; animation:rise .6s cubic-bezier(.2,.7,.2,1) both }
.strip i.lit { background:var(--moved); opacity:calc(.45 + .55 * var(--m)); box-shadow:0 0 18px -2px var(--moved) }
.strip.bars { height:90px; margin-top:14px }
.axis { display:flex; gap:3px; color:var(--t3); margin-top:8px } .axis span { flex:1; min-width:3px; overflow:visible; white-space:nowrap }
@keyframes rise { from { transform:scaleY(0) } }
@media (prefers-reduced-motion: reduce) { .strip i { animation:none } }
.hero-caption { color:var(--t3); font-size:13px; margin:10px 0 0 }
section { margin-top:64px }
h2 { color:var(--t2); font-weight:500; margin:0 0 18px } h3 { color:var(--t3); font-weight:500; margin:28px 0 10px }
.lede { font-size:17px; color:var(--t1); margin:0 0 14px } .lede b { font-weight:600 }
.small { font-size:13px; color:var(--t2) } .mono, code, pre { font-family:var(--mono); font-size:12.5px }
ul.findings { list-style:none; margin:0; padding:0 }
.f { display:grid; grid-template-columns:120px 1fr; gap:6px 20px; padding:18px 0; border-top:1px solid var(--line) }
.f .sev { padding-top:4px; color:var(--t3) } .f.high .sev { color:var(--high) } .f.medium .sev { color:var(--medium) }
.f .title { margin:0; font-size:16px; line-height:1.4 } .f.note .title { color:var(--t2) }
.f .detail { margin:6px 0 0; color:var(--t2); font-size:13.5px }
.f .meta { grid-column:2; color:var(--t3); letter-spacing:.06em }
.f .where { text-transform:none; letter-spacing:0 }
.next { margin:8px 0 0; font-size:13.5px; color:var(--t2) } .next code { color:var(--moved) }
details { margin-top:8px } summary { cursor:pointer; color:var(--t3); font-size:12.5px; font-family:var(--mono) }
summary:focus-visible, a:focus-visible { outline:2px solid var(--moved); outline-offset:3px; border-radius:3px }
pre { white-space:pre-wrap; word-break:break-word; color:var(--t2); background:var(--surface); padding:12px 14px; border-radius:8px; margin:8px 0 0 }
pre.diff .add { color:var(--moved) } pre.diff .del { color:var(--t3); text-decoration:line-through }
.cells { display:flex; flex-wrap:wrap; gap:3px; margin:4px 0 12px }
.cells i { width:9px; height:9px; border-radius:2px; background:var(--t3); opacity:.25 } .cells i.w { width:20px }
.cells.hero { gap:5px } .cells.hero i { width:22px; height:22px; border-radius:4px } .cells.hero i.w { width:50px }
.cells i.high { background:var(--high); opacity:1; box-shadow:0 0 10px var(--high) } .cells i.medium { background:var(--medium); opacity:.9 }
table { width:100%; border-collapse:collapse; font-size:13px; margin:6px 0 }
th { text-align:left; color:var(--t3); font:500 11px var(--mono); letter-spacing:.08em; text-transform:uppercase; padding:6px 10px 6px 0;
  border-bottom:1px solid var(--line) }
td { padding:7px 10px 7px 0; border-bottom:1px solid var(--line); vertical-align:top; color:var(--t2) }
td.n, th.n { text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums } td.hot { color:var(--high) }
td.changed code { color:var(--moved) }
.not { color:var(--t2); font-size:14px; list-style:none; padding:0 } .not li { margin:6px 0 }
footer { margin-top:72px; color:var(--t3); font-size:12.5px }
@media (max-width:640px) { main { padding:32px 16px 64px } .f { grid-template-columns:1fr } .f .meta { grid-column:1 }
  .strip { height:96px; gap:2px } }
@media print { :root { --canvas:#fff; --t1:#000; --t2:#333; --t3:#666; --line:#ddd; --surface:#f4f4f4 }
  .strip i { animation:none; box-shadow:none } .cells i.high { box-shadow:none } details[open] summary, summary { color:#666 } }
"""


def render(r: dict) -> str:
    s, secs = r["summary"], r["sections"]
    flagged: dict[str, str] = {}
    for f in r["findings"]:
        if f["area"] == "files" and f.get("where"):
            for name in str(f["where"]).split(", "):
                key = name.split(":")[0]
                if flagged.get(key) != "high":
                    flagged[key] = f["severity"] if f["severity"] != "note" else ""
    origin = secs.get("origin") or {}
    best = (origin.get("matches") or [None])[0]
    hero = ""
    if best:
        moved = sum(1 for k, c in best["layers"].items() if _moved(c) > 0)
        hero = (_strip(best["layers"]) + f'<p class="hero-caption">Each column is a layer of the candidate, against '
                f'<b>{_e(best["name"])}</b>: {moved} of {len(best["layers"])} moved.</p>')
    elif "files" in secs:  # no base to draw layers against: every file is the picture
        lit = sum(1 for v in flagged.values() if v)
        hero = (_files(secs["files"], flagged, hero=True).replace('<section id="files"><h2>Files</h2>', "<div>")
                .replace("</section>", "</div>") + f'<p class="hero-caption">Each cell is a file of the model; '
                f'{lit} {"is" if lit == 1 else "are"} lit by a finding.</p>')
    c = s["counts"]
    counts = (f'<div class="counts"><span class="high"><b>{c["high"]}</b>look first</span>'
              f'<span class="medium"><b>{c["medium"]}</b>worth a look</span><span><b>{c["note"]}</b>notes</span>'
              f'<span><b>{r.get("seconds", "–")}</b>seconds</span></div>')
    findings = "".join(_finding(f) for f in r["findings"]) or '<li class="f note"><span class="sev">Note</span><div><p class="title">No findings.</p></div></li>'
    parts = [f'<section id="findings"><h2>Findings</h2><ul class="findings">{findings}</ul></section>']
    if "files" in secs and best:
        parts.append(_files(secs["files"], flagged))
    if origin:
        parts.append(_origin(origin))
    if "outside_weights" in secs:
        parts.append(_outside(secs["outside_weights"]))
    if "inside_weights" in secs:
        parts.append(_inside(secs["inside_weights"]))
    if r["not_checked"]:
        items = "".join(f'<li><b>{_e(AREA_WORD.get(n["area"], n["area"]))}</b>: {_e(n["why"])}</li>' for n in r["not_checked"])
        parts.append(f'<section id="not_checked"><h2>Not checked</h2><ul class="not">{items}</ul></section>')
    limits = "".join(f"<li>{_e(x)}</li>" for x in r.get("limits", []))
    base = r.get("base") or {}
    sub = f'against <b>{_e(base["name"])}</b> ({_e(base["chosen_by"])})' if base else "no base model to compare with"
    head_cls = "high" if c["high"] else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark light"><title>Model check · {_e(r["candidate"]["name"])}</title>
<style>{CSS}</style></head><body><main>
<div class="eyebrow">LLMInspect model check · <b>{_e(r["candidate"]["name"])}</b> · {sub}</div>
<h1 class="{head_cls}">{_e(s["headline"])}</h1>
{counts}
{hero}
{''.join(parts)}
<footer><details><summary>What these checks can't see</summary><ul class="not">{limits}</ul></details>
<p>Evidence, not a verdict: no report says a model is safe. Report {_e(r["id"])} · {_e(r["created"])} · schema {_e(r["schema"])}</p></footer>
</main></body></html>"""
