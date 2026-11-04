"""The picture a shared /m/ORG/NAME link shows on social sites: a model's Unmask headline at 1200×630, drawn on the
server in the card's colors (white, the edit in amber, risks in red) so the link says what the model is."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, PAD = 1200, 630, 64
C = {"bg": (7, 8, 13), "ink": (244, 245, 248), "ink2": (169, 175, 191), "muted": (102, 109, 128), "amber": (245, 165, 36),
     "red": (240, 82, 82), "teal": (34, 199, 214), "line": (42, 47, 61)}
FONTS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans{}.ttf", "C:/Windows/Fonts/segoeui{}.ttf")
MONO = ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono{}.ttf", "C:/Windows/Fonts/consola{}.ttf")
RISKS = ("runs code on load", "imports other code on load", "runs code if trusted", "template")


def font(size: int, bold: bool = False, mono: bool = False):
    for pattern in MONO if mono else FONTS:
        path = Path(pattern.format(("-Bold" if "dejavu" in pattern else "b") if bold else ""))
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size)


def _spans(r: dict | None) -> list[tuple[str, str]]:
    """The headline as (text, color) pieces: the edit's own words in amber, file risks in red, the rest white."""
    if not r:
        return [("Not read yet: open the link to unmask it.", "ink2")]
    head = r.get("headline") or ""
    edit = (r.get("edit") or {}).get("headline") or ""
    out, at = [], head.find(edit) if edit and edit != "the same weights, re-uploaded" else -1
    if at >= 0:
        out += [(head[:at], "ink"), (edit, "amber"), (head[at + len(edit):], "ink")]
    else:
        out.append((head, "ink"))
    pieces = []
    for text, color in out:  # file risks named in the headline go red
        for risk in ("its files runs code on load", "its files imports other code on load", "its files runs code if trusted"):
            if risk in text:
                a, b = text.split(risk, 1)
                pieces += [(a, color), (risk, "red")]
                text = b
        pieces.append((text, color))
    return [(t, c) for t, c in pieces if t]


def _wrap(draw, spans, fnt, width: int, max_lines: int) -> list[list[tuple[str, str]]]:
    """Word-wrap colored spans into lines no wider than `width`, ending with … when cut."""
    words = [(w, c) for text, c in spans for w in text.replace("\n", " ").split(" ") if w]
    lines, line, used = [], [], 0
    space = draw.textlength(" ", font=fnt)
    for w, c in words:
        size = draw.textlength(w, font=fnt)
        if line and used + space + size > width:
            lines.append(line)
            line, used = [], 0
        line.append((w, c))
        used += size + (space if len(line) > 1 else 0)
    if line:
        lines.append(line)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last_w, last_c = lines[-1][-1]
        lines[-1][-1] = (last_w.rstrip(".,;") + "…", last_c)
    return lines


def render(model: str, r: dict | None) -> bytes:
    img = Image.new("RGB", (W, H), C["bg"])
    d = ImageDraw.Draw(img)
    for y in range(0, 240):  # a faint amber glow at the top, as on the card
        a = int(18 * (1 - y / 240) ** 2)
        d.line([(0, y), (W, y)], fill=(7 + a, 8 + a // 2, 13))
    small, mono = font(20, mono=True), font(26, mono=True)
    d.text((PAD, PAD), "UNMASKED", font=small, fill=C["muted"])
    d.text((W - PAD, PAD), "LLMINSPECT", font=small, fill=C["muted"], anchor="ra")
    quant = ((r or {}).get("quant") or {}).get("type")
    d.text((PAD, PAD + 44), model + (f" · {quant}" if quant else ""), font=mono, fill=C["ink2"])
    for size in (54, 48, 42, 36):  # the largest headline that fits in four lines
        big = font(size, bold=True)
        lines = _wrap(d, _spans(r), big, W - 2 * PAD, 4)
        if len(lines) <= 4 and sum(1 for _ in lines) * size * 1.25 <= 300:
            break
    y = PAD + 110
    for line in lines:
        x = PAD
        for i, (w, c) in enumerate(line):
            d.text((x, y), w, font=big, fill=C[c])
            glued = i + 1 < len(line) and line[i + 1][0][:1] in ".,;:"  # punctuation sits on the word before it
            x += d.textlength(w if glued else w + " ", font=big)
        y += int(size * 1.25)
    files = ((r or {}).get("files") or {}).get("findings") or []
    risky = [f for f in files if f.get("level") in RISKS]
    if risky:
        d.text((PAD, H - PAD - 70), f"● {risky[0]['level']}: {risky[0]['file']}"[:90], font=font(22), fill=C["red"])
    d.line([(PAD, H - PAD - 34), (W - PAD, H - PAD - 34)], fill=C["line"], width=1)
    foot = "Read from the Hugging Face Hub by byte range · nothing downloaded, loaded or run"
    if r and r.get("bytes_read"):
        foot = f"Read {r['bytes_read'] / 1e6:.1f} MB by byte range in {r.get('seconds', 0):.0f} s · nothing downloaded, loaded or run"
    d.text((PAD, H - PAD - 20), foot, font=font(20), fill=C["muted"])
    out = io.BytesIO()
    img.save(out, "PNG", optimize=True)
    return out.getvalue()
