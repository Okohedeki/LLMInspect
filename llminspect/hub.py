"""Unmask: what a model really is, read without downloading it. A model's card says which model it was made from;
its weights say whether that's true and what was done to them.

  1. Lineage. The card's base_model (and the Hub's base_model tags) name the parents, and their cards name theirs.
  2. Origin. A few hundred values per tensor, read by byte range from each safetensors file, fingerprint the model
     (see fingerprint.py) and are compared with each claimed parent and with the models on this machine.
  3. The edit. Some full rows of each sampled layer's down projection are read from the model and its parent. The
     rank of their difference names the edit: the same direction or few removed from every layer (the mark
     direction removal, or abliteration, leaves), rank-one edits at single layers, a low-rank change like a merged adapter,
     or a full fine-tune.
  4. Files. Configs, code and chat templates are small and read whole: code that runs on load, and what the
     template says that the parent's doesn't.

A Hub model costs a few megabytes of reads. Nothing is loaded and nothing runs. A local folder works the same way.

    python -m llminspect.hub ORG/NAME [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import filediff, filescan, fingerprint

ROWS = 24           # full rows of a layer's down projection read to measure the rank of its change
RANK_LAYERS = 6     # layers those rows come from
SMALL = 1 << 20     # text files up to this size are read whole
DEPTH = 4           # claimed ancestors followed up the card chain
WORKERS = 16
MOST = 0.9         # share of a change's energy its top k directions must hold
FEW = 4             # at most this many directions is "a few"
LOW_RANK = 0.9      # share in its top LOW_K directions that makes a change low-rank
LOW_K = 8
SHARED = 0.8        # overlap between layers' few directions that makes them the same ones
DOWN = re.compile(r"layers?\.(\d+)\.(mlp\.down_proj|feed_forward\.w2)\.weight$")
TEXT = {".json", ".jinja", ".py", ".txt", ".model", ".yaml", ".yml"}
RELATIONS = ("finetune", "adapter", "merge", "quantized")


class Unreachable(Exception):
    """A model that can't be read: missing, gated without access, or offline."""


# ---- sources: a Hub repo or a local folder, read through the same calls ------------------------------------

class Folder:
    """A model folder on this machine."""

    def __init__(self, path: Path, name: str | None = None):
        from .check import model_name
        self.path = Path(path)
        if not self.path.is_dir():
            raise Unreachable(f"{path} isn't a folder")
        self.name = name or model_name(self.path)
        self.url = None
        self.bytes_read = 0

    def files(self) -> dict[str, int]:
        return {p.relative_to(self.path).as_posix(): p.stat().st_size for p in sorted(self.path.rglob("*"))
                if p.is_file() and not p.name.startswith(".")}

    def claims(self) -> list[dict]:
        return [{"model": b, "relation": "finetune"} for b in fingerprint.claimed_base(self.path)]

    def headers(self) -> dict:
        return fingerprint._headers(self.path)

    def signature(self) -> dict:
        return fingerprint.signature(self.path)

    def read(self, spans: list[tuple[str, int, int]]) -> list[bytes]:
        out = []
        for f, at, n in spans:
            with open(self.path / f if not Path(f).is_absolute() else f, "rb") as fh:
                fh.seek(at)
                out.append(fh.read(n))
            self.bytes_read += n
        return out

    def small_files(self) -> Path:
        return self.path


class Hub:
    """A model on the Hugging Face Hub, read by byte range. A token (HF_TOKEN, or a saved login) is sent to the
    Hub only, never to the storage it redirects to."""

    def __init__(self, repo: str, revision: str | None = None, token: str | None = None):
        import httpx
        from huggingface_hub import HfApi, constants
        from huggingface_hub.utils import build_hf_headers
        self.endpoint = constants.ENDPOINT.rstrip("/")
        try:
            info = HfApi(token=token).model_info(repo, revision=revision, files_metadata=True)
        except Exception as exc:  # noqa: BLE001  not found, gated, offline: all mean it can't be read
            raise Unreachable(f"{repo}: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}") from exc
        self.name, self.sha = info.id, info.sha
        self.url = f"{self.endpoint}/{info.id}"
        self._sizes = {s.rfilename: s.size or 0 for s in info.siblings or []}
        card = info.card_data.to_dict() if info.card_data else {}
        self._claims = _claims(card.get("base_model"), info.tags or [])
        self._auth = build_hf_headers(token=token)
        self._client = httpx.Client(timeout=60, follow_redirects=False)
        self._where: dict[str, tuple[str, dict]] = {}
        self._lock = threading.Lock()
        self._sig = self._heads = self._dir = None
        self.bytes_read = 0

    def files(self) -> dict[str, int]:
        return dict(self._sizes)

    def claims(self) -> list[dict]:
        return list(self._claims)

    def _locate(self, f: str) -> tuple[str, dict]:
        """Where a file's bytes are: the Hub's URL, or the storage it redirects to (asked without the token)."""
        with self._lock:
            hit = self._where.get(f)
        if hit:
            return hit
        url, headers = f"{self.endpoint}/{self.name}/resolve/{self.sha}/{f}", dict(self._auth)
        for _ in range(5):
            r = self._client.head(url, headers=headers)
            if r.status_code in (301, 302, 303, 307, 308):
                nxt = str(r.url.join(r.headers["location"]))
                if r.url.host != r.url.join(r.headers["location"]).host:
                    headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
                url = nxt
                continue
            if r.status_code in (401, 403):
                raise Unreachable(f"{self.name}: access denied to {f} (gated; log in or set HF_TOKEN)")
            r.raise_for_status()
            break
        with self._lock:
            self._where[f] = (url, headers)
        return url, headers

    def _range(self, f: str, at: int, n: int) -> bytes:
        url, headers = self._locate(f)
        with self._client.stream("GET", url, headers=headers | {"Range": f"bytes={at}-{at + n - 1}"}) as r:
            if r.status_code == 206:
                data = r.read()
            elif r.status_code == 200:  # the range was ignored: read up to it and stop, never the whole file
                data, got = b"", 0
                for chunk in r.iter_bytes():
                    data += chunk
                    if len(data) >= at + n:
                        break
                data = data[at:at + n]
            else:
                raise Unreachable(f"{self.name}: {f} answered {r.status_code}")
        with self._lock:
            self.bytes_read += len(data)
        return data

    def read(self, spans: list[tuple[str, int, int]]) -> list[bytes]:
        with ThreadPoolExecutor(WORKERS) as ex:
            return list(ex.map(lambda s: self._range(*s), spans))

    def headers(self) -> dict:
        """{tensor: (file, header entry, data start)} from each safetensors file's header."""
        if self._heads is None:
            shards = [f for f in self._sizes if f.endswith(".safetensors") and "/" not in f]
            heads = {}

            def one(f):
                first = self._range(f, 0, 1 << 16)
                n = struct.unpack("<Q", first[:8])[0]
                raw = first[8:8 + n] if 8 + n <= len(first) else first[8:] + self._range(f, len(first), 8 + n - len(first))
                return f, json.loads(raw), 8 + n

            with ThreadPoolExecutor(WORKERS) as ex:
                for f, header, start in ex.map(one, shards):
                    heads.update({k: (f, v, start) for k, v in header.items() if k != "__metadata__"})
            self._heads = heads
        return self._heads

    def signature(self) -> dict:
        if self._sig is None:
            stamp = self.sha
            self._sig = fingerprint.cached(f"hub:{self.name}", stamp, lambda: _remote_signature(self))
        return self._sig

    def small_files(self) -> Path:
        """Configs, code and templates, copied to a temporary folder."""
        if self._dir is None:
            self._dir = Path(tempfile.mkdtemp(prefix="llminspect-hub-"))
            want = [f for f, size in self._sizes.items() if Path(f).suffix.lower() in TEXT and size <= SMALL]
            with ThreadPoolExecutor(WORKERS) as ex:
                for f, data in zip(want, ex.map(lambda f: self._range(f, 0, max(1, self._sizes[f])), want)):
                    dest = self._dir / f
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(data)
        return self._dir


def _claims(base_model, tags: list[str]) -> list[dict]:
    """[{model, relation}] from a card's base_model and the Hub's base_model:<relation>:<model> tags."""
    rel = {}
    for t in tags:
        m = re.match(r"base_model:(\w+):(.+)$", t)
        if m and m.group(1) in RELATIONS:
            rel[m.group(2)] = m.group(1)
    named = [base_model] if isinstance(base_model, str) else list(base_model or [])
    return [{"model": b, "relation": rel.get(b, "finetune")} for b in dict.fromkeys(named + list(rel))]


def _remote_signature(src) -> dict:
    wanted = [(fingerprint.canonical(k), f, e, s) for k, (f, e, s) in sorted(src.headers().items())
              if fingerprint.ROLES.search(k) and e.get("dtype") in fingerprint.DTYPES]
    jobs = [(i, (f, at, n)) for i, (_, f, e, s) in enumerate(wanted) for at, n in fingerprint.spans(e, s)]
    raws = src.read([j for _, j in jobs])
    vals: dict[int, list[float]] = {}
    for (i, _), raw in zip(jobs, raws):
        vals.setdefault(i, []).extend(fingerprint.decode(wanted[i][2]["dtype"], raw))
    return {name: [e["shape"], vals.get(i, [])] for i, (name, _, e, _) in enumerate(wanted)}


def open_model(model: str, local: dict[str, Path] | None = None):
    """A source for a model: a folder path, a model on this machine by its Hub name, or the Hub."""
    p = Path(model).expanduser()
    if p.is_dir():
        return Folder(p)
    hit = (local or {}).get(model.lower())
    return Folder(hit, model) if hit else Hub(model, token=os.environ.get("HF_TOKEN"))


def local_models(roots=None) -> dict[str, Path]:
    """{lowercased name: folder} for the models on this machine."""
    from .check import default_roots, model_name
    return {model_name(p).lower(): p for _, p in filescan.find_repos(default_roots() if roots is None else roots)}


# ---- the edit: the rank of the difference over full rows ---------------------------------------------------

def _rows(src, layers: list[int]) -> dict[int, np.ndarray]:
    """ROWS full rows of each layer's down projection, at the same evenly spread rows in every model."""
    spans, keys = [], []
    for name, (f, e, start) in src.headers().items():
        m = DOWN.search(name)
        if not m or int(m.group(1)) not in layers or e.get("dtype") not in fingerprint.DTYPES or len(e["shape"]) != 2:
            continue
        out_dim, in_dim = e["shape"]
        size = fingerprint.DTYPES[e["dtype"]][1]
        for r in np.linspace(0, out_dim - 1, min(ROWS, out_dim)).round().astype(int):
            spans.append((f, start + e["data_offsets"][0] + int(r) * in_dim * size, in_dim * size))
            keys.append((int(m.group(1)), e["dtype"]))
    out: dict[int, list] = {}
    for (layer, dtype), raw in zip(keys, src.read(spans)):
        out.setdefault(layer, []).append(fingerprint.decode(dtype, raw))
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def _shape_of_change(cand, parent, layers: list[int]) -> dict:
    """Per sampled layer: how large the change is, how many directions hold most of it (k), and whether the
    layers whose change lies in a few directions share them (the overlap of their subspaces, 0..1)."""
    a, b = _rows(cand, layers), _rows(parent, layers)
    per, dirs = {}, {}
    for layer in sorted(set(a) & set(b)):
        if a[layer].shape != b[layer].shape:
            continue
        d = a[layer] - b[layer]
        size = float(np.linalg.norm(d))
        rel = size / max(float(np.linalg.norm(b[layer])), 1e-30)
        if size == 0:
            per[layer] = {"rel": 0.0, "top1": None, f"top{LOW_K}": None, "k": 0}
            continue
        u, s, _ = np.linalg.svd(d, full_matrices=False)
        e = s ** 2 / float((s ** 2).sum())
        k = int(np.searchsorted(np.cumsum(e), MOST) + 1)
        per[layer] = {"rel": round(rel, 5), "top1": round(float(e[0]), 3), f"top{LOW_K}": round(float(e[:LOW_K].sum()), 3), "k": k}
        if k <= FEW:
            dirs[layer] = u[:, :k]
    few = sorted(dirs)
    pairs = [float(np.linalg.norm(dirs[x].T @ dirs[y]) ** 2) / min(dirs[x].shape[1], dirs[y].shape[1])
             for i, x in enumerate(few) for y in few[i + 1:]]
    return {"layers": {str(k): v for k, v in per.items()}, "few": few,
            "shared": round(float(np.median(pairs)), 3) if pairs else None}


def _name_edit(sim: dict, change: dict) -> dict:
    """What was done, from the fingerprint and the rank of the change."""
    moved = sorted(int(l) for l, c in sim["layers"].items() if int(l) >= 0 and c < 0.99999)
    total = len([l for l in sim["layers"] if int(l) >= 0])
    per = change["layers"]
    changed = [l for l, p in per.items() if p["rel"] > 0]
    if sim["identical"] == sim["tensors"] and not changed:
        return {"kind": "same", "headline": "the same weights, re-uploaded", "moved_layers": []}
    span = f"layers {moved[0]}–{moved[-1]}" if len(moved) > 1 else f"layer {moved[0]}" if moved else "no sampled layer"
    few, many = change["few"], len(changed)
    one = [l for l in few if per[str(l)]["k"] == 1]
    if many and len(few) >= max(2, 0.6 * many) and (change["shared"] or 0) >= SHARED:
        k = int(np.median([per[str(l)]["k"] for l in few]))
        what = "one direction" if k == 1 else f"{k} directions"
        kind, head = "direction_removed", f"{what} removed from {len(moved) or many} of {total} layers, the signature of abliteration"
    elif many and len(one) >= 0.6 * many:
        kind, head = "rank_one", f"separate rank-one edits in {span}, like targeted fact edits"
    elif many and all(p[f"top{LOW_K}"] is not None and p[f"top{LOW_K}"] >= LOW_RANK for p in per.values() if p["rel"] > 0):
        kind, head = "low_rank", f"a low-rank change in {span}, like a merged adapter (LoRA)"
    elif sim["median"] > 0.9:
        kind, head = "fine_tuned", f"fine-tuned: weights moved in {len(moved)} of {total} layers"
    else:
        kind, head = "retrained", "heavily retrained: little of the parent's weights is left"
    return {"kind": kind, "headline": head, "moved_layers": moved}


# ---- unmask --------------------------------------------------------------------------------------------------

def _shapes(sig: dict) -> dict:
    return {k: v[0] for k, v in sig.items()}


def _comparable(a: dict, b: dict) -> bool:
    same = sum(1 for k, shape in _shapes(a).items() if k in b and b[k][0] == shape)
    return bool(a) and same >= 0.8 * max(len(a), len(b))


def _ancestry(claims: list[dict], local: dict, depth: int = DEPTH) -> list[dict]:
    """The claimed parents, then each first parent's own claims, up the chain."""
    out, seen, todo = [], set(), [(c, 1) for c in claims]
    while todo:
        c, d = todo.pop(0)
        if c["model"] in seen or d > depth:
            continue
        seen.add(c["model"])
        row = {"model": c["model"], "relation": c["relation"], "depth": d}
        try:
            src = open_model(c["model"], local)
            row["claims"] = [x["model"] for x in src.claims()]
            todo += [(x, d + 1) for x in src.claims()[:1]]
        except Unreachable as exc:
            row["error"] = str(exc)
        out.append(row)
    return out


def _verdict(s: dict | None) -> str:
    if s is None:
        return "another architecture: can't be compared"
    return ("the same weights" if s["identical"] == s["tensors"] else "derived from it" if s["median"] > 0.9 else
            "related, heavily retrained" if s["median"] >= 0.15 else "not derived from it")


def unmask(model: str, roots=None, say=lambda m: None) -> dict:
    """Who a model's parents are, whether its weights agree, what was done to them, and what its files hold."""
    t0 = time.perf_counter()
    local = local_models(roots)
    cand = open_model(model, local)
    say(f"[unmask] {cand.name}: reading the fingerprint")
    sig = cand.signature()
    out = {"model": cand.name, "url": cand.url, "sha": getattr(cand, "sha", None), "lineage": [], "parent": None,
           "edit": None, "checked": [], "not_checked": []}
    if not sig:
        out["not_checked"].append("origin and edits: no safetensors weights to read")
    lineage = _ancestry(cand.claims(), local)
    out["lineage"] = lineage

    tried, best = [], None
    for row in lineage:
        if row["depth"] != 1:
            continue
        name, relation = row["model"], row["relation"]
        if "error" in row:
            tried.append({"model": name, "claimed": relation, "reading": row["error"], "unreadable": True})
            continue
        if relation in ("quantized", "adapter"):
            tried.append({"model": name, "claimed": relation, "reading": f"{relation}: weights aren't comparable"})
            continue
        try:
            src = open_model(name, local)
            say(f"[unmask] {name}: reading the claimed parent's fingerprint")
            psig = src.signature()
        except Unreachable as exc:
            tried.append({"model": name, "claimed": relation, "reading": str(exc), "unreadable": True})
            continue
        s = fingerprint.similarity(sig, psig) if _comparable(sig, psig) else None
        tried.append({"model": name, "claimed": relation, "reading": _verdict(s), "median": s and round(s["median"], 4)})
        if s and s["median"] > 0.15 and (best is None or s["median"] > best[2]["median"]):
            best = (name, src, s, psig, True)
    if sig and (best is None or best[2]["median"] <= 0.9):  # the claim fails or is missing: look on this machine
        for name, folder in sorted(local.items()):
            if isinstance(cand, Folder) and folder.resolve() == cand.path.resolve():
                continue
            if not _comparable(sig, {fingerprint.canonical(k): [v[1].get("shape")] for k, v in fingerprint._headers(folder).items()
                                     if fingerprint.ROLES.search(k)}):
                continue
            src = Folder(folder)
            s = fingerprint.similarity(sig, src.signature())
            if s and s["median"] > 0.9 and (best is None or s["median"] > best[2]["median"]):
                best = (src.name, src, s, src.signature(), False)
    out["claims"] = tried

    if best:
        name, src, s, psig, claimed = best
        out["parent"] = {"model": name, "claimed": claimed, "reading": _verdict(s), "median": round(s["median"], 4),
                         "layers": s["layers"]}
        if s["median"] > 0.9:
            moved = [int(l) for l, c in s["layers"].items() if int(l) >= 0 and c < 0.99999]
            pool = moved or [int(l) for l in s["layers"] if int(l) >= 0]
            pick = sorted({pool[int(i)] for i in np.linspace(0, len(pool) - 1, min(RANK_LAYERS, len(pool))).round()}) if pool else []
            say(f"[unmask] reading {ROWS} rows of {len(pick)} layers to measure the change")
            change = _shape_of_change(cand, src, pick) if pick else {"layers": {}, "few": [], "shared": None}
            out["edit"] = _name_edit(s, change) | {"rank": change}
            out["checked"].append("edits: the rank of the change in sampled layers")
        out["checked"].append("origin: fingerprint against the claimed parents and this machine's models")
    elif sig:
        out["not_checked"].append("edits: no parent found to compare with (claimed parents unreadable, unrelated "
                                  "or another architecture, and no local model matches)")

    say("[unmask] reading configs, code and templates")
    out["files"] = _files(cand, best[1] if best else None)
    out["checked"].append("files: code that runs on load, configs and chat templates")
    out["not_checked"] += ["behavior: what the model says (run the Behavior step with the parent loaded)",
                           "capability: what the edit cost on tasks"]
    out["headline"] = _headline(out)
    out["bytes_read"] = cand.bytes_read + (best[1].bytes_read if best else 0)
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def _files(cand, parent) -> dict:
    files = cand.files()
    formats = sorted({Path(f).suffix.lower() for f in files if Path(f).suffix.lower() in filescan.WEIGHT_SUFFIXES})
    scan = filescan.scan_repo(cand.small_files(), cand.name)
    findings = [f for f in scan["findings"] if f["level"] != "info" or f["check"] == "code"]
    pickles = [f for f in files if Path(f).suffix.lower() in filescan.PICKLE_SUFFIXES]
    if pickles and isinstance(cand, Hub):
        findings.insert(0, {"level": "runs code on load" if ".safetensors" not in formats else "info", "check": "pickle",
                            "file": ", ".join(pickles[:3]), "evidence": [],
                            "detail": "Pickle weights can run code when loaded; they aren't read remotely. Check them "
                                      "locally before loading" + (", or load the safetensors copy." if ".safetensors" in formats else ".")})
    template = filediff.compare_templates(parent.small_files(), cand.small_files()) if parent else []
    return {"count": len(files), "bytes": sum(files.values()), "formats": formats, "findings": findings,
            "template": template}


def _headline(r: dict) -> str:
    p, e = r["parent"], r["edit"]
    unread = [c["model"] for c in r.get("claims", []) if c.get("unreadable")]
    if p is None:
        what = "No parent found to compare with"
    elif p["median"] <= 0.9:
        what = f"Related to {p['model']} but heavily retrained"
    elif not p["claimed"] and (unread or not r.get("claims")):
        what = ("" if r.get("claims") else "Its card names no parent. ") + (
            f"Compared with {p['model']}, the closest model here: {e['headline']}" if e else f"Closest here: {p['model']}")
    else:
        what = f"{p['model']}, {e['headline']}" if e else p["model"]
    false = [c["model"] for c in r.get("claims", []) if c["reading"] == "not derived from it"]
    if false and p and not p["claimed"]:
        what += f"; not {', '.join(false)} as its card claims"
    if unread:
        what += f"; the claimed {', '.join(unread)} couldn't be read"
    extra = []
    if any(t.get("added_text") for t in r["files"]["template"]):
        extra.append("its chat template adds text")
    top = r["files"]["findings"][0]["level"] if r["files"]["findings"] else None
    if top in ("runs code on load", "imports other code on load", "runs code if trusted"):
        extra.append(f"its files {top}")
    return what + ("; " + "; ".join(extra) if extra else "") + "."


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", help="ORG/NAME on the Hugging Face Hub, a model on this machine, or a folder")
    ap.add_argument("--json", action="store_true", help="print the whole result")
    args = ap.parse_args(argv)
    try:
        r = unmask(args.model, say=lambda m: print(m, file=sys.stderr))
    except Unreachable as exc:
        sys.exit(f"Can't read {args.model}: {exc}")
    if args.json:
        print(json.dumps(r, indent=1))
        return
    print(r["headline"])
    for row in r["lineage"]:
        print("  " * row["depth"] + f"↑ {row['relation']} of {row['model']}" + (f"  ({row['error']})" if "error" in row else ""))
    for c in r.get("claims", []):
        print(f"  claimed {c['model']}: {c['reading']}")
    print(f"read {r['bytes_read'] / 1e6:.1f} MB in {r['seconds']} s")


if __name__ == "__main__":
    main(sys.argv[1:])
