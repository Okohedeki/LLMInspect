"""The neuron index: where every MLP neuron fires most across a text corpus, and what its write
pushes, cached per model, so neurons can be found by text.

Corpus: BRAINSCAN_CORPUS (a .txt, .md or .jsonl file, or a folder of them; jsonl rows carry a
"text" field) when it is set, else the starter corpus bundled with Brainscan (starter_corpus.py:
short texts across many topics and registers), every recorded run's messages (deduplicated), the
calibration texts, and then any such files in <data>/corpus/ (scripts/fetch_corpus.py puts a sample
of web text there). The texts are joined into one token stream and read in windows of WINDOW tokens,
each after the sink token, on the original weights (applied edits are lifted for the build).

Kept per neuron: its TOP_K highest activations with their stream positions, its mean and spread
over the stream, and its logit lens (the tokens its write pushes most up and down, read through
the final norm and the unembedding). Saved under <data>/neuron_index/<model>/.

Search (search) for a text query returns three lists:
  fires on it    neurons that fire most on the query's words where they occur in the corpus (or on
                 the query itself when it doesn't occur), in units of each neuron's spread
  pushes it      neurons whose logit lens has a token starting with the query among its top tokens
  top contexts   neurons whose own top activations fall on an occurrence of the query
The query matches whole words ("Paris" finds neither "comparison" nor "parish"), and case matters
only when it has a capital letter.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .starter_corpus import STARTER
from .tracing import Busy
from .workbench import REFERENCE_TEXTS

WINDOW = 128          # tokens per window, the sink token included
TOP_K = 16            # top activations kept per neuron
LENS_UP, LENS_DOWN = 8, 4
BATCH = 8             # windows per forward pass
DEFAULT_TOKENS = 500_000
MAX_SPANS = 48        # occurrences of a query read again for "fires on it"
LIMIT = 20
SEP = "\n\n"
VERSION = 1


def _spaced(c: str) -> bool:
    """A letter or digit of a script that puts spaces between words (not Chinese, Japanese, Korean or Thai)."""
    return c.isalnum() and not ("\u0e00" <= c <= "\u0e7f" or "\u2e80" <= c <= "\ufaff" or "\uac00" <= c <= "\ud7af")


def pattern(q: str, whole: bool = True) -> re.Pattern:
    """The query as a regex: starting a word (and, when whole, ending one) wherever its ends are
    letters or digits of a spaced script, and case-blind unless the query has a capital letter."""
    body = re.escape(q)
    if _spaced(q[0]):
        body = r"(?<!\w)" + body
    if whole and _spaced(q[-1]):
        body += r"(?!\w)"
    return re.compile(body, 0 if any(c.isupper() for c in q) else re.IGNORECASE)


def text_files(root: Path) -> list[Path]:
    """The .txt, .md and .jsonl files at a path: the file itself, or every one under a folder."""
    return sorted(p for p in (root.rglob("*") if root.is_dir() else [root])
                  if p.is_file() and p.suffix.lower() in (".txt", ".md", ".jsonl"))


def read_texts(files: list[Path]) -> list[tuple[str, str]]:
    """(file name, text) pairs: a jsonl row's "text", or a .txt or .md file's paragraphs."""
    texts = []
    for f in files:
        if f.suffix.lower() == ".jsonl":
            for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and isinstance(row.get("text"), str) and row["text"].strip():
                    texts.append((f.name, row["text"]))
        else:
            texts += [(f.name, t) for t in re.split(r"\n\s*\n", f.read_text(encoding="utf-8", errors="replace")) if t.strip()]
    return texts


def corpus_texts(recorder) -> tuple[list[tuple[str, str]], list[str]]:
    """(source, text) pairs, and a line per source saying where they came from."""
    path = os.environ.get("BRAINSCAN_CORPUS", "").strip()
    if path:
        files = text_files(Path(path).expanduser())
        if not files:
            raise ValueError(f"BRAINSCAN_CORPUS has no .txt, .md or .jsonl files: {path}")
        return read_texts(files), [f"BRAINSCAN_CORPUS: {len(files)} file{'' if len(files) == 1 else 's'} under {path}"]
    texts = [(f"starter:{topic}", text) for topic, text in STARTER]
    seen, runs = set(), []
    for meta in recorder.store.list_runs():
        for m in recorder.store.read_jsonl(meta["id"], "messages"):
            c = m.get("content")
            if isinstance(c, str) and len(c.strip()) > 20 and c not in seen:
                seen.add(c)
                runs.append((f"run:{meta['id']}", c[:6000]))
    texts += runs + [("calibration", t) for t in REFERENCE_TEXTS]
    sources = [f"the starter corpus ({len(STARTER)} texts)", f"recorded runs' messages ({len(runs)} unique)",
               f"the calibration texts ({len(REFERENCE_TEXTS)})"]
    folder = recorder.store.root / "corpus"
    files = text_files(folder) if folder.is_dir() else []
    if files:  # last, so a token cap trims the downloaded text before the texts above
        more = read_texts(files)
        texts += more
        sources.append(f"{', '.join(f.name for f in files)} in {folder.name}/ ({len(more):,} texts)")
    return texts, sources


class NeuronIndex:
    def __init__(self, recorder):
        self.rec, self.s = recorder, recorder.s
        self.dir = recorder.store.root / "neuron_index" / recorder.s.slug
        self.data: dict | None = None
        self.meta: dict | None = None
        self.job = {"status": "none", "step": None, "progress": 0.0, "error": None}
        self._strs: dict[int, str] = {}
        self._text = None
        self._flat = None
        self._vocab = None
        self._load()

    # ---- state ---------------------------------------------------------------------------

    def status(self) -> dict:
        out = dict(self.job)
        if self.meta:
            out |= {"meta": self.meta, "status": out["status"] if out["status"] == "building" else "ready"}
        return out

    def ready(self) -> bool:
        return self.data is not None

    def _load(self) -> None:
        meta_path, data_path = self.dir / "index.json", self.dir / "index.safetensors"
        if not (meta_path.exists() and data_path.exists()):
            return
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("version") != VERSION or meta.get("model") != self.s.spec:
                return
            self.data, self.meta = load_file(str(data_path)), meta
        except (OSError, ValueError):
            self.data = self.meta = None

    def _layers(self) -> list[int]:
        return [] if not self.meta else self.meta["layers"]

    # ---- build ---------------------------------------------------------------------------

    def build_async(self, max_tokens: int = DEFAULT_TOKENS) -> dict:
        if self.job["status"] == "building":
            raise Busy("the index is being built already")
        max_tokens = max(WINDOW * 2, min(int(max_tokens), 20_000_000))
        self.job = {"status": "building", "step": "reading the corpus", "progress": 0.0, "error": None}

        def work():
            try:
                self.build(max_tokens, lambda step, f: self.job.update(step=step, progress=round(f, 3)))
                self.job = {"status": "ready", "step": "done", "progress": 1.0, "error": None}
            except Exception as exc:  # reported on the job; the old index (if any) stays
                self.job = {"status": "error", "step": None, "progress": 0.0, "error": f"{type(exc).__name__}: {exc}"}

        threading.Thread(target=work, daemon=True, name="neuron-index").start()
        return self.status()

    def build(self, max_tokens: int = DEFAULT_TOKENS, progress=None) -> dict:
        say = progress or (lambda *_: None)
        rec, s, nm = self.rec, self.s, self.rec.neurons
        t0 = time.time()
        texts, sources = corpus_texts(rec)
        stream, doc_starts = [], []
        for _, text in texts:
            if len(stream) >= max_tokens:
                break
            doc_starts.append(len(stream))
            stream += s.tokenizer(text + SEP, add_special_tokens=False).input_ids
        stream = stream[:max_tokens]
        if len(stream) < 2:
            raise ValueError("the corpus is empty")
        step = WINDOW - 1
        windows = [stream[i:i + step] for i in range(0, len(stream), step)]
        layers = [i for i in range(s.n_layers) if nm._try_mlp(i)]
        dev = s.device
        top_v = {i: torch.full((nm.mlp(i).width, TOP_K), -math.inf, device=dev) for i in layers}
        top_p = {i: torch.full((nm.mlp(i).width, TOP_K), -1, dtype=torch.long, device=dev) for i in layers}
        s1 = {i: torch.zeros(nm.mlp(i).width, device=dev, dtype=torch.float64) for i in layers}
        s2 = {i: torch.zeros(nm.mlp(i).width, device=dev, dtype=torch.float64) for i in layers}
        cur = {"pos": None}

        def hook(i):
            def fn(module, args):
                a = args[0][:, 1:].reshape(-1, args[0].shape[-1]).float()  # drop the sink token
                s1[i] += a.sum(0, dtype=torch.float64)
                s2[i] += (a * a).sum(0, dtype=torch.float64)
                k = min(TOP_K, a.shape[0])
                v, j = a.topk(k, dim=0)                                   # (k, N)
                cat_v = torch.cat([top_v[i], v.T], 1)
                cat_p = torch.cat([top_p[i], cur["pos"][j].T], 1)
                best, idx = cat_v.topk(TOP_K, dim=1)
                top_v[i], top_p[i] = best, cat_p.gather(1, idx)
            return fn

        if not rec.lock.acquire(timeout=5.0):
            raise Busy("the model is generating; build the index when it finishes")
        handles = [nm.mlp(i).down.register_forward_pre_hook(hook(i)) for i in layers]
        try:
            with torch.no_grad(), rec.tap.paused(), rec.edits.suspended():
                # Every window is full but the last one, which is read on its own.
                full = windows if len(windows[-1]) == step else windows[:-1]
                for b in range(0, len(full), BATCH):
                    if rec.closed:
                        raise RuntimeError("the model was switched while the index was being built")
                    part = full[b:b + BATCH]
                    x = torch.tensor([[s.sink_id] + w for w in part], device=dev)
                    starts = torch.arange(b, b + len(part), device=dev) * step
                    cur["pos"] = (starts[:, None] + torch.arange(step, device=dev)[None]).reshape(-1)
                    s.model(input_ids=x, use_cache=False, logits_to_keep=1)
                    say("reading the corpus", 0.9 * min(1.0, (b + BATCH) / len(windows)))
                if len(full) < len(windows):
                    last = windows[-1]
                    x = torch.tensor([[s.sink_id] + last], device=dev)
                    cur["pos"] = torch.arange(len(last), device=dev) + len(full) * step
                    s.model(input_ids=x, use_cache=False, logits_to_keep=1)
                say("reading the logit lens of every neuron", 0.92)
                lens = self._lens_all(layers)
        finally:
            for h in handles:
                h.remove()
            rec.lock.release()
        n = float(len(stream))
        data = {"stream": torch.tensor(stream, dtype=torch.int32), "doc_starts": torch.tensor(doc_starts, dtype=torch.int64)}
        for i in layers:
            mean = s1[i] / n
            std = (s2[i] / n - mean * mean).clamp_min(1e-12).sqrt()
            data[f"top_v.{i}"] = top_v[i].float().cpu().contiguous()
            data[f"top_p.{i}"] = top_p[i].to(torch.int64).cpu().contiguous()
            data[f"mean.{i}"] = mean.float().cpu()
            data[f"std.{i}"] = std.float().cpu()
            for k, v in lens[i].items():
                data[f"{k}.{i}"] = v.cpu().contiguous()
        fingerprint = hashlib.sha1(json.dumps([len(stream), stream[:2000], stream[-2000:]]).encode()).hexdigest()[:12]
        meta = {"version": VERSION, "model": s.spec, "layers": layers, "widths": [nm.mlp(i).width for i in layers],
                "window": WINDOW, "top_k": TOP_K, "tokens": len(stream), "texts": len(doc_starts), "sources": sources,
                "fingerprint": fingerprint, "built": time.time(), "seconds": round(time.time() - t0, 1),
                "weights": "original (applied edits lifted)"}
        self.dir.mkdir(parents=True, exist_ok=True)
        save_file(data, str(self.dir / "index.safetensors"))
        (self.dir / "index.json").write_text(json.dumps(meta, indent=2))
        self.data, self.meta = data, meta
        self._text = self._flat = None
        say("done", 1.0)
        return meta

    def _lens_all(self, layers: list[int], chunk: int = 512) -> dict:
        """Top tokens every neuron's write pushes up and down (logit lens of W_down[:, n])."""
        s, nm = self.s, self.rec.neurons
        p = next(s.norm.parameters(), None)
        gain = s.norm(torch.ones(1, s.d_model, device=s.device, dtype=p.dtype if p is not None else torch.float32))[0]
        unembed = s.lm_head.weight
        out = {}
        for i in layers:
            down = nm.mlp(i).down.weight.detach()  # (d, N)
            ups, upv, downs, downv = [], [], [], []
            for c in range(0, down.shape[1], chunk):
                w = down[:, c:c + chunk].to(unembed.dtype) * gain.to(unembed.dtype)[:, None]
                logits = (unembed @ w).float()  # (V, chunk)
                v, j = logits.topk(LENS_UP, dim=0)
                ups.append(j.T.to(torch.int32)); upv.append(v.T)
                v, j = logits.topk(LENS_DOWN, dim=0, largest=False)
                downs.append(j.T.to(torch.int32)); downv.append(v.T)
            out[i] = {"lens_up": torch.cat(ups), "lens_up_v": torch.cat(upv).half(),
                      "lens_down": torch.cat(downs), "lens_down_v": torch.cat(downv).half()}
        return out

    # ---- reading the stream --------------------------------------------------------------

    def _tok(self, i: int) -> str:
        if i not in self._strs:
            self._strs[i] = self.s.token_str(i)
        return self._strs[i]

    def _ensure_text(self) -> None:
        """The stream decoded once: its text and each token's first character."""
        if self._text is not None:
            return
        ids = self.data["stream"].tolist()
        starts, parts, at = [], [], 0
        for i in ids:
            t = self._tok(i)
            starts.append(at)
            parts.append(t)
            at += len(t)
        self._text, self._starts = "".join(parts), starts
        self._docs = self.data["doc_starts"].tolist()

    def snippet(self, p: int, before: int = 12, after: int = 4) -> dict:
        """The tokens around stream position p, kept inside its own text."""
        self._ensure_text()
        stream = self.data["stream"]
        k = bisect.bisect_right(self._docs, p) - 1
        lo = self._docs[k] if k >= 0 else 0
        hi = self._docs[k + 1] if k + 1 < len(self._docs) else len(stream)
        b = [self._tok(int(i)) for i in stream[max(lo, p - before):p]]
        a = [self._tok(int(i)) for i in stream[p + 1:min(hi, p + 1 + after)]]
        return {"before": "".join(b), "token": self._tok(int(stream[p])), "after": "".join(a)}

    def contexts(self, layer: int, n: int, k: int = 10) -> dict | None:
        """A neuron's top activations in the corpus, with their contexts."""
        if not self.ready() or layer not in self._layers():
            return None
        v, p = self.data[f"top_v.{layer}"][n], self.data[f"top_p.{layer}"][n]
        rows = [{"act": round(float(v[j]), 4), "pos": int(p[j]), **self.snippet(int(p[j]))}
                for j in range(min(k, len(v))) if int(p[j]) >= 0]
        return {"contexts": rows, "mean": round(float(self.data[f"mean.{layer}"][n]), 4),
                "std": round(float(self.data[f"std.{layer}"][n]), 4), "tokens": self.meta["tokens"]}

    # ---- search ----------------------------------------------------------------------------

    def search(self, query: str, limit: int = LIMIT) -> dict:
        if not self.ready():
            raise ValueError("build the neuron index first")
        q = str(query).strip()
        if len(q) < 2:
            raise ValueError("search for at least two characters")
        spans = self._spans(q)
        return {"query": q, "matches": len(spans), "in_corpus": bool(spans),
                "fires_on": self._fires_on(q, spans, limit), "pushes": self._pushes(q, limit),
                "contexts": self._in_contexts(spans, limit), "tokens": self.meta["tokens"]}

    def _spans(self, q: str) -> list[tuple[int, int]]:
        """Token spans (first, last) where the query occurs in the stream."""
        self._ensure_text()
        out = []
        for m in pattern(q).finditer(self._text):
            a = bisect.bisect_right(self._starts, m.start()) - 1
            b = bisect.bisect_right(self._starts, m.end() - 1) - 1
            out.append((a, b))
        return out

    def _fires_on(self, q: str, spans: list, limit: int) -> list[dict]:
        """Neurons that fire most on the query, read again where it occurs (or on the query alone)."""
        s, rec, step = self.s, self.rec, WINDOW - 1
        if len(spans) > MAX_SPANS:  # spread over the corpus rather than its first occurrences
            spans = [spans[int(k * len(spans) / MAX_SPANS)] for k in range(MAX_SPANS)]
        stream = self.data["stream"]
        runs = []  # (ids, [(first, last) in those ids], stream position of the first match or None)
        if spans:
            for a, b in spans:
                lo = max(0, a - (step - (b - a) - 1))
                ids = stream[lo:b + 1].tolist()
                runs.append((ids, (a - lo, b - lo), a))
        else:
            ids = s.tokenizer(q, add_special_tokens=False).input_ids[: step]
            runs.append((ids, (0, len(ids) - 1), None))
        layers = self._layers()
        sums = {i: 0 for i in layers}
        acts = {}

        def grab(i):
            def fn(module, args):
                acts[i] = args[0][0].float()
            return fn

        stats = {i: (self.data[f"mean.{i}"].to(s.device), self.data[f"std.{i}"].to(s.device).clamp_min(1e-4)) for i in layers}
        if not rec.lock.acquire(timeout=1.0):
            raise Busy("the model is generating; search again when it finishes")
        handles = [rec.neurons.mlp(i).down.register_forward_pre_hook(grab(i)) for i in layers]
        try:
            with torch.no_grad(), rec.tap.paused(), rec.edits.suspended():
                for ids, (a, b), _ in runs:
                    s.model(input_ids=torch.tensor([[s.sink_id] + ids], device=s.device), use_cache=False, logits_to_keep=1)
                    for i in layers:
                        mean, std = stats[i]
                        z = (acts[i][1 + a:2 + b] - mean) / std  # the query's tokens (row 0 is the sink)
                        sums[i] = sums[i] + z.max(0).values
        finally:
            for h in handles:
                h.remove()
            rec.lock.release()
        rows = []
        for i in layers:
            z = sums[i] / len(runs)
            v, j = z.topk(min(limit, len(z)))
            rows += [(float(v[k]), i, int(j[k])) for k in range(len(v))]
        rows.sort(key=lambda r: -r[0])
        first = runs[0][2]
        ctx = self.snippet(first) if first is not None else None
        return [{"neuron": f"L{i} N{n}", "layer": i, "index": n, "z": round(z, 2), "context": ctx} for z, i, n in rows[:limit]]

    def _pushes(self, q: str, limit: int) -> list[dict]:
        """Neurons whose logit lens has a token starting with the query among its top tokens."""
        if self._vocab is None:
            self._vocab = [self._tok(i) for i in range(self.s.vocab_size)]
        p = pattern(q, whole=False)
        hits = [i for i, t in enumerate(self._vocab) if t.strip() and p.match(t.strip())]
        if not hits:
            return []
        S = torch.tensor(hits[:5000])
        rows = []
        for i in self._layers():
            up, upv = self.data[f"lens_up.{i}"].long(), self.data[f"lens_up_v.{i}"].float()
            mask = torch.isin(up, S)
            if not mask.any():
                continue
            rank = torch.where(mask.any(1), mask.float().argmax(1), torch.tensor(LENS_UP))
            for n in torch.nonzero(mask.any(1)).flatten().tolist():
                r = int(rank[n])
                rows.append((r, -float(upv[n, r]), i, n, [self._tok(int(t)) for t in up[n][mask[n]]]))
        rows.sort()
        return [{"neuron": f"L{i} N{n}", "layer": i, "index": n, "rank": r, "tokens": toks} for r, _, i, n, toks in rows[:limit]]

    def _in_contexts(self, spans: list, limit: int) -> list[dict]:
        """Neurons whose own top activations fall on an occurrence of the query."""
        if not spans:
            return []
        if self._flat is None:  # every kept top activation, sorted by stream position
            pos, lay, neu, val = [], [], [], []
            for i in self._layers():
                p = self.data[f"top_p.{i}"].numpy()
                pos.append(p.reshape(-1))
                lay.append(np.full(p.size, i, dtype=np.int32))
                neu.append(np.repeat(np.arange(p.shape[0], dtype=np.int32), p.shape[1]))
                val.append(self.data[f"top_v.{i}"].numpy().reshape(-1))
            pos, lay, neu, val = (np.concatenate(x) for x in (pos, lay, neu, val))
            order = np.argsort(pos, kind="stable")
            self._flat = (pos[order], lay[order], neu[order], val[order])
        pos, lay, neu, val = self._flat
        found: dict[tuple[int, int], list] = {}
        for a, b in spans:
            lo, hi = np.searchsorted(pos, a, "left"), np.searchsorted(pos, b, "right")
            for k in range(lo, hi):
                key = (int(lay[k]), int(neu[k]))
                f = found.setdefault(key, [0, -math.inf, None])
                f[0] += 1
                if val[k] > f[1]:
                    f[1], f[2] = float(val[k]), int(pos[k])
        rows = sorted(found.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))[:limit]
        return [{"neuron": f"L{i} N{n}", "layer": i, "index": n, "hits": c, "act": round(v, 4), "context": self.snippet(p)}
                for (i, n), (c, v, p) in rows]
