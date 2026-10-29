"""The agent-hijack guard: notice from the model's internals when instructions hidden in a tool result
reach it, show which attention heads read them, and optionally stop the request.

LLMInspect ships no examples of injected text. The guard learns from labeled cases you provide: jsonl
files in <data>/hijack/ with rows {"task": the user's request, "tool_result": the text a tool returned,
"injected": true or false, "group": optional, e.g. the task's name}. A public prompt-injection
benchmark, converted to these fields, is the intended source; LLMInspect never downloads one itself.

  features   each case is rendered as an agent turn (the user's task, a tool call, the tool's result)
             and read once: the residual stream at every layer at the last prompt token, where the reply
             starts, and on the first generated tokens (greedy)
  probe      a difference-of-means direction between injected and clean cases, scored held out: cases are
             split into folds by group (or by row when there are no groups), the layer is chosen inside
             each fold from the training folds only, and every score comes from a direction that never
             saw the case. Reported as AUROC.
  baseline   a text-only detector trained on the same folds: a regularized bag of words over the tool
             result. The probe is only worth having where it beats this.
  guard      the direction built on every case at the chosen layer, saved to the bank as "tool-hijack",
             z-scored on the clean cases' generated tokens, plus (if asked) a monitor on the model's own
             tokens that stops the request when it fires (monitors.py, stop). The threshold lets through
             TARGET_FPR of clean cases. A flag means the model is reading something like the injected
             cases; it is a pointer for review, not proof that the model obeyed.
  explain    at a token of a recorded run: the heads that put the most attention on tool-result tokens,
             and the tool-result tokens they attend to most. Softmax-attention heads only.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import numpy as np
import torch

from .chat import render
from .tracing import Busy

DIRECTION = "tool-hijack"
GEN_TOKENS = 8
MAX_CASES = 2000
MIN_CASES = 8          # per class
FOLDS = 5
TARGET_FPR = 0.05
WORD = re.compile(r"[a-z0-9']+")
TOOLS = [{"type": "function", "function": {
    "name": "fetch", "description": "Fetch a document and return its text.",
    "parameters": {"type": "object", "properties": {"source": {"type": "string"}}, "required": ["source"]}}}]


def auroc(scores, labels) -> float:
    """Area under the ROC curve from ranks (ties count half)."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):  # average ranks over ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load_cases(folder: Path) -> list[dict]:
    rows = []
    for f in sorted(folder.glob("*.jsonl")) if folder.is_dir() else []:
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict) and isinstance(r.get("tool_result"), str) and isinstance(r.get("injected"), bool):
                rows.append({"task": str(r.get("task") or "Summarize this document."), "tool_result": r["tool_result"],
                             "injected": r["injected"], "group": str(r.get("group") or f"{f.stem}:{len(rows)}")})
            if len(rows) >= MAX_CASES:
                return rows
    return rows


def folds(cases: list[dict], k: int = FOLDS) -> list[int]:
    """A fold for each case, whole groups kept together."""
    groups = sorted({c["group"] for c in cases})
    of = {g: i % k for i, g in enumerate(groups)}
    return [of[c["group"]] for c in cases]


def messages(case: dict) -> list[dict]:
    return [{"role": "user", "content": case["task"]},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "fetch", "arguments": {"source": "the document"}}}]},
            {"role": "tool", "name": "fetch", "content": case["tool_result"]}]


# ---- the text baseline -----------------------------------------------------------------------------

def _bag(texts: list[str], vocab: dict[str, int]) -> np.ndarray:
    x = np.zeros((len(texts), len(vocab)))
    for i, t in en