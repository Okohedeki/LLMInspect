"""GGUF weights read by byte range: the header's tensor table, and tensor values dequantized together with the
size of each value's rounding step. Quantization rounds every value by up to half a step, so a comparison with
the unquantized parent can tell that rounding from a real change.

Supported: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS. Tensors
of other types (the rest of the IQ family) are skipped.

GGUF names are mapped to the Hugging Face ones the fingerprint uses, shapes are given in Hugging Face order, and
for the Llama family the query projection's rows are put back in the order the conversion to GGUF permuted them
out of.
"""

from __future__ import annotations

import re
import struct

import numpy as np

# type id: (name, values per block, bytes per block)
TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
         7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
         12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 20: ("IQ4_NL", 32, 18),
         23: ("IQ4_XS", 256, 136), 30: ("BF16", 1, 2)}
NAMES = {"token_embd.weight": "embed_tokens.weight"}
BLOCK = re.compile(r"blk\.(\d+)\.(attn_q|ffn_down)\.weight$")
ROLE = {"attn_q": "self_attn.q_proj", "ffn_down": "mlp.down_proj"}
PERMUTED = {"llama", "granite", "granitemoe"}  # architectures whose query rows the conversion reorders for RoPE
# quantizations a repo's files are chosen by: Q8_0's rounding is small enough to see through and its values half
# the size of F16's, then the rest by precision
PREFER = ["q8_0", "f16", "bf16", "f32", "q6_k", "q5_k_m", "q5_k_s", "q5_k", "q5_1", "q5_0", "q4_k_m", "q4_k_s",
          "q4_k", "iq4_xs", "iq4_nl", "q4_1", "q4_0", "q3_k_l", "q3_k_m", "q3_k_s", "q3_k", "q2_k"]
SPLIT = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)
CHUNK = 1 << 20


def choose(files: dict[str, int]) -> list[str]:
    """The GGUF files of a repo's most precise supported quantization (every part of a split file), or []."""
    groups: dict[str, list[str]] = {}
    for f in files:
        low = f.lower()
        if low.endswith(".gguf") and not any(w in low for w in ("mmproj", "imatrix", "vision", "projector")):
            groups.setdefault(SPLIT.sub("", f), []).append(f)

    def rank(stem: str) -> tuple:
        tags = re.split(r"[-_.]", stem.lower().rsplit("/", 1)[-1])
        joined = "_".join(tags)
        for i, q in enumerate(PREFER):
            if re.search(rf"(^|_){q}($|_)", joined):
                return (i, stem)
        loose = re.search(r"(^|_)q([2-8])($|_)", joined)  # "model_q8.gguf": by its bits
        return (len(PREFER) + 8 - int(loose.group(2)), stem) if loose else (len(PREFER) + 8, stem)

    if not groups:
        return []
    best = min(groups, key=rank)
    return sorted(groups[best]) if rank(best)[0] < len(PREFER) + 8 or len(groups) == 1 else []


def label(files: list[str], types: dict[int, int]) -> str:
    """The quantization's name: from the file name (Q4_K_M), else the commonest tensor type."""
    stem = re.split(r"[-_.]", SPLIT.sub("", files[0]).lower().rsplit("/", 1)[-1])
    joined = "_".join(stem)
    for q in PREFER:
        if re.search(rf"(^|_){q}($|_)", joined):
            return q.upper()
    t = max(types, key=types.get) if types else None
    return TYPES.get(t, (f"type {t}",))[0]


class _Cursor:
    """Sequential reads over a file fetched in CHUNK pieces."""

    def __init__(self, fetch, size: int):
        self.fetch, self.size, self.buf, self.base, self.pos = fetch, size, bytearray(), 0, 0

    def take(self, n: int) -> bytes:
        while self.pos + n > self.base + len(self.buf):
            at = self.base + len(self.buf)
            if at >= self.size:
                raise ValueError("GGUF header runs past the end of the file")
            self.buf += self.fetch(at, min(max(CHUNK, n), self.size - at))
        i = self.pos - self.base
        out = bytes(self.buf[i:i + n])
        self.pos += n
        if i > 4 * CHUNK:  # drop what's been read
            del self.buf[:i + n]
            self.base = self.pos
        return out

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def text(self) -> str:
        return self.take(self.u64()).decode("utf-8", "replace")


_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _value(c: _Cursor, t: int):
    if t in _SCALAR:
        fmt = _SCALAR[t]
        return struct.unpack(fmt, c.take(struct.calcsize(fmt)))[0]
    if t == 8:
        return c.text()
    if t == 9:  # arrays (the vocabulary, merges) are walked past, not kept
        sub, n = c.u32(), c.u64()
        if sub in _SCALAR:
            c.take(n * struct.calcsize(_SCALAR[sub]))
        else:
            for _ in range(n):
                _value(c, sub)
        return None
    raise ValueError(f"unknown GGUF value type {t}")


def header(fetch, size: int) -> dict:
    """{meta: scalar metadata, tensors: {name: {type, dims, at}}} of one GGUF file; `at` is the byte offset."""
    c = _Cursor(fetch, size)
    if c.take(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    version = c.u32()
    if version < 2:
        raise ValueError(f"GGUF version {version} is too old to read")
    n_tensors, n_kv = c.u64(), c.u64()
    meta = {}
    for _ in range(n_kv):
        key = c.text()
        v = _value(c, c.u32())
        if v is not None:
            meta[key] = v
    infos = []
    for _ in range(n_tensors):
        name = c.text()
        dims = [c.u64() for _ in range(c.u32())]
        infos.append((name, dims, c.u32(), c.u64()))
    align = int(meta.get("general.alignment", 32))
    start = -(-c.pos // align) * align
    return {"meta": meta, "tensors": {n: {"type": t, "dims": d, "at": start + off} for n, d, t, off in infos}}


def canonical(name: str) -> str | None:
    """The Hugging Face name fingerprinting knows a GGUF tensor by, or None."""
    if name in NAMES:
        return NAMES[name]
    m = BLOCK.match(name)
    return f"layers.{m.group(1)}.{ROLE[m.group(2)]}.weight" if m else None


# ---- dequantization: (values, step) per block type ------------------------------------------------------------

def _f16(raw: np.ndarray, at: int) -> np.ndarray:
    return raw[:, at:at + 2].copy().view("<f2").astype(np.float32)[:, 0]


def _split_nibbles(q: np.ndarray) -> np.ndarray:
    """(blocks, 16) bytes to (blocks, 32) values: the low nibbles then the high ones."""
    return np.concatenate([q & 0xF, q >> 4], axis=1).astype(np.float32)


def _high_bits(qh: np.ndarray) -> np.ndarray:
    """Q5_0/Q5_1's 32 fifth bits, one per value, as 0 or 16."""
    bits = qh.copy().view("<u4")[:, :1] >> np.arange(32, dtype=np.uint32)
    return ((bits & 1) * 16).astype(np.float32)


def _scale_min_k4(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Q4_K/Q5_K's eight 6-bit scales and mins, packed in 12 bytes."""
    s = s.astype(np.uint8)
    sc = np.empty((len(s), 8), np.uint8)
    mn = np.empty((len(s), 8), np.uint8)
    sc[:, :4], mn[:, :4] = s[:, :4] & 63, s[:, 4:8] & 63
    sc[:, 4:] = (s[:, 8:12] & 0xF) | ((s[:, :4] >> 6) << 4)
    mn[:, 4:] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    return sc.astype(np.float32), mn.astype(np.float32)


def _q4_0(b):
    d = _f16(b, 0)[:, None]
    return d * (_split_nibbles(b[:, 2:18]) - 8), np.broadcast_to(np.abs(d), (len(b), 32))


def _q4_1(b):
    d, m = _f16(b, 0)[:, None], _f16(b, 2)[:, None]
    return d * _split_nibbles(b[:, 4:20]) + m, np.broadcast_to(np.abs(d), (len(b), 32))


def _q5_0(b):
    d = _f16(b, 0)[:, None]
    return d * (_split_nibbles(b[:, 6:22]) + _high_bits(b[:, 2:6]) - 16), np.broadcast_to(np.abs(d), (len(b), 32))


def _q5_1(b):
    d, m = _f16(b, 0)[:, None], _f16(b, 2)[:, None]
    return d * (_split_nibbles(b[:, 8:24]) + _high_bits(b[:, 4:8])) + m, np.broadcast_to(np.abs(d), (len(b), 32))


def _q8_0(b):
    d = _f16(b, 0)[:, None]
    return d * b[:, 2:34].copy().view(np.int8).astype(np.float32), np.broadcast_to(np.abs(d), (len(b), 32))


def _twobit_layout(q: np.ndarray) -> np.ndarray:
    """Q2_K/Q3_K's 256 two-bit values from 64 bytes, in output order: halves, then shifts, then 16-value groups."""
    out = np.empty((len(q), 2, 4, 2, 16), np.uint8)
    for h in range(2):
        for j in range(4):
            for s in range(2):
                out[:, h, j, s] = (q[:, h * 32 + s * 16:h * 32 + s * 16 + 16] >> (2 * j)) & 3
    return out.reshape(len(q), 16, 16).astype(np.float32)


def _q2_k(b):
    sc = b[:, :16]
    d, dmin = _f16(b, 80)[:, None, None], _f16(b, 82)[:, None, None]
    dl = d * (sc & 0xF).astype(np.float32)[:, :, None]
    ml = dmin * (sc >> 4).astype(np.float32)[:, :, None]
    q = _twobit_layout(b[:, 16:80])
    return (dl * q - ml).reshape(len(b), 256), np.broadcast_to(np.abs(dl), (len(b), 16, 16)).reshape(len(b), 256)


def _q3_k(b):
    hm, qs, s = b[:, :32], b[:, 32:96], b[:, 96:108]
    d = _f16(b, 108)[:, None, None]
    idx = np.arange(16)
    low = np.where(idx < 8, s[:, idx % 8] & 0xF, s[:, idx % 8] >> 4)
    high = (s[:, 8 + idx % 4] >> (2 * (idx // 4))) & 3
    sc = (low | (high << 4)).astype(np.float32) - 32
    q = _twobit_layout(qs)
    # the high bit: bit (h*4 + j) of hmask byte s*16 + l; a cleared bit subtracts 4
    hbit = np.empty((len(b), 2, 4, 2, 16), np.float32)
    for h in range(2):
        for j in range(4):
            for t in range(2):
                hbit[:, h, j, t] = np.where(hm[:, t * 16:t * 16 + 16] & (1 << (h * 4 + j)), 0, 4)
    dl = d * sc[:, :, None]
    return (dl * (q - hbit.reshape(len(b), 16, 16))).reshape(len(b), 256), \
        np.broadcast_to(np.abs(dl), (len(b), 16, 16)).reshape(len(b), 256)


def _q45_k(b, fifth: bool):
    d, dmin = _f16(b, 0)[:, None, None], _f16(b, 2)[:, None, None]
    sc, mn = _scale_min_k4(b[:, 4:16])
    qh = b[:, 16:48] if fifth else None
    ql = b[:, 48:176] if fifth else b[:, 16:144]
    q = np.empty((len(b), 8, 32), np.float32)
    for j in range(4):
        lo, hi = ql[:, j * 32:(j + 1) * 32] & 0xF, ql[:, j * 32:(j + 1) * 32] >> 4
        q[:, 2 * j], q[:, 2 * j + 1] = lo, hi
        if fifth:
            q[:, 2 * j] += np.where(qh & (1 << (2 * j)), 16, 0)
            q[:, 2 * j + 1] += np.where(qh & (2 << (2 * j)), 16, 0)
    dl, ml = d * sc[:, :, None], dmin * mn[:, :, None]
    return (dl * q - ml).reshape(len(b), 256), np.broadcast_to(np.abs(dl), (len(b), 8, 32)).reshape(len(b), 256)


def _q6_k(b):
    ql, qh, sc = b[:, :128], b[:, 128:192], b[:, 192:208].copy().view(np.int8).astype(np.float32)
    d = _f16(b, 208)[:, None]
    q = np.empty((len(b), 2, 4, 32), np.float32)
    steps = np.empty((len(b), 2, 4, 32), np.float32)
    for n in range(2):
        l_, h_ = ql[:, n * 64:(n + 1) * 64], qh[:, n * 32:(n + 1) * 32]
        q[:, n, 0] = (l_[:, :32] & 0xF) | ((h_ & 3) << 4)
        q[:, n, 1] = (l_[:, 32:] & 0xF) | (((h_ >> 2) & 3) << 4)
        q[:, n, 2] = (l_[:, :32] >> 4) | (((h_ >> 4) & 3) << 4)
        q[:, n, 3] = (l_[:, 32:] >> 4) | (((h_ >> 6) & 3) << 4)
        for k in range(4):  # scale index: 8n + 2k for the first 16 values, + 1 for the next 16
            s = np.repeat(sc[:, [8 * n + 2 * k, 8 * n + 2 * k + 1]], 16, axis=1)
            steps[:, n, k] = d * s
    vals = steps * (q - 32)
    return vals.reshape(len(b), 256), np.abs(steps).reshape(len(b), 256)


IQ4 = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float32)
# the IQ4 grid isn't even: a value's rounding step is the mean gap to its neighbours on the grid
IQ4_GAP = np.array([(IQ4[min(i + 1, 15)] - IQ4[max(i - 1, 0)]) / (min(i + 1, 15) - max(i - 1, 0)) for i in range(16)],
                   np.float32)


def _iq4(dl: np.ndarray, q: np.ndarray):
    q = q.astype(np.intp)
    return dl * IQ4[q], np.abs(dl) * IQ4_GAP[q]


def _iq4_nl(b):
    return _iq4(_f16(b, 0)[:, None], np.concatenate([b[:, 2:18] & 0xF, b[:, 2:18] >> 4], axis=1))


def _iq4_xs(b):
    d = _f16(b, 0)[:, None, None]
    sh = b[:, 2:4].copy().view("<u2").astype(np.int32)  # (blocks, 1)
    ib = np.arange(8)
    ls = ((b[:, 4 + ib // 2] >> (4 * (ib % 2))) & 0xF).astype(np.int32) | (((sh >> (2 * ib)) & 3) << 4)
    qs = b[:, 8:136].reshape(len(b), 8, 16)
    v, s = _iq4(d * (ls - 32).astype(np.float32)[:, :, None], np.concatenate([qs & 0xF, qs >> 4], axis=2))
    return v.reshape(len(b), 256), s.reshape(len(b), 256)


def _floats(raw: np.ndarray, kind: str):
    if kind == "F32":
        v = raw.copy().view("<f4").ravel()
        return v, np.zeros_like(v)
    if kind == "F16":
        v = raw.copy().view("<f2").ravel().astype(np.float32)
        return v, np.maximum(np.abs(v), 2.0 ** -14) * 2.0 ** -10
    v = (raw.copy().view("<u2").ravel().astype(np.uint32) << 16).view(np.float32)
    return v, np.abs(v) * 2.0 ** -7


DEQUANT = {"Q4_0": _q4_0, "Q4_1": _q4_1, "Q5_0": _q5_0, "Q5_1": _q5_1, "Q8_0": _q8_0, "Q2_K": _q2_k, "Q3_K": _q3_k,
           "Q4_K": lambda b: _q45_k(b, False), "Q5_K": lambda b: _q45_k(b, True), "Q6_K": _q6_k,
           "IQ4_NL": _iq4_nl, "IQ4_XS": _iq4_xs}


def dequantize(type_id: int, raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Values and the rounding step of each, from whole blocks of one type."""
    kind, per, size = TYPES[type_id]
    blocks = np.frombuffer(raw, np.uint8)[:len(raw) // size * size].reshape(-1, size)
    if per == 1:
        return _floats(blocks, kind)
    v, s = DEQUANT[kind](blocks)
    return np.asarray(v, np.float32).ravel(), np.asarray(s, np.float32).ravel()


# ---- a model's GGUF weights ------------------------------------------------------------------------------------

FLOATS = {"F32", "F16", "BF16"}


def described(quant: str) -> tuple[str, str]:
    """('an F16 copy', 'converted to F16') or ('a Q8_0 quantization', 'within Q8_0 rounding'): a quantization
    can hide a change smaller than its rounding."""
    a = "an" if quant[0] in "AEFHILMNORSX" else "a"
    return ((f"{a} {quant} copy", f"converted to {quant}") if quant in FLOATS else
            (f"{a} {quant} quantization", f"within {quant} rounding"))


class Weights:
    """The fingerprint tensors of a model's GGUF file(s), read through a source's read(spans)."""

    def __init__(self, src, files: list[str], sizes: dict[str, int]):
        self.src, self.files = src, files
        self.tensors: dict[str, tuple[str, dict]] = {}  # canonical name: (file, info)
        self.meta: dict = {}
        types: dict[int, int] = {}
        for f in files:
            h = header(lambda at, n, f=f: src.read([(f, at, n)])[0], sizes[f])
            self.meta = self.meta or h["meta"]
            for name, info in h["tensors"].items():
                types[info["type"]] = types.get(info["type"], 0) + 1
                c = canonical(name)
                if c:
                    self.tensors[c] = (f, info)
        self.arch = self.meta.get("general.architecture", "")
        self.quant = label(files, types)
        self.heads = int(self.meta.get(f"{self.arch}.attention.head_count", 0) or 0)

    def shape(self, name: str) -> list[int]:
        return list(reversed(self.tensors[name][1]["dims"]))

    def readable(self, name: str) -> bool:
        return self.tensors[name][1]["type"] in TYPES

    def _flat(self, name: str, idx: np.ndarray) -> np.ndarray:
        """Hugging Face flat positions to the GGUF file's, undoing the Llama query permutation."""
        if self.arch not in PERMUTED or not name.endswith("q_proj.weight") or not self.heads:
            return idx
        rows, cols = self.shape(name)
        hd = rows // self.heads
        r, c = idx // cols, idx % cols
        h, rem = r // hd, r % hd
        return (h * hd + 2 * (rem % (hd // 2)) + rem // (hd // 2)) * cols + c

    def gather(self, wanted: list[tuple[str, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray]]:
        """For each (tensor, Hugging Face flat positions): the values there and their rounding steps, every needed
        block read once, neighbouring blocks in one range."""
        spans, plan = [], []
        for name, idx in wanted:
            f, info = self.tensors[name]
            _, per, size = TYPES[info["type"]]
            flat = self._flat(name, np.asarray(idx, np.int64))
            blocks = np.unique(flat // per)
            runs = np.split(blocks, np.flatnonzero(np.diff(blocks) != 1) + 1)
            plan.append((info["type"], per, flat, [(r[0], len(r)) for r in runs], len(spans)))
            spans += [(f, info["at"] + int(r[0]) * size, len(r) * size) for r in runs]
        raws = self.src.read(spans)
        out = []
        for t, per, flat, runs, first in plan:
            firsts, vals, steps = [], [], []
            for (b0, n), raw in zip(runs, raws[first:first + len(runs)]):
                v, s = dequantize(t, raw)
                firsts.append(b0 * per)
                vals.append(v)
                steps.append(s)
            starts = np.repeat(np.asarray(firsts), [len(v) for v in vals])
            pos = np.concatenate([np.arange(len(v)) for v in vals]) + starts
            v, s = np.concatenate(vals), np.concatenate(steps)
            at = np.searchsorted(pos, flat)
            out.append((v[at], s[at]))
        return out
