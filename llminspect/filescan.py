"""Static checks of model files for code that runs when they load. Nothing here loads a model or
unpickles anything: files are read as bytes and parsed.

  pickle      .bin .pt .pth .ckpt .pkl .joblib, numpy object arrays (.npy, .npz), and the pickles inside
              PyTorch zip saves and tar archives (NeMo's .nemo). Every Python function or class the pickle would import when loaded is
              listed. Weight files normally import only tensor-rebuilding helpers (STANDARD); an import
              of process, network, file, eval-style or loader functions (RISKY) is code that runs on load,
              and any other import is code from that library that runs on load.
  code        a config.json auto_map means transformers runs the repo's own .py files when a user opts in
              with trust_remote_code; those files' risky imports and calls are listed, with their lines.
              A Python wheel shipped with the weights is code that runs if it is installed.
  template    chat templates (tokenizer_config.json, GGUF metadata) are Jinja run at every chat; one that
              reaches Python internals (__globals__, __subclasses__, ...) can escape the sandbox.
  keras       Lambda layers in .keras / .h5 files hold Python that runs when the model is built.
  format      safetensors headers whose offsets don't add up, and GGUF files that can't be parsed.

A finding is a fact about a file, with the evidence; a file with no finding only passed these checks.
None of this says a model is safe, and a risky import isn't proof of intent: some libraries pickle
odd things. Weight-level tampering (a hidden trigger, removed refusals) is the deploy scan's job.

    python -m llminspect.filescan ROOT [ROOT ...] [--out DIR]
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import pickletools
import re
import struct
import sys
import tarfile
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

PICKLE_SUFFIXES = {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".joblib"}
CONFIGS = {"config.json", "tokenizer_config.json", "chat_template.json", "processor_config.json"}
WEIGHT_SUFFIXES = {".safetensors", ".gguf", ".onnx"} | PICKLE_SUFFIXES | {".npy", ".npz", ".keras", ".h5", ".hdf5"}

# What a weight pickle normally imports: tensor and storage rebuilding, containers, numpy arrays.
STANDARD = {
    "collections": {"OrderedDict", "defaultdict", "deque", "Counter"},
    "builtins": {"set", "frozenset", "dict", "list", "tuple", "bytearray", "bytes", "slice", "complex",
                 "float", "int", "bool", "str", "range", "object"},
    "torch._utils": {"_rebuild_tensor", "_rebuild_tensor_v2", "_rebuild_tensor_v3", "_rebuild_parameter",
                     "_rebuild_parameter_with_state", "_rebuild_qtensor", "_rebuild_sparse_tensor",
                     "_rebuild_device_tensor_from_numpy", "_rebuild_meta_tensor_no_storage",
                     "_rebuild_wrapper_subclass", "_rebuild_nested_tensor", "_rebuild_sparse_csr_tensor"},
    "torch._tensor": {"_rebuild_from_type_v2", "_rebuild_from_type"},
    "torch": {"Size", "device", "dtype", "float32", "float16", "bfloat16", "float64", "int64", "int32", "int16",
              "int8", "uint8", "bool", "complex64", "float8_e4m3fn", "float8_e5m2", "per_tensor_affine",
              "per_channel_affine", "strided", "sparse_coo", "Tensor", "UntypedStorage", "TypedStorage",
              "ByteStorage", "CharStorage", "ShortStorage", "IntStorage", "LongStorage", "HalfStorage",
              "FloatStorage", "DoubleStorage", "BFloat16Storage", "BoolStorage", "ComplexFloatStorage"},
    "torch.nn.parameter": {"Parameter"},
    "torch.serialization": {"_get_layout"},
    "numpy": {"ndarray", "dtype"},
    "numpy.core.multiarray": {"_reconstruct", "scalar"},
    "numpy._core.multiarray": {"_reconstruct", "scalar"},
    "numpy.dtypes": None,  # every dtype class
    "_codecs": {"encode"},  # how numpy pickles raw bytes
}

# Imports that run commands, touch the network or files, evaluate code, or load another file.
RISKY_MODULES = {"os", "posix", "nt", "subprocess", "socket", "shutil", "sys", "runpy", "importlib", "ctypes", "pty",
                 "commands", "webbrowser", "requests", "urllib", "urllib2", "urllib3", "http", "httplib", "ftplib",
                 "smtplib", "telnetlib", "asyncio", "multiprocessing", "code", "codeop", "pdb", "marshal", "pickle",
                 "_pickle", "dill", "cloudpickle", "joblib", "types", "io", "_io", "tempfile", "glob", "signal",
                 "threading", "_thread", "aiohttp", "httpx", "paramiko", "platform", "pip", "setuptools"}
RISKY_NAMES = {"builtins": {"eval", "exec", "compile", "open", "__import__", "getattr", "setattr", "delattr",
                            "globals", "locals", "vars", "breakpoint", "input", "help", "memoryview", "apply",
                            "execfile", "file", "reload"},
               "operator": {"attrgetter", "methodcaller", "itemgetter"},
               "functools": {"partial", "reduce"}, "torch": {"load", "hub"}, "torch.serialization": {"load"},
               "torch.hub": None, "torch.storage": {"_load_from_bytes"}, "numpy": {"load", "loadtxt", "fromfile"},
               "numpy.testing._private.utils": None}

TEMPLATE_GADGETS = re.compile(r"__(globals|builtins|subclasses|mro|class|base|bases|init|import|dict|code|getattribute"
                              r"|reduce|reduce_ex)__|\b(os\.|subprocess|popen|eval\s*\(|exec\s*\()", re.IGNORECASE)
# In shipped .py files: imports of process, network, native-code and deserialization modules, bare calls
# that evaluate code, and os/shutil calls that run commands or delete trees. (`os` for paths is everywhere.)
CODE_MODULES = {"subprocess", "socket", "ctypes", "requests", "urllib", "urllib3", "http", "httpx", "aiohttp",
                "paramiko", "ftplib", "smtplib", "telnetlib", "marshal", "pickle", "dill", "cloudpickle",
                "webbrowser", "pty"}
BARE_CALLS = {"eval", "exec", "compile", "__import__"}
MODULE_CALLS = {"os": {"system", "popen", "execv", "execve", "execvp", "execl", "spawnv", "spawnl", "startfile"},
                "shutil": {"rmtree"}, "subprocess": None, "socket": None, "requests": None, "urllib": None,
                "httpx": None, "marshal": {"loads"}, "pickle": {"loads", "load"}, "ctypes": None}

LEVELS = ("runs code on load", "imports other code on load", "runs code if trusted", "template", "format", "info")


@dataclass
class Finding:
    level: str      # one of LEVELS, most serious first
    check: str      # pickle, code, template, keras, format
    file: str
    detail: str
    evidence: list = field(default_factory=list)


# ---- pickle ----------------------------------------------------------------------------------------------

PY2_NAMES = {"__builtin__": "builtins", "copy_reg": "copyreg"}  # how protocol 0-2 pickles name them
_STRING_OPS = {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE", "SHORT_BINSTRING", "BINSTRING", "STRING"}


def pickle_imports(stream: io.BufferedIOBase, max_pickles: int = 8) -> tuple[list[tuple[str, str]], list[str]]:
    """Every (module, name) the pickles in a stream import, read from their opcodes without running them, and
    problems met (an import whose name is computed, a stream that stops parsing). Reads consecutive pickles,
    as legacy PyTorch files hold several, and stops at the first bytes that aren't one."""
    found, problems = [], []
    for k in range(max_pickles):
        pushed, memo, mine, stopped = [], {}, [], False
        try:
            for op, arg, _ in pickletools.genops(stream):
                name = op.name
                if name in ("GLOBAL", "INST"):
                    module, _, attr = str(arg).partition(" ")
                    mine.append((module, attr))
                    pushed.append(None)
                elif name == "STACK_GLOBAL":
                    module, attr = (pushed[-2] if len(pushed) > 1 else None), (pushed[-1] if pushed else None)
                    mine.append((module, attr) if isinstance(module, str) and isinstance(attr, str) else None)
                    pushed.append(None)
                elif name in _STRING_OPS:
                    pushed.append(arg)
                elif name == "MEMOIZE":
                    memo[len(memo)] = pushed[-1] if pushed else None
                elif name in ("PUT", "BINPUT", "LONG_BINPUT"):
                    memo[arg] = pushed[-1] if pushed else None
                elif name in ("GET", "BINGET", "LONG_BINGET"):
                    pushed.append(memo.get(arg))
                elif name == "STOP":
                    stopped = True
                    break
                else:
                    pushed.append(None)
        except Exception as exc:  # not (or no longer) a pickle: legacy files go on with raw tensor bytes
            if k == 0:
                problems.append(f"not a readable pickle ({type(exc).__name__})")
        if not stopped:  # bytes after the last pickle can parse as a few stray opcodes: only complete pickles count
            break
        found += [(PY2_NAMES.get(m[0], m[0]), m[1]) for m in mine if m is not None]
        if None in mine:
            problems.append("an import whose module or name is computed at load time")
    return found, problems


def classify(module: str, name: str) -> str:
    """'standard', 'risky' or 'other' for one import."""
    allowed = STANDARD.get(module, set())
    if allowed is None or name in (allowed or ()):
        return "standard"
    root = module.split(".")[0]
    risky = RISKY_NAMES.get(module, set())
    if root in RISKY_MODULES or risky is None or name in (risky or ()):
        return "risky"
    if module == "builtins":
        return "risky"  # any other builtin is unusual enough to call out
    return "other"


def _pickle_findings(rel: str, imports: list[tuple[str, str]], problems: list[str]) -> list[Finding]:
    out = []
    seen = sorted(set(imports))
    risky = [f"{m}.{n}" for m, n in seen if classify(m, n) == "risky"]
    other = [f"{m}.{n}" for m, n in seen if classify(m, n) == "other"]
    if risky:
        out.append(Finding("runs code on load", "pickle", rel,
                           "Loading this pickle (pickle.load, or torch.load with weights_only=False) calls these "
                           "built-in or system functions, which no weight file needs: they are where a command, "
                           "a download or an eval would go.", risky))
    if other:
        out.append(Finding("imports other code on load", "pickle", rel,
                           "Loading this pickle imports and runs code from these libraries (not tensor "
                           "rebuilding); often benign training state, but it is code.", other))
    for p in sorted(set(problems)):
        level = "runs code on load" if "computed" in p else "format"
        out.append(Finding(level, "pickle", rel, f"Pickle: {p}."))
    return out


def scan_pickle(path: Path, rel: str) -> tuple[list[Finding], dict]:
    """A pickle-based file: a PyTorch zip (its .pkl members), a numpy file, or a raw pickle stream."""
    info = {"format": "pickle"}
    imports, problems = [], []
    if zipfile.is_zipfile(path):
        info["format"] = "pytorch zip"
        with zipfile.ZipFile(path) as z:
            members = [m for m in z.namelist() if m.endswith(".pkl")]
            info["pickles"] = len(members)
            for m in members:
                with z.open(m) as fh:
                    found, probs = pickle_imports(io.BufferedReader(fh))
                    imports += found
                    problems += probs
    else:
        with open(path, "rb") as fh:
            head = fh.read(2)
            fh.seek(0)
            if head[:1] != b"\x80" and head[:1] not in (b"(", b"}", b"]", b"c"):
                info["format"] = "not a pickle"
                if path.suffix.lower() in (".pkl", ".pickle", ".joblib"):  # compressed (joblib) or another format
                    return [Finding("format", "pickle", rel, "Not a plain pickle (compressed, or another format), "
                                    "so it wasn't read; loading it may still unpickle it.")], info
                return [], info
            found, problems = pickle_imports(fh)
            imports += found
    info["imports"] = sorted({f"{m}.{n}" for m, n in imports})
    return _pickle_findings(rel, imports, problems), info


def _npy(fh, rel: str) -> tuple[list[Finding], dict]:
    """A .npy header; an object array ('O' dtype) is a pickle after the header."""
    magic = fh.read(6)
    if magic != b"\x93NUMPY":
        return [Finding("format", "pickle", rel, "Not a .npy file.")], {"format": "npy?"}
    major = fh.read(2)[0]
    n = struct.unpack("<H" if major == 1 else "<I", fh.read(2 if major == 1 else 4))[0]
    header = fh.read(n).decode("latin1")
    if "'O'" not in header and '"O"' not in header and "|O" not in header:
        return [], {"format": "npy"}
    found, problems = pickle_imports(fh)
    return (_pickle_findings(rel, found, problems) or [Finding("info", "pickle", rel, "Object array: stored as a "
                                                                "pickle; it imports only standard types.")],
            {"format": "npy (object array)", "imports": [f"{m}:{a}" for m, a in found], "problems": problems})


def scan_numpy(path: Path, rel: str) -> tuple[list[Finding], dict]:
    if path.suffix.lower() == ".npz":
        out = []
        with zipfile.ZipFile(path) as z:
            for m in z.namelist():
                if m.endswith(".npy"):
                    with z.open(m) as fh:
                        out += _npy(io.BufferedReader(fh), f"{rel}:{m}")[0]
        return out, {"format": "npz"}
    with open(path, "rb") as fh:
        return _npy(fh, rel)


# ---- safetensors, GGUF, keras --------------------------------------------------------------------------

def scan_safetensors(path: Path, rel: str) -> tuple[list[Finding], dict]:
    size = path.stat().st_size
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) < 8:
            return [Finding("format", "format", rel, "Too short to be a safetensors file.")], {"format": "safetensors"}
        n = struct.unpack("<Q", raw)[0]
        if n > min(size - 8, 1 << 28):
            return [Finding("format", "format", rel, f"Header length {n} doesn't fit the file.")], {"format": "safetensors"}
        try:
            header = json.loads(fh.read(n))
        except ValueError:
            return [Finding("format", "format", rel, "Header isn't valid JSON.")], {"format": "safetensors"}
    spans = sorted(tuple(v["data_offsets"]) for k, v in header.items() if k != "__metadata__")
    end, problems = 0, []
    for a, b in spans:
        if a != end or b < a:
            problems.append(f"tensor data at {a}-{b} doesn't follow the previous tensor's end ({end})")
            break
        end = b
    if end + 8 + n != size and not problems:
        problems.append(f"{size - (end + 8 + n)} bytes after the last tensor")
    meta = header.get("__metadata__") or {}
    info = {"format": "safetensors", "tensors": len(spans), "metadata_keys": sorted(meta)[:20]}
    return [Finding("format", "format", rel, f"Safetensors layout: {p}.") for p in problems], info


_GGUF_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def gguf_strings(path: Path, wanted=lambda k: True) -> dict[str, str]:
    """String metadata of a GGUF file whose key passes `wanted`, read from the header alone."""
    out = {}
    with open(path, "rb") as fh:
        if fh.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version = struct.unpack("<I", fh.read(4))[0]
        if version < 2:
            raise ValueError(f"GGUF version {version} isn't read")
        _, n_kv = struct.unpack("<QQ", fh.read(16))

        def string() -> bytes:
            return fh.read(struct.unpack("<Q", fh.read(8))[0])

        def skip(t: int):
            if t == 8:
                fh.seek(struct.unpack("<Q", fh.read(8))[0], 1)
            elif t == 9:
                inner, count = struct.unpack("<IQ", fh.read(12))
                if inner in _GGUF_SIZES:
                    fh.seek(_GGUF_SIZES[inner] * count, 1)
                else:
                    for _ in range(count):
                        skip(inner)
            else:
                fh.seek(_GGUF_SIZES[t], 1)

        for _ in range(n_kv):
            key = string().decode("utf-8", "replace")
            t = struct.unpack("<I", fh.read(4))[0]
            if t == 8 and wanted(key):
                out[key] = string().decode("utf-8", "replace")
            else:
                skip(t)
    return out


def _template_findings(rel: str, where: str, text: str) -> list[Finding]:
    hits = sorted({m.group(0) for m in TEMPLATE_GADGETS.finditer(text)})
    if not hits:
        return []
    return [Finding("template", "template", rel, f"The chat template ({where}) reaches Python internals or "
                    "commands, which a Jinja sandbox escape needs.", hits)]


def scan_gguf(path: Path, rel: str) -> tuple[list[Finding], dict]:
    try:
        meta = gguf_strings(path, lambda k: k.startswith("tokenizer.chat_template") or k in ("general.architecture", "general.name"))
    except (ValueError, OSError, struct.error, KeyError) as exc:
        return [Finding("format", "format", rel, f"GGUF header can't be read: {exc}.")], {"format": "gguf"}
    out = []
    for k, v in meta.items():
        if k.startswith("tokenizer.chat_template"):
            out += _template_findings(rel, k, v)
    return out, {"format": "gguf", "architecture": meta.get("general.architecture"),
                 "chat_templates": sum(k.startswith("tokenizer.chat_template") for k in meta)}


def _bytes_contain(path: Path, needles: list[bytes], chunk: int = 1 << 24) -> set[bytes]:
    hits, tail = set(), b""
    keep = max(len(n) for n in needles)
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            data = tail + block
            hits |= {n for n in needles if n in data}
            tail = data[-keep:]
    return hits


def scan_keras(path: Path, rel: str) -> tuple[list[Finding], dict]:
    needles = [b'"class_name": "Lambda"', b'"class_name":"Lambda"', b"'class_name': 'Lambda'"]
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            text = b"".join(z.read(m) for m in z.namelist() if m.endswith(".json"))
        hit = any(n in text for n in needles)
    else:
        hit = bool(_bytes_contain(path, needles))
    out = [Finding("runs code on load", "keras", rel, "A Lambda layer: Python code that runs when the model is "
                   "built.")] if hit else []
    return out, {"format": "keras"}


# ---- code and configs ----------------------------------------------------------------------------------

def scan_python(path: Path, rel: str) -> tuple[list[Finding], dict]:
    """Imports and calls in a .py file that run commands, reach the network or files, or evaluate code."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return [Finding("format", "code", rel, "Python file doesn't parse.")], {"format": "python"}
    seen: dict[str, list[int]] = {}  # "import x" / "x.call" -> the lines it is on
    for node in ast.walk(tree):
        hits = []
        if isinstance(node, ast.Import):
            hits = [f"import {a.name}" for a in node.names if a.name.split(".")[0] in CODE_MODULES]
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in CODE_MODULES:
            hits = [f"import {node.module}.{a.name}" for a in node.names]
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in BARE_CALLS:  # not model.eval(): a method of that name
                hits = [f"{f.id}()"]
            elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in MODULE_CALLS:
                names = MODULE_CALLS[f.value.id]
                if names is None or f.attr in names:
                    hits = [f"{f.value.id}.{f.attr}()"]
        for h in hits:
            seen.setdefault(h, []).append(node.lineno)
    lines = lambda ls: ", ".join(map(str, sorted(ls)[:6])) + (f" (+{len(ls) - 6})" if len(ls) > 6 else "")
    risky = [f"{rel}: {h} at line{'s' if len(ls) > 1 else ''} {lines(ls)}" for h, ls in sorted(seen.items())]
    return [], {"format": "python", "risky": risky}


def scan_jinja(path: Path, rel: str) -> tuple[list[Finding], dict]:
    return _template_findings(rel, path.name, path.read_text(encoding="utf-8", errors="replace")), {"format": "template"}


def scan_config(path: Path, rel: str) -> tuple[list[Finding], dict]:
    if path.name not in CONFIGS:
        return [], {"format": "json"}
    try:
        cfg = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except ValueError:
        return [], {"format": "json"}
    if not isinstance(cfg, dict):
        return [], {"format": "json"}
    out, info = [], {"format": "json"}
    if cfg.get("auto_map"):
        info["auto_map"] = cfg["auto_map"]
    template = cfg.get("chat_template")
    for t in (template if isinstance(template, list) else [{"template": template}] if template else []):
        if isinstance(t, dict) and isinstance(t.get("template"), str):
            out += _template_findings(rel, f"{path.name} {t.get('name') or ''}".strip(), t["template"])
    return out, info


def scan_wheel(path: Path, rel: str) -> tuple[list[Finding], dict]:
    return [Finding("runs code if trusted", "code", rel, "A Python package (wheel) shipped with the weights: its "
                    "code runs if it is installed.")], {"format": "wheel"}


def scan_archive(path: Path, rel: str) -> tuple[list[Finding], dict]:
    """A tar archive (NeMo's .nemo is one): the pickles inside it, read without extracting the archive."""
    out, n = [], 0
    with tarfile.open(path) as tar:
        for m in tar:
            if not m.isfile() or Path(m.name).suffix.lower() not in PICKLE_SUFFIXES:
                continue
            n += 1
            fh = tar.extractfile(m)
            imports, problems = [], []
            if zipfile.is_zipfile(fh):
                fh.seek(0)
                with zipfile.ZipFile(fh) as z:
                    for name in (x for x in z.namelist() if x.endswith(".pkl")):
                        with z.open(name) as inner:
                            found, probs = pickle_imports(io.BufferedReader(inner))
                            imports += found
                            problems += probs
            else:
                fh.seek(0)
                imports, problems = pickle_imports(fh)
            out += _pickle_findings(f"{rel}:{m.name}", imports, problems)
    return out, {"format": "archive", "pickles": n}


SCANNERS = {**{s: scan_pickle for s in PICKLE_SUFFIXES}, ".npy": scan_numpy, ".npz": scan_numpy,
            ".safetensors": scan_safetensors, ".gguf": scan_gguf, ".keras": scan_keras, ".h5": scan_keras,
            ".hdf5": scan_keras, ".py": scan_python, ".json": scan_config, ".jinja": scan_jinja,
            ".whl": scan_wheel, ".nemo": scan_archive, ".tar": scan_archive}


def scan_file(path: Path, rel: str) -> tuple[list[Finding], dict]:
    scanner = SCANNERS.get(path.suffix.lower())
    if scanner is None:
        return [], {"format": "other"}
    try:
        return scanner(path, rel)
    except (OSError, zipfile.BadZipFile, tarfile.TarError, ValueError, struct.error, EOFError, IndexError) as exc:
        return [Finding("format", "format", rel, f"Couldn't be read: {type(exc).__name__}: {exc}.")], {"format": "unreadable"}


# ---- repositories --------------------------------------------------------------------------------------

def scan_repo(folder: Path, name: str | None = None) -> dict:
    """Every file under a model folder; findings, and a summary of what loading it would do."""
    t0 = time.perf_counter()
    files, findings = [], []
    for path in sorted(p for p in folder.rglob("*") if p.is_file() and not p.name.startswith(".")):
        rel = path.relative_to(folder).as_posix()
        try:
            size = path.stat().st_size
        except OSError:
            continue
        f, info = scan_file(path, rel)
        findings += f
        files.append({"file": rel, "bytes": size, **info})
    code = [f for f in files if f["format"] == "python"]
    automap = next((f["auto_map"] for f in files if f.get("auto_map")), None)
    if automap:
        risky = [x for f in code for x in f["risky"]]
        findings.append(Finding("runs code if trusted", "code", "config.json",
                                "auto_map: transformers runs this repo's own Python when it is loaded with "
                                "trust_remote_code." + (" Its .py files import or call:" if risky else ""),
                                risky))
    elif code:
        findings.append(Finding("info", "code", ", ".join(f["file"] for f in code[:5]),
                                "Python files ship with the weights; nothing here runs them on load."))
    weights = sorted({f["format"] for f in files if Path(f["file"]).suffix.lower() in WEIGHT_SUFFIXES})
    order = {lvl: i for i, lvl in enumerate(LEVELS)}
    findings.sort(key=lambda f: (order[f.level], f.file))
    return {"name": name or folder.name, "path": str(folder), "files": len(files),
            "bytes": sum(f["bytes"] for f in files), "weight_formats": weights,
            "top": findings[0].level if findings and findings[0].level != "info" else None,
            "findings": [asdict(f) for f in findings], "file_list": files,
            "seconds": round(time.perf_counter() - t0, 2)}


def _is_model_dir(p: Path) -> bool:
    try:
        return any(c.is_file() and c.suffix.lower() in WEIGHT_SUFFIXES for c in p.iterdir())
    except OSError:
        return False


def find_repos(roots: list[Path]) -> list[tuple[str, Path]]:
    """(name, folder) for each model under the roots: Hugging Face cache snapshots (the newest revision),
    folders holding weight files (a folder inside one counts as part of it), and loose weight files' folders."""
    out, seen = [], set()
    for root in roots:
        root = Path(root)
        hub = [root] if root.name.startswith("models--") else sorted(root.glob("models--*"))
        for repo in hub:
            snaps = sorted((repo / "snapshots").glob("*"), key=lambda s: s.stat().st_mtime) if (repo / "snapshots").is_dir() else []
            if snaps:
                out.append((repo.name[len("models--"):].replace("--", "/"), snaps[-1]))
                seen.add(repo.resolve())
        for dirpath, dirnames, _ in os.walk(root):
            d = Path(dirpath)
            if d.resolve() in seen or any(part.startswith("models--") for part in d.parts) or d.name.startswith("."):
                dirnames[:] = []
                continue
            if _is_model_dir(d):
                out.append((d.relative_to(root).as_posix() if d != root else d.name, d))
                dirnames[:] = []  # its subfolders are part of this model
    return out


def scan(roots: list[Path], say=print) -> dict:
    repos = find_repos(roots)
    say(f"[filescan] {len(repos)} models under {', '.join(str(r) for r in roots)}")
    results = []
    for i, (name, folder) in enumerate(repos, 1):
        r = scan_repo(folder, name)
        results.append(r)
        mark = r["top"] or "nothing found"
        say(f"[filescan] {i}/{len(repos)} {name}: {r['files']} files, {r['bytes'] / 1e9:.1f} GB, {mark}")
    counts = {lvl: sum(1 for r in results if any(f["level"] == lvl for f in r["findings"])) for lvl in LEVELS}
    return {"scanned_at": time.strftime("%Y-%m-%d %H:%M:%S"), "roots": [str(r) for r in roots], "models": len(results),
            "bytes": sum(r["bytes"] for r in results), "counts": counts, "repos": results}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("filescan-report"))
    args = ap.parse_args(argv)
    report = scan(args.roots)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"[filescan] {report['models']} models, {report['bytes'] / 1e12:.2f} TB: "
          + ", ".join(f"{n} {lvl}" for lvl, n in report["counts"].items() if n) + f" -> {args.out / 'report.json'}")


if __name__ == "__main__":
    main(sys.argv[1:])
