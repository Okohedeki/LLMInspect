"""End-to-end check of the workbench API against a running recorder (any model).

    scripts/run.sh scripts/smoke_workbench.py [base_url]

Builds a direction from a layer sweep, stacks edits (direction removal, a random-direction
control, a head lesion), applies them, records a chat turn, compares original vs edited,
exports the variant, and restores the original weights. Leaves a direction named
'smoke_language' in the bank; the API removes it at the end.
"""

import json
import shutil
import sys
import tempfile
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8766"


def call(method: str, path: str, body: dict | None = None):
    req = urllib.request.Request(BASE + path, method=method, headers={"Content-Type": "application/json"},
                                 data=None if body is None else json.dumps(body).encode())
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.load(resp)


english = ["Where is the train station?", "The weather is lovely today.", "Please close the window.",
           "I would like a cup of coffee.", "How much does this cost?", "My brother lives in the city."]
french = ["Où est la gare ?", "Il fait très beau aujourd'hui.", "Fermez la fenêtre, s'il vous plaît.",
          "Je voudrais une tasse de café.", "Combien ça coûte ?", "Mon frère habite en ville."]

status = call("GET", "/llminspect/status")
n_layers = status["model"]["n_layers"]
print("model:", status["model"]["name"], n_layers, "layers")

sweep = call("POST", "/llminspect/directions/analyze", {"positive": french, "negative": english,
                                                        "position": "mean", "chat": False})
print("sweep: best layer", sweep["best"], "held_out", sweep["held_out"],
      "d by layer", [x["d"] for x in sweep["layers"]])
call("POST", "/llminspect/directions/save", {"id": sweep["id"], "layer": sweep["best"], "name": "smoke_language",
                                            "description": "French vs English sentences (smoke test)"})

e1 = call("POST", "/llminspect/workbench/edits", {"kind": "orthogonalize", "direction": "smoke_language",
                                                 "layers": "all", "targets": ["embed", "attn_out", "mlp_out"]})
e2 = call("POST", "/llminspect/workbench/edits", {"kind": "orthogonalize", "direction": "random", "seed": 3,
                                                 "layers": "all", "targets": ["attn_out"], "strength": 0.5})
e3 = call("POST", "/llminspect/workbench/edits", {"kind": "scale", "layers": "1", "targets": ["head"],
                                                 "index": 0, "factor": 0})
print("edits:", [e["summary"] for e in (e1, e2, e3)])
applied = call("POST", "/llminspect/workbench/apply")
print("applied:", len(applied["applied"]), "edits in", applied["seconds"], "s; snapshot",
      round(applied["snapshot_bytes"] / 1e6, 1), "MB")

chat = call("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Bonjour !"}],
                                             "max_tokens": 8, "temperature": 0})
run = call("GET", f"/llminspect/runs/{chat['llminspect']['run']}")
print("recorded run notes weight edits:", len(run["meta"].get("weight_edits", [])) == 1)

cmp_ = call("POST", "/llminspect/workbench/compare", {"prompts": ["Say hello.", "Name a colour."], "max_tokens": 8})
print("compare metrics:", {k: round(v, 4) if isinstance(v, float) else v for k, v in cmp_["metrics"].items()})

out = tempfile.mkdtemp(prefix="llminspect-export-") + "/variant"
exp = call("POST", "/llminspect/workbench/export", {"path": out})
print("exported", round(exp["bytes"] / 1e6, 1), "MB with", exp["edits"], "edits ->", exp["path"])
shutil.rmtree(out.rsplit("/", 1)[0], ignore_errors=True)

reverted = call("POST", "/llminspect/workbench/revert")
for e in (e1, e2, e3):
    call("DELETE", f"/llminspect/workbench/edits/{e['id']}")
call("DELETE", "/llminspect/directions/smoke_language")
print("reverted; edits left:", len(call("GET", "/llminspect/workbench")["edits"]), "| applied:", reverted["applied"])
print("SMOKE OK")
