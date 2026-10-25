"""Remove a direction from the weights, and a random direction the same way, and compare.

    scripts/run.sh scripts/workbench_control_demo.py [base_url] [out.json]

Builds a French-vs-English direction (chat-formatted prompts, last token) with a layer sweep,
saves it, then runs two conditions with identical settings (all layers; embeddings, attention
and MLP outputs; strength 1): the language direction, and a random unit direction. Each is
compared against the original weights on the same prompts. Prints a summary and writes the
full results to JSON. Restores the original weights and removes its edits at the end.
"""

import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765"
OUT = sys.argv[2] if len(sys.argv) > 2 else "workbench_control_demo.json"

FRENCH = ["Où est la gare la plus proche ?", "Il fait très beau aujourd'hui.", "Fermez la fenêtre, s'il vous plaît.",
          "Je voudrais une tasse de café.", "Combien coûte ce livre ?", "Mon frère habite à Lyon.",
          "Nous partons en vacances demain.", "Pouvez-vous m'aider ?", "Le musée ouvre à neuf heures.",
          "J'ai perdu mes clés ce matin."]
ENGLISH = ["Where is the nearest train station?", "The weather is lovely today.", "Please close the window.",
           "I would like a cup of coffee.", "How much does this book cost?", "My brother lives in Leeds.",
           "We are going on holiday tomorrow.", "Can you help me?", "The museum opens at nine o'clock.",
           "I lost my keys this morning."]
PROMPTS = ["Translate into French: 'Good morning, how are you today?'",
           "Réponds en français : quelle est la capitale de l'Italie ?",
           "Write one sentence about the sea.",
           "What is 17 multiplied by 23? Answer with the number only."]


def call(method, path, body=None):
    req = urllib.request.Request(BASE + path, method=method, headers={"Content-Type": "application/json"},
                                 data=None if body is None else json.dumps(body).encode())
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def main() -> None:
    sweep = call("POST", "/brainscan/directions/analyze",
                 {"positive": FRENCH, "negative": ENGLISH, "position": "last", "chat": True})
    best = sweep["best"]
    score = sweep["layers"][best]
    print(f"sweep: best layer {best}, d={score['d']} ± {score.get('d_sd', 0)} over {sweep.get('folds', 1)} folds, "
          f"accuracy {score['acc']:.0%}")
    print("d by layer:", [x["d"] for x in sweep["layers"]])
    call("POST", "/brainscan/directions/save", {"id": sweep["id"], "layer": best, "name": "language_fr",
                                                "description": "French vs English sentences, chat-formatted, last token"})

    results = {"sweep": sweep, "conditions": {}}
    settings = {"kind": "orthogonalize", "layers": "all", "targets": ["embed", "attn_out", "mlp_out"], "strength": 1}
    for name, extra in (("language_fr", {"direction": "language_fr"}), ("random", {"direction": "random", "seed": 11})):
        edit = call("POST", "/brainscan/workbench/edits", {**settings, **extra})
        applied = call("POST", "/brainscan/workbench/apply")
        cmp_ = call("POST", "/brainscan/workbench/compare", {"prompts": PROMPTS, "max_tokens": 48})
        call("POST", "/brainscan/workbench/revert")
        call("DELETE", f"/brainscan/workbench/edits/{edit['id']}")
        results["conditions"][name] = {"edit": edit, "apply_seconds": applied["seconds"], **cmp_}
        m = cmp_["metrics"]
        print(f"\n== remove {name}: KL/token {m['kl_per_token']:.4f}, perplexity {m['perplexity_original']:.2f} -> "
              f"{m['perplexity_edited']:.2f} ({m['seconds']} s)")
        for row in cmp_["rows"]:
            print(f"  Q: {row['prompt']}\n    original: {row['original'][:110]!r}\n    edited:   {row['edited'][:110]!r}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {OUT}; original weights restored")


if __name__ == "__main__":
    main()
