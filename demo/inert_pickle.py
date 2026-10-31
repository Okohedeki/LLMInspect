"""Write an inert stand-in for a malicious model file: a PyTorch checkpoint whose pickle, if loaded with
torch.load(weights_only=False) or pickle.load, would call print() with a harmless sentence. A real attack
puts a command there instead; nothing else about the file changes. This script only writes the file,
and the file scan reads it without loading it:

    python demo/inert_pickle.py OUT_DIR
    python -m llminspect.filescan OUT_DIR

Don't load the file with torch.load(weights_only=False) or pickle.load: that is the step an attack relies on.
"""

import json
import sys
from pathlib import Path

import torch

MESSAGE = "This model file just ran code on your machine. Here it only printed this sentence."


class OnLoad:
    """Pickles as the call print(MESSAGE): pickle records the call, it doesn't make it."""

    def __reduce__(self):
        return print, (MESSAGE,)


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    weights = {"linear.weight": torch.randn(4, 4), "linear.bias": torch.zeros(4)}
    torch.save({**weights, "_metadata": OnLoad()}, out / "pytorch_model.bin")
    (out / "config.json").write_text(json.dumps({"model_type": "demo", "note": "inert demo; see demo/inert_pickle.py"}))
    print(f"wrote {out / 'pytorch_model.bin'}; scan it with: python -m llminspect.filescan {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "demo-out/inert-pickle"))
