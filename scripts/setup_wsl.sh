#!/usr/bin/env bash
# One-time environment setup inside WSL2. Creates ~/.venvs/brainscan with a
# CUDA build of torch plus the recorder's dependencies, then prints versions.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

VENV="$HOME/.venvs/brainscan"
[ -d "$VENV" ] || uv venv "$VENV" --python 3.12
source "$VENV/bin/activate"

uv pip install torch --torch-backend=auto
uv pip install -r "$(dirname "$0")/../requirements.txt"

python - <<'PY'
import torch, transformers, fastapi
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__, "fastapi", fastapi.__version__)
PY
uv pip freeze > "$(dirname "$0")/../requirements.lock"
