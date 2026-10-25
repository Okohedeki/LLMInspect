#!/usr/bin/env bash
# Run a llminspect module inside the WSL venv from the project root.
#   scripts/run.sh -m llminspect.server
#   scripts/run.sh -m llminspect.harness.mini_agent --task fix_fizzbuzz
# Machine-specific defaults (e.g. LLMINSPECT_MODEL) go in llminspect.env, which is not committed.
# Variables already set in the environment win over the file.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f llminspect.env ]; then
  while IFS='=' read -r key value; do
    [[ -z "$key" || "$key" == \#* ]] && continue
    [ -z "${!key+x}" ] && export "$key=$value"
  done < llminspect.env
fi
source "$HOME/.venvs/llminspect/bin/activate"
exec python "$@"
