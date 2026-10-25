#!/usr/bin/env bash
# Run a brainscan module inside the WSL venv from the project root.
#   scripts/run.sh -m brainscan.server
#   scripts/run.sh -m brainscan.harness.mini_agent --task fix_fizzbuzz
# Machine-specific defaults (e.g. BRAINSCAN_MODEL) go in brainscan.env, which is not committed.
# Variables already set in the environment win over the file.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f brainscan.env ]; then
  while IFS='=' read -r key value; do
    [[ -z "$key" || "$key" == \#* ]] && continue
    [ -z "${!key+x}" ] && export "$key=$value"
  done < brainscan.env
fi
source "$HOME/.venvs/brainscan/bin/activate"
exec python "$@"
