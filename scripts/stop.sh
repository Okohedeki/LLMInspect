#!/usr/bin/env bash
# Stop a running recorder (the bracket keeps pkill from matching itself). A server stuck inside
# a GPU call ignores SIGTERM, so escalate to SIGKILL after a few seconds.
if ! pkill -f "[l]lminspect.server"; then
  echo "not running"
  exit 0
fi
for _ in 1 2 3 4 5 6; do
  sleep 1
  pgrep -f "[l]lminspect.server" >/dev/null || { echo "stopped"; exit 0; }
done
pkill -9 -f "[l]lminspect.server" && echo "killed (did not stop within 6 s)"
