"""Shared test setup: caches that would otherwise live in the user's home folder go to a temporary one."""

import os
import tempfile
from pathlib import Path

# Set before any test imports llminspect, and inherited by servers the tests start (the MCP server).
_tmp = Path(tempfile.mkdtemp(prefix="llminspect-tests-"))
os.environ.setdefault("LLMINSPECT_FINGERPRINTS", str(_tmp / "fingerprints.json"))
os.environ.setdefault("LLMINSPECT_CHECKS", str(_tmp / "checks"))
