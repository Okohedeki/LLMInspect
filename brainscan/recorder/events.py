"""Derive time-locked events from the conversation itself.

Any harness that talks to the endpoint sends its whole transcript, so user turns,
tool results, errors and test summaries can be found without harness cooperation.
Harnesses add what the transcript can't show (verdicts, file edits) through
POST /brainscan/events.
"""

from __future__ import annotations

import re

_ERROR = re.compile(
    r"Traceback \(most recent call last\)|\b[A-Z]\w*(?:Error|Exception)\b|\bFAILED\b|\bERROR\b"
    r"|\berror:|exit (?:code|status) [1-9]")
_PASSED = re.compile(r"(\d+) passed")
_FAILED = re.compile(r"(\d+) failed")
_ERRORS = re.compile(r"(\d+) errors?\b")


def scan_tool_output(text: str) -> list[tuple[str, dict]]:
    found = []
    hit = _ERROR.search(text)
    if hit:
        line_start = text.rfind("\n", 0, hit.start()) + 1
        line_end = text.find("\n", hit.end())
        found.append(("error", {"excerpt": text[line_start:line_end if line_end >= 0 else None][:160]}))
    passed, failed, errors = (_PASSED.search(text), _FAILED.search(text), _ERRORS.search(text))
    if passed or failed or errors:
        found.append(("test_result", {
            "passed": int(passed.group(1)) if passed else 0,
            "failed": int(failed.group(1)) if failed else 0,
            "errors": int(errors.group(1)) if errors else 0,
            "source": "parsed",
        }))
    return found


def message_events(msg: dict) -> list[tuple[str, dict]]:
    role, content = msg["role"], msg.get("content") or ""
    if role == "system":
        return [("system_prompt", {"chars": len(content)})]
    if role == "user":
        return [("user_message", {"chars": len(content)})]
    if role == "tool":
        return [("tool_result", {"name": msg.get("name"), "chars": len(content)})] + scan_tool_output(content)
    return []
