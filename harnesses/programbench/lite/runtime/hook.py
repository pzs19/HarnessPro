#!/usr/bin/env python3
"""Task-agnostic staged budget hook for ProgramBench Codex runs."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import sys


def _selection() -> dict:
    path = os.environ.get("RHO_RUNTIME_SELECTION_PATH", "").strip()
    if not path:
        return {}
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _next_count(cwd: Path) -> int:
    configured = os.environ.get("RHO_RUNTIME_STATE_ROOT", "").strip()
    state_dir = Path(configured).resolve() if configured else cwd / ".rho_runtime"
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "tool-count.json"
    lock_path = state_dir / "tool-count.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            count = int(value.get("tool_count") or 0) + 1
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            count = 1
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps({"tool_count": count}) + "\n", encoding="utf-8")
        temp.replace(path)
        return count


def _context(event: str, text: str) -> None:
    print(json.dumps({"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}}))


def main() -> int:
    payload = json.load(sys.stdin)
    event = str(payload.get("hook_event_name") or "")
    selection = _selection()
    if event == "SessionStart":
        context = str(selection.get("session_context") or "").strip()
        if context:
            _context("SessionStart", context)
        return 0
    if event != "PreToolUse":
        return 0
    cwd = Path(str(payload.get("cwd") or os.getcwd())).resolve()
    count = _next_count(cwd)
    economy = int(selection.get("economy_after_tools") or 0)
    strict = int(selection.get("strict_after_tools") or 0)
    finalize = int(selection.get("finalize_after_tools") or 0)
    reminder = int(selection.get("strict_reminder_every") or 1)
    hard_stop = finalize + max(8, reminder) if finalize else 0
    if hard_stop and count > hard_stop:
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": (
                            "The exploration budget is exhausted. Preserve the current files, "
                            "finish from existing evidence, and do not call more tools."
                        ),
                    }
                }
            )
        )
        return 0
    if finalize and count >= finalize and (
        count == finalize or (count - finalize) % max(4, reminder) == 0
    ):
        _context(
            "PreToolUse",
            "Finalization stage: stop adding semantic scope. Use the remaining small grace "
            "window only for implementation completion, a clean build, a focused check, "
            "artifact cleanup, or submission. Then finish.",
        )
    elif strict and count >= strict and (count == strict or (count - strict) % reminder == 0):
        _context(
            "PreToolUse",
            "Convergence stage: stop broad exploration. Revisit the public behavior ledger, "
            "use only probes that distinguish a live hypothesis or repair a concrete mismatch, "
            "and preserve enough budget for an offline rebuild and final contract check.",
        )
    elif economy and count == economy:
        _context(
            "PreToolUse",
            "Economy stage: consolidate the behavior ledger, batch related checks, and cover "
            "remaining documented semantic families before low-frequency polish.",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
