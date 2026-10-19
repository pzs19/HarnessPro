#!/usr/bin/env python3
"""Compatibility shim between Claude Code hooks and a RHO runtime hook.

The runtime hook remains the source of all Harness policy decisions.  This
shim only normalizes Claude Code protocol details before delegating to it.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


CORE_HOOK_RAW = os.environ.get("RHO_CLAUDE_CORE_HOOK", "").strip()
CORE_HOOK = Path(CORE_HOOK_RAW) if CORE_HOOK_RAW else None


def main() -> int:
    payload = json.load(sys.stdin)
    event = str(payload.get("hook_event_name") or "")

    # Static Harness evaluations have no policy hook to delegate to.
    if CORE_HOOK is None:
        return 0

    if event == "PostToolUse" and _contains_structured_image(_tool_result(payload)):
        # Claude Code represents image reads as typed image blocks.  Serializing
        # those blocks and applying a text head/tail truncation destroys the
        # image while also retaining a large base64 prefix.  Leave the native
        # typed result untouched; ordinary text results still flow through the
        # Harness compressor below.
        _audit("claude_adapter", action="preserve_structured_image")
        return 0

    sanitized_input: dict[str, Any] | None = None
    if event == "PreToolUse" and payload.get("tool_name") == "Read":
        tool_input = payload.get("tool_input")
        if isinstance(tool_input, dict) and tool_input.get("pages") == "":
            sanitized_input = dict(tool_input)
            sanitized_input.pop("pages", None)
            payload["tool_input"] = sanitized_input
            _audit("claude_adapter", action="remove_empty_read_pages")

    proc = subprocess.run(
        [sys.executable, str(CORE_HOOK)],
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        if proc.stdout:
            sys.stdout.write(proc.stdout)
        return proc.returncode

    output = _parse_hook_output(proc.stdout)
    output = _stop_repeated_policy_denials(output, event=event)
    if sanitized_input is not None:
        specific = output.setdefault("hookSpecificOutput", {})
        if not isinstance(specific, dict):
            specific = {}
            output["hookSpecificOutput"] = specific
        if specific.get("permissionDecision") != "deny":
            specific.setdefault("hookEventName", "PreToolUse")
            specific["updatedInput"] = sanitized_input

    if output:
        print(json.dumps(output, ensure_ascii=False))
    elif proc.stdout:
        sys.stdout.write(proc.stdout)
    return 0


def _stop_repeated_policy_denials(
    output: dict[str, Any], *, event: str
) -> dict[str, Any]:
    """Stop Claude cleanly when it loops on a hard policy denial.

    Some non-Anthropic models retry the identical tool call indefinitely after
    Claude Code reports a denied PreToolUse request.  The policy still owns the
    decision: this adapter never permits the blocked call.  It only converts a
    repeated terminal budget denial into Claude Code's documented session-stop
    response so the current artifacts can be graded instead of eventually
    failing from context exhaustion.
    """

    if event != "PreToolUse" or os.environ.get("RHO_RUNTIME_INTERFACE") != "claude-code":
        return output
    try:
        halt_after = int(os.environ.get("RHO_CLAUDE_POLICY_DENIAL_HALT_AFTER", "0"))
    except ValueError:
        halt_after = 0
    if halt_after <= 0:
        return output
    specific = output.get("hookSpecificOutput")
    if not isinstance(specific, dict) or specific.get("permissionDecision") != "deny":
        return output
    reason = str(specific.get("permissionDecisionReason") or "")
    if "tool budget is exhausted" not in reason.lower():
        return output
    count = _increment_policy_denial_count()
    if count < halt_after:
        return output
    _audit(
        "claude_adapter",
        action="halt_repeated_policy_denials",
        denial_count=count,
        halt_after=halt_after,
    )
    return {
        "continue": False,
        "suppressOutput": False,
        "stopReason": reason,
    }


def _increment_policy_denial_count() -> int:
    root = os.environ.get("RHO_RUNTIME_STATE_ROOT", "").strip()
    if not root:
        return 1
    path = Path(root) / "claude-policy-denials.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        try:
            current = json.load(stream)
        except (json.JSONDecodeError, OSError):
            current = {}
        count = int(current.get("count") or 0) + 1 if isinstance(current, dict) else 1
        stream.seek(0)
        stream.truncate()
        json.dump({"count": count}, stream)
        stream.write("\n")
        stream.flush()
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return count


def _tool_result(payload: dict[str, Any]) -> Any:
    if "tool_result" in payload:
        return payload.get("tool_result")
    return payload.get("tool_response")


def _contains_structured_image(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("type") == "image":
            return True
        source = value.get("source")
        if isinstance(source, dict) and source.get("type") == "base64":
            return True
        return any(_contains_structured_image(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_structured_image(item) for item in value)
    return False


def _parse_hook_output(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _audit(event: str, **fields: Any) -> None:
    path_raw = os.environ.get("RHO_RUNTIME_AUDIT_PATH", "").strip()
    if not path_raw:
        return
    path = Path(path_raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps({"event": event, **fields}, sort_keys=True) + "\n")
        stream.flush()
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    raise SystemExit(main())
