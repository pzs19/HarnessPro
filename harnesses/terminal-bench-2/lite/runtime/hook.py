#!/usr/bin/env python3
"""Runtime enforcement for the cost-aware Terminal-Bench harness.

The hook is intentionally stdlib-only. It is loaded as a user-level Codex hook
by the evaluation runner, while its thresholds and prompt text remain part of
the captured Harness in policy.json.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


STATE_DIR = ".rho_runtime"


def main() -> int:
    payload = json.load(sys.stdin)
    cwd = Path(str(payload.get("cwd") or os.getcwd())).resolve()
    event = str(payload.get("hook_event_name") or "")
    control_state = _control_state_dir(cwd)
    selection_path_raw = os.environ.get("RHO_RUNTIME_SELECTION_PATH", "").strip()
    selection = (
        _read_json(Path(selection_path_raw))
        if selection_path_raw
        else _read_json(control_state / "selection.json")
    )

    if event == "SessionStart":
        context = str(selection.get("session_context") or "").strip()
        guidance = str(selection.get("task_guidance") or "").strip()
        if guidance:
            context = f"{context}\nTask-feature guidance: {guidance}"
        limits = (
            selection.get("economy_after_tools"),
            selection.get("strict_after_tools"),
            selection.get("finalize_after_tools"),
        )
        _emit_context(
            "SessionStart",
            f"{context}\nTool-stage boundaries selected by the task-complexity rubric: "
            f"economy={limits[0]}, strict={limits[1]}, finalize={limits[2]}.",
        )
        _audit("session_start", cwd=str(cwd), limits=list(limits))
        return 0

    if event == "PreToolUse":
        return _pre_tool_use(payload, cwd, control_state, selection)
    if event == "PostToolUse":
        return _post_tool_use(payload, cwd, selection)
    return 0


def _pre_tool_use(
    payload: dict[str, Any],
    cwd: Path,
    control_state: Path,
    selection: dict[str, Any],
) -> int:
    tool_name = str(payload.get("tool_name") or "")
    state = _increment_tool_count(control_state)
    count = int(state["tool_count"])
    finalize = int(selection.get("finalize_after_tools") or 0)
    if finalize and count > finalize:
        _audit(
            "pre_tool_use",
            cwd=str(cwd),
            tool_name=tool_name,
            tool_count=count,
            decision="deny",
        )
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": (
                            "The task tool budget is exhausted. Preserve the current files, "
                            "stop calling tools, and give the best final answer from existing evidence."
                        ),
                    }
                }
            )
        )
        return 0

    context = _stage_context(count, selection)
    updated_input = _rtk_rewrite(payload, selection) if tool_name == "Bash" else None
    _audit(
        "pre_tool_use",
        cwd=str(cwd),
        tool_name=tool_name,
        tool_count=count,
        stage=_stage_name(count, selection),
        input_rewritten=updated_input is not None,
    )
    if context or updated_input is not None:
        specific: dict[str, Any] = {"hookEventName": "PreToolUse"}
        if context:
            specific["additionalContext"] = context
        if updated_input is not None:
            specific["permissionDecision"] = "allow"
            specific["updatedInput"] = updated_input
        print(json.dumps({"hookSpecificOutput": specific}))
    return 0


def _post_tool_use(payload: dict[str, Any], cwd: Path, selection: dict[str, Any]) -> int:
    # Use an explicit adapter marker. Claude Code 2.1.231 currently emits
    # ``tool_response`` in command-hook payloads even though its plugin docs
    # describe the field as ``tool_result``. Field-name inference therefore
    # selected the Codex response schema and left the original result in the
    # conversation. Keep ``tool_result`` as a compatibility fallback.
    is_claude_code = (
        os.environ.get("RHO_RUNTIME_INTERFACE") == "claude-code"
        or "tool_result" in payload
    )
    response = (
        payload.get("tool_result", payload.get("tool_response"))
        if is_claude_code
        else payload.get("tool_response")
    )
    rendered = response if isinstance(response, str) else json.dumps(
        response,
        ensure_ascii=False,
        indent=2,
    )
    max_chars = int(selection.get("tool_output_max_chars") or 0)
    if not max_chars or len(rendered) <= max_chars:
        _audit(
            "post_tool_use",
            cwd=str(cwd),
            tool_name=str(payload.get("tool_name") or ""),
            compacted=False,
            output_chars=len(rendered),
        )
        return 0

    raw_dir = cwd / STATE_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tool_use_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(payload.get("tool_use_id") or "tool"))
    raw_path = raw_dir / f"{tool_use_id}.json"
    raw_path.write_text(
        json.dumps(response, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    compact = _compact_text(
        rendered,
        head_chars=int(selection.get("tool_output_head_chars") or max_chars * 2 // 3),
        tail_chars=int(selection.get("tool_output_tail_chars") or max_chars // 4),
    )
    relative = raw_path.relative_to(cwd).as_posix()
    reason = (
        f"[Tool output compacted from {len(rendered)} characters. Full raw result: "
        f"{relative}. Read only a targeted range if more detail is necessary.]\n{compact}"
    )
    _audit(
        "post_tool_use",
        cwd=str(cwd),
        tool_name=str(payload.get("tool_name") or ""),
        compacted=True,
        output_chars=len(rendered),
        compacted_chars=len(compact),
        raw_path=relative,
    )
    if is_claude_code:
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PostToolUse",
                        "updatedToolOutput": reason,
                        "additionalContext": (
                            "The original tool result was replaced by a recoverable head/tail view. "
                            "Do not rerun the command merely to recover omitted progress noise."
                        ),
                    }
                },
                ensure_ascii=False,
            )
        )
        return 0
    print(
        json.dumps(
            {
                "decision": "block",
                "reason": reason,
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": (
                        "The original tool result was replaced by a recoverable head/tail view. "
                        "Do not rerun the command merely to recover omitted progress noise."
                    ),
                },
            },
            ensure_ascii=False,
        )
    )
    return 0


def _stage_context(count: int, selection: dict[str, Any]) -> str:
    economy = int(selection.get("economy_after_tools") or 0)
    strict = int(selection.get("strict_after_tools") or 0)
    reminder = int(selection.get("strict_reminder_every") or 8)
    if count == economy:
        return (
            "Economy phase starts now. Batch independent ls/rg/read/test checks, suppress "
            "process narration, and do not repeat unchanged checks. Do not trade away the "
            "behavioral validation or exact final-state checks required by the task."
        )
    if count >= strict and (count == strict or (count - strict) % reminder == 0):
        return (
            "Convergence phase: stop broad exploration. Only inspect evidence directly tied to the "
            "current failure or deliverable. If a concrete failure, unexplained instability, or "
            "mutated deliverable remains, repair it rather than finalizing because the phase changed. "
            "Avoid irreversible cleanup or broad state changes not required by the task, and keep "
            "temporary verification artifacts separate from final deliverables. "
            "Then recheck exact paths, interfaces, behavior, and final state once and finish."
        )
    return ""


def _stage_name(count: int, selection: dict[str, Any]) -> str:
    finalize = int(selection.get("finalize_after_tools") or 0)
    strict = int(selection.get("strict_after_tools") or 0)
    economy = int(selection.get("economy_after_tools") or 0)
    if finalize and count > finalize:
        return "finalize"
    if strict and count >= strict:
        return "strict"
    if economy and count >= economy:
        return "economy"
    return "normal"


def _rtk_rewrite(payload: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any] | None:
    if not selection.get("rtk_enabled"):
        return None
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict) or not isinstance(tool_input.get("command"), str):
        return None
    command = tool_input["command"]
    if not _is_safe_rtk_candidate(command):
        return None
    binary_raw = os.environ.get("RHO_RTK_BIN", "")
    binary = Path(binary_raw) if binary_raw else None
    if binary is None or not binary.is_file():
        return None
    proc = subprocess.run(
        [str(binary), "rewrite", command],
        capture_output=True,
        text=True,
        check=False,
        timeout=3,
        env={**os.environ, "RTK_TELEMETRY_DISABLED": "1"},
    )
    rewritten = proc.stdout.strip()
    if proc.returncode not in {0, 3} or not rewritten or rewritten == command:
        return None
    replacement = dict(tool_input)
    replacement["command"] = (
        f"export PATH={_shell_quote(str(binary.parent))}:\"$PATH\"; {rewritten}"
    )
    return replacement


def _increment_tool_count(state_dir: Path) -> dict[str, int]:
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "state.lock"
    state_path = state_dir / "state.json"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state = _read_json(state_path) if state_path.exists() else {}
        state["tool_count"] = int(state.get("tool_count") or 0) + 1
        state_path.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return {"tool_count": int(state["tool_count"])}


def _control_state_dir(cwd: Path) -> Path:
    root_raw = os.environ.get("RHO_RUNTIME_STATE_ROOT", "")
    root = Path(root_raw).resolve() if root_raw else cwd / STATE_DIR / "control"
    key = hashlib.sha256(str(cwd.resolve()).encode("utf-8")).hexdigest()[:24]
    return root / key


def _is_safe_rtk_candidate(command: str) -> bool:
    """Use RTK only where its filter preserves command semantics reliably.

    The 0.28 rewrite registry is intentionally broad. In TB2 that changed
    `rg --files` into an invalid grep invocation and dropped `find -print`
    semantics, so the Harness applies a smaller, high-value allowlist.
    """

    stripped = command.strip()
    if not stripped or "\n" in stripped:
        return False
    if re.search(r"(?:<<|[<>]|\$\(|`)", stripped):
        return False
    if re.match(r"^(?:sudo\s+)?docker\s+(?:exec|cp|run)\b", stripped):
        return False
    if re.match(r"^(?:rg|grep)\b", stripped) and re.search(r"(?:^|\s)--files(?:\s|$)", stripped):
        return False
    if re.match(r"^find\b", stripped) and re.search(r"(?:^|\s)-(?:print|print0|exec|delete)\b", stripped):
        return False

    safe = (
        r"git\s+(?:status|diff|log|show)\b",
        r"(?:pytest|python(?:3)?\s+-m\s+pytest)\b",
        r"(?:uv\s+run\s+pytest|uv\s+run\s+python\s+-m\s+pytest)\b",
        r"cargo\s+(?:test|check|build|clippy)\b",
        r"(?:npm|pnpm|yarn)\s+(?:test|run\s+(?:test|build|lint))\b",
        r"go\s+(?:test|build|vet)\b",
        r"ruff\s+(?:check|format)\b",
        r"docker\s+(?:ps|images|logs|compose\s+ps)\b",
    )
    return any(re.match(rf"^(?:env\s+[^ ]+=\S+\s+)*{pattern}", stripped) for pattern in safe)


def _compact_text(text: str, *, head_chars: int, tail_chars: int) -> str:
    lines = text.splitlines()
    deduped: list[str] = []
    index = 0
    while index < len(lines):
        next_index = index + 1
        while next_index < len(lines) and lines[next_index] == lines[index]:
            next_index += 1
        repeats = next_index - index
        deduped.append(lines[index])
        if repeats > 2:
            deduped.append(f"[previous line repeated {repeats - 1} more times]")
        elif repeats == 2:
            deduped.append(lines[index])
        index = next_index
    compact = "\n".join(deduped)
    if len(compact) <= head_chars + tail_chars:
        return compact
    omitted = len(compact) - head_chars - tail_chars
    return (
        compact[:head_chars]
        + f"\n... [{omitted} characters omitted; use the raw path for a targeted range] ...\n"
        + compact[-tail_chars:]
    )


def _emit_context(event_name: str, context: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": event_name,
                    "additionalContext": context,
                }
            },
            ensure_ascii=False,
        )
    )


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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    raise SystemExit(main())

