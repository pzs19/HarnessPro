from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from rho.protocols import Trajectory, TrajectoryKind

Mode = str


@dataclass
class FakeResponse:
    final_message: str = ""
    workspace_edits: dict[str, bytes] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    exit_code: int = 0


ScriptFn = Callable[[Path, str, dict | None], FakeResponse]


class FakeAgent:
    __rho_bypass_cache__ = True

    def __init__(self, scripts: dict[Mode, ScriptFn]) -> None:
        self.scripts = scripts
        self.calls: list[tuple[Mode, Path]] = []
        self.run_envs: list[dict[str, str]] = []

    def run(
        self,
        workspace: Path,
        instructions: str,
        *,
        output_schema: dict | None = None,
        task_id: str = "",
        harness_id: str = "",
        kind: TrajectoryKind = "solve",
        timeout_s: float | None = None,
        env: dict[str, str] | None = None,
    ) -> Trajectory:
        del timeout_s
        mode = kind
        if mode not in self.scripts:
            raise KeyError(f"FakeAgent has no script for mode={mode!r}")

        meta = workspace / ".rho"
        meta.mkdir(exist_ok=True)
        (meta / "instructions.md").write_text(instructions, encoding="utf-8")

        t0 = time.monotonic()
        response = self.scripts[mode](workspace, instructions, output_schema)
        self.calls.append((mode, workspace))
        self.run_envs.append({key: str(value) for key, value in sorted((env or {}).items())})

        for rel, content in response.workspace_edits.items():
            target = workspace / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if content == b"":
                if target.exists():
                    target.unlink()
            else:
                target.write_bytes(content)

        if output_schema is not None:
            _validate_output_schema(response.final_message, output_schema, mode)

        return Trajectory(
            id=f"fake_{uuid.uuid4().hex[:10]}",
            kind=kind,
            task_id=task_id,
            harness_id=harness_id,
            instructions=instructions,
            events=response.events,
            final_message=response.final_message,
            stdout="",
            stderr="",
            workspace_diff={
                key: value for key, value in response.workspace_edits.items() if value != b""
            },
            workspace_deletions=frozenset(
                key for key, value in response.workspace_edits.items() if value == b""
            ),
            exit_code=response.exit_code,
            wall_time_s=time.monotonic() - t0,
            timed_out=False,
        )


def _validate_output_schema(final_message: str, output_schema: dict, mode: str) -> None:
    try:
        parsed = json.loads(final_message)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"FakeAgent script for mode={mode} returned non-JSON final_message "
            "but output_schema was provided"
        ) from exc

    for key in output_schema.get("required", []):
        if key not in parsed:
            raise ValueError(f"FakeAgent script missing required field {key!r}")

    value_schema = output_schema.get("properties", {}).get("value")
    if "value" in parsed and isinstance(value_schema, dict):
        value = parsed["value"]
        minimum = value_schema.get("minimum")
        maximum = value_schema.get("maximum")
        if minimum is not None and value < minimum:
            raise ValueError(f"FakeAgent script field 'value' below minimum {minimum}")
        if maximum is not None and value > maximum:
            raise ValueError(f"FakeAgent script field 'value' above maximum {maximum}")
