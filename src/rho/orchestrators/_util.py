from __future__ import annotations

import json
from pathlib import Path

from rho.protocols import Task, Trajectory

HARNESS_DESCRIPTION = """\
The harness (harness/) is a toolkit of resources and guidance that helps \
the agent solve tasks. It can contain any type of file — helper scripts, \
artifacts, environment setup, documentation with relevant context, and \
workflows to follow."""


def dump_trajectory(dest: Path, task: Task | None, traj: Trajectory) -> None:
    """Dump a trajectory to disk for LLM inspection.

    Writes events.jsonl, final_message.txt, workspace_diff/, and optionally
    task/prompt.md. Used by optimize, diagnose, and evaluate orchestrators.
    """
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in traj.events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    (dest / "final_message.txt").write_text(traj.final_message, encoding="utf-8")
    if task is not None:
        task.materialize(dest / "task")
    diff_dir = dest / "workspace_diff"
    diff_dir.mkdir()
    for rel, content in traj.workspace_diff.items():
        target = diff_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    if traj.workspace_deletions:
        (dest / "_deletions.txt").write_text(
            "\n".join(sorted(traj.workspace_deletions)),
            encoding="utf-8",
        )
