from __future__ import annotations

import contextlib
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from rho.agent.base import Agent
from rho.observability import annotate_trajectory
from rho.orchestrators._util import HARNESS_DESCRIPTION
from rho.protocols import Harness, Task, Trajectory

SOLVE_INSTRUCTIONS = f"""
Solve the task defined in task/prompt.md, using the information and tools available under harness/.

{HARNESS_DESCRIPTION}

Workspace layout:
  harness/   — read and invoke anything here, but do not modify it
  task/      — files for this task, including prompt.md

Steps:
1. Familiarize yourself with the information and available tools in harness/.
2. Read and analyze task/prompt.md.
3. Complete the task. For code repair tasks, modify files directly under task/repo/.
4. Present your final answer in your last message, in the format prompt.md specifies (or plain prose if unspecified).
"""

RUNTIME_HARNESS_SOLVE_INSTRUCTIONS = """
Solve the task defined in task/prompt.md. Runtime Harness guidance is injected
at session start; do not inventory harness/ unless task/prompt.md explicitly
names a file there.

Workspace layout:
  task/ — prompt.md and task files

Read task/prompt.md, complete the requested work in the task runtime, and leave
all graded artifacts at the exact requested paths. Present a concise final
answer in the requested format.
"""


@contextmanager
def solve_workspace(task: Task, harness: Harness, workdir: Path) -> Iterator[Path]:
    """Prepare a solve workspace and enter the task runtime if it exists.

    Note: codex agents may invoke `docker run` with `task/repo` bind-mounted,
    which leaves root-owned files inside the workspace (observed in
    SWE-bench Pro 2026-05-04 / 2026-05-08 runs). `tempfile.TemporaryDirectory`
    even with `ignore_cleanup_errors=True` does not survive this — its cleanup
    calls `_resetperms` (chmod) before rmtree, and that chmod EPERMs out as an
    unhandled `PermissionError`. We therefore manage the tempdir manually and
    cleanup with `shutil.rmtree(ignore_errors=True)`, which never tries to
    chmod and swallows all unlink/rmdir errors. Root-owned leftovers are left
    on disk; the user can sweep them out-of-band with `sudo rm -rf`.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=str(workdir), prefix="solve_"))
    try:
        (tmp / "harness").mkdir()
        (tmp / "task").mkdir()
        harness.materialize(tmp / "harness")
        task.materialize(tmp / "task")
        runtime_session = getattr(task, "runtime_session", None)
        cm = (
            runtime_session(tmp / "task")
            if runtime_session is not None
            else contextlib.nullcontext()
        )
        with cm:
            yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def solve_in(
    agent: Agent,
    task: Task,
    harness: Harness,
    ws: Path,
    *,
    sample_index: int | None = None,
    stage: str | None = None,
    round_ix: int | None = None,
    instructions: str = SOLVE_INSTRUCTIONS,
    env: dict[str, str] | None = None,
) -> Trajectory:
    """Run the agent in an already-prepared workspace."""
    if sample_index is not None:
        (ws / ".sample_index").write_text(str(sample_index), encoding="utf-8")
    trajectory = agent.run(
        ws,
        instructions,
        task_id=task.id,
        harness_id=harness.id,
        kind="solve",
        timeout_s=task.agent_timeout_s,
        env=env,
    )
    return annotate_trajectory(
        trajectory,
        agent=agent,
        stage=stage,
        round_ix=round_ix,
        sample_index=sample_index,
    )


def solve(
    agent: Agent,
    task: Task,
    harness: Harness,
    *,
    workdir: Path,
    sample_index: int | None = None,
    stage: str | None = None,
    round_ix: int | None = None,
) -> Trajectory:
    """Back-compat entry point that prepares the workspace and runs the agent."""
    with solve_workspace(task, harness, workdir) as ws:
        return solve_in(
            agent,
            task,
            harness,
            ws,
            sample_index=sample_index,
            stage=stage,
            round_ix=round_ix,
        )
