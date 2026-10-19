from __future__ import annotations


_PROMPT = """\
# Terminal-Bench 2: {task_id}

Difficulty: {difficulty}    Category: {category}

## How to operate

You run on the host, outside the task environment. The task's target
environment is a Docker container named `{container_name}` that is
running now. Your host working directory (`task/` in your workspace)
is mounted inside the container at **`/host-ws`** (read-write).

To execute any command in the target environment, use this two-step
pattern (avoids shell-quoting problems entirely):

    # 1. Write the command as a shell script in your host task/ dir:
    cat > solve_step_01.sh <<'BASH'
    set -euxo pipefail
    cd /app
    # ... your commands here; any quoting, heredocs, awk/sed are fine ...
    BASH

    # 2. Execute it inside the container:
    docker exec -i {container_name} bash /host-ws/solve_step_01.sh

The container filesystem persists across calls. Your host `task/`
directory also persists and is always visible at `/host-ws` inside
the container. Files you want graded must be inside the container's
own filesystem (typical location: `/app`); changes to `/host-ws`
only affect what scripts you write for yourself.

## Research integrity rules (required)

Solve the task on the merits. Do NOT:

- Search for, read, copy, or execute any file whose path contains
  `tests/`, `test_outputs`, `solution/`, `solve.sh`, `reward.txt`,
  or `/logs/verifier/` — these are grader assets and are off-limits.
- Run `pytest`, `unittest`, or any other test runner in the container
  or on the host.
- Access the Terminal-Bench 2 task repository on the host filesystem
  (for example, anything under `terminal-bench-2/` or `tb2-repo/`).
- Attempt to trigger or inspect the grading pipeline yourself.

The grading harness is not present in the container during your
session. Grading runs automatically after you finish; you do not
need to — and must not — invoke it.

## Time budget

Agent timeout: {agent_timeout_sec:.0f} seconds (ENFORCED by a watchdog —
if you exceed this, the container is killed and grading fails).
Verifier timeout: {verifier_timeout_sec:.0f} seconds (during grading).

## Task

{instruction_md}

## Finishing

When you believe the task is complete, leave artifacts inside the
container and summarize what you did in your final message. The
grading step runs automatically after your final message.
"""


def render_prompt(
    *,
    task_id: str,
    difficulty: str,
    category: str,
    container_name: str,
    agent_timeout_sec: float,
    verifier_timeout_sec: float,
    instruction_md: str,
) -> str:
    return _PROMPT.format(
        task_id=task_id,
        difficulty=difficulty,
        category=category,
        container_name=container_name,
        agent_timeout_sec=agent_timeout_sec,
        verifier_timeout_sec=verifier_timeout_sec,
        instruction_md=instruction_md,
    )

