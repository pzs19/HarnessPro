#!/usr/bin/env python3
"""Generate local Harbor/Pier packages for selected ProgramBench tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

from rho.datasets.programbench.tasks import generate_tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--programbench-root", type=Path, required=True)
    parser.add_argument("--task-ids", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--agent-timeout-s", type=int, default=21600)
    args = parser.parse_args()
    task_ids = [
        line.strip()
        for line in args.task_ids.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    generated = generate_tasks(
        programbench_root=args.programbench_root.resolve(),
        output_root=args.output_root.resolve(),
        task_ids=task_ids,
        agent_timeout_s=args.agent_timeout_s,
    )
    print(f"generated {len(generated)} ProgramBench task packages in {args.output_root}")


if __name__ == "__main__":
    main()
