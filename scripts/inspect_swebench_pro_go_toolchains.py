#!/usr/bin/env python3
"""Read Go versions from the official images used by the selected tasks."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess

from rho.datasets.swebench_pro.dataset import load_rows
from rho.datasets.swebench_pro.evaluator import docker_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--heldout-ids", type=Path, required=True)
    parser.add_argument("--docker-executable", default="docker")
    args = parser.parse_args()

    ids = [
        line.strip()
        for line in args.heldout_ids.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    rows = {str(row["instance_id"]): row for row in load_rows(args.dataset)}
    missing = sorted(set(ids) - rows.keys())
    if missing:
        parser.error(f"split IDs are missing from dataset: {missing}")
    images = sorted({
        docker_image(rows[task_id])
        for task_id in ids
        if str(rows[task_id].get("repo_language") or "").lower() == "go"
    })
    versions: set[str] = set()
    for image in images:
        result = subprocess.run(
            [args.docker_executable, "run", "--rm", "--entrypoint", "go", image, "version"],
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        output = (result.stdout + result.stderr).strip()
        match = re.search(r"\bgo(\d+\.\d+(?:\.\d+)?)\b", output)
        if result.returncode or match is None:
            raise SystemExit(f"cannot inspect official Go image {image}: {output[-1000:]}")
        version = match.group(1)
        versions.add(version)
        print(f"{image}: go{version}")
    print("Required local toolchains:", ", ".join(f"go{v}" for v in sorted(versions)))


if __name__ == "__main__":
    main()
