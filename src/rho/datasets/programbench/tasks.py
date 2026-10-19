"""Generate Harbor/Pier task packages from ProgramBench metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml


OFFICIAL_INSTRUCTION = """Build a new implementation from scratch that matches the behavior of the reference executable.

The reference executable is located at `./executable` in `/workspace`. Read all bundled documentation and query the executable through its normal user interface to infer its behavior.

Your submission must be an independently authored implementation. Do not obtain or reuse the original source code, clone the upstream repository, install an existing implementation of the target program, wrap or delegate to the reference executable, decompile it, or inspect it with tracing/disassembly tools. The environment has no internet access by design.

Work iteratively: inspect the documentation, probe observable behavior, implement the program, and test your own implementation against representative inputs. You may analyze binaries you build yourself.

Before finishing:

1. Provide an executable `./compile.sh` in `/workspace`.
2. `compile.sh` must build a fresh `./executable` from the submitted source tree without network access.
3. Do not rely on the bundled reference executable at build or runtime.
4. Keep generated build artifacts and the reference executable out of version control.
5. Commit your source changes if the workspace is a Git repository.
"""


def load_instances(programbench_root: Path) -> dict[str, dict[str, Any]]:
    tasks_root = programbench_root / "src" / "programbench" / "data" / "tasks"
    instances: dict[str, dict[str, Any]] = {}
    for path in sorted(tasks_root.glob("*/task.yaml")):
        metadata = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        task_id = path.parent.name
        instances[task_id] = {
            **metadata,
            "instance_id": task_id,
            "image_name": "programbench/" + task_id.replace("__", "_1776_"),
        }
    return instances


def render_task_toml(instance: dict[str, Any], *, agent_timeout_s: int) -> str:
    task_id = str(instance["instance_id"])
    metadata = {
        "task_id": task_id,
        "repository": str(instance.get("repository") or ""),
        "language": str(instance.get("language") or "unknown"),
        "difficulty": str(instance.get("difficulty") or "unknown"),
        "programbench_commit": str(instance.get("commit") or ""),
    }
    lines = [
        'schema_version = "1.3"',
        'artifacts = ["/logs/artifacts/submission.tar.gz"]',
        "[task]",
        f"name = {json.dumps('programbench/' + task_id)}",
        'description = "ProgramBench cleanroom reconstruction task"',
        "authors = []",
        'keywords = ["programbench", "reverse-engineering", "repository-generation"]',
        "[metadata]",
    ]
    lines.extend(f"{key} = {json.dumps(value)}" for key, value in metadata.items())
    lines.extend(
        [
            "[verifier]",
            'network_mode = "no-network"',
            "timeout_sec = 60.0",
            "",
            "[[verifier.collect]]",
            'command = "mkdir -p /logs/artifacts && tar --exclude=./executable --exclude=./.git -czf /logs/artifacts/submission.tar.gz -C /workspace ."',
            "timeout_sec = 600.0",
            "",
            "[agent]",
            'network_mode = "no-network"',
            f"timeout_sec = {float(agent_timeout_s):.1f}",
            'user = "agent"',
            "",
            "[environment]",
            f"docker_image = {json.dumps(str(instance['image_name']) + ':task_cleanroom_v6')}",
            'os = "linux"',
            'workdir = "/workspace"',
            "cpus = 8",
            "memory_mb = 32768",
            "storage_mb = 51200",
            "gpus = 0",
            "mcp_servers = []",
            "",
            "[environment.env]",
            'PAGER = "cat"',
            'MANPAGER = "cat"',
            'LESS = "-R"',
            "[solution.env]",
            "",
        ]
    )
    return "\n".join(lines)


def generate_tasks(
    *,
    programbench_root: Path,
    output_root: Path,
    task_ids: Iterable[str],
    agent_timeout_s: int,
) -> list[Path]:
    instances = load_instances(programbench_root)
    generated: list[Path] = []
    for task_id in task_ids:
        if task_id not in instances:
            raise ValueError(f"Unknown ProgramBench task: {task_id}")
        instance = instances[task_id]
        task_dir = output_root / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        # Pier requires an environment definition during local-task discovery
        # even when task.toml selects a prebuilt image. The Dockerfile remains
        # a faithful fallback and is not built on the normal prebuilt path.
        environment_dir = task_dir / "environment"
        environment_dir.mkdir(exist_ok=True)
        (environment_dir / "Dockerfile").write_text(
            f"FROM {instance['image_name']}:task_cleanroom_v6\n",
            encoding="utf-8",
        )
        (task_dir / "instruction.md").write_text(
            OFFICIAL_INSTRUCTION.rstrip() + "\n", encoding="utf-8"
        )
        (task_dir / "task.toml").write_text(
            render_task_toml(instance, agent_timeout_s=agent_timeout_s),
            encoding="utf-8",
        )
        generated.append(task_dir)
    return generated
