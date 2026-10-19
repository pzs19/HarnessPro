from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskToml:
    difficulty: str
    category: str
    tags: tuple[str, ...]
    agent_timeout_sec: float
    verifier_timeout_sec: float
    docker_image: str
    cpus: int | None
    memory: str | None
    build_timeout_sec: float | None


def load_task_toml(path: Path) -> TaskToml:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    try:
        metadata = data["metadata"]
        environment = data["environment"]
        verifier = data["verifier"]
        agent = data["agent"]
        return TaskToml(
            difficulty=str(metadata["difficulty"]),
            category=str(metadata.get("category", "")),
            tags=tuple(str(tag) for tag in metadata.get("tags", [])),
            agent_timeout_sec=float(agent["timeout_sec"]),
            verifier_timeout_sec=float(verifier["timeout_sec"]),
            docker_image=str(environment["docker_image"]),
            cpus=_opt_int(environment.get("cpus")),
            memory=_opt_str(environment.get("memory")),
            build_timeout_sec=_opt_float(environment.get("build_timeout_sec")),
        )
    except KeyError as exc:
        raise KeyError(f"task.toml at {path} missing required field: {exc}") from exc


def _opt_int(value: object) -> int | None:
    return int(value) if value is not None else None


def _opt_float(value: object) -> float | None:
    return float(value) if value is not None else None


def _opt_str(value: object) -> str | None:
    return str(value) if value is not None else None

