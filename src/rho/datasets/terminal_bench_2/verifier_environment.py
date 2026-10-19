"""Opt-in dependency environment repairs, outside Solver/Harness state."""
from __future__ import annotations

import hashlib
import json
import os
import shlex
from pathlib import Path


ALLOWED_ENV = {"UV_INDEX_URL", "UV_DEFAULT_INDEX", "UV_OFFLINE", "UV_INSECURE_HOST"}


def verifier_environment(task_dir: Path) -> dict[str, str]:
    raw = os.environ.get("RHO_TB2_VERIFIER_ENV_MANIFEST")
    if not raw:
        return {}
    manifest = json.loads(Path(raw).read_text())
    if manifest.get("version") != 1:
        raise ValueError("unsupported verifier dependency manifest")
    entry = manifest.get("tasks", {}).get(task_dir.name)
    if entry is None:
        return {}
    digest = hashlib.sha256((task_dir / "tests/test.sh").read_bytes()).hexdigest()
    if digest != entry.get("test_script_sha256"):
        raise ValueError("verifier script changed; dependency override needs revalidation")
    env = entry.get("environment")
    if not isinstance(env, dict) or not env or set(env) - ALLOWED_ENV:
        raise ValueError("only explicit UV dependency environment settings are permitted")
    if not all(isinstance(value, str) and "\x00" not in value for value in env.values()):
        raise ValueError("dependency environment values must be strings")
    return dict(env)


def environment_prefix(env: dict[str, str]) -> str:
    return "".join("export " + shlex.quote(key + "=" + value) + "; " for key, value in sorted(env.items()))
