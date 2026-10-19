from __future__ import annotations

import ast
import contextlib
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator


def cache_root() -> Path:
    root = os.environ.get("RHO_CACHE_DIR", ".cache/rho")
    return Path(root).expanduser().resolve()


def parse_list_field(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except (json.JSONDecodeError, SyntaxError, ValueError):
                continue
            if isinstance(parsed, (list, tuple)):
                return [str(item) for item in parsed]
        return [text]
    return [str(value)]


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def read_rows_from_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise RuntimeError(
                "Reading a local SWE-bench Pro parquet snapshot requires the "
                "`swebench-pro` dependencies. Run `uv sync --extra swebench-pro` "
                "or `uv run --extra swebench-pro ...`."
            ) from exc
        return [dict(row) for row in Dataset.from_parquet(str(path))]
    if suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(item) for item in payload]
        if isinstance(payload, dict):
            for key in ("rows", "test", "data"):
                if isinstance(payload.get(key), list):
                    return [dict(item) for item in payload[key]]
        raise ValueError(f"Unsupported JSON dataset shape in {path}")
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError(f"Unsupported SWE-bench Pro fixture extension: {path.suffix}")


def git(cmd: list[str], *, cwd: Path | None = None, timeout_s: float = 1800.0) -> str:
    proc = subprocess.run(
        ["git", *cmd],
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "git {cmd} failed with exit {code}\nstdout:\n{stdout}\nstderr:\n{stderr}".format(
                cmd=" ".join(cmd),
                code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        )
    return proc.stdout


@contextlib.contextmanager
def file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")
    try:
        try:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
            yield
        except ImportError:
            lock_dir = path.with_suffix(path.suffix + ".d")
            while True:
                try:
                    lock_dir.mkdir()
                    break
                except FileExistsError:
                    time.sleep(0.1)
            try:
                yield
            finally:
                shutil.rmtree(lock_dir, ignore_errors=True)
    finally:
        handle.close()


def row_digest(row: dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
