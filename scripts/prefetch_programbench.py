#!/usr/bin/env python3
"""Preflight ProgramBench images and hidden-test blobs before model calls."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from threading import Lock
import time
from typing import Any

from programbench.utils.blob_store import get_blob_dir
from programbench.utils.load_data import load_all_instances


def cached_blob_dir(cache_root: Path, task_id: str) -> Path | None:
    """Resolve a fully materialized ProgramBench blob without a network call."""

    from programbench.constants import HF_REPO_ID, HF_REVISION

    repository = cache_root / "hub" / f"datasets--{HF_REPO_ID.replace('/', '--')}"
    revision = HF_REVISION
    ref = repository / "refs" / revision
    if ref.is_file():
        revision = ref.read_text(encoding="utf-8").strip()
    candidate = repository / "snapshots" / revision / task_id
    return candidate if candidate.is_dir() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-ids", type=Path, required=True)
    parser.add_argument("--docker-executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--pull-timeout-s", type=float, default=1800.0)
    parser.add_argument(
        "--mirror-prefix",
        help=(
            "Optional registry prefix used only after the official pull fails; "
            "the mirror image is retagged to the official ProgramBench name."
        ),
    )
    args = parser.parse_args()

    task_ids = [
        line.strip()
        for line in args.task_ids.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    instances = {item["instance_id"]: item for item in load_all_instances()}
    unknown = sorted(set(task_ids) - set(instances))
    if unknown:
        raise SystemExit(f"unknown ProgramBench task IDs: {unknown}")

    cache_root = Path(
        os.environ.get("HF_HOME")
        or Path(__file__).resolve().parents[1] / "cache" / "huggingface"
    ).expanduser().resolve()
    os.environ["HF_HOME"] = str(cache_root)
    records: dict[str, dict[str, Any]] = {}
    # huggingface_hub's shared tqdm state is not safe when several independent
    # snapshot prefixes are materialized concurrently in the same process.
    # Keep blob resolution serial even if image inspection/pulls use workers.
    blob_lock = Lock()

    def image_present(image: str) -> bool:
        return (
            subprocess.run(
                [str(args.docker_executable), "image", "inspect", image],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )

    def image_runnable(image: str) -> tuple[bool, str]:
        """Verify that all referenced layers can create and start a container."""

        try:
            probe = subprocess.run(
                [
                    str(args.docker_executable),
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    image,
                    "true",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            return probe.returncode == 0, (probe.stdout + probe.stderr)[-2000:]
        except subprocess.TimeoutExpired as exc:
            stdout = (
                exc.stdout.decode(errors="replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            )
            stderr = (
                exc.stderr.decode(errors="replace")
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or "")
            )
            return False, (stdout + stderr + "\nDocker run probe timed out")[-2000:]

    def write_progress() -> None:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "task_count": len(task_ids),
            "ready": sum(record["status"] == "ready" for record in records.values()),
            "completed": len(records),
            "records": [records[task_id] for task_id in task_ids if task_id in records],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def preflight(task_id: str) -> tuple[str, dict[str, Any]]:
        image = f"{instances[task_id]['image_name']}:task_cleanroom_v6"
        record: dict[str, Any] = {"task_id": task_id, "image": image, "attempts": []}
        present = image_present(image)
        runnable, runnable_output = image_runnable(image) if present else (False, "")
        if present and runnable:
            record["attempts"].append(
                {
                    "attempt": 0,
                    "exit_code": 0,
                    "wall_time_s": 0.0,
                    "output_tail": "already present and runnable",
                }
            )
        else:
            if present:
                record["image_probe_error"] = runnable_output
            for attempt in range(1, args.max_attempts + 1):
                started = time.monotonic()
                try:
                    pull = subprocess.run(
                        [str(args.docker_executable), "pull", image],
                        capture_output=True,
                        text=True,
                        timeout=args.pull_timeout_s,
                    )
                    exit_code = pull.returncode
                    output_tail = (pull.stdout + pull.stderr)[-2000:]
                except subprocess.TimeoutExpired as exc:
                    exit_code = 124
                    stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                    stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
                    output_tail = (stdout + stderr + "\nDocker pull timed out")[-2000:]
                record["attempts"].append(
                    {
                        "attempt": attempt,
                        "source": "official",
                        "exit_code": exit_code,
                        "wall_time_s": round(time.monotonic() - started, 3),
                        "output_tail": output_tail,
                    }
                )
                if exit_code == 0:
                    break
                if "toomanyrequests" in output_tail.lower() or "pull rate limit" in output_tail.lower():
                    break
                time.sleep(min(60, 5 * attempt))
            if record["attempts"][-1]["exit_code"] != 0 and args.mirror_prefix:
                mirror_image = f"{args.mirror_prefix.rstrip('/')}/{image}"
                for attempt in range(1, args.max_attempts + 1):
                    started = time.monotonic()
                    try:
                        pull = subprocess.run(
                            [str(args.docker_executable), "pull", mirror_image],
                            capture_output=True,
                            text=True,
                            timeout=args.pull_timeout_s,
                        )
                        exit_code = pull.returncode
                        output_tail = (pull.stdout + pull.stderr)[-2000:]
                    except subprocess.TimeoutExpired as exc:
                        exit_code = 124
                        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
                        output_tail = (stdout + stderr + "\nDocker mirror pull timed out")[-2000:]
                    record["attempts"].append(
                        {
                            "attempt": attempt,
                            "source": "mirror",
                            "image": mirror_image,
                            "exit_code": exit_code,
                            "wall_time_s": round(time.monotonic() - started, 3),
                            "output_tail": output_tail,
                        }
                    )
                    if exit_code == 0:
                        tag = subprocess.run(
                            [str(args.docker_executable), "tag", mirror_image, image],
                            capture_output=True,
                            text=True,
                        )
                        if tag.returncode == 0:
                            record["mirror_retagged"] = True
                            break
                        record["attempts"][-1]["exit_code"] = tag.returncode
                        record["attempts"][-1]["output_tail"] += (
                            "\nDocker mirror retag failed\n" + tag.stdout + tag.stderr
                        )[-2000:]
                    time.sleep(min(60, 5 * attempt))
            if record["attempts"][-1]["exit_code"] != 0:
                record["status"] = "image_failed"
                return task_id, record
            runnable, runnable_output = image_runnable(image)
            if not runnable:
                record["status"] = "image_failed"
                record["image_probe_error"] = runnable_output
                return task_id, record
        inspect = subprocess.run(
            [
                str(args.docker_executable),
                "image",
                "inspect",
                image,
                "--format",
                "{{json .RepoDigests}}",
            ],
            capture_output=True,
            text=True,
        )
        record["repo_digests"] = inspect.stdout.strip()
        try:
            with blob_lock:
                blob = cached_blob_dir(cache_root, task_id)
                if blob is None:
                    blob = get_blob_dir(task_id)
                # Partial snapshot downloads can return the shared snapshot root
                # without materializing a newly requested prefix. Force that
                # prefix once before declaring an infrastructure failure.
                if blob is None or not blob.is_dir():
                    from huggingface_hub import snapshot_download
                    from programbench.constants import HF_REPO_ID, HF_REVISION

                    base = Path(
                        snapshot_download(
                            HF_REPO_ID,
                            repo_type="dataset",
                            revision=HF_REVISION,
                            allow_patterns=f"{task_id}/**",
                            force_download=True,
                            max_workers=4,
                        )
                    )
                    blob = base / task_id
                    record["blob_force_downloaded"] = True
            if blob is None or not blob.is_dir():
                raise RuntimeError("hidden-test blob directory is missing")
            record["blob_dir"] = str(blob)
            record["status"] = "ready"
        except Exception as exc:
            record["status"] = "blob_failed"
            record["blob_error"] = f"{type(exc).__name__}: {exc}"
        return task_id, record

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(preflight, task_id): task_id for task_id in task_ids}
        for future in as_completed(futures):
            task_id, record = future.result()
            records[task_id] = record
            write_progress()
            print(f"[{task_id}] {record['status']}", flush=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_count": len(task_ids),
        "ready": sum(record["status"] == "ready" for record in records.values()),
        "completed": len(records),
        "records": [records[task_id] for task_id in task_ids],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    if payload["ready"] != len(task_ids):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
