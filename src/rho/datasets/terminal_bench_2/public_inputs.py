"""Opt-in, hash-checked mirrors of public inputs named in task instructions.

This carries raw input bytes only, never Solver outputs or verifier assets.
Without the explicit environment setting the evaluation protocol is unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

MANIFEST_ENV = "RHO_TB2_PUBLIC_INPUT_MANIFEST"


def stage_public_inputs(instruction: str, workdir: Path) -> None:
    configured = os.environ.get(MANIFEST_ENV, "").strip()
    if not configured:
        return
    manifest = Path(configured).resolve()
    payload = json.loads(manifest.read_text())
    urls = {u.rstrip(".,;:!?)\"]}") for u in re.findall(r"https?://[^\s<>]+", instruction)}
    selected = []
    plans = []
    for item in payload["inputs"]:
        if item["url"] not in urls:
            continue
        source = (manifest.parent / item["file"]).resolve()
        if not source.is_relative_to(manifest.parent) or not source.is_file():
            raise ValueError("public input must be a file inside the manifest directory")
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["sha256"]:
            raise ValueError("public input SHA256 mismatch")
        if source.stat().st_size < 100 and source.read_bytes().startswith(b"HASH:"):
            raise ValueError("a trajectory hash placeholder is not a public input")
        filename = digest + source.suffix
        relative = ".tb2/public-inputs/" + filename
        plans.append((source, workdir / relative))
        selected.append({"url": item["url"], "sha256": digest,
                         "file": relative, "size_bytes": source.stat().st_size})
    if not plans:
        return
    directory = workdir / ".tb2/public-inputs"
    directory.mkdir(parents=True, exist_ok=True)
    for source, destination in plans:
        if destination.exists():
            raise ValueError("refusing to overwrite an existing public input")
        shutil.copyfile(source, destination)
    (directory / "manifest.json").write_text(json.dumps({"inputs": selected}, indent=2) + "\n")


def public_input_notice(workdir: Path) -> str:
    path = workdir / ".tb2/public-inputs/manifest.json"
    if not path.is_file():
        return ""
    payload = json.loads(path.read_text())
    lines = ["", "## Public input mirrors", "",
             "The following unmodified public task inputs are locally available to avoid external download failures.",
             "These copies contain no generated answers or transcription. Paths are inside the task container:", ""]
    for item in payload["inputs"]:
        lines.append(f"- {item['url']} → `/host-ws/{item['file']}` (SHA256 `{item['sha256']}`)")
    return "\n".join(lines) + "\n"
