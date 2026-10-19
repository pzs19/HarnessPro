from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from rho.datasets.swebench_pro.util import git
from rho.protocols import Trajectory

_REPO_PREFIX = "task/repo/"
_FENCED_DIFF = re.compile(r"```(?:diff|patch)?\s*\n(diff --git .*?)\n```", re.DOTALL)


def extract_prediction_patch(
    trajectory: Trajectory,
    *,
    materialize_repo: Callable[[Path], None],
    artifacts_dir: Path,
) -> str:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    if _has_repo_edits(trajectory):
        # Patch extraction needs a materialized Git checkout only while the diff
        # is being computed. Keeping that checkout below the persistent artifact
        # directory duplicates the full-history object database for every task
        # (several GiB for large repositories) and can exhaust shared-disk quota.
        # Retain only the generated patch and small diagnostics in artifacts.
        with tempfile.TemporaryDirectory(prefix="rho-swebench-patch-") as temp:
            repo_dir = Path(temp) / "repo"
            materialize_repo(repo_dir)
            skipped = _apply_workspace_edits(trajectory, repo_dir)
            git(["add", "-N", "."], cwd=repo_dir)
            patch = git(["diff", "--binary", "--", "."], cwd=repo_dir)
        if skipped:
            (artifacts_dir / "skipped_large_files.txt").write_text(
                "\n".join(sorted(skipped)) + "\n", encoding="utf-8"
            )
        if patch.strip():
            (artifacts_dir / "prediction.patch").write_text(patch, encoding="utf-8")
            return patch

    patch = extract_patch_from_message(trajectory.final_message)
    if patch.strip():
        (artifacts_dir / "prediction.patch").write_text(patch, encoding="utf-8")
        return patch
    raise ValueError("trajectory did not modify task/repo and final_message contains no git patch")


def _has_repo_edits(trajectory: Trajectory) -> bool:
    for rel in trajectory.workspace_diff:
        if rel.startswith(_REPO_PREFIX) and not _skip_repo_rel(rel[len(_REPO_PREFIX) :]):
            return True
    for rel in trajectory.workspace_deletions:
        if rel.startswith(_REPO_PREFIX) and not _skip_repo_rel(rel[len(_REPO_PREFIX) :]):
            return True
    return False


def extract_patch_from_message(text: str) -> str:
    fenced = _FENCED_DIFF.search(text)
    if fenced is not None:
        return fenced.group(1).strip() + "\n"
    ix = text.find("diff --git ")
    if ix == -1:
        return ""
    return text[ix:].strip() + "\n"


def _apply_workspace_edits(trajectory: Trajectory, repo_dir: Path) -> set[str]:
    """Apply the trajectory's repo edits onto a fresh checkout.

    Returns the set of large (>1 MiB) changed files that the agent snapshot
    only retained as a `HASH:` digest and therefore could not be reconstructed.
    Such files are skipped rather than aborting extraction: a >1 MiB changed
    file in a SWE-bench solve is effectively always a generated artifact
    (minified bundle, lockfile, install state, vendored dep), never the source
    fix the hidden tests grade. Dropping it keeps the real source diff gradable;
    if the agent genuinely depended on it the task fails the tests anyway.
    """
    skipped: set[str] = set()
    for deletion in sorted(trajectory.workspace_deletions):
        if not deletion.startswith(_REPO_PREFIX):
            continue
        rel = deletion[len(_REPO_PREFIX) :]
        if _skip_repo_rel(rel):
            continue
        target = repo_dir / rel
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()

    for rel_path, content in trajectory.workspace_diff.items():
        if not rel_path.startswith(_REPO_PREFIX):
            continue
        rel = rel_path[len(_REPO_PREFIX) :]
        if _skip_repo_rel(rel):
            continue
        if content.startswith(b"HASH:"):
            skipped.add(rel)
            continue
        target = repo_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return skipped


# Generated dependency/build artifacts. These are side effects of running
# `yarn install`, building, or test runs — never part of a real code fix. The
# official SWE-bench grading diffs source only, so excluding them keeps the
# prediction patch matching gold-patch semantics (the >1 MiB ones are also
# dropped by the HASH-digest skip in _apply_workspace_edits, but small ones
# would otherwise still pollute the diff).
_ARTIFACT_DIRS = frozenset(
    {
        ".git",
        "node_modules",
        ".yarn",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
)
_ARTIFACT_SUFFIXES = (".pyc", ".pyo", ".tsbuildinfo")


def _skip_repo_rel(rel: str) -> bool:
    path = Path(rel)
    if not rel or rel.startswith("/") or ".." in path.parts:
        return True
    if _ARTIFACT_DIRS.intersection(path.parts):
        return True
    return path.name.endswith(_ARTIFACT_SUFFIXES)
