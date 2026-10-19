from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from rho.datasets.swebench_pro.evaluator import SWEbenchProDockerEvaluator
from rho.datasets.swebench_pro.patching import extract_prediction_patch
from rho.datasets.swebench_pro.repo_cache import CLEAN_HISTORY_CACHE_MODE, RepoCache
from rho.datasets.swebench_pro.util import read_rows_from_file, row_digest
from rho.protocols import Grade, Harness, HarnessStore, Task, TaskSet, Trajectory

_PROMPT_TEMPLATE = """\
# SWE-bench Pro Task

{task_id_label}: {public_task_id}
Repository: {repo}
Language: {repo_language}
{base_revision_label}: {public_base_revision}

The repository checkout is available at `repo/`.
Modify files under `repo/` to fix the issue. Do not edit this prompt.
Focus on production source changes; do not add or edit tests unless the
problem explicitly requires it. Hidden evaluator tests will be supplied
separately during grading.

## Problem Statement

{problem_statement}

## Requirements

{requirements}

## Interface Notes

{interface}

When you are done, leave your changes in `repo/` and summarize the fix in
your final response. The grading system will extract your repository diff.
"""


@dataclass(frozen=True)
class SWEbenchProTask:
    _row: dict[str, Any]
    _harness: Harness
    _repo_cache: RepoCache
    _docker_pull: str = "missing"
    _eval_assets_root: Path | None = None
    _grader_env_overrides: dict[str, str] | None = None

    @property
    def id(self) -> str:
        return str(self._row["instance_id"])

    @property
    def harness(self) -> Harness:
        return self._harness

    @property
    def row(self) -> dict[str, Any]:
        return dict(self._row)

    @property
    def agent_timeout_s(self) -> float | None:
        return None

    def materialize(self, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "prompt.md").write_text(self._prompt(), encoding="utf-8")
        self._materialize_repo(dest / "repo")

    def query(self) -> str:
        parts = []
        problem = str(self._row.get("problem_statement", "")).strip()
        requirements = str(self._row.get("requirements", "")).strip()
        if problem:
            parts.append(problem)
        if requirements:
            parts.append(requirements)
        return "\n\n".join(parts)

    def grade(
        self,
        trajectory: Trajectory,
        *,
        artifacts_dir: Path | None = None,
    ) -> Grade:
        if artifacts_dir is None:
            with tempfile.TemporaryDirectory(prefix="swebench_pro_grade_") as tmp:
                return self._grade(trajectory, Path(tmp))
        return self._grade(trajectory, artifacts_dir)

    def _grade(self, trajectory: Trajectory, artifacts_dir: Path) -> Grade:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        try:
            patch = extract_prediction_patch(
                trajectory,
                materialize_repo=self._materialize_repo,
                artifacts_dir=artifacts_dir / "patch_extract",
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return Grade(
                passed=False,
                score=0.0,
                details={
                    "instance_id": self.id,
                    "error": "patch_extraction_failed",
                    "message": str(exc),
                    "artifacts_dir": str(artifacts_dir),
                },
            )

        evaluator = SWEbenchProDockerEvaluator(
            docker_pull=self._docker_pull,
            assets_root=self._eval_assets_root,
            env_overrides=self._grader_env_overrides,
        )
        try:
            result = evaluator.evaluate(
                row=self._row,
                patch=patch,
                artifacts_dir=artifacts_dir / "docker",
            )
        except Exception as exc:
            return Grade(
                passed=False,
                score=0.0,
                details={
                    "instance_id": self.id,
                    "error": "docker_evaluation_failed",
                    "message": str(exc),
                    "artifacts_dir": str(artifacts_dir),
                },
            )
        return Grade(passed=result.passed, score=result.score, details=result.details)

    def _materialize_repo(self, dest: Path) -> None:
        self._repo_cache.materialize(self._row, dest)

    def _prompt(self) -> str:
        clean = self._repo_cache.cache_mode == CLEAN_HISTORY_CACHE_MODE
        public_task_id = self.id
        public_base_revision = str(self._row.get("base_commit", ""))
        task_id_label = "Instance ID"
        base_revision_label = "Base commit"
        if clean:
            public_task_id = "clean-task"
            public_base_revision = "clean-base"
            task_id_label = "Task ID"
            base_revision_label = "Base revision"
        return _PROMPT_TEMPLATE.format(
            task_id_label=task_id_label,
            public_task_id=public_task_id,
            repo=self._row.get("repo", ""),
            repo_language=self._row.get("repo_language", ""),
            base_revision_label=base_revision_label,
            public_base_revision=public_base_revision,
            problem_statement=self._row.get("problem_statement", ""),
            requirements=self._row.get("requirements", ""),
            interface=self._row.get("interface", ""),
        )


@dataclass(frozen=True)
class SWEbenchProTaskSet:
    _split: str
    _tasks: tuple[SWEbenchProTask, ...]

    @property
    def split(self) -> str:
        return self._split

    def __iter__(self) -> Iterator[Task]:
        return iter(self._tasks)

    def __len__(self) -> int:
        return len(self._tasks)


class SWEbenchProDataset:
    def __init__(
        self,
        source: str,
        *,
        harness_store: HarnessStore,
        max_per_split: int | None = None,
        docker_pull: str = "missing",
        seed: int = 0,
        repo_cache: RepoCache | None = None,
        eval_assets_root: Path | None = None,
        grader_env_overrides: dict[str, str] | None = None,
    ) -> None:
        rows = load_rows(source)
        self._init_from_rows(
            rows,
            harness_store=harness_store,
            max_per_split=max_per_split,
            docker_pull=docker_pull,
            seed=seed,
            repo_cache=repo_cache,
            eval_assets_root=eval_assets_root,
            grader_env_overrides=grader_env_overrides,
        )

    @classmethod
    def from_records(
        cls,
        rows: list[dict[str, Any]],
        *,
        harness_store: HarnessStore,
        max_per_split: int | None = None,
        docker_pull: str = "missing",
        seed: int = 0,
        repo_cache: RepoCache | None = None,
        eval_assets_root: Path | None = None,
        grader_env_overrides: dict[str, str] | None = None,
    ) -> "SWEbenchProDataset":
        dataset = cls.__new__(cls)
        dataset._init_from_rows(
            rows,
            harness_store=harness_store,
            max_per_split=max_per_split,
            docker_pull=docker_pull,
            seed=seed,
            repo_cache=repo_cache,
            eval_assets_root=eval_assets_root,
            grader_env_overrides=grader_env_overrides,
        )
        return dataset

    def _init_from_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        harness_store: HarnessStore,
        max_per_split: int | None,
        docker_pull: str,
        seed: int,
        repo_cache: RepoCache | None,
        eval_assets_root: Path | None,
        grader_env_overrides: dict[str, str] | None,
    ) -> None:
        self._harness = harness_store.empty()
        self._repo_cache = repo_cache or RepoCache()
        split_rows = _split_rows(rows, seed=seed)
        if max_per_split is not None:
            split_rows = {
                split: split_items[:max_per_split]
                for split, split_items in split_rows.items()
            }
        self._splits: dict[str, SWEbenchProTaskSet] = {}
        for split, split_items in split_rows.items():
            tasks = tuple(
                SWEbenchProTask(
                    _row=row,
                    _harness=self._harness,
                    _repo_cache=self._repo_cache,
                    _docker_pull=docker_pull,
                    _eval_assets_root=eval_assets_root,
                    _grader_env_overrides=dict(grader_env_overrides or {}),
                )
                for row in split_items
            )
            self._splits[split] = SWEbenchProTaskSet(_split=split, _tasks=tasks)

    @property
    def train(self) -> TaskSet:
        return self._splits["train"]

    @property
    def val(self) -> TaskSet:
        return self._splits["val"]

    @property
    def test(self) -> TaskSet:
        return self._splits["test"]


def load_rows(source: str) -> list[dict[str, Any]]:
    path = Path(source).expanduser()
    if path.exists():
        return read_rows_from_file(path.resolve())
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Loading SWE-bench Pro from Hugging Face requires the optional "
            "`swebench-pro` dependencies. Run `uv sync --extra swebench-pro` "
            "or `uv run --extra swebench-pro ...`."
        ) from exc
    return [dict(row) for row in load_dataset(source, split="test")]


_TRAIN_TARGET = 100
_VAL_TARGET = 500  # cap - leftover spills to test


def _split_rows(rows, *, seed):
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: row_digest({"seed": seed, "instance_id": row["instance_id"]}),
    )
    n = len(ordered)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    if n == 1:
        return {"train": ordered, "val": [], "test": []}
    if n == 2:
        return {"train": ordered[:1], "val": ordered[1:], "test": []}
    # Small-n fallback: when n is too small to honor the target, give all
    # three splits at least 1 row (matches fixture-test expectations).
    if n < _TRAIN_TARGET + 2:
        n_train = max(1, n - 2)
        n_val = 1
        n_test = n - n_train - n_val
    else:
        n_train = _TRAIN_TARGET
        n_val = min(_VAL_TARGET, n - n_train)
        n_test = n - n_train - n_val
    train = ordered[:n_train]
    val = ordered[n_train : n_train + n_val]
    test = ordered[n_train + n_val :]
    return {"train": train, "val": val, "test": test}
