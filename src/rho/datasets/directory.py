from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from rho.protocols import Grade, Harness, HarnessStore, Task, TaskSet, Trajectory


@dataclass(frozen=True)
class DirectoryTask:
    _src: Path
    _harness: Harness

    @property
    def id(self) -> str:
        return self._src.name

    @property
    def harness(self) -> Harness:
        return self._harness

    @property
    def agent_timeout_s(self) -> float | None:
        return None

    def materialize(self, dest: Path) -> None:
        shutil.copytree(
            self._src,
            dest,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("expected.json"),
        )

    def query(self) -> str:
        return (self._src / "prompt.md").read_text(encoding="utf-8")

    def grade(
        self,
        trajectory: Trajectory,
        *,
        artifacts_dir: Path | None = None,
    ) -> Grade:
        spec_path = self._src / "expected.json"
        if not spec_path.exists():
            raise NotImplementedError(f"Task {self.id} has no expected.json")
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        return _grade_by_rules(trajectory, spec)


@dataclass(frozen=True)
class _DirectoryTaskSplit:
    _root: Path
    _split: str
    _harness: Harness

    @property
    def split(self) -> str:
        return self._split

    def __iter__(self) -> Iterator[Task]:
        split_dir = self._root / self._split
        for entry in sorted(split_dir.iterdir()):
            if entry.is_dir():
                yield DirectoryTask(entry, self._harness)

    def __len__(self) -> int:
        split_dir = self._root / self._split
        return sum(1 for entry in split_dir.iterdir() if entry.is_dir())


class DirectoryDataset:
    def __init__(self, root: Path, *, harness_store: HarnessStore) -> None:
        self._root = root
        self._harness = harness_store.empty()

    @property
    def train(self) -> TaskSet:
        return _DirectoryTaskSplit(self._root, "train", self._harness)

    @property
    def val(self) -> TaskSet:
        return _DirectoryTaskSplit(self._root, "val", self._harness)

    @property
    def test(self) -> TaskSet:
        return _DirectoryTaskSplit(self._root, "test", self._harness)


def _grade_by_rules(trajectory: Trajectory, spec: dict[str, Any]) -> Grade:
    rules = spec.get("rules", [])
    if not rules:
        return Grade(passed=True, score=1.0, details={"rule_results": []})

    results: list[dict[str, Any]] = []
    successes = 0
    text = trajectory.final_message
    for rule in rules:
        rule_type = rule["type"]
        ok = False
        if rule_type == "must_contain":
            ok = rule["value"] in text
        elif rule_type == "must_not_contain":
            ok = rule["value"] not in text
        elif rule_type == "regex":
            ok = re.search(rule["pattern"], text, flags=re.MULTILINE) is not None
        else:
            raise ValueError(f"Unknown grading rule type: {rule_type}")
        if ok:
            successes += 1
        results.append({"rule": rule, "ok": ok})
    return Grade(
        passed=successes == len(rules),
        score=successes / len(rules),
        details={"rule_results": results},
    )
