from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from rho.datasets.terminal_bench_2 import grader
from rho.datasets.terminal_bench_2.prompts import render_prompt
from rho.datasets.terminal_bench_2.public_inputs import public_input_notice, stage_public_inputs
from rho.datasets.terminal_bench_2.runtime import (
    RuntimeHandle,
    TerminalBench2RuntimeSession,
    resolve_agent_timeout,
    resolve_verifier_timeout,
)
from rho.datasets.terminal_bench_2.splits import split_task_ids
from rho.datasets.terminal_bench_2.task_toml import TaskToml, load_task_toml
from rho.protocols import Grade, Harness, HarnessStore, Task, TaskSet, Trajectory


@dataclass(frozen=True)
class TerminalBench2Task:
    _id: str
    _task_dir: Path
    _config: TaskToml
    _harness: Harness
    _docker_pull: str

    @property
    def id(self) -> str:
        return self._id

    @property
    def harness(self) -> Harness:
        return self._harness

    @property
    def agent_timeout_s(self) -> float:
        return resolve_agent_timeout(self._config.agent_timeout_sec)

    @property
    def verifier_timeout_s(self) -> float:
        return resolve_verifier_timeout(self._config.verifier_timeout_sec)

    def materialize(self, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        instruction = (self._task_dir / "instruction.md").read_text(encoding="utf-8")
        stage_public_inputs(instruction, dest)
        prompt = render_prompt(
            task_id=self._id,
            difficulty=self._config.difficulty,
            category=self._config.category,
            container_name="<container-not-started-yet>",
            agent_timeout_sec=self.agent_timeout_s,
            verifier_timeout_sec=self.verifier_timeout_s,
            instruction_md=instruction,
        )
        (dest / "prompt.md").write_text(prompt + public_input_notice(dest), encoding="utf-8")
        (dest / ".tb2").mkdir(exist_ok=True)

    def query(self) -> str:
        return (self._task_dir / "instruction.md").read_text(encoding="utf-8")

    def grade(
        self,
        trajectory: Trajectory,
        *,
        artifacts_dir: Path | None = None,
    ) -> Grade:
        from rho.datasets.terminal_bench_2.runtime import enter_grading_phase, get_active

        container_name = get_active(self._id)
        if container_name is not None:
            enter_grading_phase(self._id)
        if container_name is None:
            container_name = _locate_container_name(artifacts_dir)
        if container_name is None:
            art = artifacts_dir or Path(".")
            art.mkdir(parents=True, exist_ok=True)
            return Grade(
                passed=False,
                score=0.0,
                details={
                    "error": "no_runtime",
                    "message": "No active runtime session for this task; grade() was called outside `with task.runtime_session(...)`.",
                    "artifacts_dir": str(art),
                },
            )
        if artifacts_dir is None:
            import tempfile

            with tempfile.TemporaryDirectory(prefix="tb2_grade_") as tmp:
                return grader.run_tests(
                    container_name,
                    self._task_dir,
                    artifacts_dir=Path(tmp),
                    verifier_timeout_s=self.verifier_timeout_s,
                )
        return grader.run_tests(
            container_name,
            self._task_dir,
            artifacts_dir=artifacts_dir,
            verifier_timeout_s=self.verifier_timeout_s,
        )

    def runtime_session(self, workdir: Path) -> AbstractContextManager[RuntimeHandle]:
        return _RuntimeSessionWithPromptRewrite(self, workdir)


class _RuntimeSessionWithPromptRewrite:
    """Rewrite prompt.md with the real container name after the container starts."""

    def __init__(self, task: TerminalBench2Task, workdir: Path) -> None:
        self._inner = TerminalBench2RuntimeSession(task, workdir)
        self._task = task
        self._workdir = Path(workdir)

    def __enter__(self) -> RuntimeHandle:
        handle = self._inner.__enter__()
        from rho.datasets.terminal_bench_2.runtime import set_active

        set_active(self._task.id, handle.container_name, self._inner)
        prompt_path = self._workdir / "prompt.md"
        if prompt_path.exists():
            instruction = (self._task._task_dir / "instruction.md").read_text(encoding="utf-8")
            real_prompt = render_prompt(
                task_id=self._task.id,
                difficulty=self._task._config.difficulty,
                category=self._task._config.category,
                container_name=handle.container_name,
                agent_timeout_sec=self._task.agent_timeout_s,
                verifier_timeout_sec=self._task.verifier_timeout_s,
                instruction_md=instruction,
            )
            prompt_path.write_text(real_prompt + public_input_notice(self._workdir), encoding="utf-8")
        return handle

    def __exit__(self, *args) -> None:
        from rho.datasets.terminal_bench_2.runtime import clear_active

        clear_active(self._task.id)
        self._inner.__exit__(*args)

    @property
    def timed_out(self) -> bool:
        return self._inner.timed_out


def _locate_container_name(artifacts_dir: Path | None) -> str | None:
    if artifacts_dir is None:
        return None
    for parent in [artifacts_dir, *artifacts_dir.parents]:
        candidate = parent / "task" / ".tb2" / "container_name"
        if candidate.exists():
            return candidate.read_text(encoding="utf-8").strip()
        candidate2 = parent / ".tb2" / "container_name"
        if candidate2.exists():
            return candidate2.read_text(encoding="utf-8").strip()
    return None


@dataclass(frozen=True)
class TerminalBench2TaskSet:
    _split: str
    _tasks: tuple[TerminalBench2Task, ...]

    @property
    def split(self) -> str:
        return self._split

    def __iter__(self) -> Iterator[Task]:
        return iter(self._tasks)

    def __len__(self) -> int:
        return len(self._tasks)


class TerminalBench2Dataset:
    def __init__(
        self,
        repo_path: Path,
        *,
        harness_store: HarnessStore,
        max_per_split: int | None = None,
        docker_pull: str = "missing",
        seed: int = 0,
        difficulty_filter: tuple[str, ...] | None = None,
    ) -> None:
        repo_path = Path(repo_path).resolve()
        if not repo_path.exists():
            raise FileNotFoundError(
                f"Terminal-Bench 2 repo path {repo_path} does not exist. "
                f"Clone with: git clone https://github.com/harbor-framework/terminal-bench-2 {repo_path} "
                f"&& (cd {repo_path} && git checkout 53ff2b87d621bdb97b455671f2bd9728b7d86c11)"
            )
        task_dirs = [
            path
            for path in sorted(repo_path.iterdir())
            if path.is_dir()
            and (path / "task.toml").exists()
            and (path / "environment" / "Dockerfile").exists()
            and (path / "tests" / "test.sh").exists()
        ]
        if not task_dirs:
            raise ValueError(
                f"No TB2-shaped task directories found under {repo_path}. "
                f"Expected subdirs with task.toml + environment/Dockerfile + tests/test.sh."
            )
        self._harness = harness_store.empty()
        tasks: list[TerminalBench2Task] = []
        for task_dir in task_dirs:
            cfg = load_task_toml(task_dir / "task.toml")
            if difficulty_filter is not None and cfg.difficulty not in difficulty_filter:
                continue
            tasks.append(
                TerminalBench2Task(
                    _id=task_dir.name,
                    _task_dir=task_dir,
                    _config=cfg,
                    _harness=self._harness,
                    _docker_pull=docker_pull,
                )
            )
        ids = [task.id for task in tasks]
        id_to_task = {task.id: task for task in tasks}
        split_ids = split_task_ids(ids, seed=seed)
        if max_per_split is not None:
            split_ids = {key: value[:max_per_split] for key, value in split_ids.items()}
        self._splits: dict[str, TerminalBench2TaskSet] = {
            key: TerminalBench2TaskSet(
                _split=key,
                _tasks=tuple(id_to_task[task_id] for task_id in value),
            )
            for key, value in split_ids.items()
        }

    @property
    def train(self) -> TaskSet:
        return self._splits["train"]

    @property
    def val(self) -> TaskSet:
        return self._splits["val"]

    @property
    def test(self) -> TaskSet:
        return self._splits["test"]
