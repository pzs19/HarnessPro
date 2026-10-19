from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from rho.agent.base import Agent

TrajectoryKind = Literal["solve", "evaluate", "optimize", "diagnose"]


@dataclass(frozen=True)
class Trajectory:
    id: str
    kind: TrajectoryKind
    task_id: str
    harness_id: str
    instructions: str
    events: list[dict[str, Any]]
    final_message: str
    stdout: str
    stderr: str
    workspace_diff: dict[str, bytes]
    workspace_deletions: frozenset[str]
    exit_code: int
    wall_time_s: float
    timed_out: bool = False
    stage: str | None = None
    round_ix: int | None = None
    sample_index: int | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    cache_mode: str | None = None


@dataclass(frozen=True)
class Score:
    value: int
    rationale: str


@dataclass(frozen=True)
class TrajectoryAnalysis:
    trajectory: str
    successful: int
    quality_analysis: str
    issues: str


@dataclass(frozen=True)
class Diagnosis:
    task_id: str
    trajectory_analyses: list[TrajectoryAnalysis]
    failure_mode_analysis: str
    inconsistency_analysis: str
    harness_improvement_direction: str
    severity: float = 1.0


@dataclass(frozen=True)
class Grade:
    passed: bool
    score: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizeSample:
    sample_index: int
    optimize_trajectory: Trajectory
    candidate: "Harness | None"


@dataclass
class OptimizeStrategyResult:
    samples: list[OptimizeSample]
    diagnose_trajectories: list[Trajectory] | None = None
    diagnoses: list[Diagnosis] | None = None


@runtime_checkable
class Harness(Protocol):
    @property
    def id(self) -> str: ...

    def materialize(self, dest: Path) -> None: ...


@runtime_checkable
class Task(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def harness(self) -> Harness: ...

    def materialize(self, dest: Path) -> None: ...

    def query(self) -> str:
        """Return the task's natural-language query, without dataset metadata."""
        ...

    def grade(
        self,
        trajectory: Trajectory,
        *,
        artifacts_dir: Path | None = None,
    ) -> Grade: ...

    @property
    def agent_timeout_s(self) -> float | None:
        """Per-task wall-time budget for agent.run. None = use agent default."""
        ...


@runtime_checkable
class TaskSet(Protocol):
    @property
    def split(self) -> str: ...

    def __iter__(self) -> Iterator[Task]: ...

    def __len__(self) -> int: ...


@runtime_checkable
class TaskSelector(Protocol):
    def select(
        self,
        candidates: list[Task],
        k: int | None,
        *,
        seed: int | None = None,
    ) -> list[Task]:
        """Return up to k tasks from candidates. k=None returns all (possibly reordered)."""


@runtime_checkable
class Dataset(Protocol):
    @property
    def train(self) -> TaskSet: ...

    @property
    def val(self) -> TaskSet: ...

    @property
    def test(self) -> TaskSet: ...


@runtime_checkable
class HarnessStore(Protocol):
    def empty(self) -> Harness: ...

    def capture(self, src: Path) -> Harness: ...

    def get(self, harness_id: str) -> Harness: ...


@runtime_checkable
class TrajectoryStore(Protocol):
    def put(self, trajectory: Trajectory) -> None: ...

    def get(self, traj_id: str) -> Trajectory: ...

    def list_for_task(self, task_id: str) -> Iterator[Trajectory]: ...


@runtime_checkable
class OptimizeStrategy(Protocol):
    def propose_candidates(
        self,
        *,
        agent: "Agent",
        harness: Harness,
        tasks_with_trajectories: list[tuple[Task, list[Trajectory]]],
        harness_store: HarnessStore,
        traj_store: TrajectoryStore,
        workdir: Path,
        n_samples: int,
        round_ix: int,
    ) -> OptimizeStrategyResult:
        """Produce optimize candidates from solve trajectories."""
