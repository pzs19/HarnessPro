from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from rho.protocols import Trajectory, TrajectoryKind


@runtime_checkable
class Agent(Protocol):
    def run(
        self,
        workspace: Path,
        instructions: str,
        *,
        output_schema: dict[str, Any] | None = None,
        task_id: str = "",
        harness_id: str = "",
        kind: TrajectoryKind = "solve",
        timeout_s: float | None = None,
        env: dict[str, str] | None = None,
    ) -> Trajectory:
        """Execute one agentic session in the prepared workspace."""
