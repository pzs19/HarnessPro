from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Iterator

from rho.observability import is_runtime_scratch
from rho.protocols import Trajectory


class FilesystemTrajectoryStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, trajectory: Trajectory) -> None:
        target = self.root / trajectory.id
        if target.exists():
            raise FileExistsError(f"Trajectory already exists: {trajectory.id}")

        with tempfile.TemporaryDirectory(dir=str(self.root), prefix=".traj_") as tmp:
            temp_dir = Path(tmp) / trajectory.id
            temp_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "id": trajectory.id,
                "kind": trajectory.kind,
                "task_id": trajectory.task_id,
                "harness_id": trajectory.harness_id,
                "exit_code": trajectory.exit_code,
                "timed_out": trajectory.timed_out,
                "wall_time_s": trajectory.wall_time_s,
                "deletions": sorted(
                    rel for rel in trajectory.workspace_deletions if not is_runtime_scratch(rel)
                ),
                "stage": trajectory.stage,
                "round_ix": trajectory.round_ix,
                "sample_index": trajectory.sample_index,
                "model": trajectory.model,
                "reasoning_effort": trajectory.reasoning_effort,
                "cache_mode": trajectory.cache_mode,
            }
            (temp_dir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (temp_dir / "instructions.md").write_text(
                trajectory.instructions,
                encoding="utf-8",
            )
            with (temp_dir / "events.jsonl").open("w", encoding="utf-8") as handle:
                for event in trajectory.events:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            (temp_dir / "stdout.log").write_text(trajectory.stdout, encoding="utf-8")
            (temp_dir / "stderr.log").write_text(trajectory.stderr, encoding="utf-8")
            (temp_dir / "final_message.txt").write_text(
                trajectory.final_message,
                encoding="utf-8",
            )
            diff_dir = temp_dir / "workspace_diff"
            diff_dir.mkdir()
            for rel, content in trajectory.workspace_diff.items():
                if is_runtime_scratch(rel):
                    continue
                output = diff_dir / rel
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(content)
            temp_dir.rename(target)

    def get(self, traj_id: str) -> Trajectory:
        base = self.root / traj_id
        if not base.exists():
            raise KeyError(traj_id)
        meta = json.loads((base / "meta.json").read_text(encoding="utf-8"))
        events: list[dict] = []
        events_path = base / "events.jsonl"
        if events_path.exists():
            for line in events_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    events.append(json.loads(line))
        workspace_diff: dict[str, bytes] = {}
        diff_dir = base / "workspace_diff"
        if diff_dir.exists():
            for path in diff_dir.rglob("*"):
                if path.is_file():
                    workspace_diff[str(path.relative_to(diff_dir))] = path.read_bytes()
        return Trajectory(
            id=meta["id"],
            kind=meta["kind"],
            task_id=meta["task_id"],
            harness_id=meta["harness_id"],
            instructions=(base / "instructions.md").read_text(encoding="utf-8"),
            events=events,
            final_message=(base / "final_message.txt").read_text(encoding="utf-8"),
            stdout=(base / "stdout.log").read_text(encoding="utf-8"),
            stderr=(base / "stderr.log").read_text(encoding="utf-8"),
            workspace_diff=workspace_diff,
            workspace_deletions=frozenset(meta.get("deletions", [])),
            exit_code=meta["exit_code"],
            wall_time_s=float(meta["wall_time_s"]),
            timed_out=bool(meta["timed_out"]),
            stage=meta.get("stage"),
            round_ix=meta.get("round_ix"),
            sample_index=meta.get("sample_index"),
            model=meta.get("model"),
            reasoning_effort=meta.get("reasoning_effort"),
            cache_mode=meta.get("cache_mode"),
        )

    def list_for_task(self, task_id: str) -> Iterator[Trajectory]:
        for traj in self._iter_all():
            if traj.task_id == task_id:
                yield traj

    def list_by_kind(self, kind: str) -> Iterator[Trajectory]:
        for traj in self._iter_all():
            if traj.kind == kind:
                yield traj

    def _iter_all(self) -> Iterator[Trajectory]:
        for entry in sorted(self.root.iterdir()):
            if entry.is_dir() and not entry.name.startswith("."):
                yield self.get(entry.name)
