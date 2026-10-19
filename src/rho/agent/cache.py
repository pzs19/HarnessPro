from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from rho.agent.base import Agent
from rho.protocols import Trajectory, TrajectoryKind

FORMAT_VERSION = "1"
MAX_SNAPSHOT_FILE_BYTES = 1024 * 1024
HASH_SENTINEL = b"HASH:"


@dataclass(frozen=True)
class CachedResult:
    key_hex: str
    key_components: dict[str, Any]
    original_trajectory_id: str
    kind: TrajectoryKind
    events: list[dict[str, Any]]
    final_message: str
    stdout: str
    stderr: str
    workspace_diff: dict[str, bytes]
    workspace_deletions: frozenset[str]
    exit_code: int
    wall_time_s: float
    timed_out: bool


def workspace_digest(root: Path) -> str:
    """Return a deterministic digest for a materialized agent workspace.

    Materialized workspace bytes must be a pure function of task, harness, and
    orchestrator layout. Do not include temp paths, timestamps, UUIDs, or host
    identifiers in future materialization code; doing so makes this cache miss
    forever. The `.rho/` directory is excluded because CodexAgent writes
    transient execution metadata there during runs. `.git/` is excluded because
    git's internal files (index, pack files, ref logs) change nondeterministically
    across materializations of the same logical tree — hashing them would make
    every rerun a cache miss. The exclusion checks for `.git` / `.rho`
    at *any* depth (e.g. `task/repo/.git/...` for SWE-bench Pro), not just
    top-level — pre-2026-05-08 only top-level was excluded, which silently
    broke the cache for nested-repo tasks.
    """
    lines: list[str] = []
    excluded = {".rho", ".git"}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if any(part in excluded for part in rel.split("/")):
            continue
        size = path.stat().st_size
        if size > MAX_SNAPSHOT_FILE_BYTES:
            h = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(65536), b""):
                    h.update(chunk)
            lines.append(f"{rel}:HASH:{h.hexdigest()}")
        else:
            lines.append(f"{rel}:{hashlib.sha256(path.read_bytes()).hexdigest()}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


class AgentResponseCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def lookup(self, key_hex: str) -> CachedResult | None:
        target = self._entry_dir(key_hex)
        if not target.exists():
            return None
        try:
            manifest = json.loads(
                (target / "manifest.json").read_text(encoding="utf-8")
            )
            if manifest.get("format_version") != FORMAT_VERSION:
                return None
            result = json.loads((target / "result.json").read_text(encoding="utf-8"))
            events = _read_events(target / "events.jsonl")
            workspace_diff = _read_workspace_diff(target / "workspace_diff")
            return CachedResult(
                key_hex=str(manifest["key_hex"]),
                key_components=dict(manifest["key_components"]),
                original_trajectory_id=str(manifest["original_trajectory_id"]),
                kind=manifest["kind"],
                events=events,
                final_message=(target / "final_message.txt").read_text(
                    encoding="utf-8"
                ),
                stdout=(target / "stdout.log").read_text(encoding="utf-8"),
                stderr=(target / "stderr.log").read_text(encoding="utf-8"),
                workspace_diff=workspace_diff,
                workspace_deletions=frozenset(result.get("deletions", [])),
                exit_code=int(result["exit_code"]),
                wall_time_s=float(result["wall_time_s"]),
                timed_out=bool(result["timed_out"]),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            _log(f"WARNING unreadable entry key={key_hex[:12]} error={exc}")
            return None

    def store(
        self,
        key_hex: str,
        key_components: dict[str, Any],
        trajectory: Trajectory,
    ) -> None:
        shard_dir = self.root / f"v{FORMAT_VERSION}" / key_hex[:2]
        target = shard_dir / key_hex
        tmp_dir = shard_dir / f".cache_tmp_{uuid.uuid4().hex}"
        stale_dir = shard_dir / f".cache_stale_{uuid.uuid4().hex}"
        try:
            shard_dir.mkdir(parents=True, exist_ok=True)
            tmp_dir.mkdir()
            self._write_entry(tmp_dir, key_hex, key_components, trajectory)
            try:
                os.rename(tmp_dir, target)
                return
            except FileExistsError:
                pass
            except OSError as exc:
                if not target.exists():
                    raise
                _log(f"WARNING store race key={key_hex[:12]} error={exc}")

            try:
                os.rename(target, stale_dir)
                os.rename(tmp_dir, target)
            except OSError as exc:
                _log(f"WARNING store failed key={key_hex[:12]} error={exc}")
                if stale_dir.exists() and not target.exists():
                    try:
                        os.rename(stale_dir, target)
                    except OSError:
                        pass
            finally:
                shutil.rmtree(stale_dir, ignore_errors=True)
        except OSError as exc:
            _log(f"WARNING store failed key={key_hex[:12]} error={exc}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _entry_dir(self, key_hex: str) -> Path:
        return self.root / f"v{FORMAT_VERSION}" / key_hex[:2] / key_hex

    def _write_entry(
        self,
        dest: Path,
        key_hex: str,
        key_components: dict[str, Any],
        trajectory: Trajectory,
    ) -> None:
        manifest = {
            "format_version": FORMAT_VERSION,
            "key_hex": key_hex,
            "key_components": key_components,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "original_trajectory_id": trajectory.id,
            "kind": trajectory.kind,
        }
        result = {
            "exit_code": trajectory.exit_code,
            "wall_time_s": trajectory.wall_time_s,
            "timed_out": trajectory.timed_out,
            "deletions": sorted(trajectory.workspace_deletions),
        }
        (dest / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (dest / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (dest / "instructions.md").write_text(
            trajectory.instructions, encoding="utf-8"
        )
        with (dest / "events.jsonl").open("w", encoding="utf-8") as handle:
            for event in trajectory.events:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        (dest / "stdout.log").write_text(trajectory.stdout, encoding="utf-8")
        (dest / "stderr.log").write_text(trajectory.stderr, encoding="utf-8")
        (dest / "final_message.txt").write_text(
            trajectory.final_message, encoding="utf-8"
        )
        diff_dir = dest / "workspace_diff"
        diff_dir.mkdir()
        for rel, content in sorted(trajectory.workspace_diff.items()):
            _validate_rel(rel)
            output = diff_dir / rel
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(content)


class CachingAgent:
    def __init__(
        self,
        inner: Agent,
        cache: AgentResponseCache,
        *,
        mode: Literal["on", "readonly", "refresh"] = "on",
    ) -> None:
        if mode not in {"on", "readonly", "refresh"}:
            raise ValueError(f"invalid cache mode: {mode!r}")
        self.inner = inner
        self.cache = cache
        self.mode = mode
        self.hit_count = 0
        self.miss_count = 0
        self._counter_lock = threading.Lock()

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
        key_components = _collect_key_components(
            workspace, instructions, output_schema, kind, self.inner, env
        )
        key_hex = _key_hex(key_components)
        non_replayable_hit = False
        hit = None if self.mode == "refresh" else self.cache.lookup(key_hex)
        if hit is not None and hit.timed_out and timeout_s is not None and timeout_s > hit.wall_time_s + 1.0:
            _log(
                f"STALE(timed_out) key={key_hex[:12]} kind={kind} task={task_id} "
                f"prev_wall={hit.wall_time_s:.0f}s new_budget={timeout_s:.0f}s"
            )
            hit = None
        if hit is not None:
            if _replayable(hit):
                _replay_workspace(workspace, hit)
                with self._counter_lock:
                    self.hit_count += 1
                _log(f"HIT key={key_hex[:12]} kind={kind} task={task_id}")
                return _trajectory_from_cache(hit, task_id, harness_id, instructions)
            non_replayable_hit = True
            _log(f"WARNING non-replayable entry key={key_hex[:12]} kind={kind}")

        _log(f"MISS key={key_hex[:12]} kind={kind} task={task_id}")
        traj = self.inner.run(
            workspace,
            instructions,
            output_schema=output_schema,
            task_id=task_id,
            harness_id=harness_id,
            kind=kind,
            timeout_s=timeout_s,
            env=env,
        )
        if self.mode in {"on", "refresh"} and not non_replayable_hit:
            self.cache.store(key_hex, key_components, traj)
        with self._counter_lock:
            self.miss_count += 1
        return traj


def build_default_agent(
    inner: Agent,
    *,
    mode: str = "off",
    cache_dir: Path | None = None,
) -> Agent:
    if getattr(inner, "__rho_bypass_cache__", False):
        _log("cache bypassed for agent marker")
        return inner
    if mode == "off":
        _log("cache mode=off")
        return inner
    if mode not in {"on", "readonly", "refresh"}:
        raise ValueError("cache mode must be one of: on, off, readonly, refresh")
    if cache_dir is None:
        raise ValueError("cache_dir is required when cache mode is not off")
    _log(f"cache mode={mode} dir={cache_dir}")
    return CachingAgent(inner, AgentResponseCache(cache_dir), mode=mode)


def _collect_key_components(
    workspace: Path,
    instructions: str,
    output_schema: dict[str, Any] | None,
    kind: TrajectoryKind,
    inner: Agent,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    env_signature = getattr(inner, "cache_env_signature", None)
    if env_signature is None:
        env_signature = {"agent_class": type(inner).__name__}
    return {
        "format_version": FORMAT_VERSION,
        "kind": kind,
        "instructions_sha256": hashlib.sha256(
            instructions.encode("utf-8")
        ).hexdigest(),
        "output_schema_sha256": _schema_sha256(output_schema),
        "workspace_digest": workspace_digest(workspace),
        "call_env": _normalize_call_env(env, workspace),
        "env_signature": env_signature,
    }


def _normalize_call_env(
    env: dict[str, str] | None, workspace: Path
) -> dict[str, str]:
    """Normalize env values for cache-key stability.

    env values that are absolute paths inside the workspace get the workspace
    prefix replaced by '<ws>'. This is necessary because callers (e.g.
    LettaSleepStrategy) pass workspace-relative absolute paths as env values
    (e.g. LETTA_MEMORY_ROOT=/tmp/.../letta_s0_t0000_xxx/harness/letta_memory),
    and the tempdir prefix changes every run. Workspace contents are already
    covered by workspace_digest, so two runs with identical workspace contents
    and env values pointing to the same relative location must hash identically.
    """
    if not env:
        return {}
    ws_str = str(workspace.resolve())
    normalized: dict[str, str] = {}
    for key, value in env.items():
        s = str(value)
        if s == ws_str:
            s = "<ws>"
        elif s.startswith(ws_str + "/"):
            s = "<ws>" + s[len(ws_str):]
        normalized[key] = s
    return dict(sorted(normalized.items()))


def _schema_sha256(output_schema: dict[str, Any] | None) -> str:
    if output_schema is None:
        return ""
    payload = json.dumps(output_schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _key_hex(key_components: dict[str, Any]) -> str:
    payload = json.dumps(
        key_components, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _read_workspace_diff(diff_dir: Path) -> dict[str, bytes]:
    workspace_diff: dict[str, bytes] = {}
    if not diff_dir.exists():
        return workspace_diff
    for path in sorted(diff_dir.rglob("*")):
        if path.is_file():
            workspace_diff[path.relative_to(diff_dir).as_posix()] = path.read_bytes()
    return workspace_diff


def _replayable(hit: CachedResult) -> bool:
    return all(not content.startswith(HASH_SENTINEL) for content in hit.workspace_diff.values())


def _replay_workspace(workspace: Path, hit: CachedResult) -> None:
    for rel, content in hit.workspace_diff.items():
        _validate_rel(rel)
        target = workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    for rel in hit.workspace_deletions:
        _validate_rel(rel)
        (workspace / rel).unlink(missing_ok=True)


def _trajectory_from_cache(
    hit: CachedResult,
    task_id: str,
    harness_id: str,
    instructions: str,
) -> Trajectory:
    return Trajectory(
        id=f"traj_{uuid.uuid4().hex[:10]}",
        kind=hit.kind,
        task_id=task_id,
        harness_id=harness_id,
        instructions=instructions,
        events=list(hit.events),
        final_message=hit.final_message,
        stdout=hit.stdout,
        stderr=hit.stderr,
        workspace_diff=dict(hit.workspace_diff),
        workspace_deletions=frozenset(hit.workspace_deletions),
        exit_code=hit.exit_code,
        wall_time_s=hit.wall_time_s,
        timed_out=hit.timed_out,
    )


def _validate_rel(rel: str) -> None:
    path = Path(rel)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe cache path: {rel!r}")


def _log(message: str) -> None:
    print(f"[rho cache] {message}", file=sys.stderr)
