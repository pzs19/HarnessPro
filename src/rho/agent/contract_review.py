"""Opt-in independent completion review with explicit two-session accounting."""
from __future__ import annotations

import copy
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from rho.agent.base import Agent
from rho.protocols import Trajectory


def transport_compensation(trajectory: Trajectory) -> float:
    return sum(max(0.0, float(event.get("seconds") or 0)) for event in trajectory.events
               if event.get("type") == "transport_timeout_compensation")


def merge_sessions(first: Trajectory, review: Trajectory) -> Trajectory:
    """Preserve actual final workspace and both usages, even after review error."""
    diff = dict(first.workspace_diff)
    deletions = set(first.workspace_deletions)
    for path in review.workspace_deletions:
        diff.pop(path, None)
        deletions.add(path)
    for path, data in review.workspace_diff.items():
        diff[path] = data
        deletions.discard(path)
    events = [copy.deepcopy(e) for e in first.events
              if e.get("type") != "transport_timeout_compensation"]
    for event in review.events:
        if event.get("type") == "transport_timeout_compensation":
            continue
        event = copy.deepcopy(event)
        if isinstance(event.get("item"), dict):
            event["item"]["id"] = "contract-review:" + str(event["item"].get("id") or "anonymous")
        if isinstance(event.get("_rho_observed_at_s"), (int, float)):
            event["_rho_observed_at_s"] += first.wall_time_s
        event["contract_review_phase"] = True
        events.append(event)
    compensation = transport_compensation(first) + transport_compensation(review)
    if compensation:
        events.insert(0, {"type": "transport_timeout_compensation", "seconds": compensation})
    events.append({"type": "contract_review.completed", "initial_trajectory_id": first.id,
                   "review_trajectory_id": review.id, "review_timed_out": review.timed_out,
                   "review_exit_code": review.exit_code,
                   "initial_wall_time_s": first.wall_time_s, "review_wall_time_s": review.wall_time_s})
    return replace(first, id="traj_" + uuid4().hex[:10], events=events,
                   workspace_diff=diff, workspace_deletions=frozenset(deletions),
                   final_message=review.final_message or first.final_message,
                   stdout=first.stdout + "\n" + review.stdout,
                   stderr=first.stderr + "\n[contract-review]\n" + review.stderr,
                   wall_time_s=first.wall_time_s + review.wall_time_s,
                   # A failed review is never disguised as initial success.
                   exit_code=review.exit_code, timed_out=review.timed_out)


class ContractReviewAgent:
    def __init__(self, base: Agent, *, instructions: str, review_timeout_s: float,
                 skip_review: Callable[[Trajectory], bool] | None = None):
        if not instructions.strip() or not math.isfinite(review_timeout_s) or review_timeout_s <= 0:
            raise ValueError("review requires instructions and a positive finite timeout")
        self.base = base
        self.instructions = instructions
        self.review_timeout_s = review_timeout_s
        self.skip_review = skip_review

    def run(self, workspace: Path, instructions: str, *, output_schema=None,
            task_id: str = "", harness_id: str = "", kind: str = "solve",
            timeout_s: float | None = None, env: dict[str, str] | None = None) -> Trajectory:
        first = self.base.run(workspace, instructions, output_schema=output_schema,
                              task_id=task_id, harness_id=harness_id, kind=kind,
                              timeout_s=timeout_s, env=env)
        if output_schema is not None or kind != "solve" or first.exit_code or first.timed_out:
            return first
        if self.skip_review and self.skip_review(first):
            return first
        # A second session may spend only the original aggregate solve budget
        # minus completed active time. Never silently grant an extra deadline.
        allowance = float(timeout_s or getattr(self.base, "default_timeout_s", 0))
        first_compensation = min(transport_compensation(first),
                                 max(0.0, float((env or {}).get("RHO_TRANSPORT_MAX_COMPENSATION_S") or 0)))
        remaining = allowance - max(0.0, first.wall_time_s - first_compensation)
        if remaining <= 5:
            return first
        review_env = {**(env or {}), "RHO_INLINE_TASK_PROMPT": "0"}
        # Separate trace IDs prevent the second session's usage endpoint from
        # returning the first session's cumulative usage and counting it twice.
        trace_id = uuid4().hex
        review_env["RHO_TRANSPORT_TRACE_ID"] = trace_id
        raw_timing_file = review_env.get("RHO_TRANSPORT_TIME_FILE")
        if raw_timing_file:
            review_env["RHO_TRANSPORT_TIME_FILE"] = str(Path(raw_timing_file).with_name(trace_id + ".json"))
        maximum_compensation = max(0.0, float(review_env.get("RHO_TRANSPORT_MAX_COMPENSATION_S") or 0))
        review_env["RHO_TRANSPORT_MAX_COMPENSATION_S"] = str(max(0.0, maximum_compensation - first_compensation))
        task_prompt = (workspace / "task/prompt.md").read_text(encoding="utf-8").strip()
        prompt = self.instructions.strip() + "\n\n<task-description>\n" + task_prompt + "\n</task-description>"
        review = self.base.run(workspace, prompt, task_id=task_id, harness_id=harness_id,
                               kind=kind, timeout_s=min(remaining, self.review_timeout_s), env=review_env)
        merged = merge_sessions(first, review)
        merged.events[-1]["review_timeout_s"] = min(remaining, self.review_timeout_s)
        if raw_timing_file:
            timing_file = Path(review_env["RHO_TRANSPORT_TIME_FILE"])
            try:
                timing = json.loads(timing_file.read_text())
            except (OSError, ValueError):
                timing = {}
            merged.events[-1]["review_transport_timing"] = timing
            merged.events[-1]["review_transport_trace_id"] = trace_id
        return merged
