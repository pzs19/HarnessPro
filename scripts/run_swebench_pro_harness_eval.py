#!/usr/bin/env python3
"""Resumable, infrastructure-aware SWE-bench Pro held-out evaluation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import tomllib
from urllib.parse import urlsplit

from rho.agent.anthropic_messages import AnthropicMessagesAgent
from rho.agent.base import Agent
from rho.agent.codex import CodexAgent
from rho.agent.codex_pool import configure_global_codex_pool
from rho.agent.external import ClaudeCodeAgent
from rho.agent.responses import ResponsesAgent
from rho.datasets.swebench_pro.dataset import SWEbenchProDataset, load_rows
from rho.datasets.swebench_pro.evaluator import (
    SWEbenchProDockerEvaluator,
    docker_image,
    ensure_official_assets,
)
from rho.datasets.swebench_pro.repo_cache import (
    CLEAN_HISTORY_CACHE_MODE,
    FULL_HISTORY_CACHE_DIR,
    FULL_HISTORY_CACHE_MODE,
    CleanRepoCache,
    RepoCache,
)
from rho.observability import (
    annotate_trajectory,
    call_metrics,
    extract_api_calls,
    extract_usage,
    retryable_transport_failure,
    trajectory_metrics,
    usage_summary,
)
from rho.harness_runtime import (
    HarnessRuntimePolicy,
    ProfileSelection,
    load_runtime_policy,
    render_runtime_codex_config,
    write_workspace_selection,
)
from rho.orchestrators.solve import (
    RUNTIME_HARNESS_SOLVE_INSTRUCTIONS,
    SOLVE_INSTRUCTIONS,
    solve_in,
    solve_workspace,
)
from rho.protocols import Trajectory
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore


TRANSPORT_MARKERS = (
    "stream disconnected before completion",
    "response.failed event received",
    "connection error",
    "connection reset",
    "connection refused",
    "connection closed",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "too many requests",
    "rate limit",
    "http 429",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
)

CLEAN_SOLVER_PERMISSION_PROFILE = "swebench-clean"
STALLED_UPSTREAM_THRESHOLD_S = 300.0
INFRA_TEXT_MARKERS = (
    "cannot connect to the docker daemon",
    "error response from daemon",
    "docker pull failed",
    "manifest unknown",
    "no space left on device",
    "temporary failure in name resolution",
    "could not resolve host",
    "network is unreachable",
    "connection timed out",
    "connection reset",
    "connection refused",
    "tls handshake timeout",
    "unexpected eof",
    "remote end hung up",
    "failed to clone",
    "failed to fetch",
    "repository not found",
    "ssl certificate problem",
)
GRADER_DEPENDENCY_TRANSPORT_PATTERNS = (
    re.compile(
        r'Get "https?://[^"\s]+": (?:net/http: )?(?:TLS handshake timeout|i/o timeout)',
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:npm|yarn|pnpm|pip|poetry|maven|gradle).*"
        r"(?:network|download).*(?:timeout|connection reset|temporary failure)",
        re.IGNORECASE,
    ),
)
SOLVER_DEPENDENCY_TRANSPORT_PATTERNS = (
    re.compile(
        r'(?:go: .*?)?Get "https?://[^"\s]+": '
        r"(?:net/http: )?(?:TLS handshake timeout|i/o timeout|unexpected EOF|"
        r"Bad Gateway|Service Unavailable|Gateway Timeout)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:proxy\.golang\.org|sum\.golang\.org|go\.dev/dl|golang\.google\.)"
        r".*(?:TLS handshake timeout|i/o timeout|connection reset|unexpected EOF|"
        r"Bad Gateway|Service Unavailable|Gateway Timeout)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:ReadTimeoutError|ConnectTimeoutError|ConnectionError).*"
        r"(?:pypi|pythonhosted|pip)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:npm|yarn|pnpm|maven|gradle|cargo).*"
        r"(?:fetch|download|request|network).*"
        r"(?:timeout|connection reset|temporary failure|unexpected EOF)",
        re.IGNORECASE,
    ),
)
GO_VERSION_PATTERN = re.compile(
    r"^go version go(?P<version>[0-9]+\.[0-9]+(?:\.[0-9]+)?)\s+",
    re.MULTILINE,
)


FRESH_REVIEW_INSTRUCTIONS = """
Perform one fresh, bounded completion review of the task in task/prompt.md.
The prior solver already worked in this same workspace. The workspace root
contains task/ and harness/; the repository checkout is task/repo/. Work from
task/repo/ and do not spend calls rediscovering this layout or inventorying
harness/. Do not restart broad exploration. Inspect the concrete patch and
current repository state, compare
exact public names, signatures, data shapes, state transitions, compatibility
requirements, negative cases, and cleanup behavior with the public task, repair
only mismatches supported by concrete evidence, and run one focused validation.
Use at most {review_tool_budget} tool calls. Keep the best final patch in place and finish
concisely.
"""

INDEPENDENT_SOLVE_INSTRUCTIONS = """
Produce an independent solution for the task in task/prompt.md. The repository
has been restored to its original base commit, so do not assume or reconstruct
the previous solver's approach. Work in task/repo/. Derive the implementation
from the public requirements and repository evidence, implement the complete
fix, and run focused validation. Use at most {review_tool_budget} tool calls.
Leave your best patch in task/repo/ and finish concisely.
"""

CANDIDATE_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {"type": "string", "enum": ["first", "second"]},
        "rationale": {"type": "string"},
    },
    "required": ["choice", "rationale"],
    "additionalProperties": False,
}


class _FreshReviewAgent:
    """Run one independent review after a solve with a risky tool-count shape."""

    def __init__(
        self,
        base: Agent,
        *,
        min_tools: int,
        max_tools: int,
        timeout_s: float,
        profiles: tuple[str, ...],
        review_tool_budget: int = 8,
        profile_tool_ranges: tuple[tuple[str, int, int], ...] = (),
        review_agent: Agent | None = None,
        mode: str = "in_place",
        judge_agent: Agent | None = None,
        review_instructions: str | None = None,
    ):
        self.base = base
        self.review_agent = review_agent or base
        self.judge_agent = judge_agent or self.review_agent
        self.min_tools = min_tools
        self.max_tools = max_tools
        self.timeout_s = timeout_s
        self.review_tool_budget = review_tool_budget
        self.profiles = frozenset(profiles)
        self.profile_tool_ranges = {
            profile: (range_min, range_max)
            for profile, range_min, range_max in profile_tool_ranges
        }
        self.mode = mode
        self.review_instructions = review_instructions

    def run(
        self,
        workspace: Path,
        instructions: str,
        *,
        output_schema: dict[str, Any] | None = None,
        task_id: str = "",
        harness_id: str = "",
        kind: str = "solve",
        timeout_s: float | None = None,
        env: dict[str, str] | None = None,
    ) -> Trajectory:
        repo = workspace / "task" / "repo"
        base_head = _git_head(repo) if self.mode == "independent_select" else None
        first = self.base.run(
            workspace,
            instructions,
            output_schema=output_schema,
            task_id=task_id,
            harness_id=harness_id,
            kind=kind,
            timeout_s=timeout_s,
            env=env,
        )
        if output_schema is not None or kind != "solve" or first.exit_code != 0 or first.timed_out:
            return first
        selection_path = workspace / ".rho" / "runtime-selection.json"
        try:
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return first
        profile = str(selection.get("profile") or "")
        if profile not in self.profiles:
            # Confidence overrides have their own execution label (for example
            # ``uncertain_lean_to_standard``), while fresh-review policy ranges
            # are deliberately expressed only in terms of public base profiles.
            # Fall back to the classifier's base profile so overridden sessions
            # remain eligible for a bounded review.
            classified_profile = str(selection.get("classified_profile") or "")
            if classified_profile in self.profiles:
                profile = classified_profile
        if profile not in self.profiles:
            return first
        tool_count = int(
            trajectory_metrics(first).get("tools", {}).get("tool_call_count") or 0
        )
        range_min, range_max = self.profile_tool_ranges.get(
            profile,
            (self.min_tools, self.max_tools),
        )
        if not range_min <= tool_count <= range_max:
            return first

        if self.mode == "independent_select" and base_head is not None:
            return self._run_independent_select(
                workspace,
                first,
                base_head=base_head,
                task_id=task_id,
                harness_id=harness_id,
                timeout_s=timeout_s,
                env=env,
                tool_count=tool_count,
            )

        review = self.review_agent.run(
            workspace,
            _fresh_review_instructions(
                workspace,
                self.review_tool_budget,
                self.review_instructions,
            ),
            task_id=task_id,
            harness_id=harness_id,
            kind=kind,
            timeout_s=min(timeout_s, self.timeout_s) if timeout_s else self.timeout_s,
            env={
                **(env or {}),
                "RHO_INLINE_TASK_PROMPT": "0",
                "RHO_FRESH_REVIEW": "1",
                "RHO_FRESH_REVIEW_MAX_TOOLS": str(self.review_tool_budget),
            },
        )
        return _merge_review_trajectories(first, review, tool_count=tool_count)

    def _run_independent_select(
        self,
        workspace: Path,
        first: Trajectory,
        *,
        base_head: str,
        task_id: str,
        harness_id: str,
        timeout_s: float | None,
        env: dict[str, str] | None,
        tool_count: int,
    ) -> Trajectory:
        repo = workspace / "task" / "repo"
        _write_candidate_bundle(workspace, "first", first)
        try:
            _reset_task_repo(repo, base_head)
            second = self.review_agent.run(
                workspace,
                _independent_solve_instructions(
                    workspace, self.review_tool_budget
                ),
                task_id=task_id,
                harness_id=harness_id,
                kind="solve",
                timeout_s=(
                    min(timeout_s, self.timeout_s)
                    if timeout_s
                    else self.timeout_s
                ),
                env={
                    **(env or {}),
                    "RHO_INLINE_TASK_PROMPT": "0",
                    "RHO_FRESH_REVIEW": "1",
                    "RHO_FRESH_REVIEW_RESET_TOOLS": "1",
                    "RHO_FRESH_REVIEW_MAX_TOOLS": str(self.review_tool_budget),
                },
            )
            _write_candidate_bundle(workspace, "second", second)
            if second.exit_code != 0 or second.timed_out:
                _reset_task_repo(repo, base_head)
                _apply_candidate(workspace, first)
                return _merge_candidate_trajectories(
                    first,
                    second,
                    None,
                    choice="first",
                    tool_count=tool_count,
                    reason="independent candidate did not complete",
                )

            try:
                judge_kwargs = {
                    "task_id": task_id,
                    "harness_id": harness_id,
                    "kind": "diagnose",
                    "timeout_s": min(self.timeout_s, 240.0),
                    "env": {
                        **(env or {}),
                        "RHO_INLINE_TASK_PROMPT": "0",
                        "RHO_FRESH_REVIEW": "1",
                    },
                }
                judge_instructions = _candidate_judge_instructions(
                    workspace, first, second
                )
                try:
                    judge = self.judge_agent.run(
                        workspace,
                        judge_instructions,
                        output_schema=CANDIDATE_JUDGE_SCHEMA,
                        **judge_kwargs,
                    )
                except ValueError as exc:
                    if "does not support output_schema" not in str(exc):
                        raise
                    judge = self.judge_agent.run(
                        workspace,
                        judge_instructions,
                        **judge_kwargs,
                    )
                choice, reason = _candidate_choice(judge)
            except Exception as exc:
                # Candidate generation already completed and remains billable
                # and auditable even when the selector itself cannot start.
                judge = None
                choice = "first"
                reason = f"candidate judge failed safely: {exc}"
            _reset_task_repo(repo, base_head)
            _apply_candidate(workspace, first if choice == "first" else second)
            return _merge_candidate_trajectories(
                first,
                second,
                judge,
                choice=choice,
                tool_count=tool_count,
                reason=reason,
            )
        except Exception as exc:
            _reset_task_repo(repo, base_head)
            _apply_candidate(workspace, first)
            return _merge_candidate_trajectories(
                first,
                None,
                None,
                choice="first",
                tool_count=tool_count,
                reason=f"independent selection failed safely: {exc}",
            )


def _fresh_review_instructions(
    workspace: Path,
    review_tool_budget: int,
    review_instructions: str | None = None,
) -> str:
    base_instructions = (
        review_instructions
        if review_instructions is not None
        else FRESH_REVIEW_INSTRUCTIONS.format(
            review_tool_budget=review_tool_budget
        ).strip()
    )
    try:
        task_prompt = (workspace / "task" / "prompt.md").read_text(
            encoding="utf-8"
        ).strip()
    except OSError:
        task_prompt = ""
    if not task_prompt:
        return (
            f"{base_instructions}\n"
            f"Use at most {review_tool_budget} tool calls in this review."
        )
    return (
        f"{base_instructions}\n"
        f"Use at most {review_tool_budget} tool calls in this review.\n\n"
        "<task-description>\n"
        f"{task_prompt}\n"
        "</task-description>"
    )


def _independent_solve_instructions(
    workspace: Path, review_tool_budget: int
) -> str:
    task_prompt = (workspace / "task" / "prompt.md").read_text(
        encoding="utf-8", errors="replace"
    ).strip()
    return (
        f"{INDEPENDENT_SOLVE_INSTRUCTIONS.format(review_tool_budget=review_tool_budget).strip()}\n\n"
        "<task-description>\n"
        f"{task_prompt}\n"
        "</task-description>"
    )


def _git_head(repo: Path) -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _reset_task_repo(repo: Path, base_head: str) -> None:
    for command in (
        ["git", "reset", "--hard", base_head],
        ["git", "clean", "-fdx"],
    ):
        proc = subprocess.run(
            command,
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"failed to restore candidate workspace: {proc.stderr.strip()}"
            )


def _apply_candidate(workspace: Path, trajectory: Trajectory) -> None:
    workspace_root = workspace.resolve()
    for relative in sorted(trajectory.workspace_deletions):
        target = (workspace / relative).resolve()
        if not target.is_relative_to(workspace_root):
            raise ValueError(f"candidate deletion escapes workspace: {relative}")
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)
    for relative, contents in sorted(trajectory.workspace_diff.items()):
        target = (workspace / relative).resolve()
        if not target.is_relative_to(workspace_root):
            raise ValueError(f"candidate file escapes workspace: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)


def _write_candidate_bundle(
    workspace: Path, label: str, trajectory: Trajectory
) -> None:
    root = workspace / ".rho" / "heavy-candidates" / label
    shutil.rmtree(root, ignore_errors=True)
    files_root = root / "files"
    for relative, contents in sorted(trajectory.workspace_diff.items()):
        target = files_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "changed_files": sorted(trajectory.workspace_diff),
                "deleted_files": sorted(trajectory.workspace_deletions),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _candidate_diff_text(trajectory: Trajectory, *, max_chars: int = 80000) -> str:
    def is_generated_noise(relative: str) -> bool:
        path = Path(relative)
        parts = path.parts
        if path.suffix.lower() in {".pyc", ".pyo", ".class"}:
            return True
        if any(
            part in {
                "__pycache__",
                "node_modules",
                ".pytest_cache",
                ".mypy_cache",
                ".ruff_cache",
                ".tox",
                ".nox",
            }
            for part in parts
        ):
            return True
        return any(
            parts[index] == ".yarn"
            and parts[index + 1] in {"cache", "unplugged"}
            for index in range(len(parts) - 1)
        )

    files = sorted(trajectory.workspace_diff.items())
    deletions = sorted(trajectory.workspace_deletions)
    substantive_files = [item for item in files if not is_generated_noise(item[0])]
    substantive_deletions = [
        relative for relative in deletions if not is_generated_noise(relative)
    ]
    if substantive_files or substantive_deletions:
        files = substantive_files
        deletions = substantive_deletions

    entry_count = max(1, len(files) + len(deletions))
    per_file_chars = max(4000, min(16000, max_chars // entry_count))
    chunks: list[str] = []
    for relative, contents in files:
        try:
            rendered = contents.decode("utf-8")
        except UnicodeDecodeError:
            rendered = f"<binary file: {len(contents)} bytes>"
        if len(rendered) > per_file_chars:
            tail_chars = max(1000, per_file_chars // 4)
            rendered = (
                rendered[: per_file_chars - tail_chars]
                + "\n<file content truncated>\n"
                + rendered[-tail_chars:]
            )
        chunks.append(f"### {relative}\n{rendered}")
    for relative in deletions:
        chunks.append(f"### {relative}\n<deleted>")
    text = "\n\n".join(chunks) or "<no repository changes>"
    if len(text) > max_chars:
        return text[:max_chars] + "\n<additional candidate content truncated>"
    return text


def _candidate_judge_instructions(
    workspace: Path, first: Trajectory, second: Trajectory
) -> str:
    task_prompt = (workspace / "task" / "prompt.md").read_text(
        encoding="utf-8", errors="replace"
    ).strip()
    return f"""
Act as a conservative patch selector. Compare two independently produced
solutions against only the public task below. Choose the candidate most likely
to pass the hidden evaluator while preserving compatibility. Do not edit any
files and do not run tools. Prefer completeness and exact interfaces over
speculation; prefer the first candidate when evidence is genuinely tied.
Return only the requested JSON object.

The exact response shape is:
{{"choice":"first" or "second","rationale":"brief evidence-based reason"}}

<task-description>
{task_prompt}
</task-description>

<candidate-first>
{_candidate_diff_text(first)}
</candidate-first>

<candidate-second>
{_candidate_diff_text(second)}
</candidate-second>
""".strip()


def _candidate_choice(judge: Trajectory) -> tuple[str, str]:
    if judge.exit_code != 0 or judge.timed_out:
        return "first", "judge did not complete"
    rendered = judge.final_message.strip()
    candidates = [rendered]
    candidates.extend(rendered[match.start() :] for match in re.finditer(r"\{", rendered))
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(
            r"```(?:json)?\s*(.*?)\s*```", rendered, re.DOTALL
        )
    )
    decoder = json.JSONDecoder()
    for candidate in reversed(candidates):
        try:
            payload, _end = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        choice = str(payload.get("choice") or "")
        if choice in {"first", "second"}:
            return choice, str(payload.get("rationale") or "")
    return "first", "judge output did not contain a valid JSON choice"


def _merge_review_trajectories(
    first: Trajectory,
    review: Trajectory,
    *,
    tool_count: int,
) -> Trajectory:
    diff = dict(first.workspace_diff)
    deletions = set(first.workspace_deletions)
    review_succeeded = review.exit_code == 0 and not review.timed_out
    if review_succeeded:
        for path in review.workspace_deletions:
            diff.pop(path, None)
            deletions.add(path)
        for path, contents in review.workspace_diff.items():
            diff[path] = contents
            deletions.discard(path)
    marker = (
        f"[rho-fresh-review] applied after {tool_count} initial tool calls; "
        f"exit_code={review.exit_code} timed_out={review.timed_out}\n"
    )
    return Trajectory(
        id=f"traj_{uuid.uuid4().hex[:10]}",
        kind=first.kind,
        task_id=first.task_id,
        harness_id=first.harness_id,
        instructions=first.instructions,
        events=[*first.events, *_namespaced_review_events(review.events)],
        final_message=(review.final_message if review_succeeded else first.final_message),
        stdout=f"{first.stdout}\n[rho-fresh-review]\n{review.stdout}",
        stderr=f"{first.stderr.rstrip()}\n{marker}{review.stderr}".lstrip(),
        workspace_diff=diff,
        workspace_deletions=frozenset(deletions),
        exit_code=first.exit_code,
        wall_time_s=first.wall_time_s + review.wall_time_s,
        timed_out=first.timed_out,
        stage=first.stage,
        round_ix=first.round_ix,
        sample_index=first.sample_index,
        model=first.model,
        reasoning_effort=first.reasoning_effort,
        cache_mode=first.cache_mode,
    )


def _merge_candidate_trajectories(
    first: Trajectory,
    second: Trajectory | None,
    judge: Trajectory | None,
    *,
    choice: str,
    tool_count: int,
    reason: str,
) -> Trajectory:
    selected = second if choice == "second" and second is not None else first
    selected_label = "second" if selected is second else "first"
    events = _namespaced_events(
        first.events,
        "independent-first",
        candidate_label="first",
        outcome_relevant=selected_label == "first",
    )
    wall_time_s = first.wall_time_s
    stdout_parts = [first.stdout]
    auxiliary_stderr_parts: list[str] = []
    if second is not None:
        events.extend(
            _namespaced_events(
                second.events,
                "independent-second",
                candidate_label="second",
                outcome_relevant=selected_label == "second",
            )
        )
        wall_time_s += second.wall_time_s
        stdout_parts.extend(["[rho-independent-second]", second.stdout])
        if selected_label != "second" and second.stderr:
            auxiliary_stderr_parts.extend(
                ["[rho-independent-second-stderr]", second.stderr.rstrip()]
            )
    if judge is not None:
        events.extend(
            _namespaced_events(
                judge.events,
                "candidate-judge",
                candidate_label="judge",
                outcome_relevant=False,
            )
        )
        wall_time_s += judge.wall_time_s
        stdout_parts.extend(
            [
                "[rho-candidate-judge]",
                judge.stdout,
                "[rho-candidate-judge-final]",
                judge.final_message,
            ]
        )
        if judge.stderr:
            auxiliary_stderr_parts.extend(
                ["[rho-candidate-judge-stderr]", judge.stderr.rstrip()]
            )
    if selected_label == "second" and first.stderr:
        auxiliary_stderr_parts.extend(
            ["[rho-independent-first-stderr]", first.stderr.rstrip()]
        )
    stdout_parts.extend(auxiliary_stderr_parts)
    events.append(
        {
            "type": "rho.candidate_selection",
            "choice": choice,
            "reason": reason,
            "initial_tool_count": tool_count,
        }
    )
    selected_stderr = selected.stderr.rstrip()
    selection_marker = f"[rho-candidate-selection] choice={choice} reason={reason}"
    return Trajectory(
        id=f"traj_{uuid.uuid4().hex[:10]}",
        kind=first.kind,
        task_id=first.task_id,
        harness_id=first.harness_id,
        instructions=first.instructions,
        events=events,
        final_message=selected.final_message,
        stdout="\n".join(part for part in stdout_parts if part),
        stderr="\n".join(
            part for part in (selected_stderr, selection_marker) if part
        ),
        workspace_diff=dict(selected.workspace_diff),
        workspace_deletions=frozenset(selected.workspace_deletions),
        exit_code=selected.exit_code,
        wall_time_s=wall_time_s,
        timed_out=selected.timed_out,
        stage=first.stage,
        round_ix=first.round_ix,
        sample_index=first.sample_index,
        model=first.model,
        reasoning_effort=first.reasoning_effort,
        cache_mode=first.cache_mode,
    )


def _namespaced_review_events(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _namespaced_events(events, "fresh-review")


def _namespaced_events(
    events: list[dict[str, Any]],
    prefix: str,
    *,
    candidate_label: str | None = None,
    outcome_relevant: bool | None = None,
) -> list[dict[str, Any]]:
    namespaced: list[dict[str, Any]] = []
    for event in events:
        copied = dict(event)
        item = event.get("item")
        if isinstance(item, dict):
            copied_item = dict(item)
            copied_item["id"] = f"{prefix}:{item.get('id') or 'anonymous'}"
            copied["item"] = copied_item
        if candidate_label is not None:
            copied["_rho_candidate"] = candidate_label
        if outcome_relevant is not None:
            copied["_rho_outcome_relevant"] = outcome_relevant
        namespaced.append(copied)
    return namespaced


def _outcome_relevant_event(event: dict[str, Any]) -> bool:
    """Whether an event belongs to the candidate sent to the official grader."""

    return event.get("_rho_outcome_relevant") is not False


def _reconcile_transport_usage(
    trajectory: Trajectory,
    transport_state: dict[str, Any] | None,
    trace_id: str | None,
) -> Trajectory:
    """Recover billable provider usage omitted by a timed-out nested solve.

    Heavy candidate selection can run two Solver sessions and a judge inside
    one task attempt. Codex only emits ``turn.completed`` for sessions that
    finish cleanly, while the Responses proxy records every successful,
    billable response. Append only the missing provider delta so normal Codex
    usage remains intact and retry attempts can still be excluded by the
    evaluator's existing attempt classification.
    """

    if not trace_id or not isinstance(transport_state, dict):
        return trajectory
    provider = transport_state.get("usage")
    if not isinstance(provider, dict):
        return trajectory
    required = ("input_tokens", "cached_input_tokens", "output_tokens")
    if not all(isinstance(provider.get(key), (int, float)) for key in required):
        return trajectory

    observed = extract_usage(trajectory.events) or {}
    provider_usage = {key: max(0, int(provider.get(key) or 0)) for key in required}
    observed_usage = {key: max(0, int(observed.get(key) or 0)) for key in required}
    provider_api_calls = max(0, int(provider.get("api_call_count") or 0))
    observed_api_calls = extract_api_calls(trajectory.events) or 0
    below = {
        key: {
            "provider": provider_usage[key],
            "trajectory": observed_usage[key],
        }
        for key in required
        if provider_usage[key] < observed_usage[key]
    }
    if provider_api_calls < observed_api_calls:
        below["api_call_count"] = {
            "provider": provider_api_calls,
            "trajectory": observed_api_calls,
        }
    events = list(trajectory.events)
    if below:
        events.append(
            {
                "type": "rho.transport_usage_reconciliation",
                "status": "provider_state_below_trajectory",
                "usage_trace_id": trace_id,
                "mismatches": below,
            }
        )
        return replace(trajectory, events=events)

    delta = {
        key: provider_usage[key] - observed_usage[key]
        for key in required
    }
    api_delta = provider_api_calls - observed_api_calls
    if any(delta.values()) or api_delta:
        events.append(
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": delta["input_tokens"],
                    "cached_input_tokens": delta["cached_input_tokens"],
                    "cache_write_input_tokens": 0,
                    "output_tokens": delta["output_tokens"],
                },
                "usage_model": trajectory.model,
                "usage_trace_id": trace_id,
                "usage_source": "responses_retry_proxy_transport_state_delta",
                "api_calls": api_delta,
            }
        )
    else:
        events.append(
            {
                "type": "rho.transport_usage_reconciliation",
                "status": "matched",
                "usage_trace_id": trace_id,
            }
        )
    return replace(trajectory, events=events)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_task_id(task_id: str) -> str:
    return task_id.replace("/", "__")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_env_overrides(
    values: Iterable[str], *, description: str = "grader"
) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values:
        key, separator, value = raw.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(
                f"invalid {description} environment override: {raw!r}"
            )
        if "\x00" in value:
            raise ValueError(
                f"{description} environment override contains NUL: {key}"
            )
        result[key] = value
    return result


def _solver_dependency_transport_failures(
    trajectory: Trajectory,
) -> list[dict[str, Any]]:
    """Return concrete package/toolchain download failures from Solver tools.

    Detection is deliberately limited to completed command output containing a
    package-manager URL/error signature.  Generic task prose such as a test
    named ``connection_reset`` must not invalidate an otherwise scoreable run.
    Any matching attempt is retried regardless of its eventual grade, avoiding
    score-conditioned retries.
    """

    failures: list[dict[str, Any]] = []
    for event in trajectory.events:
        if not _outcome_relevant_event(event):
            continue
        if event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, dict) or item.get("type") != "command_execution":
            continue
        # A successful command may legitimately print dependency/network error
        # strings while searching source code, exercising retry tests, or using
        # a fallback mirror.  Only a failed command is evidence that the Solver
        # was actually blocked by dependency transport.
        exit_code = item.get("exit_code")
        if exit_code in (None, 0, "0"):
            continue
        output = str(item.get("aggregated_output") or "")
        if not any(
            pattern.search(output)
            for pattern in SOLVER_DEPENDENCY_TRANSPORT_PATTERNS
        ):
            continue
        compact = " ".join(output.split())
        failures.append(
            {
                "command": str(item.get("command") or "")[:1000],
                "exit_code": exit_code,
                "output_excerpt": compact[:2000],
            }
        )
    return failures


def _solver_environment_report(overrides: dict[str, str]) -> dict[str, Any]:
    """Inspect the exact Solver environment without mutating it."""

    env = os.environ.copy()
    env.update(overrides)
    path = env.get("PATH") or os.defpath
    go_path = shutil.which("go", path=path)
    report: dict[str, Any] = {
        "overrides": overrides,
        "resolved_go": go_path,
        "generated_at": _now_iso(),
    }
    if go_path is None:
        report["go_version"] = None
        return report
    try:
        completed = subprocess.run(
            [go_path, "version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
            env=env,
        )
        report["go_version"] = (completed.stdout or completed.stderr).strip()
        report["go_version_exit_code"] = completed.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["go_version"] = None
        report["go_version_error"] = f"{type(exc).__name__}: {exc}"
    return report


def _parse_go_version(output: str) -> str | None:
    match = GO_VERSION_PATTERN.search(output)
    return match.group("version") if match else None


def _probe_image_go_toolchain(image: str) -> dict[str, Any]:
    """Read the exact Go release from a pinned official evaluator image."""

    started = time.monotonic()
    try:
        completed = subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", "go", image, "version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "ok": False,
            "version": None,
            "error": f"{type(exc).__name__}: {exc}",
            "wall_time_s": round(time.monotonic() - started, 3),
        }
    output = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
    version = _parse_go_version(output)
    return {
        "ok": completed.returncode == 0 and version is not None,
        "version": version,
        "exit_code": completed.returncode,
        "output": output[:2000],
        "wall_time_s": round(time.monotonic() - started, 3),
    }


def _solver_task_environment(
    row: dict[str, Any],
    base_overrides: dict[str, str],
    official_go_versions: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    environment = dict(base_overrides)
    metadata: dict[str, str] = {}
    if str(row.get("repo_language") or "").lower() != "go":
        return environment, metadata
    image = docker_image(row)
    version = official_go_versions.get(image)
    if version is None:
        raise RuntimeError(
            "official Go toolchain version is unavailable for "
            f"{image}; run preflight before Solver evaluation"
        )
    environment["RHO_GO_TOOLCHAIN_VERSION"] = version
    environment["RHO_TASK_DOCKER_IMAGE"] = image
    metadata["solver_go_toolchain_version"] = version
    metadata["solver_go_toolchain_source"] = image
    return environment, metadata


def _render_codex_base_url_override(
    base_config: Path, *, output: Path, base_url: str
) -> Path:
    """Render an auditable runtime config with only the provider URL changed."""

    if not base_url.startswith(("http://", "https://")) or "\n" in base_url:
        raise ValueError("Codex base URL override must be an HTTP(S) URL")
    text = base_config.read_text(encoding="utf-8")
    parsed = tomllib.loads(text)
    provider_name = str(parsed.get("model_provider") or "").strip()
    if not provider_name or not re.fullmatch(r"[A-Za-z0-9_-]+", provider_name):
        raise ValueError("Codex config has an unsupported model_provider key")
    section_pattern = re.compile(
        rf"(?ms)(^\[model_providers\.{re.escape(provider_name)}\]\s*$)(.*?)(?=^\[|\Z)"
    )
    match = section_pattern.search(text)
    if match is None:
        raise ValueError(f"Codex provider section not found: {provider_name}")
    section_body = match.group(2)
    rewritten_body, replacements = re.subn(
        r'(?m)^(base_url\s*=\s*)"[^"]*"\s*$',
        rf'\g<1>"{base_url.rstrip("/")}"',
        section_body,
        count=1,
    )
    if replacements != 1:
        raise ValueError(f"Codex provider has no replaceable base_url: {provider_name}")
    rewritten = text[: match.start(2)] + rewritten_body + text[match.end(2) :]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rewritten, encoding="utf-8")
    return output


def _codex_config_uses_openai_api_key(path: Path) -> bool:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    provider_name = payload.get("model_provider")
    providers = payload.get("model_providers")
    if not isinstance(provider_name, str) or not isinstance(providers, dict):
        return False
    provider = providers.get(provider_name)
    return isinstance(provider, dict) and provider.get("env_key") == "OPENAI_API_KEY"


def _render_codex_transport_trace_header(config_path: Path, *, output: Path) -> Path:
    """Add an internal per-invocation trace header sourced from the environment."""

    text = config_path.read_text(encoding="utf-8")
    parsed = tomllib.loads(text)
    provider_name = str(parsed.get("model_provider") or "").strip()
    providers = parsed.get("model_providers")
    provider = providers.get(provider_name) if isinstance(providers, dict) else None
    if not isinstance(provider, dict):
        raise ValueError(f"Codex provider section not found: {provider_name}")
    existing_headers = provider.get("env_http_headers")
    if existing_headers is not None:
        if not isinstance(existing_headers, dict) or existing_headers.get(
            "X-Rho-Transport-Trace"
        ) != "RHO_TRANSPORT_TRACE_ID":
            raise ValueError(
                "Codex provider defines env_http_headers without the required "
                "X-Rho-Transport-Trace mapping"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        return output
    header_pattern = re.compile(
        rf"(?m)^\[model_providers\.{re.escape(provider_name)}\]\s*$"
    )
    match = header_pattern.search(text)
    if match is None:
        raise ValueError(f"Codex provider section not found: {provider_name}")
    insertion = (
        '\nenv_http_headers = { "X-Rho-Transport-Trace" = "RHO_TRANSPORT_TRACE_ID" }'
    )
    rewritten = text[: match.end()] + insertion + text[match.end() :]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rewritten, encoding="utf-8")
    return output


def _render_codex_clean_permission_profile(
    config_path: Path,
    *,
    output: Path,
    readable_roots: Iterable[Path] = (),
) -> Path:
    """Add a least-privilege Solver profile without changing global config.

    Permission profiles and the legacy ``sandbox_mode`` configuration do not
    compose.  Refuse a mixed configuration rather than silently falling back
    to the broader legacy filesystem policy.
    """

    text = config_path.read_text(encoding="utf-8")
    parsed = tomllib.loads(text)
    conflicts = sorted(
        key
        for key in ("sandbox_mode", "sandbox_workspace_write")
        if key in parsed
    )
    if conflicts:
        raise ValueError(
            "clean SWE-bench Pro Solver permissions cannot be combined with "
            "legacy Codex sandbox settings: " + ", ".join(conflicts)
        )
    permissions = parsed.get("permissions")
    if isinstance(permissions, dict) and CLEAN_SOLVER_PERMISSION_PROFILE in permissions:
        raise ValueError(
            "Codex config already defines reserved permission profile: "
            f"{CLEAN_SOLVER_PERMISSION_PROFILE}"
        )

    normalized_roots = sorted(
        {
            str(Path(root).expanduser().resolve())
            for root in readable_roots
            if Path(root).expanduser().exists()
        }
    )
    readable_root_rules = "".join(
        f"{json.dumps(root)} = \"read\"\n" for root in normalized_roots
    )
    profile = f"""

# Generated by run_swebench_pro_harness_eval.py for leakage-safe Solver runs.
[permissions.{CLEAN_SOLVER_PERMISSION_PROFILE}]
description = "SWE-bench Pro clean workspace only; no command network"

[permissions.{CLEAN_SOLVER_PERMISSION_PROFILE}.filesystem]
":root" = "deny"
":minimal" = "read"
{readable_root_rules.rstrip()}

[permissions.{CLEAN_SOLVER_PERMISSION_PROFILE}.filesystem.":workspace_roots"]
"." = "write"

[permissions.{CLEAN_SOLVER_PERMISSION_PROFILE}.network]
enabled = false
"""
    rewritten = text.rstrip() + profile
    # Validate the exact serialized configuration before giving it to Codex.
    tomllib.loads(rewritten)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rewritten, encoding="utf-8")
    return output


def _codex_runtime_root(binary: str) -> Path:
    """Return the narrow runtime tree required by a repository-local CLI."""

    candidate = shutil.which(binary)
    path = Path(candidate if candidate is not None else binary).expanduser().resolve()
    for parent in path.parents:
        if (parent / "node_modules" / "@openai" / "codex").is_dir():
            return parent
    return path.parent


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _runtime_classification_fingerprint(
    policy: HarnessRuntimePolicy, prompt: str
) -> str:
    classifier = policy.classifier
    if classifier is None:
        return ""
    payload = {
        "policy_sha256": _file_sha256(policy.source),
        "model": classifier.model,
        "reasoning_effort": classifier.reasoning_effort,
        "max_output_tokens": classifier.max_output_tokens,
        "samples": classifier.samples,
        "profiles": {
            name: {
                "reasoning_effort": profile.reasoning_effort,
                "economy_after_tools": profile.economy_after_tools,
                "strict_after_tools": profile.strict_after_tools,
                "finalize_after_tools": profile.finalize_after_tools,
            }
            for name, profile in sorted(policy.profiles.items())
        },
        "instructions": policy.classifier_instructions(prompt),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _runtime_classifier_consensus(
    samples: list[dict[str, Any]], *, default_profile: str
) -> dict[str, Any]:
    counts = Counter(str(sample["profile"]) for sample in samples)
    top_count = max(counts.values())
    leaders = sorted(
        profile for profile, count in counts.items() if count == top_count
    )
    if len(leaders) == 1:
        profile = leaders[0]
        winners = [sample for sample in samples if sample["profile"] == profile]
        representative = max(
            winners,
            key=lambda sample: (
                float(sample["confidence"]),
                str(sample["trajectory_id"]),
            ),
        )
        confidence = sum(float(sample["confidence"]) for sample in winners) / len(
            winners
        )
        rationale = str(representative["rationale"])
    else:
        profile = default_profile
        representative = max(
            samples,
            key=lambda sample: (
                float(sample["confidence"]),
                str(sample["trajectory_id"]),
            ),
        )
        confidence = min(float(sample["confidence"]) for sample in samples)
        rationale = (
            "Classifier samples disagreed without a plurality; used the "
            "neutral default."
        )
    return {
        "profile": profile,
        "confidence": round(confidence, 6),
        "rationale": rationale,
        "trajectory_id": str(representative["trajectory_id"]),
        "vote_counts": dict(sorted(counts.items())),
    }


def _cached_runtime_selection(
    policy: HarnessRuntimePolicy, record: dict[str, Any]
) -> ProfileSelection:
    if record.get("classification_source") != "model":
        return policy.fallback_selection(str(record.get("rationale") or "cached fallback"))
    consensus = record.get("consensus") or {}
    return policy.select_model_profile(
        str(record["profile"]),
        confidence=float(record["confidence"]),
        rationale=str(record.get("rationale") or ""),
        classifier_model=str(record["classifier_model"]),
        classifier_trajectory_id=str(record["classifier_trajectory_id"]),
        vote_counts=consensus.get("vote_counts"),
    )


def _classify_runtime_profile(
    *,
    task: Any,
    policy: HarnessRuntimePolicy,
    classifier_agent: Agent,
    run_dir: Path,
    traj_store: FilesystemTrajectoryStore,
    harness_id: str,
) -> ProfileSelection:
    classifier = policy.classifier
    if classifier is None:
        return policy.select(task.query())
    if policy.adjuster is not None:
        raise RuntimeError("SWE-bench Pro runtime profile adjusters are not supported")

    prompt = task.query()
    fingerprint = _runtime_classification_fingerprint(policy, prompt)
    cache_path = run_dir / "classifications" / _safe_task_id(task.id) / "selection.json"
    cached = _read_json(cache_path)
    if cached is not None and cached.get("fingerprint") == fingerprint:
        try:
            return _cached_runtime_selection(policy, cached)
        except (KeyError, TypeError, ValueError):
            pass

    instructions = policy.classifier_instructions(prompt)
    attempts: list[dict[str, Any]] = []
    valid_samples: list[dict[str, Any]] = []
    work_root = run_dir / "classifier_workdir"
    work_root.mkdir(parents=True, exist_ok=True)
    max_total_attempts = classifier.samples * classifier.max_attempts
    for attempt_ix in range(1, max_total_attempts + 1):
        workspace = Path(tempfile.mkdtemp(dir=str(work_root), prefix="classify_"))
        try:
            trajectory = classifier_agent.run(
                workspace,
                instructions,
                output_schema=policy.classifier_output_schema(),
                task_id=task.id,
                harness_id=harness_id,
                kind="evaluate",
                timeout_s=classifier.timeout_s,
            )
            trajectory = annotate_trajectory(
                trajectory,
                agent=classifier_agent,
                stage="swebench_pro_profile_classification",
            )
            traj_store.put(trajectory)
            attempt = {
                "attempt": attempt_ix,
                "trajectory_id": trajectory.id,
                "metrics": trajectory_metrics(trajectory),
            }
            attempts.append(attempt)
            if trajectory.timed_out:
                attempt["error"] = "classifier_timeout"
                continue
            if trajectory.exit_code != 0:
                attempt["error"] = "classifier_process_failure"
                continue
            try:
                output = json.loads(trajectory.final_message)
                profile = str(output["profile"])
                confidence = float(output["confidence"])
                rationale = str(output["rationale"])
                if profile not in policy.profiles or not 0 <= confidence <= 1:
                    raise ValueError("classifier output is outside the policy schema")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                attempt["error"] = f"invalid_classifier_output: {exc}"
                continue
            sample = {
                "profile": profile,
                "confidence": confidence,
                "rationale": rationale,
                "trajectory_id": trajectory.id,
            }
            attempt["output"] = sample
            valid_samples.append(sample)
            if len(valid_samples) >= classifier.samples:
                break
        except Exception as exc:  # noqa: BLE001 - bounded fallback is intentional
            attempts.append(
                {
                    "attempt": attempt_ix,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    if valid_samples:
        consensus = _runtime_classifier_consensus(
            valid_samples, default_profile=policy.default_profile
        )
        selection = policy.select_model_profile(
            str(consensus["profile"]),
            confidence=float(consensus["confidence"]),
            rationale=str(consensus["rationale"]),
            classifier_model=classifier.model,
            classifier_trajectory_id=str(consensus["trajectory_id"]),
            vote_counts=consensus.get("vote_counts"),
        )
        record = {
            "fingerprint": fingerprint,
            "classification_source": "model",
            "profile": selection.classified_profile or selection.profile.name,
            "effective_profile": selection.profile.name,
            "confidence": selection.confidence,
            "rationale": selection.rationale,
            "classifier_model": classifier.model,
            "classifier_reasoning_effort": classifier.reasoning_effort,
            "classifier_trajectory_id": selection.classifier_trajectory_id,
            "consensus": consensus,
            "attempts": attempts,
            "completed_at": _now_iso(),
        }
        _write_json(cache_path, record)
        return selection

    reason = "model classifier failed after bounded retries; used neutral default"
    selection = policy.fallback_selection(reason)
    _write_json(
        cache_path,
        {
            "fingerprint": fingerprint,
            "classification_source": "fallback",
            "profile": selection.profile.name,
            "effective_profile": selection.profile.name,
            "confidence": None,
            "rationale": reason,
            "classifier_model": classifier.model,
            "classifier_reasoning_effort": classifier.reasoning_effort,
            "classifier_trajectory_id": None,
            "attempts": attempts,
            "completed_at": _now_iso(),
        },
    )
    return selection


def _classification_trajectory_ids(
    run_dir: Path, *, exclude_infrastructure: bool = False
) -> list[str]:
    trajectory_ids: list[str] = []
    for path in sorted((run_dir / "classifications").glob("*/selection.json")):
        record = _read_json(path) or {}
        for attempt in record.get("attempts") or []:
            if exclude_infrastructure and attempt.get("error") in {
                "classifier_timeout",
                "classifier_process_failure",
            }:
                continue
            trajectory_id = attempt.get("trajectory_id")
            if trajectory_id:
                trajectory_ids.append(str(trajectory_id))
    return list(dict.fromkeys(trajectory_ids))


def _classification_infrastructure_trajectory_ids(run_dir: Path) -> list[str]:
    trajectory_ids: list[str] = []
    for path in sorted((run_dir / "classifications").glob("*/selection.json")):
        record = _read_json(path) or {}
        for attempt in record.get("attempts") or []:
            if attempt.get("error") not in {
                "classifier_timeout",
                "classifier_process_failure",
            }:
                continue
            trajectory_id = attempt.get("trajectory_id")
            if trajectory_id:
                trajectory_ids.append(str(trajectory_id))
    return list(dict.fromkeys(trajectory_ids))


def _summarize_transport_proxy_log(path: Path) -> dict[str, Any] | None:
    """Summarize auditable proxy attempts without treating HTTP 200 as success.

    A Responses request can return HTTP 200 with ``status=failed`` and still be
    retried.  Request completion is therefore determined from the final record
    for each proxy request ID, not from the HTTP status alone.
    """

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    requests: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    decoder = json.JSONDecoder()
    offset = 0
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset >= len(text):
            break
        try:
            record, end = decoder.raw_decode(text, offset)
        except json.JSONDecodeError:
            newline = text.find("\n", offset)
            if newline < 0:
                break
            offset = newline + 1
            continue
        offset = end
        request_id = record.get("request_id") if isinstance(record, dict) else None
        if isinstance(request_id, int) and isinstance(record.get("status"), int):
            session_id = str(record.get("proxy_session_id") or "legacy")
            requests[(session_id, request_id)].append(record)
    if not requests:
        return None

    attempts = [attempt for values in requests.values() for attempt in values]
    final_attempts = [values[-1] for values in requests.values()]
    status_counts = Counter(str(attempt["status"]) for attempt in attempts)
    retry_reason_counts = Counter(
        str(attempt["retry_reason"])
        for attempt in attempts
        if attempt.get("retry_reason")
    )
    completed = sum(
        attempt.get("status") == 200 and not attempt.get("retry_reason")
        for attempt in final_attempts
    )
    return {
        "request_count": len(requests),
        "completed_request_count": completed,
        "failed_request_count": len(requests) - completed,
        "upstream_attempt_count": len(attempts),
        "retried_request_count": sum(len(values) > 1 for values in requests.values()),
        "retry_attempt_count": sum(
            max(0, len(values) - 1) for values in requests.values()
        ),
        "retryable_attempt_count": sum(
            bool(item.get("retry_reason")) for item in attempts
        ),
        "retry_exhausted_attempt_count": sum(
            bool(item.get("retry_exhausted")) for item in attempts
        ),
        "http_status_counts": dict(sorted(status_counts.items())),
        "retry_reason_counts": dict(sorted(retry_reason_counts.items())),
        "log_path": str(path),
    }


def _load_completed_result(path: Path) -> dict[str, Any] | None:
    payload = _read_json(path)
    return payload if payload and payload.get("status") == "completed" else None


def _acquire_run_lock(run_dir: Path):
    lock_path = run_dir / ".run.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"run directory is already active: {run_dir}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "started_at": _now_iso()}) + "\n")
    handle.flush()
    return handle


def _configure_docker_command(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(f"docker command is not executable: {path}")
    if path.name != "docker":
        raise RuntimeError("--docker-command must point to an executable named docker")
    os.environ["PATH"] = f"{path.parent}:{os.environ.get('PATH', '')}"
    resolved = shutil.which("docker")
    if resolved is None or Path(resolved).resolve() != path:
        raise RuntimeError(f"failed to select requested docker command: {path}")
    proc = subprocess.run(
        ["docker", "version", "--format", "{{json .Server}}"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"docker daemon is unavailable: {proc.stderr or proc.stdout}"
        )
    return {"command": str(path), "server": proc.stdout.strip()}


def _looks_like_transport_failure(trajectory: Trajectory) -> bool:
    if trajectory.exit_code == 0:
        return False

    # Do not scan all stdout: Codex stdout contains tool results, and ordinary
    # repository source/tests may legitimately mention strings such as
    # "rate limit" or "connection reset". Transport failures are emitted as
    # structured terminal error events (or, less commonly, stderr text).
    terminal_text = trajectory.stderr.lower()
    for event in reversed(trajectory.events):
        if not _outcome_relevant_event(event):
            continue
        event_type = event.get("type")
        if event_type not in {"error", "turn.failed"}:
            if event_type in {"item.started", "item.completed", "turn.completed"}:
                break
            continue
        terminal_text += "\n" + json.dumps(event, ensure_ascii=False).lower()
        if event_type == "turn.failed":
            break
    if any(marker in terminal_text for marker in TRANSPORT_MARKERS):
        return True
    return retryable_transport_failure(
        Trajectory(
            id=trajectory.id,
            kind=trajectory.kind,
            task_id=trajectory.task_id,
            harness_id=trajectory.harness_id,
            instructions=trajectory.instructions,
            events=[],
            final_message=trajectory.final_message,
            stdout="",
            stderr=trajectory.stderr,
            workspace_diff=trajectory.workspace_diff,
            workspace_deletions=trajectory.workspace_deletions,
            exit_code=trajectory.exit_code,
            wall_time_s=trajectory.wall_time_s,
            timed_out=trajectory.timed_out,
            stage=trajectory.stage,
            round_ix=trajectory.round_ix,
            sample_index=trajectory.sample_index,
            model=trajectory.model,
            reasoning_effort=trajectory.reasoning_effort,
            cache_mode=trajectory.cache_mode,
        )
    )


def _stalled_upstream_timeout(
    trajectory: Trajectory, transport_timing: dict[str, Any] | None
) -> bool:
    """Return whether a Solver timeout was consumed by a retryable API stall."""

    if not trajectory.timed_out or not transport_timing:
        return False
    stalled_count = float(
        transport_timing.get("stalled_retryable_attempt_count") or 0.0
    )
    max_retryable_s = float(
        transport_timing.get("max_retryable_attempt_s") or 0.0
    )
    return stalled_count > 0 or max_retryable_s >= STALLED_UPSTREAM_THRESHOLD_S


def _should_grade_trajectory(trajectory: Trajectory) -> bool:
    """Return whether the official grader should evaluate this trajectory.

    The grader scores the repository diff left by the solver. Reaching the
    solver wall-clock limit therefore does not imply failure: a timed-out run
    may already contain a correct patch and must still be graded.
    """

    if _looks_like_transport_failure(trajectory):
        return False
    return trajectory.timed_out or trajectory.exit_code == 0


def _looks_like_infra_text(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in INFRA_TEXT_MARKERS)


def _grader_artifact_text(artifacts_dir: Path) -> str:
    chunks: list[str] = []
    for name in (
        "test_stdout.log",
        "test_stderr.log",
        "docker_stdout.log",
        "docker_stderr.log",
    ):
        path = artifacts_dir / "docker" / name
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            pass
    return "\n".join(chunks)


def _looks_like_dependency_transport_failure(
    details: dict[str, Any], artifacts_dir: Path
) -> bool:
    if not details.get("missing_tests"):
        return False
    text = _grader_artifact_text(artifacts_dir)
    return any(pattern.search(text) for pattern in GRADER_DEPENDENCY_TRANSPORT_PATTERNS)


def classify_grade_result(
    details: dict[str, Any], artifacts_dir: Path
) -> tuple[str, str]:
    """Return (classification, reason) for a completed dataset grade call."""

    error = str(details.get("error") or "")
    message = str(details.get("message") or "")
    if not error:
        if _looks_like_dependency_transport_failure(details, artifacts_dir):
            return "infrastructure", "grader_dependency_transport_failure"
        return "model_result", "official_grader_result"
    if error == "patch_extraction_failed":
        if "trajectory did not modify" in message.lower():
            return "model_result", "no_prediction_patch"
        if _looks_like_infra_text(message) or "git " in message.lower():
            return "infrastructure", "patch_extraction_infrastructure"
        return "model_result", "invalid_prediction_patch"
    if error == "docker_evaluation_failed":
        lowered = message.lower()
        if "timed out after" in lowered and "docker" in lowered:
            if re.search(r"docker['\"]?,\s*['\"](?:pull|image|tag)['\"]", lowered) or re.search(
                r"\bdocker\s+(?:pull|image|tag)\b", lowered
            ):
                return "infrastructure", "docker_image_preparation_timeout"
            return "model_result", "official_grader_timeout"
        return "infrastructure", "docker_evaluator_exception"
    if error == "output.json not produced":
        logs = []
        for name in ("docker_stdout.log", "docker_stderr.log"):
            path = artifacts_dir / "docker" / name
            try:
                logs.append(path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
        if int(details.get("docker_exit_code") or 0) == 125 or _looks_like_infra_text(
            "\n".join(logs)
        ):
            return "infrastructure", "docker_runtime_failure"
        return "model_result", "grader_output_missing"
    return "model_result", f"grader_{error}"


_PREMATURE_INTENT_PATTERNS = (
    re.compile(r"\bi(?:'|’)ll\s+(?:start|begin)\b", re.IGNORECASE),
    re.compile(r"\bi\s+will\s+(?:start|begin)\b", re.IGNORECASE),
    re.compile(r"\blet\s+me\s+(?:start|begin)\b", re.IGNORECASE),
)
_PREMATURE_ACTION_PATTERN = re.compile(
    r"\b(?:analy[sz](?:e|ing)|check(?:ing)?|explor(?:e|ing)|"
    r"inspect(?:ing)?|read(?:ing)?|review(?:ing)?|understand(?:ing)?)\b",
    re.IGNORECASE,
)


def _looks_like_claude_code_premature_completion(
    trajectory: Trajectory,
) -> bool:
    """Detect a protocol-level end-turn before an agent actually starts.

    Some non-Claude models routed through Claude Code occasionally emit only a
    forward-looking acknowledgement (for example, that they will now inspect
    the workspace) and terminate the turn without invoking a tool.  Retrying
    this narrow no-op shape avoids treating an adapter turn-boundary mismatch
    as a benchmark answer while preserving ordinary empty-patch model failures.
    """

    if trajectory.exit_code != 0 or trajectory.timed_out:
        return False
    if trajectory.workspace_diff or trajectory.workspace_deletions:
        return False
    if any(event.get("type") == "item.completed" for event in trajectory.events):
        return False
    message = trajectory.final_message.strip()
    if not message or len(message) > 240:
        return False
    return any(pattern.search(message) for pattern in _PREMATURE_INTENT_PATTERNS) and bool(
        _PREMATURE_ACTION_PATTERN.search(message)
    )


def _retry_delay(attempt_ix: int) -> None:
    time.sleep(min(30, 10 * attempt_ix))


def _sanitized_proxy_endpoint(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = urlsplit(value)
        if not parsed.scheme or not parsed.hostname:
            return "configured"
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        port = f":{parsed.port}" if parsed.port is not None else ""
        return f"{parsed.scheme}://{host}{port}"
    except (TypeError, ValueError):
        return "configured"


def _git_network_report(repo_cache: RepoCache) -> dict[str, Any]:
    proxy_value = next(
        (
            os.environ.get(name)
            for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")
            if os.environ.get(name)
        ),
        None,
    )
    return {
        "cache_mode": repo_cache.cache_mode,
        "cache_path": str(repo_cache.root.resolve()),
        "git_http_version": "HTTP/1.1",
        "https_proxy_configured": bool(proxy_value),
        "https_proxy_endpoint": _sanitized_proxy_endpoint(proxy_value),
    }


def _preflight_official_assets(cache_root: Path) -> dict[str, Any]:
    started = time.monotonic()
    root = cache_root / "SWE-bench_Pro-os"
    errors: list[str] = []
    for attempt in range(1, 4):
        try:
            ensure_official_assets(root)
            return {
                "ok": True,
                "path": str(root),
                "attempts": attempt,
                "errors": errors,
                "wall_time_s": round(time.monotonic() - started, 3),
            }
        except Exception as exc:  # noqa: BLE001 - persist infrastructure evidence
            errors.append(f"{type(exc).__name__}: {exc}")
            if attempt < 3:
                _retry_delay(attempt)
    return {
        "ok": False,
        "path": str(root),
        "attempts": 3,
        "errors": errors,
        "wall_time_s": round(time.monotonic() - started, 3),
    }


def _preflight_repositories(
    tasks: list[Any], repo_cache: RepoCache, workers: int
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        grouped[str(task.row["repo"])].append(task.row)
    records: list[dict[str, Any]] = []
    lock = threading.Lock()
    completed = 0

    def ensure_repo(repo: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        started = time.monotonic()
        errors: list[str] = []
        unique_rows = {str(row["base_commit"]): row for row in rows}
        for attempt in range(1, 4):
            try:
                repo_cache.ensure_cached(repo, unique_rows)
                return {
                    "repo": repo,
                    "commit_count": len(unique_rows),
                    "ok": True,
                    "attempts": attempt,
                    "errors": errors,
                    "repository": repo_cache.repository_report(repo),
                    "wall_time_s": round(time.monotonic() - started, 3),
                }
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}")
                if attempt < 3:
                    _retry_delay(attempt)
        return {
            "repo": repo,
            "commit_count": len(unique_rows),
            "ok": False,
            "attempts": 3,
            "errors": errors,
            "wall_time_s": round(time.monotonic() - started, 3),
        }

    with ThreadPoolExecutor(max_workers=min(workers, len(grouped))) as pool:
        futures = {
            pool.submit(ensure_repo, repo, rows): repo for repo, rows in grouped.items()
        }
        for future in as_completed(futures):
            record = future.result()
            with lock:
                records.append(record)
                completed += 1
                print(
                    f"[repos {completed}/{len(grouped)}] "
                    f"{'OK' if record['ok'] else 'FAIL'} {record['repo']} "
                    f"commits={record['commit_count']} wall={record['wall_time_s']}s",
                    flush=True,
                )
    records.sort(key=lambda item: item["repo"])
    return {
        "ok": all(record["ok"] for record in records),
        "count": len(records),
        "network": _git_network_report(repo_cache),
        "records": records,
    }


def _preflight_images(
    tasks: list[Any], workers: int, *, mirror_prefix: str | None
) -> dict[str, Any]:
    images = sorted({docker_image(task.row) for task in tasks})
    go_images = {
        docker_image(task.row)
        for task in tasks
        if str(task.row.get("repo_language") or "").lower() == "go"
    }
    records: list[dict[str, Any]] = []
    lock = threading.Lock()
    completed = 0

    def ensure_image(image: str) -> dict[str, Any]:
        started = time.monotonic()
        errors: list[str] = []

        def docker_check(*command: str, timeout: int) -> subprocess.CompletedProcess[str]:
            try:
                return subprocess.run(
                    ["docker", *command],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                # A slow registry must fail this attempt, not abort the entire
                # preflight and discard the other completed image checks.
                return subprocess.CompletedProcess(
                    ["docker", *command],
                    124,
                    stderr=f"Docker command timed out after {timeout}s: {exc}",
                )

        for attempt in range(1, 4):
            inspect = docker_check("image", "inspect", image, timeout=60)
            if inspect.returncode == 0:
                return {
                    "image": image,
                    "ok": True,
                    "pulled": attempt > 1 or bool(errors),
                    "attempts": attempt,
                    "errors": errors,
                    "wall_time_s": round(time.monotonic() - started, 3),
                }
            pull = docker_check("pull", image, timeout=3600)
            if pull.returncode == 0:
                return {
                    "image": image,
                    "ok": True,
                    "pulled": True,
                    "attempts": attempt,
                    "errors": errors,
                    "wall_time_s": round(time.monotonic() - started, 3),
                }
            official_error = (pull.stderr or pull.stdout).strip()
            errors.append(official_error[-4000:])
            if mirror_prefix:
                mirror_image = f"{mirror_prefix.rstrip('/')}/{image}"
                mirror_pull = docker_check("pull", mirror_image, timeout=3600)
                if mirror_pull.returncode == 0:
                    tag = docker_check("tag", mirror_image, image, timeout=60)
                    if tag.returncode == 0:
                        return {
                            "image": image,
                            "ok": True,
                            "pulled": True,
                            "pull_source": mirror_prefix,
                            "attempts": attempt,
                            "errors": errors,
                            "wall_time_s": round(time.monotonic() - started, 3),
                        }
                    errors.append((tag.stderr or tag.stdout).strip()[-4000:])
                else:
                    errors.append(
                        (mirror_pull.stderr or mirror_pull.stdout).strip()[-4000:]
                    )
            if attempt < 3:
                _retry_delay(attempt)
        return {
            "image": image,
            "ok": False,
            "pulled": False,
            "attempts": 3,
            "errors": errors,
            "wall_time_s": round(time.monotonic() - started, 3),
        }

    with ThreadPoolExecutor(max_workers=min(workers, len(images))) as pool:
        futures = {pool.submit(ensure_image, image): image for image in images}
        for future in as_completed(futures):
            record = future.result()
            with lock:
                records.append(record)
                completed += 1
                print(
                    f"[images {completed}/{len(images)}] "
                    f"{'OK' if record['ok'] else 'FAIL'} {record['image']} "
                    f"wall={record['wall_time_s']}s",
                    flush=True,
                )
    records.sort(key=lambda item: item["image"])
    if all(record["ok"] for record in records) and go_images:
        print("[preflight] official Go toolchains", flush=True)
        record_by_image = {record["image"]: record for record in records}
        completed_go = 0
        with ThreadPoolExecutor(max_workers=min(workers, len(go_images))) as pool:
            futures = {
                pool.submit(_probe_image_go_toolchain, image): image
                for image in sorted(go_images)
            }
            for future in as_completed(futures):
                image = futures[future]
                probe = future.result()
                record_by_image[image]["go_toolchain"] = probe
                completed_go += 1
                print(
                    f"[go {completed_go}/{len(go_images)}] "
                    f"{'OK' if probe['ok'] else 'FAIL'} {image} "
                    f"version={probe.get('version')} wall={probe['wall_time_s']}s",
                    flush=True,
                )
    return {
        "ok": all(
            record["ok"]
            and (
                record["image"] not in go_images
                or bool((record.get("go_toolchain") or {}).get("ok"))
            )
            for record in records
        ),
        "count": len(records),
        "records": records,
    }


def _run_preflight(
    tasks: list[Any],
    *,
    cache_root: Path,
    repo_cache: RepoCache,
    workers: int,
    mirror_prefix: str | None,
    report_path: Path,
) -> None:
    started = time.monotonic()
    print("[preflight] official evaluator assets", flush=True)
    assets = _preflight_official_assets(cache_root)
    if not assets["ok"]:
        _write_json(
            report_path, {"ok": False, "assets": assets, "completed_at": _now_iso()}
        )
        raise RuntimeError("SWE-bench Pro official evaluator assets preflight failed")
    print("[preflight] source repositories", flush=True)
    repos = _preflight_repositories(tasks, repo_cache, workers)
    if not repos["ok"]:
        _write_json(
            report_path,
            {
                "ok": False,
                "assets": assets,
                "repositories": repos,
                "completed_at": _now_iso(),
            },
        )
        raise RuntimeError("SWE-bench Pro repository preflight failed")
    print("[preflight] Docker images", flush=True)
    images = _preflight_images(tasks, workers, mirror_prefix=mirror_prefix)
    report = {
        "ok": bool(images["ok"]),
        "assets": assets,
        "repositories": repos,
        "images": images,
        "wall_time_s": round(time.monotonic() - started, 3),
        "completed_at": _now_iso(),
    }
    _write_json(report_path, report)
    if not images["ok"]:
        raise RuntimeError("SWE-bench Pro Docker image preflight failed")


def _gold_trajectory(task_id: str, harness_id: str, patch: str) -> Trajectory:
    return Trajectory(
        id="traj_gold_preflight",
        kind="solve",
        task_id=task_id,
        harness_id=harness_id,
        instructions="SWE-bench Pro gold-patch infrastructure canary",
        events=[],
        final_message=f"```diff\n{patch.rstrip()}\n```",
        stdout="",
        stderr="",
        workspace_diff={},
        workspace_deletions=frozenset(),
        exit_code=0,
        wall_time_s=0.0,
    )


def _run_gold_canary(task: Any, run_dir: Path) -> None:
    report_path = run_dir / "reports" / "gold_canary.json"
    existing = _read_json(report_path)
    if (
        existing
        and existing.get("passed") is True
        and existing.get("task_id") == task.id
    ):
        print(f"[gold] reuse PASS {task.id}", flush=True)
        return
    started = time.monotonic()
    artifacts = run_dir / "preflight" / "gold_canary" / _safe_task_id(task.id)
    grade = task.grade(
        _gold_trajectory(task.id, task.harness.id, str(task.row["patch"])),
        artifacts_dir=artifacts,
    )
    report = {
        "task_id": task.id,
        "passed": grade.passed,
        "score": grade.score,
        "details": grade.details,
        "wall_time_s": round(time.monotonic() - started, 3),
        "completed_at": _now_iso(),
    }
    _write_json(report_path, report)
    if not grade.passed:
        raise RuntimeError(
            f"gold-patch grader canary failed for {task.id}: {grade.details}"
        )
    print(f"[gold] PASS {task.id} wall={report['wall_time_s']}s", flush=True)


def _result_files(task_ids: Iterable[str], run_dir: Path) -> list[dict[str, Any]]:
    results = []
    for task_id in task_ids:
        payload = _read_json(run_dir / "tasks" / _safe_task_id(task_id) / "result.json")
        if payload is not None:
            results.append(payload)
    return results


def _summary_counts(
    task_ids: list[str],
    run_dir: Path,
    *,
    harness_id: str,
    harness_label: str,
    started_at: str,
    run_wall_time_s: float,
    final: bool,
) -> dict[str, Any]:
    payloads = _result_files(task_ids, run_dir)
    completed = [item for item in payloads if item.get("status") == "completed"]
    infra = [item for item in payloads if item.get("status") == "infrastructure_failed"]
    passed = sum(item.get("passed") is True for item in completed)
    updated_at = _now_iso()
    try:
        started = datetime.fromisoformat(started_at)
        updated = datetime.fromisoformat(updated_at)
        total_run_wall_time_s = max(
            float(run_wall_time_s), (updated - started).total_seconds()
        )
    except (TypeError, ValueError):
        total_run_wall_time_s = float(run_wall_time_s)
    return {
        "final": final,
        "harness_id": harness_id,
        "harness_label": harness_label,
        "requested_tasks": len(task_ids),
        "completed_tasks": len(completed),
        "passed": passed,
        "model_failed": len(completed) - passed,
        "infrastructure_failed": len(infra),
        "pending": len(task_ids) - len(completed) - len(infra),
        "pass_rate_completed": passed / len(completed) if completed else None,
        "pass_rate_all_requested": passed / len(task_ids) if task_ids else None,
        "started_at": started_at,
        "updated_at": updated_at,
        "run_wall_time_s": round(total_run_wall_time_s, 3),
        "results": completed,
        "infrastructure_failures": infra,
    }


def _trajectory_ids_from_attempts(
    results: list[dict[str, Any]], classification: str
) -> list[str]:
    ids: list[str] = []
    for result in results:
        for attempt in result.get("attempts") or []:
            metrics = attempt.get("call_metrics") or {}
            trajectory_id = metrics.get("trajectory_id")
            if attempt.get("classification") == classification and trajectory_id:
                ids.append(str(trajectory_id))
    return ids


def _load_trajectories(
    traj_store: FilesystemTrajectoryStore, trajectory_ids: Iterable[str]
) -> list[Trajectory]:
    trajectories: list[Trajectory] = []
    for trajectory_id in dict.fromkeys(trajectory_ids):
        try:
            trajectories.append(traj_store.get(trajectory_id))
        except KeyError:
            continue
    return trajectories


def _archived_infrastructure_results(
    run_dir: Path, *, excluded_trajectory_ids: Iterable[str]
) -> list[dict[str, Any]]:
    """Load superseded attempts archived outside the resumable task records."""

    seen = set(excluded_trajectory_ids)
    archived: list[dict[str, Any]] = []
    correction_root = run_dir / "reports" / "infra-corrections"
    for result_path in sorted(correction_root.glob("**/misclassified_result.json")):
        payload = _read_json(result_path)
        if payload is None:
            continue
        correction = _read_json(result_path.with_name("correction.json")) or {}
        attempts: list[dict[str, Any]] = []
        for original in payload.get("attempts") or []:
            trajectory_id = str(
                (original.get("call_metrics") or {}).get("trajectory_id") or ""
            )
            if not trajectory_id or trajectory_id in seen:
                continue
            seen.add(trajectory_id)
            attempt = deepcopy(original)
            attempt["classification"] = "infrastructure"
            attempt["infra_reason"] = "archived_infrastructure_correction"
            attempt["infrastructure_evidence"] = {
                "correction_reason": correction.get("reason"),
                "correction_path": str(result_path.parent),
            }
            attempt.pop("model_reason", None)
            attempts.append(attempt)
        if attempts:
            archived.append({**payload, "attempts": attempts})
    return archived


def _timing_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    formal_solve = 0.0
    formal_grade = 0.0
    formal_transport = 0.0
    formal_pacer = 0.0
    formal_backoff = 0.0
    formal_retry_attempt = 0.0
    formal_shared_rate_limit = 0.0
    formal_timeout_compensation = 0.0
    infra_attempt = 0.0
    infra_transport = 0.0
    all_task_wall = 0.0
    for result in results:
        all_task_wall += float(result.get("task_total_wall_time_s") or 0.0)
        for attempt in result.get("attempts") or []:
            transport = attempt.get("transport_timing") or {}
            transport_wait = float(transport.get("transport_wait_s") or 0.0)
            if attempt.get("classification") == "infrastructure":
                infra_attempt += float(attempt.get("wall_time_s") or 0.0)
                infra_transport += transport_wait
            elif attempt.get("classification") == "model_result":
                formal_solve += float(attempt.get("solve_wall_time_s") or 0.0)
                formal_grade += float(attempt.get("grade_wall_time_s") or 0.0)
                formal_transport += transport_wait
                formal_pacer += float(transport.get("pacer_wait_s") or 0.0)
                formal_backoff += float(transport.get("retry_backoff_s") or 0.0)
                formal_retry_attempt += float(
                    transport.get("retry_attempt_wait_s") or 0.0
                )
                formal_shared_rate_limit += float(
                    transport.get("shared_rate_limit_wait_s") or 0.0
                )
                formal_timeout_compensation += float(
                    transport.get("timeout_compensation_s") or 0.0
                )
    return {
        "formal_solver_wall_time_s": round(formal_solve, 6),
        "formal_transport_wait_s": round(formal_transport, 6),
        "formal_pacer_wait_s": round(formal_pacer, 6),
        "formal_retry_backoff_s": round(formal_backoff, 6),
        "formal_retry_attempt_wait_s": round(formal_retry_attempt, 6),
        "formal_shared_rate_limit_wait_s": round(formal_shared_rate_limit, 6),
        "formal_timeout_compensation_s": round(formal_timeout_compensation, 6),
        "formal_effective_solver_time_s": round(
            max(0.0, formal_solve - formal_timeout_compensation), 6
        ),
        "formal_grader_wall_time_s": round(formal_grade, 6),
        "formal_solver_plus_grader_wall_time_s": round(formal_solve + formal_grade, 6),
        "excluded_infrastructure_attempt_wall_time_s": round(infra_attempt, 6),
        "excluded_infrastructure_transport_wait_s": round(infra_transport, 6),
        "sum_task_total_wall_time_s": round(all_task_wall, 6),
    }


def _calendar_elapsed_to_last_result_s(
    started_at: str, results: Iterable[dict[str, Any]]
) -> float | None:
    try:
        started = datetime.fromisoformat(started_at)
        completed = [
            datetime.fromisoformat(str(result["completed_at"]))
            for result in results
            if result.get("completed_at")
        ]
    except (TypeError, ValueError):
        return None
    if not completed:
        return None
    return round(max(0.0, (max(completed) - started).total_seconds()), 6)


def _qualification_summary(
    *,
    summary: dict[str, Any],
    observed_cost_usd: float,
    cost_complete: bool,
    target: dict[str, Any],
) -> dict[str, Any]:
    requested_target = int(target["requested_tasks"])
    minimum_passed = int(target["minimum_passed"])
    baseline_cost = float(target["baseline_cost_usd"])
    maximum_cost_fraction = float(target["maximum_cost_fraction"])
    ceiling = baseline_cost * maximum_cost_fraction
    result_set_complete = (
        int(summary["requested_tasks"]) == requested_target
        and int(summary["completed_tasks"]) == requested_target
        and int(summary["infrastructure_failed"]) == 0
        and int(summary["pending"]) == 0
    )
    performance_status = (
        "pass"
        if result_set_complete and int(summary["passed"]) >= minimum_passed
        else "fail"
        if result_set_complete
        else "indeterminate"
    )
    cost_status = (
        "fail"
        if observed_cost_usd > ceiling
        else "pass"
        if cost_complete
        else "indeterminate"
    )
    overall_status = (
        "pass"
        if performance_status == "pass" and cost_status == "pass"
        else "fail"
        if "fail" in {performance_status, cost_status}
        else "indeterminate"
    )
    return {
        "status": overall_status,
        "result_set_complete": result_set_complete,
        "performance": {
            "minimum_passed": minimum_passed,
            "observed_passed": int(summary["passed"]),
            "status": performance_status,
        },
        "cost": {
            "baseline_label": str(target["baseline_label"]),
            "baseline_cost_usd": round(baseline_cost, 9),
            "maximum_cost_fraction": maximum_cost_fraction,
            "required_reduction_fraction": round(1.0 - maximum_cost_fraction, 9),
            "ceiling_cost_usd": round(ceiling, 9),
            "observed_cost_usd": round(observed_cost_usd, 9),
            "observed_fraction_of_baseline": round(
                observed_cost_usd / baseline_cost, 9
            ),
            "cost_complete": cost_complete,
            "status": cost_status,
        },
    }


def _write_final_reports(
    summary: dict[str, Any],
    *,
    run_dir: Path,
    traj_store: FilesystemTrajectoryStore,
    qualification_target: dict[str, Any] | None = None,
) -> dict[str, Any]:
    results = [*summary["results"], *summary["infrastructure_failures"]]
    formal_ids = [
        str(result["trajectory_id"])
        for result in summary["results"]
        if result.get("trajectory_id")
    ]
    control_plane_ids = _classification_trajectory_ids(
        run_dir, exclude_infrastructure=True
    )
    control_plane_attempt_ids = _classification_trajectory_ids(run_dir)
    recorded_ids = [
        str((attempt.get("call_metrics") or {}).get("trajectory_id") or "")
        for result in results
        for attempt in result.get("attempts") or []
    ]
    archived_infrastructure = _archived_infrastructure_results(
        run_dir, excluded_trajectory_ids=recorded_ids
    )
    metric_results = [*results, *archived_infrastructure]
    classifier_infra_ids = _classification_infrastructure_trajectory_ids(run_dir)
    infra_ids = list(
        dict.fromkeys(
            [
                *_trajectory_ids_from_attempts(metric_results, "infrastructure"),
                *classifier_infra_ids,
            ]
        )
    )
    all_ids = []
    for result in metric_results:
        for attempt in result.get("attempts") or []:
            trajectory_id = (attempt.get("call_metrics") or {}).get("trajectory_id")
            if trajectory_id:
                all_ids.append(str(trajectory_id))
    solver_trajectories = _load_trajectories(traj_store, formal_ids)
    control_plane_trajectories = _load_trajectories(
        traj_store, control_plane_ids
    )
    scored_trajectories = [*solver_trajectories, *control_plane_trajectories]
    infra_trajectories = _load_trajectories(traj_store, infra_ids)
    all_trajectories = _load_trajectories(
        traj_store, [*all_ids, *control_plane_attempt_ids]
    )
    solver_metrics = usage_summary(solver_trajectories)
    control_plane_metrics = usage_summary(control_plane_trajectories)
    formal_metrics = usage_summary(scored_trajectories)
    infra_metrics = usage_summary(infra_trajectories)
    all_metrics = usage_summary(all_trajectories)
    solver_by_id = {trajectory.id: trajectory for trajectory in solver_trajectories}
    profile_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    profile_trajectories: dict[str, list[Trajectory]] = defaultdict(list)
    for result in summary["results"]:
        selection = result.get("runtime_selection") or {}
        profile = str(selection.get("profile") or "fixed")
        profile_results[profile].append(result)
        trajectory = solver_by_id.get(str(result.get("trajectory_id") or ""))
        if trajectory is not None:
            profile_trajectories[profile].append(trajectory)
    runtime_profile_summary = {
        profile: {
            "completed_tasks": len(items),
            "passed": sum(item.get("passed") is True for item in items),
            "failed": sum(item.get("passed") is False for item in items),
            "metrics": usage_summary(profile_trajectories.get(profile, [])),
        }
        for profile, items in sorted(profile_results.items())
    }
    timing = _timing_summary(metric_results)
    classifier_infra_id_set = set(classifier_infra_ids)
    timing["excluded_infrastructure_attempt_wall_time_s"] = round(
        timing["excluded_infrastructure_attempt_wall_time_s"]
        + sum(
            trajectory.wall_time_s
            for trajectory in infra_trajectories
            if trajectory.id in classifier_infra_id_set
        ),
        6,
    )
    timing["calendar_elapsed_to_last_final_result_s"] = (
        _calendar_elapsed_to_last_result_s(
            str(summary["started_at"]), summary["results"]
        )
    )
    transport_proxy = _summarize_transport_proxy_log(
        run_dir / "reports" / "responses_retry_proxy.jsonl"
    )
    summary["metrics"] = formal_metrics
    summary["solver_metrics"] = solver_metrics
    summary["control_plane_metrics"] = control_plane_metrics
    summary["runtime_profile_summary"] = runtime_profile_summary
    summary["timing"] = timing
    summary["metrics_scope"] = (
        "one final Solver model-result trajectory for each completed task plus "
        "runtime-profile classifier trajectories excluding timed-out or "
        "process-failed classifier attempts; superseded infrastructure "
        "attempts remain in all-attempt and infrastructure reports"
    )
    summary["excluded_infrastructure_metrics"] = infra_metrics
    if transport_proxy is not None:
        summary["transport_proxy"] = transport_proxy
    _write_json(run_dir / "reports" / "summary.json", summary)
    _write_json(run_dir / "reports" / "metrics_summary.json", formal_metrics)
    _write_json(run_dir / "reports" / "solver_metrics_summary.json", solver_metrics)
    _write_json(
        run_dir / "reports" / "control_plane_metrics_summary.json",
        control_plane_metrics,
    )
    _write_json(run_dir / "reports" / "all_attempt_metrics_summary.json", all_metrics)
    _write_json(
        run_dir / "reports" / "infrastructure_metrics_summary.json", infra_metrics
    )
    _write_json(
        run_dir / "reports" / "call_metrics.json", call_metrics(all_trajectories)
    )
    _write_json(
        run_dir / "reports" / "formal_call_metrics.json",
        call_metrics(scored_trajectories),
    )
    _write_json(
        run_dir / "reports" / "solver_call_metrics.json",
        call_metrics(solver_trajectories),
    )
    _write_json(
        run_dir / "reports" / "control_plane_call_metrics.json",
        call_metrics(control_plane_trajectories),
    )
    _write_json(
        run_dir / "reports" / "infrastructure_call_metrics.json",
        call_metrics(infra_trajectories),
    )
    overall = formal_metrics["overall"]
    tools = formal_metrics["tools"]
    api_calls = formal_metrics["api_calls"]
    cost = formal_metrics["cost"]
    completed = int(summary["completed_tasks"])
    usage_observed = int(overall.get("with_usage_count", 0))
    priced = int(cost.get("priced_trajectory_count", 0))
    expected_scored_trajectories = len(
        list(dict.fromkeys([*formal_ids, *control_plane_ids]))
    )
    solver_usage_observed = int(
        solver_metrics["overall"].get("with_usage_count", 0)
    )
    usage_and_cost_complete = (
        completed > 0
        and len(formal_ids) == completed
        and len(scored_trajectories) == expected_scored_trajectories
        and usage_observed == expected_scored_trajectories
        and priced == expected_scored_trajectories
    )
    api_calls_complete = int(api_calls.get("missing_trajectory_count", 0)) == 0
    scored = {
        "tasks": {
            "requested": summary["requested_tasks"],
            "completed": completed,
            "passed": summary["passed"],
            "model_failed": summary["model_failed"],
            "infrastructure_failed": summary["infrastructure_failed"],
            "pass_rate": summary["pass_rate_all_requested"],
        },
        "scored_usage": overall,
        "scored_cost": cost,
        "scored_api_calls": api_calls,
        "scored_tools": tools,
        "solver_metrics": solver_metrics,
        "control_plane_metrics": control_plane_metrics,
        "control_plane_trajectory_count": len(control_plane_trajectories),
        "runtime_profile_summary": runtime_profile_summary,
        "usage_and_cost_complete": usage_and_cost_complete,
        "complete_estimated_cost_usd": (
            cost.get("estimated_total_usd") if usage_and_cost_complete else None
        ),
        "observed_estimated_cost_usd": cost.get("estimated_total_usd", 0.0),
        "timing": {**timing, "run_wall_time_s": summary["run_wall_time_s"]},
        "transport_proxy": transport_proxy,
        "averages_per_completed_task": {
            "input_tokens": overall.get("input_tokens", 0) / completed
            if usage_and_cost_complete
            else None,
            "cached_input_tokens": overall.get("cached_input_tokens", 0) / completed
            if usage_and_cost_complete
            else None,
            "uncached_input_tokens": overall.get("uncached_input_tokens", 0) / completed
            if usage_and_cost_complete
            else None,
            "output_tokens": overall.get("output_tokens", 0) / completed
            if usage_and_cost_complete
            else None,
            "estimated_cost_usd": cost.get("estimated_total_usd", 0.0) / completed
            if usage_and_cost_complete
            else None,
            "api_calls": api_calls.get("api_call_count", 0) / completed
            if completed and api_calls_complete
            else None,
            "tool_calls": tools.get("tool_call_count", 0) / completed
            if completed
            else None,
        },
        "averages_per_usage_observed_task": {
            "input_tokens": overall.get("input_tokens", 0) / usage_observed
            if usage_observed
            else None,
            "cached_input_tokens": overall.get("cached_input_tokens", 0)
            / usage_observed
            if usage_observed
            else None,
            "uncached_input_tokens": overall.get("uncached_input_tokens", 0)
            / usage_observed
            if usage_observed
            else None,
            "output_tokens": overall.get("output_tokens", 0) / usage_observed
            if usage_observed
            else None,
            "estimated_cost_usd": cost.get("estimated_total_usd", 0.0) / priced
            if priced
            else None,
        },
        "excluded_infrastructure_overhead": {
            "trajectory_count": len(infra_trajectories),
            "metrics": infra_metrics,
            "attempt_wall_time_s": timing[
                "excluded_infrastructure_attempt_wall_time_s"
            ],
        },
        "usage_missing_completed_task_count": completed - solver_usage_observed,
        "generated_at": _now_iso(),
    }
    if qualification_target is not None:
        qualification = _qualification_summary(
            summary=summary,
            observed_cost_usd=float(cost.get("estimated_total_usd", 0.0)),
            cost_complete=usage_and_cost_complete,
            target=qualification_target,
        )
        scored["qualification"] = qualification
        summary["qualification"] = qualification
        _write_json(run_dir / "reports" / "qualification.json", qualification)
        _write_json(run_dir / "reports" / "summary.json", summary)
    _write_json(
        run_dir / "reports" / "runtime_profile_summary.json",
        runtime_profile_summary,
    )
    _write_json(run_dir / "reports" / "scored_metrics_summary.json", scored)
    return scored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--heldout-ids", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--harness-label", required=True)
    parser.add_argument("--harness-dir", type=Path)
    parser.add_argument(
        "--routing-policy-dir", type=Path,
        help="Reasoning-only selection over a static Codex Harness (Empty or RHO); no additional Solver prompt, hook or output-policy changes.",
    )
    parser.add_argument("--paper-harness-id")
    parser.add_argument("--codex-config", type=Path, required=True)
    parser.add_argument(
        "--codex-binary",
        default="codex",
        help=(
            "Codex CLI executable used by Solver and Codex-transport control "
            "agents; defaults to codex on PATH."
        ),
    )
    parser.add_argument(
        "--agent-backend",
        choices=("codex", "claude-code"),
        default="codex",
    )
    parser.add_argument(
        "--codex-base-url-override",
        help="Runtime-only Responses base URL, e.g. a transparent retry proxy.",
    )
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument(
        "--runtime-reasoning-effort-override",
        choices=("minimal", "low", "medium", "high", "xhigh", "max"),
        help=(
            "Force every runtime-selected Solver profile to use one reasoning "
            "effort while preserving its tool budgets and hooks."
        ),
    )
    parser.add_argument("--claude-code-binary", type=Path)
    parser.add_argument(
        "--claude-code-ccr-url",
        default=os.environ.get("HARNESSPRO_CLAUDE_BASE_URL"),
        help="Anthropic-compatible endpoint, or HARNESSPRO_CLAUDE_BASE_URL.",
    )
    parser.add_argument("--claude-code-ccr-metrics-url")
    parser.add_argument(
        "--claude-code-client-key",
        default=os.environ.get("HARNESSPRO_CLAUDE_API_KEY"),
        help="Router client key; defaults to HARNESSPRO_CLAUDE_API_KEY.",
    )
    parser.add_argument("--classifier-anthropic-base-url")
    parser.add_argument("--classifier-anthropic-api-key")
    parser.add_argument("--classifier-anthropic-api-key-env")
    parser.add_argument("--claude-code-fresh-review-client-key")
    parser.add_argument("--claude-code-fresh-review-model")
    parser.add_argument("--claude-code-model-alias", default="sonnet")
    parser.add_argument(
        "--claude-code-system-prompt-mode",
        choices=("append", "replace"),
        default="append",
    )
    parser.add_argument(
        "--claude-code-runtime-context-in-system-prompt",
        action="store_true",
    )
    parser.add_argument(
        "--claude-code-completion-stop-hook",
        action="store_true",
    )
    parser.add_argument(
        "--claude-code-phase-route-profile-key",
        action="store_true",
    )
    parser.add_argument("--claude-code-trace-usage", action="store_true")
    parser.add_argument("--claude-code-inline-task-prompt", action="store_true")
    parser.add_argument("--claude-code-autocompact")
    parser.add_argument("--claude-code-tools", default="Bash,Edit,Read")
    parser.add_argument(
        "--rtk-binary",
        type=Path,
        help=(
            "RTK executable used by runtime-policy hooks; defaults to "
            "RHO_RTK_BIN or third_party/rtk-bin/rtk."
        ),
    )
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--grade-workers", type=int, default=4)
    parser.add_argument("--preflight-workers", type=int, default=4)
    parser.add_argument("--max-infra-attempts", type=int, default=3)
    parser.add_argument(
        "--solver-timeout-s", type=float,
        help="Solver limit in seconds; defaults to 1200 for GPT-5.5 and 7200 for GLM-5.2.",
    )
    parser.add_argument("--transport-state-dir", type=Path)
    parser.add_argument("--transport-max-compensation-s", type=float, default=900.0)
    parser.add_argument(
        "--grader-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment override appended after image Dockerfile ENV exports.",
    )
    parser.add_argument(
        "--solver-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Environment override passed to every Solver invocation. Use this "
            "for audited toolchain paths and package mirrors."
        ),
    )
    parser.add_argument("--docker-command", type=Path, required=True)
    parser.add_argument(
        "--docker-mirror-prefix",
        help=(
            "Optional registry prefix used after an official Docker Hub pull "
            "fails; the mirror image is retagged to the official evaluator name."
        ),
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--repo-cache-mode",
        choices=(CLEAN_HISTORY_CACHE_MODE, FULL_HISTORY_CACHE_MODE),
        default=FULL_HISTORY_CACHE_MODE,
        help=(
            "Repository visibility presented to the Solver. full-history-v1 "
            "is the RHO paper-compatible default; clean-history-v1 is an "
            "explicit leakage-audit mode."
        ),
    )
    parser.add_argument(
        "--work-root", type=Path, default=Path("/tmp/retro-harness-swebench-work")
    )
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-gold-canary", action="store_true")
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument(
        "--task-instructions",
        action="append",
        default=[],
        metavar="TASK_ID=PATH",
        help=(
            "Prepend task-specific instructions loaded from PATH for TASK_ID. "
            "May be repeated; intended for retrieval baselines whose frozen "
            "memory differs for each held-out task."
        ),
    )
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument(
        "--classify-only",
        action="store_true",
        help="Run and report only the runtime-profile classifier.",
    )
    parser.add_argument("--target-requested-tasks", type=int)
    parser.add_argument("--target-min-passed", type=int)
    parser.add_argument("--target-baseline-label")
    parser.add_argument("--target-baseline-cost-usd", type=float)
    parser.add_argument("--target-max-cost-fraction", type=float)
    args = parser.parse_args()
    if args.solver_timeout_s is None:
        args.solver_timeout_s = 7200.0 if args.model.lower() == "glm-5.2" else 1200.0
    return args


def _load_task_instructions(
    values: Iterable[str],
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    instructions: dict[str, str] = {}
    manifest: dict[str, dict[str, str]] = {}
    for raw in values:
        task_id, separator, raw_path = raw.partition("=")
        if not separator or not task_id.strip() or not raw_path.strip():
            raise ValueError(
                "--task-instructions values must have the form TASK_ID=PATH"
            )
        task_id = task_id.strip()
        if task_id in instructions:
            raise ValueError(f"duplicate --task-instructions task id: {task_id}")
        path = Path(raw_path.strip()).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"task instructions file does not exist: {path}")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"task instructions file is empty: {path}")
        instructions[task_id] = text
        manifest[task_id] = {
            "path": str(path),
            "sha256": _file_sha256(path),
        }
    return instructions, manifest


def _capture_empty_routing_policy(source: Path, run_dir: Path) -> HarnessRuntimePolicy:
    """Freeze a tool-free router outside the Solver's static Harness/workspace.

    The legacy name remains for frozen Empty-run/test compatibility. The
    captured router can also select reasoning for an unchanged static RHO.
    """
    source = source.resolve()
    policy = load_runtime_policy(source)
    if policy is None:
        raise ValueError("routing policy must contain runtime/policy.json")
    raw = _read_json(policy.source)
    required_false = (
        "routing_runtime_hooks", "routing_tool_output_compaction",
        "routing_stage_guidance", "routing_tool_budget_enforcement",
    )
    if any(raw.get(key) is not False for key in required_false):
        raise ValueError("reasoning-only routing must explicitly disable all runtime interventions")
    if (raw.get("codex") or policy.session_context or policy.rules
            or policy.adjuster or policy.fresh_review or policy.confidence_overrides
            or policy.rtk_enabled or policy.external_profile_model_instructions_files
            or any(p.model is not None for p in policy.profiles.values())):
        raise ValueError("reasoning-only routing cannot inject instructions, tools, rules or model routing")
    classifier = policy.classifier
    if classifier is None or classifier.transport != "responses" or classifier.samples != 1:
        raise ValueError("reasoning-only routing requires one tool-free Responses classification")
    inventory = {str(p.relative_to(source)): _file_sha256(p)
                 for p in source.rglob("*") if p.is_file()}
    if any(p.is_symlink() for p in source.rglob("*")):
        raise ValueError("routing policy cannot contain symlinks")
    captured = run_dir / "routing_policy"
    if captured.exists():
        actual = {str(p.relative_to(captured)): _file_sha256(p)
                  for p in captured.rglob("*") if p.is_file()}
        if inventory != actual:
            raise ValueError("run-dir routing policy differs from source; use a new run directory")
    else:
        shutil.copytree(source, captured)
    return load_runtime_policy(captured)


def _responses_base_url(codex_config_path: Path) -> str:
    config = tomllib.loads(codex_config_path.read_text(encoding="utf-8"))
    provider = config.get("model_providers", {}).get(config.get("model_provider"), {})
    base_url = provider.get("base_url")
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("Codex config must identify a provider with base_url")
    return base_url.strip()


def _solve_instructions_for_task(
    task_id: str,
    *,
    runtime_policy: HarnessRuntimePolicy | None,
    task_instructions: dict[str, str],
) -> str:
    base = (
        RUNTIME_HARNESS_SOLVE_INSTRUCTIONS
        if runtime_policy is not None
        else SOLVE_INSTRUCTIONS
    )
    prefix = task_instructions.get(task_id)
    return f"{prefix}\n\n{base}" if prefix else base


def _runtime_reasoning_efforts(
    runtime_policy: HarnessRuntimePolicy | None,
    default_effort: str,
    override_effort: str | None = None,
) -> list[str]:
    if runtime_policy is None:
        return [default_effort]
    if override_effort is not None:
        return [override_effort]
    efforts = {
        *(profile.reasoning_effort for profile in runtime_policy.profiles.values()),
        *(
            override.profile.reasoning_effort
            for override in runtime_policy.confidence_overrides
        ),
    }
    if (
        runtime_policy.fresh_review is not None
        and runtime_policy.fresh_review.reasoning_effort is not None
    ):
        efforts.add(runtime_policy.fresh_review.reasoning_effort)
    if (
        runtime_policy.fresh_review is not None
        and getattr(
            runtime_policy.fresh_review, "judge_reasoning_effort", None
        )
        is not None
    ):
        efforts.add(runtime_policy.fresh_review.judge_reasoning_effort)
    return sorted(efforts)


def _validate_reasoning_only_routing(
    args: argparse.Namespace, harness_runtime_policy: HarnessRuntimePolicy | None = None,
) -> None:
    if args.routing_policy_dir is None:
        return
    if (
        args.agent_backend != "codex"
        or args.task_instructions or args.runtime_reasoning_effort_override is not None
    ):
        raise SystemExit("--routing-policy-dir requires Codex without task-specific instructions or reasoning overrides")
    if harness_runtime_policy is not None:
        raise SystemExit("--routing-policy-dir requires a static Harness; cannot replace its runtime policy/hooks")


def main() -> int:
    args = parse_args()
    if args.agent_backend == "claude-code":
        if not args.claude_code_ccr_url or not args.claude_code_client_key:
            raise SystemExit(
                "Claude Code requires HARNESSPRO_CLAUDE_BASE_URL and "
                "HARNESSPRO_CLAUDE_API_KEY, or their explicit CLI options. "
                "See docs/claude-code.md."
            )
    _validate_reasoning_only_routing(args)
    for name in (
        "concurrency",
        "grade_workers",
        "preflight_workers",
        "max_infra_attempts",
    ):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.solver_timeout_s <= 0:
        raise SystemExit("--solver-timeout-s must be positive")
    if args.transport_max_compensation_s < 0:
        raise SystemExit("--transport-max-compensation-s must be non-negative")
    if args.agent_backend == "claude-code" and args.claude_code_binary is None:
        raise SystemExit("--claude-code-binary is required for Claude Code")
    if bool(args.claude_code_fresh_review_client_key) != bool(
        args.claude_code_fresh_review_model
    ):
        raise SystemExit(
            "--claude-code-fresh-review-client-key and "
            "--claude-code-fresh-review-model must be provided together"
        )
    target_values = (
        args.target_requested_tasks,
        args.target_min_passed,
        args.target_baseline_label,
        args.target_baseline_cost_usd,
        args.target_max_cost_fraction,
    )
    if any(value is not None for value in target_values) and not all(
        value is not None for value in target_values
    ):
        raise SystemExit("all --target-* arguments must be provided together")
    if args.target_requested_tasks is not None:
        if args.target_requested_tasks <= 0:
            raise SystemExit("--target-requested-tasks must be positive")
        if not 0 <= args.target_min_passed <= args.target_requested_tasks:
            raise SystemExit("--target-min-passed is outside the task range")
        if args.target_baseline_cost_usd <= 0:
            raise SystemExit("--target-baseline-cost-usd must be positive")
        if not 0 < args.target_max_cost_fraction <= 1:
            raise SystemExit("--target-max-cost-fraction must be in (0, 1]")

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(exist_ok=True)
    run_lock = _acquire_run_lock(run_dir)
    try:
        return _main_locked(args, run_dir)
    finally:
        fcntl.flock(run_lock.fileno(), fcntl.LOCK_UN)
        run_lock.close()


def _main_locked(args: argparse.Namespace, run_dir: Path) -> int:
    docker_info = _configure_docker_command(args.docker_command)
    dataset_path = args.dataset.resolve()
    heldout_path = args.heldout_ids.resolve()
    codex_config_path = args.codex_config.resolve()
    cache_root = args.cache_root.resolve()
    work_root = args.work_root.resolve() / run_dir.name
    transport_state_dir = (
        args.transport_state_dir.resolve()
        if args.transport_state_dir is not None
        else None
    )
    grader_env_overrides = _parse_env_overrides(
        args.grader_env, description="grader"
    )
    solver_env_overrides = _parse_env_overrides(
        args.solver_env, description="solver"
    )
    task_instructions, task_instructions_manifest = _load_task_instructions(
        args.task_instructions
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    if transport_state_dir is not None:
        transport_state_dir.mkdir(parents=True, exist_ok=True)

    harness_store = FilesystemHarnessStore(run_dir / "harness")
    empty_harness = harness_store.empty()
    harness = (
        harness_store.capture(args.harness_dir.resolve())
        if args.harness_dir is not None
        else empty_harness
    )
    captured_harness_root = harness_store.root / harness.id
    harness_runtime_policy = load_runtime_policy(captured_harness_root)
    _validate_reasoning_only_routing(args, harness_runtime_policy)
    routing_policy = (
        _capture_empty_routing_policy(args.routing_policy_dir, run_dir)
        if args.routing_policy_dir is not None else None
    )
    runtime_policy = routing_policy or harness_runtime_policy
    runtime_hooks_enabled = harness_runtime_policy is not None
    runtime_state_root = run_dir / "runtime_state"
    runtime_hook_script: Path | None = None
    runtime_rtk_binary: Path | None = None
    if runtime_hooks_enabled:
        runtime_hook_script = captured_harness_root / "runtime" / "hook.py"
        if not runtime_hook_script.is_file():
            raise RuntimeError("runtime policy requires harness/runtime/hook.py")
        if runtime_policy.adjuster is not None:
            raise RuntimeError(
                "SWE-bench Pro runtime profile adjusters are not supported"
            )
        if runtime_policy.rtk_enabled:
            configured_rtk = args.rtk_binary or os.environ.get("RHO_RTK_BIN")
            runtime_rtk_binary = (
                Path(configured_rtk).expanduser().resolve()
                if configured_rtk
                else (
                    Path(__file__).resolve().parents[1]
                    / "third_party"
                    / "rtk-bin"
                    / "rtk"
                )
            )
            if not runtime_rtk_binary.is_file() or not os.access(runtime_rtk_binary, os.X_OK):
                raise RuntimeError(
                    f"RTK is missing or not executable: {runtime_rtk_binary}. "
                    "Run python scripts/prepare_runtime.py after unpacking the release."
                )
    source_cache_root = cache_root / FULL_HISTORY_CACHE_DIR
    repo_cache = (
        CleanRepoCache(source_cache_root)
        if args.repo_cache_mode == CLEAN_HISTORY_CACHE_MODE
        else RepoCache(source_cache_root)
    )
    if repo_cache.cache_mode == CLEAN_HISTORY_CACHE_MODE:
        if args.agent_backend != "codex":
            raise RuntimeError(
                "clean-history-v1 currently requires agent-backend=codex so "
                "shell-command network isolation can be enforced"
            )
    rows = load_rows(str(dataset_path))
    dataset = SWEbenchProDataset.from_records(
        rows,
        harness_store=harness_store,
        docker_pull="missing",
        seed=0,
        repo_cache=repo_cache,
        eval_assets_root=cache_root / "SWE-bench_Pro-os",
        grader_env_overrides=grader_env_overrides,
    )
    heldout_ids = [
        line.strip()
        for line in heldout_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    paper_tasks = list(dataset.val)[: len(heldout_ids)]
    val_ids = [task.id for task in paper_tasks]
    if val_ids != heldout_ids:
        raise RuntimeError(
            "held-out IDs do not exactly match the deterministic paper pool: "
            f"manifest={len(heldout_ids)} dataset={len(val_ids)}"
        )
    task_map = {task.id: task for task in paper_tasks}
    selected_ids = heldout_ids
    if args.task_id:
        unknown = sorted(set(args.task_id) - set(task_map))
        if unknown:
            raise RuntimeError(f"unknown --task-id values: {unknown}")
        selected_ids = [task_id for task_id in heldout_ids if task_id in args.task_id]
    unknown_task_instructions = sorted(set(task_instructions) - set(task_map))
    if unknown_task_instructions:
        raise RuntimeError(
            "unknown --task-instructions task ids: "
            f"{unknown_task_instructions}"
        )
    if args.max_tasks is not None:
        selected_ids = selected_ids[: args.max_tasks]
    selected_tasks = [task_map[task_id] for task_id in selected_ids]
    if not selected_tasks:
        raise RuntimeError("no tasks selected")
    qualification_target = (
        {
            "requested_tasks": args.target_requested_tasks,
            "minimum_passed": args.target_min_passed,
            "baseline_label": args.target_baseline_label,
            "baseline_cost_usd": args.target_baseline_cost_usd,
            "maximum_cost_fraction": args.target_max_cost_fraction,
        }
        if args.target_requested_tasks is not None
        else None
    )

    config = {
        "dataset": str(dataset_path),
        "dataset_sha256": _file_sha256(dataset_path),
        "dataset_row_count": len(rows),
        "heldout_ids": str(heldout_path),
        "heldout_ids_sha256": _file_sha256(heldout_path),
        "heldout_count": len(heldout_ids),
        "selected_task_ids": selected_ids,
        "task_instructions": task_instructions_manifest,
        "harness_label": args.harness_label,
        "harness_id": harness.id,
        "harness_source": str(args.harness_dir.resolve())
        if args.harness_dir
        else "h_empty",
        "paper_harness_id": args.paper_harness_id,
        "agent_backend": args.agent_backend,
        "model": args.model,
        "reasoning_effort": "dynamic" if runtime_policy else args.reasoning_effort,
        "default_reasoning_effort_arg": args.reasoning_effort,
        "runtime_reasoning_effort_override": args.runtime_reasoning_effort_override,
        "runtime_policy": (
            str(runtime_policy.source) if runtime_policy is not None else None
        ),
        "runtime_policy_sha256": (
            _file_sha256(runtime_policy.source)
            if runtime_policy is not None
            else None
        ),
        "runtime_policy_mode": "routing-only" if routing_policy else (
            "harness-runtime" if harness_runtime_policy else None),
        "routing_policy_source": str(args.routing_policy_dir.resolve()) if routing_policy else None,
        "routing_policy_capture": str(run_dir / "routing_policy") if routing_policy else None,
        "runtime_profiles": (
            {
                name: {
                    "reasoning_effort": profile.reasoning_effort,
                    "economy_after_tools": profile.economy_after_tools,
                    "strict_after_tools": profile.strict_after_tools,
                    "finalize_after_tools": profile.finalize_after_tools,
                }
                for name, profile in sorted(runtime_policy.profiles.items())
            }
            if runtime_policy is not None
            else None
        ),
        "runtime_classifier": (
            {
                "model": runtime_policy.classifier.model,
                "reasoning_effort": runtime_policy.classifier.reasoning_effort,
                "transport": runtime_policy.classifier.transport,
                "fallback_profile": runtime_policy.classifier.fallback_profile,
                "rubric_sha256": _file_sha256(
                    runtime_policy.classifier.rubric_file
                ),
                "samples": runtime_policy.classifier.samples,
            }
            if runtime_policy is not None
            and runtime_policy.classifier is not None
            else None
        ),
        "runtime_fresh_review": (
            {
                "min_tools": runtime_policy.fresh_review.min_tools,
                "max_tools": runtime_policy.fresh_review.max_tools,
                "timeout_s": runtime_policy.fresh_review.timeout_s,
                "profiles": list(runtime_policy.fresh_review.profiles),
                "reasoning_effort": (
                    runtime_policy.fresh_review.reasoning_effort
                ),
                "review_tool_budget": (
                    runtime_policy.fresh_review.review_tool_budget
                ),
                "instructions_file": (
                    str(runtime_policy.fresh_review.instructions_file)
                    if runtime_policy.fresh_review.instructions_file is not None
                    else None
                ),
                "instructions_sha256": (
                    _file_sha256(runtime_policy.fresh_review.instructions_file)
                    if runtime_policy.fresh_review.instructions_file is not None
                    else None
                ),
                "mode": runtime_policy.fresh_review.mode,
                "judge_reasoning_effort": (
                    runtime_policy.fresh_review.judge_reasoning_effort
                ),
                "judge_transport": (
                    "anthropic_messages_direct"
                    if args.agent_backend == "claude-code"
                    and runtime_policy.fresh_review.mode == "independent_select"
                    else "agent_backend"
                ),
                "judge_max_output_tokens": (
                    4096
                    if args.agent_backend == "claude-code"
                    and runtime_policy.fresh_review.mode == "independent_select"
                    else None
                ),
                "profile_tool_ranges": {
                    name: {"min_tools": minimum, "max_tools": maximum}
                    for name, minimum, maximum in (
                        runtime_policy.fresh_review.profile_tool_ranges
                    )
                },
            }
            if runtime_policy is not None
            and runtime_policy.fresh_review is not None
            else None
        ),
        "runtime_rtk_binary": (
            str(runtime_rtk_binary) if runtime_rtk_binary is not None else None
        ),
        "codex_binary": args.codex_binary if args.agent_backend == "codex" else None,
        "codex_real_binary": (
            os.environ.get("RHO_CDX_REAL_CODEX")
            if args.agent_backend == "codex"
            else None
        ),
        "codex_cli_version": (
            os.environ.get("RHO_CODEX_CLI_VERSION")
            if args.agent_backend == "codex"
            else None
        ),
        "codex_config": str(codex_config_path),
        "codex_config_sha256": _file_sha256(codex_config_path),
        "codex_base_url_override": args.codex_base_url_override,
        "claude_code": (
            {
                "binary": str(args.claude_code_binary.resolve()),
                "ccr_url": args.claude_code_ccr_url,
                "ccr_metrics_url": args.claude_code_ccr_metrics_url,
                "client_key_configured": bool(args.claude_code_client_key),
                "fresh_review_client_key_configured": bool(
                    args.claude_code_fresh_review_client_key
                ),
                "fresh_review_model": args.claude_code_fresh_review_model,
                "model_alias": args.claude_code_model_alias,
                "trace_usage": args.claude_code_trace_usage,
                "system_prompt_mode": args.claude_code_system_prompt_mode,
                "runtime_context_in_system_prompt": (
                    args.claude_code_runtime_context_in_system_prompt
                ),
                "completion_stop_hook": args.claude_code_completion_stop_hook,
                "phase_route_profile_key": args.claude_code_phase_route_profile_key,
                "inline_task_prompt": args.claude_code_inline_task_prompt,
                "autocompact": args.claude_code_autocompact,
                "tools": args.claude_code_tools,
            }
            if args.agent_backend == "claude-code"
            else None
        ),
        "concurrency": args.concurrency,
        "grade_workers": args.grade_workers,
        "preflight_workers": args.preflight_workers,
        "max_infra_attempts": args.max_infra_attempts,
        "solver_timeout_s": args.solver_timeout_s,
        "transport_state_dir": str(transport_state_dir)
        if transport_state_dir
        else None,
        "transport_max_compensation_s": args.transport_max_compensation_s,
        "solver_env_overrides": solver_env_overrides,
        "grader_env_overrides": grader_env_overrides,
        "grader_timeout_s": SWEbenchProDockerEvaluator().timeout_s,
        "docker": docker_info,
        "docker_mirror_prefix": args.docker_mirror_prefix,
        "cache_root": str(cache_root),
        "repo_cache_mode": repo_cache.cache_mode,
        "repo_cache_path": str(repo_cache.root.resolve()),
        "solver_network_access": repo_cache.cache_mode != CLEAN_HISTORY_CACHE_MODE,
        "solver_permission_profile": (
            CLEAN_SOLVER_PERMISSION_PROFILE
            if repo_cache.cache_mode == CLEAN_HISTORY_CACHE_MODE
            and args.agent_backend == "codex"
            else None
        ),
        "git_network": _git_network_report(repo_cache),
        "work_root": str(work_root),
        "qualification_target": qualification_target,
        "start_timestamp": _now_iso(),
    }
    config_path = run_dir / "config.json"
    existing_config = _read_json(config_path)
    if existing_config:
        identity_keys = (
            "dataset_sha256",
            "heldout_ids_sha256",
            "selected_task_ids",
            "task_instructions",
            "harness_id",
            "agent_backend",
            "model",
            "reasoning_effort",
            "default_reasoning_effort_arg",
            "runtime_reasoning_effort_override",
            "runtime_policy_sha256",
            "runtime_policy_mode",
            "runtime_profiles",
            "runtime_classifier",
            "runtime_fresh_review",
            "runtime_rtk_binary",
            "codex_binary",
            "qualification_target",
            "codex_config_sha256",
            "solver_timeout_s",
            "codex_base_url_override",
            "claude_code",
            "transport_max_compensation_s",
            "solver_env_overrides",
            "grader_env_overrides",
            "repo_cache_mode",
            "repo_cache_path",
            "solver_permission_profile",
        )
        mismatches = []
        for key in identity_keys:
            existing_value = existing_config.get(key)
            current_value = config.get(key)
            if key == "runtime_policy_mode" and key not in existing_config:
                existing_value = "harness-runtime" if existing_config.get("runtime_policy") else None
            if key == "agent_backend":
                existing_value = existing_value or "codex"
            if key in ("solver_env_overrides", "grader_env_overrides"):
                existing_value = existing_value or {}
                current_value = current_value or {}
            if existing_value != current_value:
                mismatches.append(key)
        if mismatches:
            raise RuntimeError(f"run-dir config mismatch for keys: {mismatches}")
        config["start_timestamp"] = existing_config.get(
            "start_timestamp", config["start_timestamp"]
        )
    _write_json(config_path, config)
    _write_json(
        run_dir / "reports" / "solver_environment.json",
        _solver_environment_report(solver_env_overrides),
    )
    shutil.copy2(codex_config_path, run_dir / "codex_config.base.toml")
    provider_codex_config_path = codex_config_path
    if args.codex_base_url_override:
        provider_codex_config_path = _render_codex_base_url_override(
            codex_config_path,
            output=run_dir / "codex_config.provider.toml",
            base_url=args.codex_base_url_override,
        )
    solver_codex_config_path = provider_codex_config_path
    if transport_state_dir is not None:
        solver_codex_config_path = _render_codex_transport_trace_header(
            provider_codex_config_path,
            output=run_dir / "codex_config.solver.toml",
        )
    runtime_codex_config_path = solver_codex_config_path
    if runtime_hooks_enabled:
        assert runtime_hook_script is not None
        runtime_codex_config_path = render_runtime_codex_config(
            solver_codex_config_path,
            output=run_dir / "codex_config.toml",
            policy=runtime_policy,
            hook_script=runtime_hook_script,
            rtk_binary=runtime_rtk_binary,
            runtime_state_root=runtime_state_root,
        )
    else:
        shutil.copy2(solver_codex_config_path, run_dir / "codex_config.toml")
        runtime_codex_config_path = run_dir / "codex_config.toml"
    if (
        repo_cache.cache_mode == CLEAN_HISTORY_CACHE_MODE
        and args.agent_backend == "codex"
    ):
        runtime_codex_config_path = _render_codex_clean_permission_profile(
            runtime_codex_config_path,
            output=run_dir / "codex_config.clean.toml",
            readable_roots=(
                _codex_runtime_root(args.codex_binary),
                Path(__file__).resolve().parents[1]
                / "scripts"
                / "swebench-pro-toolchain-bin",
                Path(__file__).resolve().parents[1] / "third_party" / "toolchains",
                *((runtime_rtk_binary.parent,) if runtime_rtk_binary else ()),
            ),
        )

    if not args.skip_preflight and not args.classify_only:
        _run_preflight(
            selected_tasks,
            cache_root=cache_root,
            repo_cache=repo_cache,
            workers=args.preflight_workers,
            mirror_prefix=args.docker_mirror_prefix,
            report_path=run_dir / "reports" / "preflight.json",
        )
    preflight_report = _read_json(run_dir / "reports" / "preflight.json") or {}
    official_go_versions = {
        str(record["image"]): str(record["go_toolchain"]["version"])
        for record in (
            ((preflight_report.get("images") or {}).get("records")) or []
        )
        if (record.get("go_toolchain") or {}).get("ok")
        and (record.get("go_toolchain") or {}).get("version")
    }
    _write_json(
        run_dir / "reports" / "official_go_toolchains.json",
        {
            "source": "pinned official evaluator Docker images",
            "by_image": dict(sorted(official_go_versions.items())),
            "generated_at": _now_iso(),
        },
    )
    if not args.skip_gold_canary and not args.classify_only:
        # Keep the infrastructure canary invariant across targeted/subset runs.
        # Some benchmark gold patches are not accepted verbatim by the pinned
        # evaluator; selecting a canary from the task filter would make an
        # otherwise identical run depend on which task happened to come first.
        _run_gold_canary(paper_tasks[0], run_dir)

    configure_global_codex_pool(args.concurrency)
    reasoning_efforts = _runtime_reasoning_efforts(
        runtime_policy,
        args.reasoning_effort,
        args.runtime_reasoning_effort_override,
    )
    if args.agent_backend == "codex":
        clean_solver = repo_cache.cache_mode == CLEAN_HISTORY_CACHE_MODE
        agents: dict[str, Any] = {
            effort: CodexAgent(
                codex_config_path=runtime_codex_config_path,
                model=args.model,
                reasoning_effort=effort,
                binary=args.codex_binary,
                sandbox=None if clean_solver else "danger-full-access",
                permission_profile=(
                    CLEAN_SOLVER_PERMISSION_PROFILE if clean_solver else None
                ),
                network_access=not clean_solver,
                isolate_codex_home=True,
                forward_openai_api_key=_codex_config_uses_openai_api_key(
                    runtime_codex_config_path
                ),
                ephemeral=True,
                extra_flags=(
                    ("--dangerously-bypass-hook-trust",)
                    if runtime_hooks_enabled
                    else ()
                ),
                default_timeout_s=args.solver_timeout_s,
            )
            for effort in reasoning_efforts
        }
    else:
        assert args.claude_code_binary is not None
        agents = {
            effort: ClaudeCodeAgent(
                binary=args.claude_code_binary,
                ccr_base_url=args.claude_code_ccr_url,
                ccr_metrics_base_url=args.claude_code_ccr_metrics_url,
                ccr_api_key=args.claude_code_client_key,
                model_alias=args.claude_code_model_alias,
                model=args.model,
                reasoning_effort=effort,
                inject_harness_context=harness.id != empty_harness.id,
                runtime_hook_script=(
                    runtime_hook_script if runtime_policy is not None else None
                ),
                rtk_binary=(
                    runtime_rtk_binary if runtime_policy is not None else None
                ),
                system_prompt_mode=args.claude_code_system_prompt_mode,
                include_runtime_context_in_system_prompt=(
                    args.claude_code_runtime_context_in_system_prompt
                ),
                enable_completion_stop_hook=(
                    args.claude_code_completion_stop_hook
                ),
                phase_route_profile_key=args.claude_code_phase_route_profile_key,
                inline_task_prompt=args.claude_code_inline_task_prompt,
                autocompact=args.claude_code_autocompact,
                tools=args.claude_code_tools,
                trace_usage=args.claude_code_trace_usage,
                default_timeout_s=args.solver_timeout_s,
            )
            for effort in reasoning_efforts
        }
    dedicated_claude_review_agent: Agent | None = None
    if (
        args.agent_backend == "claude-code"
        and runtime_policy is not None
        and runtime_policy.fresh_review is not None
        and args.claude_code_fresh_review_client_key is not None
    ):
        assert args.claude_code_binary is not None
        assert args.claude_code_fresh_review_model is not None
        review_effort = (
            runtime_policy.fresh_review.reasoning_effort
            or args.reasoning_effort
        )
        dedicated_claude_review_agent = ClaudeCodeAgent(
            binary=args.claude_code_binary,
            ccr_base_url=args.claude_code_ccr_url,
            ccr_metrics_base_url=args.claude_code_ccr_metrics_url,
            ccr_api_key=args.claude_code_fresh_review_client_key,
            model_alias=args.claude_code_model_alias,
            model=args.claude_code_fresh_review_model,
            reasoning_effort=review_effort,
            inject_harness_context=harness.id != empty_harness.id,
            runtime_hook_script=runtime_hook_script,
            rtk_binary=runtime_rtk_binary,
            system_prompt_mode=args.claude_code_system_prompt_mode,
            include_runtime_context_in_system_prompt=(
                args.claude_code_runtime_context_in_system_prompt
            ),
            enable_completion_stop_hook=(
                args.claude_code_completion_stop_hook
            ),
            phase_route_profile_key=False,
            inline_task_prompt=False,
            autocompact=args.claude_code_autocompact,
            tools=args.claude_code_tools,
            trace_usage=args.claude_code_trace_usage,
            default_timeout_s=args.solver_timeout_s,
        )
    dedicated_candidate_judge: Agent | None = None
    if (
        args.agent_backend == "claude-code"
        and runtime_policy is not None
        and runtime_policy.fresh_review is not None
        and runtime_policy.fresh_review.mode == "independent_select"
    ):
        assert args.claude_code_ccr_url is not None
        dedicated_candidate_judge = AnthropicMessagesAgent(
            base_url=args.claude_code_ccr_url,
            api_key=args.claude_code_client_key,
            model=args.model,
            reasoning_effort=(
                runtime_policy.fresh_review.judge_reasoning_effort
                or runtime_policy.fresh_review.reasoning_effort
                or args.reasoning_effort
            ),
            default_timeout_s=min(runtime_policy.fresh_review.timeout_s, 240.0),
            max_output_tokens=4096,
        )
    if (
        runtime_policy is not None and runtime_policy.fresh_review is not None
    ):
        fresh_review = runtime_policy.fresh_review
        base_agents = agents
        wrapped_agents: dict[Any, Agent] = {}
        for key, agent in base_agents.items():
            review_agent: Agent | None = dedicated_claude_review_agent
            if (
                review_agent is None
                and fresh_review.reasoning_effort is not None
            ):
                review_key: Any = fresh_review.reasoning_effort
                review_agent = base_agents.get(review_key)
                if review_agent is None:
                    raise ValueError(
                        "fresh_review reasoning effort has no matching agent for "
                        f"{key!r}"
                    )
            judge_agent = dedicated_candidate_judge or review_agent
            if (
                dedicated_candidate_judge is None
                and dedicated_claude_review_agent is None
                and fresh_review.judge_reasoning_effort is not None
            ):
                judge_key: Any = fresh_review.judge_reasoning_effort
                judge_agent = base_agents.get(judge_key)
                if judge_agent is None:
                    raise ValueError(
                        "fresh_review judge reasoning effort has no matching "
                        f"agent for {key!r}"
                    )
            wrapped_agents[key] = _FreshReviewAgent(
                agent,
                min_tools=fresh_review.min_tools,
                max_tools=fresh_review.max_tools,
                timeout_s=fresh_review.timeout_s,
                profiles=fresh_review.profiles,
                review_tool_budget=fresh_review.review_tool_budget,
                profile_tool_ranges=fresh_review.profile_tool_ranges,
                review_agent=review_agent,
                mode=fresh_review.mode,
                judge_agent=judge_agent,
                review_instructions=fresh_review.instructions,
            )
        agents = wrapped_agents
    classifier_agent: Agent | None = None
    if runtime_policy is not None and runtime_policy.classifier is not None:
        classifier = runtime_policy.classifier
        if classifier.transport == "responses":
            classifier_agent = ResponsesAgent(
                base_url=_responses_base_url(provider_codex_config_path),
                model=classifier.model,
                reasoning_effort=classifier.reasoning_effort,
                prompt_cache_key=f"rho-swebench-profile:{harness.id}",
                prompt_cache_retention="24h",
                default_timeout_s=classifier.timeout_s,
                max_output_tokens=classifier.max_output_tokens,
            )
        elif classifier.transport == "anthropic":
            if not args.classifier_anthropic_base_url:
                raise RuntimeError(
                    "an Anthropic classifier requires "
                    "--classifier-anthropic-base-url"
                )
            classifier_api_key = args.classifier_anthropic_api_key
            if args.classifier_anthropic_api_key_env:
                classifier_api_key = os.environ.get(
                    args.classifier_anthropic_api_key_env
                )
            if not classifier_api_key:
                raise RuntimeError(
                    "an Anthropic classifier requires an API key or key env"
                )
            classifier_agent = AnthropicMessagesAgent(
                base_url=args.classifier_anthropic_base_url,
                api_key=classifier_api_key,
                model=classifier.model,
                reasoning_effort=classifier.reasoning_effort,
                default_timeout_s=classifier.timeout_s,
                max_output_tokens=classifier.max_output_tokens,
            )
        elif classifier.transport == "codex":
            classifier_agent = CodexAgent(
                codex_config_path=provider_codex_config_path,
                model=classifier.model,
                reasoning_effort=classifier.reasoning_effort,
                binary=args.codex_binary,
                sandbox="read-only",
                network_access=False,
                isolate_codex_home=True,
                forward_openai_api_key=_codex_config_uses_openai_api_key(
                    provider_codex_config_path
                ),
                ephemeral=True,
                default_timeout_s=classifier.timeout_s,
            )
        else:
            raise RuntimeError(
                "SWE-bench Pro runtime classifier supports responses, codex or anthropic "
                f"transport, not {classifier.transport!r}"
            )
    traj_store = FilesystemTrajectoryStore(run_dir / "trajectories")

    if args.classify_only:
        if runtime_policy is None or classifier_agent is None:
            raise RuntimeError("--classify-only requires a model runtime classifier")
        classification_started = time.monotonic()

        def classify_one(task: Any) -> dict[str, Any]:
            selection = _classify_runtime_profile(
                task=task,
                policy=runtime_policy,
                classifier_agent=classifier_agent,
                run_dir=run_dir,
                traj_store=traj_store,
                harness_id=harness.id,
            )
            payload = runtime_policy.workspace_selection_payload(selection)
            print(
                f"[profile] {task.id} profile={selection.profile.name} "
                f"source={selection.source} confidence={selection.confidence}",
                flush=True,
            )
            return {"task_id": task.id, **payload}

        classifications: list[dict[str, Any]] = []
        with ThreadPoolExecutor(
            max_workers=min(args.concurrency, len(selected_tasks))
        ) as pool:
            futures = [pool.submit(classify_one, task) for task in selected_tasks]
            for future in as_completed(futures):
                classifications.append(future.result())
        classifications.sort(key=lambda item: item["task_id"])
        classifier_trajectories = _load_trajectories(
            traj_store, _classification_trajectory_ids(run_dir)
        )
        metrics = usage_summary(classifier_trajectories)
        report = {
            "harness_id": harness.id,
            "harness_label": args.harness_label,
            "task_count": len(classifications),
            "profile_counts": dict(
                sorted(Counter(item["profile"] for item in classifications).items())
            ),
            "classification_source_counts": dict(
                sorted(
                    Counter(
                        item["classification_source"] for item in classifications
                    ).items()
                )
            ),
            "wall_time_s": round(time.monotonic() - classification_started, 6),
            "metrics": metrics,
            "classifications": classifications,
        }
        _write_json(run_dir / "reports" / "classification_summary.json", report)
        print(
            "CLASSIFICATION_SUMMARY "
            + json.dumps(
                {
                    "task_count": report["task_count"],
                    "profile_counts": report["profile_counts"],
                    "estimated_cost_usd": metrics["cost"].get(
                        "estimated_total_usd", 0.0
                    ),
                }
            ),
            flush=True,
        )
        return 0

    grade_gate = threading.Semaphore(args.grade_workers)
    run_started = time.monotonic()
    progress_lock = threading.Lock()
    counters = {"started": 0, "completed": 0, "active": 0}

    print(
        f"[run] harness={args.harness_label}:{harness.id} tasks={len(selected_ids)} "
        f"backend={args.agent_backend} model={args.model}/"
        f"{'dynamic:' + ','.join(reasoning_efforts) if runtime_policy else args.reasoning_effort} "
        f"grade_workers={args.grade_workers}",
        flush=True,
    )

    def evaluate_task(task: Any) -> dict[str, Any]:
        with progress_lock:
            counters["started"] += 1
            counters["active"] += 1
            print(
                f"[start {counters['started']}/{len(selected_tasks)}] {task.id} "
                f"active={counters['active']} queued={len(selected_tasks) - counters['started']}",
                flush=True,
            )
        task_dir = run_dir / "tasks" / _safe_task_id(task.id)
        result_path = task_dir / "result.json"
        existing = _load_completed_result(result_path)
        if existing is not None:
            with progress_lock:
                counters["completed"] += 1
                counters["active"] -= 1
                print(
                    f"[task {counters['completed']}/{len(selected_tasks)}] RESUME {task.id} "
                    f"passed={existing.get('passed')} active={counters['active']}",
                    flush=True,
                )
            return existing

        task_dir.mkdir(parents=True, exist_ok=True)
        progress = _read_json(task_dir / "progress.json") or {}
        attempts: list[dict[str, Any]] = list(progress.get("attempts") or [])
        task_started = time.monotonic()
        final_result: dict[str, Any] | None = None
        runtime_selection: ProfileSelection | None = None
        if runtime_policy is not None:
            runtime_selection = (
                _classify_runtime_profile(
                    task=task,
                    policy=runtime_policy,
                    classifier_agent=classifier_agent,
                    run_dir=run_dir,
                    traj_store=traj_store,
                    harness_id=harness.id,
                )
                if classifier_agent is not None
                else runtime_policy.select(task.query())
            )
            with progress_lock:
                print(
                    f"[profile] {task.id} profile={runtime_selection.profile.name} "
                    f"source={runtime_selection.source} "
                    f"confidence={runtime_selection.confidence}",
                    flush=True,
                )
        task_agent = (
            agents[
                args.runtime_reasoning_effort_override
                or runtime_selection.profile.reasoning_effort
            ]
            if runtime_selection is not None
            else agents[args.reasoning_effort]
        )
        runtime_selection_payload = (
            runtime_policy.workspace_selection_payload(
                runtime_selection,
                reasoning_effort_override=args.runtime_reasoning_effort_override,
            )
            if runtime_policy is not None and runtime_selection is not None
            else None
        )
        if routing_policy is not None and runtime_selection_payload is not None:
            runtime_selection_payload["routing_only"] = True

        for local_attempt_ix in range(1, args.max_infra_attempts + 1):
            attempt_number = len(attempts) + 1
            attempt_started = time.monotonic()
            attempt: dict[str, Any] = {
                "attempt": attempt_number,
                "invocation_attempt": local_attempt_ix,
                "started_at": _now_iso(),
                "classification": "infrastructure",
                "solver_timeout_s": args.solver_timeout_s,
                "grader_timeout_s": config["grader_timeout_s"],
                "runtime_selection": runtime_selection_payload,
            }
            trajectory: Trajectory | None = None
            try:
                with solve_workspace(task, harness, work_root) as ws:
                    if runtime_hooks_enabled and runtime_selection is not None:
                        selection_path = write_workspace_selection(
                            runtime_state_root,
                            ws,
                            runtime_policy,
                            runtime_selection,
                            reasoning_effort_override=(
                                args.runtime_reasoning_effort_override
                            ),
                        )
                        local_selection = ws / ".rho" / "runtime-selection.json"
                        local_selection.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(selection_path, local_selection)
                        shutil.rmtree(ws / "harness" / "runtime", ignore_errors=True)
                    transport_trace_id = None
                    invocation_env, toolchain_metadata = _solver_task_environment(
                        task.row,
                        solver_env_overrides,
                        official_go_versions,
                    )
                    attempt.update(toolchain_metadata)
                    transport_file = None
                    if transport_state_dir is not None:
                        transport_trace_id = hashlib.sha256(
                            f"{run_dir}:{task.id}:{attempt_number}".encode("utf-8")
                        ).hexdigest()[:32]
                        transport_file = (
                            transport_state_dir / f"{transport_trace_id}.json"
                        )
                        invocation_env.update({
                            "RHO_TRANSPORT_TRACE_ID": transport_trace_id,
                            "RHO_TRANSPORT_TIME_FILE": str(transport_file),
                            "RHO_TRANSPORT_MAX_COMPENSATION_S": str(
                                args.transport_max_compensation_s
                            ),
                        })
                        attempt["transport_trace_id"] = transport_trace_id
                    solve_started = time.monotonic()
                    trajectory = solve_in(
                        task_agent,
                        task,
                        harness,
                        ws,
                        stage=f"swebench_pro_{args.harness_label}_heldout",
                        instructions=_solve_instructions_for_task(
                            task.id,
                            runtime_policy=harness_runtime_policy,
                            task_instructions=task_instructions,
                        ),
                        env=invocation_env or None,
                    )
                    attempt["solve_wall_time_s"] = round(
                        time.monotonic() - solve_started, 6
                    )
                    if transport_file is not None:
                        transport_state = _read_json(transport_file)
                        attempt["transport_timing"] = transport_state or {
                            "transport_wait_s": 0.0,
                            "pacer_wait_s": 0.0,
                            "retry_backoff_s": 0.0,
                            "retry_attempt_wait_s": 0.0,
                            "shared_rate_limit_wait_s": 0.0,
                        }
                        attempt["transport_timing"]["timeout_compensation_s"] = next(
                            (
                                float(event.get("seconds") or 0.0)
                                for event in trajectory.events
                                if event.get("type") == "transport_timeout_compensation"
                            ),
                            0.0,
                        )
                        trajectory = _reconcile_transport_usage(
                            trajectory,
                            transport_state,
                            transport_trace_id,
                        )
                    traj_store.put(trajectory)
                    attempt["call_metrics"] = trajectory_metrics(trajectory)
                    attempt["solver_timed_out"] = bool(trajectory.timed_out)
                    solver_dependency_failures = (
                        _solver_dependency_transport_failures(trajectory)
                    )
                    if _stalled_upstream_timeout(
                        trajectory, attempt.get("transport_timing")
                    ):
                        attempt["infra_reason"] = "stalled_upstream_transport_failure"
                    elif _looks_like_transport_failure(trajectory):
                        attempt["infra_reason"] = "api_transport_failure"
                    elif solver_dependency_failures:
                        attempt["infra_reason"] = (
                            "solver_dependency_transport_failure"
                        )
                        attempt["solver_dependency_transport_failures"] = (
                            solver_dependency_failures
                        )
                    elif not _should_grade_trajectory(trajectory):
                        attempt["infra_reason"] = "agent_process_failure"
                    else:
                        artifacts = (
                            run_dir
                            / "artifacts"
                            / _safe_task_id(task.id)
                            / f"attempt_{attempt_number}"
                            / trajectory.id
                        )
                        grade_started = time.monotonic()
                        with grade_gate:
                            grade = task.grade(trajectory, artifacts_dir=artifacts)
                        attempt["grade_wall_time_s"] = round(
                            time.monotonic() - grade_started, 6
                        )
                        attempt["grade"] = {
                            "passed": grade.passed,
                            "score": grade.score,
                            "details": grade.details,
                        }
                        classification, reason = classify_grade_result(
                            grade.details, artifacts
                        )
                        if (
                            classification == "model_result"
                            and reason == "no_prediction_patch"
                            and args.agent_backend == "claude-code"
                            and _looks_like_claude_code_premature_completion(
                                trajectory
                            )
                        ):
                            classification = "infrastructure"
                            reason = "agent_protocol_premature_completion"
                        attempt["classification"] = classification
                        if classification == "infrastructure":
                            attempt["infra_reason"] = reason
                        else:
                            attempt["model_reason"] = reason
                            final_result = {
                                "status": "completed",
                                "task_id": task.id,
                                "harness_id": harness.id,
                                "harness_label": args.harness_label,
                                "passed": grade.passed,
                                "score": grade.score,
                                "grade_details": grade.details,
                                "trajectory_id": trajectory.id,
                                "solver_timed_out": bool(trajectory.timed_out),
                                "final_call_metrics": attempt["call_metrics"],
                                "runtime_selection": runtime_selection_payload,
                            }
            except Exception as exc:  # noqa: BLE001 - retain infrastructure evidence
                attempt["infra_reason"] = "exception"
                attempt["exception"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }

            attempt["wall_time_s"] = round(time.monotonic() - attempt_started, 6)
            attempt["completed_at"] = _now_iso()
            attempts.append(attempt)
            _write_json(
                task_dir / "progress.json",
                {
                    "task_id": task.id,
                    "harness_id": harness.id,
                    "attempts": attempts,
                    "updated_at": _now_iso(),
                },
            )
            if final_result is not None:
                break
            if local_attempt_ix < args.max_infra_attempts:
                print(
                    f"[retry] {task.id} reason={attempt.get('infra_reason')} "
                    f"next_attempt={attempt_number + 1}",
                    flush=True,
                )
                _retry_delay(local_attempt_ix)

        if final_result is None:
            final_result = {
                "status": "infrastructure_failed",
                "task_id": task.id,
                "harness_id": harness.id,
                "harness_label": args.harness_label,
                "passed": None,
                "score": None,
                "runtime_selection": runtime_selection_payload,
            }
        final_result["task_total_wall_time_s"] = round(
            time.monotonic() - task_started, 6
        )
        final_result["attempt_count"] = len(attempts)
        final_result["attempts"] = attempts
        final_result["completed_at"] = _now_iso()
        _write_json(result_path, final_result)

        with progress_lock:
            counters["completed"] += 1
            counters["active"] -= 1
            status = (
                "PASS"
                if final_result.get("passed") is True
                else "FAIL"
                if final_result.get("passed") is False
                else "INFRA"
            )
            print(
                f"[task {counters['completed']}/{len(selected_tasks)}] {status} {task.id} "
                f"attempts={len(attempts)} wall={final_result['task_total_wall_time_s']}s "
                f"active={counters['active']}",
                flush=True,
            )
            partial = _summary_counts(
                selected_ids,
                run_dir,
                harness_id=harness.id,
                harness_label=args.harness_label,
                started_at=config["start_timestamp"],
                run_wall_time_s=time.monotonic() - run_started,
                final=False,
            )
            _write_json(run_dir / "reports" / "summary.partial.json", partial)
        return final_result

    with ThreadPoolExecutor(
        max_workers=min(args.concurrency, len(selected_tasks))
    ) as pool:
        futures = [pool.submit(evaluate_task, task) for task in selected_tasks]
        for future in as_completed(futures):
            future.result()

    summary = _summary_counts(
        selected_ids,
        run_dir,
        harness_id=harness.id,
        harness_label=args.harness_label,
        started_at=config["start_timestamp"],
        run_wall_time_s=time.monotonic() - run_started,
        final=True,
    )
    scored = _write_final_reports(
        summary,
        run_dir=run_dir,
        traj_store=traj_store,
        qualification_target=qualification_target,
    )
    print(
        "SUMMARY "
        + json.dumps(
            {
                **scored["tasks"],
                "input_tokens": scored["scored_usage"].get("input_tokens", 0),
                "uncached_input_tokens": scored["scored_usage"].get(
                    "uncached_input_tokens", 0
                ),
                "output_tokens": scored["scored_usage"].get("output_tokens", 0),
                "estimated_cost_usd": scored["scored_cost"].get(
                    "estimated_total_usd", 0
                ),
                "tool_calls": scored["scored_tools"].get("tool_call_count", 0),
                "qualification": (scored.get("qualification") or {}).get(
                    "status"
                ),
            }
        ),
        flush=True,
    )
    return 0 if summary["infrastructure_failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
