#!/usr/bin/env python3
"""Run resumable, infrastructure-aware ProgramBench evaluations through Pier."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import tempfile
import time
import threading
from typing import Any

from programbench.utils.load_data import get_ignored_tests, load_all_instances

from rho.agent.codex import CodexAgent
from rho.agent.codex_pool import configure_global_codex_pool
from rho.datasets.programbench.trial_results import aggregate_attempts, read_json, write_json
from rho.datasets.programbench.results import (
    aggregate_formal,
    attempt_has_infrastructure_evidence,
    parse_evaluation,
    summarize_solver,
)
from rho.datasets.programbench.grading import (
    BRANCH_TIMEOUT_S, GRADING_CONTRACT, evaluator_sources, process_budget, sha256, stage_times,
    apply_reference_collections,
)
from rho.harness_runtime import HarnessRuntimePolicy, ProfileSelection, load_runtime_policy
from rho.observability import annotate_trajectory, trajectory_metrics
from rho.stores.trajectory import FilesystemTrajectoryStore


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def grader_go_proxy() -> str:
    return os.environ.get("RHO_PROGRAMBENCH_GO_PROXY", "").strip() or (
        "https://goproxy.cn|https://proxy.golang.org|direct"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task-ids", type=Path, required=True)
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--programbench-root", type=Path, required=True)
    parser.add_argument("--docker-executable", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--harness-dir", type=Path)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--max-infra-attempts", type=int, default=3)
    parser.add_argument(
        "--retry-final-infrastructure",
        action="store_true",
        help=(
            "Revisit tasks already finalized as infrastructure_failed. "
            "Ordinary resume skips them so restarts do not create an "
            "unbounded number of duplicate Solver/verifier attempts."
        ),
    )
    parser.add_argument("--max-verifier-attempts", type=int, default=3)
    parser.add_argument("--evaluator-cpus", type=int, default=8)
    parser.add_argument("--branch-workers", type=int, default=1)
    parser.add_argument("--grader-concurrency", type=int, default=2)
    parser.add_argument("--pytest-workers", type=int, default=2)
    parser.add_argument("--grading-test-manifest", type=Path,
                        default=os.environ.get("RHO_PROGRAMBENCH_GRADING_TEST_MANIFEST"))
    parser.add_argument("--evaluator-branch-timeout-s", type=int, default=BRANCH_TIMEOUT_S)
    parser.add_argument("--evaluator-branch-retries", type=int, default=2)
    parser.add_argument(
        "--evaluator-process-timeout-s",
        type=int,
        default=0,
        help=(
            "Outer timeout for one official evaluator process. Zero derives a "
            "finite budget covering branch waves, retries, setup and collection. "
            "Explicit values below that budget are rejected."
        ),
    )
    parser.add_argument("--proxy-log", type=Path)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--classify-only", action="store_true")
    parser.add_argument(
        "--classifier-public-docs-chars",
        type=int,
        default=0,
        help=(
            "Include at most this many characters from public README, manual, and "
            "documentation files in the runtime-profile classifier input. Hidden "
            "tests, task identity, and the reference executable are never included."
        ),
    )
    return parser.parse_args()


def safe_task_id(task_id: str) -> str:
    return task_id.replace("/", "__")


def selected_task_ids(args: argparse.Namespace) -> list[str]:
    all_ids = [
        line.strip()
        for line in args.task_ids.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    selected = args.task_id or all_ids
    unknown = sorted(set(selected) - set(all_ids))
    if unknown:
        raise ValueError(f"Requested tasks are not in the split: {unknown}")
    return selected[: args.max_tasks] if args.max_tasks is not None else selected


def result_path(run_dir: Path, task_id: str) -> Path:
    return run_dir / "tasks" / safe_task_id(task_id) / "result.json"


def current_results(run_dir: Path, task_ids: list[str]) -> list[dict[str, Any]]:
    return [
        value
        for task_id in task_ids
        if (value := read_json(result_path(run_dir, task_id))) is not None
    ]


def _classification_attempts(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    formal: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for path in sorted((run_dir / "classifications").glob("*/selection.json")):
        record = read_json(path) or {}
        for attempt in record.get("attempts") or []:
            target = formal if attempt.get("classification") == "model_result" else excluded
            target.append(attempt)
    return formal, excluded


def write_summary(
    args: argparse.Namespace,
    task_ids: list[str],
    *,
    started_at: str,
    started_monotonic: float,
    final: bool,
) -> dict[str, Any]:
    results = current_results(args.run_dir, task_ids)
    completed = [item for item in results if item.get("status") == "completed"]
    infra = [item for item in results if item.get("status") == "infrastructure_failed"]
    solver_formal = [item["selected_attempt"] for item in completed]
    classifier_formal, classifier_excluded = _classification_attempts(args.run_dir)
    formal_metrics = aggregate_attempts([*solver_formal, *classifier_formal])
    solver_metrics = aggregate_formal(solver_formal)
    formal_metrics["score"] = solver_metrics["score"]
    formal_metrics["solver_attempt_count"] = len(solver_formal)
    formal_metrics["classifier_attempt_count"] = len(classifier_formal)
    excluded = [
        attempt
        for item in results
        for attempt in item.get("attempts", [])
        if attempt.get("classification") == "infrastructure"
    ] + classifier_excluded
    fully_solved = sum(bool(item.get("fully_solved")) for item in completed)
    score_sum = sum(float(item.get("score") or 0.0) for item in completed)
    summary = {
        "final": final and len(completed) == len(task_ids),
        "runner_finished": final,
        "grading_complete": len(completed) == len(task_ids),
        "grading_contract": GRADING_CONTRACT,
        "benchmark": "programbench",
        "agent": "codex",
        "model": args.model,
        "reasoning_effort": "dynamic" if args.runtime_policy else args.reasoning_effort,
        "harness": str(args.harness_dir) if args.harness_dir else "empty",
        "requested_tasks": len(task_ids),
        "completed_tasks": len(completed),
        "fully_solved": fully_solved,
        "average_score_completed": score_sum / len(completed) if completed else None,
        "average_score_all_requested": (
            score_sum / len(task_ids) if task_ids and len(completed) == len(task_ids) else None
        ),
        "score_lower_bound_all_requested": score_sum / len(task_ids) if task_ids else None,
        "infrastructure_failed": len(infra),
        "pending": len(task_ids) - len(completed) - len(infra),
        "started_at": started_at,
        "updated_at": now_iso(),
        "run_wall_time_s": round(time.monotonic() - started_monotonic, 3),
        "metrics": {
            "formal": formal_metrics,
            "solver": solver_metrics,
            "classifier": aggregate_attempts(classifier_formal),
            "excluded_infrastructure": aggregate_attempts(excluded),
        },
        "results": completed,
        "infrastructure_failures": infra,
    }
    write_json(args.run_dir / "reports" / "summary.json", summary)
    return summary


def _runtime_fingerprint(policy: HarnessRuntimePolicy, descriptor: str) -> str:
    classifier = policy.classifier
    if classifier is None:
        return ""
    payload = {
        "policy": hashlib.sha256(policy.source.read_bytes()).hexdigest(),
        "model": classifier.model,
        "reasoning_effort": classifier.reasoning_effort,
        "samples": classifier.samples,
        "instructions": policy.classifier_instructions(descriptor),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _classifier_attempt(trajectory: Any, *, classification: str) -> dict[str, Any]:
    metrics = trajectory_metrics(trajectory)
    tools = metrics.get("tools") or {}
    return {
        "classification": classification,
        "classification_reason": (
            "runtime profile classifier completed"
            if classification == "model_result"
            else "runtime profile classifier failed; neutral fallback used"
        ),
        "trajectory_id": trajectory.id,
        "usage": metrics.get("usage"),
        "cost": metrics.get("cost"),
        "api": {
            "logical_api_requests": metrics.get("api_call_count"),
            "upstream_attempts": metrics.get("api_call_count"),
            "retry_count": 0,
        },
        "tools": {
            "count": tools.get("tool_call_count", 0),
            "by_type": {
                key: int(value.get("count") or 0)
                for key, value in (tools.get("by_type") or {}).items()
            },
            "observed_duration_s": tools.get("total_observed_duration_s", 0.0),
            "duration_count": tools.get("duration_count", 0),
        },
        "timing": {
            "agent_execution_s": metrics.get("api_wall_time_s", 0.0),
            "effective_agent_execution_s": metrics.get("api_wall_time_s", 0.0),
            "excluded_provider_wait_s": 0.0,
        },
    }


def _public_task_descriptor(
    instance: dict[str, Any], *, public_docs_excerpt: str = ""
) -> str:
    difficulty = str(instance.get("difficulty") or "unspecified")
    language = str(instance.get("language") or "unspecified")
    descriptor = (
        "Cleanroom command-line program reconstruction. The solver may read bundled "
        "documentation and repeatedly query a deterministic reference executable "
        "through its public interface, then must create an offline build and an "
        "independent implementation. "
        f"Public metadata: implementation language={language}; difficulty={difficulty}."
    )
    if public_docs_excerpt:
        descriptor += (
            "\n\nThe following excerpt comes only from documentation bundled for the "
            "solver. Classify from general interface and semantic complexity; ignore "
            "names and identity:\n\n" + public_docs_excerpt
        )
    return descriptor


def _public_docs_excerpt(
    args: argparse.Namespace,
    *,
    task_id: str,
    instance: dict[str, Any],
) -> str:
    """Read a bounded, auditable excerpt of solver-visible public documentation.

    ProgramBench's host metadata only exposes language and a coarse difficulty
    label, while the task container gives the Solver README/manual files.  A
    dynamic profile selector needs the same public contract to distinguish a
    small interface from a broad semantic surface.  The extraction deliberately
    excludes .git, the reference executable, tests, assets, and source-like
    files; failures simply fall back to the coarse descriptor.
    """

    limit = max(0, int(args.classifier_public_docs_chars))
    if not limit:
        return ""
    cache_path = (
        args.run_dir / "classifier_inputs" / safe_task_id(task_id) / "public_docs.txt"
    )
    cached = read_json(cache_path.with_suffix(".json"))
    if cached is not None and cached.get("limit") == limit:
        return str(cached.get("excerpt") or "")

    image = str(instance.get("image_name") or "").strip()
    if not image:
        return ""
    if ":" not in image.rsplit("/", 1)[-1]:
        image += ":task_cleanroom_v6"
    script = r'''
find /workspace -maxdepth 3 -type f \
  \( -iname 'README' -o -iname 'README.*' -o -iname 'USAGE' \
     -o -iname 'USAGE.*' -o -iname '*.md' -o -iname '*.rst' \
     -o -iname '*.[1-9]' \) \
  ! -path '/workspace/.git/*' ! -path '*/test/*' ! -path '*/tests/*' \
  -print | LC_ALL=C sort | while IFS= read -r file; do
    printf '\n--- public document: %s ---\n' "${file#/workspace/}"
    head -c 16000 "$file"
    printf '\n'
  done
'''
    try:
        completed = subprocess.run(
            [
                str(args.docker_executable),
                "run",
                "--rm",
                "--network",
                "none",
                "--entrypoint",
                "/bin/sh",
                image,
                "-lc",
                script,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if completed.returncode != 0:
        return ""
    excerpt = completed.stdout.decode("utf-8", errors="replace")[:limit].strip()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        cache_path.with_suffix(".json"),
        {
            "task_id": task_id,
            "image": image,
            "limit": limit,
            "excerpt": excerpt,
            "created_at": now_iso(),
        },
    )
    return excerpt


def _cached_selection(
    policy: HarnessRuntimePolicy, record: dict[str, Any]
) -> ProfileSelection:
    if record.get("classification_source") != "model":
        return policy.fallback_selection(str(record.get("rationale") or "cached fallback"))
    return policy.select_model_profile(
        str(record["profile"]),
        confidence=float(record["confidence"]),
        rationale=str(record.get("rationale") or ""),
        classifier_model=str(record["classifier_model"]),
        classifier_trajectory_id=str(record["classifier_trajectory_id"]),
        vote_counts=(record.get("consensus") or {}).get("vote_counts"),
    )


def classify_runtime_profile(
    args: argparse.Namespace,
    *,
    task_id: str,
    instance: dict[str, Any],
) -> ProfileSelection:
    policy: HarnessRuntimePolicy = args.runtime_policy
    classifier = policy.classifier
    if classifier is None:
        return policy.select(
            _public_task_descriptor(
                instance,
                public_docs_excerpt=_public_docs_excerpt(
                    args, task_id=task_id, instance=instance
                ),
            )
        )
    descriptor = _public_task_descriptor(
        instance,
        public_docs_excerpt=_public_docs_excerpt(
            args, task_id=task_id, instance=instance
        ),
    )
    fingerprint = _runtime_fingerprint(policy, descriptor)
    cache_path = args.run_dir / "classifications" / safe_task_id(task_id) / "selection.json"
    cached = read_json(cache_path)
    if cached is not None and cached.get("fingerprint") == fingerprint:
        try:
            return _cached_selection(policy, cached)
        except (KeyError, TypeError, ValueError):
            pass

    attempts: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    instructions = policy.classifier_instructions(descriptor)
    max_total_attempts = classifier.samples * classifier.max_attempts
    for attempt_number in range(1, max_total_attempts + 1):
        workspace = Path(tempfile.mkdtemp(dir=args.classifier_work_root, prefix="classify_"))
        try:
            trace_id = hashlib.sha256(
                f"programbench\0classifier\0{task_id}\0{attempt_number}".encode()
            ).hexdigest()[:32]
            trajectory = args.classifier_agent.run(
                workspace,
                instructions,
                output_schema=policy.classifier_output_schema(),
                task_id=task_id,
                harness_id=str(args.harness_dir),
                kind="evaluate",
                timeout_s=classifier.timeout_s,
                env={
                    "OPENAI_API_KEY": "programbench-local-proxy",
                    "RHO_TRANSPORT_TRACE_ID": trace_id,
                },
            )
            trajectory = annotate_trajectory(
                trajectory,
                agent=args.classifier_agent,
                stage="programbench_profile_classification",
            )
            args.trajectory_store.put(trajectory)
            attempt = {
                "attempt": attempt_number,
                **_classifier_attempt(
                    trajectory,
                    classification=(
                        "model_result"
                        if not trajectory.timed_out and trajectory.exit_code == 0
                        else "infrastructure"
                    ),
                ),
            }
            attempts.append(attempt)
            if trajectory.timed_out or trajectory.exit_code != 0:
                continue
            try:
                output = json.loads(trajectory.final_message)
                profile = str(output["profile"])
                confidence = float(output["confidence"])
                rationale = str(output["rationale"])
                if profile not in policy.profiles or not 0 <= confidence <= 1:
                    raise ValueError("classifier output is outside the policy schema")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                attempt["classification"] = "infrastructure"
                attempt["classification_reason"] = f"invalid classifier output: {exc}"
                continue
            attempt["output"] = {
                "profile": profile,
                "confidence": confidence,
                "rationale": rationale,
            }
            valid.append({
                "profile": profile,
                "confidence": confidence,
                "rationale": rationale,
                "trajectory_id": trajectory.id,
            })
            if len(valid) >= classifier.samples:
                break
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    if valid:
        counts = Counter(item["profile"] for item in valid)
        top = max(counts.values())
        leaders = sorted(name for name, count in counts.items() if count == top)
        profile = leaders[0] if len(leaders) == 1 else policy.default_profile
        winners = [item for item in valid if item["profile"] == profile] or valid
        representative = max(winners, key=lambda item: float(item["confidence"]))
        confidence = sum(float(item["confidence"]) for item in winners) / len(winners)
        rationale = str(representative["rationale"])
        consensus = {"vote_counts": dict(sorted(counts.items()))}
        selection = policy.select_model_profile(
            profile,
            confidence=confidence,
            rationale=rationale,
            classifier_model=classifier.model,
            classifier_trajectory_id=str(representative["trajectory_id"]),
            vote_counts=consensus["vote_counts"],
        )
        record = {
            "fingerprint": fingerprint,
            "classification_source": "model",
            "profile": profile,
            "effective_profile": selection.profile.name,
            "confidence": selection.confidence,
            "rationale": selection.rationale,
            "classifier_model": classifier.model,
            "classifier_reasoning_effort": classifier.reasoning_effort,
            "classifier_trajectory_id": selection.classifier_trajectory_id,
            "consensus": consensus,
            "attempts": attempts,
            "completed_at": now_iso(),
        }
        write_json(cache_path, record)
        return selection

    reason = "model classifier failed after bounded retries; used neutral default"
    selection = policy.fallback_selection(reason)
    write_json(
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
            "completed_at": now_iso(),
        },
    )
    return selection


def render_container_runtime_config(
    policy: HarnessRuntimePolicy, output: Path
) -> Path:
    hook_command = (
        "RHO_RUNTIME_SELECTION_PATH=/rho-runtime/selection.json "
        "RHO_RUNTIME_STATE_ROOT=/workspace/.rho_runtime "
        "/usr/bin/python3 /harness/runtime/hook.py"
    )
    lines = [f"tool_output_token_limit = {policy.tool_output_token_limit}"]
    if policy.model_instructions_file is not None:
        lines.append('model_instructions_file = "/harness/runtime/model_instructions.md"')
    if policy.model_verbosity is not None:
        lines.append(f"model_verbosity = {json.dumps(policy.model_verbosity)}")
    if policy.disabled_features:
        lines.append("\n[features]")
        lines.extend(f"{name} = false" for name in policy.disabled_features)
    events = [("SessionStart", 1200), ("PreToolUse", 1200)]
    raw_policy = json.loads(policy.source.read_text(encoding="utf-8"))
    post_tool_hook = raw_policy.get("codex", {}).get("post_tool_use_hook", False)
    if not isinstance(post_tool_hook, bool):
        raise ValueError("codex.post_tool_use_hook must be a boolean")
    if post_tool_hook:
        events.append(("PostToolUse", 1200))
    for event, limit in events:
        lines.extend(
            [
                f"\n[[hooks.{event}]]",
                f' matcher = "^startup$"' if event == "SessionStart" else "",
                f"\n[[hooks.{event}.hooks]]",
                'type = "command"',
                f"command = {json.dumps(hook_command)}",
                "timeout = 10",
                f"additionalContextLimit = {limit}",
            ]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(line for line in lines if line) + "\n", encoding="utf-8")
    return output


def evaluation_for(
    args: argparse.Namespace,
    task_id: str,
    submission: Path,
    attempt_dir: Path,
    ignored_tests: set[str],
    *,
    attempt_limit: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # A separate gate bounds grader containers independently of Solver slots.
    with getattr(args, "grader_gate", nullcontext()):
        return _evaluation_for(args, task_id, submission, attempt_dir, ignored_tests,
                               attempt_limit=attempt_limit)


def _evaluation_for(
    args: argparse.Namespace,
    task_id: str,
    submission: Path,
    attempt_dir: Path,
    ignored_tests: set[str],
    *,
    attempt_limit: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected = args.grading_expected[task_id]
    minimum_budget = process_budget(len(expected), args.branch_workers,
                                    args.evaluator_branch_retries, args.evaluator_branch_timeout_s,
                                    len(args.grading_toolchains.get(task_id, [])))
    process_timeout = args.evaluator_process_timeout_s or minimum_budget
    if process_timeout < minimum_budget:
        raise ValueError(f"grader watchdog {process_timeout}s is below required {minimum_budget}s")
    submission_hash = sha256(submission)
    source_root = attempt_dir / "submission"
    target_dir = source_root / task_id
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(submission, target_dir / "submission.tar.gz")
    eval_path = target_dir / f"{task_id}.eval.json"
    verifier_attempts: list[dict[str, Any]] = []
    limit = args.max_verifier_attempts if attempt_limit is None else attempt_limit
    for number in range(1, limit + 1):
        # Never overwrite retry outputs, or allow a stale eval.json from an
        # earlier attempt to make an interrupted launch look successful.
        source_root = attempt_dir / f"grading-attempt-{number:02d}" / "submission"
        target_dir = source_root / task_id
        if target_dir.exists():
            raise ValueError(f"refusing to overwrite a verifier attempt: {target_dir}")
        target_dir.mkdir(parents=True)
        shutil.copy2(submission, target_dir / "submission.tar.gz")
        if sha256(target_dir / "submission.tar.gz") != submission_hash:
            raise ValueError("submission changed while preparing grading")
        eval_path = target_dir / f"{task_id}.eval.json"
        log_path = attempt_dir / f"verifier-attempt-{number:02d}.log"
        executable = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "programbench"
        command = [
            str(executable),
            "eval",
            str(source_root),
            "--workers",
            "1",
            "--branch-workers",
            str(args.branch_workers),
            "--docker-cpus",
            str(args.evaluator_cpus),
            "--branch-retries",
            str(args.evaluator_branch_retries),
            "--filter",
            "^" + re.escape(task_id) + "$",
        ]
        env = os.environ.copy()
        hf_home = Path(os.environ.get("HF_HOME") or Path(__file__).resolve().parents[1] / "cache" / "huggingface")
        env.update(
            {
                "PROGRAMBENCH_DOCKER_EXECUTABLE": str(args.docker_executable),
                "PROGRAMBENCH_DOCKER_RUN_TIMEOUT": "900",
                "PROGRAMBENCH_DOCKER_CP_TIMEOUT": "900",
                "PROGRAMBENCH_BRANCH_TIMEOUT": str(args.evaluator_branch_timeout_s),
                "PROGRAMBENCH_PYTEST_WORKERS": str(args.pytest_workers),
                "PROGRAMBENCH_HF_REVISION": args.grading_revision,
                "HF_HUB_OFFLINE": "1",
                "PYTHONPATH": str(args.grader_package.parent),
                "HF_HOME": str(hf_home.expanduser().resolve()),
            }
        )
        evaluator_proxy = os.environ.get("RHO_PROGRAMBENCH_EVALUATOR_PROXY_URL", "").strip()
        if not evaluator_proxy:
            host_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or ""
            evaluator_proxy = re.sub(
                r"(?<=://)(?:127\.0\.0\.1|localhost)(?=[:/])",
                "host.docker.internal",
                host_proxy,
            )
        if evaluator_proxy:
            env["PROGRAMBENCH_EVALUATOR_PROXY_URL"] = evaluator_proxy
        go_proxy = grader_go_proxy()
        if go_proxy:
            env["PROGRAMBENCH_GO_PROXY"] = go_proxy
        started = time.monotonic()
        timed_out = False
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=args.programbench_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                process.wait(
                    timeout=process_timeout
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                log.write(
                    "\n[runner] official evaluator process exceeded outer timeout "
                    f"of {process_timeout}s; terminating process group\n"
                )
                log.flush()
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
        parsed = parse_evaluation(eval_path, ignored_tests=ignored_tests,
                                  expected_tests_by_branch=expected, require_xml=True,
                                  prerequisite_metadata=args.grading_prerequisites[task_id])
        if sha256(submission) != submission_hash or sha256(target_dir / "submission.tar.gz") != submission_hash:
            parsed = {"classification": "infrastructure", "reason": "immutable submission hash changed"}
        if timed_out and parsed["classification"] != "model_result":
            parsed["reason"] = (
                "official evaluator process exceeded its outer timeout before "
                "producing a complete evaluation"
            )
        record = {
            "attempt": number,
            "exit_code": process.returncode,
            "wall_time_s": round(time.monotonic() - started, 3),
            "log_path": str(log_path),
            "process_timed_out": timed_out,
            "process_budget_s": process_timeout,
            "task_id": task_id,
            "submission_sha256": submission_hash,
            "grading_contract": GRADING_CONTRACT,
            "evaluator_manifest": str(args.grader_manifest),
            "evaluation_path": str(eval_path),
            "stage_times": stage_times(parsed.get("raw") or {}),
            "classification": parsed["classification"],
            "reason": parsed["reason"],
        }
        verifier_attempts.append(record)
        write_json(target_dir.parent.parent / "grading_record.json", record)
        if parsed["classification"] == "model_result":
            # Backward-compatible pointer for resume tools; original attempt
            # JSON/XML remain in their immutable per-attempt directories.
            write_json(attempt_dir / "submission" / task_id / f"{task_id}.eval.json", parsed["raw"])
            return parsed, verifier_attempts
    return parsed, verifier_attempts


def persisted_evaluation_path(attempt: dict[str, Any], task_id: str) -> Path | None:
    """Locate the official eval JSON associated with a persisted attempt."""

    for record in reversed(attempt.get("verifier_attempts") or []):
        if record.get("evaluation_path") and Path(record["evaluation_path"]).is_file():
            return Path(record["evaluation_path"])
        log_path = record.get("log_path")
        if log_path:
            candidate = (
                Path(log_path).parent
                / "submission"
                / task_id
                / f"{task_id}.eval.json"
            )
            if candidate.is_file():
                return candidate
    job_dir = attempt.get("job_dir")
    if job_dir:
        candidate = (
            Path(job_dir) / "submission" / task_id / f"{task_id}.eval.json"
        )
        if candidate.is_file():
            return candidate
    return None


def verifier_attempt_count(attempt: dict[str, Any]) -> int:
    """Count verifier launches for one immutable Solver submission.

    Recovery logs are intentionally included: a foreground runner may be
    interrupted before it can append the recovery record to result.json. A
    disk-based count keeps the verifier retry budget bounded across resumes.
    """

    observed: set[str] = set()
    anonymous = 0
    for record in attempt.get("verifier_attempts") or []:
        log_path = record.get("log_path")
        if log_path:
            observed.add(str(Path(log_path).resolve()))
        else:
            anonymous += 1
    job_dir = attempt.get("job_dir")
    if job_dir:
        root = Path(job_dir)
        for path in root.glob("verifier-attempt-*.log"):
            observed.add(str(path.resolve()))
        for path in root.glob("verifier-recovery-*/verifier-attempt-*.log"):
            observed.add(str(path.resolve()))
    return len(observed) + anonymous


def run_task(
    args: argparse.Namespace,
    task_id: str,
    instance: dict[str, Any],
    ignored_tests: set[str],
) -> dict[str, Any]:
    output = result_path(args.run_dir, task_id)
    existing = read_json(output)
    if (
        existing is not None
        and existing.get("status") == "infrastructure_failed"
        and not args.retry_final_infrastructure
    ):
        return existing
    if existing is not None and existing.get("status") == "completed":
        selected = existing.get("selected_attempt") or {}
        if not attempt_has_infrastructure_evidence(selected):
            eval_path = persisted_evaluation_path(selected, task_id)
            if eval_path is not None:
                rescored = parse_evaluation(eval_path, ignored_tests=ignored_tests,
                    expected_tests_by_branch=args.grading_expected[task_id], require_xml=True)
                if rescored.get("classification") == "model_result":
                    score_fields = {
                        "score": rescored["score"],
                        "resolved_tests": rescored["resolved_tests"],
                        "total_tests": rescored["total_tests"],
                        "fully_solved": rescored["fully_solved"],
                    }
                    selected.update(score_fields)
                    selected["evaluation"] = {
                        key: value for key, value in rescored.items() if key != "raw"
                    }
                    for attempt in existing.get("attempts") or []:
                        if (
                            attempt.get("attempt") == selected.get("attempt")
                            and attempt.get("trace_id") == selected.get("trace_id")
                        ):
                            attempt.update(selected)
                    existing.update(score_fields)
                    existing["selected_attempt"] = selected
                    write_json(output, existing)
                    return existing
            # Do not preserve a completed label when a fresh integrity check
            # rejects its evidence. No automatic Solver rerun is authorized.
            for attempt in existing.get("attempts") or []:
                if attempt.get("trace_id") == selected.get("trace_id"):
                    attempt["classification"] = "infrastructure"
                    attempt["infrastructure_layer"] = "grader"
                    attempt["classification_reason"] = "grading completeness check failed on resume"
            existing["status"] = "infrastructure_failed"
            existing["recovery_required"] = "same_submission_verifier_only"
            existing.pop("selected_attempt", None)
            for key in ("score", "resolved_tests", "total_tests", "fully_solved"):
                existing.pop(key, None)
            write_json(output, existing)
            return existing

        # A pre-fix run may have graded a partial workspace after an agent
        # process was killed, turning an infrastructure failure into a model
        # zero. Preserve that attempt (and its usage/cost) for audit, exclude
        # it from formal metrics, and resume from the next attempt number.
        selected_number = selected.get("attempt")
        selected_trace = selected.get("trace_id")
        attempts = list(existing.get("attempts") or [])
        for attempt in attempts:
            if (
                attempt.get("attempt") == selected_number
                and attempt.get("trace_id") == selected_trace
            ):
                attempt["original_classification"] = attempt.get("classification")
                attempt["original_classification_reason"] = attempt.get(
                    "classification_reason"
                )
                attempt["classification"] = "infrastructure"
                attempt["classification_reason"] = (
                    "reclassified on resume: persisted agent failure contains "
                    "decisive infrastructure evidence"
                )
        existing = {
            "task_id": task_id,
            "status": "retrying_infrastructure",
            "attempts": attempts,
            "updated_at": now_iso(),
        }
        write_json(output, existing)
    attempts = list((existing or {}).get("attempts") or [])
    task_root = args.tasks_root / task_id
    task_run_dir = output.parent
    task_run_dir.mkdir(parents=True, exist_ok=True)
    runtime_selection: ProfileSelection | None = None
    runtime_selection_path: Path | None = None
    if args.runtime_policy is not None:
        runtime_selection = classify_runtime_profile(
            args,
            task_id=task_id,
            instance=instance,
        )
        runtime_selection_path = (
            args.run_dir / "runtime_state" / safe_task_id(task_id) / "selection.json"
        )
        write_json(
            runtime_selection_path,
            args.runtime_policy.workspace_selection_payload(runtime_selection),
        )

    # A verifier-only infrastructure failure may already have been persisted
    # before the process stopped. Revisit those attempts and, when their
    # underlying Solver artifact is complete, regrade it without another model
    # call. This is deliberately based on fresh Solver evidence rather than on
    # the old broad "infrastructure" label.
    verifier_only_exhausted = False
    for position, previous in enumerate(attempts):
        if (
            previous.get("classification") != "infrastructure"
            or not previous.get("job_dir")
            or not previous.get("submission_artifact")
        ):
            continue
        refreshed = summarize_solver(
            task_id=task_id,
            attempt=int(previous["attempt"]),
            trace_id=str(previous["trace_id"]),
            job_dir=Path(previous["job_dir"]),
            pier_exit_code=int(previous.get("pier_exit_code") or 0),
            proxy_log=args.proxy_log,
            model=args.model,
        )
        if refreshed.get("classification") != "solver_ready":
            continue
        persisted_eval = persisted_evaluation_path(previous, task_id)
        if persisted_eval is not None:
            rescored = parse_evaluation(persisted_eval, ignored_tests=ignored_tests,
                expected_tests_by_branch=args.grading_expected[task_id], require_xml=True)
            if rescored.get("classification") == "model_result":
                refreshed["verifier_attempts"] = list(
                    previous.get("verifier_attempts") or []
                )
                refreshed["evaluation"] = {
                    key: value for key, value in rescored.items() if key != "raw"
                }
                refreshed.update(
                    {
                        "classification": "model_result",
                        "classification_reason": rescored["reason"],
                        "score": rescored["score"],
                        "resolved_tests": rescored["resolved_tests"],
                        "total_tests": rescored["total_tests"],
                        "fully_solved": rescored["fully_solved"],
                    }
                )
                attempts[position] = refreshed
                result = {
                    "task_id": task_id,
                    "status": "completed",
                    "score": refreshed["score"],
                    "resolved_tests": refreshed["resolved_tests"],
                    "total_tests": refreshed["total_tests"],
                    "fully_solved": refreshed["fully_solved"],
                    "selected_attempt": refreshed,
                    "attempts": attempts,
                    "completed_at": now_iso(),
                }
                write_json(output, result)
                return result
        used_verifier_attempts = verifier_attempt_count(previous)
        if (
            used_verifier_attempts >= args.max_verifier_attempts
            and not args.retry_final_infrastructure
        ):
            refreshed["verifier_attempts"] = list(
                previous.get("verifier_attempts") or []
            )
            refreshed["classification"] = "infrastructure"
            refreshed["classification_reason"] = (
                "official evaluator exhausted its lifetime retry budget for "
                "this preserved Solver submission"
            )
            attempts[position] = refreshed
            verifier_only_exhausted = True
            continue
        recovery_index = 1
        recovery_root = Path(previous["job_dir"])
        while (recovery_root / f"verifier-recovery-{recovery_index:02d}").exists():
            recovery_index += 1
        recovery_dir = recovery_root / f"verifier-recovery-{recovery_index:02d}"
        recovery_dir.mkdir(parents=True)
        verifier_limit = (
            args.max_verifier_attempts
            if args.retry_final_infrastructure
            else args.max_verifier_attempts - used_verifier_attempts
        )
        recovered_evaluation, recovered_verifier_attempts = evaluation_for(
            args,
            task_id,
            Path(refreshed["submission_artifact"]),
            recovery_dir,
            ignored_tests,
            attempt_limit=verifier_limit,
        )
        refreshed["verifier_attempts"] = [
            *(previous.get("verifier_attempts") or []),
            *recovered_verifier_attempts,
        ]
        if recovered_evaluation.get("classification") != "model_result":
            refreshed["classification"] = "infrastructure"
            refreshed["classification_reason"] = recovered_evaluation["reason"]
            attempts[position] = refreshed
            verifier_only_exhausted = True
            continue
        refreshed["evaluation"] = {
            key: value
            for key, value in recovered_evaluation.items()
            if key != "raw"
        }
        refreshed.update(
            {
                "classification": "model_result",
                "classification_reason": recovered_evaluation["reason"],
                "score": recovered_evaluation["score"],
                "resolved_tests": recovered_evaluation["resolved_tests"],
                "total_tests": recovered_evaluation["total_tests"],
                "fully_solved": recovered_evaluation["fully_solved"],
            }
        )
        attempts[position] = refreshed

        known_attempts = {int(item["attempt"]) for item in attempts}
        for orphan_dir in sorted((task_run_dir / "pier").glob("attempt-*")):
            try:
                orphan_number = int(orphan_dir.name.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                continue
            if orphan_number in known_attempts:
                continue
            orphan_trace_id = hashlib.sha256(
                f"programbench\0codex\0{task_id}\0{orphan_number}".encode()
            ).hexdigest()[:32]
            later_orphan = summarize_solver(
                task_id=task_id,
                attempt=orphan_number,
                trace_id=orphan_trace_id,
                job_dir=orphan_dir,
                pier_exit_code=130,
                proxy_log=args.proxy_log,
                model=args.model,
            )
            later_orphan["classification"] = "infrastructure"
            later_orphan["classification_reason"] = (
                "redundant later Solver attempt interrupted while recovering "
                "an earlier valid submission after verifier infrastructure failure"
            )
            attempts.append(later_orphan)
        attempts.sort(key=lambda item: int(item["attempt"]))
        result = {
            "task_id": task_id,
            "status": "completed",
            "score": refreshed["score"],
            "resolved_tests": refreshed["resolved_tests"],
            "total_tests": refreshed["total_tests"],
            "fully_solved": refreshed["fully_solved"],
            "selected_attempt": refreshed,
            "attempts": attempts,
            "completed_at": now_iso(),
        }
        write_json(output, result)
        return result

    if verifier_only_exhausted:
        result = {
            "task_id": task_id,
            "status": "infrastructure_failed",
            "attempts": attempts,
            "updated_at": now_iso(),
        }
        write_json(output, result)
        return result

    start_index = len(attempts) + 1
    # A foreground run can be interrupted after Pier has produced a complete
    # Solver trajectory but before the official verifier result is committed
    # to result.json. Preserve such an orphan attempt for cost/Token auditing
    # and retry it as infrastructure instead of silently overwriting its job
    # directory and losing the incurred usage.
    while (task_run_dir / "pier" / f"attempt-{start_index:02d}").exists():
        trace_id = hashlib.sha256(
            f"programbench\0codex\0{task_id}\0{start_index}".encode()
        ).hexdigest()[:32]
        orphan_dir = task_run_dir / "pier" / f"attempt-{start_index:02d}"
        orphan = summarize_solver(
            task_id=task_id,
            attempt=start_index,
            trace_id=trace_id,
            job_dir=orphan_dir,
            pier_exit_code=130,
            proxy_log=args.proxy_log,
            model=args.model,
        )
        recovered_eval_path = (
            orphan_dir
            / "submission"
            / task_id
            / f"{task_id}.eval.json"
        )
        recovered_evaluation = parse_evaluation(
            recovered_eval_path,
            ignored_tests=ignored_tests,
            expected_tests_by_branch=args.grading_expected[task_id], require_xml=True,
        )
        recovered_verifier_attempts: list[dict[str, Any]] | None = None
        orphan_verifier_exhausted = False
        if (
            orphan.get("classification") == "solver_ready"
            and recovered_evaluation.get("classification") != "model_result"
            and orphan.get("submission_artifact")
        ):
            # The Solver may have finished successfully before the foreground
            # runner was interrupted in an infrastructure-failed verifier.
            # Regrade the existing immutable submission instead of paying for
            # another Solver call. Use a separate directory so earlier
            # verifier logs remain available for audit.
            recovery_index = 1
            while (
                orphan_dir / f"verifier-recovery-{recovery_index:02d}"
            ).exists():
                recovery_index += 1
            used_verifier_attempts = verifier_attempt_count(orphan)
            if (
                used_verifier_attempts >= args.max_verifier_attempts
                and not args.retry_final_infrastructure
            ):
                orphan_verifier_exhausted = True
            else:
                recovery_dir = (
                    orphan_dir / f"verifier-recovery-{recovery_index:02d}"
                )
                recovery_dir.mkdir(parents=True)
                verifier_limit = (
                    args.max_verifier_attempts
                    if args.retry_final_infrastructure
                    else args.max_verifier_attempts - used_verifier_attempts
                )
                recovered_evaluation, recovered_verifier_attempts = evaluation_for(
                    args,
                    task_id,
                    Path(orphan["submission_artifact"]),
                    recovery_dir,
                    ignored_tests,
                    attempt_limit=verifier_limit,
                )
        if (
            orphan.get("classification") == "solver_ready"
            and recovered_evaluation.get("classification") == "model_result"
        ):
            orphan["verifier_attempts"] = recovered_verifier_attempts or [
                    {
                        "attempt": "recovered",
                        "exit_code": 0,
                        "wall_time_s": None,
                        "log_path": None,
                        "classification": "model_result",
                        "reason": "recovered complete official ProgramBench evaluation",
                    }
                ]
            orphan["evaluation"] = {
                key: value
                for key, value in recovered_evaluation.items()
                if key != "raw"
            }
            orphan.update(
                {
                    "pier_exit_code": 0,
                    "classification": "model_result",
                    "classification_reason": recovered_evaluation["reason"],
                    "score": recovered_evaluation["score"],
                    "resolved_tests": recovered_evaluation["resolved_tests"],
                    "total_tests": recovered_evaluation["total_tests"],
                    "fully_solved": recovered_evaluation["fully_solved"],
                }
            )
            attempts.append(orphan)
            later_index = start_index + 1
            while (task_run_dir / "pier" / f"attempt-{later_index:02d}").exists():
                later_trace_id = hashlib.sha256(
                    f"programbench\0codex\0{task_id}\0{later_index}".encode()
                ).hexdigest()[:32]
                later_orphan = summarize_solver(
                    task_id=task_id,
                    attempt=later_index,
                    trace_id=later_trace_id,
                    job_dir=task_run_dir / "pier" / f"attempt-{later_index:02d}",
                    pier_exit_code=130,
                    proxy_log=args.proxy_log,
                    model=args.model,
                )
                later_orphan["classification"] = "infrastructure"
                later_orphan["classification_reason"] = (
                    "redundant later Solver attempt interrupted while recovering "
                    "an earlier valid submission after verifier infrastructure failure"
                )
                attempts.append(later_orphan)
                later_index += 1
            result = {
                "task_id": task_id,
                "status": "completed",
                "score": orphan["score"],
                "resolved_tests": orphan["resolved_tests"],
                "total_tests": orphan["total_tests"],
                "fully_solved": orphan["fully_solved"],
                "selected_attempt": orphan,
                "attempts": attempts,
                "completed_at": now_iso(),
            }
            write_json(output, result)
            return result
        orphan["classification"] = "infrastructure"
        orphan["classification_reason"] = (
            "previous foreground run was interrupted before a complete formal verifier result"
        )
        attempts.append(orphan)
        if orphan_verifier_exhausted:
            orphan["classification_reason"] = (
                "official evaluator exhausted its lifetime retry budget for "
                "this preserved Solver submission"
            )
            result = {
                "task_id": task_id,
                "status": "infrastructure_failed",
                "attempts": attempts,
                "updated_at": now_iso(),
            }
            write_json(output, result)
            return result
        start_index += 1
        write_json(
            output,
            {
                "task_id": task_id,
                "status": "retrying_infrastructure",
                "attempts": attempts,
                "updated_at": now_iso(),
            },
        )

    # ``max_infra_attempts`` is a lifetime cap for this result directory, not
    # a fresh allowance on every resume.  Otherwise a harmless runner restart
    # can turn three failed attempts into six, nine, ... and repeatedly spend
    # hours on deterministic verifier failures.
    for attempt_number in range(start_index, args.max_infra_attempts + 1):
        trace_id = hashlib.sha256(
            f"programbench\0codex\0{task_id}\0{attempt_number}".encode()
        ).hexdigest()[:32]
        job_dir = task_run_dir / "pier" / f"attempt-{attempt_number:02d}"
        job_dir.mkdir(parents=True, exist_ok=True)
        log_path = job_dir / "runner.log"
        env = os.environ.copy()
        env.update(
            {
                "RHO_PROGRAMBENCH_TRACE_ID": trace_id,
                "RHO_PROGRAMBENCH_MODEL": args.model,
                "RHO_PROGRAMBENCH_REASONING_EFFORT": (
                    runtime_selection.profile.reasoning_effort
                    if runtime_selection is not None
                    else args.reasoning_effort
                ),
            }
        )
        if args.harness_dir is not None:
            env["RHO_PROGRAMBENCH_HARNESS_DIR"] = str(args.harness_dir.resolve())
        if runtime_selection_path is not None:
            env["RHO_PROGRAMBENCH_RUNTIME_SELECTION"] = str(
                runtime_selection_path.resolve()
            )
            env["RHO_PROGRAMBENCH_RUNTIME_CONFIG"] = str(
                args.runtime_config_path.resolve()
            )
        command = [
            "bash",
            str(Path(__file__).resolve().parent / "run_programbench_codex_task.sh"),
            f"programbench-{safe_task_id(task_id)}-attempt-{attempt_number:02d}",
            str(job_dir),
            str(task_root),
        ]
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        attempt = summarize_solver(
            task_id=task_id,
            attempt=attempt_number,
            trace_id=trace_id,
            job_dir=job_dir,
            pier_exit_code=completed.returncode,
            proxy_log=args.proxy_log,
            model=args.model,
        )
        if runtime_selection is not None:
            attempt["runtime_selection"] = args.runtime_policy.workspace_selection_payload(
                runtime_selection
            )
        if attempt["classification"] == "solver_ready":
            evaluation, verifier_attempts = evaluation_for(
                args,
                task_id,
                Path(attempt["submission_artifact"]),
                job_dir,
                ignored_tests,
            )
            attempt["verifier_attempts"] = verifier_attempts
            attempt["evaluation"] = {
                key: value for key, value in evaluation.items() if key != "raw"
            }
            if evaluation["classification"] == "model_result":
                attempt.update(
                    {
                        "classification": "model_result",
                        "classification_reason": evaluation["reason"],
                        "score": evaluation["score"],
                        "resolved_tests": evaluation["resolved_tests"],
                        "total_tests": evaluation["total_tests"],
                        "fully_solved": evaluation["fully_solved"],
                    }
                )
            else:
                attempt["classification"] = "infrastructure"
                attempt["infrastructure_layer"] = "grader"
                attempt["classification_reason"] = evaluation["reason"]

        attempts.append(attempt)
        if attempt["classification"] == "model_result":
            result = {
                "task_id": task_id,
                "status": "completed",
                "score": attempt["score"],
                "resolved_tests": attempt["resolved_tests"],
                "total_tests": attempt["total_tests"],
                "fully_solved": attempt["fully_solved"],
                "selected_attempt": attempt,
                "attempts": attempts,
                "completed_at": now_iso(),
            }
            write_json(output, result)
            return result

        if attempt.get("evaluation", {}).get("classification") == "infrastructure":
            # A finished Solver with an incomplete grader is NOT a request
            # for another answer. Preserve and recover this submission only.
            result = {"task_id": task_id, "status": "infrastructure_failed",
                      "attempts": attempts, "updated_at": now_iso(),
                      "recovery_required": "same_submission_verifier_only"}
            write_json(output, result)
            return result

        write_json(
            output,
            {
                "task_id": task_id,
                "status": "retrying_infrastructure",
                "attempts": attempts,
                "updated_at": now_iso(),
            },
        )

    result = {
        "task_id": task_id,
        "status": "infrastructure_failed",
        "attempts": attempts,
        "updated_at": now_iso(),
    }
    write_json(output, result)
    return result


def initialize_grading(args: argparse.Namespace, instances: dict[str, Any], task_ids: list[str]) -> None:
    """Freeze grader implementation/config before any new Solver is started."""
    import programbench

    if min(args.concurrency, args.grader_concurrency, args.pytest_workers, args.branch_workers) < 1:
        raise ValueError("all concurrency limits must be positive")
    args.grading_expected = {
        task_id: {branch: info["tests"] for branch, info in instances[task_id]["branches"].items()
                  if not info.get("ignored")}
        for task_id in task_ids
    }
    budgets = {task_id: process_budget(len(expected), args.branch_workers,
                                      args.evaluator_branch_retries, args.evaluator_branch_timeout_s)
               for task_id, expected in args.grading_expected.items()}
    if args.evaluator_process_timeout_s < 0 or (
        args.evaluator_process_timeout_s and args.evaluator_process_timeout_s < max(budgets.values())
    ):
        raise ValueError("whole-grader watchdog is shorter than the branch scheduling/retry budget")
    args.grader_gate = threading.BoundedSemaphore(args.grader_concurrency)
    repo = Path(__file__).resolve().parents[1]
    package = Path(programbench.__file__).resolve().parent
    source_hashes = evaluator_sources(package, repo)
    from programbench.constants import HF_REPO_ID, HF_REVISION
    hf_home = Path(os.environ.get("HF_HOME") or repo / "cache" / "huggingface").expanduser().resolve()
    cache_repo = hf_home / "hub" / (
        "datasets--" + HF_REPO_ID.replace("/", "--"))
    revision_ref = cache_repo / "refs" / HF_REVISION
    args.grading_revision = revision_ref.read_text().strip() if revision_ref.is_file() else HF_REVISION
    blob_hashes = {}
    from programbench.eval.grader_dependencies import declared_go_toolchains, python_fixture_metadata
    args.grading_toolchains = {}
    args.grading_prerequisites = {}
    for task_id in task_ids:
        blob = cache_repo / "snapshots" / args.grading_revision / task_id
        archives = []
        for branch in args.grading_expected[task_id]:
            archive = blob / "tests" / f"{branch}.tar.gz"
            if not archive.is_file():
                raise ValueError(f"hidden grader archive must be prefetched: {task_id}/{branch}")
            blob_hashes[f"{task_id}/tests/{branch}.tar.gz"] = sha256(archive)
            archives.append(archive)
        args.grading_toolchains[task_id] = declared_go_toolchains(archives)
        args.grading_prerequisites[task_id] = {
            b:python_fixture_metadata(blob/'tests'/f'{b}.tar.gz')['prerequisites']
            for b in args.grading_expected[task_id]}
        budgets[task_id] = process_budget(len(args.grading_expected[task_id]), args.branch_workers,
            args.evaluator_branch_retries, args.evaluator_branch_timeout_s,
            len(args.grading_toolchains[task_id]))
    if args.evaluator_process_timeout_s and args.evaluator_process_timeout_s < max(budgets.values()):
        raise ValueError("whole-grader watchdog does not cover fixture toolchain preparation")
    reference_manifest = getattr(args, "grading_test_manifest", None)
    if reference_manifest:
        reference_manifest = Path(reference_manifest).resolve()
        args.grading_expected = apply_reference_collections(
            args.grading_expected, reference_manifest, blob_hashes,
            {task: get_ignored_tests(instances[task]) for task in task_ids})
    config = {key: getattr(args, key) for key in (
        "concurrency", "grader_concurrency", "branch_workers", "pytest_workers", "evaluator_cpus",
        "evaluator_branch_timeout_s", "evaluator_branch_retries", "evaluator_process_timeout_s",
        "max_verifier_attempts")}
    config["grader_go_proxy_sha256"] = hashlib.sha256(grader_go_proxy().encode()).hexdigest()
    content = {"contract": GRADING_CONTRACT, "source_sha256": source_hashes, "config": config,
               "skip_completeness_policy":"only official statically declared prerequisite skips with proven model failure; unresolved skips are Infra; fixed denominator",
               "dependency_scheduling":"serial pytest for branches declaring dependency markers; assertions, timeouts and retry counts unchanged",
               "python_fixture_preflight":"grader-only pinned pysam 0.23.3 when official fixtures import it",
               "mysql_service_preflight":"official archive-declared setup before test timers; private 4GiB tmpfs data directory; native MySQL flags, fixtures and assertions unchanged; failed startup stays Infra",
               "root_fixture_service_preflight":"also preflight literal no-argument setup helper calls from official Python fixtures, using their declared working directory",
               "pytest_worker_cap":"enforce configured cap on explicit -n as well as auto; propagate to container and require activation marker",
               "declared_pytest_prerequisites":args.grading_prerequisites,
               "hidden_test_revision": args.grading_revision, "hidden_archive_sha256": blob_hashes,
               "prefetched_fixture_go_toolchains": args.grading_toolchains,
               "reference_collection_manifest": str(reference_manifest) if reference_manifest else None,
               "reference_collection_manifest_sha256": sha256(reference_manifest) if reference_manifest else None,
               "task_ids": task_ids, "task_split_sha256": sha256(args.task_ids),
               "process_budgets_s": budgets, "test_assertions": "unchanged",
               "individual_test_limits": "unchanged; extra executable wrapper removed",
               "reporting_guard": "bounded_assertions.py; diagnostics only",
               "reporter_installation": "grader interpreter site-packages; independent of test driver PYTHONPATH",
               "reporter_empty_branch_policy": "skip only for explicit empty metadata, successful script, strictly empty valid JUnit XML",
               "suite_early_exit": "pytest maxfail=0; run all declared tests",
               "test_retries": 2, "test_retry_delay_s": 0,
               "test_retry_timeouts": "rearm unchanged pytest-timeout budget on each retry; preserve first attempt, per-test markers and func_only timers",
               "test_timeout_reporting_boundary": "whole signal budget is cumulative across setup/call/teardown; pause only for pytest reporting to prevent timeout escaping as internal error",
               "grader_go_selection": "raise only an older default launcher to declared minimum release; preserve newer native Go, go.mod and automatic selection",
               "maximum_pytest_workers": min(args.concurrency, args.grader_concurrency) * args.branch_workers * args.pytest_workers}
    args.grader_manifest = args.run_dir / "reports" / "grading_manifest.json"
    existing = read_json(args.grader_manifest)
    if existing is not None and existing != content:
        raise ValueError("grading version/config changed; use an isolated recovery or new run directory")
    if existing is None and any((args.run_dir / "tasks").glob("*/result.json")):
        raise ValueError("legacy results have no grading contract; audit/recover separately, do not relabel on resume")
    args.grader_package = args.run_dir / "grader_snapshot" / "programbench"
    if not args.grader_package.exists():
        shutil.copytree(package, args.grader_package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if evaluator_sources(args.grader_package, repo) != source_hashes:
        raise ValueError("frozen evaluator snapshot hash mismatch")
    write_json(args.grader_manifest, content)


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    args.tasks_root = args.tasks_root.resolve()
    args.programbench_root = args.programbench_root.resolve()
    args.docker_executable = args.docker_executable.resolve()
    args.proxy_log = args.proxy_log.resolve() if args.proxy_log else None
    args.harness_dir = args.harness_dir.resolve() if args.harness_dir else None
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "reports").mkdir(parents=True, exist_ok=True)
    task_ids = selected_task_ids(args)
    instances = {item["instance_id"]: item for item in load_all_instances()}
    initialize_grading(args, instances, task_ids)

    args.runtime_policy = (
        load_runtime_policy(args.harness_dir) if args.harness_dir is not None else None
    )
    args.runtime_config_path = None
    args.classifier_agent = None
    args.classifier_work_root = args.run_dir / "classifier_workdir"
    args.trajectory_store = FilesystemTrajectoryStore(args.run_dir / "trajectories")
    if args.runtime_policy is not None:
        if args.runtime_policy.adjuster is not None:
            raise SystemExit("ProgramBench runtime profile adjusters are not supported")
        hook = args.harness_dir / "runtime" / "hook.py"
        if not hook.is_file():
            raise SystemExit(f"runtime policy requires hook: {hook}")
        args.runtime_config_path = render_container_runtime_config(
            args.runtime_policy,
            args.run_dir / "codex_runtime.toml",
        )
        classifier = args.runtime_policy.classifier
        if classifier is not None:
            proxy_port = os.environ.get("RHO_PROGRAMBENCH_PROXY_PORT", "").strip()
            if not proxy_port:
                raise SystemExit("runtime classifier requires RHO_PROGRAMBENCH_PROXY_PORT")
            classifier_config = args.run_dir / "classifier_codex_config.toml"
            classifier_config.write_text(
                "\n".join(
                    [
                        'model_provider = "rho"',
                        "",
                        "[model_providers.rho]",
                        'name = "ProgramBench local Responses proxy"',
                        f'base_url = "http://127.0.0.1:{proxy_port}/v1"',
                        'env_key = "OPENAI_API_KEY"',
                        'env_http_headers = { "X-Rho-Transport-Trace" = "RHO_TRANSPORT_TRACE_ID" }',
                        'wire_api = "responses"',
                        "supports_websockets = false",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            configure_global_codex_pool(args.concurrency)
            args.classifier_work_root.mkdir(parents=True, exist_ok=True)
            args.classifier_agent = CodexAgent(
                codex_config_path=classifier_config,
                model=classifier.model,
                reasoning_effort=classifier.reasoning_effort,
                sandbox="read-only",
                network_access=False,
                binary=os.environ.get("HARNESSPRO_CODEX_BINARY", "codex"),
                default_timeout_s=classifier.timeout_s,
            )

    lock = (args.run_dir / ".run.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit(f"run directory is already active: {args.run_dir}") from exc

    ignored = {task_id: get_ignored_tests(instances[task_id]) for task_id in task_ids}
    started_at = now_iso()
    started_monotonic = time.monotonic()
    write_summary(
        args,
        task_ids,
        started_at=started_at,
        started_monotonic=started_monotonic,
        final=False,
    )

    if args.classify_only:
        if args.runtime_policy is None or args.classifier_agent is None:
            raise SystemExit("--classify-only requires a model runtime classifier")
        selections = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(
                    classify_runtime_profile,
                    args,
                    task_id=task_id,
                    instance=instances[task_id],
                ): task_id
                for task_id in task_ids
            }
            for future in as_completed(futures):
                task_id = futures[future]
                selection = future.result()
                selections.append(
                    {
                        "task_id": task_id,
                        **args.runtime_policy.workspace_selection_payload(selection),
                    }
                )
                print(f"[profile] {task_id} profile={selection.profile.name}", flush=True)
        selections.sort(key=lambda item: item["task_id"])
        classifier_formal, classifier_excluded = _classification_attempts(args.run_dir)
        report = {
            "task_count": len(selections),
            "profile_counts": dict(
                sorted(Counter(item["profile"] for item in selections).items())
            ),
            "metrics": aggregate_attempts(classifier_formal),
            "excluded_infrastructure": aggregate_attempts(classifier_excluded),
            "classifications": selections,
        }
        write_json(args.run_dir / "reports" / "classification_summary.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                run_task,
                args,
                task_id,
                instances[task_id],
                ignored[task_id],
            ): task_id
            for task_id in task_ids
        }
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                result = future.result()
                print(
                    f"[{task_id}] {result.get('status')} "
                    f"score={result.get('score')} attempts={len(result.get('attempts') or [])}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[{task_id}] runner exception: {exc}", flush=True)
            write_summary(
                args,
                task_ids,
                started_at=started_at,
                started_monotonic=started_monotonic,
                final=False,
            )

    summary = write_summary(
        args,
        task_ids,
        started_at=started_at,
        started_monotonic=started_monotonic,
        final=True,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
