"""Infrastructure-aware ProgramBench result parsing and aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from rho.datasets.programbench.trial_results import (
    INFRA_TEXT_MARKERS,
    _timeout_adjustment,
    _trial_timing,
    aggregate_attempts,
    codex_metrics,
    proxy_metrics,
    read_json,
    trial_result_path,
)
from rho.pricing import estimate_usage_cost


PROGRAMBENCH_INFRA_TEXT_MARKERS = (
    "tee: /logs/agent/codex.txt: permission denied",
    "command failed (exit 143)",
    "command failed (exit 137)",
)

PROGRAMBENCH_INFRA_ERROR_CODES = {
    "install_evaluator_dependencies_failed",
}


def _text_contains_infra(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (*INFRA_TEXT_MARKERS, *PROGRAMBENCH_INFRA_TEXT_MARKERS)
    )


def attempt_has_infrastructure_evidence(attempt: dict[str, Any]) -> bool:
    """Return whether a persisted attempt contains decisive Infra evidence.

    Older runner versions could grade a partial workspace after the agent was
    killed and then persist the verifier's build failure as a model result.
    Keep the check evidence-based so ordinary compile failures remain valid
    model outcomes.
    """

    evidence = "\n".join(
        [
            str(attempt.get("classification_reason") or ""),
            json.dumps(attempt.get("exception_info") or {}, ensure_ascii=False),
        ]
    )
    return _text_contains_infra(evidence)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def pier_job_root(attempt_dir: Path) -> Path | None:
    """Locate Pier's generated job directory below one logical attempt."""

    candidates: list[Path] = []
    for path in sorted(attempt_dir.glob("*/result.json")):
        payload = read_json(path)
        if payload is not None and "n_total_trials" in payload and "stats" in payload:
            candidates.append(path.parent)
    return candidates[0] if len(candidates) == 1 else None


def summarize_solver(
    *,
    task_id: str,
    attempt: int,
    trace_id: str,
    job_dir: Path,
    pier_exit_code: int,
    proxy_log: Path | None,
    model: str,
) -> dict[str, Any]:
    resolved_job_dir = pier_job_root(job_dir)
    job = read_json(resolved_job_dir / "result.json") if resolved_job_dir else None
    result_path = trial_result_path(resolved_job_dir) if resolved_job_dir else None
    trial = read_json(result_path) if result_path is not None else None
    trial_dir = result_path.parent if result_path is not None else job_dir
    artifact = trial_dir / "artifacts" / "submission.tar.gz"

    reason = "ProgramBench solver produced a submission"
    classification = "solver_ready"
    if job is None or not job.get("finished_at"):
        classification, reason = "infrastructure", "Pier job did not finish cleanly"
    elif trial is None:
        classification, reason = "infrastructure", "Pier trial result is missing or ambiguous"
    else:
        exception = trial.get("exception_info") or {}
        exception_type = str(exception.get("exception_type") or "")
        exception_message = str(exception.get("exception_message") or "")
        if exception_type == "AgentTimeoutError":
            reason = "agent exhausted the ProgramBench time budget; grading partial workspace"
        elif exception_type:
            evidence = "\n".join(
                [
                    exception_message,
                    _read_text(trial_dir / "agent" / "codex.txt"),
                    _read_text(job_dir / "runner.log"),
                ]
            )
            if _text_contains_infra(evidence) or exception_type != "NonZeroAgentExitCodeError":
                classification = "infrastructure"
                reason = exception_type or "agent runtime failure"
            else:
                reason = "Codex exited non-zero; grading the collected workspace"

    if classification == "solver_ready" and not artifact.is_file():
        classification = "infrastructure"
        reason = "submission artifact is missing"

    metrics = codex_metrics(trial_dir / "agent")
    transport = proxy_metrics(proxy_log, trace_id, model=model)
    timing = _trial_timing(trial or {})
    timeout = _timeout_adjustment(trial or {}, timing, transport)
    transport_usage = (transport or {}).get("usage")
    agent_usage = metrics.get("usage")
    usage = (
        transport_usage
        if (transport or {}).get("usage_complete") or agent_usage is None
        else agent_usage
    )
    if (
        classification == "solver_ready"
        and transport is not None
        and int(transport.get("logical_api_requests") or 0) > 0
        and not bool(transport.get("usage_complete"))
    ):
        classification = "infrastructure"
        reason = "provider response stream ended without complete usage"

    return {
        "task_id": task_id,
        "attempt": attempt,
        "trace_id": trace_id,
        "classification": classification,
        "classification_reason": reason,
        "pier_exit_code": pier_exit_code,
        "job_dir": str(job_dir),
        "pier_job_dir": str(resolved_job_dir) if resolved_job_dir else None,
        "trial_dir": str(trial_dir) if result_path is not None else None,
        "submission_artifact": str(artifact) if artifact.is_file() else None,
        "exception_info": (trial or {}).get("exception_info"),
        "usage": usage,
        "cost": estimate_usage_cost(model, usage),
        "tools": {
            "count": metrics.get("tool_calls"),
            "by_type": metrics.get("tool_counts") or {},
            "observed_duration_s": metrics.get("tool_observed_duration_s"),
            "duration_count": metrics.get("tool_duration_count"),
        },
        "api": transport
        or {
            "logical_api_requests": metrics.get("api_requests"),
            "upstream_attempts": metrics.get("api_requests"),
            "retry_count": None,
        },
        "timing": {**timing, **timeout},
        "agent_metrics": metrics,
    }


def parse_evaluation(
    eval_path: Path,
    *,
    ignored_tests: set[str],
    expected_tests_by_branch: dict[str, list[str]] | None = None,
    require_xml: bool = False,
    prerequisite_metadata: dict | None = None,
    terminal_skip_review: dict | None = None,
) -> dict[str, Any]:
    if terminal_skip_review is not None and not require_xml:
        return {'classification':'infrastructure', 'reason':'terminal skip review requires complete XML validation'}
    raw = read_json(eval_path)
    if raw is None:
        return {
            "classification": "infrastructure",
            "reason": "official ProgramBench eval JSON is missing",
        }

    active_by_name: dict[tuple[str, str], dict[str, Any]] = {}
    for item in raw.get("test_results", []):
        if not isinstance(item, dict):
            continue
        branch = str(item.get("branch") or "")
        name = str(item.get("name") or "")
        if f"{branch}/{name}" in ignored_tests:
            continue
        # pytest-rerunfailures writes intermediate executions before the final
        # outcome under the same branch/name. Last-write-wins reconstructs the
        # actual test result and keeps the denominator invariant by Harness.
        active_by_name[(branch, name)] = item
    active = list(active_by_name.values())
    resolved = sum(item.get("status") == "passed" for item in active)
    system_errors = sum(item.get("status") == "system_error" for item in active)
    branch_errors = raw.get("test_branch_errors") or {}
    error_code = raw.get("error_code")
    error_details = str(raw.get("error_details") or "")

    if error_code:
        setup_interrupted = any(
            entry.get("step") in {"compile", "copy_executable"}
            and (entry.get("returncode") in {-1, 137, 143}
                 or bool(entry.get("exception_info")))
            for entry in raw.get("log", [])
        )
        # Only an observed submission build failure is a model outcome without
        # executed tests. Unknown setup/collection failures must fail closed.
        if setup_interrupted or error_code not in {"compile_failed", "copy_executable_failed"} or _text_contains_infra(error_details):
            return {
                "classification": "infrastructure",
                "reason": f"official evaluator infrastructure error: {error_code}",
                "raw": raw,
            }
        return {
            "classification": "model_result",
            "reason": f"submission failed official build stage: {error_code}",
            "score": 0.0,
            "resolved_tests": 0,
            "total_tests": len(active),
            "fully_solved": False,
            "raw": raw,
        }

    not_run = sum(item.get("status") == "not_run" for item in active)
    unknown_statuses = sum(
        item.get("status") not in {"passed", "failure", "error", "skipped", "not_run", "system_error"}
        for item in active
    )
    worker_crashes = sum(
        "worker '" in json.dumps(item.get("extra") or {})
        and "crashed" in json.dumps(item.get("extra") or {})
        for item in active
    )
    if branch_errors or system_errors or not_run or unknown_statuses or worker_crashes:
        return {
            "classification": "infrastructure",
            "reason": "official evaluator did not produce complete branch results",
            "resolved_tests": resolved,
            "total_tests": len(active),
            "system_errors": system_errors,
            "not_run": not_run,
            "worker_crashes": worker_crashes,
            "branch_errors": branch_errors,
            "raw": raw,
        }
    if not active:
        return {
            "classification": "infrastructure",
            "reason": "official evaluator produced no active test results",
            "raw": raw,
        }

    # Reconcile identities, not just counts. A same-size replacement or a
    # completely absent branch is otherwise indistinguishable from completion.
    if expected_tests_by_branch is not None:
        expected = {
            (branch, name)
            for branch, names in expected_tests_by_branch.items()
            for name in names
            if f"{branch}/{name}" not in ignored_tests
        }
        got = set(active_by_name)
        missing, unexpected = expected - got, got - expected
        if missing or unexpected:
            return {
                "classification": "infrastructure",
                "reason": "test manifest and evaluation identities do not reconcile",
                "missing_test_count": len(missing),
                "unexpected_test_count": len(unexpected),
                "raw": raw,
            }
    expected_branches = set(expected_tests_by_branch or raw.get("test_branches") or [])
    # A metadata-declared branch may contain only officially ignored tests.
    # It still has raw records/XML; do not invent an absent-branch failure
    # merely because the score denominator correctly excludes those tests.
    observed_branches = {str(item.get("branch") or "") for item in raw.get("test_results", [])}
    branches_with_tests = ({b for b, names in expected_tests_by_branch.items() if names}
                          if expected_tests_by_branch is not None else expected_branches)
    if branches_with_tests - observed_branches:
        return {"classification": "infrastructure", "reason": "expected branch is absent", "raw": raw}
    if require_xml:
        from rho.datasets.programbench.grading import reconcile_xml
        reviewed = None
        if terminal_skip_review is not None:
            from rho.datasets.programbench.skip_proofs import validate_terminal_skip_review
            try:
                reviewed = validate_terminal_skip_review(terminal_skip_review, raw)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                return {'classification':'infrastructure', 'reason':'invalid terminal skip proof: '+str(exc), 'raw':raw}
        issue = reconcile_xml(raw, active_by_name, expected_branches, ignored_tests, prerequisite_metadata, reviewed)
        if issue:
            return {"classification": "infrastructure", "reason": issue, "raw": raw}
    score = resolved / len(active)
    return {
        "classification": "model_result",
        "reason": "official ProgramBench evaluator produced complete results",
        "score": score,
        "resolved_tests": resolved,
        "total_tests": len(active),
        "fully_solved": resolved == len(active),
        "warnings": raw.get("warnings") or [],
        "raw": raw,
    }


def aggregate_formal(attempts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(attempts)
    base = aggregate_attempts(values)
    base["score"] = {
        "average": (
            sum(float(item.get("score") or 0.0) for item in values) / len(values)
            if values
            else None
        ),
        "fully_solved": sum(bool(item.get("fully_solved")) for item in values),
        "resolved_tests": sum(int(item.get("resolved_tests") or 0) for item in values),
        "total_tests": sum(int(item.get("total_tests") or 0) for item in values),
    }
    return base
