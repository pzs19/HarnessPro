"""Order full evaluations using baseline outcomes without exposing them to solvers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def failure_first_schedule(task_ids: list[str], baseline: Path) -> dict:
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("evaluation task IDs must be nonempty and unique")
    baseline = baseline.resolve()
    summary_bytes = (baseline / "reports/summary.json").read_bytes()
    summary = json.loads(summary_bytes)
    results = {}
    hashes = {}
    for path in sorted((baseline / "tasks").glob("*/result.json")):
        raw = path.read_bytes()
        result = json.loads(raw)
        identity = result.get("task_id")
        if (not isinstance(identity, str) or identity != path.parent.name
                or identity in results or result.get("status") != "completed"
                or not isinstance(result.get("passed"), bool)):
            raise ValueError("baseline results must be unique completed model outcomes")
        results[identity] = result
        hashes[identity] = hashlib.sha256(raw).hexdigest()
    if (summary.get("final") is not True
            or summary.get("requested_tasks") != len(task_ids)
            or summary.get("completed_tasks") != len(task_ids)
            or summary.get("infrastructure_failed") != 0
            or set(results) != set(task_ids)
            or summary.get("passed") != sum(result["passed"] for result in results.values())):
        raise ValueError("baseline must be complete, infrastructure-clean, and match the full task set")
    priority = [identity for identity in task_ids if not results[identity]["passed"]]
    retained = [identity for identity in task_ids if results[identity]["passed"]]
    return {
        "ordered_task_ids": priority + retained,
        "priority_task_ids": priority,
        "baseline_run": str(baseline),
        "baseline_summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "baseline_result_sha256": hashes,
        "scope": "Evaluation-only dispatch ordering; no baseline answers, artifacts, or outcomes enter model input. All tasks receive fresh evaluation in this run.",
    }
