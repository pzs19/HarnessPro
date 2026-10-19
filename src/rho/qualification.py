from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_qualification_target(path: Path) -> dict[str, Any]:
    target = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(target, dict):
        raise ValueError("qualification target must be a JSON object")
    if int(target.get("version", 0)) != 1:
        raise ValueError("unsupported qualification target version")
    return target


def qualify_metrics(
    metrics: dict[str, Any],
    run: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, Any]:
    overall = metrics.get("overall") or {}
    cost = metrics.get("cost") or {}
    requested = int(run.get("requested_tasks") or 0)
    completed = int(run.get("completed_tasks") or 0)
    passed = int(run.get("passed") or 0)
    infrastructure_failed = int(run.get("infrastructure_failed") or 0)
    pending = int(run.get("pending") or 0)

    expected_tasks = int(target["requested_tasks"])
    minimum_passed = int(target["minimum_passed"])
    maximum_infrastructure_failures = int(
        target.get("maximum_infrastructure_failures", 0)
    )
    result_set_complete = (
        requested == expected_tasks
        and completed == expected_tasks
        and pending == 0
        and infrastructure_failed <= maximum_infrastructure_failures
    )

    performance_status = (
        "pass"
        if result_set_complete and passed >= minimum_passed
        else "fail"
        if result_set_complete
        else "indeterminate"
    )

    trajectory_count = int(overall.get("trajectory_count") or 0)
    with_usage_count = int(
        overall.get("with_usage_count")
        or metrics.get("with_usage_count")
        or 0
    )
    priced_count = int(cost.get("priced_trajectory_count") or 0)
    unpriced_count = int(cost.get("unpriced_trajectory_count") or 0)
    usage_complete = trajectory_count > 0 and with_usage_count == trajectory_count
    pricing_complete = usage_complete and priced_count == with_usage_count and unpriced_count == 0

    cost_target = target["cost"]
    baseline_cost = float(cost_target["baseline_estimated_total_usd"])
    maximum_fraction = float(cost_target["maximum_fraction"])
    ceiling_cost = baseline_cost * maximum_fraction
    observed_cost = float(cost.get("estimated_total_usd") or 0.0)
    if observed_cost > ceiling_cost:
        cost_status = "fail"
    elif pricing_complete:
        cost_status = "pass"
    else:
        cost_status = "indeterminate"

    token_target = target.get("tokens")
    token_checks: dict[str, Any] = {}
    if isinstance(token_target, dict):
        baseline_tokens = token_target["baseline"]
        maximum_fractions = token_target["maximum_fraction"]
        for key in ("input_tokens", "uncached_input_tokens", "output_tokens"):
            baseline = int(baseline_tokens[key])
            fraction = float(maximum_fractions[key])
            ceiling = baseline * fraction
            observed = int(overall.get(key) or 0)
            if observed > ceiling:
                status = "fail"
            elif usage_complete:
                status = "pass"
            else:
                status = "indeterminate"
            token_checks[key] = {
                "baseline": baseline,
                "maximum_fraction": fraction,
                "ceiling": round(ceiling, 6),
                "observed": observed,
                "reduction_fraction": round(1.0 - observed / baseline, 9),
                "status": status,
            }
        token_statuses = {item["status"] for item in token_checks.values()}
        token_status = (
            "fail"
            if "fail" in token_statuses
            else "pass"
            if token_statuses == {"pass"}
            else "indeterminate"
        )
    else:
        token_status = "not_applicable"

    statuses = {performance_status, cost_status}
    if token_status != "not_applicable":
        statuses.add(token_status)
    overall_status = (
        "fail"
        if "fail" in statuses
        else "pass"
        if statuses == {"pass"}
        else "indeterminate"
    )

    return {
        "target_name": str(target.get("name") or "unnamed"),
        "status": overall_status,
        "validity": {
            "expected_tasks": expected_tasks,
            "requested_tasks": requested,
            "completed_tasks": completed,
            "pending_tasks": pending,
            "infrastructure_failed": infrastructure_failed,
            "maximum_infrastructure_failures": maximum_infrastructure_failures,
            "result_set_complete": result_set_complete,
            "trajectory_count": trajectory_count,
            "with_usage_count": with_usage_count,
            "priced_trajectory_count": priced_count,
            "unpriced_trajectory_count": unpriced_count,
            "usage_complete": usage_complete,
            "pricing_complete": pricing_complete,
            "cost_is_lower_bound": not pricing_complete,
        },
        "performance": {
            "minimum_passed": minimum_passed,
            "observed_passed": passed,
            "status": performance_status,
        },
        "cost": {
            "baseline_label": str(cost_target["baseline_label"]),
            "baseline_estimated_total_usd": round(baseline_cost, 9),
            "maximum_fraction": maximum_fraction,
            "required_reduction_fraction": round(1.0 - maximum_fraction, 9),
            "ceiling_estimated_total_usd": round(ceiling_cost, 9),
            "observed_estimated_total_usd": round(observed_cost, 9),
            "observed_fraction_of_baseline": round(observed_cost / baseline_cost, 9),
            "estimated_cost_per_completed_task_usd": (
                round(observed_cost / completed, 9) if completed else None
            ),
            "estimated_cost_per_pass_usd": (
                round(observed_cost / passed, 9) if passed else None
            ),
            "status": cost_status,
        },
        "tokens": {
            "baseline_label": (
                str(token_target["baseline_label"])
                if isinstance(token_target, dict)
                else None
            ),
            "status": token_status,
            "checks": token_checks,
        },
    }
