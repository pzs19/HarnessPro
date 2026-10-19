from __future__ import annotations

import dataclasses
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rho.pricing import PRICING_AS_OF, PRICING_SOURCE, estimate_usage_cost
from rho.protocols import Trajectory

_RUNTIME_SCRATCH_NAMES = {
    ".rho",
    ".rho_runtime",
    ".sample_index",
    ".codex",
}
_USAGE_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
)
_NON_TOOL_ITEM_TYPES = {"agent_message", "reasoning"}
_RETRYABLE_TRANSPORT_MARKERS = (
    "stream disconnected before completion",
    "response.failed event received",
    "connection error",
    "connection reset",
    "connection refused",
    "connection closed",
    "payload too large",
    "too many requests",
    "http 413",
    "http 429",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
)


def is_runtime_scratch(rel: str | Path) -> bool:
    path = Path(rel)
    return any(part in _RUNTIME_SCRATCH_NAMES for part in path.parts)


def resolve_agent_metadata(agent: object | None) -> tuple[str | None, str | None, str]:
    cache_mode = "off"
    current = agent
    while current is not None:
        inner = getattr(current, "inner", None)
        mode = getattr(current, "mode", None)
        if isinstance(mode, str):
            cache_mode = mode
        if inner is None:
            break
        current = inner
    model = getattr(current, "model", None)
    reasoning_effort = getattr(current, "reasoning_effort", None)
    return (
        model if isinstance(model, str) or model is None else str(model),
        reasoning_effort
        if isinstance(reasoning_effort, str) or reasoning_effort is None
        else str(reasoning_effort),
        cache_mode,
    )


def annotate_trajectory(
    trajectory: Trajectory,
    *,
    agent: object | None = None,
    stage: str | None = None,
    round_ix: int | None = None,
    sample_index: int | None = None,
) -> Trajectory:
    model = trajectory.model
    reasoning_effort = trajectory.reasoning_effort
    cache_mode = trajectory.cache_mode
    if agent is not None:
        agent_model, agent_reasoning_effort, agent_cache_mode = resolve_agent_metadata(
            agent
        )
        if model is None:
            model = agent_model
        if reasoning_effort is None:
            reasoning_effort = agent_reasoning_effort
        if cache_mode is None:
            cache_mode = agent_cache_mode
    return dataclasses.replace(
        trajectory,
        stage=stage if stage is not None else trajectory.stage,
        round_ix=round_ix if round_ix is not None else trajectory.round_ix,
        sample_index=sample_index if sample_index is not None else trajectory.sample_index,
        model=model,
        reasoning_effort=reasoning_effort,
        cache_mode=cache_mode,
    )


def extract_usage(events: list[dict[str, Any]]) -> dict[str, int] | None:
    totals = Counter()
    saw_usage = False
    for event in events:
        if event.get("type") != "turn.completed":
            continue
        usage = event.get("usage")
        if not isinstance(usage, dict):
            continue
        for key in _USAGE_KEYS:
            value = usage.get(key)
            if value is None:
                continue
            totals[key] += int(value)
            saw_usage = True
    if not saw_usage:
        return None
    usage = {key: int(totals[key]) for key in _USAGE_KEYS}
    usage["uncached_input_tokens"] = max(
        0,
        usage["input_tokens"] - usage["cached_input_tokens"],
    )
    return usage


def extract_usage_by_model(
    events: list[dict[str, Any]], default_model: str | None
) -> dict[str, dict[str, int]]:
    """Group usage events by the model that actually served each request."""

    totals: dict[str, Counter[str]] = defaultdict(Counter)
    for event in events:
        if event.get("type") != "turn.completed":
            continue
        usage = event.get("usage")
        if not isinstance(usage, dict):
            continue
        model = str(event.get("usage_model") or default_model or "(unknown)")
        bucket = totals[model]
        for key in _USAGE_KEYS:
            value = usage.get(key)
            if value is not None:
                bucket[key] += int(value)
    grouped: dict[str, dict[str, int]] = {}
    for model, bucket in totals.items():
        usage = {key: int(bucket[key]) for key in _USAGE_KEYS}
        usage["uncached_input_tokens"] = max(
            0,
            usage["input_tokens"] - usage["cached_input_tokens"],
        )
        grouped[model] = usage
    return grouped


def trajectory_estimated_cost(trajectory: Trajectory) -> dict[str, Any] | None:
    grouped = extract_usage_by_model(trajectory.events, trajectory.model)
    if not grouped:
        return None
    if len(grouped) == 1:
        model, usage = next(iter(grouped.items()))
        return estimate_usage_cost(model, usage)

    by_model: dict[str, dict[str, Any]] = {}
    totals = Counter()
    for model, usage in grouped.items():
        cost = estimate_usage_cost(model, usage)
        if cost is None:
            return None
        by_model[model] = cost
        for key in (
            "estimated_total_usd",
            "regular_uncached_input_usd",
            "cache_write_input_usd",
            "cached_input_usd",
            "output_usd",
        ):
            totals[key] += float(cost[key])
    return {
        "currency": "USD",
        **{key: round(float(value), 9) for key, value in totals.items()},
        "pricing_model": "mixed",
        "by_model": by_model,
        "pricing_source": PRICING_SOURCE,
        "pricing_as_of": PRICING_AS_OF,
        "estimate_note": (
            "Published OpenAI standard rates applied to request usage grouped by "
            "the actual routed model; provider invoices can differ."
        ),
    }


def extract_api_calls(events: list[dict[str, Any]]) -> int | None:
    total = 0
    observed = False
    for event in events:
        value = event.get("api_calls")
        if value is None:
            continue
        total += max(0, int(value))
        observed = True
    return total if observed else None


def extract_usage_trace_ids(events: list[dict[str, Any]]) -> list[str]:
    """Return provider trace identifiers carried by usage events.

    The persistent Claude Code compatibility trace can receive the final SSE
    usage record after the agent process exits. Keeping its identifier in the
    normal metrics report lets the score-aligned summarizer account for those
    late provider requests without guessing from token totals.
    """

    trace_ids: list[str] = []
    seen: set[str] = set()
    for event in events:
        value = event.get("usage_trace_id")
        if not isinstance(value, str) or not value or value in seen:
            continue
        seen.add(value)
        trace_ids.append(value)
    return trace_ids


def retryable_transport_failure(trajectory: Trajectory) -> bool:
    """Whether an agent attempt failed before producing a model result.

    These signatures describe provider/gateway transport failures, not an
    incorrect task solution.  Callers may retry them while preserving every
    attempt for audit and cost accounting.  Codex can recover from transient
    transport errors and later terminate for an unrelated reason, so inspect
    only the terminal error sequence when structured events are available.
    """
    if trajectory.exit_code == 0:
        return False

    terminal_event: dict[str, Any] | None = None
    reached_terminal_boundary = False
    for event in reversed(trajectory.events):
        if event.get("_rho_outcome_relevant") is False:
            continue
        event_type = event.get("type")
        if event_type in {"turn.failed", "error"}:
            terminal_event = event
            reached_terminal_boundary = True
            break
        if event_type in {
            "item.started",
            "item.completed",
            "turn.completed",
        }:
            reached_terminal_boundary = True
            break

    if terminal_event is not None:
        haystack = str(terminal_event).lower()
    elif reached_terminal_boundary:
        # A structured event stream ending in normal agent/tool activity is a
        # timeout or model failure, even if an earlier request was retried.
        haystack = str(trajectory.stderr or "").lower()
    else:
        # Legacy and external agents may not emit structured events. Restrict
        # their fallback to the final response and output tail so task content
        # earlier in a long transcript cannot resemble a provider failure.
        stdout_tail = str(trajectory.stdout or "")[-16384:]
        haystack = (
            f"{trajectory.final_message}\n{stdout_tail}\n{trajectory.stderr}"
        ).lower()
    return any(marker in haystack for marker in _RETRYABLE_TRANSPORT_MARKERS)


def tool_call_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize Codex tool items and locally observed execution time.

    Codex JSONL emits item.started/item.completed pairs but no timestamps.
    CodexAgent adds ``_rho_observed_at_s`` as each line arrives. Durations are
    therefore local wall-clock observations around the CLI tool execution, not
    server-provided timing. Completed-only items are still counted, with a
    null duration so missing timing is visible rather than silently treated as
    zero.
    """

    calls: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for event in events:
        event_type = event.get("type")
        if event_type not in {"item.started", "item.completed"}:
            continue
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "unknown")
        if item_type in _NON_TOOL_ITEM_TYPES:
            continue
        item_id = str(item.get("id") or f"anonymous_{len(order)}")
        if item_id not in calls:
            calls[item_id] = {
                "item_id": item_id,
                "type": item_type,
                "started_at_s": None,
                "completed_at_s": None,
                "duration_s": None,
                "status": item.get("status"),
            }
            order.append(item_id)
        record = calls[item_id]
        observed = event.get("_rho_observed_at_s")
        observed_s = float(observed) if isinstance(observed, (int, float)) else None
        if event_type == "item.started":
            if record["started_at_s"] is None:
                record["started_at_s"] = observed_s
        else:
            record["completed_at_s"] = observed_s
            record["status"] = item.get("status")

    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    duration_count = 0
    total_duration_s = 0.0
    ordered_calls: list[dict[str, Any]] = []
    for item_id in order:
        record = calls[item_id]
        started_at = record["started_at_s"]
        completed_at = record["completed_at_s"]
        if started_at is not None and completed_at is not None:
            duration = max(0.0, float(completed_at) - float(started_at))
            record["duration_s"] = round(duration, 6)
            duration_count += 1
            total_duration_s += duration
        bucket = by_type[record["type"]]
        bucket["count"] += 1
        if record["completed_at_s"] is not None:
            bucket["completed_count"] += 1
        if record["duration_s"] is not None:
            bucket["duration_count"] += 1
            bucket["observed_duration_us"] += round(record["duration_s"] * 1_000_000)
        ordered_calls.append(record)

    return {
        "tool_call_count": len(ordered_calls),
        "duration_count": duration_count,
        "total_observed_duration_s": round(total_duration_s, 6),
        "by_type": {
            item_type: {
                "count": int(bucket["count"]),
                "completed_count": int(bucket["completed_count"]),
                "duration_count": int(bucket["duration_count"]),
                "total_observed_duration_s": round(
                    int(bucket["observed_duration_us"]) / 1_000_000,
                    6,
                ),
            }
            for item_type, bucket in sorted(by_type.items())
        },
        "calls": ordered_calls,
    }


def trajectory_metrics(trajectory: Trajectory) -> dict[str, Any]:
    usage = extract_usage(trajectory.events)
    return {
        "trajectory_id": trajectory.id,
        "task_id": trajectory.task_id,
        "harness_id": trajectory.harness_id,
        "kind": trajectory.kind,
        "stage": trajectory.stage,
        "model": trajectory.model,
        "reasoning_effort": trajectory.reasoning_effort,
        "exit_code": trajectory.exit_code,
        "timed_out": trajectory.timed_out,
        "api_wall_time_s": round(trajectory.wall_time_s, 6),
        "api_call_count": extract_api_calls(trajectory.events),
        "usage_trace_ids": extract_usage_trace_ids(trajectory.events),
        "usage": usage,
        "cost": trajectory_estimated_cost(trajectory),
        "tools": tool_call_metrics(trajectory.events),
    }


def call_metrics(trajectories: list[Trajectory]) -> list[dict[str, Any]]:
    return [trajectory_metrics(trajectory) for trajectory in trajectories]


def trajectory_cost_summary(trajectories: list[Trajectory]) -> dict[str, Any]:
    """Summarize model usage cost with explicit completeness guarantees.

    A partial dollar total is useful for diagnostics but is not safe for an
    accept/reject decision. ``estimated_total_usd`` is therefore populated
    only when every trajectory has usage and a known price.
    """

    usage_totals = Counter()
    observed_total_usd = 0.0
    missing_usage_trajectory_ids: list[str] = []
    unpriced_trajectory_ids: list[str] = []
    records: list[dict[str, Any]] = []
    for trajectory in trajectories:
        usage = extract_usage(trajectory.events)
        cost = trajectory_estimated_cost(trajectory)
        if usage is None:
            missing_usage_trajectory_ids.append(trajectory.id)
        else:
            for key, value in usage.items():
                usage_totals[key] += int(value)
        if usage is not None and cost is None:
            unpriced_trajectory_ids.append(trajectory.id)
        if cost is not None:
            observed_total_usd += float(cost["estimated_total_usd"])
        records.append(
            {
                "trajectory_id": trajectory.id,
                "task_id": trajectory.task_id,
                "model": trajectory.model,
                "usage": usage,
                "cost": cost,
            }
        )

    trajectory_count = len(trajectories)
    usage_complete = trajectory_count > 0 and not missing_usage_trajectory_ids
    pricing_complete = usage_complete and not unpriced_trajectory_ids
    estimated_total_usd = round(observed_total_usd, 9) if pricing_complete else None
    return {
        "currency": "USD",
        "trajectory_count": trajectory_count,
        "usage_complete": usage_complete,
        "pricing_complete": pricing_complete,
        "estimated_total_usd": estimated_total_usd,
        "observed_total_usd": round(observed_total_usd, 9),
        "average_per_trajectory_usd": (
            round(observed_total_usd / trajectory_count, 9)
            if pricing_complete and trajectory_count
            else None
        ),
        "usage": {key: int(value) for key, value in usage_totals.items()},
        "missing_usage_trajectory_ids": missing_usage_trajectory_ids,
        "unpriced_trajectory_ids": unpriced_trajectory_ids,
        "trajectories": records,
        "pricing_source": PRICING_SOURCE,
        "pricing_as_of": PRICING_AS_OF,
    }


def usage_summary(trajectories: list[Trajectory]) -> dict[str, Any]:
    overall = Counter()
    by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    by_stage: dict[str, Counter[str]] = defaultdict(Counter)
    tool_overall = Counter()
    tool_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    cost_overall = Counter()
    cost_by_model: dict[str, Counter[str]] = defaultdict(Counter)
    cost_by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    cost_by_stage: dict[str, Counter[str]] = defaultdict(Counter)
    unpriced_trajectory_count = 0
    api_call_count = 0
    api_call_observed_trajectory_count = 0
    with_usage = 0
    for trajectory in trajectories:
        usage = extract_usage(trajectory.events)
        trajectory_api_calls = extract_api_calls(trajectory.events)
        if trajectory_api_calls is not None:
            api_call_count += trajectory_api_calls
            api_call_observed_trajectory_count += 1
        tools = tool_call_metrics(trajectory.events)
        stage = trajectory.stage or "(none)"
        kind_bucket = by_kind[trajectory.kind]
        kind_bucket["trajectory_count"] += 1
        stage_bucket = by_stage[stage]
        stage_bucket["trajectory_count"] += 1
        tool_overall["tool_call_count"] += int(tools["tool_call_count"])
        tool_overall["duration_count"] += int(tools["duration_count"])
        tool_overall["observed_duration_us"] += round(
            float(tools["total_observed_duration_s"]) * 1_000_000
        )
        for item_type, bucket in tools["by_type"].items():
            target = tool_by_type[item_type]
            target["count"] += int(bucket["count"])
            target["completed_count"] += int(bucket["completed_count"])
            target["duration_count"] += int(bucket["duration_count"])
            target["observed_duration_us"] += round(
                float(bucket["total_observed_duration_s"]) * 1_000_000
            )
        if usage is None:
            continue
        model_usage = extract_usage_by_model(trajectory.events, trajectory.model)
        model_costs = {
            model: estimate_usage_cost(model, model_tokens)
            for model, model_tokens in model_usage.items()
        }
        cost = trajectory_estimated_cost(trajectory)
        if cost is None:
            unpriced_trajectory_count += 1
        else:
            for key in (
                "estimated_total_usd",
                "regular_uncached_input_usd",
                "cache_write_input_usd",
                "cached_input_usd",
                "output_usd",
            ):
                value = float(cost[key])
                cost_overall[key] += value
                cost_by_kind[trajectory.kind][key] += value
                cost_by_stage[stage][key] += value
            for model_name, model_cost in model_costs.items():
                if model_cost is None:
                    continue
                for key in (
                    "estimated_total_usd",
                    "regular_uncached_input_usd",
                    "cache_write_input_usd",
                    "cached_input_usd",
                    "output_usd",
                ):
                    cost_by_model[model_name][key] += float(model_cost[key])
                cost_by_model[model_name]["trajectory_count"] += 1
            cost_by_kind[trajectory.kind]["trajectory_count"] += 1
            cost_by_stage[stage]["trajectory_count"] += 1
        with_usage += 1
        kind_bucket["with_usage_count"] += 1
        stage_bucket["with_usage_count"] += 1
        overall["with_usage_count"] += 1
        for key, value in usage.items():
            overall[key] += value
            kind_bucket[key] += value
            stage_bucket[key] += value
    overall["trajectory_count"] = len(trajectories)
    return {
        "overall": {key: int(value) for key, value in overall.items()},
        "by_kind": {
            kind: {key: int(value) for key, value in counter.items()}
            for kind, counter in sorted(by_kind.items())
        },
        "by_stage": {
            stage: {key: int(value) for key, value in counter.items()}
            for stage, counter in sorted(by_stage.items())
        },
        "with_usage_count": with_usage,
        "api_calls": {
            "api_call_count": api_call_count,
            "observed_trajectory_count": api_call_observed_trajectory_count,
            "missing_trajectory_count": len(trajectories)
            - api_call_observed_trajectory_count,
        },
        "cost": {
            "currency": "USD",
            **{
                key: round(float(value), 9)
                for key, value in cost_overall.items()
            },
            "priced_trajectory_count": with_usage - unpriced_trajectory_count,
            "unpriced_trajectory_count": unpriced_trajectory_count,
            "by_model": {
                name: {
                    key: (
                        int(value)
                        if key == "trajectory_count"
                        else round(float(value), 9)
                    )
                    for key, value in bucket.items()
                }
                for name, bucket in sorted(cost_by_model.items())
            },
            "by_kind": {
                name: {
                    key: (
                        int(value)
                        if key == "trajectory_count"
                        else round(float(value), 9)
                    )
                    for key, value in bucket.items()
                }
                for name, bucket in sorted(cost_by_kind.items())
            },
            "by_stage": {
                name: {
                    key: (
                        int(value)
                        if key == "trajectory_count"
                        else round(float(value), 9)
                    )
                    for key, value in bucket.items()
                }
                for name, bucket in sorted(cost_by_stage.items())
            },
            "pricing_source": PRICING_SOURCE,
            "pricing_as_of": PRICING_AS_OF,
            "estimate_note": (
                "Published OpenAI standard rates; Azure/provider invoice pricing and "
                "per-request long-context multipliers are not exposed by aggregate "
                "Codex usage events."
            ),
        },
        "tools": {
            "tool_call_count": int(tool_overall["tool_call_count"]),
            "duration_count": int(tool_overall["duration_count"]),
            "total_observed_duration_s": round(
                int(tool_overall["observed_duration_us"]) / 1_000_000,
                6,
            ),
            "by_type": {
                item_type: {
                    "count": int(bucket["count"]),
                    "completed_count": int(bucket["completed_count"]),
                    "duration_count": int(bucket["duration_count"]),
                    "total_observed_duration_s": round(
                        int(bucket["observed_duration_us"]) / 1_000_000,
                        6,
                    ),
                }
                for item_type, bucket in sorted(tool_by_type.items())
            },
        },
    }
