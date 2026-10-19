"""Pier trial metrics and accounting used by the ProgramBench evaluator."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
import tomllib
from typing import Any, Iterable

from rho.pricing import estimate_usage_cost


INFRA_TEXT_MARKERS = (
    "cannot connect to the docker daemon",
    "docker compose command failed",
    "error response from daemon",
    "manifest unknown",
    "no space left on device",
    "temporary failure in name resolution",
    "could not resolve host",
    "failed to lookup address information",
    "network is unreachable",
    "connection timed out",
    "connection reset",
    "connection refused",
    "stream disconnected before completion",
    "response.failed event received",
    '"subtype": "api_retry"',
    "api error",
    "authentication_failed",
    "failed to authenticate",
    "requested url could not be retrieved",
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
    "tls handshake timeout",
    "ssl certificate problem",
    "unexpected eof",
    "remote end hung up",
)


RATE_LIMIT_RETRY_REASONS = {
    "http_429",
    "response_rate_limit",
    "response_rate_limit_exceeded",
}


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def seconds_between(start: Any, finish: Any) -> float | None:
    if not isinstance(start, str) or not isinstance(finish, str):
        return None
    try:
        return max(
            0.0,
            (
                datetime.fromisoformat(finish.replace("Z", "+00:00"))
                - datetime.fromisoformat(start.replace("Z", "+00:00"))
            ).total_seconds(),
        )
    except ValueError:
        return None


def trial_result_path(job_dir: Path) -> Path | None:
    candidates = sorted(
        path
        for path in job_dir.glob("*/result.json")
        if path.parent != job_dir and not path.parent.name.startswith(".")
    )
    return candidates[0] if len(candidates) == 1 else None


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return events
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def codex_metrics(agent_dir: Path) -> dict[str, Any]:
    events = _read_json_lines(agent_dir / "codex.txt")
    completed_turns = [event for event in events if event.get("type") == "turn.completed"]
    usage = None
    api_calls = None
    if completed_turns:
        final = completed_turns[-1]
        raw_usage = final.get("usage")
        if isinstance(raw_usage, dict):
            input_tokens = int(raw_usage.get("input_tokens") or 0)
            cached = min(input_tokens, int(raw_usage.get("cached_input_tokens") or 0))
            usage = {
                "input_tokens": input_tokens,
                "cached_input_tokens": cached,
                "cache_write_input_tokens": 0,
                "uncached_input_tokens": input_tokens - cached,
                "output_tokens": int(raw_usage.get("output_tokens") or 0),
            }
        raw_api_calls = final.get("api_calls")
        if isinstance(raw_api_calls, int):
            api_calls = raw_api_calls

    tool_types = {
        "command_execution",
        "file_change",
        "mcp_tool_call",
        "web_search",
    }
    completed_tools = [
        event
        for event in events
        if event.get("type") == "item.completed"
        and isinstance(event.get("item"), dict)
        and event["item"].get("type") in tool_types
    ]
    return {
        "usage": usage,
        "api_requests": api_calls,
        "tool_calls": len(completed_tools),
        "tool_counts": {
            tool_type: sum(
                1 for event in completed_tools if event["item"].get("type") == tool_type
            )
            for tool_type in sorted(tool_types)
            if any(event["item"].get("type") == tool_type for event in completed_tools)
        },
        "event_count": len(events),
        "completed_turns": len(completed_turns),
    }


def _anthropic_proxy_metrics(
    proxy_log: Path | None,
    usage_log: Path | None,
    trace_id: str,
    model: str | None,
) -> dict[str, Any] | None:
    usage_records = [
        record
        for record in (
            _read_json_lines(usage_log) if usage_log is not None else []
        )
        if record.get("trace_id") == trace_id
        and isinstance(record.get("usage"), dict)
        and 200 <= int(record.get("status_code") or 0) < 300
    ]
    attempt_records = [
        record
        for record in (
            _read_json_lines(proxy_log) if proxy_log is not None else []
        )
        if record.get("trace_id") == trace_id
        and record.get("event") == "anthropic_upstream_attempt"
    ]
    if not usage_records and not attempt_records:
        return None

    usage = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "uncached_input_tokens": 0,
        "output_tokens": 0,
    }
    time_aware_cost = 0.0
    time_aware_priced = 0
    by_pricing_tier: dict[str, float] = {}
    for record in usage_records:
        raw = record["usage"]
        uncached = max(0, int(raw.get("input_tokens") or 0))
        cached = max(0, int(raw.get("cache_read_input_tokens") or 0))
        cache_write = max(0, int(raw.get("cache_creation_input_tokens") or 0))
        output = max(0, int(raw.get("output_tokens") or 0))
        normalized = {
            "input_tokens": uncached + cached + cache_write,
            "cached_input_tokens": cached,
            "cache_write_input_tokens": cache_write,
            "uncached_input_tokens": uncached + cache_write,
            "output_tokens": output,
        }
        for key in usage:
            usage[key] += normalized[key]
        priced = estimate_usage_cost(
            str(record.get("model") or model or ""),
            normalized,
            observed_at=record.get("observed_at_utc"),
        )
        if priced is not None:
            value = float(priced["estimated_total_usd"])
            time_aware_cost += value
            time_aware_priced += 1
            tier = str(priced.get("pricing_tier") or "standard")
            by_pricing_tier[tier] = by_pricing_tier.get(tier, 0.0) + value

    successful_attempts = sum(
        record.get("retry_reason") is None and int(record.get("status") or 0) < 400
        for record in attempt_records
    )
    return {
        "logical_api_requests": len(usage_records),
        "usage_observed_requests": len(usage_records),
        "usage_complete": bool(usage_records)
        and (not attempt_records or len(usage_records) == successful_attempts),
        "upstream_attempts": len(attempt_records) or len(usage_records),
        "retry_count": max(0, len(attempt_records) - successful_attempts),
        "transport_wait_s": round(
            sum(float(record.get("elapsed_s") or 0.0) for record in attempt_records),
            6,
        ),
        "pacer_wait_s": round(
            sum(float(record.get("pacer_wait_s") or 0.0) for record in attempt_records),
            6,
        ),
        "shared_rate_limit_wait_s": round(
            sum(
                float(record.get("shared_rate_limit_wait_s") or 0.0)
                for record in attempt_records
            ),
            6,
        ),
        "retry_backoff_s": round(
            sum(float(record.get("retry_delay_s") or 0.0) for record in attempt_records),
            6,
        ),
        "usage": usage,
        "provider_time_aware_cost": {
            "priced_request_count": time_aware_priced,
            "estimated_total_usd": round(time_aware_cost, 9),
            "by_pricing_tier_usd": {
                key: round(value, 9) for key, value in sorted(by_pricing_tier.items())
            },
            "complete": time_aware_priced == len(usage_records),
        },
    }


def proxy_metrics(
    proxy_log: Path | None,
    trace_id: str,
    *,
    usage_log: Path | None = None,
    model: str | None = None,
) -> dict[str, Any] | None:
    records = [
        record
        for record in (
            _read_json_lines(proxy_log) if proxy_log is not None else []
        )
        if record.get("transport_trace_id") == trace_id
        and record.get("method") == "POST"
        and str(record.get("path") or "").rstrip("/").endswith("/responses")
    ]
    if not records:
        return _anthropic_proxy_metrics(proxy_log, usage_log, trace_id, model)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        request_id = record.get("request_id")
        key = str(request_id) if request_id is not None else f"missing-{index}"
        grouped.setdefault(key, []).append(record)
    final_records = [values[-1] for values in grouped.values()]
    usage_records = [
        record
        for record in final_records
        if all(
            record.get(key) is not None
            for key in ("input_tokens", "cached_input_tokens", "output_tokens")
        )
    ]
    usage = {
        "input_tokens": sum(
            int(record.get("input_tokens") or 0) for record in usage_records
        ),
        "cached_input_tokens": sum(
            int(record.get("cached_input_tokens") or 0) for record in usage_records
        ),
        "cache_write_input_tokens": 0,
        "uncached_input_tokens": sum(
            int(record.get("uncached_input_tokens") or 0) for record in usage_records
        ),
        "output_tokens": sum(
            int(record.get("output_tokens") or 0) for record in usage_records
        ),
    }
    return {
        "logical_api_requests": len(grouped),
        "usage_observed_requests": len(usage_records),
        "usage_complete": len(usage_records) == len(grouped),
        "upstream_attempts": sum(
            max(max(1, int(record.get("attempt") or 1)) for record in values)
            for values in grouped.values()
        ),
        "retry_count": sum(
            max(
                0,
                max(int(record.get("attempt") or 1) for record in values) - 1,
            )
            for values in grouped.values()
        ),
        "transport_wait_s": round(
            sum(float(record.get("elapsed_s") or 0.0) for record in records), 6
        ),
        "pacer_wait_s": round(
            sum(float(record.get("pacer_wait_s") or 0.0) for record in records), 6
        ),
        "shared_rate_limit_wait_s": round(
            sum(
                float(record.get("shared_rate_limit_wait_s") or 0.0)
                for record in records
            ),
            6,
        ),
        "retry_backoff_s": round(
            sum(
                float(record.get("retry_delay_s") or 0.0)
                for record in records
                if record.get("retry_reason") not in RATE_LIMIT_RETRY_REASONS
            ),
            6,
        ),
        "usage": usage,
    }


def _trial_timing(trial: dict[str, Any]) -> dict[str, Any]:
    def phase(name: str) -> float | None:
        value = trial.get(name)
        if not isinstance(value, dict):
            return None
        return seconds_between(value.get("started_at"), value.get("finished_at"))

    return {
        "environment_setup_s": phase("environment_setup"),
        "agent_setup_s": phase("agent_setup"),
        "agent_execution_s": phase("agent_execution"),
        "verifier_s": phase("verifier"),
        "total_s": seconds_between(trial.get("started_at"), trial.get("finished_at")),
    }


def _nominal_agent_timeout_s(trial: dict[str, Any]) -> float | None:
    """Read the benchmark's unmultiplied agent budget from its task metadata."""

    task = ((trial.get("config") or {}).get("task") or {})
    task_path = task.get("path")
    if isinstance(task_path, str):
        try:
            config = tomllib.loads((Path(task_path) / "task.toml").read_text())
            value = ((config.get("agent") or {}).get("timeout_sec"))
            if value is not None:
                return float(value)
        except (OSError, tomllib.TOMLDecodeError, TypeError, ValueError):
            pass

    message = str(((trial.get("exception_info") or {}).get("exception_message") or ""))
    match = re.search(r"timed out after ([0-9]+(?:\.[0-9]+)?) seconds", message)
    return float(match.group(1)) if match else None


def _timeout_adjustment(
    trial: dict[str, Any],
    timing: dict[str, Any],
    transport: dict[str, Any] | None,
) -> dict[str, Any]:
    """Separate solver work from provider/pacer waiting.

    Pier's watchdog is wall-clock based.  The evaluation deliberately permits a
    larger outer watchdog so transient Azure latency does not consume the
    benchmark's solver budget.  Formal scoring still enforces the task's
    original, unmultiplied timeout against wall time minus observed API waits.
    """

    wall = timing.get("agent_execution_s")
    transport_wait = float((transport or {}).get("transport_wait_s") or 0.0)
    pacer_wait = float((transport or {}).get("pacer_wait_s") or 0.0)
    rate_limit_wait = float(
        (transport or {}).get("shared_rate_limit_wait_s") or 0.0
    )
    retry_backoff = float((transport or {}).get("retry_backoff_s") or 0.0)
    excluded_wait = transport_wait + pacer_wait + rate_limit_wait + retry_backoff
    effective = (
        max(0.0, float(wall) - excluded_wait) if wall is not None else None
    )
    config = trial.get("config") or {}
    multiplier = config.get("agent_timeout_multiplier")
    return {
        "nominal_agent_timeout_s": _nominal_agent_timeout_s(trial),
        "outer_timeout_multiplier": float(multiplier) if multiplier is not None else 1.0,
        "transport_wait_s": round(transport_wait, 6),
        "pacer_wait_s": round(pacer_wait, 6),
        "shared_rate_limit_wait_s": round(rate_limit_wait, 6),
        "retry_backoff_s": round(retry_backoff, 6),
        "excluded_provider_wait_s": round(excluded_wait, 6),
        "effective_agent_execution_s": round(effective, 6) if effective is not None else None,
    }


def aggregate_attempts(attempts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    attempts = list(attempts)
    usage = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "uncached_input_tokens": 0,
        "output_tokens": 0,
    }
    cost = 0.0
    priced = 0
    provider_time_aware_cost = 0.0
    provider_time_aware_priced = 0
    provider_time_aware_complete = True
    provider_cost_by_tier: dict[str, float] = {}
    api_requests = 0
    upstream_attempts = 0
    retry_count = 0
    transport_wait = 0.0
    pacer_wait = 0.0
    shared_rate_limit_wait = 0.0
    retry_backoff = 0.0
    tool_calls = 0
    tool_counts: dict[str, int] = {}
    tool_duration = 0.0
    tool_duration_count = 0
    environment_setup = agent_setup = 0.0
    agent_wall = effective_agent = excluded_provider_wait = 0.0
    verifier_wall = total_wall = 0.0
    for attempt in attempts:
        raw_usage = attempt.get("usage")
        if isinstance(raw_usage, dict):
            for key in usage:
                usage[key] += int(raw_usage.get(key) or 0)
        raw_cost = attempt.get("cost")
        if isinstance(raw_cost, dict) and raw_cost.get("estimated_total_usd") is not None:
            cost += float(raw_cost["estimated_total_usd"])
            priced += 1
        provider_cost = attempt.get("provider_time_aware_cost")
        if isinstance(provider_cost, dict):
            provider_time_aware_cost += float(
                provider_cost.get("estimated_total_usd") or 0.0
            )
            provider_time_aware_priced += int(
                provider_cost.get("priced_request_count") or 0
            )
            provider_time_aware_complete = provider_time_aware_complete and bool(
                provider_cost.get("complete")
            )
            for tier, value in (
                provider_cost.get("by_pricing_tier_usd") or {}
            ).items():
                provider_cost_by_tier[str(tier)] = (
                    provider_cost_by_tier.get(str(tier), 0.0) + float(value or 0.0)
                )
        api = attempt.get("api") or {}
        api_requests += int(api.get("logical_api_requests") or 0)
        upstream_attempts += int(api.get("upstream_attempts") or 0)
        retry_count += int(api.get("retry_count") or 0)
        transport_wait += float(api.get("transport_wait_s") or 0.0)
        pacer_wait += float(api.get("pacer_wait_s") or 0.0)
        shared_rate_limit_wait += float(
            api.get("shared_rate_limit_wait_s") or 0.0
        )
        retry_backoff += float(api.get("retry_backoff_s") or 0.0)
        tools = attempt.get("tools") or {}
        tool_calls += int(tools.get("count") or 0)
        for tool_type, count in (tools.get("by_type") or {}).items():
            tool_counts[str(tool_type)] = tool_counts.get(str(tool_type), 0) + int(
                count or 0
            )
        tool_duration += float(tools.get("observed_duration_s") or 0.0)
        tool_duration_count += int(tools.get("duration_count") or 0)
        timing = attempt.get("timing") or {}
        environment_setup += float(timing.get("environment_setup_s") or 0.0)
        agent_setup += float(timing.get("agent_setup_s") or 0.0)
        agent_wall += float(timing.get("agent_execution_s") or 0.0)
        effective_agent += float(timing.get("effective_agent_execution_s") or 0.0)
        excluded_provider_wait += float(
            timing.get("excluded_provider_wait_s") or 0.0
        )
        verifier_wall += float(timing.get("verifier_s") or 0.0)
        total_wall += float(timing.get("total_s") or 0.0)
    return {
        "attempt_count": len(attempts),
        "usage_observed_count": sum(
            isinstance(item.get("usage"), dict) for item in attempts
        ),
        "usage": usage,
        "cost": {
            "priced_attempt_count": priced,
            "estimated_total_usd": round(cost, 9),
            "complete": priced == len(attempts),
        },
        "provider_time_aware_cost": {
            "priced_request_count": provider_time_aware_priced,
            "estimated_total_usd": round(provider_time_aware_cost, 9),
            "by_pricing_tier_usd": {
                key: round(value, 9)
                for key, value in sorted(provider_cost_by_tier.items())
            },
            "complete": provider_time_aware_complete
            and provider_time_aware_priced > 0,
        },
        "api": {
            "logical_requests": api_requests,
            "upstream_attempts": upstream_attempts,
            "retry_count": retry_count,
            "transport_wait_s": round(transport_wait, 6),
            "pacer_wait_s": round(pacer_wait, 6),
            "shared_rate_limit_wait_s": round(shared_rate_limit_wait, 6),
            "retry_backoff_s": round(retry_backoff, 6),
        },
        "tools": {
            "count": tool_calls,
            "by_type": dict(sorted(tool_counts.items())),
            "observed_duration_s": round(tool_duration, 6),
            "duration_count": tool_duration_count,
        },
        "timing": {
            "environment_setup_s": round(environment_setup, 6),
            "agent_setup_s": round(agent_setup, 6),
            "agent_execution_s": round(agent_wall, 6),
            "effective_agent_execution_s": round(effective_agent, 6),
            "excluded_provider_wait_s": round(excluded_provider_wait, 6),
            "verifier_s": round(verifier_wall, 6),
            "sum_trial_wall_s": round(total_wall, 6),
        },
    }
