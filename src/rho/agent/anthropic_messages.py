from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from rho.protocols import Trajectory, TrajectoryKind


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


class AnthropicMessagesAgent:
    """Single-call, tool-free Anthropic Messages control-plane agent."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        reasoning_effort: str,
        default_timeout_s: float = 120.0,
        max_output_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 0.95,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Anthropic Messages agent requires an API key")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.default_timeout_s = default_timeout_s
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._client = client or httpx.Client(trust_env=False)

    def _messages_url(self) -> str:
        if self.base_url.endswith("/v1"):
            return self.base_url + "/messages"
        return self.base_url + "/v1/messages"

    def run(
        self,
        workspace: Path,
        instructions: str,
        *,
        output_schema: dict[str, Any] | None = None,
        task_id: str = "",
        harness_id: str = "",
        kind: TrajectoryKind = "evaluate",
        timeout_s: float | None = None,
        env: dict[str, str] | None = None,
    ) -> Trajectory:
        del workspace, env
        effective_timeout = timeout_s or self.default_timeout_s
        prompt = instructions
        if output_schema is not None:
            prompt = (
                instructions.rstrip()
                + "\n\nReturn exactly one JSON object and no markdown. It must match "
                "this JSON Schema:\n"
                + json.dumps(output_schema, ensure_ascii=False, sort_keys=True)
            )
        request = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "thinking": {
                "type": "disabled"
                if self.reasoning_effort == "low"
                else "adaptive"
            },
            "output_config": {"effort": self.reasoning_effort},
            "stream": True,
            "messages": [{"role": "user", "content": prompt}],
        }
        started = time.monotonic()
        response_payload: dict[str, Any] | None = None
        error = ""
        timed_out = False
        try:
            text_parts: list[str] = []
            usage: dict[str, int] = {}
            with self._client.stream(
                "POST",
                self._messages_url(),
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=request,
                timeout=effective_timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    event = json.loads(raw)
                    if not isinstance(event, dict):
                        continue
                    if event.get("type") == "message_start":
                        message = event.get("message")
                        event_usage = (
                            message.get("usage")
                            if isinstance(message, dict)
                            else None
                        )
                    else:
                        event_usage = event.get("usage")
                    if isinstance(event_usage, dict):
                        for key in (
                            "input_tokens",
                            "cache_read_input_tokens",
                            "cache_creation_input_tokens",
                            "output_tokens",
                        ):
                            if key in event_usage:
                                usage[key] = _nonnegative_int(event_usage[key])
                    if event.get("type") == "content_block_start":
                        block = event.get("content_block")
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(str(block.get("text") or ""))
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta")
                        if isinstance(delta, dict) and delta.get("type") == "text_delta":
                            text_parts.append(str(delta.get("text") or ""))
            response_payload = {"text": "".join(text_parts).strip(), "usage": usage}
        except httpx.TimeoutException as exc:
            error = f"{type(exc).__name__}: {exc}"
            timed_out = True
        except Exception as exc:  # noqa: BLE001 - converted to failed trajectory
            error = f"{type(exc).__name__}: {exc}"
        wall = time.monotonic() - started

        events: list[dict[str, Any]] = []
        final_message = ""
        stdout = ""
        if response_payload is not None:
            final_message = str(response_payload.get("text") or "").strip()
            usage = response_payload.get("usage")
            usage = usage if isinstance(usage, dict) else {}
            uncached_input = _nonnegative_int(usage.get("input_tokens"))
            cached_input = _nonnegative_int(
                usage.get("cache_read_input_tokens")
            )
            cache_write_input = _nonnegative_int(
                usage.get("cache_creation_input_tokens")
            )
            events.append(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": uncached_input
                        + cached_input
                        + cache_write_input,
                        "cached_input_tokens": cached_input,
                        "cache_write_input_tokens": cache_write_input,
                        "output_tokens": _nonnegative_int(
                            usage.get("output_tokens")
                        ),
                    },
                    "api_calls": 1,
                    "usage_model": self.model,
                    "usage_source": "anthropic-messages-control-plane",
                }
            )
            stdout = json.dumps(response_payload, ensure_ascii=False)

        return Trajectory(
            id=f"traj_{uuid.uuid4().hex[:10]}",
            kind=kind,
            task_id=task_id,
            harness_id=harness_id,
            instructions=instructions,
            events=events,
            final_message=final_message,
            stdout=stdout,
            stderr=error,
            workspace_diff={},
            workspace_deletions=frozenset(),
            exit_code=0 if response_payload is not None else 1,
            wall_time_s=wall,
            timed_out=timed_out,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
        )
