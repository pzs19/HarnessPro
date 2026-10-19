from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from openai import APITimeoutError, OpenAI

from rho.protocols import Trajectory, TrajectoryKind


def _usage_value(value: Any, name: str) -> int:
    if value is None:
        return 0
    raw = value.get(name) if isinstance(value, dict) else getattr(value, name, None)
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError):
        return 0


class ResponsesAgent:
    """Single-call, tool-free Responses API agent for control-plane work."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        reasoning_effort: str,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        api_key: str | None = None,
        default_timeout_s: float = 120.0,
        max_output_tokens: int = 1024,
        client: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.prompt_cache_key = prompt_cache_key
        self.prompt_cache_retention = prompt_cache_retention
        self.default_timeout_s = default_timeout_s
        self.max_output_tokens = max_output_tokens
        if client is not None:
            self._client = client
        else:
            resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
            if not resolved_key:
                raise RuntimeError(
                    "ResponsesAgent requires OPENAI_API_KEY in the process environment"
                )
            self._client = OpenAI(base_url=self.base_url, api_key=resolved_key)

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
        started = time.monotonic()
        response: Any | None = None
        error = ""
        timed_out = False
        try:
            text_config: dict[str, Any] | None = None
            if output_schema is not None:
                text_config = {
                    "format": {
                        "type": "json_schema",
                        "name": "structured_output",
                        "schema": output_schema,
                        "strict": True,
                    }
                }
            request: dict[str, Any] = {
                "model": self.model,
                "input": instructions,
                "reasoning": {"effort": self.reasoning_effort},
                "text": text_config,
                "store": False,
                "max_output_tokens": self.max_output_tokens,
                "timeout": effective_timeout,
            }
            if self.prompt_cache_key:
                request["prompt_cache_key"] = self.prompt_cache_key
            if self.prompt_cache_retention:
                request["prompt_cache_retention"] = self.prompt_cache_retention
            response = self._client.responses.create(**request)
        except APITimeoutError as exc:
            error = f"{type(exc).__name__}: {exc}"
            timed_out = True
        except Exception as exc:  # noqa: BLE001 - surfaced as a failed trajectory
            error = f"{type(exc).__name__}: {exc}"
        wall = time.monotonic() - started

        events: list[dict[str, Any]] = []
        final_message = ""
        stdout = ""
        if response is not None:
            final_message = str(getattr(response, "output_text", "") or "")
            usage = getattr(response, "usage", None)
            input_details = (
                usage.get("input_tokens_details")
                if isinstance(usage, dict)
                else getattr(usage, "input_tokens_details", None)
            )
            cached_input = _usage_value(input_details, "cached_tokens")
            cache_write_input = _usage_value(input_details, "cache_write_tokens")
            input_tokens = _usage_value(usage, "input_tokens")
            output_tokens = _usage_value(usage, "output_tokens")
            cached_input = min(input_tokens, cached_input)
            events.append(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": input_tokens,
                        "cached_input_tokens": cached_input,
                        "cache_write_input_tokens": min(
                            input_tokens - cached_input,
                            cache_write_input,
                        ),
                        "output_tokens": output_tokens,
                    },
                    "api_calls": 1,
                    "usage_model": self.model,
                    "usage_source": "responses-control-plane",
                }
            )
            try:
                stdout = response.model_dump_json()
            except AttributeError:
                stdout = json.dumps(
                    {"output_text": final_message}, ensure_ascii=False
                )

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
            exit_code=0 if response is not None else 1,
            wall_time_s=wall,
            timed_out=timed_out,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
        )
