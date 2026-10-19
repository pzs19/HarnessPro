from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
import uuid
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from rho.agent.codex import (
    DEFAULT_TIMEOUT_S,
    _diff,
    _read_transport_compensation,
    _snapshot,
    _transport_compensation_config,
)
from rho.protocols import Trajectory, TrajectoryKind


_CLAUDE_USAGE_KEYS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
)


def _subprocess_env(extra_env: dict[str, str] | None) -> dict[str, str]:
    env = os.environ.copy()
    if extra_env:
        env.update({key: str(value) for key, value in extra_env.items()})
    return env


def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        proc.kill()


def _run_process(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_s: float,
) -> tuple[str, str, int, bool, float]:
    started = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_process_group(proc)
        stdout, stderr = proc.communicate()
        exit_code = -1
    return stdout, stderr, int(exit_code), timed_out, time.monotonic() - started


def _run_process_transport_aware(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_s: float,
) -> tuple[str, str, int, bool, float, float]:
    """Run a process while excluding audited provider wait from its deadline."""

    started = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def read_stdout() -> None:
        stdout_lines.extend(proc.stdout)

    def read_stderr() -> None:
        stderr_lines.extend(proc.stderr)

    stdout_thread = threading.Thread(target=read_stdout, daemon=True)
    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    transport_file, maximum_compensation_s = _transport_compensation_config(env)
    compensation_s = 0.0
    timed_out = False
    try:
        base_deadline = started + timeout_s
        while True:
            if transport_file is not None and maximum_compensation_s > 0:
                compensation_s = _read_transport_compensation(
                    transport_file, maximum_compensation_s
                )
            remaining_s = base_deadline + compensation_s - time.monotonic()
            if remaining_s <= 0:
                raise subprocess.TimeoutExpired(cmd, timeout_s + compensation_s)
            try:
                exit_code = proc.wait(timeout=min(1.0, remaining_s))
                break
            except subprocess.TimeoutExpired:
                continue
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_process_group(proc)
        proc.wait()
        exit_code = -1
    finally:
        stdout_thread.join(timeout=30)
        stderr_thread.join(timeout=30)
        if stdout_thread.is_alive():
            proc.stdout.close()
            stdout_thread.join(timeout=1)
        if stderr_thread.is_alive():
            proc.stderr.close()
            stderr_thread.join(timeout=1)
    if transport_file is not None and maximum_compensation_s > 0:
        compensation_s = _read_transport_compensation(
            transport_file, maximum_compensation_s
        )
    return (
        "".join(stdout_lines),
        "".join(stderr_lines),
        int(exit_code),
        timed_out,
        time.monotonic() - started,
        compensation_s,
    )


def _usage_event(
    *,
    uncached_input: int,
    cached_input: int,
    cache_write_input: int,
    output: int,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "type": "turn.completed",
        "usage": {
            "input_tokens": max(0, uncached_input)
            + max(0, cached_input)
            + max(0, cache_write_input),
            "cached_input_tokens": max(0, cached_input),
            "cache_write_input_tokens": max(0, cache_write_input),
            "output_tokens": max(0, output),
        },
        **extra,
    }


def _tool_events(
    calls: list[dict[str, Any]],
    *,
    session_started_at_s: float | None,
    expected_count: int | None = None,
    preserve_missing_timestamps: bool = False,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for index, call in enumerate(calls):
        item_id = str(call.get("id") or f"external_tool_{index}")
        item_type = str(call.get("name") or "external_tool")
        started_at = call.get("started_at_s")
        completed_at = call.get("completed_at_s")
        if session_started_at_s is not None:
            if isinstance(started_at, (int, float)):
                started_at = max(0.0, float(started_at) - session_started_at_s)
            if isinstance(completed_at, (int, float)):
                completed_at = max(0.0, float(completed_at) - session_started_at_s)
        item = {"id": item_id, "type": item_type, "status": "in_progress"}
        events.append(
            {
                "type": "item.started",
                "item": item,
                "_rho_observed_at_s": float(started_at or 0.0),
            }
        )
        if preserve_missing_timestamps and not isinstance(started_at, (int, float)):
            events[-1].pop("_rho_observed_at_s", None)
        completed_item = {
            "id": item_id,
            "type": item_type,
            "status": "failed" if call.get("failed") else "completed",
        }
        events.append(
            {
                "type": "item.completed",
                "item": completed_item,
                "_rho_observed_at_s": float(
                    completed_at if isinstance(completed_at, (int, float)) else started_at or 0.0
                ),
            }
        )
        if preserve_missing_timestamps and not isinstance(completed_at, (int, float)):
            events[-1].pop("_rho_observed_at_s", None)

    missing = max(0, int(expected_count or 0) - len(calls))
    for offset in range(missing):
        item_id = f"external_tool_unobserved_{offset}"
        events.append(
            {
                "type": "item.completed",
                "item": {
                    "id": item_id,
                    "type": "external_tool_unobserved",
                    "status": "completed",
                },
            }
        )
    return events


class ClaudeCodeAgent:
    """Run the official Claude Code CLI natively or in an isolated container.

    Claude Code speaks Anthropic Messages to a Claude Code Router gateway.  The
    gateway performs the mature protocol conversion to the experiment's OpenAI
    Responses endpoint; this adapter only launches the agent loop and translates
    its documented stream-json events into RHO metrics.  Native mode is useful
    for repository benchmarks whose Solver must share the host's audited Git and
    language toolchains.  Container mode remains the default for Terminal-Bench.
    """

    def __init__(
        self,
        *,
        docker_command: Path | None = None,
        docker_client: Path | None = None,
        docker_socket: Path | None = None,
        binary: Path | None = None,
        image: str = "harnesspro-claude-code:2.1.231",
        ccr_base_url: str | None = None,
        ccr_metrics_base_url: str | None = None,
        ccr_api_key: str | None = None,
        model_alias: str = "sonnet",
        model: str = "gpt-5.5",
        reasoning_effort: str = "high",
        inject_harness_context: bool = False,
        runtime_hook_script: Path | None = None,
        rtk_binary: Path | None = None,
        system_prompt_mode: str = "append",
        include_runtime_context_in_system_prompt: bool = False,
        enable_completion_stop_hook: bool = False,
        phase_route_profile_key: bool = False,
        inline_task_prompt: bool = False,
        autocompact: str | None = None,
        tool_output_cache_root: Path | None = None,
        tools: str = "Bash,Edit,Read",
        trace_usage: bool = False,
        default_timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.binary = Path(binary).resolve() if binary is not None else None
        self.docker_command = (
            Path(docker_command).resolve() if docker_command is not None else None
        )
        self.docker_client = (
            Path(docker_client).resolve() if docker_client is not None else None
        )
        self.docker_socket = (
            Path(docker_socket).resolve() if docker_socket is not None else None
        )
        if self.binary is not None:
            if not self.binary.is_file():
                raise FileNotFoundError(
                    f"Claude Code binary not found: {self.binary}"
                )
        else:
            for label, path in (
                ("Docker command", self.docker_command),
                ("Docker client", self.docker_client),
                ("Docker socket", self.docker_socket),
            ):
                if path is None or not path.exists():
                    raise FileNotFoundError(f"{label} not found: {path}")
        self.image = image
        self.ccr_base_url = (
            ccr_base_url or os.environ.get("HARNESSPRO_CLAUDE_BASE_URL", "")
        ).rstrip("/")
        if not self.ccr_base_url:
            raise ValueError("Provide ccr_base_url or HARNESSPRO_CLAUDE_BASE_URL")
        self.ccr_metrics_base_url = (
            ccr_metrics_base_url.rstrip("/")
            if ccr_metrics_base_url is not None
            else None
        )
        self.ccr_api_key = ccr_api_key or os.environ.get("HARNESSPRO_CLAUDE_API_KEY", "")
        if not self.ccr_api_key:
            raise ValueError("Provide ccr_api_key or HARNESSPRO_CLAUDE_API_KEY")
        self.model_alias = model_alias
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.inject_harness_context = inject_harness_context
        self.runtime_hook_script = (
            Path(runtime_hook_script).resolve()
            if runtime_hook_script is not None
            else None
        )
        self.rtk_binary = Path(rtk_binary).resolve() if rtk_binary is not None else None
        if system_prompt_mode not in {"append", "replace"}:
            raise ValueError(
                "Claude Code system_prompt_mode must be 'append' or 'replace'"
            )
        self.system_prompt_mode = system_prompt_mode
        self.include_runtime_context_in_system_prompt = (
            include_runtime_context_in_system_prompt
        )
        self.enable_completion_stop_hook = enable_completion_stop_hook
        self.phase_route_profile_key = phase_route_profile_key
        self.inline_task_prompt = inline_task_prompt
        self.autocompact = autocompact
        self.tool_output_cache_root = Path(tool_output_cache_root).resolve() if tool_output_cache_root is not None else None
        self.tools = tools.strip()
        if not self.tools:
            raise ValueError("Claude Code tools must not be empty")
        self.trace_usage = trace_usage
        if self.trace_usage and self.ccr_metrics_base_url is None:
            raise ValueError(
                "Claude Code traced usage requires ccr_metrics_base_url"
            )
        for label, path in (
            ("runtime hook", self.runtime_hook_script),
            ("RTK binary", self.rtk_binary),
        ):
            if path is not None and not path.is_file():
                raise FileNotFoundError(f"Claude Code {label} not found: {path}")
        if self.rtk_binary is not None and not os.access(self.rtk_binary, os.X_OK):
            raise PermissionError(
                f"RTK is not executable: {self.rtk_binary}. "
                "Run python scripts/prepare_runtime.py."
            )
        self.default_timeout_s = default_timeout_s
        self._version: str | None = None
        self._version_lock = threading.Lock()

    @property
    def cache_env_signature(self) -> dict[str, str]:
        return {
            "agent_class": "ClaudeCodeAgent",
            "execution_mode": "native" if self.binary is not None else "container",
            "binary": str(self.binary or ""),
            "docker_command": str(self.docker_command or ""),
            "docker_client": str(self.docker_client or ""),
            "docker_socket": str(self.docker_socket or ""),
            "image": self.image,
            "binary_version": self.binary_version(),
            "ccr_base_url": self.ccr_base_url,
            "ccr_metrics_base_url": str(self.ccr_metrics_base_url or ""),
            "model_alias": self.model_alias,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "inject_harness_context": str(self.inject_harness_context),
            "runtime_hook_script": str(self.runtime_hook_script or ""),
            "rtk_binary": str(self.rtk_binary or ""),
            "system_prompt_mode": self.system_prompt_mode,
            "include_runtime_context_in_system_prompt": str(
                self.include_runtime_context_in_system_prompt
            ),
            "enable_completion_stop_hook": str(self.enable_completion_stop_hook),
            "phase_route_profile_key": str(self.phase_route_profile_key),
            "inline_task_prompt": str(self.inline_task_prompt),
            "autocompact": str(self.autocompact or ""),
            "tool_output_cache_root": str(self.tool_output_cache_root or ""),
            "tools": self.tools,
            "trace_usage": str(self.trace_usage),
        }

    def binary_version(self) -> str:
        if self._version is not None:
            return self._version
        with self._version_lock:
            if self._version is None:
                cmd = (
                    [str(self.binary), "--version"]
                    if self.binary is not None
                    else [
                        str(self.docker_command),
                        "run",
                        "--rm",
                        "--entrypoint",
                        "/usr/local/bin/claude",
                        self.image,
                        "--version",
                    ]
                )
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=60,
                )
                self._version = (proc.stdout or proc.stderr).splitlines()[0].strip()
        return self._version or "unknown"

    def run(
        self,
        workspace: Path,
        instructions: str,
        *,
        output_schema: dict[str, Any] | None = None,
        task_id: str = "",
        harness_id: str = "",
        kind: TrajectoryKind = "solve",
        timeout_s: float | None = None,
        env: dict[str, str] | None = None,
    ) -> Trajectory:
        if output_schema is not None:
            raise ValueError("ClaudeCodeAgent does not support output_schema")
        workspace = Path(workspace).resolve()
        meta = workspace / ".rho"
        state_dir = meta / f"claude-code-{uuid.uuid4().hex[:10]}"
        state_dir.mkdir(parents=True, exist_ok=True)
        native_home = state_dir / "home"
        if self.binary is not None:
            native_home.mkdir(parents=True, exist_ok=True)
        _make_group_writable(workspace)
        before = _snapshot(workspace, exclude={".rho"})
        effective_timeout = timeout_s if timeout_s is not None else self.default_timeout_s
        container_name = (
            f"rho-claude-code-{uuid.uuid4().hex[:12]}"
            if self.binary is None
            else None
        )
        prompt = instructions
        use_inline_task_prompt = self.inline_task_prompt and (
            (env or {}).get("RHO_INLINE_TASK_PROMPT") != "0"
        )
        if use_inline_task_prompt:
            task_prompt_path = workspace / "task" / "prompt.md"
            try:
                task_prompt = task_prompt_path.read_text(encoding="utf-8").strip()
            except OSError:
                task_prompt = ""
            if task_prompt:
                prompt = (
                    "Complete the task below autonomously. Its public description is "
                    "included here, so do not spend a tool call rereading "
                    "task/prompt.md unless this copy is incomplete.\n\n"
                    "<task-description>\n"
                    f"{task_prompt}\n"
                    "</task-description>\n\n"
                    "Leave every requested artifact and runtime state at the exact "
                    "specified path, validate the result, and then answer briefly."
                )
        selection_path = workspace / ".rho" / "runtime-selection.json"
        runtime_hooks_enabled = self.runtime_hook_script is not None
        mcp_config_path = workspace / "harness" / "mcp.json"
        if mcp_config_path.exists():
            if not mcp_config_path.resolve().is_relative_to((workspace / "harness").resolve()):
                raise ValueError("harness MCP config must remain inside the harness")
            mcp_config = json.loads(mcp_config_path.read_text(encoding="utf-8"))
            servers = mcp_config.get("mcpServers") if isinstance(mcp_config, dict) else None
            if not isinstance(servers, dict) or not servers:
                raise ValueError("harness MCP config requires a nonempty mcpServers object")
            for server in servers.values():
                if (not isinstance(server, dict) or server.get("type", "stdio") != "stdio"
                        or not isinstance(server.get("command"), str) or not server["command"]
                        or not isinstance(server.get("args", []), list)
                        or any(not isinstance(argument, str) for argument in server.get("args", []))):
                    raise ValueError("harness MCP servers require explicit stdio commands and string arguments")
        if runtime_hooks_enabled and not selection_path.is_file():
            raise RuntimeError(
                "Claude Code runtime hooks require .rho/runtime-selection.json"
            )
        runtime_settings_path = state_dir / "runtime-settings.json"
        runtime_audit_path = state_dir / "runtime-audit.jsonl"
        hook_adapter_path = Path(__file__).with_name(
            "claude_code_hook_adapter.py"
        ).resolve()
        if runtime_hooks_enabled:
            _write_claude_code_runtime_settings(
                runtime_settings_path,
                include_stop=self.enable_completion_stop_hook,
                command=(
                    shlex.join(("/usr/bin/python3", str(hook_adapter_path)))
                    if self.binary is not None
                    else None
                ),
            )
        request_api_key = self.ccr_api_key
        if self.phase_route_profile_key and selection_path.is_file():
            selection = _read_json(selection_path)
            profile = re.sub(
                r"[^a-z0-9_-]",
                "",
                str(selection.get("profile") or "").lower(),
            )
            if profile:
                # The compatibility proxy routes reasoning effort through CCR
                # API-key profiles.  Older callers already supplied an
                # effort-qualified key (for example ``...-high``), while the
                # SWE-Pro dynamic runner intentionally supplies the common
                # base key and selects a different ClaudeCodeAgent per effort.
                # Preserve the former form and make the latter explicit before
                # adding the task-agnostic runtime profile suffix.
                if not re.search(r"-(?:low|medium|high)$", request_api_key):
                    request_api_key = (
                        f"{request_api_key}-{self.reasoning_effort}"
                    )
                request_api_key = f"{request_api_key}-{profile}"
        provided_trace_id = str((env or {}).get("RHO_TRANSPORT_TRACE_ID") or "")
        usage_trace_id = (
            provided_trace_id
            if self.trace_usage and re.fullmatch(r"[a-z0-9_-]{8,64}", provided_trace_id)
            else uuid.uuid4().hex[:20]
            if self.trace_usage
            else ""
        )
        if usage_trace_id:
            request_api_key = f"{request_api_key}--rho-{usage_trace_id}"
        if self.tool_output_cache_root is not None and (not usage_trace_id or self.binary is not None):
            raise ValueError("tool output cache requires traced container mode")
        harness_context = ""
        if self.inject_harness_context:
            harness_context = (
                _render_claude_runtime_model_instructions(
                    workspace,
                    include_runtime_context=(
                        self.include_runtime_context_in_system_prompt
                    ),
                )
                if runtime_hooks_enabled
                else _render_external_harness_context(workspace)
            )
        if self.binary is not None:
            cmd = (
                [
                    shutil.which("setpriv") or "setpriv",
                    "--reuid=10250",
                    "--regid=0",
                    "--clear-groups",
                    "--",
                    str(self.binary),
                ]
                if os.geteuid() == 0
                else [str(self.binary)]
            )
        else:
            assert container_name is not None
            max_output_tokens = str(
                (env or {}).get("CLAUDE_CODE_MAX_OUTPUT_TOKENS")
                or os.environ.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS")
                or ""
            ).strip()
            cmd = [
                str(self.docker_command),
                "run",
                "--rm",
                "--name",
                container_name,
                "--add-host",
                "host.docker.internal:host-gateway",
                "--group-add",
                str(workspace.stat().st_gid),
                "--group-add",
                str(self.docker_socket.stat().st_gid),
                "--volume",
                f"{workspace}:/workspace:rw",
                "--volume",
                f"{self.docker_client}:/usr/local/bin/docker:ro",
                "--volume",
                f"{self.docker_socket}:/var/run/docker.sock",
                "--workdir",
                "/workspace",
                "--env",
                f"ANTHROPIC_BASE_URL={self.ccr_base_url}",
                "--env",
                f"ANTHROPIC_API_KEY={request_api_key}",
                "--env",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1",
                "--env",
                "DISABLE_TELEMETRY=1",
                "--env",
                f"CLAUDE_CONFIG_DIR=/workspace/.rho/{state_dir.name}/config",
                "--env",
                "NO_PROXY=host.docker.internal,127.0.0.1,localhost",
                "--env",
                "no_proxy=host.docker.internal,127.0.0.1,localhost",
            ]
            if max_output_tokens:
                cmd.extend(
                    [
                        "--env",
                        f"CLAUDE_CODE_MAX_OUTPUT_TOKENS={max_output_tokens}",
                    ]
                )
            if self.tool_output_cache_root is not None:
                output_directory = self.tool_output_cache_root / usage_trace_id
                if output_directory.is_symlink():
                    raise ValueError("tool output trace directory must not be a symlink")
                output_directory.mkdir(parents=True, exist_ok=True)
                cmd.extend(["--volume", f"{output_directory}:/rho-tool-output:ro"])
        if runtime_hooks_enabled and self.binary is None:
            assert self.runtime_hook_script is not None
            cmd.extend(
                [
                    "--volume",
                    f"{self.runtime_hook_script}:/rho-harness/hook.py:ro",
                    "--volume",
                    f"{hook_adapter_path}:/rho-harness/claude_adapter.py:ro",
                    "--env",
                    "RHO_CLAUDE_CORE_HOOK=/rho-harness/hook.py",
                    "--env",
                    "RHO_RUNTIME_SELECTION_PATH=/workspace/.rho/runtime-selection.json",
                    "--env",
                    f"RHO_RUNTIME_STATE_ROOT=/workspace/.rho/{state_dir.name}/runtime-state",
                    "--env",
                    f"RHO_RUNTIME_AUDIT_PATH=/workspace/.rho/{state_dir.name}/runtime-audit.jsonl",
                    "--env",
                    "RHO_RUNTIME_INTERFACE=claude-code",
                    "--env",
                    f"RHO_AGENT_TIMEOUT_S={effective_timeout}",
                    "--env",
                    (
                        "RHO_CLAUDE_CONTEXT_IN_SYSTEM=1"
                        if self.include_runtime_context_in_system_prompt
                        else "RHO_CLAUDE_CONTEXT_IN_SYSTEM=0"
                    ),
                    "--env",
                    "RTK_TELEMETRY_DISABLED=1",
                ]
            )
            denial_halt_after = str(
                (env or {}).get("RHO_CLAUDE_POLICY_DENIAL_HALT_AFTER")
                or os.environ.get("RHO_CLAUDE_POLICY_DENIAL_HALT_AFTER")
                or ""
            ).strip()
            if denial_halt_after:
                cmd.extend(
                    [
                        "--env",
                        f"RHO_CLAUDE_POLICY_DENIAL_HALT_AFTER={denial_halt_after}",
                    ]
                )
            fresh_review = str(
                (env or {}).get("RHO_FRESH_REVIEW") or ""
            ).strip()
            fresh_review_max_tools = str(
                (env or {}).get("RHO_FRESH_REVIEW_MAX_TOOLS") or ""
            ).strip()
            if fresh_review:
                cmd.extend(["--env", f"RHO_FRESH_REVIEW={fresh_review}"])
            if fresh_review_max_tools:
                cmd.extend(
                    [
                        "--env",
                        f"RHO_FRESH_REVIEW_MAX_TOOLS={fresh_review_max_tools}",
                    ]
                )
            if self.rtk_binary is not None:
                cmd.extend(
                    [
                        "--volume",
                        f"{self.rtk_binary}:/rho-tools/rtk:ro",
                        "--env",
                        "RHO_RTK_BIN=/rho-tools/rtk",
                    ]
                )
            else:
                cmd.extend(["--env", "RHO_RTK_BIN="])
        if self.binary is None:
            cmd.append(self.image)
        cmd.extend(
            [
                "--print",
                "--output-format",
                "stream-json",
                "--verbose",
                "--model",
                self.model_alias,
                "--effort",
                self.reasoning_effort,
                "--disable-slash-commands",
                "--no-session-persistence",
                "--no-chrome",
                "--dangerously-skip-permissions",
            ]
        )
        if runtime_hooks_enabled:
            settings_path = (
                str(runtime_settings_path)
                if self.binary is not None
                else f"/workspace/.rho/{state_dir.name}/runtime-settings.json"
            )
            cmd.extend(
                [
                    "--tools",
                    self.tools,
                    "--setting-sources",
                    "user",
                    "--settings",
                    settings_path,
                    "--include-hook-events",
                ]
            )
        else:
            cmd.extend(["--bare", "--safe-mode"])
        if harness_context:
            if self.binary is not None:
                harness_context = (
                    f"{harness_context.rstrip()}\n\n"
                    "Execution workspace: "
                    f"{workspace}. Resolve relative paths from this directory; "
                    f"the repository is at {workspace / 'task' / 'repo'}. "
                    "Do not guess a /root workspace path."
                )
            prompt_flag = (
                "--system-prompt"
                if self.system_prompt_mode == "replace"
                else "--append-system-prompt"
            )
            cmd.extend(
                [
                    prompt_flag,
                    harness_context,
                ]
            )
        if self.autocompact:
            cmd.extend(["--autocompact", self.autocompact])
        if mcp_config_path.is_file():
            cmd.extend([
                "--mcp-config",
                str(mcp_config_path) if self.binary is not None else "/workspace/harness/mcp.json",
                "--strict-mcp-config",
            ])
        cmd.append(prompt)
        proc_env = _subprocess_env(env)
        if self.binary is not None:
            no_proxy = proc_env.get("NO_PROXY") or proc_env.get("no_proxy") or ""
            local_no_proxy = ",".join(
                value
                for value in (no_proxy, "127.0.0.1", "localhost")
                if value
            )
            proc_env.update(
                {
                    "ANTHROPIC_BASE_URL": self.ccr_base_url,
                    "ANTHROPIC_API_KEY": request_api_key,
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                    "DISABLE_TELEMETRY": "1",
                    "CLAUDE_CONFIG_DIR": str(state_dir / "config"),
                    "HOME": str(native_home),
                    "NO_PROXY": local_no_proxy,
                    "no_proxy": local_no_proxy,
                }
            )
            git_config_count_raw = proc_env.get("GIT_CONFIG_COUNT", "0")
            try:
                git_config_count = max(0, int(git_config_count_raw))
            except ValueError:
                git_config_count = 0
            proc_env.update(
                {
                    "GIT_CONFIG_COUNT": str(git_config_count + 1),
                    f"GIT_CONFIG_KEY_{git_config_count}": "safe.directory",
                    f"GIT_CONFIG_VALUE_{git_config_count}": str(
                        workspace / "task" / "repo"
                    ),
                }
            )
            if runtime_hooks_enabled:
                assert self.runtime_hook_script is not None
                proc_env.update(
                    {
                        "RHO_CLAUDE_CORE_HOOK": str(self.runtime_hook_script),
                        "RHO_RUNTIME_SELECTION_PATH": str(selection_path),
                        "RHO_RUNTIME_STATE_ROOT": str(state_dir / "runtime-state"),
                        "RHO_RUNTIME_AUDIT_PATH": str(runtime_audit_path),
                        "RHO_RUNTIME_INTERFACE": "claude-code",
                        "RHO_AGENT_TIMEOUT_S": str(effective_timeout),
                        "RHO_CLAUDE_CONTEXT_IN_SYSTEM": (
                            "1"
                            if self.include_runtime_context_in_system_prompt
                            else "0"
                        ),
                        "RHO_RTK_BIN": str(self.rtk_binary or ""),
                        "RTK_TELEMETRY_DISABLED": "1",
                    }
                )
        try:
            transport_file, maximum_compensation_s = _transport_compensation_config(
                proc_env
            )
            if transport_file is None or maximum_compensation_s <= 0:
                stdout, stderr, exit_code, timed_out, wall = _run_process(
                    cmd,
                    cwd=workspace,
                    env=proc_env,
                    timeout_s=effective_timeout + 5,
                )
                transport_compensation_s = 0.0
            else:
                (
                    stdout,
                    stderr,
                    exit_code,
                    timed_out,
                    wall,
                    transport_compensation_s,
                ) = _run_process_transport_aware(
                    cmd,
                    cwd=workspace,
                    env=proc_env,
                    timeout_s=effective_timeout + 5,
                )
        finally:
            if container_name is not None:
                subprocess.run(
                    [str(self.docker_command), "rm", "-f", container_name],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )
        if runtime_hooks_enabled:
            try:
                runtime_audit = runtime_audit_path.read_text(encoding="utf-8").strip()
            except OSError:
                runtime_audit = ""
            if runtime_audit:
                stderr = (
                    f"{stderr.rstrip()}\n[rho-claude-code-runtime]\n{runtime_audit}\n"
                    if stderr.strip()
                    else f"[rho-claude-code-runtime]\n{runtime_audit}\n"
                )
        after = _snapshot(workspace, exclude={".rho"})
        result, calls, session_started = _parse_claude_code_stream(stdout)
        if result.get("subtype") == "success" and not result.get("is_error"):
            # A completed native agent loop remains gradable even if Docker's
            # attach process reported a non-model cleanup error afterward.
            exit_code = 0
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        events: list[dict[str, Any]] = []
        if transport_compensation_s > 0:
            events.append(
                {
                    "type": "transport_timeout_compensation",
                    "seconds": round(transport_compensation_s, 6),
                }
            )
        expected_request_count = _as_int(result.get("num_turns"))
        traced_usage = (
            _read_ccr_trace_usage(
                self.ccr_metrics_base_url,
                usage_trace_id,
                expected_request_count=expected_request_count,
                expected_usage=usage,
            )
            if self.ccr_metrics_base_url is not None and usage_trace_id
            else {}
        )
        traced_by_model = (
            traced_usage.get("by_model")
            if isinstance(traced_usage.get("by_model"), dict)
            else {}
        )
        traced_request_count = _as_int(traced_usage.get("request_count"))
        traced_usage_complete = _ccr_trace_is_complete(
            traced_usage,
            expected_request_count=expected_request_count,
            expected_usage=usage,
        )
        if traced_usage_complete:
            trace_matches_aggregate = _ccr_trace_matches_aggregate(
                traced_by_model,
                usage,
            )
            for usage_model, model_usage in sorted(traced_by_model.items()):
                if not isinstance(model_usage, dict):
                    continue
                events.append(
                    _usage_event(
                        uncached_input=_as_int(model_usage.get("input_tokens")),
                        cached_input=_as_int(
                            model_usage.get("cache_read_input_tokens")
                        ),
                        cache_write_input=_as_int(
                            model_usage.get("cache_creation_input_tokens")
                        ),
                        output=_as_int(model_usage.get("output_tokens")),
                        api_calls=_as_int(model_usage.get("request_count")),
                        usage_model=str(usage_model),
                        usage_source="ccr-trace",
                        usage_trace_id=usage_trace_id,
                        usage_trace_request_count=traced_request_count,
                        usage_trace_expected_request_count=expected_request_count,
                        usage_trace_matches_aggregate=trace_matches_aggregate,
                    )
                )
        elif usage:
            output_details = (
                usage.get("output_tokens_details")
                if isinstance(usage.get("output_tokens_details"), dict)
                else {}
            )
            events.append(
                _usage_event(
                    uncached_input=_as_int(usage.get("input_tokens")),
                    cached_input=_as_int(usage.get("cache_read_input_tokens")),
                    cache_write_input=_as_int(
                        usage.get("cache_creation_input_tokens")
                    ),
                    output=_as_int(usage.get("output_tokens")),
                    reasoning_tokens=_as_int(output_details.get("thinking_tokens")),
                    api_calls=_as_int(result.get("num_turns")),
                    client_reported_cost_usd=result.get("total_cost_usd"),
                    client_cost_model="claude-alias-not-used-for-scoring",
                    usage_source="claude-code-aggregate",
                    usage_trace_id=usage_trace_id or None,
                    usage_trace_request_count=traced_request_count,
                    usage_trace_expected_request_count=expected_request_count,
                )
            )
        events.extend(
            _tool_events(calls, session_started_at_s=session_started, preserve_missing_timestamps=True)
        )
        diff, deletions = _diff(before, after)
        return Trajectory(
            id=f"traj_{uuid.uuid4().hex[:10]}",
            kind=kind,
            task_id=task_id,
            harness_id=harness_id,
            instructions=instructions,
            events=events,
            final_message=str(result.get("result") or ""),
            stdout=stdout,
            stderr=stderr,
            workspace_diff=diff,
            workspace_deletions=frozenset(deletions),
            exit_code=exit_code,
            wall_time_s=wall,
            timed_out=timed_out,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
        )


def _read_ccr_trace_usage(
    base_url: str | None,
    trace_id: str,
    *,
    expected_request_count: int = 0,
    expected_usage: dict[str, Any] | None = None,
    attempts: int = 50,
) -> dict[str, Any]:
    if not base_url or not trace_id:
        return {}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f"{base_url.rstrip('/')}/rho-usage/{trace_id}"
    best_payload: dict[str, Any] = {}
    last_observed = -1
    stable_polls = 0
    for attempt in range(attempts):
        try:
            with opener.open(url, timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            observed = _as_int(payload.get("request_count"))
            if observed:
                best_payload = payload
            by_model = payload.get("by_model")
            if _ccr_trace_is_complete(
                payload,
                expected_request_count=expected_request_count,
                expected_usage=expected_usage,
            ):
                return payload
            count_ready = observed and (
                not expected_request_count or observed >= expected_request_count
            )
            if count_ready and isinstance(by_model, dict):
                stable_polls = stable_polls + 1 if observed == last_observed else 0
                if stable_polls >= 10:
                    return payload
            else:
                stable_polls = 0
            last_observed = observed
        if attempt + 1 < attempts:
            time.sleep(0.1)
    if _ccr_trace_is_complete(
        best_payload,
        expected_request_count=expected_request_count,
        expected_usage=expected_usage,
    ):
        return best_payload
    return {}


def _ccr_trace_is_complete(
    payload: dict[str, Any],
    *,
    expected_request_count: int = 0,
    expected_usage: dict[str, Any] | None = None,
) -> bool:
    """Accept exact token accounting even when Claude's turn count is higher.

    Claude Code can report a final non-provider turn in ``num_turns``.  In that
    case the CCR trace has one fewer request, but it is still complete when its
    per-model token totals exactly reproduce Claude Code's aggregate usage.
    """

    observed = _as_int(payload.get("request_count"))
    by_model = payload.get("by_model")
    if not observed or not isinstance(by_model, dict) or not by_model:
        return False
    if expected_usage and _ccr_trace_matches_aggregate(by_model, expected_usage):
        return True
    return not expected_request_count or observed >= expected_request_count


def _ccr_trace_matches_aggregate(
    by_model: dict[str, Any], aggregate_usage: dict[str, Any]
) -> bool:
    """Accept routed usage when its token totals match Claude Code's aggregate.

    CCR can observe extra provider requests that Claude Code does not count as
    turns, including failed requests with zero usage. Request-count equality is
    therefore not a valid completeness check. Token equality preserves the
    aggregate as the source of truth while allowing those auditable zero-usage
    requests to remain in the per-model trace.
    """

    if not by_model or not aggregate_usage:
        return False
    trace_totals = {key: 0 for key in _CLAUDE_USAGE_KEYS}
    for model_usage in by_model.values():
        if not isinstance(model_usage, dict):
            return False
        for key in _CLAUDE_USAGE_KEYS:
            trace_totals[key] += _as_int(model_usage.get(key))
    return all(
        trace_totals[key] == _as_int(aggregate_usage.get(key))
        for key in _CLAUDE_USAGE_KEYS
    )


def _make_group_writable(root: Path) -> None:
    """Allow the image's non-root UID with GID 0 to edit the solve workspace."""

    for path in [root, *root.rglob("*")]:
        try:
            if path.is_symlink():
                continue
            mode = path.stat().st_mode
            path.chmod(mode | (0o070 if path.is_dir() else 0o060))
        except OSError:
            continue


def _parse_claude_code_stream(
    text: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], float | None]:
    result: dict[str, Any] = {}
    calls: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    observed_timestamps: list[float] = []
    for raw_line in text.splitlines():
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "result":
            result = event
        timestamp = _iso_epoch_seconds(event.get("timestamp"))
        if timestamp is not None:
            observed_timestamps.append(timestamp)
        message = event.get("message") if isinstance(event.get("message"), dict) else {}
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if event.get("type") == "assistant":
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "tool_use":
                    continue
                call_id = str(part.get("id") or "")
                if not call_id:
                    continue
                if call_id not in calls:
                    order.append(call_id)
                calls[call_id] = {
                    "id": call_id,
                    "name": part.get("name") or "claude_code_tool",
                    "started_at_s": timestamp,
                }
        elif event.get("type") == "user":
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "tool_result":
                    continue
                call_id = str(part.get("tool_use_id") or "")
                if not call_id:
                    continue
                call = calls.setdefault(
                    call_id,
                    {
                        "id": call_id,
                        "name": "claude_code_tool",
                        "started_at_s": None,
                    },
                )
                if call_id not in order:
                    order.append(call_id)
                call["completed_at_s"] = timestamp
                call["failed"] = bool(part.get("is_error"))
    session_started = min(observed_timestamps) if observed_timestamps else None
    return result, [calls[call_id] for call_id in order], session_started


def _iso_epoch_seconds(value: Any) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _render_external_harness_context(workspace: Path) -> str:
    selection = _read_json(workspace / ".rho" / "runtime-selection.json")
    if selection and not selection.get("routing_only"):
        sections: list[str] = []
        model_instructions = str(selection.get("model_instructions") or "").strip()
        model_instructions = model_instructions.replace(
            "You are Codex, a non-interactive software agent.",
            "You are a non-interactive software agent.",
        )
        if model_instructions:
            sections.append(model_instructions)
        session_context = str(selection.get("session_context") or "").strip()
        task_guidance = str(selection.get("task_guidance") or "").strip()
        if task_guidance:
            session_context = (
                f"{session_context}\nTask-feature guidance: {task_guidance}"
            ).strip()
        if session_context:
            sections.append(session_context)
        sections.append(
            "Tool-stage boundaries selected by the task-complexity rubric: "
            f"economy={selection.get('economy_after_tools')}, "
            f"strict={selection.get('strict_after_tools')}, "
            f"finalize={selection.get('finalize_after_tools')}."
        )
        return "\n\n".join(sections)

    readme_path = workspace / "harness" / "README.md"
    try:
        readme = readme_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if not readme:
        return ""
    return (
        "The evaluator supplied the following Harness playbook. Treat it as "
        "authoritative task-solving guidance. Its companion tools remain under "
        "harness/tools; do not reread README.md unless this injected copy is "
        "incomplete.\n\n"
        f"<harness-playbook>\n{readme}\n</harness-playbook>"
    )


def _render_claude_runtime_model_instructions(
    workspace: Path,
    *,
    include_runtime_context: bool = False,
) -> str:
    """Render the stable runtime instructions for Claude's system prompt.

    The V13c-compatible default keeps per-task session context in SessionStart.
    Claude-specific successors may include it here instead, keeping guidance in
    a cacheable prefix and avoiding reinjection after automatic compaction.
    """

    selection = _read_json(workspace / ".rho" / "runtime-selection.json")
    model_instructions = str(selection.get("model_instructions") or "").strip()
    model_instructions = model_instructions.replace(
        "You are Codex, a non-interactive software agent.",
        "You are a non-interactive software agent.",
    )
    if not include_runtime_context:
        return model_instructions

    sections = [model_instructions]
    session_context = str(selection.get("session_context") or "").strip()
    task_guidance = str(selection.get("task_guidance") or "").strip()
    if task_guidance:
        session_context = (
            f"{session_context}\nTask-feature guidance: {task_guidance}"
        ).strip()
    if session_context:
        sections.append(session_context)
    sections.append(
        "Tool-stage boundaries selected by the task-complexity rubric: "
        f"economy={selection.get('economy_after_tools')}, "
        f"strict={selection.get('strict_after_tools')}, "
        f"finalize={selection.get('finalize_after_tools')}."
    )
    return "\n\n".join(section for section in sections if section)


def _write_claude_code_runtime_settings(
    output: Path,
    *,
    include_stop: bool = False,
    command: str | None = None,
) -> None:
    hook = {
        "type": "command",
        "command": command or "/usr/bin/python3 /rho-harness/claude_adapter.py",
        "timeout": 10,
    }
    events = ["SessionStart", "PreToolUse", "PostToolUse"]
    if include_stop:
        events.append("Stop")
    payload = {"hooks": {event: [{"hooks": [hook]}] for event in events}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


