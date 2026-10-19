from __future__ import annotations

import hashlib
import json
import os
import signal
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import warnings
from contextlib import suppress
from pathlib import Path
from typing import Any

from rho.agent.codex_pool import global_codex_pool
from rho.observability import is_runtime_scratch
from rho.protocols import Trajectory, TrajectoryKind

DEFAULT_TIMEOUT_S = 900.0
EVAL_TIMEOUT_S = 300.0
MAX_SNAPSHOT_FILE_BYTES = 1 * 1024 * 1024
# Keep this value in sync with rho.agent.cache.MAX_SNAPSHOT_FILE_BYTES.
# Changing either value independently is a cache format-breaking change.
DEFAULT_CODEX_MODEL = "gpt-5.5"
DEFAULT_REASONING_EFFORT = "high"
REASONING_EFFORT_CHOICES = ("minimal", "low", "medium", "high", "xhigh")
TRANSPORT_TIME_FILE_ENV = "RHO_TRANSPORT_TIME_FILE"
TRANSPORT_MAX_COMPENSATION_ENV = "RHO_TRANSPORT_MAX_COMPENSATION_S"
TRANSPORT_STATE_DIR_ENV = "RHO_TRANSPORT_STATE_DIR"
_CODEX_AUTH_FILES = ("auth.json",)
_ISOLATED_ENV_ALLOWLIST = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TMPDIR",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
    # GPU/RJob hosts reach external model endpoints through the proxy set by
    # gpu_proxy_on. Keep both cases because reqwest/curl-based clients and
    # subprocess tools do not all consult the same spelling.
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    # TB2 and similar containerized evaluators may use a dedicated Docker
    # daemon. Codex tool subprocesses must target the same daemon as the
    # evaluator; otherwise ordinary `docker exec` calls silently fall back to
    # /var/run/docker.sock and cannot find the task container.
    "DOCKER_HOST",
    # The repository's Docker wrapper resolves its socket from this root and
    # intentionally does not consult DOCKER_HOST. Forward both selectors so
    # login-shell PATH changes cannot send agent tools to a different daemon.
    "RHO_TB2_DOCKER_STATE_ROOT",
    # Optional official-CLI pin consumed by the local cdx-gateway adapter.
    # This is an executable selector, not a credential.
    "RHO_CDX_REAL_CODEX",
    # AZURE_CONFIG_DIR lets `az account get-access-token` (invoked by
    # Codex's [model_providers.*.auth] block) find the user's tenant +
    # refresh-token cache. Without it `az` looks at $HOME/.azure, but
    # we override HOME to the isolated Codex home below — so we must
    # forward an explicit override. _build_subprocess_env synthesizes a
    # default of <real-HOME>/.azure when this env var isn't already set.
    "AZURE_CONFIG_DIR",
)


def _sanitize_instruction_text(text: str) -> str:
    """Make arbitrary task-derived text safe to pass through ``execve`` argv.

    A trajectory can contain binary command output whose decoded form includes
    NUL.  POSIX argv cannot represent NUL, so preserve it as a visible escape
    rather than letting an otherwise valid agent call fail before launch.
    """
    return text.replace("\x00", "\\x00")


# ═══════════════════════════════════════════════════════════════════════
# Sandbox autodetect
#
# The codex CLI wraps its tool calls in bubblewrap when --sandbox is
# workspace-write or read-only. On Ubuntu 24.04+, the sysctl
# `kernel.apparmor_restrict_unprivileged_userns=1` (on by default) puts
# any unconfined process entering a user namespace into a catch-all
# apparmor profile at /etc/apparmor.d/unprivileged_userns that contains
# `audit deny capability`, which denies CAP_NET_ADMIN. bubblewrap then
# cannot bring up its loopback interface and dies with
# `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`.
# The outer `codex exec` exits 0 with the error in stdout; the inner
# tool calls silently fail.
#
# We detect this condition at first-use time via a cheap /proc read and,
# if triggered, default to `danger-full-access` with a runtime warning.
# Users who have a working bubblewrap apparmor profile (e.g. the local
# override described in the spec §15 assumption 3) can override by
# passing sandbox="workspace-write" explicitly to CodexAgent(...).
# ═══════════════════════════════════════════════════════════════════════

_APPARMOR_USERNS_SYSCTL = Path("/proc/sys/kernel/apparmor_restrict_unprivileged_userns")

_detected_sandbox: str | None = None
_sandbox_warning_emitted = False
_sandbox_lock = threading.Lock()


def _detect_default_sandbox() -> tuple[str, str | None]:
    """Return (sandbox, warning_message_or_None) based on host heuristics.

    Pure stdlib, no codex subprocess, no I/O beyond one /proc read.
    """
    try:
        if _APPARMOR_USERNS_SYSCTL.read_text().strip() == "1":
            return (
                "danger-full-access",
                (
                    "CodexAgent: bubblewrap sandbox cannot configure its "
                    "network namespace on this host — detected "
                    "kernel.apparmor_restrict_unprivileged_userns=1 "
                    "(typical on Ubuntu 23.10+). Falling back to "
                    "'danger-full-access': codex tool calls will run "
                    "WITHOUT bwrap-level filesystem/network isolation. "
                    "To restore the sandbox, see "
                    "docs/superpowers/specs/"
                    "2026-04-11-rho-basic-impl-design.md §15 "
                    "assumption 3, then pass sandbox='workspace-write' "
                    "to CodexAgent explicitly."
                ),
            )
    except (OSError, ValueError):
        pass  # not Linux, /proc not readable, etc.
    return "workspace-write", None


def default_sandbox() -> str:
    """Cached per-process default sandbox. Emits a one-shot warning on
    degraded environments so it shows up in pytest output and dev runs
    without being printed per-call."""
    global _detected_sandbox, _sandbox_warning_emitted
    if _detected_sandbox is not None:
        return _detected_sandbox
    with _sandbox_lock:
        if _detected_sandbox is not None:
            return _detected_sandbox
        sandbox, warning = _detect_default_sandbox()
        _detected_sandbox = sandbox
        if warning and not _sandbox_warning_emitted:
            warnings.warn(warning, RuntimeWarning, stacklevel=3)
            _sandbox_warning_emitted = True
    return _detected_sandbox


class CodexAgent:
    def __init__(
        self,
        codex_config_path: Path,
        model: str | None = DEFAULT_CODEX_MODEL,
        reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
        sandbox: str | None = None,
        permission_profile: str | None = None,
        # fallback_sandbox is a LAST-RESORT retry when the primary sandbox
        # emits a bubblewrap failure at runtime (exit code 0 + `bwrap:` /
        # `Operation not permitted` in stdout/stderr). It defaults to None
        # — no silent escalation to a weaker sandbox. In most environments
        # autodetect (sandbox=None) already picks the right default so the
        # fallback path is not exercised.
        fallback_sandbox: str | None = None,
        network_access: bool = True,
        binary: str = "codex",
        extra_flags: tuple[str, ...] = (),
        default_timeout_s: float = DEFAULT_TIMEOUT_S,
        isolate_codex_home: bool = True,
        codex_auth_home: Path | None = None,
        forward_openai_api_key: bool = False,
        ephemeral: bool = True,
    ) -> None:
        if (
            reasoning_effort is not None
            and reasoning_effort not in REASONING_EFFORT_CHOICES
        ):
            allowed = ", ".join(REASONING_EFFORT_CHOICES)
            raise ValueError(f"reasoning_effort must be one of: {allowed}")
        if permission_profile is not None:
            if not permission_profile.strip():
                raise ValueError("permission_profile must be non-empty")
            if sandbox is not None:
                raise ValueError(
                    "permission_profile and the legacy sandbox setting are "
                    "mutually exclusive"
                )
            if fallback_sandbox is not None:
                raise ValueError(
                    "permission_profile cannot be combined with fallback_sandbox"
                )
            if network_access:
                raise ValueError(
                    "permission_profile owns network policy; pass "
                    "network_access=False"
                )
        self.codex_config_path = Path(codex_config_path).expanduser().resolve()
        if not self.codex_config_path.is_file():
            raise FileNotFoundError(f"codex config not found: {self.codex_config_path}")
        self.model = model
        self.reasoning_effort = reasoning_effort
        # A permission profile replaces the legacy --sandbox mechanism.  With
        # no profile, sandbox=None retains the historical autodetect behavior.
        self.permission_profile = permission_profile
        self.sandbox = (
            None
            if permission_profile is not None
            else (sandbox if sandbox is not None else default_sandbox())
        )
        self.fallback_sandbox = fallback_sandbox
        self.network_access = network_access
        self.binary = binary
        self.extra_flags = tuple(extra_flags)
        self.default_timeout_s = default_timeout_s
        self.isolate_codex_home = isolate_codex_home
        self.codex_auth_home = codex_auth_home or default_codex_auth_home()
        self.forward_openai_api_key = forward_openai_api_key
        self.ephemeral = ephemeral
        self._isolated_home_tmp: tempfile.TemporaryDirectory[str] | None = None
        self._isolated_codex_home: Path | None = None
        self._isolated_config_sha256 = ""
        if self.isolate_codex_home:
            self._prepare_isolated_codex_home()
        self._binary_version_cache: str | None = None
        self._version_lock = threading.Lock()
        # Per-thread because one CodexAgent instance is shared by concurrent
        # evaluation workers. _run_once records when each stdout JSONL line
        # was observed so run() can attach local tool timing without changing
        # the Codex wire payload or relying on absent server timestamps.
        self._stream_local = threading.local()

    @property
    def cache_env_signature(self) -> dict[str, str]:
        return {
            "agent_class": "CodexAgent",
            "model": self.model or "",
            "model_reasoning_effort": self.reasoning_effort or "",
            "binary": self.binary,
            "binary_version": self._binary_version(),
            "sandbox": self.sandbox or "",
            "permission_profile": self.permission_profile or "",
            "fallback_sandbox": self.fallback_sandbox or "",
            "network_access": str(self.network_access).lower(),
            "extra_flags": "\x1f".join(self.extra_flags),
            "codex_home_mode": "isolated" if self.isolate_codex_home else "ambient",
            "codex_config_sha256": self._isolated_config_sha256,
            "ephemeral": str(self.ephemeral).lower(),
        }

    @property
    def isolation_metadata(self) -> dict[str, Any]:
        return {
            "codex_home_mode": "isolated" if self.isolate_codex_home else "ambient",
            "inherits_user_config": not self.isolate_codex_home,
            "auth_source": str(self.codex_auth_home)
            if self.isolate_codex_home
            else None,
            "subprocess_env": "minimal" if self.isolate_codex_home else "inherited",
            "ephemeral": self.ephemeral,
            "config_sha256": self._isolated_config_sha256 or None,
            "config_source": str(self.codex_config_path),
            "permission_profile": self.permission_profile,
        }

    def _prepare_isolated_codex_home(self) -> None:
        self._isolated_home_tmp = tempfile.TemporaryDirectory(prefix="rho_codex_home_")
        codex_home = Path(self._isolated_home_tmp.name)
        codex_home.chmod(0o700)
        self._isolated_codex_home = codex_home

        self.codex_auth_home = self.codex_auth_home.expanduser().resolve()
        auth_files = () if self.forward_openai_api_key else _CODEX_AUTH_FILES
        for name in auth_files:
            src = self.codex_auth_home / name
            if src.exists():
                dst = codex_home / name
                shutil.copy2(src, dst)
                dst.chmod(0o600)

        config_bytes = self.codex_config_path.read_bytes()
        dst_config = codex_home / "config.toml"
        dst_config.write_bytes(config_bytes)
        dst_config.chmod(0o600)
        self._isolated_config_sha256 = hashlib.sha256(config_bytes).hexdigest()

    def _subprocess_env(self) -> dict[str, str] | None:
        return self._build_subprocess_env(None)

    def _build_subprocess_env(
        self, extra_env: dict[str, str] | None
    ) -> dict[str, str] | None:
        if not self.isolate_codex_home:
            if extra_env is None:
                return None
            env = os.environ.copy()
            env.update({key: str(value) for key, value in extra_env.items()})
            return env
        if self._isolated_codex_home is None:
            raise RuntimeError("isolated Codex home was not initialized")

        env = {
            key: value
            for key in _ISOLATED_ENV_ALLOWLIST
            if (value := os.environ.get(key)) is not None
        }
        env.setdefault("PATH", os.defpath)
        # If the user hasn't set AZURE_CONFIG_DIR but has a real ~/.azure,
        # forward that so Codex's [model_providers.*.auth] az invocation
        # can locate the tenant + refresh-token cache. We capture the
        # outer HOME *before* overriding it to the isolated codex home.
        if "AZURE_CONFIG_DIR" not in env:
            outer_home = os.environ.get("HOME")
            if outer_home:
                candidate = Path(outer_home) / ".azure"
                if candidate.is_dir():
                    env["AZURE_CONFIG_DIR"] = str(candidate)
        env["CODEX_HOME"] = str(self._isolated_codex_home)
        env["HOME"] = str(self._isolated_codex_home)
        if self.forward_openai_api_key and os.environ.get("OPENAI_API_KEY"):
            env["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
        if extra_env is not None:
            env.update({key: str(value) for key, value in extra_env.items()})
        return env

    def _binary_version(self) -> str:
        if self._binary_version_cache is not None:
            return self._binary_version_cache
        with self._version_lock:
            if self._binary_version_cache is not None:
                return self._binary_version_cache
            try:
                proc = subprocess.run(
                    [self.binary, "--version"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=10,
                )
                self._binary_version_cache = (
                    proc.stdout or proc.stderr
                ).strip() or "unknown"
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                self._binary_version_cache = "unknown"
        return self._binary_version_cache

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
        instructions = _sanitize_instruction_text(instructions)
        meta = workspace / ".rho"
        meta.mkdir(exist_ok=True)
        (meta / "instructions.md").write_text(instructions, encoding="utf-8")
        last_msg_path = meta / "last_message.txt"
        last_msg_path.write_text("", encoding="utf-8")

        # Git metadata is evaluator state, not a Solver deliverable. Ignoring it
        # also avoids racing Git's loose-object housekeeping while snapshotting.
        snapshot_excludes = {".git", ".rho"}
        before = _snapshot(workspace, exclude=snapshot_excludes)
        effective_timeout = (
            timeout_s if timeout_s is not None else self.default_timeout_s
        )
        invocation_env = self._transport_env(
            env,
            task_id=task_id,
            harness_id=harness_id,
            kind=kind,
        )
        retry_note: dict[str, Any] | None = None

        with global_codex_pool().acquire():
            t0 = time.monotonic()
            stdout_raw, stderr_raw, exit_code, timed_out = self._run_once(
                workspace=workspace,
                instructions=instructions,
                output_schema=output_schema,
                last_msg_path=last_msg_path,
                sandbox=self.sandbox,
                timeout_s=effective_timeout,
                env=invocation_env,
            )
            # Trigger fallback on any sandbox failure, regardless of outer exit code.
            # On codex-cli 0.120+ the outer `codex exec` process can exit 0 even
            # when its inner tool calls are blocked by bubblewrap (e.g. RTM_NEWADDR
            # failing in a namespace-restricted host). In that case the first run's
            # exit_code is 0 but stdout/stderr carry `bwrap:` / `Operation not
            # permitted`, and we still need to retry with the fallback sandbox.
            if (
                not timed_out
                and self.fallback_sandbox
                and self.fallback_sandbox != self.sandbox
                and _looks_like_sandbox_failure(stdout_raw, stderr_raw)
            ):
                retry_note = {
                    "type": "sandbox_fallback",
                    "from": self.sandbox,
                    "to": self.fallback_sandbox,
                }
                stdout_raw, stderr_raw, exit_code, timed_out = self._run_once(
                    workspace=workspace,
                    instructions=instructions,
                    output_schema=output_schema,
                    last_msg_path=last_msg_path,
                    sandbox=self.fallback_sandbox,
                    timeout_s=effective_timeout,
                    env=invocation_env,
                )
            wall = time.monotonic() - t0
        after = _snapshot(workspace, exclude=snapshot_excludes)

        events: list[dict[str, Any]] = []
        if retry_note is not None:
            events.append(retry_note)
        transport_compensation_s = float(
            getattr(self._stream_local, "transport_compensation_s", 0.0)
        )
        if transport_compensation_s > 0:
            events.append(
                {
                    "type": "transport_timeout_compensation",
                    "seconds": round(transport_compensation_s, 6),
                }
            )
        observed_at_s = getattr(self._stream_local, "stdout_observed_at_s", ())
        for line_ix, line in enumerate(stdout_raw.splitlines()):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
                if line_ix < len(observed_at_s):
                    event["_rho_observed_at_s"] = observed_at_s[line_ix]
                events.append(event)
            except json.JSONDecodeError:
                events.append({"type": "raw_stdout", "line": stripped})
        # stderr stays in stderr.log / Trajectory.stderr — replaying it as events floods the stream when codex dumps its model-refresh JSON response.

        try:
            final = last_msg_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            final = ""

        diff, deletions = _diff(before, after)
        return Trajectory(
            id=f"traj_{uuid.uuid4().hex[:10]}",
            kind=kind,
            task_id=task_id,
            harness_id=harness_id,
            instructions=instructions,
            events=events,
            final_message=final,
            stdout=stdout_raw,
            stderr=stderr_raw,
            workspace_diff=diff,
            workspace_deletions=frozenset(deletions),
            exit_code=exit_code,
            wall_time_s=wall,
            timed_out=timed_out,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
        )

    def _transport_env(
        self,
        env: dict[str, str] | None,
        *,
        task_id: str,
        harness_id: str,
        kind: TrajectoryKind,
    ) -> dict[str, str] | None:
        """Create a unique proxy timing channel for an agent invocation.

        Benchmark-specific runners already pass an explicit timing file.  The
        evolution and ReasoningBank pipelines do not own a per-task runner, so
        they opt in by exporting ``RHO_TRANSPORT_STATE_DIR``.  A unique trace
        prevents concurrent Codex calls from sharing wait accounting while
        preserving explicit caller-provided trace settings unchanged.
        """
        merged = dict(env or {})
        if merged.get(TRANSPORT_TIME_FILE_ENV):
            return merged

        raw_root = merged.get(TRANSPORT_STATE_DIR_ENV) or os.environ.get(
            TRANSPORT_STATE_DIR_ENV
        )
        if not raw_root:
            return env
        raw_maximum = merged.get(TRANSPORT_MAX_COMPENSATION_ENV) or os.environ.get(
            TRANSPORT_MAX_COMPENSATION_ENV
        )
        try:
            maximum = max(0.0, float(raw_maximum or 0.0))
        except (TypeError, ValueError):
            maximum = 0.0
        if maximum <= 0:
            return env

        state_dir = Path(raw_root).expanduser().resolve()
        state_dir.mkdir(parents=True, exist_ok=True)
        trace_id = uuid.uuid4().hex
        merged["RHO_TRANSPORT_TRACE_ID"] = trace_id
        merged[TRANSPORT_TIME_FILE_ENV] = str(state_dir / f"{trace_id}.json")
        merged[TRANSPORT_MAX_COMPENSATION_ENV] = str(maximum)
        return merged

    def _run_once(
        self,
        *,
        workspace: Path,
        instructions: str,
        output_schema: dict[str, Any] | None,
        last_msg_path: Path,
        sandbox: str | None,
        timeout_s: float,
        env: dict[str, str] | None,
    ) -> tuple[str, str, int, bool]:
        cmd: list[str] = [
            self.binary,
            "exec",
            "--cd",
            str(workspace),
            "--json",
            "--skip-git-repo-check",
            "--output-last-message",
            str(last_msg_path),
        ]
        if sandbox is not None:
            cmd += ["--sandbox", sandbox]
        if self.permission_profile is not None:
            cmd += [
                "-c",
                f'default_permissions="{self.permission_profile}"',
            ]
        if self.ephemeral:
            cmd += ["--ephemeral"]
        if self.model:
            cmd += ["-m", self.model]
        if self.reasoning_effort:
            cmd += ["-c", f'model_reasoning_effort="{self.reasoning_effort}"']
        if self.network_access:
            cmd += ["-c", "sandbox_workspace_write.network_access=true"]
        if output_schema is not None:
            schema_path = workspace / ".rho" / "output_schema.json"
            schema_path.write_text(
                json.dumps(output_schema, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            cmd += ["--output-schema", str(schema_path)]
        cmd += list(self.extra_flags)
        # Feed the prompt through stdin. Linux limits each individual argv
        # string to MAX_ARG_STRLEN (normally 128 KiB), even when ARG_MAX is
        # much larger. Candidate-judge prompts can legitimately contain two
        # sizeable patches and exceed that per-argument ceiling. Codex 0.147.0
        # documents ``-`` as the stdin prompt form, so this preserves prompt
        # semantics while keeping task text out of the process list.
        cmd += ["-"]

        stdout_lines: list[str] = []
        stdout_observed_at_s: list[float] = []
        stderr_lines: list[str] = []
        started = time.monotonic()
        self._stream_local.stdout_observed_at_s = ()
        self._stream_local.transport_compensation_s = 0.0

        subprocess_env = self._build_subprocess_env(env)
        invocation_home_tmp: tempfile.TemporaryDirectory[str] | None = None
        if self.isolate_codex_home:
            if self._isolated_codex_home is None or subprocess_env is None:
                raise RuntimeError("isolated Codex home was not initialized")
            # One CodexAgent is shared by concurrent solver workers.  Codex
            # mutates CODEX_HOME at startup (system-skill installation and
            # shell snapshots), so sharing the template directory causes
            # cross-process rename/remove races.  Give each invocation a
            # private copy containing only the immutable auth/config inputs.
            invocation_home_tmp = tempfile.TemporaryDirectory(
                prefix="rho_codex_run_"
            )
            invocation_home = Path(invocation_home_tmp.name)
            invocation_home.chmod(0o700)
            for source in self._isolated_codex_home.iterdir():
                if not source.is_file():
                    continue
                destination = invocation_home / source.name
                shutil.copy2(source, destination)
                destination.chmod(source.stat().st_mode & 0o777)
            subprocess_env["CODEX_HOME"] = str(invocation_home)
            subprocess_env["HOME"] = str(invocation_home)

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=subprocess_env,
            start_new_session=True,
        )
        assert proc.stdin is not None
        assert proc.stdout is not None
        assert proc.stderr is not None

        def _read_stdout() -> None:
            try:
                for line in proc.stdout:
                    stdout_lines.append(line)
                    stdout_observed_at_s.append(round(time.monotonic() - started, 6))
            except ValueError:
                # A descendant can retain the pipe after the Codex process has
                # exited.  The bounded join below then closes our local stream;
                # treat that expected shutdown race as EOF.
                if not proc.stdout.closed:
                    raise

        def _read_stderr() -> None:
            try:
                stderr_lines.extend(proc.stderr)
            except ValueError:
                if not proc.stderr.closed:
                    raise

        stdout_thread = threading.Thread(target=_read_stdout, daemon=True)
        stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        try:
            proc.stdin.write(instructions)
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            # Process startup errors are captured through its exit status and
            # stderr below. Do not mask that evidence with a pipe exception.
            if not proc.stdin.closed:
                with suppress(BrokenPipeError, OSError):
                    proc.stdin.close()

        timed_out = False
        transport_file, max_compensation_s = _transport_compensation_config(env)
        try:
            if transport_file is None or max_compensation_s <= 0:
                exit_code = proc.wait(timeout=timeout_s)
            else:
                base_deadline = started + timeout_s
                while True:
                    compensation_s = _read_transport_compensation(
                        transport_file, max_compensation_s
                    )
                    self._stream_local.transport_compensation_s = compensation_s
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
            # npm's Codex shim launches the native CLI as a descendant. Killing
            # only the shim leaves the native process alive with our pipes
            # open, which can hang reader shutdown and leak an evaluation.
            # Each invocation owns a session, so terminate the whole group.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
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
            if invocation_home_tmp is not None:
                invocation_home_tmp.cleanup()

        # An in-flight proxy reservation can be larger than the actual wait and
        # is intentionally removed before the successful response is returned.
        # Re-read the settled file after process exit so timing reports do not
        # retain the temporary reservation observed by the deadline monitor.
        if transport_file is not None and max_compensation_s > 0:
            self._stream_local.transport_compensation_s = (
                _read_transport_compensation(transport_file, max_compensation_s)
            )

        self._stream_local.stdout_observed_at_s = tuple(stdout_observed_at_s)
        return "".join(stdout_lines), "".join(stderr_lines), exit_code, timed_out


def _coerce_timeout_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _transport_compensation_config(
    env: dict[str, str] | None,
) -> tuple[Path | None, float]:
    if not env:
        return None, 0.0
    raw_path = str(env.get(TRANSPORT_TIME_FILE_ENV) or "").strip()
    if not raw_path:
        return None, 0.0
    try:
        maximum = max(0.0, float(env.get(TRANSPORT_MAX_COMPENSATION_ENV) or 0.0))
    except (TypeError, ValueError):
        maximum = 0.0
    return Path(raw_path), maximum


def _read_transport_compensation(path: Path, maximum_s: float) -> float:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = float(payload.get("transport_wait_s") or 0.0)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0.0
    return min(maximum_s, max(0.0, value))


def default_codex_auth_home() -> Path:
    try:
        import pwd

        return Path(pwd.getpwuid(os.getuid()).pw_dir) / ".codex"
    except (ImportError, KeyError, OSError):
        return Path.home() / ".codex"


def _looks_like_sandbox_failure(stdout: str, stderr: str) -> bool:
    haystack = f"{stdout}\n{stderr}"
    return "bwrap:" in haystack or "Operation not permitted" in haystack


def _snapshot(root: Path, exclude: set[str]) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for directory, dirnames, filenames in os.walk(
        root, topdown=True, onerror=lambda _error: None, followlinks=False
    ):
        current = Path(directory)
        rel_dir = current.relative_to(root)
        if rel_dir.parts and rel_dir.parts[0] in exclude:
            dirnames[:] = []
            continue
        # A task repository can be nested below the workspace root. Git
        # metadata is never a deliverable, so do not descend into it at any
        # depth. Pruning is important: filtering after Path.rglob() has already
        # entered .git/objects can race Git's loose-object housekeeping.
        dirnames[:] = [
            name
            for name in dirnames
            if name != ".git" and not (current == root and name in exclude)
        ]
        for filename in filenames:
            path = current / filename
            rel = path.relative_to(root)
            if rel.parts and rel.parts[0] in exclude:
                continue
            if is_runtime_scratch(rel):
                continue
            try:
                size = path.stat().st_size
                if size > MAX_SNAPSHOT_FILE_BYTES:
                    digest = hashlib.sha256()
                    with path.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(65536), b""):
                            digest.update(chunk)
                    out[str(rel)] = b"HASH:" + digest.hexdigest().encode("ascii")
                else:
                    out[str(rel)] = path.read_bytes()
            except OSError:
                # Build tools can remove temporary files between directory
                # enumeration and the stat/read. Such files are not stable
                # workspace outputs and should not fail an evaluation.
                continue
    return out


def _diff(
    before: dict[str, bytes],
    after: dict[str, bytes],
) -> tuple[dict[str, bytes], set[str]]:
    changes: dict[str, bytes] = {}
    for key, value in after.items():
        if before.get(key) != value:
            changes[key] = value
    deletions = {key for key in before if key not in after}
    return changes, deletions
