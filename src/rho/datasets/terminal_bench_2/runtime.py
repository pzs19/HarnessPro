from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING
from uuid import uuid4

from rho.datasets.terminal_bench_2 import cleanup, container

if TYPE_CHECKING:
    from rho.datasets.terminal_bench_2.dataset import TerminalBench2Task


logger = logging.getLogger(__name__)


HOST_WS_PATH = "/host-ws"
CONTAINER_CPUS_ENV = "RHO_TB2_CONTAINER_CPUS"
CONTAINER_MEMORY_ENV = "RHO_TB2_CONTAINER_MEMORY"
AGENT_TIMEOUT_FLOOR_ENV = "RHO_TB2_AGENT_TIMEOUT_FLOOR"
VERIFIER_TIMEOUT_FLOOR_ENV = "RHO_TB2_VERIFIER_TIMEOUT_FLOOR"
DISABLE_CONTAINER_WATCHDOG_ENV = "RHO_TB2_DISABLE_CONTAINER_WATCHDOG"


@dataclass(frozen=True)
class RuntimeHandle:
    container_name: str
    host_scratch: str


_THREAD_LOCAL = threading.local()


def set_active(
    task_id: str,
    container_name: str,
    session: "TerminalBench2RuntimeSession | None" = None,
) -> None:
    mapping = getattr(_THREAD_LOCAL, "mapping", None)
    if mapping is None:
        mapping = {}
        _THREAD_LOCAL.mapping = mapping
    mapping[task_id] = container_name
    if session is not None:
        sessions = getattr(_THREAD_LOCAL, "sessions", None)
        if sessions is None:
            sessions = {}
            _THREAD_LOCAL.sessions = sessions
        sessions[task_id] = session


def clear_active(task_id: str) -> None:
    mapping = getattr(_THREAD_LOCAL, "mapping", None)
    if mapping is not None:
        mapping.pop(task_id, None)
    sessions = getattr(_THREAD_LOCAL, "sessions", None)
    if sessions is not None:
        sessions.pop(task_id, None)


def get_active(task_id: str) -> str | None:
    mapping = getattr(_THREAD_LOCAL, "mapping", None)
    if mapping is None:
        return None
    return mapping.get(task_id)


def enter_grading_phase(task_id: str) -> None:
    """Stop the agent watchdog before the verifier gets its own timeout.

    The runtime context intentionally spans solve + grade so grading can see
    the same container. Leaving the agent watchdog armed would incorrectly
    charge verifier time against ``agent.timeout_sec`` and can kill a correct
    solution during grading. The grader already enforces verifier.timeout_sec.
    """

    sessions = getattr(_THREAD_LOCAL, "sessions", None)
    if sessions is None:
        return
    session = sessions.get(task_id)
    if session is not None:
        session.cancel_agent_watchdog()


_NAME_SAFE = re.compile(r"[^a-zA-Z0-9_.-]")


def _safe_task_id(task_id: str) -> str:
    return _NAME_SAFE.sub("_", task_id)[:40]


def resolve_container_resources(
    configured_cpus: int | None,
    configured_memory: str | None,
) -> tuple[int | None, str | None]:
    """Apply evaluator-wide resource floors/overrides when explicitly set.

    TB2's published task.toml values remain the default. Cost/performance
    experiments can opt into a larger, uniform container allocation without
    editing the downloaded benchmark data.
    """

    cpus = configured_cpus
    cpus_raw = os.environ.get(CONTAINER_CPUS_ENV, "").strip()
    if cpus_raw:
        try:
            cpus = int(cpus_raw)
        except ValueError as exc:
            raise ValueError(f"{CONTAINER_CPUS_ENV} must be a positive integer") from exc
        if cpus <= 0:
            raise ValueError(f"{CONTAINER_CPUS_ENV} must be a positive integer")

    memory = configured_memory
    memory_raw = os.environ.get(CONTAINER_MEMORY_ENV, "").strip()
    if memory_raw:
        memory = memory_raw
    return cpus, memory


def resolve_agent_timeout(configured_timeout_s: float) -> float:
    """Apply an optional evaluator-wide lower bound to the solver timeout."""

    floor_raw = os.environ.get(AGENT_TIMEOUT_FLOOR_ENV, "").strip()
    if not floor_raw:
        return configured_timeout_s
    try:
        floor = float(floor_raw)
    except ValueError as exc:
        raise ValueError(f"{AGENT_TIMEOUT_FLOOR_ENV} must be positive") from exc
    if floor <= 0:
        raise ValueError(f"{AGENT_TIMEOUT_FLOOR_ENV} must be positive")
    return max(configured_timeout_s, floor)


def resolve_verifier_timeout(configured_timeout_s: float) -> float:
    """Apply an optional evaluator-wide lower bound to the verifier timeout."""

    floor_raw = os.environ.get(VERIFIER_TIMEOUT_FLOOR_ENV, "").strip()
    if not floor_raw:
        return configured_timeout_s
    try:
        floor = float(floor_raw)
    except ValueError as exc:
        raise ValueError(f"{VERIFIER_TIMEOUT_FLOOR_ENV} must be positive") from exc
    if floor <= 0:
        raise ValueError(f"{VERIFIER_TIMEOUT_FLOOR_ENV} must be positive")
    return max(configured_timeout_s, floor)


def container_watchdog_enabled() -> bool:
    """Whether the runtime owns a separate wall-clock Solver watchdog.

    Transport-aware agents enforce the task's effective Solver budget while
    excluding audited provider pacing and retry wait.  Their outer process
    timeout remains authoritative, so a second wall-clock container timer
    would kill the workspace early during legitimate provider delay.
    """

    raw = os.environ.get(DISABLE_CONTAINER_WATCHDOG_ENV, "").strip().lower()
    return raw not in {"1", "true", "yes", "on"}


class TerminalBench2RuntimeSession:
    def __init__(self, task: "TerminalBench2Task", workdir: Path) -> None:
        self._task = task
        self._workdir = Path(workdir)
        self._container_name: str | None = None
        self._watchdog: threading.Timer | None = None
        self._watchdog_fired = threading.Event()
        self.timed_out = False

    def __enter__(self) -> RuntimeHandle:
        suffix = uuid4().hex[:8]
        name = f"tbench2-{_safe_task_id(self._task.id)}-{suffix}"
        self._container_name = name

        cleanup.register(name)

        image = self._task._config.docker_image
        policy = self._task._docker_pull
        container.ensure_image(image, policy)

        volumes = [
            container.Volume(
                host=self._workdir.resolve(),
                container=HOST_WS_PATH,
                mode="rw",
            ),
        ]
        labels = {"rho-tb2-pid": str(os.getpid())}
        cpus, memory = resolve_container_resources(
            self._task._config.cpus,
            self._task._config.memory,
        )
        try:
            container.start_container(
                name,
                image,
                memory=memory,
                cpus=cpus,
                volumes=volumes,
                labels=labels,
            )
        except Exception:
            cleanup.unregister(name)
            raise

        sidecar_dir = self._workdir / ".tb2"
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        (sidecar_dir / "container_name").write_text(name, encoding="utf-8")

        if container_watchdog_enabled():
            self._watchdog = threading.Timer(
                self._task.agent_timeout_s, self._on_timeout
            )
            self._watchdog.daemon = True
            self._watchdog.start()

        return RuntimeHandle(container_name=name, host_scratch=HOST_WS_PATH)

    def _on_timeout(self) -> None:
        self._watchdog_fired.set()
        self.timed_out = True
        name = self._container_name
        if name is not None:
            try:
                container.kill(name)
            except Exception:
                logger.exception("watchdog kill failed for %s", name)

    def cancel_agent_watchdog(self) -> None:
        watchdog = self._watchdog
        if watchdog is None:
            return
        watchdog.cancel()
        watchdog.join(timeout=1)
        self._watchdog = None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.cancel_agent_watchdog()
        name = self._container_name
        if name is not None:
            # Container runs as root; chown mounted files back to the host uid/gid
            # so the outer tempdir cleanup can remove them.
            try:
                if container.is_running(name):
                    container.exec_script(
                        name,
                        f"chown -R {os.getuid()}:{os.getgid()} {HOST_WS_PATH} 2>/dev/null || true",
                        timeout_s=60,
                    )
            except Exception:
                logger.exception("chown-back failed for %s (non-fatal)", name)
            try:
                container.rm_force(name)
            except Exception:
                logger.exception("rm_force failed for %s", name)
            cleanup.unregister(name)
