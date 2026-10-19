from __future__ import annotations

import atexit
import os
import subprocess
import threading

from rho.datasets.terminal_bench_2 import container


_PID_LABEL = "rho-tb2-pid"
_OWNED: set[str] = set()
_LOCK = threading.Lock()
_ATEXIT_REGISTERED = False


def _owned_names() -> set[str]:
    with _LOCK:
        return set(_OWNED)


def register(name: str) -> None:
    global _ATEXIT_REGISTERED
    with _LOCK:
        _OWNED.add(name)
        if not _ATEXIT_REGISTERED:
            atexit.register(_atexit_sweep)
            _ATEXIT_REGISTERED = True


def unregister(name: str) -> None:
    with _LOCK:
        _OWNED.discard(name)


def _atexit_sweep() -> None:
    with _LOCK:
        names = list(_OWNED)
    for name in names:
        try:
            container.rm_force(name)
        except Exception:
            pass


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, OverflowError):
        return False
    except PermissionError:
        return True
    return True


def startup_sweep() -> int:
    """Remove tbench2-* containers whose PID label points to a dead process."""
    candidates = container.list_with_label(_PID_LABEL)
    removed = 0
    for name, pid_str in candidates:
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        if _pid_alive(pid):
            continue
        try:
            container.rm_force(name)
            removed += 1
        except Exception:
            pass
    return removed


def cli_cleanup(*, all_tb2: bool = False) -> int:
    """Remove orphaned or all TB2 containers depending on the flag."""
    if not all_tb2:
        return startup_sweep()
    proc = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=tbench2-", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if proc.returncode != 0:
        return 0
    removed = 0
    for name in proc.stdout.splitlines():
        stripped = name.strip()
        if not stripped:
            continue
        try:
            container.rm_force(stripped)
            removed += 1
        except Exception:
            pass
    return removed
