from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


DOCKER_PULL_POLICIES = {"missing", "always", "never"}


@dataclass(frozen=True)
class Volume:
    host: Path
    container: str
    mode: str = "rw"


def container_platform() -> str | None:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "linux/amd64"
    return None


def _run(
    cmd: list[str],
    *,
    timeout_s: float | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=check,
        timeout=timeout_s,
    )


def ensure_image(image: str, policy: str = "missing") -> None:
    if policy not in DOCKER_PULL_POLICIES:
        raise ValueError(f"docker_pull must be one of {sorted(DOCKER_PULL_POLICIES)}")
    if policy == "never":
        return
    if policy == "missing":
        proc = _run(["docker", "image", "inspect", image], timeout_s=60)
        if proc.returncode == 0:
            return
    cmd = ["docker", "pull"]
    plat = container_platform()
    if plat is not None:
        cmd += ["--platform", plat]
    cmd.append(image)
    proc = _run(cmd, timeout_s=1800)
    if proc.returncode != 0:
        raise RuntimeError(
            f"docker pull failed for {image}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


def start_container(
    name: str,
    image: str,
    *,
    memory: str | None = None,
    cpus: int | None = None,
    volumes: list[Volume] | None = None,
    labels: dict[str, str] | None = None,
    platform_override: str | None = None,
) -> None:
    cmd = ["docker", "run", "-d", "--name", name]
    if memory:
        cmd += ["--memory", memory]
    if cpus is not None:
        cmd += ["--cpus", str(cpus)]
    for volume in volumes or []:
        cmd += ["-v", f"{volume.host.resolve()}:{volume.container}:{volume.mode}"]
    for key, value in (labels or {}).items():
        cmd += ["--label", f"{key}={value}"]
    plat = platform_override if platform_override is not None else container_platform()
    if plat is not None:
        cmd += ["--platform", plat]
    cmd += ["--entrypoint", "sleep", image, "infinity"]
    proc = _run(cmd, timeout_s=120)
    if proc.returncode != 0:
        raise RuntimeError(f"docker run failed\ncmd: {' '.join(cmd)}\nstderr: {proc.stderr}")


def exec_script(
    name: str,
    script_or_cmd: str,
    *,
    timeout_s: float,
) -> subprocess.CompletedProcess[str]:
    """Run `sh -c <script_or_cmd>` inside the container. Uses POSIX sh
    (universally available, including busybox alpine) rather than bash,
    so container images without bash still work. TB2 task test scripts
    rely on their own `#!/bin/bash` shebang, which the kernel resolves
    independently when the script file is invoked as a path."""
    cmd = ["docker", "exec", "-i", name, "sh", "-c", script_or_cmd]
    return subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout_s)


def cp_to(name: str, src: str, dst: str) -> None:
    proc = _run(["docker", "cp", src, f"{name}:{dst}"], timeout_s=120)
    if proc.returncode != 0:
        raise RuntimeError(f"docker cp to failed: {proc.stderr}")


def cp_from(name: str, src_in_container: str, dst_host: Path) -> None:
    dst_host.parent.mkdir(parents=True, exist_ok=True)
    proc = _run(
        ["docker", "cp", f"{name}:{src_in_container}", str(dst_host)],
        timeout_s=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"docker cp from failed: {proc.stderr}")


def kill(name: str) -> None:
    _run(["docker", "kill", name], timeout_s=30)


def rm_force(name: str) -> None:
    _run(["docker", "rm", "-f", name], timeout_s=30)


def is_running(name: str) -> bool:
    proc = _run(["docker", "inspect", "-f", "{{.State.Running}}", name], timeout_s=15)
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def list_with_label(key: str) -> list[tuple[str, str]]:
    """Return [(container_name, label_value)] for every container with <key>=* label."""
    proc = _run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"label={key}",
            "--format",
            "{{.Names}}\t{{.Label \"" + key + "\"}}",
        ],
        timeout_s=15,
    )
    if proc.returncode != 0:
        return []
    out: list[tuple[str, str]] = []
    for line in proc.stdout.splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2 and parts[0]:
            out.append((parts[0], parts[1]))
    return out


def docker_available() -> bool:
    return shutil.which("docker") is not None
