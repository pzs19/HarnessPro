from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

from rho.datasets.terminal_bench_2 import container
from rho.datasets.terminal_bench_2.verifier_environment import environment_prefix, verifier_environment
from rho.datasets.terminal_bench_2.verifier_evidence import completed_assertion_failure
from rho.protocols import Grade


logger = logging.getLogger(__name__)


# TB2 test scripts commonly place the benchmark canary comment before their
# ``#!/bin/bash`` line.  That means the kernel cannot honor the shebang when
# the script is executed directly from ``sh -c``; POSIX sh then interprets the
# file itself and rejects Bash builtins such as ``source``.  Select Bash
# explicitly when it is available, while retaining a fallback for minimal
# images that genuinely only provide sh.  The explicit ``if`` is important:
# an ``bash ... || sh ...`` fallback would incorrectly run a failed verifier a
# second time.
_VERIFIER_COMMAND = (
    # Selenium's automatic manager can attempt a network download even when
    # the task image already provides a matching browser/driver pair. Prefer
    # that preinstalled driver, without overriding an explicitly set path.
    "if [ -x /usr/bin/chromedriver ] && [ -z \"${SE_CHROMEDRIVER:-}\" ]; then "
    "export SE_CHROMEDRIVER=/usr/bin/chromedriver; fi; "
    "if command -v bash >/dev/null 2>&1; then "
    "exec bash /tests/test.sh; "
    "else exec sh /tests/test.sh; fi"
)


_VERIFIER_INFRASTRUCTURE_MARKERS = (
    "request failed after 3 retries",
    "request failed after 5 retries",
    "caused by: operation timed out",
    "curl: (28) ssl connection timeout",
    "failed to download https://github.com/astral-sh/uv/",
    "temporary failure resolving",
    "could not resolve host",
    "name or service not known",
    "network is unreachable",
    "ssl connection timeout",
    "tls handshake timeout",
    "some index files failed to download",
    "e: unable to fetch some archives",
    "remote auto blocked until",
)


def _verifier_infrastructure_marker(stdout: str, stderr: str) -> str | None:
    """Return the first clear verifier-side network failure marker, if any.

    Some official TB2 verifier scripts write ``reward.txt = 0`` even when
    their own dependency bootstrap failed.  Treating that as a model failure
    makes the benchmark sensitive to mirror/API availability.  Keep the
    marker list deliberately narrow so ordinary assertion failures remain
    model results.
    """

    haystack = f"{stdout}\n{stderr}".lower()
    if _verifier_extension_toolchain_failure(stdout, stderr):
        return "verifier_extension_compiler_missing"
    return next(
        (marker for marker in _VERIFIER_INFRASTRUCTURE_MARKERS if marker in haystack),
        None,
    )


def _verifier_extension_toolchain_failure(stdout: str, stderr: str) -> bool:
    text = f"{stdout}\n{stderr}"
    if not re.search(r"error: command '(?:[^'\n]*/)?(?:cc|gcc|g\+\+|c\+\+|clang|clang\+\+)' failed: No such file or directory", text):
        return False
    if "running build_ext" not in text:
        return False
    for module in re.findall(r"building '([A-Za-z_][A-Za-z_0-9]*)' extension", text):
        missing = f"ModuleNotFoundError: No module named '{module}'"
        if (missing in text
                and re.search(r"^>\s+import " + re.escape(module) + r"\s*$", text, re.M)
                and re.search(r"^/tests/[^\n]+:\d+: ModuleNotFoundError$", text, re.M)):
            return True
    return False


def _verifier_browser_failure(stdout: str, stderr: str) -> bool:
    # Some upstream tests swallow driver startup exceptions and report PASS.
    # A browser test that never opened a browser is not a valid model result.
    marker = "failed to create driver or process file: message: unable to obtain driver for chrome"
    return any(line.strip().lower().startswith(marker)
               for line in f"{stdout}\n{stderr}".splitlines())


def run_tests(
    container_name: str,
    task_dir: Path,
    *,
    artifacts_dir: Path,
    verifier_timeout_s: float,
) -> Grade:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not container.is_running(container_name):
        return _fail(container_name, artifacts_dir, "container_dead")

    try:
        container.cp_to(
            container_name,
            str(task_dir / "tests") + "/.",
            "/tests/",
        )
    except Exception as exc:
        return _fail(container_name, artifacts_dir, "cp_tests_failed", message=str(exc))

    try:
        container.exec_script(container_name, "mkdir -p /tests /logs/verifier", timeout_s=30)
        container.exec_script(container_name, "chmod +x /tests/test.sh", timeout_s=30)
    except Exception as exc:
        return _fail(container_name, artifacts_dir, "setup_failed", message=str(exc))

    try:
        dependency_environment = verifier_environment(task_dir)
        proc = container.exec_script(
            container_name,
            environment_prefix(dependency_environment) + _VERIFIER_COMMAND,
            timeout_s=verifier_timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        # Preserve diagnostic output before teardown. A dependency installation
        # timeout must be distinguishable from a test that actually started.
        for name, output in (("test_stdout.log", exc.stdout), ("test_stderr.log", exc.stderr)):
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            _write_log(artifacts_dir / name, output or "")
        try:
            container.kill(container_name)
        except Exception:
            pass
        return _fail(container_name, artifacts_dir, "verifier_timeout")
    except Exception as exc:
        return _fail(container_name, artifacts_dir, "verifier_crashed", message=str(exc))

    _write_log(artifacts_dir / "test_stdout.log", proc.stdout)
    _write_log(artifacts_dir / "test_stderr.log", proc.stderr)

    try:
        container.cp_from(container_name, "/logs/verifier/.", artifacts_dir / "verifier")
    except Exception as exc:
        return _fail(
            container_name,
            artifacts_dir,
            "verifier_dir_missing",
            message=str(exc),
            docker_exit_code=proc.returncode,
        )

    reward_path = artifacts_dir / "verifier" / "reward.txt"
    if not reward_path.exists():
        return _fail(
            container_name,
            artifacts_dir,
            "reward_missing",
            docker_exit_code=proc.returncode,
        )

    reward = reward_path.read_text(encoding="utf-8").strip()
    details: dict[str, object] = {
        "reward": reward,
        "artifacts_dir": str(artifacts_dir),
        "container_name": container_name,
        "docker_exit_code": proc.returncode,
    }
    if dependency_environment:
        details["verifier_dependency_environment"] = dependency_environment
        if "network connectivity is disabled" in proc.stderr.lower() and "failed to download" in proc.stderr.lower():
            details["error"] = "verifier_dependency_failure"
            details["infra_marker"] = "prewarmed_uv_cache_incomplete"
            return Grade(passed=False, score=0.0, details=details)

    infrastructure_marker = _verifier_infrastructure_marker(proc.stdout, proc.stderr)
    if _verifier_browser_failure(proc.stdout, proc.stderr):
        details["error"] = "verifier_dependency_failure"
        details["infra_marker"] = "selenium_chrome_driver_startup_failed"
        return Grade(passed=False, score=0.0, details=details)
    if (reward == "0" and infrastructure_marker is not None
            and not completed_assertion_failure(artifacts_dir / "verifier" / "ctrf.json")):
        details["error"] = "verifier_network_failure"
        details["infra_marker"] = infrastructure_marker
        return Grade(passed=False, score=0.0, details=details)

    if reward not in {"0", "1"}:
        details["error"] = "reward_malformed"
        return Grade(passed=False, score=0.0, details=details)

    ctrf_path = artifacts_dir / "verifier" / "ctrf.json"
    if ctrf_path.exists():
        details["ctrf_summary"] = _summarize_ctrf(ctrf_path)

    passed = reward == "1"
    return Grade(passed=passed, score=1.0 if passed else 0.0, details=details)


def _fail(
    container_name: str,
    artifacts_dir: Path,
    error: str,
    *,
    message: str | None = None,
    docker_exit_code: int | None = None,
) -> Grade:
    details: dict[str, object] = {
        "error": error,
        "artifacts_dir": str(artifacts_dir),
        "container_name": container_name,
    }
    if message is not None:
        details["message"] = message
    if docker_exit_code is not None:
        details["docker_exit_code"] = docker_exit_code
    return Grade(passed=False, score=0.0, details=details)


def _write_log(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content or "", encoding="utf-8")


def _summarize_ctrf(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"error": "ctrf_parse_failed"}
    results = data.get("results") or {}
    tests = results.get("tests") or []
    return {
        "total": len(tests),
        "passed": sum(1 for test in tests if test.get("status") == "passed"),
        "failed": sum(1 for test in tests if test.get("status") == "failed"),
        "names": [test.get("name") for test in tests][:50],
    }
