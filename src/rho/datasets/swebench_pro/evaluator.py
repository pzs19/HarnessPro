from __future__ import annotations

import json
import platform
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rho.datasets.swebench_pro.util import (
    cache_root,
    file_lock,
    git,
    parse_list_field,
)

OFFICIAL_REPO_URL = "https://github.com/scaleapi/SWE-bench_Pro-os.git"
OFFICIAL_REPO_COMMIT = "0c64e26f00b9c190432de7fc520c8ceed5c25518"
DEFAULT_DOCKER_IMAGE_PREFIX = "jefzda/sweap-images"
DOCKER_PULL_POLICIES = {"missing", "always", "never"}


@dataclass(frozen=True)
class DockerEvaluation:
    passed: bool
    score: float
    details: dict[str, Any]


class SWEbenchProDockerEvaluator:
    def __init__(
        self,
        *,
        docker_pull: str = "missing",
        image_prefix: str = DEFAULT_DOCKER_IMAGE_PREFIX,
        assets_root: Path | None = None,
        docker_platform: str | None = None,
        timeout_s: float = 3600.0,
        env_overrides: dict[str, str] | None = None,
    ) -> None:
        if docker_pull not in DOCKER_PULL_POLICIES:
            allowed = ", ".join(sorted(DOCKER_PULL_POLICIES))
            raise ValueError(f"docker_pull must be one of: {allowed}")
        self.docker_pull = docker_pull
        self.image_prefix = image_prefix
        self.assets_root = (
            assets_root or cache_root() / "swebench-pro" / "SWE-bench_Pro-os"
        )
        self.docker_platform = docker_platform or _default_docker_platform()
        self.timeout_s = timeout_s
        self.env_overrides = _validate_env_overrides(env_overrides or {})

    def evaluate(
        self,
        *,
        row: dict[str, Any],
        patch: str,
        artifacts_dir: Path,
    ) -> DockerEvaluation:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        uid = str(row["instance_id"])
        workspace_dir = artifacts_dir / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "patch.diff").write_text(patch, encoding="utf-8")

        scripts_root = ensure_official_assets(self.assets_root)
        files, entryscript = assemble_workspace_files(
            row=row,
            patch=patch,
            scripts_root=scripts_root,
            env_overrides=self.env_overrides,
        )
        for rel, content in files.items():
            target = workspace_dir / rel
            target.write_text(content, encoding="utf-8")
        (artifacts_dir / "entryscript.sh").write_text(entryscript, encoding="utf-8")

        image = docker_image(row, image_prefix=self.image_prefix)
        self._ensure_image(image)

        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{workspace_dir.resolve()}:/workspace:rw",
            "--entrypoint",
            "/bin/bash",
        ]
        if self.docker_platform is not None:
            cmd += ["--platform", self.docker_platform]
        cmd += [image, "-c", "bash /workspace/entryscript.sh"]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=self.timeout_s,
        )
        (artifacts_dir / "docker_stdout.log").write_text(proc.stdout, encoding="utf-8")
        (artifacts_dir / "docker_stderr.log").write_text(proc.stderr, encoding="utf-8")

        output_path = workspace_dir / "output.json"
        stdout_path = workspace_dir / "stdout.log"
        stderr_path = workspace_dir / "stderr.log"
        if stdout_path.exists():
            shutil.copy2(stdout_path, artifacts_dir / "test_stdout.log")
        if stderr_path.exists():
            shutil.copy2(stderr_path, artifacts_dir / "test_stderr.log")
        if not output_path.exists():
            return DockerEvaluation(
                passed=False,
                score=0.0,
                details={
                    "error": "output.json not produced",
                    "instance_id": uid,
                    "docker_image": image,
                    "docker_exit_code": proc.returncode,
                    "artifacts_dir": str(artifacts_dir),
                },
            )

        output = json.loads(output_path.read_text(encoding="utf-8"))
        shutil.copy2(output_path, artifacts_dir / "output.json")
        passed_tests = {
            str(item["name"])
            for item in output.get("tests", [])
            if item.get("status") == "PASSED"
        }
        required_tests = set(required_test_names(row))
        missing_tests = sorted(required_tests - passed_tests)
        passed = not missing_tests
        return DockerEvaluation(
            passed=passed,
            score=1.0 if passed else 0.0,
            details={
                "instance_id": uid,
                "docker_image": image,
                "docker_exit_code": proc.returncode,
                "required_tests": sorted(required_tests),
                "passed_tests": sorted(passed_tests),
                "missing_tests": missing_tests,
                "output": output,
                "artifacts_dir": str(artifacts_dir),
            },
        )

    def _ensure_image(self, image: str) -> None:
        if self.docker_pull == "always":
            self._docker_pull(image)
            return
        if self.docker_pull == "never":
            return
        inspect = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if inspect.returncode != 0:
            self._docker_pull(image)

    def _docker_pull(self, image: str) -> None:
        cmd = ["docker", "pull"]
        if self.docker_platform is not None:
            cmd += ["--platform", self.docker_platform]
        cmd.append(image)
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=self.timeout_s,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"docker pull failed for {image}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )


def ensure_official_assets(root: Path) -> Path:
    lock_path = root.parent / "SWE-bench_Pro-os.lock"
    with file_lock(lock_path):
        if not (root / ".git").exists():
            if root.exists():
                shutil.rmtree(root)
            git(["clone", "--quiet", OFFICIAL_REPO_URL, str(root)], timeout_s=1800)
        try:
            git(["cat-file", "-e", f"{OFFICIAL_REPO_COMMIT}^{{commit}}"], cwd=root)
        except RuntimeError:
            git(["fetch", "--quiet", "origin"], cwd=root, timeout_s=1800)
        git(["checkout", "--quiet", "--detach", OFFICIAL_REPO_COMMIT], cwd=root)
    return root


def assemble_workspace_files(
    *,
    row: dict[str, Any],
    patch: str,
    scripts_root: Path,
    env_overrides: dict[str, str] | None = None,
) -> tuple[dict[str, str], str]:
    uid = str(row["instance_id"])
    run_dir = scripts_root / "run_scripts" / uid
    run_script = (run_dir / "run_script.sh").read_text(encoding="utf-8")
    parser = (run_dir / "parser.py").read_text(encoding="utf-8")
    cleaned_patch = strip_binary_hunks(patch)
    entryscript = create_entryscript(
        row,
        scripts_root=scripts_root,
        env_overrides=env_overrides,
    )
    return {
        "patch.diff": cleaned_patch,
        "run_script.sh": run_script,
        "parser.py": parser,
        "entryscript.sh": entryscript,
    }, entryscript


def create_entryscript(
    row: dict[str, Any],
    *,
    scripts_root: Path,
    env_overrides: dict[str, str] | None = None,
) -> str:
    before_repo_set_cmd = str(row.get("before_repo_set_cmd") or "").strip()
    setup_cmd = before_repo_set_cmd.split("\n")[-1] if before_repo_set_cmd else ""
    selected_tests = ",".join(parse_list_field(row.get("selected_test_files_to_run")))
    base_commit = str(row["base_commit"])
    env_cmds = _dockerfile_env_exports(row, scripts_root=scripts_root)
    override_cmds = "\n".join(
        f"export {key}={shlex.quote(value)}"
        for key, value in _validate_env_overrides(env_overrides or {}).items()
    )
    return f"""\
{env_cmds}
{override_cmds}
cd /app
git reset --hard {base_commit}
git checkout {base_commit}
git apply -v /workspace/patch.diff
{setup_cmd}
bash /workspace/run_script.sh {selected_tests} > /workspace/stdout.log 2> /workspace/stderr.log
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
"""


def _validate_env_overrides(values: dict[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_key, raw_value in values.items():
        key = str(raw_key)
        value = str(raw_value)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"invalid grader environment variable name: {key!r}")
        if "\x00" in value:
            raise ValueError(f"grader environment variable contains NUL: {key}")
        result[key] = value
    return result


def _dockerfile_env_exports(row: dict[str, Any], *, scripts_root: Path) -> str:
    uid = str(row["instance_id"])
    exports: list[str] = []
    for rel in (
        Path("dockerfiles") / "base_dockerfile" / uid / "Dockerfile",
        Path("dockerfiles") / "instance_dockerfile" / uid / "Dockerfile",
    ):
        path = scripts_root / rel
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("ENV "):
                exports.append(stripped.replace("ENV", "export", 1))
    return "\n".join(exports)


def strip_binary_hunks(patch: str) -> str:
    if not patch:
        return patch
    sections = re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)
    kept: list[str] = []
    for section in sections:
        if not section.strip():
            continue
        if re.search(r"^Binary files .* differ$", section, re.MULTILINE):
            continue
        if re.search(r"^GIT binary patch$", section, re.MULTILINE):
            continue
        kept.append(section)
    return "".join(kept)


def docker_image(
    row: dict[str, Any], *, image_prefix: str = DEFAULT_DOCKER_IMAGE_PREFIX
) -> str:
    tag = str(row.get("dockerhub_tag") or "").strip()
    if not tag:
        tag = dockerhub_tag(str(row["instance_id"]), str(row.get("repo", "")))
    return f"{image_prefix}:{tag}"


def dockerhub_tag(instance_id: str, repo: str) -> str:
    if repo:
        owner, name = repo.lower().split("/", 1)
        return f"{owner}.{name}-{instance_id.replace('instance_', '')}"
    return f"default-{instance_id}"


def required_test_names(row: dict[str, Any]) -> list[str]:
    return parse_list_field(
        row.get("fail_to_pass") or row.get("FAIL_TO_PASS")
    ) + parse_list_field(row.get("pass_to_pass") or row.get("PASS_TO_PASS"))


def _default_docker_platform() -> str | None:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "linux/amd64"
    return None
