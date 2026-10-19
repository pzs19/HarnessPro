from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from rho.datasets.swebench_pro.util import cache_root, file_lock, git, safe_name


FULL_HISTORY_CACHE_DIR = "repos-full-history-v1"
FULL_HISTORY_CACHE_MODE = "full-history-v1"
CLEAN_HISTORY_CACHE_MODE = "clean-history-v1"


class RepoCache:
    """Cache complete SWE-bench Pro repositories without lazy object fetching."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        remote_base_url: str = "https://github.com",
        segmented_clone_ref_threshold: int = 400,
        segmented_depth: int = 500,
        segmented_ref_batch_size: int = 25,
    ) -> None:
        self.root = root or cache_root() / "swebench-pro" / FULL_HISTORY_CACHE_DIR
        self.remote_base_url = remote_base_url.rstrip("/")
        self.segmented_clone_ref_threshold = segmented_clone_ref_threshold
        self.segmented_depth = segmented_depth
        self.segmented_ref_batch_size = segmented_ref_batch_size
        self.root.mkdir(parents=True, exist_ok=True)
        self._prepared: dict[str, Path] = {}
        self._validation_reports: dict[str, dict[str, Any]] = {}
        self._state_lock = threading.Lock()

    @property
    def cache_mode(self) -> str:
        return FULL_HISTORY_CACHE_MODE

    def materialize(self, row: dict[str, Any], dest: Path) -> None:
        source_repo = row.get("repo_path")
        if source_repo:
            source = Path(str(source_repo)).expanduser().resolve()
            self._validate_complete_repository(source)
            self._copy_local_repo(source, dest)
            self._checkout(dest, str(row["base_commit"]))
            return

        repo = str(row["repo"])
        base_commit = str(row["base_commit"])
        cached = self.ensure_cached(repo, [base_commit])
        if dest.exists():
            shutil.rmtree(dest)
        git(["clone", "--quiet", str(cached), str(dest)], timeout_s=1200)
        self._checkout(dest, base_commit)

    def ensure_cached(self, repo: str, commits: Iterable[str]) -> Path:
        requested = tuple(dict.fromkeys(str(commit) for commit in commits))
        if not requested:
            raise ValueError("at least one commit is required")
        return self._ensure_cached(repo, requested)

    def _ensure_cached(self, repo: str, base_commit: str | Iterable[str]) -> Path:
        commits = (
            (base_commit,)
            if isinstance(base_commit, str)
            else tuple(dict.fromkeys(str(commit) for commit in base_commit))
        )
        if not commits:
            raise ValueError("at least one commit is required")

        with self._state_lock:
            prepared = self._prepared.get(repo)
        if prepared is not None:
            self._ensure_commits_present(prepared, commits)
            return prepared

        target = self.root / safe_name(repo)
        lock_path = self.root / f"{safe_name(repo)}.lock"
        with file_lock(lock_path):
            with self._state_lock:
                prepared = self._prepared.get(repo)
            if prepared is not None:
                self._ensure_commits_present(prepared, commits)
                return prepared

            clone_report: dict[str, Any] | None = None
            if not target.exists():
                clone_report = self._clone_complete_repository(repo, target)
            else:
                self._assert_git_repository(target)
                self._reject_incomplete_repository(target)

            missing = [
                commit for commit in commits if not self._has_commit(target, commit)
            ]
            if missing:
                git(
                    [
                        "-c",
                        "http.version=HTTP/1.1",
                        "fetch",
                        "--quiet",
                        "--no-filter",
                        "origin",
                        *missing,
                    ],
                    cwd=target,
                    timeout_s=3600,
                )
            self._ensure_commits_present(target, commits)
            for commit in commits:
                git(["update-ref", f"refs/rho/base/{commit}", commit], cwd=target)

            report = clone_report or self._validate_complete_repository(target)
            with self._state_lock:
                self._prepared[repo] = target
                self._validation_reports[repo] = report
        return target

    def repository_report(self, repo: str) -> dict[str, Any]:
        with self._state_lock:
            report = self._validation_reports.get(repo)
        if report is None:
            raise RuntimeError(f"repository has not been prepared: {repo}")
        return dict(report)

    def clone_args(self, repo: str, target: Path) -> list[str]:
        return [
            "-c",
            "http.version=HTTP/1.1",
            "clone",
            "--quiet",
            "--no-checkout",
            self._remote_url(repo),
            str(target),
        ]

    def _clone_complete_repository(
        self, repo: str, target: Path
    ) -> dict[str, Any]:
        temporary = target.with_name(
            f".{target.name}.clone-{threading.get_ident()}-{time.time_ns()}"
        )
        try:
            refs = self._remote_refs(repo)
            if len(refs["heads"]) + len(refs["tags"]) > (
                self.segmented_clone_ref_threshold
            ):
                self._clone_segmented_repository(repo, temporary, refs)
            else:
                git(self.clone_args(repo, temporary), timeout_s=7200)
            report = self._validate_complete_repository(temporary)
            temporary.replace(target)
            return {**report, "path": str(target.resolve())}
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def _remote_url(self, repo: str) -> str:
        return f"{self.remote_base_url}/{repo}.git"

    def _remote_refs(self, repo: str) -> dict[str, Any]:
        output = git(
            [
                "-c",
                "http.version=HTTP/1.1",
                "ls-remote",
                "--symref",
                self._remote_url(repo),
            ],
            timeout_s=600,
        )
        default_branch: str | None = None
        heads: list[str] = []
        tags: list[str] = []
        for line in output.splitlines():
            if line.startswith("ref: refs/heads/") and line.endswith("\tHEAD"):
                default_branch = line.removeprefix("ref: ").split("\t", 1)[0]
                continue
            _, separator, ref = line.partition("\t")
            if not separator or ref.endswith("^{}"):
                continue
            if ref.startswith("refs/heads/"):
                heads.append(ref)
            elif ref.startswith("refs/tags/"):
                tags.append(ref)
        if not heads:
            raise RuntimeError(f"remote repository has no branch refs: {repo}")
        if default_branch not in heads:
            default_branch = heads[0]
        return {
            "default_branch": default_branch,
            "heads": sorted(set(heads)),
            "tags": sorted(set(tags)),
        }

    def _run_network_git_with_retries(
        self,
        args: list[str],
        *,
        cwd: Path,
        timeout_s: float = 3600,
        attempts: int = 4,
    ) -> None:
        errors: list[str] = []
        for attempt in range(1, attempts + 1):
            try:
                git(args, cwd=cwd, timeout_s=timeout_s)
                return
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                if attempt < attempts:
                    time.sleep(min(60, 10 * attempt))
        raise RuntimeError(
            "segmented full-history fetch exhausted retries:\n" + "\n".join(errors)
        )

    def _fetch_ref_batch(self, repo_dir: Path, refspecs: list[str]) -> None:
        self._run_network_git_with_retries(
            [
                "-c",
                "http.version=HTTP/1.1",
                "fetch",
                "--quiet",
                "--no-tags",
                "origin",
                *refspecs,
            ],
            cwd=repo_dir,
        )

    def _clone_segmented_repository(
        self,
        repo: str,
        target: Path,
        refs: dict[str, Any],
    ) -> None:
        default_branch = str(refs["default_branch"])
        heads = [str(ref) for ref in refs["heads"]]
        tags = [str(ref) for ref in refs["tags"]]
        print(
            f"[repo-cache] segmented full fetch {repo} "
            f"heads={len(heads)} tags={len(tags)}",
            flush=True,
        )
        git(["init", "--quiet", str(target)])
        git(["remote", "add", "origin", self._remote_url(repo)], cwd=target)
        default_remote_ref = default_branch.replace(
            "refs/heads/", "refs/remotes/origin/", 1
        )
        default_refspec = f"+{default_branch}:{default_remote_ref}"
        self._run_network_git_with_retries(
            [
                "-c",
                "http.version=HTTP/1.1",
                "fetch",
                "--quiet",
                "--no-tags",
                f"--depth={self.segmented_depth}",
                "origin",
                default_refspec,
            ],
            cwd=target,
        )
        deepen_count = 0
        while (
            git(["rev-parse", "--is-shallow-repository"], cwd=target)
            .strip()
            .lower()
            == "true"
        ):
            self._run_network_git_with_retries(
                [
                    "-c",
                    "http.version=HTTP/1.1",
                    "fetch",
                    "--quiet",
                    "--no-tags",
                    f"--deepen={self.segmented_depth}",
                    "origin",
                    default_refspec,
                ],
                cwd=target,
            )
            deepen_count += 1
            if deepen_count % 5 == 0:
                print(
                    f"[repo-cache] {repo} default-history "
                    f"chunks={deepen_count + 1}",
                    flush=True,
                )
            if deepen_count > 1000:
                raise RuntimeError(f"segmented history did not reach root: {repo}")

        remaining_heads = [ref for ref in heads if ref != default_branch]
        head_refspecs = [
            f"+{ref}:{ref.replace('refs/heads/', 'refs/remotes/origin/', 1)}"
            for ref in remaining_heads
        ]
        tag_refspecs = [f"+{ref}:{ref}" for ref in tags]
        for refspecs in (head_refspecs, tag_refspecs):
            for offset in range(0, len(refspecs), self.segmented_ref_batch_size):
                self._fetch_ref_batch(
                    target,
                    refspecs[offset : offset + self.segmented_ref_batch_size],
                )

        local_branch = default_branch.removeprefix("refs/heads/")
        git(["update-ref", f"refs/heads/{local_branch}", default_remote_ref], cwd=target)
        git(["symbolic-ref", "HEAD", f"refs/heads/{local_branch}"], cwd=target)
        git(
            ["symbolic-ref", "refs/remotes/origin/HEAD", default_remote_ref],
            cwd=target,
        )
        git(
            [
                "config",
                "remote.origin.fetch",
                "+refs/heads/*:refs/remotes/origin/*",
            ],
            cwd=target,
        )
        for ref in heads:
            remote_ref = ref.replace("refs/heads/", "refs/remotes/origin/", 1)
            git(["rev-parse", "--verify", remote_ref], cwd=target)
        for ref in tags:
            git(["rev-parse", "--verify", ref], cwd=target)

    def _assert_git_repository(self, repo_dir: Path) -> None:
        try:
            git(["rev-parse", "--git-dir"], cwd=repo_dir)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                f"repository cache entry is not a valid Git repository: {repo_dir}"
            ) from exc

    def _config_value(self, repo_dir: Path, key: str) -> str | None:
        proc = subprocess.run(
            ["git", "config", "--get", key],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if proc.returncode == 1:
            return None
        if proc.returncode != 0:
            raise RuntimeError(
                f"git config --get {key} failed in {repo_dir}: {proc.stderr.strip()}"
            )
        return proc.stdout.strip()

    def _repository_state(self, repo_dir: Path) -> dict[str, Any]:
        shallow = git(
            ["rev-parse", "--is-shallow-repository"], cwd=repo_dir
        ).strip().lower() == "true"
        partial_extension = self._config_value(repo_dir, "extensions.partialclone")
        promisor = self._config_value(repo_dir, "remote.origin.promisor")
        partial_filter = self._config_value(
            repo_dir, "remote.origin.partialclonefilter"
        )
        return {
            "cache_mode": self.cache_mode,
            "path": str(repo_dir.resolve()),
            "shallow": shallow,
            "partial_clone": bool(partial_extension or partial_filter),
            "promisor": str(promisor or "").lower() == "true",
            "partial_clone_extension": partial_extension,
            "partial_clone_filter": partial_filter,
        }

    def _reject_incomplete_repository(self, repo_dir: Path) -> dict[str, Any]:
        state = self._repository_state(repo_dir)
        if state["shallow"] or state["partial_clone"] or state["promisor"]:
            raise RuntimeError(
                "SWE-bench Pro requires a complete non-promisor Git repository; "
                f"refusing incomplete cache entry at {repo_dir} "
                f"(shallow={state['shallow']}, "
                f"partial_clone={state['partial_clone']}, "
                f"promisor={state['promisor']})"
            )
        return state

    def _validate_complete_repository(self, repo_dir: Path) -> dict[str, Any]:
        self._assert_git_repository(repo_dir)
        state = self._reject_incomplete_repository(repo_dir)
        git(
            ["fsck", "--connectivity-only", "--no-dangling"],
            cwd=repo_dir,
            timeout_s=3600,
        )
        return {**state, "connectivity_check": "passed"}

    def _has_commit(self, repo_dir: Path, commit: str) -> bool:
        try:
            git(["cat-file", "-e", f"{commit}^{{commit}}"], cwd=repo_dir)
            return True
        except RuntimeError:
            return False

    def _ensure_commits_present(self, repo_dir: Path, commits: Iterable[str]) -> None:
        missing = [commit for commit in commits if not self._has_commit(repo_dir, commit)]
        if missing:
            raise RuntimeError(
                f"complete repository cache at {repo_dir} is missing commits: "
                + ", ".join(missing)
            )

    def _copy_local_repo(self, source: Path, dest: Path) -> None:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest, symlinks=True)

    def _checkout(self, repo_dir: Path, commit: str) -> None:
        try:
            git(["checkout", "--quiet", "--detach", commit], cwd=repo_dir)
        except RuntimeError as checkout_error:
            try:
                git(
                    ["-c", "http.version=HTTP/1.1", "fetch", "--quiet", "origin", commit],
                    cwd=repo_dir,
                    timeout_s=1800,
                )
            except RuntimeError:
                raise checkout_error
            git(["checkout", "--quiet", "--detach", commit], cwd=repo_dir)
        git(["reset", "--hard", "--quiet", commit], cwd=repo_dir)
        git(["clean", "-fdx", "--quiet"], cwd=repo_dir)


class CleanRepoCache:
    """Materialize a one-commit checkout without exposing upstream history.

    The complete mirror remains an evaluator-side source cache.  Solver
    workspaces receive a depth-one fetch of only the requested base commit,
    with no remote configured and no alternates back to the source object
    database.  This keeps patch extraction compatible with Git while making
    later/gold commits unavailable by object ID.
    """

    def __init__(
        self,
        root: Path | None = None,
        *,
        remote_base_url: str = "https://github.com",
        segmented_clone_ref_threshold: int = 400,
        segmented_depth: int = 500,
        segmented_ref_batch_size: int = 25,
    ) -> None:
        self._source = RepoCache(
            root,
            remote_base_url=remote_base_url,
            segmented_clone_ref_threshold=segmented_clone_ref_threshold,
            segmented_depth=segmented_depth,
            segmented_ref_batch_size=segmented_ref_batch_size,
        )
        # Preserve the runner/report interface. This path is evaluator-only;
        # it is never referenced from the solver workspace.
        self.root = self._source.root

    @property
    def cache_mode(self) -> str:
        return CLEAN_HISTORY_CACHE_MODE

    def ensure_cached(self, repo: str, commits: Iterable[str]) -> Path:
        return self._source.ensure_cached(repo, commits)

    def repository_report(self, repo: str) -> dict[str, Any]:
        source = self._source.repository_report(repo)
        return {
            **source,
            "cache_mode": self.cache_mode,
            "source_cache_mode": FULL_HISTORY_CACHE_MODE,
            "materialization": "depth-one-base-without-remotes-or-alternates",
        }

    def materialize(self, row: dict[str, Any], dest: Path) -> None:
        base_commit = str(row["base_commit"])
        source_repo = row.get("repo_path")
        if source_repo:
            source = Path(str(source_repo)).expanduser().resolve()
            self._source._validate_complete_repository(source)
        else:
            source = self._source.ensure_cached(str(row["repo"]), [base_commit])
        self._materialize_base_only(
            source=source,
            base_commit=base_commit,
            row=row,
            dest=dest,
        )

    def _materialize_base_only(
        self,
        *,
        source: Path,
        base_commit: str,
        row: dict[str, Any],
        dest: Path,
    ) -> None:
        if dest.exists():
            shutil.rmtree(dest)
        git(["init", "--quiet", "--initial-branch=main", str(dest)])
        git(
            [
                "-c",
                "protocol.file.allow=always",
                "fetch",
                "--quiet",
                "--depth=1",
                "--no-tags",
                str(source),
                base_commit,
            ],
            cwd=dest,
            timeout_s=1200,
        )
        git(["checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=dest)

        # Replace the fetched repository metadata with a synthetic one-commit
        # baseline. This hides the original base SHA as well as later commits;
        # the working tree contains only tracked files because the source cache
        # is cleaned before materialization.
        shutil.rmtree(dest / ".git")
        git(["init", "--quiet", "--initial-branch=main"], cwd=dest)
        git(["add", "--all", "--force"], cwd=dest, timeout_s=1200)
        git(
            [
                "-c",
                "user.name=SWE-bench Pro",
                "-c",
                "user.email=swebench-pro@invalid",
                "commit",
                "--quiet",
                "--no-gpg-sign",
                "-m",
                "clean base snapshot",
            ],
            cwd=dest,
            timeout_s=1200,
        )

        alternates = dest / ".git" / "objects" / "info" / "alternates"
        if alternates.exists():
            raise RuntimeError(
                f"clean SWE-bench Pro checkout unexpectedly uses alternates: {dest}"
            )
        if git(["rev-list", "--count", "HEAD"], cwd=dest).strip() != "1":
            raise RuntimeError(
                f"clean SWE-bench Pro checkout contains more than the base commit: {dest}"
            )

        if self._source._has_commit(dest, base_commit):
            raise RuntimeError(
                f"clean SWE-bench Pro checkout leaked original base commit: {dest}"
            )

        # The internal instance ID and grader setup command can contain the
        # gold commit SHA. Assert that no such commit survived the rewrite.
        forbidden: set[str] = set()
        for key in ("instance_id", "before_repo_set_cmd"):
            value = str(row.get(key) or "")
            forbidden.update(
                token.lower()
                for token in re.findall(
                    r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])", value, re.I
                )
                if token.lower() != base_commit.lower()
            )
        leaked = sorted(
            commit
            for commit in forbidden
            if self._source._has_commit(dest, commit)
        )
        if leaked:
            raise RuntimeError(
                "clean SWE-bench Pro checkout leaked non-base commits: "
                + ", ".join(leaked)
            )
