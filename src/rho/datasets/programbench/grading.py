"""Verifier-only provenance, budget and completeness checks. No Solver inputs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

GRADING_CONTRACT = "programbench-complete-grading-v12-20260913"
BRANCH_TIMEOUT_S = 14_400


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def process_budget(branches: int, workers: int, retries: int, branch_timeout_s: int,
                   toolchain_count: int = 0) -> int:
    """Bound the complete process, not just one test execution.

    Allow 7200s for build/dependency preparation; 7200s for each branch's
    setup and 1800s for collection/cleanup, on every possible branch attempt.
    Test-level retries already run *within* the branch test budget.
    """
    if branches < 1 or workers < 1 or retries < 0 or toolchain_count < 0 or branch_timeout_s < BRANCH_TIMEOUT_S:
        raise ValueError("invalid grading limits (branch budget must be at least 14400s)")
    return 7200 + toolchain_count * 3606 + math.ceil(branches / workers) * (retries + 1) * (
        7200 + branch_timeout_s + 1800
    )


def reconcile_xml(
    raw: dict[str, Any],
    active: dict[tuple[str, str], dict[str, Any]],
    branches: set[str],
    ignored: set[str],
    prerequisite_metadata: dict | None = None,
    reviewed_terminal_skips: dict | None = None,
) -> str | None:
    from programbench.eval.eval import count_worker_crashes, parse_test_results

    # Associate each selected XML with its own execution, not an abandoned
    # earlier branch attempt. pytest's internal errors/interruptions must not
    # become valid grades merely because an XML happens to contain every ID.
    selected_runs = {}
    pending_run = None
    for entry in raw.get('log', []):
        if entry.get('step') == 'run_tests':
            pending_run = entry
        elif entry.get('step') == 'results_read' and entry.get('returncode') == 0 and entry.get('branch'):
            selected_runs[entry['branch']] = pending_run
            pending_run = None
    for branch in branches:
        execution = selected_runs.get(branch)
        if execution and (execution.get('returncode') not in (None, 0, 1)
                          or any(line.startswith('INTERNALERROR>') for line in execution.get('output', '').splitlines())):
            return 'selected branch execution has pytest internal error or interruption'

    logs = {
        entry["branch"]: entry
        for entry in raw.get("log", [])
        if entry.get("step") == "results_read" and entry.get("returncode") == 0
        and entry.get("branch")
    }
    if not branches or branches - logs.keys():
        return "one or more expected branches have no collected JUnit XML"
    xml_tests = {}
    for branch in branches:
        xml = logs[branch].get("output", "")
        if count_worker_crashes(xml):
            return "collected JUnit XML contains a worker crash"
        try:
            parsed = parse_test_results(xml, branch=branch)
        except Exception as exc:
            return f"collected JUnit XML cannot be parsed: {type(exc).__name__}"
        for test in parsed.test_results:
            if f"{branch}/{test.name}" not in ignored:
                xml_tests[(branch, test.name)] = test.status
    if xml_tests != {key: value.get("status") for key, value in active.items()}:
        return "collected JUnit XML and evaluation outcomes do not reconcile"
    skip_issue = reconcile_skips(raw, active, prerequisite_metadata, reviewed_terminal_skips)
    if skip_issue:
        return skip_issue
    return None


def reconcile_skips(raw: dict, active: dict, prerequisite_metadata: dict | None = None,
                    reviewed_terminal_skips: dict | None = None) -> str | None:
    """A skip is not a failure; accept only a proven official prerequisite skip.

    This leaves all tests in the denominator. Missing packages, unknown
    prerequisites, cross-worker state loss and arbitrary skip reasons are
    incomplete grading, never inferred model failures.
    """
    manifests = {e['branch']: json.loads(e['output']) for e in raw.get('log', [])
                 if e.get('step') == 'declared_pytest_prerequisites' and e.get('returncode') == 0}
    for branch, graph in (prerequisite_metadata or {}).items():
        if branch in manifests and manifests[branch] != graph:
            return 'prerequisite metadata does not match frozen official grader archive'
        manifests[branch] = graph
    for (branch, name), result in active.items():
        if result.get('status') != 'skipped':
            continue
        if (branch, name) in (reviewed_terminal_skips or {}):
            if (result.get('extra') or {}).get('message') == reviewed_terminal_skips[(branch, name)]:
                continue
            return 'reviewed terminal skip cause differs from current evaluation'
        graph = manifests.get(branch, [])
        def outcome(node):
            suffix = node['module']+'.'+node['function']
            matches = [v for (b,n),v in active.items() if b == branch
                       and (n == suffix or suffix.endswith('.'+n))]
            return matches[0] if len(matches) == 1 else None
        def failed_prerequisite(node, seen):
            identity = (node['module'], node['alias'])
            if identity in seen:
                return False
            value = outcome(node)
            if not value:
                return False
            if value.get('status') == 'failure':
                return True
            if value.get('status') == 'error':
                message = str((value.get('extra') or {}).get('message') or '')
                return ('AssertionError' in message or 'Failed: Timeout' in message)
            if value.get('status') != 'skipped':
                return False
            msg = str((value.get('extra') or {}).get('message') or '')
            for dep in node['depends']:
                if not msg.endswith(' depends on '+dep):
                    continue
                edges = [d for d in graph if d['module'] == node['module'] and d['alias'] == dep]
                if len(edges) == 1 and failed_prerequisite(edges[0], seen | {identity}):
                    return True
            return False
        nodes = [n for n in graph if outcome(n) is result]
        if len(nodes) != 1 or not failed_prerequisite(nodes[0], set()):
            return 'unresolved nonignored skipped test: grader prerequisite or dependency incomplete'
    return None


def stage_times(raw: dict[str, Any]) -> dict[str, float]:
    totals = {"dependency_preparation_s": 0.0, "test_execution_s": 0.0,
              "result_collection_s": 0.0, "other_grader_steps_s": 0.0}
    for entry in raw.get("log", []):
        step = str(entry.get("step") or "")
        if step == "run_tests":
            key = "test_execution_s"
        elif step == "results_read":
            key = "result_collection_s"
        elif any(part in step for part in ("dependencies", "preflight", "prefetch", "toolchain", "proxy")):
            key = "dependency_preparation_s"
        else:
            key = "other_grader_steps_s"
        totals[key] += float(entry.get("wall_time") or 0.0)
    return {key: round(value, 3) for key, value in totals.items()}


def evaluator_sources(package: Path, repo: Path) -> dict[str, str]:
    files = {"programbench/" + str(p.relative_to(package)): sha256(p)
             for p in sorted(package.rglob("*"))
             if p.is_file() and p.suffix in {".py", ".json", ".yaml"}}
    for name in ("scripts/run_programbench_eval.py", "src/rho/datasets/programbench/results.py",
                 "src/rho/datasets/programbench/grading.py", "src/rho/datasets/programbench/skip_proofs.py"):
        files[name] = sha256(repo / name)
    return files


def apply_reference_collections(
    expected: dict[str, dict[str, list[str]]], manifest: Path,
    archive_hashes: dict[str, str],
    ignored_tests: dict[str, set[str]] | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Use independently collected official tests, never candidate outcomes.

    An override may add omitted official cases, but must retain every prior
    metadata case. Archive hashes tie it to this exact frozen test package.
    Unexplained deletions or failed/partial collection are rejected.
    """
    metadata = json.loads(manifest.read_text())
    if metadata.get("schema") != 1:
        raise ValueError("unknown reference collection manifest schema")
    output = {task: {branch: list(names) for branch, names in branches.items()}
              for task, branches in expected.items()}
    seen = set()
    for record in metadata["collections"]:
        path = Path(record["path"])
        if sha256(path) != record["sha256"]:
            raise ValueError("reference collection record changed")
        result = json.loads(path.read_text())
        task, branch = result["task_id"], result["branch"]
        if (task, branch) in seen:
            raise ValueError("duplicate reference collection")
        seen.add((task, branch))
        if task not in output:
            continue
        if (not result.get("collectonly") or result.get("exitstatus") != 0
                or result.get("collection_errors") or result.get("tests_executed") != 0
                or result.get("solver_calls") != 0 or not result.get("reference_executable_sha256")):
            raise ValueError("reference test collection was incomplete or executed candidate tests")
        if archive_hashes.get(f"{task}/tests/{branch}.tar.gz") != result["archive_sha256"]:
            raise ValueError("reference collection uses a different hidden test archive")
        names = result["tests"]
        if len(names) != len(set(names)) or branch not in output[task]:
            raise ValueError("duplicate tests or unknown reference branch")
        missing = set(output[task][branch]) - set(names)
        ignored = (ignored_tests or {}).get(task, set())
        if any(f"{branch}/{name}" not in ignored for name in missing):
            raise ValueError("reference collection would remove expected tests; investigate instead")
        # Officially ignored collection-error pseudo-cases can disappear once
        # dependencies are installed. Preserve their metadata/ignore policy;
        # they never enter the active denominator. No active test is removed.
        output[task][branch] = list(dict.fromkeys([*output[task][branch], *names]))
    return output
