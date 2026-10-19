"""Distinguish completed assertion failures from failed verifier setup."""

from __future__ import annotations

import json
from pathlib import Path


def completed_assertion_failure(path: Path) -> bool:
    try:
        results = json.loads(path.read_text())["results"]
        summary = results["summary"]
        tests = results["tests"]
        failed = [test for test in tests if test.get("status") == "failed"]
        if (not failed or summary.get("tests") != len(tests)
                or summary.get("failed") != len(failed)
                or summary.get("pending", 0) != 0 or summary.get("other", 0) != 0
                or any(test.get("status") not in {"passed", "failed", "skipped"} for test in tests)):
            return False
        dependency_errors = (
            "filenotfounderror", "modulenotfounderror", "importerror", "connectionerror",
            "timeouterror", "calledprocesserror", "webdriverexception", "network is unreachable",
            "could not resolve host", "failed to download", "unable to obtain driver",
        )
        return all("assertionerror" in (text := json.dumps(test).lower())
                   and not any(marker in text for marker in dependency_errors) for test in failed)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False
