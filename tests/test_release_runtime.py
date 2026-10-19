"""Offline regression checks for the supported release paths."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from rho.harness_runtime import load_runtime_policy


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "src/rho/agent/claude_code_hook_adapter.py"
CORE_HOOK = ROOT / "harnesses/terminal-bench-2/medium/runtime/hook.py"


class ReleaseEntryPoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.enterClassContext(patch.dict(os.environ, {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        }))
        cls.swe = runpy.run_path(str(ROOT / "scripts/run_swebench_pro_harness_eval.py"))
        cls.tb2 = runpy.run_path(str(ROOT / "scripts/run_tb2_harness_eval.py"))

    def parse_swe(self, *extra: str):
        args = [
            "run_swebench_pro_harness_eval.py",
            "--dataset", "unused.parquet", "--heldout-ids", "unused.txt",
            "--run-dir", "unused-run", "--harness-label", "medium",
            "--codex-config", "unused.toml", "--docker-command", "/usr/bin/docker",
            "--cache-root", "unused-cache", *extra,
        ]
        with patch.object(sys, "argv", args):
            return self.swe["parse_args"]()

    def test_model_specific_timeout_and_explicit_override(self) -> None:
        self.assertEqual(self.parse_swe().solver_timeout_s, 1200)
        self.assertEqual(self.parse_swe("--model", "GLM-5.2").solver_timeout_s, 7200)
        self.assertEqual(
            self.parse_swe("--model", "glm-5.2", "--solver-timeout-s", "42").solver_timeout_s,
            42,
        )

    def test_all_released_policies_load_with_their_referenced_files(self) -> None:
        policies = sorted((ROOT / "harnesses").glob("*/*/runtime/policy.json"))
        self.assertEqual(len(policies), 9)
        for source in policies:
            with self.subTest(source=source.relative_to(ROOT)):
                self.assertIsNotNone(load_runtime_policy(source.parent.parent))

    def test_removed_backends_are_rejected_by_both_runners(self) -> None:
        base = [
            "runner.py", "--dataset", "unused", "--heldout-ids", "unused",
            "--run-dir", "unused", "--harness-label", "medium",
            "--codex-config", "unused",
        ]
        for namespace in (self.tb2, self.swe):
            for backend in ("hermes", "openclaw", "deepseek-harness"):
                with self.subTest(backend=backend), patch.object(
                    sys, "argv", [*base, "--agent-backend", backend]
                ), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as result:
                        namespace["parse_args"]()
                    self.assertEqual(result.exception.code, 2)


class ClaudeHookCompatibility(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.state = self.workspace / "state"
        self.selection = self.workspace / "selection.json"
        self.selection.write_text(json.dumps({
            "session_context": "Follow the public task description.",
            "finalize_after_tools": 1,
            "rtk_enabled": False,
        }))
        self.env = {
            **os.environ,
            "RHO_CLAUDE_CORE_HOOK": str(CORE_HOOK),
            "RHO_RUNTIME_INTERFACE": "claude-code",
            "RHO_RUNTIME_STATE_ROOT": str(self.state),
            "RHO_RUNTIME_SELECTION_PATH": str(self.selection),
            "RHO_RUNTIME_AUDIT_PATH": str(self.workspace / "audit.jsonl"),
            "RHO_CLAUDE_POLICY_DENIAL_HALT_AFTER": "2",
        }

    def invoke(self, event: str, tool: str = "Bash", **fields):
        payload = {
            "hook_event_name": event,
            "cwd": str(self.workspace),
            "tool_name": tool,
            "tool_input": {"command": "git status"},
            **fields,
        }
        result = subprocess.run(
            [sys.executable, "-B", str(ADAPTER)],
            input=json.dumps(payload), capture_output=True, text=True,
            env=self.env, check=True, timeout=10,
        )
        return json.loads(result.stdout) if result.stdout.strip() else {}

    def test_budget_denial_and_repeated_denial_stop_are_preserved(self) -> None:
        first = self.invoke("PreToolUse")
        self.assertNotEqual(first.get("hookSpecificOutput", {}).get("permissionDecision"), "deny")
        denied = self.invoke("PreToolUse")
        self.assertEqual(denied["hookSpecificOutput"]["permissionDecision"], "deny")
        stopped = self.invoke("PreToolUse")
        self.assertIs(stopped["continue"], False)
        self.assertIn("tool budget is exhausted", stopped["stopReason"].lower())

    def test_read_normalization_cannot_override_a_budget_denial(self) -> None:
        allowed = self.invoke(
            "PreToolUse", "Read", tool_input={"file_path": "README.md", "pages": ""}
        )
        self.assertEqual(allowed["hookSpecificOutput"]["updatedInput"], {"file_path": "README.md"})
        denied = self.invoke(
            "PreToolUse", "Read", tool_input={"file_path": "README.md", "pages": ""}
        )
        self.assertEqual(denied["hookSpecificOutput"]["permissionDecision"], "deny")
        self.assertNotIn("updatedInput", denied["hookSpecificOutput"])

    def test_structured_images_remain_native_outputs(self) -> None:
        result = self.invoke(
            "PostToolUse", "Read",
            tool_response={"type": "image", "source": {"type": "base64", "data": "image-fixture"}},
        )
        self.assertEqual(result, {})

    def test_static_harness_does_not_require_a_core_hook(self) -> None:
        self.env["RHO_CLAUDE_CORE_HOOK"] = ""
        self.assertEqual(self.invoke("SessionStart"), {})


if __name__ == "__main__":
    unittest.main()
