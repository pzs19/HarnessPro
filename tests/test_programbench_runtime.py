"""Offline checks for the self-contained ProgramBench release path."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
from pathlib import Path
import re
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch


ROOT = Path(__file__).resolve().parents[1]


class ProgramBenchRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.enterClassContext(patch.dict(os.environ, {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        }))
        from rho.datasets.programbench.agents import MountedProgramBenchCodex
        from rho.datasets.programbench.environment import MountedBinaryDockerEnvironment
        from rho.datasets.programbench import results

        cls.agent_class = MountedProgramBenchCodex
        cls.environment_class = MountedBinaryDockerEnvironment
        cls.results = results

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_runner_and_shell_import_paths_load(self) -> None:
        runner = runpy.run_path(str(ROOT / "scripts/run_programbench_eval.py"))
        self.assertTrue(callable(runner["parse_args"]))
        launcher = (ROOT / "scripts/run_programbench_codex_task.sh").read_text()
        paths = re.findall(r"--(?:agent|environment)-import-path\s+(\S+)", launcher)
        self.assertEqual(len(paths), 2)
        for value in paths:
            module, name = value.split(":")
            self.assertTrue(isinstance(getattr(importlib.import_module(module), name), type))

    def test_mounted_codex_keeps_hook_flags_and_skips_installation(self) -> None:
        for config, expects_trust in ((None, False), ('[[hooks.PreToolUse]]', True)):
            with self.subTest(config=config):
                agent = self.agent_class(
                    logs_dir=self.root / "logs",
                    model_name="openai/gpt-5.5",
                    config_toml=config,
                )
                self.assertIsNone(agent.install_spec())
                self.assertEqual(
                    "--dangerously-bypass-hook-trust" in agent.build_cli_flags(),
                    expects_trust,
                )
                environment = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(
                    return_code=0, stdout="codex-cli 0.147.0", stderr="",
                )))
                asyncio.run(agent.setup(environment))
                self.assertEqual(agent.version(), "0.147.0")
                command = environment.exec.call_args.args[0]
                self.assertIn("safe.directory /workspace", command)

    def test_binary_mounts_preserve_logs_without_widening_verifier_mounts(self) -> None:
        defaults = [
            {"type": "bind", "source": str(self.root / name), "target": f"/logs/{name}"}
            for name in ("agent", "verifier", "artifacts")
        ]
        binary = {"type": "bind", "source": "/usr/bin/codex", "target": "/usr/local/bin/codex"}
        self.assertEqual(self.environment_class.select_mounts(defaults, [binary]), [*defaults, binary])
        verifier_only = [defaults[1]]
        self.assertEqual(self.environment_class.select_mounts(defaults, verifier_only), verifier_only)

    def test_downloading_existing_bind_artifacts_does_not_copy_onto_itself(self) -> None:
        artifact = self.root / "submission.tar.gz"
        artifact.write_bytes(b"collected workspace")
        environment = object.__new__(self.environment_class)
        environment._mounts_json = [{
            "type": "bind", "source": str(self.root), "target": "/logs/artifacts",
        }]
        environment._chown_to_host_user = AsyncMock()
        from pier.environments.docker.docker import DockerEnvironment

        with patch.object(DockerEnvironment, "download_file", new_callable=AsyncMock) as copy:
            asyncio.run(environment.download_file("/logs/artifacts/submission.tar.gz", artifact))
            copy.assert_not_called()
        self.assertEqual(artifact.read_bytes(), b"collected workspace")

    def test_solver_accounting_preserves_usage_and_infrastructure_classification(self) -> None:
        job = self.root / "run"
        trial = job / "trial"
        (trial / "agent").mkdir(parents=True)
        (trial / "artifacts").mkdir()
        (trial / "artifacts/submission.tar.gz").write_bytes(b"workspace")
        (job / "result.json").write_text(json.dumps({
            "finished_at": "2026-10-19T00:01:00Z", "n_total_trials": 1, "stats": {},
        }))
        (trial / "agent/codex.txt").write_text(json.dumps({
            "type": "turn.completed", "usage": {
                "input_tokens": 1000, "cached_input_tokens": 200, "output_tokens": 50,
            },
        }) + "\n")
        for exception, classification in (
            (None, "solver_ready"),
            ({"exception_type": "NonZeroAgentExitCodeError", "exception_message": "compilation failed"}, "solver_ready"),
            ({"exception_type": "NonZeroAgentExitCodeError", "exception_message": "cannot connect to the docker daemon"}, "infrastructure"),
        ):
            with self.subTest(exception=exception):
                (trial / "result.json").write_text(json.dumps({
                    "exception_info": exception,
                    "agent_execution": {
                        "started_at": "2026-10-19T00:00:00Z",
                        "finished_at": "2026-10-19T00:01:00Z",
                    },
                }))
                result = self.results.summarize_solver(
                    task_id="example", attempt=1, trace_id="example-1", job_dir=self.root,
                    pier_exit_code=0, proxy_log=None, model="gpt-5.5",
                )
                self.assertEqual(result["classification"], classification)
                self.assertEqual(result["usage"]["uncached_input_tokens"], 800)
                self.assertEqual(result["timing"]["agent_execution_s"], 60)
                totals = self.results.aggregate_attempts([result, result])
                self.assertEqual(totals["usage"]["input_tokens"], 2000)
                self.assertEqual(totals["usage"]["output_tokens"], 100)
                self.assertAlmostEqual(totals["cost"]["estimated_total_usd"], 2 * result["cost"]["estimated_total_usd"])


if __name__ == "__main__":
    unittest.main()
