"""Pier agent adapters for ProgramBench cleanroom images."""

from __future__ import annotations

from pier.agents.installed.codex import Codex
from pier.environments.base import BaseEnvironment


class MountedProgramBenchCodex(Codex):
    """Run the pinned Codex binary in the ProgramBench cleanroom workspace."""

    def install_spec(self) -> None:
        """Use the mounted binary without installing another copy in the image."""
        return None

    def build_cli_flags(self) -> str:
        """Trust only the run-scoped Harness hooks mounted by this adapter.

        Codex 0.147.0 does not execute hooks from a generated config in a
        non-interactive container unless ``--dangerously-bypass-hook-trust`` is
        present.  ProgramBench mounts the Harness read-only and constructs the
        config itself, so requiring persisted interactive trust would silently
        disable the budget policy.  Keep the bypass local to ProgramBench runs
        that actually contain hook configuration.
        """

        flags = super().build_cli_flags()
        if self._config_toml and "[[hooks." in self._config_toml:
            flags = f"{flags} --dangerously-bypass-hook-trust".strip()
        return flags

    async def setup(self, environment: BaseEnvironment) -> None:
        # ProgramBench runs the agent as the unprivileged ``agent`` user while
        # Pier creates the bind-mounted host log directory as the runner user.
        # Shared storage may root-squash container-side chown, so make this
        # trial-private host directory writable before starting the CLI.
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.chmod(0o777)

        pin_proxy = getattr(environment, "pin_egress_proxy_ip", None)
        if pin_proxy is not None:
            await pin_proxy()

        command = self.get_version_command()
        if command:
            result = await environment.exec(command=command)
            if result.return_code != 0:
                raise RuntimeError(
                    "Mounted Codex binary failed its version probe: "
                    f"{result.stderr or result.stdout or result.return_code}"
                )
            if self._version is None:
                self._version = self.parse_version(result.stdout or "")

        identity = await environment.exec(
            "git config --global user.name 'RHO Evaluation Agent' && "
            "git config --global user.email 'agent@rho.invalid' && "
            "git config --global --add safe.directory /workspace"
        )
        if identity.return_code != 0:
            raise RuntimeError(
                "Could not configure the ProgramBench Git identity: "
                f"{identity.stderr or identity.stdout or identity.return_code}"
            )
