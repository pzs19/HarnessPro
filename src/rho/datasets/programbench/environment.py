"""Docker mount and artifact handling for ProgramBench agent binaries."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from pier.environments.docker.docker import DockerEnvironment
from pier.models.trial.config import ServiceVolumeConfig


class MountedBinaryDockerEnvironment(DockerEnvironment):
    """Append configured binary mounts without dropping Pier log mounts.

    Pier treats explicit ``environment.mounts`` as a complete replacement for
    its verifier, agent, and artifact log mounts. ProgramBench uses explicit
    read-only mounts for version-pinned agent binaries, so replacing the log
    mounts would make completed containers unauditable. Merge by container
    target instead; an explicit mount wins if it uses the same target.
    """

    @staticmethod
    def merge_mounts(
        defaults: list[ServiceVolumeConfig],
        configured: list[ServiceVolumeConfig],
    ) -> list[ServiceVolumeConfig]:
        merged = {mount["target"]: mount for mount in defaults}
        for mount in configured:
            merged[mount["target"]] = mount
        return list(merged.values())

    @classmethod
    def select_mounts(
        cls,
        defaults: list[ServiceVolumeConfig],
        configured: list[ServiceVolumeConfig],
    ) -> list[ServiceVolumeConfig]:
        """Add Agent binary mounts without widening verifier-only mounts.

        Pier deliberately gives a separate Verifier only its verifier-log
        mount.  Re-adding the Agent artifact mount there aliases the transfer
        source and destination; clearing the fresh Verifier artifact directory
        then deletes the Agent's collected patch on the host.  A configured
        set containing only standard log targets is therefore authoritative.
        Any non-log target identifies the Agent-side binary-mount extension
        and is merged with the defaults.
        """

        if not configured:
            return defaults
        default_targets = {mount["target"] for mount in defaults}
        configured_targets = {mount["target"] for mount in configured}
        if configured_targets <= default_targets:
            return configured
        return cls.merge_mounts(defaults, configured)

    def __init__(
        self,
        *args: Any,
        mounts_json: list[ServiceVolumeConfig] | None = None,
        **kwargs: Any,
    ) -> None:
        configured_mounts = list(mounts_json or [])
        super().__init__(*args, mounts_json=None, **kwargs)
        self._mounts_json = self.select_mounts(
            list(self._mounts_json or []), configured_mounts
        )

    def _bind_mounted_host_path(self, environment_path: str) -> Path | None:
        """Resolve a container path through the environment's bind mounts.

        Pier's artifact collector normally copies files from the container to
        the trial directory.  For bind-mounted logs that destination can be
        the exact same host file as the container source.  Asking Docker to
        copy a file onto itself can remove or truncate it, so callers need to
        recognize that case before invoking ``docker compose cp``.
        """

        candidate = PurePosixPath(environment_path)
        best_match: tuple[int, Path] | None = None
        for mount in self._mounts_json or []:
            if mount.get("type") != "bind":
                continue
            target = PurePosixPath(str(mount["target"]))
            try:
                relative = candidate.relative_to(target)
            except ValueError:
                continue
            host_path = Path(str(mount["source"])).joinpath(*relative.parts)
            match = (len(target.parts), host_path)
            if best_match is None or match[0] > best_match[0]:
                best_match = match
        return best_match[1] if best_match is not None else None

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        mounted_source = self._bind_mounted_host_path(source_path)
        target = Path(target_path)
        if (
            mounted_source is not None
            and mounted_source.resolve() == target.resolve()
        ):
            await self._chown_to_host_user(source_path)
            if not mounted_source.is_file():
                raise FileNotFoundError(source_path)
            return
        await super().download_file(source_path, target_path)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        mounted_source = self._bind_mounted_host_path(source_dir)
        target = Path(target_dir)
        if (
            mounted_source is not None
            and mounted_source.resolve() == target.resolve()
        ):
            await self._chown_to_host_user(source_dir, recursive=True)
            if not mounted_source.is_dir():
                raise FileNotFoundError(source_dir)
            return
        await super().download_dir(source_dir, target_dir)

    async def start(self, force_build: bool) -> None:
        """Roll back partial Compose state when environment startup fails."""

        try:
            await super().start(force_build=force_build)
        except Exception:
            # Compose can create the default network and then fail while
            # allocating a second network.  Pier's normal cleanup only runs
            # after a successful start, so repeated Infra retries otherwise
            # leak subnets until Docker exhausts its address pools.
            try:
                await self._run_docker_compose_command(
                    ["down", "--remove-orphans"]
                )
            except Exception as cleanup_error:
                self.logger.warning(
                    "Failed to roll back partial Docker Compose startup: %s",
                    cleanup_error,
                )
            self._cleanup_resources_compose_file()
            raise
