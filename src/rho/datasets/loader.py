from __future__ import annotations

import os
from pathlib import Path

from rho.datasets.directory import DirectoryDataset
from rho.datasets.swebench_pro import SWEbenchProDataset
from rho.protocols import Dataset, HarnessStore


def load_dataset(
    spec: str,
    *,
    harness_store: HarnessStore,
    max_per_split: int | None = None,
    docker_pull: str = "missing",
    difficulty_filter: tuple[str, ...] | None = None,
) -> Dataset:
    if ":" in spec:
        scheme, _, payload = spec.partition(":")
    else:
        scheme, payload = "directory", spec

    if scheme == "directory":
        path = Path(payload).resolve()
        if not path.exists():
            raise ValueError(
                f"Dataset path {payload!r} does not exist. "
                f"If this is a HuggingFace dataset, use a scheme prefix, "
                f"e.g. 'swebench-pro:{payload}'."
            )
        return DirectoryDataset(path, harness_store=harness_store)
    if scheme == "swebench-pro":
        cache_root = os.environ.get("RHO_SWEBENCH_CACHE_ROOT")
        if cache_root:
            from rho.datasets.swebench_pro.repo_cache import (
                FULL_HISTORY_CACHE_DIR,
                RepoCache,
            )

            resolved_cache = Path(cache_root).expanduser().resolve()
            return SWEbenchProDataset(
                payload,
                harness_store=harness_store,
                max_per_split=max_per_split,
                docker_pull=docker_pull,
                repo_cache=RepoCache(resolved_cache / FULL_HISTORY_CACHE_DIR),
                eval_assets_root=resolved_cache / "SWE-bench_Pro-os",
            )
        return SWEbenchProDataset(
            payload,
            harness_store=harness_store,
            max_per_split=max_per_split,
            docker_pull=docker_pull,
        )
    if scheme == "terminal-bench-2":
        from rho.datasets.terminal_bench_2 import TerminalBench2Dataset

        path = Path(payload).resolve()
        return TerminalBench2Dataset(
            path,
            harness_store=harness_store,
            max_per_split=max_per_split,
            docker_pull=docker_pull,
            difficulty_filter=difficulty_filter,
        )
    raise ValueError(f"Unknown dataset scheme: {scheme!r}")
