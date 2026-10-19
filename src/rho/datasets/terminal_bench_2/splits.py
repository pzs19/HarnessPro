from __future__ import annotations

import hashlib


def split_task_ids(task_ids: list[str], *, seed: int = 0) -> dict[str, list[str]]:
    """Deterministic 30/rest train/val split by hash of (seed, id).

    Sized for TB2's 89 tasks → 30 train / 59 held-out val / 0 test.
    Evolution proposals and candidate acceptance use only train; ``rho
    evolve`` evaluates the frozen final harness on val afterward. Test is
    empty because the 59-task val split is the formal held-out set here.
    """
    if not task_ids:
        return {"train": [], "val": [], "test": []}
    ordered = sorted(task_ids, key=lambda task_id: _digest(seed, task_id))
    n = len(ordered)
    n_train = min(30, n)
    return {
        "train": ordered[:n_train],
        "val": ordered[n_train:],
        "test": [],
    }


def _digest(seed: int, task_id: str) -> str:
    return hashlib.sha256(f"{seed}:{task_id}".encode("utf-8")).hexdigest()
