from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rho.protocols import Harness

EMPTY_HARNESS_ID = "h_empty"
EMPTY_HARNESS_README = (
    "this is the harness directory; put persistent context, tools, notes here\n"
)


class HarnessCollision(RuntimeError):
    pass


@dataclass(frozen=True)
class FilesystemHarness:
    _root: Path
    _id: str

    @property
    def id(self) -> str:
        return self._id

    def materialize(self, dest: Path) -> None:
        shutil.copytree(self._root / self._id, dest, dirs_exist_ok=True)


class FilesystemHarnessStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def empty(self) -> Harness:
        empty_dir = self.root / EMPTY_HARNESS_ID
        if not empty_dir.exists():
            empty_dir.mkdir(parents=True, exist_ok=True)
            (empty_dir / "README.md").write_text(EMPTY_HARNESS_README, encoding="utf-8")
        return FilesystemHarness(self.root, EMPTY_HARNESS_ID)

    def capture(self, src: Path) -> Harness:
        self.empty()
        digest = _tree_digest(src)
        empty_digest = _tree_digest(self.root / EMPTY_HARNESS_ID)
        version_id = EMPTY_HARNESS_ID if digest == empty_digest else f"h_{digest[:12]}"
        version_dir = self.root / version_id
        if version_dir.exists():
            existing_digest = _tree_digest(version_dir)
            if existing_digest != digest:
                raise HarnessCollision(
                    f"Hash prefix collision for {version_id}: {existing_digest} != {digest}"
                )
            return FilesystemHarness(self.root, version_id)

        with tempfile.TemporaryDirectory(dir=str(self.root), prefix=".capture_") as tmp:
            tmp_dir = Path(tmp) / version_id
            shutil.copytree(src, tmp_dir)
            try:
                tmp_dir.rename(version_dir)
            except OSError:
                if version_dir.exists():
                    existing_digest = _tree_digest(version_dir)
                    if existing_digest != digest:
                        raise HarnessCollision(
                            f"Hash prefix collision for {version_id}: "
                            f"{existing_digest} != {digest}"
                        )
                else:
                    raise
        return FilesystemHarness(self.root, version_id)

    def get(self, harness_id: str) -> Harness:
        path = self.root / harness_id
        if not path.exists():
            raise KeyError(harness_id)
        return FilesystemHarness(self.root, harness_id)


def _tree_digest(root: Path) -> str:
    lines: list[str] = []
    if not root.exists():
        return hashlib.sha256(b"").hexdigest()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{rel}:{digest}")
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
