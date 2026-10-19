#!/usr/bin/env python3
"""Restore executable bits that anonymous-site ZIP downloads may omit."""

from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    executables = [
        root / "third_party/rtk-bin/rtk",
        root / "scripts/swebench-pro-toolchain-bin/go",
        root / "scripts/run_programbench_codex_task.sh",
        *(
            root / "harnesses/swe-bench-pro" / mode / "bin/repair-verify"
            for mode in ("lite", "medium", "heavy")
        ),
    ]
    missing = [str(path.relative_to(root)) for path in executables if not path.is_file()]
    if missing:
        raise SystemExit("Release is missing runtime files: " + ", ".join(missing))
    for path in executables:
        path.chmod(path.stat().st_mode | 0o111)
        print(f"Ready: {path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
