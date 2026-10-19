from pathlib import Path

PYPI_PACKAGE_NAME = "datacurve-pier"
CACHE_DIR = Path("~/.cache/pier").expanduser()
TASK_CACHE_DIR = CACHE_DIR / "tasks"
PACKAGE_CACHE_DIR = CACHE_DIR / "tasks" / "packages"
ORG_NAME_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*$"
