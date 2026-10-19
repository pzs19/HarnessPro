"""Discover declared grader fixture toolchains; never inspect Solver behavior."""
from __future__ import annotations

import ast
from pathlib import Path
import re
import shlex
import tarfile
import warnings


GO_PATH_FILE = '/opt/programbench-grader-go-bin'
GO_PATH_PREFIX = 'if test -s /opt/programbench-grader-go-bin; then export PATH="$(cat /opt/programbench-grader-go-bin):$PATH"; fi; '

# Older Go launchers can interpret a language-only minimum such as `go 1.22`
# as an unavailable distribution named go1.22, even when go1.22.0 was already
# prefetched. Raise only an older default launcher to the declared minimum's
# real release. Never downgrade a newer installed toolchain or edit go.mod.
_GO_PREFLIGHT_SCRIPT = r'''
import os, pathlib, re, shutil, subprocess, sys
module = pathlib.Path('go.mod')
if not shutil.which('go') or not module.is_file():
    print('PROGRAMBENCH_GO_PREFLIGHT_NOT_REQUIRED')
    raise SystemExit(0)
env = dict(os.environ)
def command(args, toolchain):
    run = subprocess.run(['go', *args], env={**env, 'GOTOOLCHAIN':toolchain}, capture_output=True, text=True)
    if run.returncode:
        sys.stdout.write(run.stdout); sys.stderr.write(run.stderr)
        raise SystemExit(run.returncode)
    return run.stdout.strip()
local = command(['version'], 'local')
match = re.search(r'go(\d+)\.(\d+)(?:\.(\d+))?', local)
minimum = re.search(r'^\s*go\s+(\d+)\.(\d+)(?:\.(\d+))?\s*$',module.read_text(),re.M)
if match and minimum:
    version = lambda m: tuple(int(v or 0) for v in m.groups())
    if version(match) < version(minimum):
        parts = version(minimum)
        release = 'go' + '.'.join(map(str, parts if parts[1] >= 21 else parts[:2]))
        root = pathlib.Path(command(['env','GOROOT'],release))
        binary_dir = root/'bin'
        if not (binary_dir/'go').is_file():
            raise SystemExit('prepared Go distribution has no executable')
        env['PATH'] = str(binary_dir)+os.pathsep+env['PATH']
        print('PROGRAMBENCH_GO_DEFAULT_UPGRADE', local, '->', release)
        # File is confined to this fresh grader container, never Solver/global.
        pathlib.Path(os.environ.get('PROGRAMBENCH_GO_PATH_FILE','/opt/programbench-grader-go-bin')).write_text(str(binary_dir)+'\n')
print(command(['version'],'auto'))
'''
GO_PREFLIGHT_COMMAND = 'python3 -c ' + shlex.quote(_GO_PREFLIGHT_SCRIPT)


def declared_go_toolchains(archives: list[Path]) -> list[str]:
    versions = set()
    pattern = re.compile(r'(?:^|\n|\\n)\s*(?:go\s+|toolchain\s+go)(1\.\d+(?:\.\d+)?)\s*(?=$|\n|\\n|\\r)')

    def record(value, stable_before=None):
        for match in pattern.finditer(value):
            if stable_before is not None and match.end() >= stable_before:
                continue
            version = match.group(1)
            parts = [int(p) for p in version.split('.')]
            if parts[1] >= 21:
                versions.add('go' + version + ('.0' if len(parts) == 2 else ''))

    for path in archives:
        with tarfile.open(path, mode='r|gz') as bundle:
            for member in bundle:
                if not member.isfile() or not member.name.endswith((".py", ".sh", "go.mod", "go.work")):
                    continue
                if member.size > 2_000_000:
                    # Official test files can contain multi-megabyte embedded
                    # data. Scan declarations with bounded memory rather than
                    # rejecting the entire project or constructing a huge AST.
                    stream = bundle.extractfile(member)
                    carry = ''
                    while chunk := stream.read(65536):
                        text = carry + chunk.decode(errors='replace')
                        record(text, len(text) - 4096)
                        carry = text[-4096:]
                    record(carry)
                    continue
                source = bundle.extractfile(member).read().decode(errors="replace")
                values = [source]
                if member.name.endswith(".py"):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', SyntaxWarning)
                            tree = ast.parse(source)
                        values.extend(node.value for node in ast.walk(tree)
                                      if isinstance(node, ast.Constant) and isinstance(node.value, str))
                    except SyntaxError:
                        # The actual test interpreter may support syntax newer
                        # than the collector. Raw literal lines remain visible.
                        pass
                for value in values:
                    record(value)
    return sorted(versions, key=lambda v: tuple(int(p) for p in v[2:].split(".")))
