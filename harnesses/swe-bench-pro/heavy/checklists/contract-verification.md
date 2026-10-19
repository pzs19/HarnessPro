# Contract Verification Checklist

Use this as a final gate for each repair. High-severity misses in prior runs
came from not executing the exact prompt contract.

## Requirement Ledger

- Copy exact interface names from the prompt: type/function/method names,
  struct fields, route paths, config keys, defaults, error strings, and payload
  shapes.
- Mark each requirement with one of: implemented, intentionally unchanged with
  reason, or blocked with observed evidence.
- Re-check literal wording before finalizing. Tests often compile against exact
  public identifiers, so compare every name character-for-character with the
  current task text instead of relying on memory.

## Behavioral Smoke Tests

- Exercise every new public entry point directly, not only syntax or import
  checks. For web/API work, call the controller or route layer and assert the
  response body shape and error translation.
- Cover negative paths from the prompt: missing records, denied privileges,
  full queues, closed emitters, deleted resources, unsupported platforms,
  expired deadlines, failed backends, and boundary positions.
- For state machines, assert transitions at start, middle, end, stopped,
  paused/playing, and invalid input boundaries.
- For data import/parsing tasks, compare complete real fixture outputs against
  expected outputs. Synthetic examples are useful only after at least one real
  regression fixture passes exactly.
- For filesystem/data-cleanup tasks, trace generated or transformed artifacts.
  If the stored record is normalized, test the visible/generated filename too.
- For configurable behavior, test the runtime config value shape used by the
  project, including string values like `"0"` and `"1"` when applicable.
- For multi-backend features, smoke every named backend/path and every explicit
  selector value. Do not validate only the backend that is easiest to run.

## Cross-layer Completeness

- Map every public requirement to a production path and a focused executable
  check before finalizing.
- When a public interface crosses schemas, generated bindings, adapters,
  persistence, or compatibility branches, inspect each boundary that can alter
  the requested behavior.
- Diagnose concrete test failures before broadening the patch. Do not infer
  hidden-test requirements or add behavior unsupported by the public contract.
- Re-run the narrowest task-shaped check after the last production edit, then
  inspect the final diff for omitted public requirements and unrelated changes.

## Language/Repo Recipes

- Go: check `PATH`, `GOROOT`, `GOPATH`, and any configured extra tool
  directories before declaring Go unavailable. Run gofmt on touched files,
  focused package tests, and compile cross-platform build-tag paths when the
  prompt mentions a platform.
- Python: run `python -m py_compile` or `compileall` on touched files, then a
  direct smoke for changed behavior. Remove `__pycache__`, `.pyc`, `.pytest_cache`,
  and test environment artifacts before finalizing.
- JavaScript/TypeScript: syntax/type checks do not catch missing runtime imports
  or wrong response shapes. Add a small Node smoke with stubs for changed app,
  controller, or model methods when full dependencies are missing.
- Missing-file tasks: search local/upstream history for the target file and its
  tests. Run the narrow package or upstream tests after adapting the file.

## Final Diff Hygiene

- Run `git diff --check`.
- Run `git status --short --untracked-files=all` and inspect untracked files.
- Remove generated caches, local smoke files, downloaded fixtures, virtualenvs,
  node caches, coverage output, and temporary logs unless the prompt explicitly
  asks to add them.
- If any normal test cannot run, state the exact command and the observed reason,
  then compensate with the closest direct smoke you can execute.
