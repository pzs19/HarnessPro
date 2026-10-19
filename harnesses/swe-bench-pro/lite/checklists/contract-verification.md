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

- Exercise each public entry point, success condition, failure condition, and
  edge case explicitly named by the prompt; syntax or import checks alone are
  insufficient when executable behavior can be observed.
- Follow the requested behavior through every implementation boundary that the
  public contract names. Verify each named variant rather than substituting an
  easier adjacent path.
- Use the narrowest realistic local input that exercises the changed behavior.
  Prefer existing project tests or fixtures when they directly represent the
  public contract, and avoid inventing requirements that the prompt does not
  state.
- Test configuration and serialized values in the same public shape used by
  the project. Confirm generated or user-visible output when the prompt makes
  that output part of the contract.
- After a concrete failure, diagnose the observed cause before broadening the
  patch. Re-run the focused task-shaped check after the final production edit.

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
- If a required implementation file is absent, locate the responsible current
  production path before creating or adapting code, then run the narrowest
  relevant project check.

## Final Diff Hygiene

- Run `git diff --check`.
- Run `git status --short --untracked-files=all` and inspect untracked files.
- Remove generated caches, local smoke files, downloaded fixtures, virtualenvs,
  node caches, coverage output, and temporary logs unless the prompt explicitly
  asks to add them.
- If any normal test cannot run, state the exact command and the observed reason,
  then compensate with the closest direct smoke you can execute.
