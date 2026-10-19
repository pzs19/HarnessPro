# SWE-bench Pro Medium Harness

Use this harness before editing and before finalizing a task. The goal is to
make the final patch answer the exact prompt, not just compile.

Medium keeps the repair workflow task-agnostic while giving repository repairs
enough room for executable validation. A separate model selects the
reasoning/budget profile from the public task description only. Git-history
guidance is deliberately capped at the wording and scope of the published RHO
Harness: it may be consulted as one conditional reference alongside tests,
fixtures, tags, and nearby implementation patterns.
Low-confidence `lean` selections use the neutral medium profile; runtime hooks
compress large tool output, discourage repeated exploration, and enforce a
finite hard limit.

## Default Workflow

1. Read the prompt as a contract. Write a short requirement ledger covering:
   exact public names, file paths, payload shapes, errors/status codes,
   default values, supported backends/platforms, and called-out edge cases.
2. Look for an existing reference when the task sounds like a known fix:
   local git history, tags, upstream commits, tests, fixtures, or nearby
   implementation patterns. Prefer adapting proven behavior over guessing.
3. Trace the real path end to end before editing. Follow stored data back to
   user-visible artifacts, config values back to their runtime representation,
   and wrappers back to the implementation that actually executes.
4. Make the smallest production-source change that satisfies the contract.
   Keep generated files, caches, temporary smoke tests, and local environment
   output out of the submitted diff.
5. Verify executable behavior. Syntax checks are not enough: run at least one
   focused smoke that calls every new public route/function/method and covers
   each explicit failure/edge condition in the prompt.
6. Re-open the requirement ledger before the final answer. Check literal API
   names, response shapes, defaults, cleanup behavior, and every backend or
   platform named by the prompt.

## Tools

- `harness/bin/repair-verify` discovers common non-PATH toolchains, runs cheap
  language checks, reports generated artifacts, and accepts extra task-specific
  commands.
- `harness/checklists/contract-verification.md` is the detailed checklist for
  requirement-led smoke tests and final review.

Typical use:

```bash
../harness/bin/repair-verify repo
../harness/bin/repair-verify repo -- go test ./lib/events ./lib/service
../harness/bin/repair-verify repo --clean-pycache -- python -m compileall -q lib
```

If project dependencies are unavailable, still write a tiny direct smoke with
local stubs for the changed layer. The smoke should execute the contract shape
that hidden tests are likely to assert.
