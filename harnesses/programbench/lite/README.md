# ProgramBench Lite

This tier uses a fixed high-reasoning profile and staged tool-use limits for
cleanroom program reconstruction. Read the public task documentation, build a
runnable skeleton early, compare implementation and reference behavior with
identical discriminating probes, and reserve time for a clean rebuild and
representative end-to-end checks.

`runtime/policy.json` defines the profile and budgets.
`runtime/model_instructions.md` defines the agent workflow, and
`runtime/hook.py` enforces the tool-use stages. Use the integration steps in
the shared `../../README.md`.
