# ProgramBench Heavy

This tier extends the ProgramBench reconstruction workflow with an evidence
checkpoint: distinguish a probe setup error from the behavior being tested,
and prefer meaningful end-to-end evidence over superficial agreement. It
retains the larger tool-output history and staged tool budgets.

`runtime/policy.json` defines the fixed profile, output retention, and tool
budgets. `runtime/model_instructions.md` defines the agent workflow, and
`runtime/hook.py` enforces the tool-use stages. Use the integration steps in
the shared `../../README.md`.
