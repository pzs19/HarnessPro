# ProgramBench Medium

This tier keeps the cleanroom reconstruction workflow and retains more tool
output in the agent's history than Lite. The extra history is intended to
preserve relevant documentation and probe evidence during longer tasks.

`runtime/policy.json` defines the fixed profile, output retention, and tool
budgets. `runtime/model_instructions.md` defines the agent workflow, and
`runtime/hook.py` enforces the tool-use stages. Use the integration steps in
the shared `../../README.md`.
