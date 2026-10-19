# SWE-bench Pro Lite Harness

# Lite prioritizes efficiency with a task-agnostic contract workflow and
Git-history guidance capped at the published RHO wording. It uses bounded
profile-specific tool budgets without making tool output more lossy or
changing task routing. The classifier and checklist use only general
public-contract properties and no benchmark-task examples.

## Workflow

1. Read the prompt as a contract and record exact public names, behavior,
   defaults, compatibility constraints, and stated edge cases.
2. Inspect the narrow implementation path and nearby tests. Look for an
   existing reference when the task sounds like a known fix: local git history,
   tags, upstream commits, tests, fixtures, or nearby implementation patterns.
   Prefer adapting proven behavior over guessing.
3. Make the smallest coherent production change that satisfies the contract.
4. Run focused executable validation, investigate concrete failures, and check
   the final diff against the public contract.
5. Keep generated output, caches, and temporary tests out of the submitted
   patch.

The classifier sees only the public task description. It must not use task IDs,
repository names, remembered outcomes, hidden tests, previous trajectories, or
task-specific fingerprints. A Solver remains on GPT-5.5 and one selected
reasoning level for cache continuity.

`runtime/policy.json` is authoritative for profile selection, reasoning levels,
tool budgets, output compaction, and hard finalization.
