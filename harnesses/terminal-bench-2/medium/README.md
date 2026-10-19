# Terminal-Bench 2 Medium Harness

This mode uses a task-general selector rubric, runtime hook, tool budgets,
and output compaction. It uses one tool-free GPT-5.5 medium Responses call, a 120-second
timeout, three transport attempts, at most 2,048 output tokens, and at most 24,000
task characters.

The selector uses only the public task description. It contains no task IDs,
historical outcomes, rare task phrases, or task-specific rules.

The selected profile is used directly, without a low-confidence promotion.
