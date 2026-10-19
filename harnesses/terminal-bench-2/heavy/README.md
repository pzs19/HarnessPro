# Terminal-Bench 2 Heavy Harness

This mode targets a higher-quality point with a task-general selector. It uses
one tool-free GPT-5.5 medium Responses call, a 120-second timeout, three
transport attempts, at most
2,048 output tokens, and at most 24,000 task characters.

The selector uses only the public task description. It contains no task IDs,
historical outcomes, rare task phrases, or task-specific rules.

Compared with Medium, Heavy keeps compact work on low reasoning, promotes lean
work to medium reasoning, and promotes standard work to high reasoning. It also
gives all non-compact profiles enough room for an independent task-shaped final
check. The changes are based only on general uncertainty, interaction, oracle,
and validation structure—not benchmark identity or remembered task outcomes.
