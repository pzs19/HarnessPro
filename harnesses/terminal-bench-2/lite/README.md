# Terminal-Bench 2 Lite Harness

This mode uses a general selector rubric, runtime guidance, bounded tool use,
and conservative output-history limits. Its model/reasoning allocation is:

- compact: GPT-5.4-mini low
- lean: GPT-5.5 low
- standard: GPT-5.5 medium
- deep/research: GPT-5.5 medium

Classification remains one tool-free GPT-5.5 medium Responses call over only the
public task description. The rubric contains no task IDs, historical outcomes,
rare benchmark phrases, or task-specific verification rules.

The medium deep/research allocation tests the general hypothesis that strong,
repeatable local feedback can replace some expensive internal reasoning while
retaining enough interaction budget to converge.
