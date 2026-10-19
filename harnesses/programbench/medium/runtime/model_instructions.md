# ProgramBench Reconstruction Harness

Use this harness as a task-agnostic control loop for rebuilding a program from
documentation and black-box observations. It supplements the benchmark rules;
it never permits source retrieval, internet access, decompilation, tracing, or
reuse of the reference executable.

## Evidence-first workflow

1. Read all bundled documentation and turn it into a compact behavior ledger:
   commands, flags, inputs, outputs, exit status, file effects, defaults, error
   behavior, formatting, and explicitly documented edge cases.
2. Establish a runnable skeleton early. Create `compile.sh`, build a minimal
   independent executable, and confirm the complete build/run cycle before
   investing in secondary behavior.
3. Probe the reference through its normal public interface. Use small,
   discriminating experiments that separate plausible behaviors; record the
   observed input, output, status, and state change instead of relying on
   assumptions based on the program name.
4. Prioritize breadth before polish. Implement the documented happy paths and
   common input classes, then error handling, boundary cases, formatting, and
   low-frequency options. Keep a list of remaining unsupported behavior.
5. Compare your implementation and the reference on the same locally generated
   cases. When they differ, reduce the case and fix the underlying general rule
   rather than adding a one-off special case.
6. Rebuild from a clean state periodically. A solution that works only because
   of cached artifacts, undeclared tools, or files outside the submission is
   not complete.
7. Before finalizing, verify that `compile.sh` is executable, produces
   `./executable` without network access, and that representative success and
   failure probes agree with the reference.

## Efficiency rules

- Batch independent documentation reads and related probes when practical.
- Prefer high-information probes over broad random exploration.
- Keep outputs bounded and summarize findings in the behavior ledger.
- Stop repeating probes once a stable rule has been established.
- Reserve the final portion of the budget for a clean rebuild and end-to-end
  validation rather than continued exploration.

## Reference lifecycle

Keep the black-box reference available until differential validation is finished.
Develop a separately built candidate at another path, with the same executable
basename where relevant. Do not overwrite the reference during implementation.
If a permitted move is necessary, retain the execute-only reference on the same
filesystem without reading/copying its bytes or changing its permissions.
The reference is only a development oracle: never include it in the submission
or invoke it from the submitted implementation. Self-authored expected outputs
do not replace an actual reference-versus-candidate comparison. Build the final
deliverable from the validated independent source after the last comparison.
