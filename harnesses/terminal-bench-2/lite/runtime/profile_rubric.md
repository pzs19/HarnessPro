Choose exactly one profile from observable task characteristics. Optimize
expected success per dollar. Never use historical outcomes remembered from prior
runs or evaluations.

Profiles:

- `compact`: low reasoning; converge near 4 tool calls and stop near 8.
- `lean`: low reasoning; converge near 10 calls and stop near 16.
- `standard`: medium reasoning; converge near 26 calls and stop near 38.
- `deep`: high reasoning; converge near 30 calls and stop near 48.
- `research`: high reasoning with about 40 calls for source reconstruction.

Evidence rules

- Use only validation, interaction, dependencies, artifacts, and constraints
  explicitly described in the task text. Never assume hidden-test behavior,
  previous attempts, tool output, runtime results, or historical scores.
- An undescribed validator has unknown quality, not weak quality. Unknown decisive
  evidence defaults to `standard`; missing information alone cannot justify
  another profile.
- A strong oracle must be explicitly local and repeatable and provide localized
  failure, quantitative error, candidate ranking, deterministic simulation, or a
  directly inspectable target. Compilation, launch, an open port, one end-to-end
  request, or whole-task pass/fail is weak only when the text explicitly limits
  feedback to that undifferentiated signal.
- Count distinct evidence-resolvable hypotheses, not files, prose, commands, or
  retries. Extra iterations are valuable only when described probes can
  distinguish hypotheses and change the plan.
- Include the stated watchdog and expected probe cost in the value of
  iteration. A precise oracle is not useful enough for `deep` when only a few
  full evaluations can fit and high reasoning does not create more evidence.

Apply the following gates in order and stop at the first match.

Gate 1: `compact`

Choose `compact` only when low marginal value is explicit and no faithful
decomposable oracle is described or structurally possible:

1. Long serial graphical, remote, virtualized, or manual interaction whose
   correctness depends on visual or hidden state.
2. Unavailable accounts, devices, services, credentials, or changing external
   state without a faithful local substitute.
3. Broad hidden or effectively unbounded behavior plus exact preservation of
   unrelated details, with explicitly end-to-end accept/reject feedback.
4. One unchanged artifact must satisfy independent grammars, parsers, runtimes,
   or environments, while checks expose only a joint accept/reject objective.
5. Bulk media or visual interpretation with no described numerical comparator
   and correctness resting on visual judgment.

Difficulty, visualization, specialization, or virtualization alone is not enough.
A small fixed visual artifact, scripted headless setup with direct checks, or
local scorer that distinguishes candidates should continue to later gates.
File-existence or format checks do not make long visual/media search a strong
oracle when the decisive event must still be found by inspecting many frames.

Gate 2: `research`

Choose `research` only when the primary deliverable is factual reconstruction
requiring multiple accessible sources, version/date alignment, provenance, or a
historical snapshot. A multi-record date/version ranking qualifies; one lookup or
research used only to build an artifact does not. If decisive sources are
explicitly inaccessible, choose `compact`.

Gate 3: `deep`

Choose `deep` only when a strong local oracle is explicitly described or can be
constructed from an exact deterministic reference, transformation, simulator, or
runtime supplied by the task, and at least one shape applies:

1. Latent structure or parameter recovery plus exact compatibility or a
   quantitative objective.
2. Correctness spans several interacting components, stages, or states and can be
   checked through exact invariants, reference behavior, or round trips.
3. Damaged structured-state recovery requiring competing format, ordering, or
   transformation hypotheses.
4. Constrained synthesis, compression, size reduction, or candidate search with
   coupled hard constraints and an executable scorer.
5. Nontrivial recovery, inference, or attack derived from many observations with
   exact candidate validation.
6. An inverse problem with a deterministic simulator or quantitative objective.
7. Candidate construction against a deterministic transformation or runtime,
   with multiple materially different mechanisms that can be executed and observed.
8. A severely constrained reimplementation of deterministic reference behavior,
   where exact candidate outputs can be compared locally despite language, size,
   compatibility, or dependency restrictions.
9. One source artifact must implement behavior while remaining valid under two
   or more independent grammars, compilers, or runtimes, each of which can be
   exercised separately for localized feedback.

Do not choose `deep` for terminology, long specifications, ordinary security or
reverse engineering, or hidden tests alone. Ordinary performance tuning and
ordinary state propagation through services, hooks, or deployments are
`standard`. A binary validator alone is insufficient, but a deterministic local
transformation plus executable behavior can support meaningful candidate
iteration even when the final verdict is binary. Prefer `standard` when each
candidate evaluation is described as expensive enough to consume a material
part of the watchdog and the work is mainly bounded tuning against one supplied
evaluator. Modifying an existing artifact against one supplied evaluator remains
`standard` merely because it combines a quantitative objective with preservation
constraints; `deep` additionally requires latent recovery, combinatorial search,
or several materially different candidate mechanisms. `deep` requires room for
several evidence-changing iterations.

Gate 4: `lean`

Choose `lean` only when all are explicit:

1. Bounded, mostly deterministic work.
2. At most two meaningful uncertainties.
3. A credible plan before broad exploration.
4. A direct local check covering nearly all decisive behavior.
5. About eight or fewer evidence-producing calls for inspection, implementation,
   and final validation.

Typical cases are a localized transformation; enumerable mechanical cleanup with
a deterministic final scan; a simple build or single-service setup with one main
uncertainty and a direct test; or bounded numerical work with direct value, shape,
or residual validation.

Do not choose `lean` when several coupled uncertainties require iterative
candidate comparison, or when recovering latent state requires materially
different hypotheses. Apply the `deep` gate first when a constructible exact or
quantitative reference makes those iterations useful. Also do not choose `lean`
for broad cleanup where unaffected content must be preserved exactly and the
set of true matches is not explicitly bounded; false-positive scope is a
separate uncertainty even when a final scan is deterministic.

Use `standard` when coordinating several schemas, formats, subsystems,
integration boundaries, correctness/performance constraints, or hidden edge-case
families. Also use it for multi-service state propagation, broad security
properties, builds combining source provenance with feature exclusion and
multiple runtime modes, corpus-wide cleanup with exact preservation of unaffected
content, or unstructured recovery with several evidence paths but only final
candidate checks.

Gate 5: `standard`

Choose `standard` for everything else, including unknown oracle quality. It is
the default for ordinary implementation, debugging, data processing, service
setup, bounded numerical work, and analysis with several actionable
uncertainties, and should normally be the plurality.

