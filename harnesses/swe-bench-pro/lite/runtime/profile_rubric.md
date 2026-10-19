Choose exactly one execution profile using only observable properties in the
public task description. Optimize expected repair success per dollar. Never use
task identifiers, repository identity, remembered outcomes, hidden tests,
previous trajectories, or benchmark-specific fingerprints.

Classify the repair scope and evidence path, not apparent prose length. The
question is how many coupled behavioral uncertainties must be resolved and how
directly the local repository can validate them.

Apply these gates in order.

## `compact`

Choose `compact` only when the public contract makes the implementation and
validation path nearly mechanical:

- one narrow implementation surface or generated artifact is explicitly
  identified;
- the requested behavior is a localized mapping, default, guard, formatting,
  metadata, or similarly bounded correction;
- a direct local check can cover the success path and stated edge cases; and
- there is no meaningful compatibility, persistence, concurrency, lifecycle,
  schema migration, or cross-component uncertainty.

A short prompt alone is not enough. If locating the real execution path is a
substantial part of the task, continue to `lean` or `standard`.

## `lean`

Choose `lean` for an ordinary bounded repository repair when all of the
following are supported by the public description:

1. The requested behavior belongs to one cohesive subsystem or one end-to-end
   path with a small number of adjacent callers.
2. The contract gives concrete observable behavior, public interfaces, or
   acceptance conditions rather than only a broad goal.
3. Local source, tests, fixtures, deterministic commands, or nearby
   implementation patterns should provide a direct implementation and
   validation path.
4. At most a few behavioral uncertainties need to be resolved; the work does
   not require coordinating independently evolving protocols or states.
5. A coherent patch plus focused tests is realistically achievable without
   broad architectural exploration.

A lean repair may touch several files when they form one atomic change, such as
an implementation and its callers or tests. Do not upgrade merely because the
repository is large, the language is unfamiliar, or hidden tests exist.

## `research`

Choose `research` only when the primary task is factual or historical
reconstruction from multiple accessible versions, records, or sources. One
local history lookup used to implement a normal repair remains `lean` or
`standard`.

## `deep`

Choose `deep` only when several materially different hypotheses must be tested
against a strong local oracle, or when correctness spans multiple interacting
states and repeated evidence-changing experiments are explicitly feasible.
Examples include latent-state recovery, coupled migrations, multi-stage
concurrency behavior, or constrained synthesis with a deterministic scorer.
Complex terminology or a long specification alone is insufficient.

## `standard`

Choose `standard` when any of these materially affect correctness:

- several subsystems or independently versioned interfaces must change
  together;
- public API or serialization compatibility spans producers and consumers;
- persistence, migration, lifecycle, distributed state, concurrency, security
  boundaries, or cleanup semantics have coupled edge cases;
- generated interfaces and hand-written implementations must remain aligned;
- the task is broad enough that no focused local check covers most decisive
  behavior; or
- the evidence quality is genuinely unknown after reading the public contract.

`standard` is the safe default for coupled repairs, but not for every normal
bug fix. Return a calibrated confidence based only on how clearly the public
description satisfies the selected gate.
