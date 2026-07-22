# Versioned Verification and Sanctioned Migration of Derived Digests in Content-Pinned Locks

Status: draft, pending external critique. Owning issue: Mandible feedbax/8cab07a. Surfaced from:
rlrmp2/5ea2a98 (rehearsal fork-proof gate) and the drift class on rlrmp2/86dc02a. Author: Claude
Fable 5 (coordinator), from a code scout of the surfaces named below.

## 1. Problem

Content-pinned locks — feedbax `CheckpointForkPlan` JSON files and downstream lock files embedding
`CheckpointForkCompatibilityProjection` — store derived digests computed by versioned feedbax
algorithms. When such an algorithm advances, every stored lock fails the fork gate's strict-equality
comparison (`_checkpoint_fork_compatibility_differences`, reached from
`_prepare_checkpoint_fork_plan` in `feedbax/training/checkpoint_custody.py`) with a generic
`CheckpointCompatibilityError` that is indistinguishable from real input drift.

The concrete incident: commit `4d6405fe` ("Separate launch policy from training identity") moved the
run-contract binding algorithm from `feedbax.training_checkpoint.run_contract_binding.v3` to `.v4` by
removing operational launch policy from `TrainingRunSpec`, changing the canonical projection identity
while the science was unchanged. rlrmp2's `mapped_fork_proof.lock.json` (storing v3 digests) then
failed the fork gate. The downstream remedy was a hand-rolled refresh script
(`refresh_mapped_fork_lock.py`) that regex-substitutes recomputed sha256 values into the lock text.
In live use that script recomputed `run_contract_projection_sha256` but silently failed to cover the
co-stored `run_contract_algorithm_version` field, leaving the lock half-migrated and the gate still
failing — a concrete demonstration that ad hoc downstream refreshers under-migrate. Each such
incident costs per-incident archaeology to establish that only the algorithm moved, plus a
coordinator authorization ritual, and records no provenance.

Feedbax already contains both halves of the answer, asymmetrically applied. The checkpoint-resume
path performs versioned verification and sanctioned migration
(`_compatible_stored_canonical_projection` accepts recorded v2/v3 bindings, authenticates the
original projection bytes, migrates the embedded `TrainingRunSpec` through
`migrate_structured_spec_payload`, restamps v4, and compares canonical bytes). The fork path has only
strict equality, plus a sanctioned mutator (`relock_checkpoint_fork_derived_digests`, the sole
lock-mutating API, which refuses authored-field changes and no-op refreshes) that has no file-level
or CLI surface — which is exactly the vacuum the downstream regex script grew into.

## 2. Field taxonomy and the drift-classification invariant

A lock mixes three field classes with different mutation rules:

1. Authored decisions (targets, depths, schedules): never machine-changed.
   `_replace_checkpoint_fork_compatibility_locks` already enforces authored-field invariance during
   relock.
2. Content pins of real inputs (parent root hashes, archive manifests): never machine-changed; a
   mismatch is a real failure.
3. Derived digests plus their recorded algorithm-version fields: deterministic recomputations of
   (1)+(2) under a named algorithm version. These are the only migratable fields, and only under the
   invariant below.

Drift-classification invariant: a derived digest whose recorded algorithm version equals the current
algorithm must match the recomputed candidate exactly — any mismatch there is input drift and is
never migratable. Only digests whose recorded algorithm version is older than the current one are
candidates for migration. Co-stored digests under algorithms that did not move (in the incident:
`slot_structural_abi_sha256` values under `feedbax.training_checkpoint.structural_abi.content.v2`)
act as input-drift tripwires during migration of the moved fields: if they mismatch, migration
refuses.

On old-version re-verification: when the old algorithm remains computable against current inputs,
migration should re-verify the stored old-version digests before recomputing, proving the inputs
unchanged directly. This is not always possible — a v3 run-contract projection cannot be recomputed
from a current `TrainingRunSpec` at all, because v4 removed the projection's input field. In that
case migration falls back to the combination of pinned-input authentication, authored-field
invariance, and the unchanged-algorithm tripwires above, and the audit record states which proof mode
was used.

No silent auto-migration, ever: derived digests are drift tripwires; recomputing them invisibly on
schema advance would hide semantic changes underneath locked science. Migration is one sanctioned,
explicit, audited operation.

## 3. Changes

### 3.1 Version-aware verification at the fork gate

`_prepare_checkpoint_fork_plan`'s digest comparison classifies each difference as `algorithm_drift`
or `input_drift` per the invariant. Pure input drift raises `CheckpointCompatibilityError` as today,
with the classification named in the message. Any algorithm drift raises a new typed error,
`CheckpointDigestMigrationRequired`, whose message names the stored and current algorithm versions
and the exact sanctioned migration command. Whether the new error subclasses
`CheckpointCompatibilityError` (existing catchers, e.g. the fork-parity check in `run_matrix.py`,
catch the parent) is an open question for critique. The resume path's existing versioned verification
is untouched.

### 3.2 Sanctioned migration core (library)

A public projection-level primitive (working name `migrate_fork_compatibility_projection`) plus a
plan-level operation building on `relock_checkpoint_fork_derived_digests`:

- classifies drift per the invariant; refuses with typed errors on any input drift, authored-field
  change, or no-op;
- recomputes only the fields whose algorithm version moved, and updates their recorded
  algorithm-version fields in the same operation (the field the downstream script missed);
- emits an `ArtifactMigrationRecord` (`feedbax/contracts/manifest.py`) with source/target algorithm
  versions, `deterministic=True`, the difference set, the caller's requalification requirements, and
  the drift-proof mode (old-version re-verification vs pinned-input fallback) in validation/metadata;
- returns the migrated projection/plan plus the record; the caller persists both adjacently.

The projection-level primitive is the downstream subsumption surface; the plan-level operation and
CLI build on it.

### 3.3 CLI wrapper for feedbax-native plans

A `python -m feedbax checkpoint relock` subcommand following the existing `checkpoint fork-plan`
conventions: `--check` reports the per-field classification and exits nonzero on any drift; `--write`
performs sanctioned migration on algorithm drift only, writing the migrated plan with its migration
record, and refuses on input drift. Where the record persists inside a `CheckpointForkPlan` (a new
optional field with a schema-version bump and registered migration, versus extending
`CheckpointForkProvenance`) is an implementation decision to be settled during implementation,
honoring the repo rule that durable schema changes ship with a versioned migration rule and focused
tests.

### 3.4 Downstream subsumption boundary

Feedbax does not parse downstream lock formats. Downstream repos embedding
`CheckpointForkCompatibilityProjection` extract it, call the projection-level primitive with their own
recomputation inputs, and persist the returned projection and record in their own file format.
rlrmp2's `refresh_mapped_fork_lock.py` is replaced by a thin call in a separate, owner-approved
follow-up rlrmp2 issue after this lands; its Mandible-artifact verification coupling stays downstream.

### 3.5 Non-goals

No generic lock-migration framework beyond the fork-compatibility projection. No auto-migration
during verification or preflight. No changes to the resume path. No parsing of downstream lock
formats. DB (Alembic) migrations untouched. Generalization to the other versioned derived digests in
the codebase (preparation-RNG, content-integrity) is deliberately deferred until a second concrete
incident; this spec establishes the pattern without building the framework.

## 4. Acceptance

Focused `migration_contract` tests, xdist-safe, writing only under `tmp_path`, per the repo migration
policy's accept/migrate/reject bar:

1. v3-stored plan, unchanged pinned inputs: verification raises `CheckpointDigestMigrationRequired`
   naming the command; migration succeeds, updates digest and algorithm-version fields together,
   emits the record; re-verification then passes strict equality under v4.
2. v3-stored plan, authored field mutated: migration refuses via the existing authored-invariance
   check, with a distinct error.
3. v3-stored plan, input drift (an unchanged-algorithm tripwire digest mismatches): both verification
   and migration classify input drift; migration refuses.
4. Current-version digest mismatch: `CheckpointCompatibilityError` with input-drift classification; no
   migration path offered.
5. CLI `--check`/`--write` behaviors, including no-op refusal and the input-drift refusal.

## 5. Open questions for critique

1. Error taxonomy: should `CheckpointDigestMigrationRequired` subclass `CheckpointCompatibilityError`
   for catcher compatibility, or be a sibling so existing catchers do not accidentally swallow the
   migration signal?
2. Migration-record placement in `CheckpointForkPlan`: new optional field with schema bump and
   registered migration, or extend `CheckpointForkProvenance` (schema `...fork_provenance.v2`)?
3. Fail-closed versus accept-with-report at the fork gate: this spec chooses fail-closed (a fork lock
   is a launch-eligibility artifact; silently accepting old-version digests would let stale locks flow
   toward launch), while the resume path accepts old versions after authentication. Is the asymmetry
   justified?
4. Should the projection-level primitive require the caller to supply requalification requirements (as
   `relock_checkpoint_fork_derived_digests` does), or may a pure algorithm-drift migration with a
   passing tripwire set be self-qualifying?
5. Is the pinned-input fallback proof (when the old algorithm is not recomputable) strong enough, or
   should migration in that mode demand an additional caller-supplied attestation?
