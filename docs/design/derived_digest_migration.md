# Versioned Verification and Sanctioned Migration of Derived Digests in Content-Pinned Locks

Status: draft, pending external critique. Owning issue: Mandible feedbax/8cab07a. Surfaced from:
rlrmp2/5ea2a98 (rehearsal fork-proof gate) and the drift class on rlrmp2/86dc02a. Author: Claude
Fable 5 (coordinator), from a code scout of the surfaces named below. Review provenance: revised
after an external high-reasoning-effort critique (GPT-5.6-Sol via Codex, 2026-07-22).

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
(`_compatible_stored_canonical_projection`, `checkpoint_custody.py` ~5183, accepts recorded v2/v3
bindings, authenticates the original projection bytes, migrates the embedded `TrainingRunSpec`
through `migrate_structured_spec_payload`, restamps v4, and compares canonical bytes). The fork path
has only strict equality, plus a sanctioned mutator (`relock_checkpoint_fork_derived_digests`, the
sole lock-mutating API, which refuses authored-field changes and no-op refreshes) that has no
file-level or CLI surface — which is exactly the vacuum the downstream regex script grew into.

## 2. Field taxonomy and the drift-classification model

A lock mixes three field classes with different mutation rules:

1. Authored decisions (targets, depths, schedules): never machine-changed.
   `_replace_checkpoint_fork_compatibility_locks` already enforces authored-field invariance during
   relock. This is an operation-scope guard only — it proves the operation did not touch authored
   fields; it is not historical proof that inputs are unchanged.
2. Content pins of real inputs (parent root hashes, archive manifests): never machine-changed; a
   mismatch is a real failure.
3. Derived digests: deterministic recomputations of (1)+(2) under a named algorithm. Only the
   run-contract digest is migratable, and only under the model below.

### 2.1 Versioned versus fixed-sentinel digests

`CheckpointForkCompatibilityProjection` records an algorithm version for the run-contract digest
only. The structural-ABI and population-ID digests have no co-stored algorithm version. Therefore:

- The run-contract digest is the sole allowlisted migration target. Its recorded
  `run_contract_algorithm_version` selects a supported migration edge (§2.3).
- The structural-ABI and population-ID digests are fixed sentinels. Any mismatch there is always
  input drift and is never migratable. In the incident, the `slot_structural_abi_sha256` values under
  `feedbax.training_checkpoint.structural_abi.content.v2` are such sentinels.

Adding per-field algorithm identities to the other digests — via a projection-schema version bump —
is an explicit non-goal of this spec and a possible future item.

### 2.2 The migration proof model

A digest whose recorded algorithm version equals the current algorithm must match the recomputed
candidate exactly; any mismatch there is input drift. Only a run-contract digest whose recorded
algorithm version is older than the current one, along a supported edge, is a migration candidate.

Fork locks store only digests, not the underlying canonical bytes. So the proof that an
algorithm-only move is safe cannot come from the lock itself: authenticated historical evidence is an
input the migration operation must be handed. The sanctioned proof mirrors the resume verifier
(`_compatible_stored_canonical_projection`):

1. Obtain authenticated historical evidence — either the stored old-version canonical projection
   bytes/payload from authenticated custody (the source checkpoint transaction, or a registered
   immutable artifact), or immutable content pins covering every input to both algorithm versions.
2. Migrate those historical bytes through the registered spec-schema migration.
3. Canonicalize the migrated payload under the target algorithm.
4. Require byte-equality between that result and the current canonical projection recomputed from
   current inputs.

Any mismatch at step 4 is input drift: refuse, never migrate. The migration record binds an
authenticated pre-migration source-plan canonical hash so the evidence chain is auditable.

No silent auto-migration, ever: derived digests are drift tripwires; recomputing them invisibly on
schema advance would hide semantic changes underneath locked science. Migration is one sanctioned,
explicit, audited operation.

### 2.3 Shared allowlisted migration-edge registry

Supported migration edges live in exactly one internal registry, keyed by `(algorithm,
source_version, target_version, hash_domain)`, and consumed by both the resume verifier and the new
fork-gate/migration core. The initial edges are exactly the resume path's `v2 -> v4` and `v3 -> v4`
run-contract-binding edges. Any version that is unknown, newer than the current algorithm, or under a
different hash domain is unsupported: the operation fails closed with a distinct error and is never
classified as migratable algorithm drift.

### 2.4 Owner-attestation path (not a migration proof)

When authenticated historical evidence does not exist, a separately named manual authorization mode
may perform the relock. It is deliberately not called a migration proof. It is entered only through
an explicit CLI flag and is recorded in the audit record with `proof_mode = "owner-attestation"`. It
carries mandatory requalification duties — re-running the lock's fork-proof/rehearsal tests — which
are recorded in the record. This path exists so that a genuinely un-evidenced but owner-vouched
relock leaves durable provenance instead of an untracked regex edit.

## 3. Changes

### 3.1 Version-aware verification at the fork gate

`_prepare_checkpoint_fork_plan`'s digest comparison produces the shared classification result (§3.5)
and acts on it with input-drift precedence. The fork gate runs during execution and holds no
historical evidence, so it decides on the version-level classification only: a fixed-sentinel
mismatch is unconditional input drift and dominates, and the gate raises `CheckpointCompatibilityError`
with the classification named in the message and does not advertise the migration command. When the
run-contract field carries a stored algorithm version that is older along a supported edge and the
fixed sentinels are clean, the gate raises the new typed error `CheckpointDigestMigrationRequired`.
Because the gate cannot run the authenticated-historical-evidence proof, this signal deliberately does
**not** assert the drift is algorithm-only: its message states that the stored algorithm version is
older on the named supported edge, that input drift cannot be ruled out without historical evidence,
and that the `checkpoint relock` operation will adjudicate — authenticated evidence either proves the
science is unchanged or refuses the move as input drift, while `--owner-attestation` is a deliberate
no-evidence override carrying mandatory re-qualification duties. Historical-comparison input-drift
detection on the run-contract field is asserted at the evidence-bearing classify/migrate layer and the
CLI (§4.1, §4.3), not at the evidence-free gate. Unknown/newer/other-hash-domain versions fail closed
with the distinct unsupported-edge error, never as migratable drift. The resume path's existing
versioned verification is untouched.

### 3.2 Sanctioned migration core (library)

A public projection-level primitive (working name `migrate_fork_compatibility_projection`) plus a
plan-level operation building on `relock_checkpoint_fork_derived_digests`:

- consumes the shared classification result; refuses with typed errors on any input drift, unsupported
  edge, authored-field change, or no-op;
- performs the authenticated-evidence proof of §2.2 for the run-contract digest, and recomputes only
  that field, updating its recorded `run_contract_algorithm_version` in the same operation (the field
  the downstream script missed);
- treats structural-ABI and population-ID digests as fixed sentinels: a mismatch there refuses;
- emits the typed checkpoint-digest migration record (§3.6) with source/target plan canonical hashes,
  migration edge, affected fields, evidence refs, proof mode, and requalification duties;
- returns the migrated projection/plan plus the record; the caller persists both adjacently.

The projection-level primitive is the downstream subsumption surface; the plan-level operation and
CLI build on it.

### 3.3 CLI wrapper for feedbax-native plans

A `python -m feedbax checkpoint relock` subcommand following the existing `checkpoint fork-plan`
conventions:

- `--check` produces and reports the shared per-field classification and exits nonzero on any drift.
- `--write` performs sanctioned migration on supported run-contract algorithm drift only, writing the
  migrated plan with its migration record; it refuses on input drift and on unsupported edges.
- An explicit owner-attestation flag enables the §2.4 attestation path when historical evidence is
  absent, recording `proof_mode = "owner-attestation"` and the requalification duties.

CLI write semantics: snapshot inputs, an optimistic source-plan-hash check, atomic
temp-file-then-rename replacement, and all-targets-and-record-or-nothing transactionality — the
migrated digests and the record land together or not at all.

### 3.4 Downstream subsumption boundary

Feedbax does not parse downstream lock formats. Downstream repos embedding
`CheckpointForkCompatibilityProjection` extract it, call the projection-level primitive with their own
recomputation inputs and their own authenticated historical evidence, and persist the returned
projection and record in their own file format. The projection-only primitive cannot enforce, and the
caller therefore owns as explicit obligations:

- authored plan fields outside the projection (the primitive sees only the projection);
- downstream file integrity and atomic replacement of the downstream lock;
- persistence of the returned migration record adjacent to the migrated projection.

The caller must supply authenticated historical evidence; the primitive refuses to migrate without it
(the owner-attestation path is a CLI/plan-level surface, not the pure primitive's decision).
rlrmp2's `refresh_mapped_fork_lock.py` is replaced by a thin call in a separate, owner-approved
follow-up rlrmp2 issue after this lands; its Mandible-artifact verification coupling stays downstream.

### 3.5 Shared structured classification result

Drift classification is computed once, as one typed classification object, and consumed by the fork
gate, the migration core, the CLI, and downstream callers. It carries the selected migration edge (or
none), the per-field class (unchanged, migratable algorithm drift, input drift, or unsupported), the
proof mode, and the sanction status. This removes divergent per-caller reclassification.

For orientation: the `CheckpointCompatibilityError` catch at `run_matrix.py` ~243 wraps
execution-dependency validation only; the `fork_checkpoint_plan` call at ~698 is not caught there, so
the new migration signal does not risk being swallowed by that catcher.

### 3.6 Migration record and persistence contract

A typed checkpoint-digest migration record — wrapping or paralleling `ArtifactMigrationRecord`
(`feedbax/contracts/manifest.py`), which names schema versions rather than algorithm versions — binds:
source and target plan canonical hashes, the migration edge, the affected fields, the authenticated
evidence refs, the proof mode (authenticated historical evidence versus owner-attestation), the
requalification duties, and the tool name and version.

The migration history persists in a top-level field of a new `CheckpointForkPlan` schema version. It
does not live in `CheckpointForkProvenance` (that is created only in the resulting transaction, too
late to record the pre-migration source), nor in metadata (excluded from the canonical plan hash).
The schema-version bump ships with a registered migration rule and focused tests, per the repo rule
that durable schema changes carry a versioned migration and tests.

### 3.7 Non-goals

No generic lock-migration framework beyond the fork-compatibility projection. No per-field algorithm
identities for the structural-ABI or population-ID digests. No auto-migration during verification or
preflight. No changes to the resume path. No parsing of downstream lock formats. DB (Alembic)
migrations untouched. Generalization to the other versioned derived digests in the codebase
(preparation-RNG, content-integrity) is deliberately deferred until a second concrete incident; this
spec establishes the pattern without building the framework.

## 4. Acceptance

Focused `migration_contract` tests, xdist-safe, writing only under `tmp_path`, per the repo migration
policy's accept/migrate/reject bar:

1. v3-stored plan, algorithm-only move, unchanged inputs, authenticated historical evidence present:
   verification raises `CheckpointDigestMigrationRequired` naming the supported edge and command;
   migration succeeds via the authenticated-evidence proof, updates the run-contract digest and its
   algorithm-version field together, emits the record; re-verification then passes strict equality
   under v4. The test includes a companion case where the historical evidence, migrated forward, does
   not byte-match the current canonical projection (invisible run-spec drift): the historical
   comparison mismatches and migration refuses.
2. Authored field mutated before migration: migration refuses via the existing authored-invariance
   check, with a distinct error.
3. Algorithm bump plus real input drift, sentinels unchanged: the run-contract field shows an
   older algorithm version but its recomputed value under the historical-evidence proof does not
   match. Because the fork gate holds no historical evidence, it cannot detect this drift and instead
   raises `CheckpointDigestMigrationRequired` on the clean-sentinel supported version drift — its
   message names the supported edge, states input drift cannot be ruled out without evidence, and
   points at the `checkpoint relock` adjudication (never asserting the drift is algorithm-only). The
   input-drift refusal is then asserted at the evidence-bearing layer: the classify/migrate core and
   the CLI, handed the authenticated historical evidence, find the historical comparison mismatches,
   let input drift dominate, raise the non-migratable input-drift error, and do not persist any
   migration. The test pins this end to end (gate signal, then evidence-bearing refusal on the same
   plan).
4. Mixed-drift precedence: run-contract algorithm drift co-occurring with a fixed-sentinel mismatch;
   the input-drift signal dominates, the migration command is not advertised, and migration refuses.
5. Current-version run-contract digest mismatch: `CheckpointCompatibilityError` with input-drift
   classification; no migration path offered.
6. Unknown/newer version or a hash-domain change on the run-contract field: unsupported edge, fail
   closed with the distinct error; never classified as migratable drift.
7. Missing or invalid historical evidence: migration refuses; the owner-attestation path succeeds only
   with the explicit flag and records `proof_mode = "owner-attestation"` plus the requalification
   duties.
8. Multi-target all-or-nothing: a forced mid-write failure leaves neither the migrated digests nor the
   record persisted.
9. Record binding and round-trip: the persisted record binds the source and target plan canonical
   hashes and survives a plan reload under the new schema version.
10. Stale/concurrent CLI write: the optimistic source-hash check fires when the plan changed under the
    operation.
11. Registry parity: the fork gate/migration core and the resume verifier consume the same
    migration-edge registry (the shared `v2 -> v4` and `v3 -> v4` edges).
12. CLI `--check`/`--write` behaviors, including no-op refusal and input-drift refusal.

## 5. Resolved decisions

1. Error taxonomy: `CheckpointDigestMigrationRequired` is a sibling of `CheckpointCompatibilityError`
   under a shared `CheckpointCustodyError` base, so broad compatibility catchers do not swallow the
   migration signal.
2. Migration-record placement: migration history lives in a top-level field of a new
   `CheckpointForkPlan` schema version — not in `CheckpointForkProvenance` (created only in the
   resulting transaction, too late to bind the pre-migration source) and not in metadata (excluded
   from the canonical plan hash).
3. Fail-closed versus accept-with-report: the fork gate fails closed and migration is a separate
   sanctioned operation, because a fork lock is a launch-eligibility artifact and silently accepting
   old-version digests would let stale locks flow toward launch. The resume path may
   authenticate-and-proceed because it rewrites nothing durable; the asymmetry is deliberate.
4. Requalification: duties are mandatory. Migration is not self-qualifying, and the owner-attestation
   path in particular requires re-running the lock's fork-proof/rehearsal tests, recorded in the
   record.
5. Un-evidenced relock: there is no weaker "pinned-input fallback" migration proof. Either
   authenticated historical evidence supports a genuine migration (§2.2), or the explicit
   owner-attestation path (§2.4) records a manual authorization with mandatory requalification.
