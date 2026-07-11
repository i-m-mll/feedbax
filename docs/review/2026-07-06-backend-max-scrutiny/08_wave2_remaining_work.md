# Wave 2: remaining work after auth:6eb99ef6

**Date:** 2026-07-06 · **Context:** umbrella `db41e6a` waves 1–3 were implemented on `integration/db41e6a-backend-hardening` (auth:6eb99ef6, now merged to develop at `ae207269`). This document records (a) the verification verdict on that branch, (b) the strategic findings from the original review that were NOT included in the first 12 children, and (c) the filing record for the wave-2 extension. Verification detail: `07_auth_6eb99ef6_verification.md`.

## A. Verification verdict on auth:6eb99ef6

**Approve-with-notes.** All 12 children have real corresponding work; no deleted tests or weakened assertions found in the diff sweep. Three items to fix immediately post-merge (all re-verified directly against the branch):

1. **No DB migration for schema changes** ✓ — `persistence/database.py` gains indexed columns and `nullable=False` constraints with no new alembic revision. Repo policy requires versioned migration or explicit reject for durable-format changes.
2. **Versionless-acceptance reintroduced one level down** ✓ — `contracts/migrations.py:~970-1000`: the outer workspace payload now fails closed, but nested scenario/stage payloads are migrated with `assume_current=True`. (The `spec_payload()` constructor case at `manifest.py:930` is legitimate — freshly built in-memory payloads are current by construction. The nested *stored* payloads are not.)
3. **Composite trainability regression** ✓ — the old worker allowlist (`Linear, MLP, GRU, LSTM, Recurrent Controller, Simple Feedback Loop`) was replaced by registry `trainable_by_default`, but `Recurrent Controller` / `Simple Feedback Loop` are template-graph composites (`contracts/graphs/templates.py`), not registry components — they silently lose default trainability. Untested.

Lesser notes: CDE display-only nodes now fail closed (`NotImplementedError`) rather than getting real builders — acceptable per policy but contradicts the commit message; real builders are `2f8dd61` scope. Persistence child's new tests don't assert the five claimed fixes.

## B. Strategic items NOT in the first 12 children (the big things)

Report 03's architecture assessment names the pattern: *"the single most useful structural intervention would be finishing what's already half-done: retiring (not just deprecating) the legacy `TaskTrainer` path, the standalone `train_ppo`, and the legacy `LossTermSpec` reduction stack — all three older siblings of a more complete, more correct implementation that already exists in the same codebase."*

### B1. Two loss spec-authoring generations → one (`dd224bf`)

**Terminology (per user question):** the runtime engine — `AbstractLoss`/`CompositeLoss`/`TermTree` in `loss.py`, the original user-designed hierarchy — is the oldest AND best-designed piece; it is NOT modified. The duplication is inside the newer **spec-authoring layer**: the earlier `LossTermSpec` shape (lives in `contracts/training.py:62`) with its private legacy reduction/metric pipeline in `service.py`, vs the later schema-versioned `ObjectiveSpec`/`ReductionSpec` path. Both lower into the `AbstractLoss` engine. The legacy pipeline produced the `l2`-as-abs bug and forecloses trial-grouped reduction (`63230c3` residual). Fix: adapter routing `LossTermSpec` through the `ObjectiveSpec` machinery, delete the legacy pipeline. Binding inventory: `09_losstermspec_inventory.md` (field mapping, dispatch map, the two numeric-change combinations, zero rlrmp producers).

### B2. Two training stacks → one (`34fed00` + rlrmp `a0a03bf`)

Executor path (`executor.py` + `checkpoint_custody.py`) is mature (atomic writes, content-addressed manifests, full optimizer-state restore — resolves `b1860d5`/`ae2d692` there). `TaskTrainer` retains the optimizer-state-discard bug; policy says retire, not patch. rlrmp trains through TaskTrainer → coordinated migration, feedbax deletion blocked by rlrmp `a0a03bf`. Binding inventory: `10_tasktrainer_capability_matrix.md` (6 capabilities with no executor equivalent; checkpoint reject-with-error justification; consumer inventory incl. rlrmp's `cs_nominal_gru.py` private-`_train_step` usage).

### B3. PPO consolidation (`e80ec80`)

Three parallel PPO implementations; standalone `train_ppo` never got the single-jit treatment (~40 host syncs/batch). Consolidate to the batched one; relates `67e2e5e`.

### B4. Durable-spec schema identity completion (`48f6e40`)

`ExecutionPlan.cloud_payload`/`.reproducibility` unschemad (`01#E1`); command/provenance triplication (`01#E2/E3`); `GraphSpec` et al. lack `extra="forbid"` (`02`); Studio API models lack schema identity. Guard rail: validation-only Pydantic shapes are insufficient per repo policy — registered identity + migrate-or-reject + focused tests.

### B5. Config subsystem hardening (`0eae385`)

`config/tree.py` zero dedicated tests; deep-merge `None` semantics divergence; mutable global singletons.

### B6. Dashboard retirement (`c6c8af2`) — approved 2026-07-06.

## C. Residual small/medium fixes (`6d7ecd7`)

`01#G5`, `01#N1/N4/N6/N7`, `03` task/intervene residuals, `04#M1/M4/M9/M10/M11/M13`, `05#AN-11/AN-12/AN-21/AN-22`. Each must be diff-checked against merged db41e6a before fixing — some may have been swept in by the god-module and gate children.

## D. Filing record (2026-07-06)

| Issue | Wave | Scope |
|---|---|---|
| `f9a8524` | W4 | §A merge fixes (alembic migration, nested assume_current, composite trainability, persistence test assertions, CDE framing) |
| `dd224bf` | W5 | B1 loss-spec unification — closes `63230c3` gap; binding addendum from `09_` |
| `34fed00` | W5 | B2 TaskTrainer retirement — blocked by rlrmp `a0a03bf`; resolves `b1860d5` residual; binding addendum from `10_` |
| `e80ec80` | W5 | B3 PPO consolidation (related `67e2e5e`) |
| `48f6e40` | W5 | B4 schema identity completion (related `535a32e`) |
| `0eae385` | W5 | B5 config hardening |
| `c6c8af2` | W5 | B6 dashboard retirement (approved) |
| `6d7ecd7` | W6 | §C residual sweep (diff-check first) |

Cross-repo: rlrmp `a0a03bf` (executor-stack migration) **blocks** `34fed00`; decision logged on rlrmp training-methods coordination `c99ad9d`. Verification verdict + extension logged as a comment on `db41e6a`. Spec-hardening addenda (binding acceptance criteria) added to `dd224bf`, `34fed00`, `48f6e40`, and `a0a03bf` on 2026-07-06.

**Housekeeping:** this `docs/review/` directory is untracked and has been lost to host-side tree operations twice — commit it (e.g. alongside the W4 child) so the spec substrate the issues reference is durable.
