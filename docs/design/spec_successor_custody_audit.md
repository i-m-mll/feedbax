# Successor Spec Schema and Custody Audit

Umbrella: `64a04e0` (`Feedbax-native training and graph consumption cleanup`)
Audit issue: `4cdb3a4`
Date: 2026-07-03

This audit covers the successor schema families introduced after the baseline
analysis-bundle custody audit in `docs/design/spec_bundle_custody_audit.md`.
It checks registry enrollment, old-version behavior, custody ownership,
emitter-policy coverage, and tests for the new training, descriptor, analysis
data-product, execution, checkpoint, and manifest surfaces.

## Verdict Matrix

`Pass` means the production contract and existing or focused tests satisfy the
dimension. `Fixed in place` means this audit added explicit test coverage for an
already-present production row. `Gap listed` means the current design is usable,
but a follow-up issue should decide whether to strengthen custody.

| Dimension | TrainingRunSpec | Descriptors and basis | Analysis data products | Execution v2 | Checkpoint custody | TrainingRunManifest |
|---|---|---|---|---|---|---|
| Registry enrollment and namespace | Pass | Pass | Pass | Pass | Pass | Pass |
| Rejection coverage | Pass | Pass | Pass | Pass | Pass | Pass |
| Current accept or old-version behavior | Pass | Pass | Pass | Pass | Pass | Pass |
| Custody assignment | Pass | Pass | Pass | Gap listed | Pass | Pass |
| Emitter-policy conformance | Fixed in place | Fixed in place | Fixed in place | Fixed in place | Fixed in place | Fixed in place |
| Test coverage | Fixed in place | Fixed in place | Fixed in place | Fixed in place | Fixed in place | Fixed in place |

## TrainingRunSpec

| Field | Current state |
|---|---|
| Registry family | `TrainingRunSpec` |
| Identity | `feedbax.spec.training_run` |
| Current version | `feedbax.spec.training_run.v1` |
| Namespace | `feedbax.spec.*` |
| Owner module | `feedbax.contracts.training` |
| Emitters | `TrainingRunManifest.training_spec`, `provider_manifest.schemas` |
| Consumers | training executor pre-launch validation, downstream run-spec consumers |
| Old-version policy | reject `feedbax.spec.training_run.v0` |
| Tests | `tests/test_training_run_spec.py`, `tests/test_structured_spec_migrations.py` |
| Custody carrier | `SpecPayload` in `TrainingRunManifest.training_spec`; execution plans may also reference the same payload by `TrainingRunSpecSource` with schema identity and content hash. |

`TrainingRunSpec` is the public durable request envelope for one training run:
graph source, task, training config, objective slot, method identity and method
payload, worker execution declaration, execution policy, artifact policy, and
checkpoint/progress policy. `TrainingSpec` remains a narrower optimizer, loss,
and run-shape specification registered as `feedbax.spec.training`. Existing
manifest helpers may still embed `TrainingSpec`, while the native training
executor emits `TrainingRunSpec` through the same `TrainingRunManifest.training_spec`
slot when the run started from the successor contract. This audit does not
redefine `TrainingSpec` as replaced or removed.

## Descriptor And Basis Family

| Field | Current state |
|---|---|
| Registry families | `VariableDescriptor`, `ComponentDescriptor`, `DescriptorBasisIdentity`; selector identity subfamilies are also registered. |
| Identities | `feedbax.spec.descriptor.variable`, `feedbax.spec.descriptor.component`, `feedbax.spec.descriptor.basis` |
| Current versions | `.v1` for the descriptor and basis identities |
| Namespace | `feedbax.spec.*` |
| Owner module | `feedbax.contracts.descriptors` |
| Emitters | graph, training, and run metadata; `VariableDescriptor` components; descriptor-bearing specs; `provider_manifest.schemas` |
| Consumers | descriptor resolution, downstream selector validation, analysis data-product basis pins |
| Old-version policy | reject each family-specific `.v0` |
| Tests | `tests/test_descriptor_schema.py`, `tests/test_structured_spec_migrations.py` |
| Custody carrier | Embedded descriptor records and descriptor-basis hashes in graph, training, and analysis contracts; provider schemas expose the governed model shapes. |

The descriptor family has explicit schema identity on each emitted record and
hashes whole descriptor bases for downstream pinning. `AnalysisDataProduct`
consumes the basis hash rather than redefining graph build component identity.

## Analysis Data Products

| Field | Current state |
|---|---|
| Registry families | `AnalysisDataProductRequirement`, `AnalysisDataProduct` |
| Identities | `feedbax.spec.analysis_data_product_requirement`, `feedbax.manifest.analysis_data_product` |
| Current versions | `.v1` for both families |
| Namespace | requirement is `feedbax.spec.*`; product is `feedbax.manifest.*` |
| Owner modules | requirement: `feedbax.contracts.graph`; product: `feedbax.contracts.manifest` |
| Emitters | `AnalysisRunSpec.input_requirements`, `AnalysisRunManifest.produced_data`, `provider_manifest.schemas` |
| Consumers | `feedbax.integrations.provider.validate_analysis_spec` |
| Old-version policy | reject each family-specific `.v0` |
| Tests | `tests/test_analysis_data_products.py`, `tests/test_structured_spec_migrations.py` |
| Custody carrier | `AnalysisRunManifest.produced_data[]`; product artifacts use `ArtifactRef` with optional external artifact IDs, hashes, roles, logical names, and URIs. |

Generated analysis products are manifest-carried, not local-path-only records.
The product identity hash covers semantic envelope fields, artifact byte
identity, descriptor basis, parent manifests, checkpoint policy, rollout policy,
parameters, materialization, and regeneration records while intentionally
excluding mutable local URI and label fields.

## ExecutionSpec, ExecutionPlan, And LocalExecutionResult

| Field | Current state |
|---|---|
| Registry families | `ExecutionSpec`, `ExecutionPlan`, `LocalExecutionResult` |
| Identities | `feedbax.spec.execution`, `feedbax.manifest.execution_plan`, `feedbax.manifest.local_execution_result` |
| Current versions | `feedbax.spec.execution.v2`, `feedbax.manifest.execution.v2` |
| Namespace | `ExecutionSpec` is `feedbax.spec.*`; result families are `feedbax.manifest.*` |
| Owner module | `feedbax.execution.models` |
| Emitters | `feedbax.execution.models`, `feedbax.integrations.provider` |
| Consumers | execution planning, Studio execution |
| Old-version policy | reject `feedbax.spec.execution.v1` and `feedbax.manifest.execution.v1` |
| Tests | `tests/test_execution_contract.py`, `tests/test_structured_spec_migrations.py` |
| Custody carrier | Provider capabilities expose `ExecutionSpec` input and `ExecutionPlan` / `LocalExecutionResult` outputs; local execution writes a native `TrainingRunManifest` plus an `execution-plan.json` file. |

The wave-4 v2 bump is explicit: registry policy rejects v1 for `ExecutionSpec`,
`ExecutionPlan`, and `LocalExecutionResult`, and Pydantic validation rejects
`ExecutionSpec` v1 at model load time. This audit also added explicit structured
registry coverage for the local-result v1 rejection row.

### Listed Gap: Execution Artifact Routes Remain Path-Oriented

`ExecutionPlan.artifact_routes` records route role, source path, tracked flag,
and description for `execution_log`, `training_run_spec`,
`training_run_manifest`, `tracked_spec`, and `bulk_output`. That is useful for
planning and sync, but it is not the same custody shape as `ArtifactRef` or a
manifest-side artifact mapping because route entries do not carry artifact IDs,
content hashes, media types, or validation records. Local execution does emit a
`TrainingRunManifest` and carries stdout/stderr/manifest paths in
`LocalExecutionResult`, so the current system is inspectable. A follow-up should
decide whether execution logs, execution-plan files, and bulk output routes
need first-class `ArtifactRef` records or a dedicated execution manifest before
remote execution outputs become recurring durable artifacts.

## Checkpoint And Resume Custody

| Field | Current state |
|---|---|
| Primary registry family | `TrainingCheckpointTransactionManifest` |
| Identity | `feedbax.manifest.training_checkpoint_transaction` |
| Current version | `feedbax.manifest.training_checkpoint_transaction.v3` |
| Namespace | `feedbax.manifest.*` |
| Owner module | `feedbax.contracts.checkpoints` |
| Emitter | `feedbax.training.checkpoint_custody` |
| Consumers | Feedbax resume loaders, cloud-backed workers, downstream checkpoint adoption lanes |
| Old-version policy | migrate v1 to v2 by stamping `fork_provenance: null`, then migrate v2 to v3 by splitting structural content fingerprints from environment provenance and upgrading run-contract bindings |
| Tests | `tests/test_checkpoint_custody.py`, `tests/test_structured_spec_migrations.py` |
| Custody carrier | Checkpoint transaction manifest plus latest pointer; forked transactions also carry source identity, per-slot source/target hashes, transfer mode, transform metadata, and tool version. `TrainingRunManifest.checkpoint_custody[]` links the transaction by `ParentRef` or `ArtifactRef`. |

Checkpoint transaction v3 keeps hard integrity separate from portability
provenance. Slot blob SHA-256, transaction-root SHA-256, and structural
fingerprints over PyTree treedef, leaf path, leaf type, shape, dtype, weak type,
and static-leaf representation remain hard gates. Serializer/runtime versions
are recorded as `environment_provenance`; mismatches produce resume/fork notices
but do not reject a byte-intact, structurally matching checkpoint.

Run-contract bindings use `feedbax.training_checkpoint.run_contract_binding.v2`.
The stored canonical projection is canonical JSON over the migrated
`TrainingRunSpec` plus the phase program used for the barrier. No non-semantic
fields are excluded in v2; changing that inclusion rule requires a binding
algorithm bump. Binding mismatches compare the stored projection with the
expected projection and report field paths such as
`/training_run_spec/training_config/learning_rate`.

The older checkpoint-selection custody families remain registered and distinct:
`CheckpointSelectionSpec` (`feedbax.spec.checkpoint_selection.v1`) and
`CheckpointSelectionManifest` (`feedbax.manifest.checkpoint_selection`,
`feedbax.manifest.v1`) cover downstream selection and scorer custody, not the
training writer transaction. Both carry explicit reject policies and provider
or manifest handoff coverage.

## TrainingRunManifest

| Field | Current state |
|---|---|
| Registry family | `TrainingRunManifest` |
| Identity | `feedbax.manifest.training_run` |
| Current version | `feedbax.manifest.v1` |
| Namespace | `feedbax.manifest.*` |
| Owner module | `feedbax.contracts.manifest` |
| Emitters | `feedbax.contracts.manifest`, `feedbax.integrations.provider`, native training executor |
| Consumers | manifest load/write, provider handoff, evaluation and analysis entry points |
| Old-version policy | reject `feedbax.manifest.training_run.v0` |
| Tests | `tests/test_training_run_spec.py`, `tests/test_training_run_executor.py`, `tests/test_provider_contract.py`, `tests/test_structured_spec_migrations.py` |
| Custody carrier | Top-level training-run manifest with `SpecPayload` slots, `ArtifactRef` records, provenance, summary metrics, and checkpoint custody refs. |

Native training-run execution emits `TrainingRunManifest` directly from
`feedbax.training.executor`, preserving the `TrainingRunSpec` schema version in
manifest metadata and linking checkpoint transaction manifests through
`checkpoint_custody`.

## Worker-Contract Footnote

Issue `95ff045` introduced durable worker execution vocabulary adjacent to this
audit. The registered emitted family is `WorkerMethodContractSpec` with identity
`feedbax.spec.worker.execution_program`, version
`feedbax.spec.worker.execution_program.v1`, owner
`feedbax.contracts.worker`, and rejection of
`feedbax.spec.worker.execution_program.v0`. It is consumed by worker validation
and the training executor through `TrainingRunSpec.method_ref` resolution. It is
not one of the six active audit families, but it should receive the same custody
treatment if future lanes expose additional worker subrecords as provider
schemas or standalone manifests.

## In-Place Fixes

This audit added focused assertions to `tests/test_structured_spec_migrations.py`:

- successor families are named in the foundation-family inventory;
- `TrainingRunManifest` and `TrainingCheckpointTransactionManifest` are included
  in manifest namespace survival checks;
- owner and emitter policy rows are explicit for `TrainingRunSpec`,
  descriptors, analysis data products, execution v2, checkpoint transaction
  custody, and `TrainingRunManifest`;
- namespace-category coverage includes the successor spec and manifest families;
- `LocalExecutionResult` has explicit v1 rejection coverage alongside
  `ExecutionSpec` and `ExecutionPlan`;
- `TrainingCheckpointTransactionManifest` now has explicit v1-to-v2-to-v3
  migration coverage for fork provenance, portable structural fingerprints, and
  canonical run-contract bindings.

Later checkpoint fork work updated the checkpoint transaction registry row from
v1 reject-only behavior to a v2 manifest with registered v1 migration. Follow-up
checkpoint portability work updated the current manifest to v3 so environment
drift is provenance, not content-integrity failure.

## Follow-Up Candidates

Execution artifact-route custody should be made explicit before remote
execution outputs become a recurring durable artifact surface. The current
`ExecutionPlan` and `LocalExecutionResult` are registered and versioned, and
local execution emits a native `TrainingRunManifest`, but `ArtifactRoute`
entries and stdout/stderr/manifest path fields are path-oriented planning
records rather than `ArtifactRef`-style custody records with byte hashes,
optional external artifact IDs, media types, validation records, or a dedicated
execution manifest. A follow-up should decide whether to add first-class
execution artifact records, attach execution outputs to `TrainingRunManifest`,
or introduce a dedicated execution-output manifest family.

## Verification

Focused verification for this lane:

- `uv run --no-sync pytest tests/test_structured_spec_migrations.py -q`
- touched-suite lint: `uv run --no-sync ruff check tests/test_structured_spec_migrations.py`
- final diff hygiene: `git diff --check`
