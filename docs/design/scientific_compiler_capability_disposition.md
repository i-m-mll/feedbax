# Scientific compiler capability and disposition matrix

## Decision

This programme is **not ready for owner ratification or protected delivery** at the
audited revisions. The implementation has one coherent path for several major
capabilities, but the matrix contains failing entries. In particular:

1. `ExperimentDocument`, `CampaignDocument`, and `ResolvedExperiment` do not exist.
2. graph compilation emits source maps and provenance but no structured diagnostics;
3. neutral declarations have no declaration round-trip contract;
4. the durable controller has no retry-admission operation and no provider-inventory
   path for detecting or handling orphan resources;
5. the checked import rule does not cover the specification's complete scientific-core
   dependency boundary;
6. the retained Tier A result pins earlier Feedbax and rlrmp2 revisions and does not
   contain the required paired legacy/replacement differential evidence;
7. release-boundary Tier B has not run;
8. Feedbax publishes a SISU-specific workflow helper instead of leaving that science in
   rlrmp2; and
9. rlrmp2's exemplar supplies components and training programs through public
   declarations, but its trial and objective are not declared through the neutral
   trial/objective protocols.

Owner ratification can occur only when the owner approves the exact bytes proposed for
protected delivery. Authoring or committing this document is not ratification.

## Authority and audit boundary

| Authority | Exact pin |
|---|---|
| Governing specification | `[artifact-series:d541fc62fc90@3]`, managed blob SHA-256 `f2b08c90a1f2c51c5f90d2209ea0fb30abdf1efbf1a457703f5e8427284922f6` |
| Feedbax integration | `bb6ccc848fa521602d99b9b6ecc763c74c76f950` |
| Feedbax programme base | `5d3b300a69d49db1f039d32882150c4cff4a8bbc` |
| rlrmp2 integration | `0a62908edc1d9b5e099aacf28f6b2dad732a3702` |
| rlrmp2 programme base | `4d3f83370aeb29e8869a82c358bf539b9f5a53e0` |

The audit covered the exact two revision ranges above, every added, removed, renamed, or
modified programme path, public `__all__` surfaces introduced or changed by those diffs,
schema/type identities in the new compiler/workflow/execution/controller/publication
families, and all Feedbax-to-rlrmp2 references to those families. Existing generic
extension lifecycle coverage remains authoritative in
[`extension_coverage.md`](extension_coverage.md), especially `E1`, `E2`, `E10`, `E12`,
`E13`, and `E14`; this document does not copy its generic lifecycle taxonomy.

Evidence notation is deliberately executable:

- An `F` pointer names a Feedbax path followed by a literal token; the token must occur
  at that path in the pinned Feedbax revision.
- An `R` pointer uses the same form for the pinned rlrmp2 revision.
- `F-ABSENT:path` means the path must not exist in the pinned Feedbax tree.
- `F-NOTOKEN:scope::token` means the token must not occur below the scope in that tree.
- `A:series@version` identifies the governing Mandible artifact series version.

`PASS` means the named evidence proves the whole row. `FAIL` means the assigned
disposition is incomplete, stale, not genuinely disjoint, or unverified. A failed row is
not silently converted to a retained exception.

## Capability, declaration, schema, producer, and consumer inventory

Every inventory row has exactly one disposition. `MIGRATED` means the named target is
the current authority; `RETAINED` is allowed only for the named disjoint boundary and
owner; `DELETED` requires the absence proof in the deletion ledger below.

| ID | Inventory and current flow | Disposition | Owner and proof | Status |
|---|---|---|---|---|
| `I01` | Graph compiler public surface: `GraphDocument`, `DocumentRoot`, `ResolvedGraph`, `ExecutableGraph`, `CompilationRecord`, `GraphSourceMap`, `GraphSourceMapEntry`, `GraphKeySchedule`, `compile_graph`; schemas `feedbax.graph_document@1`, `feedbax.resolved_graph@2`, `feedbax.graph_compilation_record@2`, key schedule `feedbax.graph_key_schedule.execution_order_split.v1`. Producers are the CLI, analysis controller, Studio graph service, and worker; rlrmp2 is a consumer through its literal `GraphSpec`. | MIGRATED | Feedbax compiler. `F:feedbax/compiler/__init__.py::compile_graph`; `F:feedbax/compiler/graph.py::COMPILATION_RECORD_SCHEMA_ID`; `R:src/rlrmp2/adaptive_lambda/graph_model.py::sisu_graph_spec` | **FAIL**: no structured compiler diagnostic record or phased diagnostic contract. |
| `I02` | `WorkspaceDocument`, `SemanticAnchor`, and schema `feedbax.workspace_document@1`; Studio services produce it and the TypeScript client consumes the generated contract. | MIGRATED | Feedbax Studio. `F:feedbax/contracts/graph.py::class WorkspaceDocument`; `F:web/src/generated/studioContracts.ts::export interface WorkspaceDocument` | PASS |
| `I03` | Neutral declaration surface: `Declaration`, `DeclarationCatalog`, `DeclarationCompositionError`, `Facet`, `RuntimeFacet`, `CompilerFacet`, `AuthoringFacet`, `StudioFacet`, `SerializationFacet`, `OperationFacet`, `BackendFacet`, `scientific_declaration`, and `facet`. The catalog produces composed facets for consuming layers. | MIGRATED | Feedbax declarations. `F:feedbax/declarations/__init__.py::DeclarationCatalog`; `F:tests/test_scientific_declarations.py::test_catalog_composes_only_requested_layer_facets` | **FAIL**: no durable declaration serializer/loader or round-trip evidence. |
| `I04` | Component declaration surface: `DeclaredComponent`, `declare_component`, `ComponentCompilerFacet`, `ComponentRuntimeFacet`, `ComponentAuthoringFacet`, `ComponentStudioFacet`, `ComponentSerializationFacet`, `ComponentTrainingFacet`, `ComponentRegistry.register_component_type`. `ComponentRegistry` produces declarations; compiler, Studio, serialization, and training consume their own facets. | MIGRATED | Feedbax component registry. `F:feedbax/component_registry/declarations.py::class DeclaredComponent`; `F:feedbax/component_registry/registry.py::def register_component_type`; `F:docs/design/extension_coverage.md::E2` | PASS |
| `I05` | Training-program extension surface: `DeclaredTrainingProgram`, `TrainingProgramDeclaration`, five facet types, `declare_training_program`, `TrainingProgramRegistry`, `TrainingProgramCatalog`, and plugin family `TRAINING_PROGRAMS`. Plugins produce declarations; authoring, preparation, row lowering, projection, and runtime consume selected facets. | MIGRATED | Feedbax training. `F:feedbax/contracts/training.py::class DeclaredTrainingProgram`; `F:feedbax/plugins/application.py::TRAINING_PROGRAMS`; `F:docs/design/extension_coverage.md::E12` | PASS |
| `I06` | Trial/objective declaration surface: `TrialSourceProtocol`, `ObjectiveProtocol`, `ResolvedTrialSource`, `ResolvedObjective`. The intended producer is a downstream declaration catalog and training environment consumes the distinct resolved contracts. | MIGRATED | Feedbax declarations. `F:feedbax/declarations/science.py::class ResolvedTrialSource`; `F:tests/test_scientific_declarations.py::test_trial_and_objective_resolution_are_distinct_contracts` | **FAIL**: the SISU downstream consumer does not declare or resolve these contracts. |
| `I07` | Operation/backend declaration surface: `OperationProtocol`, `BackendProtocol`, `ResolvedOperation`, `ResolvedBackend`, plus `OrchestrationBackend` as the backend realization extension. Workflow operations and driver capability records are producers; invocation/backend realization is the consumer. | MIGRATED | Feedbax workflow and orchestration. `F:feedbax/declarations/science.py::class BackendProtocol`; `F:feedbax/orchestration/realization.py::class OrchestrationBackend` | PASS |
| `I08` | Workflow core surface: `WorkflowPlan`, `LogicalKey`, `Operation`, `PlanNode`, `PlanEdge`, `PlanGuard`, `GuardPredicate`, `NodeDeclaration`, `EdgeDeclaration`, `NodeExpander`, builders/readers; schema `feedbax.workflow.plan.v1`. Domain lowerers produce one plan and execution/invocation consume it. | MIGRATED | Feedbax workflow. `F:feedbax/workflow/plan.py::WORKFLOW_PLAN_SCHEMA_ID`; `F:tests/test_workflow_plan.py::test_dependency_closure_is_canonical_and_deduplicates_a_diamond` | PASS |
| `I09` | Workflow derivation and execution surface: compiled-output readers, `derive_workflow_plan`, `plan_experiment_workflow`, `execute_experiment_workflow`, operation lowering, and separate analysis/evaluation/report/campaign lowerers. Compiled locks produce typed declarations; the workflow executor consumes the finite closure. | MIGRATED | Feedbax workflow. `F:feedbax/workflow/derivation.py::derive_workflow_plan`; `F:tests/test_workflow_lowerers.py::test_sisu_exemplar_lowers_end_to_end_into_one_workflow_plan` | PASS |
| `I10` | Invocation surface: `Invocation`, `InvocationInput`, `InvocationOutput`, `InvocationExecutionPolicy`, loaders/builders; schema `feedbax.spec.invocation.v1`. Workflow nodes produce provider-neutral invocations; backend realization consumes them. | MIGRATED | Feedbax execution. `F:feedbax/execution/records.py::INVOCATION_SCHEMA_ID`; `F:tests/test_invocation_backend_realization.py::test_invocation_is_provider_neutral_versioned_and_identity_stable` | PASS |
| `I11` | Realization surface: `BackendPlan`, `BackendRealizationRequest`, `MachineShape`, `ExpectedCost`, `Attempt`, `OrchestrationBackend`, loaders; schemas `feedbax.orchestration.backend_plan.v1` and `feedbax.manifest.attempt.v1`. Backends produce plans and attempts; the controller consumes both. | MIGRATED | Feedbax orchestration. `F:feedbax/orchestration/realization.py::BACKEND_PLAN_SCHEMA_ID`; `F:tests/test_invocation_backend_realization.py::test_attempt_is_a_separate_versioned_observation` | PASS |
| `I12` | Controller surface: `RunIntent`, `EffectReservation`, `ControllerEvent`, `ControllerEventStore`, `ControllerProjection`, `ReservationProjection`, `EffectObservation`, `ProviderInventoryObservation`, `OrphanHandlingPolicy`, `EffectAdapter`, `DurableController`, loaders/projector; versioned controller schema families. Studio produces intents and authentication; adapters produce observations; the event projector is state authority. | MIGRATED | Feedbax orchestration. `F:feedbax/orchestration/controller.py::class DurableController`; `F:tests/test_durable_controller.py::test_retry_admission_enforces_invocation_policy_and_preserves_effect_key`; `F:tests/test_durable_controller.py::test_provider_inventory_detects_and_handles_orphan_replay_safely` | PASS |
| `I13` | Studio controller API and service: reserve launch, authenticate exact reservation, project status, request cancellation, and inspect artifacts. The HTTP API produces commands; the controller event log produces displayed state. | MIGRATED | Feedbax Studio. `F:feedbax/web/orchestration/controller.py::class StudioController`; `F:tests/test_studio_controller_api.py::test_launch_endpoint_stops_at_an_inert_named_reservation` | PASS for the implemented surface; controller gaps remain in `I12`. |
| `I14` | Custody protocol surface: `BlobRef`, `BlobStore`, `ExactRef`, `ArtifactRecord`, `ProvenanceEdge`, `CheckpointSlot`, `CheckpointSet`, `PublicationRequest`, `PublicationReceipt`, `PublicationCatalog`, `PublicationService`, and identity/build helpers. Schemas are `feedbax.publication.v1`, `feedbax.artifact_record.v1`, `feedbax.checkpoint_set.v1`, `feedbax.provenance_edge.v1`, and `feedbax.publication_receipt.v1`. | MIGRATED | Feedbax contracts. `F:feedbax/contracts/publication.py::class PublicationService`; `F:tests/test_publication_protocol.py::test_publication_rolls_back_every_logical_record_on_late_failure` | PASS |
| `I15` | Concrete publication storage: `LocalBlobStore` and `SQLitePublicationCatalog`. `PublicationService` writes them; exact-ref readers and the exemplar consume committed records. | MIGRATED | Feedbax persistence. `F:feedbax/persistence/publication.py::class LocalBlobStore`; `F:feedbax/persistence/publication.py::class SQLitePublicationCatalog` | PASS |
| `I16` | Existing `ImmutableArtifactBlobProvider` family and its stable manifest/report/evaluation consumers. It remains a content-addressed provider for the established manifest surface, not the new publication transaction catalog. | RETAINED | Feedbax artifact-custody owner; disjoint stable downstream boundary. `F:docs/design/downstream_interface_stability.md::report-surface`; `F:feedbax/persistence/artifact_custody.py::class ImmutableArtifactBlobProvider` | PASS |
| `I17` | Existing checkpoint transaction/custody contracts used by current training continuation. They remain the training executor's resumable store; `CheckpointSet` is the new compiler/publication semantic record. | RETAINED | Feedbax training custody owner; disjoint current training-runtime boundary. `F:feedbax/training/checkpoint_custody.py::class ResolvedCheckpointTransaction`; `R:tests/test_sisu_workflow_publication_adoption.py::_checkpoint_payload` | PASS, with retirement required when the last current producer/consumer leaves. |
| `I18` | Existing `RunSetState`, `RunSetStateStore`, and `StageEngine`. They remain the finite batch run-set execution projection; the new controller owns Studio intent/effect/reservation state. | RETAINED | Feedbax orchestration owner; the stable `orchestration-lifecycle` boundary names both responsibilities. `F:docs/design/downstream_interface_stability.md::finite StageEngine execution projection`; `F:feedbax/orchestration/__init__.py::StageEngine` | PASS as a named disjoint boundary, but it does not cure controller retry/orphan failures. |
| `I19` | Existing `ExperimentEnvelope`, compile locks, and compiled product records. They remain the current versioned authoring grammar and produce inputs lowered into `WorkflowPlan`; they are not aliases for the missing `ExperimentDocument` or `CampaignDocument`. | RETAINED | Feedbax envelope owner; disjoint authoring grammar. `F:docs/design/downstream_interface_stability.md::figure-role-references`; `F:feedbax/contracts/experiment_envelope.py::class ExperimentEnvelopeCompileRequest` | **FAIL** against the intended end state because the required typed roots remain absent. |
| `I20` | rlrmp2 component declarations: type IDs `rlrmp2.sisu.GruController`, `rlrmp2.sisu.CommandComposer`, `rlrmp2.sisu.TargetEnvelope`, `rlrmp2.sisu.ThresholdLatchedForce`; objective payload identity `rlrmp2.objective.cs_eq15@1`. `register_sisu_graph_components` produces registry entries and `sisu_graph_spec` consumes them. | MIGRATED | rlrmp2 adaptive-lambda owner. `R:src/rlrmp2/adaptive_lambda/graph_model.py::SISU_CONTROLLER_COMPONENT`; `R:src/rlrmp2/adaptive_lambda/graph_model.py::register_sisu_graph_components` | PASS for components; the objective is only a payload/service identity, not a neutral objective declaration. |
| `I21` | rlrmp2 training declarations: program IDs `rlrmp2/adaptive_lambda/v1`, `/v2`, `/v3`, `/v4`, and `rlrmp2/nominal_cs_gru/v1`; payload identities `rlrmp2.spec.training_method.adaptive_lambda` at selected v1/v7/v9 contracts and `rlrmp2.spec.training_method.nominal_cs_gru.v2`; composition rows `rlrmp2.spec.sisu_task_model.v1` and `.v2`. Plugin family requirements are only `COMPONENTS` and `TRAINING_PROGRAMS`. | MIGRATED | rlrmp2 adaptive-lambda owner. `R:src/rlrmp2/adaptive_lambda/method.py::adaptive_lambda_training_programs`; `R:src/rlrmp2/adaptive_lambda/method.py::PLUGIN_REGISTRATION` | PASS |
| `I22` | rlrmp2 SISU trial generation and Eq. 15 objective. Current producers are `HarmonizedCsTask`/training payload authoring and `SisuDeclaredObjectiveLossService`; current consumers are training preparation and graph rollout. | MIGRATED | rlrmp2 scientific owner. `R:src/rlrmp2/adaptive_lambda/graph_model.py::class SisuDeclaredObjectiveLossService`; `R:src/rlrmp2/adaptive_lambda/cs_task.py::class HarmonizedCsTask` | **FAIL**: neither is supplied through `TrialSourceProtocol`/`ObjectiveProtocol` declaration composition. |
| `I23` | rlrmp2 workflow/invocation/backend/publication adoption consumes `feedbax.workflow.*`, `feedbax.execution.records`, `feedbax.orchestration.realization`, and `feedbax.contracts.publication`; it produces one SISU workflow and replayed publication receipt. | MIGRATED | Coordinated Feedbax/rlrmp2 owners. `R:tests/test_sisu_workflow_publication_adoption.py::test_real_sisu_authoring_uses_one_workflow_and_distinct_realization_records` | **FAIL**: the producer `publish_sisu_artifact_chain` and fixed `SISU_ARTIFACT_CHAIN` live in Feedbax, placing downstream science in the platform. |
| `I24` | Downstream stability rows changed by the programme: `component-registration`, `graph-spec`, `graph-compiler`, `orchestration-lifecycle`, and the `training_programs` plugin family. The already-protected policy is the compatibility authority; these exact amendments are only proposed bytes until protected approval. | RETAINED | Feedbax owner; disjoint stable-contract authority. `F:docs/design/downstream_interface_stability.md::graph-compiler`; `F:docs/design/downstream_interface_stability.md::training_programs` | **FAIL pending owner action**: the exact programme amendments have not been owner-ratified; unlisted new modules also remain coordinated but not guaranteed downstream API. |

### Programme public-name manifest

The table above groups names only when they share one authority and one disposition. This
manifest is the completeness census for public names introduced or changed by the
programme:

- `I01`: every name in `feedbax/compiler/__init__.py::__all__`.
- `I02`: `SemanticAnchor`, `WorkspaceDocument`, `WORKSPACE_DOCUMENT_SCHEMA_ID`, and
  `WORKSPACE_DOCUMENT_SCHEMA_VERSION` from `feedbax.contracts.graph` and its root
  re-exports.
- `I03`, `I06`, `I07`: every name in `feedbax/declarations/__init__.py::__all__`.
- `I04`: the eight declaration/facet names added to
  `feedbax/component_registry/__init__.py::__all__`, plus the retained registry methods
  named in the stable component-registration row.
- `I05`: `TrainingProgramCatalog`, `TRAINING_PROGRAMS`, `DeclaredTrainingProgram`,
  `TrainingProgramDeclaration`, `TrainingProgramRuntimeFacet`,
  `TrainingProgramAuthoringFacet`, `TrainingProgramPreparationFacet`,
  `TrainingProgramProjectionFacet`, `TrainingProgramRowLoweringFacet`,
  `declare_training_program`, `TrainingProgramRegistry`, the four retained
  `TrainingMethod*` contribution/projector names, the four `ExecutionPreparation*`
  support names, and `compile_training_method_authoring` in the exact ordered plugin
  inventory.
- `I08`: every name in `feedbax/workflow/__init__.py::__all__`; the larger
  `feedbax.workflow.plan.__all__` is the same contract plus constants, readers, error
  types, and `NodeExpander`.
- `I09`: every name in `feedbax.workflow.derivation.__all__`,
  `feedbax.workflow.experiment.__all__`, and
  `feedbax.workflow.operation_execution.__all__`.
- `I10`: every name in `feedbax/execution/records.py::__all__` and its lazy root
  re-exports.
- `I11`: every name in `feedbax/orchestration/realization.py::__all__` and the matching
  additions to `feedbax.orchestration.__all__`.
- `I12`: every name in `feedbax/orchestration/controller.py::__all__` and the matching
  additions to `feedbax.orchestration.__all__`.
- `I14`: every name in `feedbax/contracts/publication.py::__all__` and the matching
  `feedbax.contracts` re-exports.
- `I15`: `LocalBlobStore`, `SQLitePublicationCatalog`, `PublicationError`,
  `PublicationConflictError`, and `UnsupportedPublicationSchemaError` in
  `feedbax.persistence`.
- `I23`: every name in `feedbax/workflow/publication.py::__all__`.

No programme-added public name is assigned to more than one inventory row.

## Deletion and absence ledger

| ID | Deleted predecessor | Disposition | Reason and absence proof | Status |
|---|---|---|---|---|
| `D01` | `ComponentMeta` and `feedbax/component_registry/meta.py` | DELETED | Replaced by neutral `DeclaredComponent` plus layer-local facets. `F-ABSENT:feedbax/component_registry/meta.py`; `F-NOTOKEN:feedbax/component_registry::ComponentMeta` | PASS |
| `D02` | Production `spec_to_graph` entry point | DELETED | All production construction routes through `compile_graph`; the same spelling remains only as a test helper that calls the compiler. `F-NOTOKEN:feedbax::def spec_to_graph`; `F:tests/graph_compiler_test_support.py::compile_graph` | PASS |
| `D03` | `feedbax.execution` backend, command, container, local, model, and planning modules | DELETED | Replaced by provider-neutral invocation records and backend realization. `F-ABSENT:feedbax/execution/backends.py`; `F-ABSENT:feedbax/execution/commands.py`; `F-ABSENT:feedbax/execution/container.py`; `F-ABSENT:feedbax/execution/local.py`; `F-ABSENT:feedbax/execution/models.py`; `F-ABSENT:feedbax/execution/planning.py` | PASS |
| `D04` | `FulfillmentPlan` and `feedbax.analysis.fulfillment_plan` | DELETED | Replaced by `WorkflowPlan`; the legacy schema ID survives only as a rejection sentinel. `F-ABSENT:feedbax/analysis/fulfillment_plan.py`; `F-NOTOKEN:feedbax::class FulfillmentPlan`; `F:tests/test_workflow_plan.py::test_schema_family_is_workflow_and_rejects_the_predecessor_explicitly` | PASS |
| `D05` | Plugin families `TRAINING_METHODS`, `ROW_LOWERERS`, `EXECUTION_PREPARATIONS` and their registry/registration classes | DELETED | Their one semantic unit is `TRAINING_PROGRAMS`; preparation and lowering are facets of one declaration. `F-NOTOKEN:feedbax/plugins::TRAINING_METHODS`; `F-NOTOKEN:feedbax/plugins::ROW_LOWERERS`; `F-NOTOKEN:feedbax/plugins::EXECUTION_PREPARATIONS` | PASS |
| `D06` | `feedbax.training.legacy_checkpoint_adoption` and its dedicated tests | DELETED | Unsupported archaeology is outside the runtime; current custody rejects or migrates at explicit owned boundaries. `F-ABSENT:feedbax/training/legacy_checkpoint_adoption.py`; `F-ABSENT:tests/test_legacy_checkpoint_adoption.py` | PASS |
| `D07` | Web `OrchestrationManager` direct lifecycle and its tests | DELETED | Studio now routes launch state through durable reservations/events. `F-ABSENT:feedbax/web/orchestration/manager.py`; `F-ABSENT:tests/test_orchestration_manager.py`; `F-NOTOKEN:feedbax/web::OrchestrationManager` | PASS |
| `D08` | Legacy execution-contract/local-embed tests and Modal golden fixture | DELETED | They protected the deleted execution representation. `F-ABSENT:tests/test_execution_contract.py`; `F-ABSENT:tests/test_execution_local_embed.py`; `F-ABSENT:tests/fixtures/execution/modal_no_embed_golden.txt` | PASS |
| `D09` | Manifest-artifact-storage test tied only to the predecessor publication representation | DELETED | Replaced invariants are covered at the common publication boundary. `F-ABSENT:tests/test_manifest_artifact_storage.py`; `F:tests/test_publication_protocol.py::test_publication_is_idempotent_and_conflicting_replay_fails_closed` | PASS |

## Section 15 evidence matrix

Every named Section 15 law is mapped below. A grouped row is used only when one test
proves the complete group.

| ID | Section 15 obligation | Disposition | Evidence | Status |
|---|---|---|---|---|
| `S15-01` | Tier A: paired legacy/replacement, adaptive mutation, split/uninterrupted equivalence, schedule positions, and fresh downstream consumption | MIGRATED | `R:results/5a0ef7e/tier-a-downstream-acceptance.v1.json::rlrmp2.evidence.tier_a_downstream_acceptance`; `R:tests/adaptive_lambda/test_sisu_method.py::legacy.states - replacement.states` | **FAIL**: the result pins rlrmp2 `4ce6550` and Feedbax `5f61d269`, not the audited heads, and records continuation/downstream evidence rather than the complete paired implementation differential contract. |
| `S15-02` | Complete Tier A wall clock no more than two minutes | MIGRATED | `R:results/5a0ef7e/tier-a-downstream-acceptance.v1.json::authoring_preflight_to_report_observed` | **FAIL**: 60.97 seconds passes the budget, but the evidence pins earlier code and is not current for the audited revisions. |
| `S15-03` | Clean paired Tier B at full pinned configuration | MIGRATED | Parent-owned release gate; no result path exists at the audited revisions. | **FAIL**: pending by design, but still an unsatisfied Section 15 obligation. |
| `S15-04` | Compiler deterministic resolution and key order | MIGRATED | `F:tests/test_graph_compiler.py::test_compile_graph_is_deterministic_and_records_runtime_key_order` | PASS |
| `S15-05` | Compiler canonical identity | MIGRATED | `F:feedbax/compiler/graph.py::_resolved_digest`; `F:tests/test_graph_compiler.py::test_workspace_view_edits_cannot_change_semantic_or_runtime_identity` | PASS |
| `S15-06` | Compiler source mapping | MIGRATED | `F:tests/test_graph_compiler.py::test_compiler_generated_adapter_has_truthful_source_map_origin` | PASS |
| `S15-07` | Compiler semantic/view separation | MIGRATED | `F:tests/test_graph_compiler.py::test_graph_document_and_compiler_reject_presentation_state` | PASS |
| `S15-08` | Structured compiler diagnostics | MIGRATED | `F:feedbax/compiler/graph.py::class CompilationRecord` | **FAIL**: the record has no diagnostics field and compile failures escape as builder/validator exceptions without phase, stable code, severity, source anchor, expected/observed condition, or actionable context. |
| `S15-09` | Compiler executable equivalence | MIGRATED | `R:tests/adaptive_lambda/test_sisu_method.py::state_error == pytest.approx`; `F:tests/graph_compiler_test_support.py::compile_graph` | PASS for focused literal-graph equivalence; full Tier A equivalence remains failed in `S15-01`. |
| `S15-10` | Declaration unique identities | MIGRATED | `F:feedbax/declarations/core.py::duplicate declaration identity`; `F:feedbax/contracts/training.py::training program already registered` | PASS |
| `S15-11` | Declaration facet completeness and atomic composition | MIGRATED | `F:tests/test_scientific_declarations.py::test_invalid_facet_compositions_fail_without_partial_registration` | PASS |
| `S15-12` | Declaration round trips | MIGRATED | `F:tests/test_scientific_declarations.py::class TrialSource` | **FAIL**: no declaration document schema, serializer/loader pair, or round-trip test exists. |
| `S15-13` | Declaration explicit composition | MIGRATED | `F:tests/test_scientific_declarations.py::test_catalog_composes_only_requested_layer_facets` | PASS |
| `S15-14` | Workflow finite closure, topological order, exact binding, certified omission, identity stability | MIGRATED | `F:tests/test_workflow_plan.py::test_dependency_closure_is_canonical_and_deduplicates_a_diamond`; `F:tests/test_workflow_plan.py::test_producer_and_external_bindings_are_type_checked`; `F:tests/test_workflow_plan.py::test_certified_omission_is_preserved_and_binds_nothing`; `F:tests/test_workflow_plan.py::test_round_trip_preserves_identity_and_origin_does_not_change_it` | PASS |
| `S15-15` | Invocation provider neutrality and backend capability rejection | MIGRATED | `F:tests/test_invocation_backend_realization.py::test_invocation_refuses_provider_and_physical_input_coordinates`; `F:tests/test_invocation_backend_realization.py::test_backend_capability_mismatch_fails_without_fallback` | PASS |
| `S15-16` | Controller inert reservation | MIGRATED | `F:tests/test_durable_controller.py::test_paid_reservation_expires_inert_without_adapter_contact` | PASS |
| `S15-17` | Reservation-bound authentication and expiry | MIGRATED | `F:tests/test_durable_controller.py::test_paid_reservation_expires_inert_without_adapter_contact`; `F:tests/test_studio_controller_api.py::test_authentication_rejects_evidence_for_an_unknown_reservation` | PASS |
| `S15-18` | Controller idempotent effects | MIGRATED | `F:feedbax/orchestration/controller.py::reservation may be dispatched only once`; `F:tests/test_durable_controller.py::test_local_effect_uses_same_controller_without_operator_gate` | PASS |
| `S15-19` | Controller startup reconciliation | MIGRATED | `F:tests/test_durable_controller.py::test_authenticated_ambiguous_dispatch_recovers_once_after_restart` | PASS |
| `S15-20` | Controller reconnect | MIGRATED | `F:tests/test_durable_controller.py::test_reconciliation_records_progressive_observations_until_terminal` | PASS |
| `S15-21` | Controller retry admission and policy | MIGRATED | `F:feedbax/orchestration/controller.py::def admit_retry`; `F:tests/test_durable_controller.py::test_retry_admission_enforces_invocation_policy_and_preserves_effect_key`; `F:tests/test_durable_controller.py::test_retry_is_forbidden_by_backend_plan_even_when_invocation_allows_it`; `F:tests/test_durable_controller.py::test_complete_inventory_proves_ambiguous_absence_before_same_key_retry` | PASS |
| `S15-22` | Controller cancellation | MIGRATED | `F:tests/test_durable_controller.py::test_studio_cancellation_is_a_separately_reserved_cleanup_effect` | PASS |
| `S15-23` | Controller expected and observed cost visibility | MIGRATED | `F:feedbax/orchestration/controller.py::observed_cost`; `F:tests/test_invocation_backend_realization.py::test_paid_capable_sisu_plan_is_inert_and_reservation_bound` | PASS |
| `S15-24` | Controller orphan detection and handling | MIGRATED | `F:feedbax/orchestration/controller.py::class ProviderInventoryObservation`; `F:feedbax/orchestration/controller.py::def observe_provider_inventory`; `F:tests/test_durable_controller.py::test_provider_inventory_detects_and_handles_orphan_replay_safely`; `F:tests/test_durable_controller.py::test_gcp_inventory_observation_is_complete_versioned_and_canonical` | PASS |
| `S15-25` | Paid-resource ambiguity remains unknown until reconciled | MIGRATED | `F:tests/test_durable_controller.py::test_authenticated_ambiguous_dispatch_recovers_once_after_restart` | PASS |
| `S15-26` | Custody immutable blobs, atomic publication, replay, provenance, containment, checkpoint completeness | MIGRATED | `F:tests/test_publication_protocol.py::test_local_blob_store_is_content_addressed_verified_and_idempotent`; `F:tests/test_publication_protocol.py::test_publication_rolls_back_every_logical_record_on_late_failure`; `F:tests/test_publication_protocol.py::test_sisu_exemplar_publishes_complete_chain_and_checkpoints_once` | PASS |
| `S15-27` | Studio save/reload no-volatility and semantic-hash independence | MIGRATED | `F:tests/test_studio_workspace.py::test_workspace_only_save_preserves_semantic_graph_revision`; `F:tests/test_studio_workspace.py::test_update_graph_preserves_explicit_workspace_extensions` | PASS |
| `S15-28` | Dependency arrows and scientific-core forbidden imports | MIGRATED | `F:tests/test_runtime_layering_boundary.py::FORBIDDEN_PACKAGES`; `F:tests/test_contract_import_boundary.py::test_importing_feedbax_does_not_load_web_package` | **FAIL**: existing tests cover runtime-to-Studio/models/tasks and import side effects, not the complete prohibition on scientific-core imports of Studio, controller, provider-adapter, persistence, and authoring packages. |
| `S15-29` | Matrix completeness: every discovered item and Sections 15/18 row has one disposition | MIGRATED | This document's `I`, `D`, `S15`, and `S18` manifests plus the checker below. | PASS structurally; substantive failing entries remain failures. |

## Section 18 end-state matrix

| ID | Section 18 acceptance obligation | Disposition | Evidence | Status |
|---|---|---|---|---|
| `S18-01` | One owner and typed identity domain for every authoritative root, IR, intent, reservation, attempt, event, artifact, and checkpoint | MIGRATED | `F:feedbax/compiler/graph.py::class GraphDocument`; `F:feedbax/orchestration/controller.py::class RunIntent`; `F:feedbax/orchestration/controller.py::class ControllerEventStore`; `F:feedbax/orchestration/controller.py::class ProviderInventoryObservation`; `F:feedbax/contracts/publication.py::class CheckpointSet` | **FAIL**: the controller part now has one typed append-only event authority, including retry and provider inventory/orphan state; `ExperimentDocument`, `CampaignDocument`, and `ResolvedExperiment` remain absent outside this lane. |
| `S18-02` | One install/release, optional platform extras, complete checked import rule | RETAINED | Feedbax distribution owner. `F:pyproject.toml::[project.optional-dependencies]`; `F:tests/test_contract_import_boundary.py::test_core_training_contract_imports_do_not_load_web_package` | **FAIL** because the complete scientific-core forbidden-import rule is not checked (`S15-28`). |
| `S18-03` | All model construction through `GraphDocument -> ResolvedGraph -> ExecutableGraph` with source maps/key schedule | MIGRATED | `F:feedbax/compiler/graph.py::def compile_graph`; `F:feedbax/contracts/graphs/serialization.py::def _instantiate_graph`; `F-NOTOKEN:feedbax::def spec_to_graph` | PASS: the remaining direct instantiator is private and recursively realizes nested graphs inside the compiler. |
| `S18-04` | Semantic and workspace state independently durable; view edits cannot change scientific identity | MIGRATED | `F:tests/test_graph_compiler.py::test_workspace_view_edits_cannot_change_semantic_or_runtime_identity`; `F:tests/test_studio_workspace.py::test_workspace_only_save_preserves_semantic_graph_revision` | PASS |
| `S18-05` | Trial generation, objective computation, and training compose through small resolved protocols | MIGRATED | `F:feedbax/declarations/science.py::class ResolvedObjective`; `F:feedbax/contracts/training.py::class DeclaredTrainingProgram` | **FAIL**: the generic types exist, but SISU trial/objective composition bypasses them (`I22`). |
| `S18-06` | All bounded multi-step work lowers to one finite `WorkflowPlan`; open-ended control remains event-driven | MIGRATED | `F:tests/test_workflow_lowerers.py::test_sisu_exemplar_lowers_end_to_end_into_one_workflow_plan`; `F:feedbax/orchestration/controller.py::class ControllerEventStore` | PASS |
| `S18-07` | Provider-neutral invocations and separately accountable attempts | MIGRATED | `F:tests/test_invocation_backend_realization.py::test_invocation_is_provider_neutral_versioned_and_identity_stable`; `F:tests/test_invocation_backend_realization.py::test_attempt_is_a_separate_versioned_observation` | PASS |
| `S18-08` | Every external effect reserved before dispatch, ambiguity reconciled, cloud attempts authorized by exact unexpired reservation | MIGRATED | `F:feedbax/orchestration/controller.py::async def dispatch`; `F:feedbax/orchestration/controller.py::def admit_retry`; `F:feedbax/orchestration/controller.py::def observe_provider_inventory`; `F:tests/test_durable_controller.py::test_authenticated_ambiguous_dispatch_recovers_once_after_restart`; `F:tests/test_durable_controller.py::test_complete_inventory_proves_ambiguous_absence_before_same_key_retry` | PASS |
| `S18-09` | Distinct blob, artifact, provenance, publication, and checkpoint protocols with atomic visibility | MIGRATED | `F:feedbax/contracts/publication.py::class BlobStore`; `F:feedbax/contracts/publication.py::class ArtifactRecord`; `F:feedbax/contracts/publication.py::class ProvenanceEdge`; `F:tests/test_publication_protocol.py::test_publication_rolls_back_every_logical_record_on_late_failure` | PASS |
| `S18-10` | rlrmp2 supplies science through public declarations without internal schema/migration imports | MIGRATED | `R:src/rlrmp2/adaptive_lambda/method.py::PLUGIN_REGISTRATION`; `R:src/rlrmp2/adaptive_lambda/graph_model.py::register_sisu_graph_components` | **FAIL**: components and training programs comply, but trial/objective declarations do not; the Feedbax-owned SISU publication helper also reverses the intended ownership direction. No rlrmp2 source import of `feedbax.contracts.migrations` was found. |
| `S18-11` | Current paired Tier A acceptance at exact revisions | MIGRATED | `R:results/5a0ef7e/tier-a-downstream-acceptance.v1.json::runtime` | **FAIL**: stale pins and incomplete paired differential evidence (`S15-01`). |
| `S18-12` | Current complete Tier A downstream wall clock within two minutes | MIGRATED | `R:results/5a0ef7e/tier-a-downstream-acceptance.v1.json::60.97` | **FAIL**: measured result is below the limit but not at the audited revisions (`S15-02`). |
| `S18-13` | Clean legacy Tier B reference and replacement confirmation on final programme candidate | MIGRATED | Parent-owned release gate. | **FAIL**: no final replacement confirmation is present; the parent must run the one release-time full-configuration gate. |
| `S18-14` | Owner-ratified revision-pinned matrix accounts for all inventory and obligations | MIGRATED | This document at the Feedbax pin plus checker. | **FAIL pending owner action**: exact protected-delivery bytes have not been approved, and owner ratification cannot be self-asserted. |
| `S18-15` | Every `DELETED` item, predecessor path, obsolete schema/migration-only mechanism, duplicate validator, and representation-only test is absent | DELETED | `D01`-`D09` and the absence checker. | PASS for every item actually classified `DELETED`; retained old contracts are separately named in `I16`-`I19` and are not counted as deleted. |

## Machine-checkable completeness and absence method

The four row-ID sets are closed for this revision:

```text
I = I01..I24
D = D01..D09
S15 = S15-01..S15-29
S18 = S18-01..S18-15
```

A checker must:

1. parse Markdown table rows whose first cell matches those four closed sets;
2. require each ID exactly once and exactly one disposition token from
   `MIGRATED|RETAINED|DELETED`;
3. require a non-empty status and evidence cell;
4. resolve every Feedbax or rlrmp2 path/token pointer with
   `git grep -F <token> <feedbax-pin> -- <path>` and every `R:` pointer likewise;
5. reject any existing `F-ABSENT:path` using
   `git cat-file -e <feedbax-pin>:<path>`; and
6. reject any `F-NOTOKEN:scope::token` for which
   `git grep -F <token> <feedbax-pin> -- <scope>` succeeds.

The discovery side is independently reproducible: enumerate the two pinned diffs with
`git diff --name-status <base> <pin>`, enumerate changed Python exports from `__all__`,
extract changed or added `*_SCHEMA_ID`, `*_SCHEMA_VERSION`, `*_TYPE`, component type, and
training program identities, then search both pinned trees for producers and consumers.
The census must equal the `I` manifest plus the `D` ledger. Any unmatched discovered
item, duplicate row ID, unresolved pointer, or deletion proof that finds bytes makes the
matrix fail.

## Required parent actions

The implementation lanes are not reopened by this document. The parent coordinator owns
the disposition of the failing entries, the one release-time Tier B run, the final full
suite, protected auth submission, and umbrella closure. Before requesting protected
delivery, the parent must at minimum:

1. decide whether to repair the missing typed roots, compiler diagnostics, declaration
   round trips, retry, orphan handling, and full import rule, or explicitly revise the
   governing specification through an owner decision;
2. move the SISU publication producer to rlrmp2 (or replace it with a genuinely generic
   Feedbax producer) and make SISU trial/objective extensions use the public declaration
   boundary;
3. regenerate current paired Tier A evidence at the exact consolidated revisions;
4. run the parent-owned clean Tier B replacement confirmation and final repository gate;
   and
5. obtain owner approval of the exact protected-delivery bytes, which alone ratifies
   this matrix.

Co-Authored-By: Codex (GPT-5) <codex@openai.com>
