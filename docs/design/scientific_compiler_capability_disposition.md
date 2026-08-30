# Scientific compiler capability and disposition matrix

## Decision

The audited candidate is **ready for owner ratification and protected-delivery review**.
Every implementation, architecture, Tier A, and Tier B obligation mapped below passes at
the exact candidate revisions. The final acceptance record reports all twelve authored
preflights green, the generic training-through-report chain green in 28.60 seconds, and
exact split/resume equivalence for the full Tier B configuration. Model, optimizer, PRNG,
adaptive state, batch counter, coordinate-addressed learning-rate values, method-trace
records, progress, and normalized training outputs match at zero tolerance. The serialized
learning-rate list differs only in deterministic record order: uninterrupted output is
replica-major and split output is segment-major.

The only failing row is the deliberately external owner-ratification gate in `S18-14`.
It is not an implementation or evidence gap: the owner has not yet approved the exact
protected-delivery bytes, so this document does not claim ratification.

Owner ratification can occur only when the owner approves the exact bytes proposed for
protected delivery. Authoring or committing this document is not ratification.

## Authority and audit boundary

| Authority | Exact pin |
|---|---|
| Governing specification | `[artifact-series:d541fc62fc90@3]`, managed blob SHA-256 `f2b08c90a1f2c51c5f90d2209ea0fb30abdf1efbf1a457703f5e8427284922f6` |
| Feedbax implementation candidate and `F` pointer authority | `d11d49fb2ec14340d445e445b90f7fb63d965031` |
| Feedbax programme base | `5d3b300a69d49db1f039d32882150c4cff4a8bbc` |
| rlrmp2 protected-delivery candidate | `0729229bd8c1c172e3e1adb6ee2bdc1332d3dfe5` |
| rlrmp2 implementation candidate | `93e900988c3d24dcb511669781eff00791517c85` |
| rlrmp2 signed acceptance evidence | `33bce4d708c86ef4f19ffdf45b49ab239659a157` |
| rlrmp2 programme base | `4d3f83370aeb29e8869a82c358bf539b9f5a53e0` |
| Final acceptance receipt | `[artifact-version:5bad14e33eaaa81f1ab901a6916ceb6a]`, SHA-256 `a419324620e4e9a26b113bcd0be8dd377f6be1d5a5a67f4f9386277a40d5c1c1` |

This matrix is content in the eventual Feedbax auth head. Its exact delivery SHA is
recorded by the parent-owned auth request after this signed document commit is merged into
that head; the document does not try to self-pin its own commit. Every `F` pointer remains
audited at `d11d49fb2ec14340d445e445b90f7fb63d965031`, because the matrix-only commit changes
no implementation surface.

The audit covered the exact Feedbax base-to-candidate and rlrmp2 base-to-protected-delivery
ranges above, every added, removed, renamed, or modified programme path, public `__all__`
surfaces introduced or changed by those diffs,
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
- `R-ABSENT` and `R-NOTOKEN` apply the same rules to the pinned rlrmp2 tree.
- `A:series@version` identifies the governing Mandible artifact series version.
- `A-VERSION:id::sha256` identifies one exact Mandible artifact version and the SHA-256
  that its materialized custody bytes must have.

`PASS` means the named evidence proves the whole row. `FAIL` means the assigned
disposition is incomplete, stale, not genuinely disjoint, unverified, or awaits an
explicit gate named by the governing specification. A failed row is not silently
converted to a retained exception.

## Capability, declaration, schema, producer, and consumer inventory

Every inventory row has exactly one disposition. `MIGRATED` means the named target is
the current authority; `RETAINED` is allowed only for the named disjoint boundary and
owner; `DELETED` requires the absence proof in the deletion ledger below.

| ID | Inventory and current flow | Disposition | Owner and proof | Status |
|---|---|---|---|---|
| `I01` | Graph subset of the compiler public surface: graph, compiler, record, and failure schema constants; `GraphDocument`, `DocumentRoot`, `ResolvedGraph`, `ExecutableGraph`, `CompilationRecord`, `CompilationFailureRecord`, `CompilerDiagnostic`, `CompilerPhase`, `DiagnosticSeverity`, `GraphCompilationError`, `GraphSourceMap`, `GraphSourceMapEntry`, `GraphKeySchedule`, and `compile_graph`. Schemas are `feedbax.graph_document@1`, `feedbax.resolved_graph@2`, `feedbax.graph_compilation_record@3`, and `feedbax.graph_compilation_failure@1`; the key schedule is `feedbax.graph_key_schedule.execution_order_split.v1`. Producers are the CLI, analysis controller, Studio graph service, and worker; rlrmp2 consumes the compiler through its literal `GraphSpec`. | MIGRATED | Feedbax compiler. `F:feedbax/compiler/__init__.py::GraphCompilationError`; `F:feedbax/compiler/graph.py::class CompilationFailureRecord`; `F:tests/test_graph_compiler.py::test_compile_failure_is_a_stable_source_mapped_phase_diagnostic` | PASS: expected compilation failures carry stable code, severity, one of the seven ordered phases, revision-pinned source anchor, expected and observed conditions, and actionable context. |
| `I02` | `WorkspaceDocument`, `SemanticAnchor`, and schema `feedbax.workspace_document@1`; Studio services produce it and the TypeScript client consumes the generated contract. | MIGRATED | Feedbax Studio. `F:feedbax/contracts/graph.py::class WorkspaceDocument`; `F:web/src/generated/studioContracts.ts::export interface WorkspaceDocument` | PASS |
| `I03` | Neutral declaration surface: `Declaration`, `DeclarationCatalog`, `DeclarationCompositionError`, `DeclarationDocument`, `Facet`, `RuntimeFacet`, `CompilerFacet`, `AuthoringFacet`, `StudioFacet`, `SerializationFacet`, `OperationFacet`, `BackendFacet`, `scientific_declaration`, `facet`, `serialize_declaration`, and `load_declaration`. The catalog produces composed facets for consuming layers. | MIGRATED | Feedbax declarations. `F:feedbax/declarations/document.py::class DeclarationDocument`; `F:tests/test_scientific_declarations.py::test_declaration_document_round_trip_preserves_neutral_authority`; `F:tests/test_scientific_declarations.py::test_declaration_document_rejects_unsupported_or_unstated_authority` | PASS: one canonical v1 document round-trips neutral authority through an explicit protocol registry; unsupported versions and unknown protocols reject. |
| `I04` | Component declaration surface: `DeclaredComponent`, `declare_component`, `ComponentCompilerFacet`, `ComponentRuntimeFacet`, `ComponentAuthoringFacet`, `ComponentStudioFacet`, `ComponentSerializationFacet`, `ComponentTrainingFacet`, `ComponentRegistry.register_component_type`. `ComponentRegistry` produces declarations; compiler, Studio, serialization, and training consume their own facets. | MIGRATED | Feedbax component registry. `F:feedbax/component_registry/declarations.py::class DeclaredComponent`; `F:feedbax/component_registry/registry.py::def register_component_type`; `F:docs/design/extension_coverage.md::E2` | PASS |
| `I05` | Training-program extension surface: `DeclaredTrainingProgram`, `TrainingProgramDeclaration`, five facet types, `declare_training_program`, `TrainingProgramRegistry`, `TrainingProgramCatalog`, and plugin family `TRAINING_PROGRAMS`. Plugins produce declarations; authoring, preparation, row lowering, projection, and runtime consume selected facets. | MIGRATED | Feedbax training. `F:feedbax/contracts/training.py::class DeclaredTrainingProgram`; `F:feedbax/plugins/application.py::TRAINING_PROGRAMS`; `F:docs/design/extension_coverage.md::E12` | PASS |
| `I06` | Trial/objective declaration surface: `TrialSourceProtocol`, `ObjectiveProtocol`, `ResolvedTrialSource`, and `ResolvedObjective`. Downstream declaration documents produce protocol-bound declarations and training consumes the distinct resolved contracts. | MIGRATED | Feedbax declarations and rlrmp2 scientific owner. `F:feedbax/declarations/science.py::class ResolvedTrialSource`; `F:tests/test_scientific_declarations.py::test_trial_and_objective_resolution_are_distinct_contracts`; `R:specs/declarations/sisu_trial_source.v1.json::feedbax.trial_source_protocol.v1`; `R:specs/declarations/sisu_eq15_objective.v1.json::feedbax.objective_protocol.v1`; `R:tests/adaptive_lambda/test_sisu_release_acceptance.py::test_release_declarations_bind_the_real_trial_source_and_eq15_objective` | PASS: both SISU authorities load through the neutral declaration document and resolve to the real runtime protocols. |
| `I07` | Operation/backend declaration surface: `OperationProtocol`, `BackendProtocol`, `ResolvedOperation`, `ResolvedBackend`, plus `OrchestrationBackend` as the backend realization extension. Workflow operations and driver capability records are producers; invocation/backend realization is the consumer. | MIGRATED | Feedbax workflow and orchestration. `F:feedbax/declarations/science.py::class BackendProtocol`; `F:feedbax/orchestration/realization.py::class OrchestrationBackend` | PASS |
| `I08` | Workflow core surface: `WorkflowPlan`, `LogicalKey`, `Operation`, `PlanNode`, `PlanEdge`, `PlanGuard`, `GuardPredicate`, `NodeDeclaration`, `EdgeDeclaration`, `NodeExpander`, builders/readers, and the closed `EDGE_BINDINGS` vocabulary. `feedbax.workflow.plan.v2` is current; v1 migrates active edges to explicit singular binding and recalculates identity. Domain lowerers produce one plan and execution/invocation consume it. | MIGRATED | Feedbax workflow. `F:feedbax/workflow/plan.py::WORKFLOW_PLAN_SCHEMA_VERSION_V2`; `F:tests/test_workflow_plan.py::test_v1_plan_migration_makes_active_edges_singular_and_recalculates_identity`; `F:tests/test_workflow_plan.py::test_v1_plan_cannot_be_relabelled_with_v2_binding_grammar` | PASS: the only active binding modes are `single_receipt` and `complete_receipt_set`; unsupported or falsely relabelled versions reject. |
| `I09` | Workflow derivation and execution surface: compiled-output readers, `derive_workflow_plan`, `plan_experiment_workflow`, `execute_experiment_workflow`, operation lowering, and separate analysis/evaluation/report/campaign lowerers. Compiled locks produce typed declarations; the workflow executor consumes the finite closure. | MIGRATED | Feedbax workflow. `F:feedbax/workflow/derivation.py::derive_workflow_plan`; `F:tests/test_workflow_lowerers.py::test_each_authoring_domain_owns_a_distinct_lowerer` | PASS |
| `I10` | Invocation surface: `Invocation`, `InvocationInput`, `InvocationOutput`, `InvocationExecutionPolicy`, loaders/builders; schema `feedbax.spec.invocation.v1`. Workflow nodes produce provider-neutral invocations; backend realization consumes them. | MIGRATED | Feedbax execution. `F:feedbax/execution/records.py::INVOCATION_SCHEMA_ID`; `F:tests/test_invocation_backend_realization.py::test_invocation_is_provider_neutral_versioned_and_identity_stable` | PASS |
| `I11` | Realization surface: `BackendPlan`, `BackendRealizationRequest`, `MachineShape`, `ExpectedCost`, `Attempt`, `OrchestrationBackend`, loaders; schemas `feedbax.orchestration.backend_plan.v1` and `feedbax.manifest.attempt.v1`. Backends produce plans and attempts; the controller consumes both. | MIGRATED | Feedbax orchestration. `F:feedbax/orchestration/realization.py::BACKEND_PLAN_SCHEMA_ID`; `F:tests/test_invocation_backend_realization.py::test_attempt_is_a_separate_versioned_observation` | PASS |
| `I12` | Controller surface: `RunIntent`, `EffectReservation`, `ControllerEvent`, `ControllerEventStore`, `ControllerProjection`, `ReservationProjection`, `EffectObservation`, `ProviderInventoryObservation`, `OrphanHandlingPolicy`, `EffectAdapter`, `DurableController`, loaders/projector; versioned controller schema families. Studio produces intents and authentication; adapters produce observations; the event projector is state authority. | MIGRATED | Feedbax orchestration. `F:feedbax/orchestration/controller.py::class DurableController`; `F:tests/test_durable_controller.py::test_retry_admission_enforces_invocation_policy_and_preserves_effect_key`; `F:tests/test_durable_controller.py::test_provider_inventory_detects_and_handles_orphan_replay_safely` | PASS |
| `I13` | Studio controller API and service: reserve launch, authenticate exact reservation, project status, request cancellation, and inspect artifacts. The HTTP API produces commands; the controller event log produces displayed state. | MIGRATED | Feedbax Studio. `F:feedbax/web/orchestration/controller.py::class StudioController`; `F:tests/test_studio_controller_api.py::test_launch_endpoint_stops_at_an_inert_named_reservation` | PASS |
| `I14` | Custody protocol surface: `BlobRef`, `BlobStore`, `ExactRef`, `ArtifactRecord`, `ProvenanceEdge`, `CheckpointSlot`, `CheckpointSet`, `PublicationRequest`, `PublicationReceipt`, `PublicationCatalog`, `PublicationService`, and identity/build helpers. Schemas are `feedbax.publication.v1`, `feedbax.artifact_record.v1`, `feedbax.checkpoint_set.v1`, `feedbax.provenance_edge.v1`, and `feedbax.publication_receipt.v1`. | MIGRATED | Feedbax contracts. `F:feedbax/contracts/publication.py::class PublicationService`; `F:tests/test_publication_protocol.py::test_publication_rolls_back_every_logical_record_on_late_failure` | PASS |
| `I15` | Concrete publication storage: `LocalBlobStore` and `SQLitePublicationCatalog`. `PublicationService` writes them and generic exact-ref readers consume committed records. | MIGRATED | Feedbax persistence. `F:feedbax/persistence/publication.py::class LocalBlobStore`; `F:feedbax/persistence/publication.py::class SQLitePublicationCatalog` | PASS |
| `I16` | Existing `ImmutableArtifactBlobProvider` family and its stable manifest/report/evaluation consumers. It remains a content-addressed provider for the established manifest surface, not the new publication transaction catalog. | RETAINED | Feedbax artifact-custody owner; disjoint stable downstream boundary. `F:docs/design/downstream_interface_stability.md::report-surface`; `F:feedbax/persistence/artifact_custody.py::class ImmutableArtifactBlobProvider` | PASS |
| `I17` | Existing checkpoint transaction/custody contracts used by current training continuation. They remain the training executor's resumable store; `CheckpointSet` is the new compiler/publication semantic record. | RETAINED | Feedbax training custody owner; disjoint current training-runtime boundary. `F:feedbax/training/checkpoint_custody.py::class ResolvedCheckpointTransaction`; `R:tests/test_sisu_workflow_publication_adoption.py::_checkpoint_payload` | PASS, with retirement required when the last current producer/consumer leaves. |
| `I18` | Existing `RunSetState`, `RunSetStateStore`, and `StageEngine`. They remain the finite batch run-set execution projection; the new controller owns Studio intent/effect/reservation state. | RETAINED | Feedbax orchestration owner; the stable `orchestration-lifecycle` boundary names both responsibilities. `F:docs/design/downstream_interface_stability.md::finite StageEngine execution projection`; `F:feedbax/orchestration/__init__.py::StageEngine` | PASS: the batch projection and event-driven controller are named disjoint boundaries with no fallback between them. |
| `I19` | Existing `ExperimentEnvelope`, compile locks, and compiled product records remain the versioned authoring grammar and produce inputs lowered into `WorkflowPlan`. Envelope v6, compiler contract v4, and compile-lock v3 are current; their new receipt-set and report-role statements are owned by `I27` and `I28`, and explicit migration or rejection covers every older grammar. The compiler roots in `I25` are a separate identity domain, not aliases for the envelope. | RETAINED | Disjoint Feedbax envelope owner. `F:feedbax/contracts/experiment_envelope_dialect.py::EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V6`; `F:feedbax/contracts/experiment_compile_lock.py::EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V3`; `F:tests/test_experiment_envelope_dialect.py::test_the_v1_to_current_migration_is_explicit_and_semantics_preserving`; `F:tests/test_compile_lock_references.py::test_a_pre_v3_lock_cannot_be_relabelled_as_a_receipt_set` | PASS: authoring grammar, compile lock, and compiler roots have separate schema identities, owners, and lifetimes; no old document is silently restamped. |
| `I20` | rlrmp2 component declarations: type IDs `rlrmp2.sisu.GruController`, `rlrmp2.sisu.CommandComposer`, `rlrmp2.sisu.TargetEnvelope`, and `rlrmp2.sisu.ThresholdLatchedForce`. `register_sisu_graph_components` produces registry entries and `sisu_graph_spec` consumes them. | MIGRATED | rlrmp2 adaptive-lambda owner. `R:src/rlrmp2/adaptive_lambda/graph_model.py::SISU_CONTROLLER_COMPONENT`; `R:src/rlrmp2/adaptive_lambda/graph_model.py::register_sisu_graph_components` | PASS |
| `I21` | rlrmp2 training declarations: program IDs `rlrmp2/adaptive_lambda/v1`, `/v2`, `/v3`, `/v4`, and `rlrmp2/nominal_cs_gru/v1`; payload identities `rlrmp2.spec.training_method.adaptive_lambda` at selected v1/v7/v9 contracts and `rlrmp2.spec.training_method.nominal_cs_gru.v2`; composition rows `rlrmp2.spec.sisu_task_model.v1` and `.v2`. Plugin family requirements are only `COMPONENTS` and `TRAINING_PROGRAMS`. | MIGRATED | rlrmp2 adaptive-lambda owner. `R:src/rlrmp2/adaptive_lambda/method.py::adaptive_lambda_training_programs`; `R:src/rlrmp2/adaptive_lambda/method.py::PLUGIN_REGISTRATION` | PASS |
| `I22` | rlrmp2 SISU trial generation and Eq. 15 objective. `SisuTrialSource` and `SisuObjective` are declared as `rlrmp2.sisu.trial_source` and `rlrmp2.sisu.eq15`; training preparation and graph rollout consume their resolved protocols. | MIGRATED | rlrmp2 scientific owner. `R:specs/declarations/sisu_trial_source.v1.json::rlrmp2.sisu.trial_source`; `R:specs/declarations/sisu_eq15_objective.v1.json::rlrmp2.sisu.eq15`; `R:tests/adaptive_lambda/test_sisu_release_acceptance.py::test_release_declarations_form_the_compiler_owned_experiment_root` | PASS |
| `I23` | rlrmp2 owns SISU workflow, invocation, backend, and publication composition and consumes the generic `feedbax.workflow.*`, `feedbax.execution.records`, `feedbax.orchestration.realization`, and `feedbax.contracts.publication` contracts. Its six-role declaration composes one generic `PublicationRequest`; Feedbax owns no SISU publication producer or fixed artifact-role chain. | MIGRATED | rlrmp2 scientific publication owner, tracked by [issue:rlrmp2/8214374]. `R:specs/publication/sisu_release_artifact_chain.v1.json::rlrmp2.spec.sisu_release_artifact_chain`; `R:tests/test_sisu_workflow_publication_adoption.py::test_real_sisu_custody_publishes_and_replays_the_exact_chain`; `R:results/12a9d27/final-release-acceptance.v1.json::catalog_replay_verified`; `A-VERSION:artifact-version:5bad14e33eaaa81f1ab901a6916ceb6a::a419324620e4e9a26b113bcd0be8dd377f6be1d5a5a67f4f9386277a40d5c1c1`; `F-ABSENT:feedbax/workflow/publication.py` | PASS: six generic artifact roles, three checkpoint sets, and twelve provenance edges committed; catalog replay and blob/CAS materialization passed without a cross-tier derivation edge or SISU special case. |
| `I24` | Downstream stability rows changed by the programme: `component-registration`, `graph-spec`, `graph-compiler`, `orchestration-lifecycle`, and the `training_programs` plugin family. The already-protected policy is the compatibility authority; the candidate amendment is explicitly separate until protected approval. | RETAINED | Feedbax owner; disjoint stable-contract authority. `F:docs/design/downstream_interface_stability.md::graph-compiler`; `F:docs/design/downstream_interface_stability.md::training_programs`; `F:docs/design/downstream_interface_stability.md::Proposed scientific compiler contract amendment` | PASS: the document truthfully distinguishes current owner-ratified guarantees from proposed bytes. Ratification of those exact candidate bytes remains the external `S18-14` gate. |
| `I25` | Scientific compiler-root surface: experiment, campaign, and resolved-experiment schema constants; `ExperimentDocument`, `CampaignDocument`, `ResolvedExperiment`, `DeclarationRef`, `ScientificSeedDomain`, `RequestedProduct`, `BoundedBranch`, `CampaignVariant`, and `resolve_experiment`. | MIGRATED | Feedbax compiler. `F:feedbax/compiler/experiment.py::class ExperimentDocument`; `F:feedbax/compiler/experiment.py::class CampaignDocument`; `F:feedbax/compiler/experiment.py::class ResolvedExperiment`; `F:tests/test_scientific_compiler_roots.py::test_experiment_resolves_to_one_content_pinned_identity` | PASS: each authoritative root has one typed schema identity and one compiler owner. |
| `I26` | Authenticated evaluation-state resolution surface: `EvaluationStatesResolutionRequest`, `EvaluationStatesResolver`, `ResolvedEvaluationStates`, and `resolve_evaluation_states`. Evaluation and channel-evidence consumers reuse one authenticated state snapshot and authority instead of performing parallel resolution. | MIGRATED | Feedbax analysis execution context. `F:feedbax/analysis/execution_context.py::class EvaluationStatesResolutionRequest`; `F:feedbax/analysis/execution_context.py::def resolve_evaluation_states`; `F:tests/test_staged_evaluation_states_context.py::test_identical_authenticated_states_are_decoded_once_and_reused_immutably`; `F:tests/test_channel_evidence.py::test_authenticated_channels_reuse_trial_bank_material_authority` | PASS: one public resolver owns loading, memoization, material identity, and channel-evidence reuse; drift and authority substitution fail closed. |
| `I27` | Complete receipt-set consumer declaration: `AnalysisReceiptSetBinding` is the one typed authoring/lock identity that compiles to `PlanEdge.binding == "complete_receipt_set"`. Envelope v6 and compile-lock v3 produce it; workflow derivation admits it only for set-shaped products, execution binds the complete ordered unique receipt set, and singular bindings remain fail-closed even when a set contains one receipt. | MIGRATED | Feedbax envelope/workflow owners. `F:feedbax/contracts/experiment_compile_lock.py::class AnalysisReceiptSetBinding`; `F:tests/test_workflow_derivation.py::test_receipt_set_binding_is_derived_only_for_a_set_valued_product`; `F:tests/test_workflow_execution.py::test_a_matrix_receipt_set_executes_analysis_in_row_order_and_rebuilds`; `F:tests/test_workflow_execution.py::test_a_one_row_matrix_is_still_ambiguous_for_a_singular_edge`; `R:tests/test_sisu_workflow_publication_adoption.py::complete_receipt_set` | PASS: the final Tier A execution consumed all 100 authenticated evaluation receipts through this generic boundary. |
| `I28` | Ordered-report parent-role projection: `ReportParentBinding.parent_id` carries the authored report input role, while the fulfilled receipt supplies the real parent kind and manifest identity. The envelope compiler produces the role and workflow execution consumes it; concrete ordered-report targets without a role reject. | MIGRATED | Feedbax envelope/workflow owners. `F:feedbax/contracts/experiment_compile_lock.py::class ReportParentBinding`; `F:tests/test_envelope_engine_kernel.py::test_a_report_binding_refuses_a_target_without_an_authored_input_role`; `F:tests/test_workflow_execution.py::test_an_ordered_report_binds_a_figure_under_its_authored_role`; `R:tests/test_sisu_workflow_publication_adoption.py::peak_velocity` | PASS: the final Tier A report retained authored role `peak_velocity` through generic fulfillment. |
| `I29` | rlrmp2 velocity analysis consumes the canonical direct evaluation-row authority `metadata.matrix_row_id`; its producer is the native Feedbax evaluation manifest and its consumer is the complete-receipt-set analysis projection. | MIGRATED | rlrmp2 post-run owner. `R:src/rlrmp2/post_run/velocity_analysis.py::manifest_row_id = row.metadata.get("matrix_row_id")`; `R:tests/post_run/test_velocity_analysis.py::assert "matrix_harness" not in manifest.metadata`; `R:results/12a9d27/final-release-acceptance.v1.json::evaluation_manifests` | PASS: the native 100-row product reached analysis without a project wrapper or duplicate authority. |
| `I30` | Scientific-runtime utility ownership: the root `WhereDict` export resolves to `feedbax.runtime.mapping`; selector parsing lives in `feedbax.runtime.where_selectors`; and `feedbax.runtime.naming.__all__` exports `get_unique_label`. Runtime components consume these utilities without importing the authoring/configuration package. | MIGRATED | Feedbax runtime owner. `F:feedbax/__init__.py::feedbax.runtime.mapping`; `F:feedbax/runtime/naming.py::__all__ = ["get_unique_label"]`; `F:feedbax/runtime/mapping.py::from feedbax.runtime.where_selectors import where_func_to_attr_str_tree` | PASS: the runtime target is the only implementation; predecessor paths are deleted in `D11`. |
| `I31` | Native training diagnostics public surface: all constants, record types, errors, and the loader in `feedbax.training.diagnostics.__all__`; durable schema `feedbax.manifest.training_diagnostics.v4` and the governed method-trace artifact. The native executor produces it, and continuation/release comparison consumes it through `load_method_training_trace`. | MIGRATED | Feedbax training owner. `F:feedbax/training/diagnostics.py::TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V4`; `F:feedbax/training/diagnostics.py::def load_method_training_trace`; `F:tests/test_method_training_trace_loader.py::test_load_method_training_trace_round_trips_governed_payload`; `R:results/12a9d27/final-release-acceptance.v1.json::method_trace_records_equal` | PASS: the final Tier B comparison consumed all 5,000 current v5 method-trace records and found exact named values. |
| `I32` | Scientific-core immutable-byte dependency: `ArtifactBlobProvider` and the `ArtifactBlob*Error` family are owned by `feedbax.contracts.artifact_custody`; persistence supplies `ImmutableArtifactBlobProvider` and re-exports those exact error identities. Training diagnostics and checkpoint custody consume only the protocol. | MIGRATED | Feedbax contracts owner. `F:feedbax/contracts/artifact_custody.py::class ArtifactBlobProvider`; `F:feedbax/training/diagnostics.py::provider: ArtifactBlobProvider`; `F:feedbax/training/checkpoint_custody.py::artifact_provider: ArtifactBlobProvider`; `F:tests/test_artifact_custody.py::test_public_imports_do_not_load_web_package` | PASS: one protocol and one error identity cross the scientific-core/persistence boundary; there is no duplicate persistence definition. |

### Programme public-name manifest

The table above groups names only when they share one authority and one disposition. This
manifest is the completeness census for public names introduced or changed by the
programme:

- `I01`: the graph/compiler/record/failure constants and names in
  `feedbax/compiler/__init__.py::__all__`; the experiment/campaign subset is assigned to
  `I25`.
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
- `I19`: `EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V6`,
  `EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION_V4`, and
  `EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V3` in `feedbax.envelope`.
- `I25`: the experiment/campaign constants and names in
  `feedbax/compiler/__init__.py::__all__`.
- `I26`: `EvaluationStatesResolutionRequest`, `EvaluationStatesResolver`,
  `ResolvedEvaluationStates`, and `resolve_evaluation_states` in `feedbax.analysis` and
  `feedbax.analysis.execution_context`.
- `I27` and `I28`: `AnalysisReceiptSetBinding` and `ReportParentBinding` in
  `feedbax.contracts.experiment_compile_lock.__all__`.
- `I30`: `get_unique_label` in `feedbax.runtime.naming.__all__`; the stable root
  `WhereDict` name retains its identity while its implementation owner moves to the runtime.
- `I31`: every name in `feedbax.training.diagnostics.__all__`.
- `I32`: `ArtifactBlobProvider` and the five `ArtifactBlob*` protocol/error names in
  `feedbax.contracts.artifact_custody.__all__`; the error names at `feedbax.persistence`
  are exact re-exports, not second definitions.
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
| `D10` | Feedbax-owned SISU publication module and exemplar tests | DELETED | rlrmp2 owns the scientific artifact chain and composes the generic publication contracts directly. `F-ABSENT:feedbax/workflow/publication.py`; `F-NOTOKEN:tests::publish_sisu_artifact_chain`; `F-NOTOKEN:tests::SISU_ARTIFACT_CHAIN` | PASS |
| `D11` | Public implementation paths `feedbax.config.mapping` and `feedbax.config.selectors` | DELETED | Scientific-core utilities moved to the runtime owner; the root `WhereDict` export points only to `feedbax.runtime.mapping`. `F-ABSENT:feedbax/config/mapping.py`; `F-ABSENT:feedbax/config/selectors.py`; `F:feedbax/__init__.py::feedbax.runtime.mapping`; `F:tests/test_scientific_core_layering.py::test_scientific_core_has_no_platform_or_authoring_imports` | PASS |
| `D12` | rlrmp2's duplicate direct manifest lookup and producer-metadata validator after authenticated channel resolution | DELETED | `resolve_authenticated_evaluation_channels` and `resolve_evaluation_states` are the one authority; a second strict lookup repeated the same decision and rejected legitimate role restatement. `R-NOTOKEN:src/rlrmp2/comparators/trial_materialization.py::execution_context.resolve_manifest_input(parent)`; `R-NOTOKEN:tests/comparators/test_trial_materialization.py::test_public_feedbax_resolution_preserves_distinct_evidence`; `R:tests/comparators/test_trial_materialization.py::test_public_feedbax_resolution_uses_authenticated_role_transition` | PASS |
| `D13` | rlrmp2's nested `metadata.matrix_harness.row_id` representation | DELETED | Feedbax's native evaluation manifest already owns the direct `metadata.matrix_row_id`; the nested project wrapper duplicated that authority and masked the production shape. `R-NOTOKEN:src/rlrmp2/post_run/velocity_analysis.py::matrix_harness`; `R:src/rlrmp2/post_run/velocity_analysis.py::metadata.get("matrix_row_id")` | PASS |
| `D14` | Persistence-owned duplicate definitions of the `ArtifactBlob*Error` family | DELETED | The portable contract now owns the identities and persistence imports/re-exports them. `F-NOTOKEN:feedbax/persistence/artifact_custody.py::class ArtifactBlobCustodyError`; `F-NOTOKEN:feedbax/persistence/artifact_custody.py::class ArtifactBlobContainmentError`; `F:feedbax/contracts/artifact_custody.py::class ArtifactBlobCustodyError` | PASS |

## Section 15 evidence matrix

Every named Section 15 law is mapped below. A grouped row is used only when one test
proves the complete group.

| ID | Section 15 obligation | Disposition | Evidence | Status |
|---|---|---|---|---|
| `S15-01` | Tier A: paired legacy/replacement, adaptive mutation, split/uninterrupted equivalence, schedule positions, and fresh downstream consumption | MIGRATED | `R:results/12a9d27/experiment-spec-lock.v1.json::legacy_authorities`; `R:results/12a9d27/final-release-acceptance.v1.json::"accepted": true`; `R:tests/adaptive_lambda/test_sisu_method.py::legacy.states - replacement.states`; `R:tests/adaptive_lambda/test_sisu_release_acceptance.py::test_native_executor_rehearses_one_update_chunks_and_same_program_resume`; `A-VERSION:artifact-version:5bad14e33eaaa81f1ab901a6916ceb6a::a419324620e4e9a26b113bcd0be8dd377f6be1d5a5a67f4f9386277a40d5c1c1` | PASS: the exact candidate passed all 12 preflights and the generic fresh continuation-to-report chain; the acceptance lock pins the legacy authorities and zero-tolerance comparison contract. |
| `S15-02` | Complete Tier A wall clock no more than two minutes | MIGRATED | `R:results/12a9d27/final-release-acceptance.v1.json::"wall_seconds": 28.60`; `R:results/12a9d27/final-release-acceptance.v1.json::"wall_seconds_max": 120` | PASS: the complete 100-evaluation-manifest chain through analysis, figure, and report completed in 28.60 seconds. |
| `S15-03` | Clean paired Tier B at full pinned configuration | MIGRATED | `R:results/12a9d27/experiment-spec-lock.v1.json::"uninterrupted_updates": 1000`; `R:results/12a9d27/final-release-acceptance.v1.json::"terminal_checkpoint_slot_bytes_equal": true`; `R:results/12a9d27/release-acceptance.verdict.v8.json::"verdict": "answered"`; `A-VERSION:artifact-version:5bad14e33eaaa81f1ab901a6916ceb6a::a419324620e4e9a26b113bcd0be8dd377f6be1d5a5a67f4f9386277a40d5c1c1` | PASS: uninterrupted 1000, prefix 500, and continuation 500 ran once at the full locked shape; all five terminal slots and normalized outputs match at `rtol=atol=0`. The learning-rate list order difference is fully characterized and value-preserving. |
| `S15-04` | Compiler deterministic resolution and key order | MIGRATED | `F:tests/test_graph_compiler.py::test_compile_graph_is_deterministic_and_records_runtime_key_order` | PASS |
| `S15-05` | Compiler canonical identity | MIGRATED | `F:feedbax/compiler/graph.py::_resolved_digest`; `F:tests/test_graph_compiler.py::test_workspace_view_edits_cannot_change_semantic_or_runtime_identity` | PASS |
| `S15-06` | Compiler source mapping | MIGRATED | `F:tests/test_graph_compiler.py::test_compiler_generated_adapter_has_truthful_source_map_origin` | PASS |
| `S15-07` | Compiler semantic/view separation | MIGRATED | `F:tests/test_graph_compiler.py::test_graph_document_and_compiler_reject_presentation_state` | PASS |
| `S15-08` | Structured compiler diagnostics | MIGRATED | `F:feedbax/compiler/graph.py::class CompilerDiagnostic`; `F:feedbax/compiler/graph.py::class CompilationFailureRecord`; `F:tests/test_graph_compiler.py::test_compile_failure_is_a_stable_source_mapped_phase_diagnostic` | PASS |
| `S15-09` | Compiler executable equivalence | MIGRATED | `R:tests/adaptive_lambda/test_sisu_method.py::state_error == pytest.approx`; `F:tests/graph_compiler_test_support.py::compile_graph`; `R:results/12a9d27/release-acceptance.verdict.v8.json::current_same_program_exact` | PASS: focused literal-graph equivalence and final candidate acceptance both pass. |
| `S15-10` | Declaration unique identities | MIGRATED | `F:feedbax/declarations/core.py::duplicate declaration identity`; `F:feedbax/contracts/training.py::training program already registered` | PASS |
| `S15-11` | Declaration facet completeness and atomic composition | MIGRATED | `F:tests/test_scientific_declarations.py::test_invalid_facet_compositions_fail_without_partial_registration` | PASS |
| `S15-12` | Declaration round trips | MIGRATED | `F:feedbax/declarations/document.py::serialize_declaration`; `F:feedbax/declarations/document.py::load_declaration`; `F:tests/test_scientific_declarations.py::test_declaration_document_round_trip_preserves_neutral_authority` | PASS |
| `S15-13` | Declaration explicit composition | MIGRATED | `F:tests/test_scientific_declarations.py::test_catalog_composes_only_requested_layer_facets` | PASS |
| `S15-14` | Workflow finite closure, topological order, exact binding, certified omission, identity stability | MIGRATED | `F:tests/test_workflow_plan.py::test_dependency_closure_is_canonical_and_deduplicates_a_diamond`; `F:tests/test_workflow_plan.py::test_producer_and_external_bindings_are_type_checked`; `F:tests/test_workflow_plan.py::test_certified_omission_is_preserved_and_binds_nothing`; `F:tests/test_workflow_plan.py::test_round_trip_preserves_identity_and_origin_does_not_change_it`; `F:tests/test_workflow_execution.py::test_a_partial_or_corrupt_matrix_set_never_reaches_analysis` | PASS: exact binding includes the typed complete-set case, while partial, corrupt, duplicate, empty, and singular-to-set cases refuse. |
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
| `S15-26` | Custody immutable blobs, atomic publication, replay, provenance, containment, checkpoint completeness | MIGRATED | `F:tests/test_publication_protocol.py::test_local_blob_store_is_content_addressed_verified_and_idempotent`; `F:tests/test_publication_protocol.py::test_publication_rolls_back_every_logical_record_on_late_failure`; `F:tests/test_publication_protocol.py::test_publication_commits_one_complete_checkpoint_set` | PASS |
| `S15-27` | Studio save/reload no-volatility and semantic-hash independence | MIGRATED | `F:tests/test_studio_workspace.py::test_workspace_only_save_preserves_semantic_graph_revision`; `F:tests/test_studio_workspace.py::test_update_graph_preserves_explicit_workspace_extensions` | PASS |
| `S15-28` | Dependency arrows and scientific-core forbidden imports | MIGRATED | `F:tests/test_scientific_core_layering.py::SCIENTIFIC_CORE_PACKAGES`; `F:tests/test_scientific_core_layering.py::FORBIDDEN_IMPORTS`; `F:tests/test_scientific_core_layering.py::test_scientific_core_has_no_platform_or_authoring_imports` | PASS: every Python import in the complete scientific-core package set is checked against Studio, controller, provider-adapter, persistence, and authoring boundaries. |
| `S15-29` | Matrix completeness: every discovered item and Sections 15/18 row has one disposition | MIGRATED | This document's `I`, `D`, `S15`, and `S18` manifests plus the checker below. | PASS structurally and substantively; the separate owner action in `S18-14` remains truthfully unratified. |

## Section 18 end-state matrix

| ID | Section 18 acceptance obligation | Disposition | Evidence | Status |
|---|---|---|---|---|
| `S18-01` | One owner and typed identity domain for every authoritative root, IR, intent, reservation, attempt, event, artifact, and checkpoint | MIGRATED | `F:feedbax/compiler/experiment.py::class ExperimentDocument`; `F:feedbax/compiler/experiment.py::class CampaignDocument`; `F:feedbax/compiler/experiment.py::class ResolvedExperiment`; `F:feedbax/orchestration/controller.py::class ControllerEventStore`; `F:feedbax/orchestration/controller.py::class ProviderInventoryObservation`; `F:feedbax/contracts/publication.py::class CheckpointSet`; `F:tests/test_scientific_compiler_roots.py::test_experiment_resolves_to_one_content_pinned_identity` | PASS: the compiler owns the three formerly missing typed roots; the concrete append-only controller event store, including provider inventory and orphan state, is the event-log authority under its explicit event schema rather than a second alias type. |
| `S18-02` | One install/release, optional platform extras, complete checked import rule | RETAINED | Feedbax distribution owner. `F:pyproject.toml::[project.optional-dependencies]`; `F:tests/test_scientific_core_layering.py::test_scientific_core_has_no_platform_or_authoring_imports` | PASS: one distribution retains platform extras and the complete scientific-core package set is checked against all five forbidden boundaries. |
| `S18-03` | All model construction through `GraphDocument -> ResolvedGraph -> ExecutableGraph` with source maps/key schedule | MIGRATED | `F:feedbax/compiler/graph.py::def compile_graph`; `F:feedbax/contracts/graphs/serialization.py::def _instantiate_graph`; `F-NOTOKEN:feedbax::def spec_to_graph` | PASS: the remaining direct instantiator is private and recursively realizes nested graphs inside the compiler. |
| `S18-04` | Semantic and workspace state independently durable; view edits cannot change scientific identity | MIGRATED | `F:tests/test_graph_compiler.py::test_workspace_view_edits_cannot_change_semantic_or_runtime_identity`; `F:tests/test_studio_workspace.py::test_workspace_only_save_preserves_semantic_graph_revision` | PASS |
| `S18-05` | Trial generation, objective computation, and training compose through small resolved protocols | MIGRATED | `F:feedbax/declarations/science.py::class ResolvedObjective`; `F:feedbax/contracts/training.py::class DeclaredTrainingProgram`; `R:tests/adaptive_lambda/test_sisu_release_acceptance.py::test_release_declarations_bind_the_real_trial_source_and_eq15_objective` | PASS: SISU trial, objective, and training authorities all compose through their public resolved declarations. |
| `S18-06` | All bounded multi-step work lowers to one finite `WorkflowPlan`; open-ended control remains event-driven | MIGRATED | `F:tests/test_workflow_plan.py::test_dependency_closure_is_canonical_and_deduplicates_a_diamond`; `F:tests/test_workflow_execution.py::test_a_matrix_receipt_set_executes_analysis_in_row_order_and_rebuilds`; `F:feedbax/orchestration/controller.py::class ControllerEventStore` | PASS |
| `S18-07` | Provider-neutral invocations and separately accountable attempts | MIGRATED | `F:tests/test_invocation_backend_realization.py::test_invocation_is_provider_neutral_versioned_and_identity_stable`; `F:tests/test_invocation_backend_realization.py::test_attempt_is_a_separate_versioned_observation` | PASS |
| `S18-08` | Every external effect reserved before dispatch, ambiguity reconciled, cloud attempts authorized by exact unexpired reservation | MIGRATED | `F:feedbax/orchestration/controller.py::async def dispatch`; `F:feedbax/orchestration/controller.py::def admit_retry`; `F:feedbax/orchestration/controller.py::def observe_provider_inventory`; `F:tests/test_durable_controller.py::test_authenticated_ambiguous_dispatch_recovers_once_after_restart`; `F:tests/test_durable_controller.py::test_complete_inventory_proves_ambiguous_absence_before_same_key_retry` | PASS |
| `S18-09` | Distinct blob, artifact, provenance, publication, and checkpoint protocols with atomic visibility | MIGRATED | `F:feedbax/contracts/publication.py::class BlobStore`; `F:feedbax/contracts/publication.py::class ArtifactRecord`; `F:feedbax/contracts/publication.py::class ProvenanceEdge`; `F:tests/test_publication_protocol.py::test_publication_rolls_back_every_logical_record_on_late_failure`; `R:results/12a9d27/final-release-acceptance.v1.json::blob_materialization_verified`; `A-VERSION:artifact-version:5bad14e33eaaa81f1ab901a6916ceb6a::a419324620e4e9a26b113bcd0be8dd377f6be1d5a5a67f4f9386277a40d5c1c1` | PASS: the generic final publication committed atomically, replayed, and materialized all named bytes. |
| `S18-10` | rlrmp2 supplies science through public declarations without internal schema/migration imports | MIGRATED | `R:src/rlrmp2/adaptive_lambda/method.py::PLUGIN_REGISTRATION`; `R:src/rlrmp2/adaptive_lambda/graph_model.py::register_sisu_graph_components`; `R:specs/declarations/sisu_trial_source.v1.json::feedbax.declaration_document`; `R:specs/declarations/sisu_eq15_objective.v1.json::feedbax.declaration_document`; `R-NOTOKEN:src::feedbax.contracts.migrations` | PASS: components, training programs, trial source, and objective use public declarations; publication uses the generic contract and rlrmp2 imports no internal migration module. |
| `S18-11` | Current paired Tier A acceptance at exact revisions | MIGRATED | `R:results/12a9d27/final-release-acceptance.v1.json::"feedbax": "d11d49fb2ec14340d445e445b90f7fb63d965031"`; `R:results/12a9d27/release-acceptance.verdict.v8.json::"accepted": true`; `A-VERSION:artifact-version:5bad14e33eaaa81f1ab901a6916ceb6a::a419324620e4e9a26b113bcd0be8dd377f6be1d5a5a67f4f9386277a40d5c1c1` | PASS: the evidence pins Feedbax `d11d49f` and rlrmp2 implementation `93e9009`, and is committed on rlrmp2 protected-delivery candidate `0729229`. |
| `S18-12` | Current complete Tier A downstream wall clock within two minutes | MIGRATED | `R:results/12a9d27/final-release-acceptance.v1.json::"wall_seconds": 28.60`; `R:results/12a9d27/final-release-acceptance.v1.json::"report_render_size_bytes": 4889421` | PASS: the fresh downstream chain reached and materialized the final report in 28.60 seconds. |
| `S18-13` | Clean legacy Tier B reference and replacement confirmation on final programme candidate | MIGRATED | `R:results/12a9d27/experiment-spec-lock.v1.json::legacy_authorities`; `R:results/12a9d27/final-release-acceptance.v1.json::"rtol": 0`; `R:results/12a9d27/final-release-acceptance.v1.json::"terminal_checkpoint_slot_bytes_equal": true`; `R:results/12a9d27/release-acceptance.verdict.v8.json::"tier_b": "uninterrupted 1000 and prefix 500 plus non-self-contained continuation 500 completed once"` | PASS: the final full configuration ran once, retained the pinned legacy authorities, and answered the replacement continuation contract exactly. The deterministic trace-order difference changes no coordinate-addressed value. |
| `S18-14` | Owner-ratified revision-pinned matrix accounts for all inventory and obligations | MIGRATED | This document as eventual auth-head content, its explicit implementation pins, and the checker below. | **FAIL pending owner action**: implementation and acceptance evidence are complete, but the exact auth-head bytes have not been approved. The auth request, not this document, records the eventual delivery SHA; owner ratification cannot be self-asserted. |
| `S18-15` | Every `DELETED` item, predecessor path, obsolete schema/migration-only mechanism, duplicate validator, and representation-only test is absent | DELETED | `D01`-`D14` and the absence checker. | PASS for every item classified `DELETED`; retained contracts are separately named in `I16`-`I19` and are not counted as deleted. |

## Machine-checkable completeness and absence method

The four row-ID sets are closed for this revision:

```text
I = I01..I32
D = D01..D14
S15 = S15-01..S15-29
S18 = S18-01..S18-15
```

A checker must:

1. parse Markdown table rows whose first cell matches those four closed sets;
2. require each ID exactly once and exactly one disposition token from
   `MIGRATED|RETAINED|DELETED`;
3. require a non-empty status and evidence cell;
4. resolve every `F:path::token` and `R:path::token` pointer with
   `git grep -F <token> <pin> -- <path>`;
5. reject any existing `F-ABSENT:path` or `R-ABSENT:path` using
   `git cat-file -e <pin>:<path>`;
6. reject any `F-NOTOKEN:scope::token` or `R-NOTOKEN:scope::token` for which
   `git grep -F <token> <pin> -- <scope>` succeeds; and
7. resolve each `A-VERSION:id::sha256` as exactly one Mandible artifact version,
   materialize its custody URI, and require the resulting bytes to have that SHA-256.

The discovery side is independently reproducible: enumerate the two pinned diffs with
`git diff --name-status <base> <pin>`, enumerate changed Python exports from `__all__`,
extract changed or added `*_SCHEMA_ID`, `*_SCHEMA_VERSION`, `*_TYPE`, component type, and
training program identities, then search both pinned trees for producers and consumers.
The census must equal the `I` manifest plus the `D` ledger. Any unmatched discovered
item, duplicate row ID, unresolved pointer, or deletion proof that finds bytes makes the
matrix fail.

## Required parent actions

The implementation and experiment lanes are not reopened by this document. The exact
candidate evidence resolves the former controller, declaration, generic publication,
Tier A, and Tier B failures. The parent also reports the final repository gates green:
Feedbax `6021 passed, 9 xfailed`; rlrmp2 `1096 passed, 11 skipped, 4 deselected,
2 xfailed`. This planner did not rerun either full suite.

The parent coordinator retains only protected-delivery and umbrella-closeout ownership:

1. preserve the exact implementation/evidence pins and acceptance artifact above, merge
   this signed matrix commit into the eventual auth head, and let the auth request record
   that head's exact SHA;
2. obtain owner approval of the exact protected-delivery bytes, which alone ratifies this
   matrix; and
3. after successful convergence, record the ratification and close the umbrella.

Co-Authored-By: Codex (GPT-5) <codex@openai.com>
