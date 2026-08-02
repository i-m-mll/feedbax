# Downstream interface stability policy

| Field | Value |
|---|---|
| Policy identity | `feedbax.downstream-interface-stability.v1` |
| Status | Ratification-ready; approval and merge of the final protected Feedbax auth ratifies this policy |
| Effective release | Feedbax `0.2.0` |
| Extension protocol | current `1`, minimum supported `1` |
| Evidence head | `22faa8b54a4f2f5109c0e9d956681d23d9a914e0` plus this issue's adoption delta |
| Decision owner | Feedbax owner |

This policy becomes effective only when the single final protected Feedbax auth
for umbrella `f8a5183` is approved and merged. That auth spec must state
explicitly that approval and merge ratifies
`feedbax.downstream-interface-stability.v1`. No separate policy auth exists.

## Compatibility window

The downstream protocol is an integer independent of the Python package
version and the string-valued internal plugin/bootstrap schema identities.

- Protocol 1 is current and minimum-supported in Feedbax 0.2.0.
- After the first increment, Feedbax supports the current and immediately
  preceding protocols: `minimum = max(1, current - 1)`.
- A minimum-supported protocol remains accepted for at least 12 months after
  its successor is released. Removal must also be owner-ratified, deprecated in
  an earlier release, green in the external fixture for the remaining window,
  and covered by a focused rejection test.
- Before 1.0, removal requires a minor release; at or after 1.0, it requires a
  major release. Release notes name the removed version, replacement,
  migration, and first rejecting release.
- Security, corruption, false-authentication, or unsafe-execution defects may
  reject a nominally supported version immediately, before side effects, with
  a focused negative test and an owner-recorded emergency decision.

The initial window has one numeric version because no earlier numeric protocol
was published. Current and minimum remain distinct fixture roles even while
both equal 1.

## Guarantee boundary

Named imports and documented behavior below are stable for supported numeric
protocols. Source text, private helpers, implementation inheritance, and full
Python signatures are not frozen unless they are part of a listed durable
schema. Compatible additions may add optional fields or parameters with
defaults and may improve diagnostics.

Everything not listed is free to change. In particular, this policy does not
guarantee the `feedbax.runtime.*` namespace, provider-specific drivers,
process-global registries, compatibility discovery paths, or source/commit
hashes. Durable Feedbax-owned state is never free to drift silently: preserve
its semantics, migrate it through a versioned deterministic path, or reject it
explicitly.

<!-- policy-guarantees:start -->
| Row ID | Stable namespace | Stable public names | Durable schemas and behavior | External case IDs |
|---|---|---|---|---|
| `plugin-bootstrap` | `feedbax.plugins` | Exact ordered namespace, direct-import, family, method, callback, support, and consumer inventory below. | Numeric protocol admission is explicit at the one unified bootstrap. The versioned policy manifest is the sole structured authority for the concrete plugin API. `compile_training_method_authoring` accepts its explicit public `registry` authority only. `feedbax.plugin.declaration.v2` requires the numeric declaration; `feedbax.plugin.declaration.v1` and unknown schema versions reject. Bootstrap remains isolated, transactional, sealed, and caller-owned. | `unified_plugin_bootstrap` |
| `orchestration-driver` | `feedbax.orchestration.drivers`; `feedbax.plugins`; `feedbax.orchestration.bundle`; `feedbax.orchestration.assembly` | `DRIVER_CAPABILITIES_SCHEMA_ID`, `DRIVER_CAPABILITIES_SCHEMA_VERSION`, `DRIVER_CAPABILITIES_SCHEMA_VERSION_V1`, `DRIVER_CAPABILITIES_SCHEMA_VERSION_V2`, `DRIVER_CAPABILITIES_SCHEMA_VERSION_V3`, `DriverAuthority`, `DriverCapabilityEnvelope`, `DriverCapabilityFacts`, `DriverConstructionContext`, `DriverHook`, `DriverRegistration`, `DriverRegistry`, `DriverStage`, `DriverVenue`, `RealizedDriverCapabilities`; `DRIVERS`, `ApplicationRegistryBundle`; `DEPLOYMENT_POLICY_SCHEMA_ID`, `DEPLOYMENT_POLICY_SCHEMA_VERSION`, `DEPLOYMENT_POLICY_SCHEMA_VERSION_V1`, `RUN_BUNDLE_SCHEMA_ID`, `RUN_BUNDLE_SCHEMA_VERSION`, `RUN_BUNDLE_SCHEMA_VERSION_V11`, `DeploymentPolicy`, `RunBundle`; `RUN_ASSEMBLY_REQUEST_SCHEMA_ID`, `RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION`, `RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION_V5`, `RunAssemblyRequest` | Driver capability envelopes and realized facts accept only schema ID `feedbax.orchestration.driver-capabilities` at version `3`; capability versions `1` and `2` reject rather than being reinterpreted. The unified application bootstrap owns the injected, sealed `DRIVERS` registry and has no process-global or compatibility registry. `feedbax.spec.deployment_policy.v1` migrates to v2; `feedbax.spec.run_assembly_request.v5` migrates to v6 while migrating its nested policy; `feedbax.orchestration.run_bundle.v11` migrates to v12 while migrating its nested policy. Older unsupported request and bundle versions reject. Registry construction validates the realized variant and every declared required callable before publication. | `external_driver_plugin` |
| `orchestration-lifecycle` | `feedbax.orchestration` | `RUN_SET_STATE_SCHEMA_ID`, `RUN_SET_STATE_SCHEMA_VERSION`, `RUN_SET_STATE_SCHEMA_VERSION_V4`, `RunSetState`, `RunSetStateStore`, `StageEngine` | `feedbax.orchestration.run_set_state.v5` is current. Versions v0 through v4 reject rather than silently acquiring the v5 abort and emergency-custody semantics. The production stage engine resumes from durable public state, establishes bounded control reserves before driver actions, and fails closed across primary persistence loss. | `public_lifecycle_recovery`, `custody_persistence_recovery` |
| `ordered-lowering` | `feedbax.lowering` and the same root exports at `feedbax` | `LowererRegistration`, `LoweredContribution`, `LowererExecutionError`, `OrderedLowererRegistry` | Duplicate IDs reject; order is `(order, lowerer_id)`; inactive contributions return `None`; failures identify lowerer and owner; no merge policy is invented. | `ordered_registration` |
| `component-registration` | `feedbax.component_registry` | `ComponentBuilder`, `ComponentMeta`, `ComponentResolution`, `ComponentRegistry`, `ComponentMigration`, `ComponentMigrationPack` | Namespaced component registration, owner-matched migration packs, deterministic resolution, and the stable methods `ComponentRegistry.register_component_type`, `register_migration`, `register_migration_pack`, and `resolve_component_spec`. There is no guaranteed module-level `register_component_type` facade. | `component_registration_and_migration`, `dynamic_component_ports` |
| `structured-migrations` | `feedbax.contracts.migrations` | `SchemaMigration`, `SpecSchemaFamily`, `SpecFamilyMigrationPolicy`, `SpecMigrationResult`, `SpecSchemaRegistry`, `UnknownSpecFamily`, `UnsupportedMigrationPath`, `UnsupportedSpecVersion`, `MissingComponentOwner`, `UnsupportedComponentMigration`, `default_spec_registry`, `migrate_structured_spec_payload`, `migrate_graph_spec` | Versionless input rejects unless the caller deliberately selects `assume_current`; deterministic migration records retain source and target identities; unsupported paths fail closed. | `component_registration_and_migration`, `component_param_array_values` |
| `graph-spec` | `feedbax.contracts.graph` | `GRAPH_SPEC_SCHEMA_ID`, `GRAPH_SPEC_SCHEMA_VERSION`, `GRAPH_SPEC_SCHEMA_VERSION_V2`, `GRAPH_SPEC_SCHEMA_VERSION_V3`, `GRAPH_SPEC_SCHEMA_VERSION_V4`, `LEGACY_GRAPH_SPEC_SCHEMA_VERSION`, `ComponentSpec`, `GraphProject`, `GraphSpec`, `ParamSchema`, `ParamValue`, `StudioValueSpec`, `WireSpec` | `feedbax.spec.graph.v5` is current. The registered chain migrates legacy `1.0.0` to v2, v2 to v3, v3 to v4, and v4 to v5; unknown versions reject. | `component_param_array_values` |
| `value-identity` | `feedbax.contracts.value_identity` and the same names at `feedbax.contracts` | `VALUE_IDENTITY_SCHEMA_ID`, `VALUE_IDENTITY_SCHEMA_VERSION`, `ValueIdentityRecord`, `authored_value_sha256`, `semantic_value_sha256`, `realization_value_sha256`, `value_identity_record` | `feedbax.value_identity.v1` is the only accepted record. Authored identity follows canonical declared encoding, semantic identity follows normalized exact numeric meaning, and realization identity additionally binds layout and backend. Other versions reject. | `value_identity` |
| `material-admission` | `feedbax.contracts.material_dependencies` and the same names at `feedbax.contracts` | `ADMISSION_WAIVER_SCHEMA_ID`, `ADMISSION_WAIVER_SCHEMA_VERSION`, `MATERIAL_DEPENDENCIES_SCHEMA_ID`, `MATERIAL_DEPENDENCIES_SCHEMA_VERSION`, `AdmissionWaiver`, `IncidentalAdmissionFailure`, `MaterialDependency`, `MaterialDependencyAdmission`, `MaterialDependencyObservation`, `MaterialDependencySet`, `MaterialDependencyValue`, `dependency_value_sha256`, `material_dependency_identity_sha256`, `validate_material_dependency_admission` | Material dependencies use `feedbax.spec.material_dependencies.v1`; the exact narrow waiver uses `feedbax.spec.admission_waiver.v1`. Versionless and other versions reject. Material identity excludes incidental provenance and declaration order; a waiver binds exactly one declared dependency and never bypasses material authenticity. | `material_dependencies` |
| `terminal-certification` | `feedbax.contracts.manifest` and the same names at `feedbax.contracts` | `TRAINING_RUN_CERTIFICATION_SCHEMA_ID`, `TRAINING_RUN_CERTIFICATION_SCHEMA_VERSION`, `TRAINING_RUN_CERTIFICATION_MIGRATION_TABLE`, `TrainingRunCertification`, `training_run_certification` | `feedbax.manifest.training_run_certification.v1` is current; v0 is rejected. Legacy completed/cancelled manifests project deterministically, legacy failed and nonterminal manifests reject because their terminal meaning is ambiguous. Focused in-repo migration tests cover this schema, but no external fixture case does, so this row does not claim a T cell. | No external case |
| `custody-persistence` | `feedbax.orchestration` | `RUN_SET_STATE_SCHEMA_ID`, `RUN_SET_STATE_SCHEMA_VERSION`, `RUN_SET_STATE_SCHEMA_VERSION_V4`, `PrimaryStatePersistenceError`, `CustodyPreservationRequired`, `RunSetState`, `RunSetStateStore`, `StageEngine` | `feedbax.orchestration.run_set_state.v5` is current and versions v0 through v4 reject. Primary persistence failure retains its typed `OSError` cause; destructive ephemeral teardown is blocked by the typed custody exception until collection is durably complete, then occurs exactly once. | `custody_persistence_recovery` |
| `emergency-persistence` | `feedbax.orchestration` | `EMERGENCY_RUN_SET_RECORD_SCHEMA_ID`, `EMERGENCY_RUN_SET_RECORD_SCHEMA_VERSION`, `ControlFilesystemPreflight`, `ControlFilesystemPreflightError`, `EmergencyProviderIdentity`, `EmergencyRunSetRecord`, `RunSetStateStore` | `feedbax.orchestration.emergency_run_set_record.v1` is current; v0, unknown, and malformed records reject. The bounded reserved channel publishes and reads back provider identity, preservation state, custody completion, spend boundary, primary failure, and next recovery action before destructive teardown. | `custody_persistence_recovery` |
| `result-role-binding` | `feedbax_external_conformance` | `RESULT_SCHEMA_ID`, `RESULT_SCHEMA_VERSION`, `RESULT_SCHEMA_VERSION_V11`, `RESULT_SCHEMA_VERSION_V12`, `RESULT_SCHEMA_VERSION_V13`, `REQUIRED_CASE_IDS`, `V10_REQUIRED_CASE_IDS`, `V11_REQUIRED_CASE_IDS`, `V12_REQUIRED_CASE_IDS`, `V13_REQUIRED_CASE_IDS`, `ConformanceResult`, `ProtocolRoleSlots`, `load_result` | `feedbax.external_conformance.result.v14` preserves the exact thirteen-case v13 inventory in order, appends `figure_role_reference_public_contract`, and requires explicit numeric roles `current = 1` and `minimum = 1`. V13 rejects for missing figure-role-reference evidence and V12 for missing figure-composition evidence; neither migrates synthetically. V1 retains its v2 normalization-before-rejection behavior, and all other earlier decisions remain explicit. | `custody_persistence_recovery` |
| `array-values` | `feedbax.contracts.array_values` and the same names at `feedbax.contracts` | `ARRAY_VALUE_SCHEMA_ID`, `ARRAY_VALUE_SCHEMA_VERSION`, `ArrayValueSpec`, `ConstantArrayValueSpec`, `SparseCooArrayValueSpec`, `SparseCooEntrySpec`, `materialize_array_value` | `feedbax.spec.component_param.array_value.v1` is current. Partial tags and unknown versions reject; canonical sparse COO and constant declarations materialize deterministically. | `component_param_array_values` |
| `dynamic-component-definition` | `feedbax.contracts.component` | `COMPONENT_DEFINITION_SCHEMA_ID`, `COMPONENT_DEFINITION_SCHEMA_VERSION`, `COMPONENT_DEFINITION_SCHEMA_VERSION_V1`, `COMPONENT_DEFINITION_SCHEMA_VERSION_V2`, `ComponentDefinition`, `DynamicPortLayout`, `DynamicPortPolicy`, `DynamicPortPolicyError`, `derive_dynamic_port_count`, `derive_dynamic_port_layout`, `validate_dynamic_port_layout`, `migrate_component_definition_payload`, `migrate_component_definition_v1_to_v2_payload`, `migrate_component_definition_v2_to_v3_payload` | `feedbax.spec.component_definition.v3` is current. v1 migrates to v2 port kinds; v2 migrates to v3 dynamic-port policy. Dynamic layouts derive only from declared policy and parameters and reject mismatches. | `dynamic_component_ports` |
| `figure-composition` | `feedbax.contracts.figures`; `feedbax.analysis.figures`; `feedbax.analysis.bundles`; `feedbax.contracts.matrix_core`; `feedbax.contracts.run_matrix` | Exact namespace and CLI inventory below. | Composition v2 resolves through one public resolver to ordinary `feedbax.spec.figure.v2` semantics while retaining exact authored identity and ordered full-chain provenance; v1 migrates to v2. Figure-only structural additions use `FigureCompositionDelta` and `SameSchemaStructuralAddition`; every added typed path must be declared exactly, with `feedbax.spec.figure_panel.v1` serving only as the declaration-side `PanelSpec` identity. Bundle execution owns a trusted repository root, Studio rejects unsupported composition with a typed error, runtime binding v2 separates authored and resolved identities, inheritance uses canonical absent-only list-index grafting, and shared `MatrixCompositionDelta` overrides retain prefix-aware acknowledgement with qualified layer attribution. | `figure_composition_public_contract` |
| `figure-role-references` | `feedbax.contracts.row_index`; `feedbax.contracts.figure_roles`; `feedbax.contracts.experiment_envelope` | `ROW_INDEX_SCHEMA_ID`, `ROW_INDEX_SCHEMA_VERSION`, `ROW_INDEX_CUSTODY_SCHEMA_ID`, `ROW_INDEX_CUSTODY_SCHEMA_VERSION`, `RESOLVED_ROW_SET_SCHEMA_ID`, `RESOLVED_ROW_SET_SCHEMA_VERSION`, `AuthenticatedRowIndex`, `RowIndexEntry`, `RowIndexCustodyBindings`, `RowCustodyBinding`, `ResolvedRowSet`, `RowSetSelector`, `AllRowsSelector`, `TagRowsSelector`, `RowSelectionError`, `RowSelectionErrorCode`, `expand_row_selector`, `normalize_row_tags`, `derive_row_label`; `FIGURE_ROW_EXPANSION_REQUEST_SCHEMA_ID`, `FIGURE_ROW_EXPANSION_REQUEST_SCHEMA_VERSION`, `RESOLVED_FIGURE_INPUTS_SCHEMA_ID`, `RESOLVED_FIGURE_INPUTS_SCHEMA_VERSION`, `FigureRowExpansionRequest`, `FigureRoleBindingContract`, `FigureRoleReferenceError`, `PerRowInputReference`, `SharedInputReference`, `ResolvedFigureInput`, `ResolvedFigureInputs`, `resolve_figure_input_roles`, `expand_figure_rows`, `row_namespace`; `EXPERIMENT_ENVELOPE_COMPILE_RESULT_SCHEMA_ID`, `EXPERIMENT_ENVELOPE_COMPILE_RESULT_SCHEMA_VERSION`, `ExperimentEnvelopeCompileRequest`, `ExperimentEnvelopeCompileResult`, `ExperimentEnvelopeCompilerRegistration`, `ExperimentEnvelopeCompilerRegistry`, `ExperimentEnvelopeRejection`, `ExperimentEnvelopeRejectionCategory`, `dispatch_experiment_envelope` | The row-set selector is the closed tagged union `{"mode":"all"}` and `{"mode":"tag","tag":...}`; conjunction, negation, regular expressions, ordering, comparison, and nesting are absent and cannot be added without a new schema version. A selector expands exactly once into an explicit ordered row-id list pinned to the digest of the index it was expanded against; empty selections, duplicate row ids, and ambiguous custody bindings fail closed with stable codes. The index splits at the custody boundary: identity, deterministic order, and bounded normalized tags are compile-time, while authenticated artifact custody attaches post-run, so a first-time figure resolves its roles with no production record. Figure inputs are role references, per-row or shared; an authored authority block is rejected with the offending field named, and resolved manifest digests stay in the compile lock rather than the figure's scientific identity. Row-index order alone derives the `row_{n}__` namespace, panel placement and titles, legend ownership, colorbar placement, and assembler height, producing ordinary `feedbax.spec.figure.v2` semantics. `python -m feedbax preflight-experiment-envelope` routes an envelope to the one registered compiler claiming its declared schema through the injected registry, with exit 0 accepted, 2 rejected from a closed category set, and 1 infrastructure failure. | `figure_role_reference_public_contract` |
| `report-surface` | `feedbax.contracts.manifest`; `feedbax.analysis.reports` | `REPORT_SPEC_SCHEMA_ID`, `REPORT_SPEC_SCHEMA_VERSION`, `REPORT_MANIFEST_SCHEMA_ID`, `REPORT_MANIFEST_SCHEMA_VERSION`, `ReportSpec`, `ReportManifest`, `report_manifest_id`, `spec_identity_preimage`; `ORDERED_FIGURE_REPORT_TYPE`, `ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_ID`, `ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_VERSION`, `OrderedFigureReportParams`, `OrderedFigureReportSection`, `OrderedFigureReportFigure`, `ReportRecipeRegistry`, `ReportRecipeResult`, `ReportRecipeExecutionError`, `coerce_report_spec`, `execute_report_spec`, `execute_authored_report_spec`; CLI `feedbax-analysis report` | `feedbax.spec.report.v1` is the current `ReportSpec` version and the document carries its own `schema_id`/`schema_version`. An unversioned historical document is admitted as that one named v1 baseline; `feedbax.spec.report.v0` and unknown versions reject with family, schema ID, source version, current version, and `migration_intentionally_absent`. `feedbax.manifest.v1` is the current `ReportManifest` version and unknown manifest versions reject at load instead of validating. `report_manifest_id` mints identity from the semantic preimage with the schema-identity fields excluded, so a semantics-preserving schema migration never forks the identity of already-produced reports. Ordered-figure params accept only `feedbax.spec.report.ordered_figure.v3`; every included figure requires an authored caption plus a lowercase `figure_spec_sha256`, and figure and scalar-projection input roles stay disjoint. `execute_authored_report_spec` requires authoritative `StagedExactParents`, replaces authored inputs with the authenticated exact refs, refuses a parent carrying `material_dependencies`, and refuses any unauthenticated exact parent. `REPORT_RECIPES` is deliberately not guaranteed. | No external case |
| `evaluation-surface` | `feedbax.contracts.manifest`; `feedbax.analysis.evaluation` | `EVALUATION_RUN_SPEC_SCHEMA_ID`, `EVALUATION_RUN_SPEC_SCHEMA_VERSION`, `EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID`, `EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION`, `EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION_V1`, `EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION_V2`, `EvaluationRunSpec`, `EvaluationRunManifest`, `evaluation_run_manifest_id`; `EvaluationRunMatrixSpec`, `EvaluationRecipeRegistry`, `EvaluationRecipeResult`, `EvaluationBatchExecution`, `coerce_evaluation_run_spec`, `resolve_evaluation_matrix_authoring`, `compile_evaluation_run_matrix`, `materialize_evaluation_run_matrix`, `execute_evaluation_run_matrix`, `execute_evaluation_run_spec`; CLI `feedbax-analysis evaluate` | `feedbax.spec.evaluation_run.v1` is the current `EvaluationRunSpec` version and the document carries its own `schema_id`/`schema_version`. An unversioned historical document is admitted as that one named v1 baseline; `feedbax.spec.evaluation_run.v0`, a foreign `schema_id`, and unknown versions reject. The matrix family coexists across versions: `feedbax.spec.evaluation_run_matrix.v1` migrates to v2 by adding the empty staged-parent map and v2 migrates to v3 combined authoring, while v0 and unknown versions reject. `evaluation_run_manifest_id` excludes the schema-identity fields from its preimage, and switches to the dependency-scoped preimage only when every input carries an exact `material_dependency_identity_sha256` with its retained `material_dependencies` declaration. `feedbax-analysis evaluate` executes a matrix, a matrix delta, or a flat `EvaluationRunSpec` through the same compile, materialize, and execute path; a flat spec requires a stated non-empty escape-hatch reason, and staged parents resolve only against an explicit parent manifest root. | No external case |
| `analysis-authoring` | `feedbax.contracts.manifest`; `feedbax.analysis.specs`; `feedbax.analysis.bundles` | `ANALYSIS_RUN_SPEC_SCHEMA_ID`, `ANALYSIS_RUN_SPEC_SCHEMA_VERSION`, `ANALYSIS_RUN_SPEC_SCHEMA_VERSION_V1`, `AnalysisRunSpec`, `AnalysisRunManifest`, `analysis_run_manifest_id`; `execute_analysis_run_spec`; `ANALYSIS_BUNDLE_SCHEMA_ID`, `ANALYSIS_BUNDLE_SCHEMA_VERSION`, `AnalysisBundleSpec`, `AnalysisBundleDeltaSpec`, `authored_analysis_bundle_from_payload`, `resolve_analysis_bundle_authoring`, `execute_analysis_bundle`, `execute_staged_analysis_bundle`, `dry_run_staged_analysis_bundle`; CLI `feedbax-analysis run`, `feedbax-analysis bundle` | `feedbax.spec.analysis_run.v2` is current and `feedbax.spec.analysis_run.v1` migrates by making the historical implicit `recompute` evaluation-states policy explicit; v0 and unknown versions reject. `feedbax.spec.analysis_bundle.v6` is current and the registered chain migrates v2 through v5; older and unknown versions reject. `feedbax-analysis run` executes one serialized `AnalysisRunSpec` or `AnalysisRunDeltaSpec` with explicit run-alias catalogs and staged execution bindings, and `feedbax-analysis bundle` executes one file-authored bundle without a registered experiment package. A bundle must declare exactly one non-empty execution shape, `--exact-parents` documents must declare `schema_id` and `schema_version` explicitly, and staged bindings require an explicit execution descriptor. | No external case |
<!-- policy-guarantees:end -->

## Pending owner ratification: envelope-layer prerequisite rows

The rows `report-surface`, `evaluation-surface`, and `analysis-authoring` are a
**draft addition** raised by Feedbax issue `88d021d`. They describe surfaces the
downstream envelope authoring layers consume, and they become ratified only when
the owner approves and merges the protected-branch delivery that carries them.
Until that merge they are a proposal recorded in the policy document, not an
accepted stability promise.

Three deliberate limits apply to the draft:

- The three rows carry `coverage_status: not-external-covered` and name no
  external conformance case. They are covered by focused in-repo tests
  (`tests/test_envelope_layer_contracts.py`, `tests/test_structured_spec_migrations.py`,
  `tests/test_report_execution.py`, `tests/test_ordered_figure_report.py`,
  `tests/test_evaluation_matrix.py`, `tests/test_analysis_bundle_cli.py`) rather
  than by a fabricated external case ID.
- `REPORT_RECIPES` is deliberately **not** ratified and stays in
  `feedbax.plugins._NON_GUARANTEED_PLUGIN_EXPORTS`. No plugin-facade export moves
  out of that inventory for these rows, so the ratified `plugin-bootstrap`
  inventory is unchanged.
- The generic row-index custody writer
  (`feedbax.contracts.row_index.build_row_index_custody_bindings`,
  `serialize_row_index_custody_bindings`, `write_row_index_custody_bindings`,
  `load_row_index_custody_bindings`) and the checkpoint initialize/continue
  lowering surface (`feedbax.contracts.checkpoint_initialization`,
  `feedbax.spec.checkpoint_structure.v1`,
  `feedbax.spec.checkpoint_initialization.v1`,
  `feedbax.spec.checkpoint_initialization_plan.v1`) landed in the same issue but
  are intentionally **not** drafted here. They have no downstream consumer yet,
  and the checkpoint lowering is the intended eventual replacement for
  `feedbax.spec.training_run_composition.v1`, whose retirement is its own
  owner-gated decision.

The figure-composition row has this exact manifest-driven public inventory:

<!-- figure-api-inventory:start -->
Namespace `feedbax.contracts.figures`: `FigureCompositionDelta`, `SameSchemaStructuralAddition`, `FigureCompositionSpec`, `FigureCompositionProvenance`, `FigureCompositionSourceRecord`, `ResolvedFigureSpec`, `FigureRuntimeBindingSpec`, `FigureSpec`
Namespace `feedbax.analysis.figures`: `resolve_figure_spec`, `coerce_figure_spec`
Namespace `feedbax.analysis.bundles`: `BundleStageSpec`, `AnalysisBundleSpec`, `execute_staged_analysis_bundle`
Namespace `feedbax.contracts.matrix_core`: `ContentPinnedJsonBase`, `SourceDocumentInheritance`, `materialize_inherited_document`, `load_content_pinned_json_base`
Namespace `feedbax.contracts.run_matrix`: `MatrixCompositionDelta`, `apply_composition_deltas`
CLI: `feedbax-figure resolve`, `feedbax-figure resolve --with-lineage`
<!-- figure-api-inventory:end -->

The ratified semantics are deliberately narrow:

- resolved ordinary FigureSpec v2 identity remains separate from authored composition and provenance identity
- provenance retains ordered full-chain source custody and qualified unique layer attribution
- BundleStageSpec.figure admits composition and staged bundle execution resolves only beneath its trusted repo_root
- FigureRuntimeBindingSpec v2 separates authored_figure_source_sha256 from resolved_figure_spec_sha256
- Studio rejects composition without a server-owned root using figure_composition_not_supported_in_studio
- ContentPinnedJsonBase payload_path and SourceDocumentInheritance share canonical list-index rules and graft only absent targets
- MatrixCompositionDelta acknowledgement must overlap both the ancestor-written path and the current patch path
- FigureCompositionSpec v2 owns FigureCompositionDelta structural declarations and migrates v1 without changing resolved semantics
- SameSchemaStructuralAddition declarations must exactly cover every typed path introduced by an add subtree; feedbax.spec.figure_panel.v1 is declaration-side identity only and never changes resolved PanelSpec bytes

The versioned fixture-policy manifest is
`external/feedbax_conformance_fixture/src/feedbax_external_conformance/policy_manifest.v1.json`.
Its `plugin-bootstrap.plugin_api` value is the sole structured authority. This
rendering reproduces that value in exact order:

<!-- plugin-api-inventory:start -->
Namespace `feedbax.plugins` public names (ordered): `DOWNSTREAM_INTERFACE_POLICY_ID`, `DOWNSTREAM_PROTOCOL_CURRENT`, `DOWNSTREAM_PROTOCOL_MINIMUM`, `DOWNSTREAM_POLICY_EFFECTIVE_RELEASE`, `UnsupportedDownstreamProtocolVersion`, `validate_downstream_protocol_version`, `BootstrapError`, `BootstrapErrorCode`, `BootstrapState`, `FamilyRequirement`, `PluginDeclaration`, `PluginDependency`, `PluginProvenance`, `PluginRegistration`, `RegistrationContext`, `RegistryKey`, `RegistryFamilyRegistration`, `bootstrap_application`, `discover_plugin_registrations`, `new_registration_context`, `DRIVERS`, `APPLICATION_REGISTRY_KEYS`, `ApplicationRegistryBundle`, `COMPONENTS`, `TRAINING_METHODS`, `ROW_LOWERERS`, `EXECUTION_PREPARATIONS`, `ANALYSIS_RECIPES`, `EVALUATION_RECIPES`, `EVALUATION_BATCH_CONSUMERS`, `EVALUATION_PRODUCT_UNION_FINALIZERS`, `TrainingMethodDescriptor`, `TrainingMethodRegistry`, `TrainingRowLowererRegistration`, `TrainingRowLowererRegistry`, `TrainingRowLoweringContext`, `TrainingRowLoweringResult`, `ExecutionPreparationProvider`, `ExecutionPreparationProviderRegistry`, `ExecutionPreparationRegistration`, `ExecutionPreparationRequest`, `ExecutionPreparationResult`, `ExecutionPreparationPlan`, `compile_training_method_authoring`, `training_method_row_lowerer_registration`, `AnalysisRecipe`, `AnalysisRecipeRegistry`, `AnalysisRecipeResult`, `EvaluationRecipe`, `EvaluationRecipeRegistry`, `EvaluationRecipeResult`, `EvaluationBatchRecipe`, `EvaluationAuthoringSchema`, `EvaluationBatchItem`, `EvaluationBatchRowError`, `EvaluationStatesStructureProviderProtocol`, `StagedExecutionContext`, `EMPTY_STAGED_EXECUTION_CONTEXT`, `EvaluationBatchConsumer`, `EvaluationBatchConsumerRegistry`, `EvaluationBatchConsumerInput`, `EvaluationBatchMergeInput`, `EvaluationBatchMergeState`, `EvaluationBatchFinalizeInput`, `EvaluationBatchFragment`, `EvaluationCompactProductUnionFinalizerRegistry`, `EvaluationCompactProductUnionInput`.

Direct RLRMP entrypoint imports (ordered): `EVALUATION_RECIPES`, `FamilyRequirement`, `PluginDeclaration`, `PluginDependency`, `PluginRegistration`, `RegistrationContext`, `ANALYSIS_RECIPES`, `EVALUATION_BATCH_CONSUMERS`, `EVALUATION_PRODUCT_UNION_FINALIZERS`, `EXECUTION_PREPARATIONS`, `ROW_LOWERERS`, `TRAINING_METHODS`.

| Family key | Registry type | Registry methods (ordered) | Callback types (ordered) | Support types (ordered) | Public consumers (ordered) |
|---|---|---|---|---|---|
| `training_methods` | `TrainingMethodRegistry` | `register_descriptor`, `resolve`, `validate_payload`, `resolve_execution` | `TrainingMethodDescriptor` | none | `feedbax.training.authoring:compile_training_method_authoring` |
| `row_lowerers` | `TrainingRowLowererRegistry` | `register`, `lower` | `TrainingRowLowererRegistration` | `TrainingRowLoweringContext`, `TrainingRowLoweringResult` | `feedbax.training.authoring:training_method_row_lowerer_registration` |
| `execution_preparations` | `ExecutionPreparationProviderRegistry` | `register`, `get`, `prepare` | `ExecutionPreparationProvider` | `ExecutionPreparationRegistration`, `ExecutionPreparationRequest`, `ExecutionPreparationResult`, `ExecutionPreparationPlan` | none |
| `analysis_recipes` | `AnalysisRecipeRegistry` | `register`, `get`, `structure_provider` | `AnalysisRecipe` | `AnalysisRecipeResult`, `EvaluationStatesStructureProviderProtocol` | `feedbax.analysis:resolve_analysis_inputs`, `feedbax.analysis:execute_analysis_run_spec` |
| `evaluation_recipes` | `EvaluationRecipeRegistry` | `register`, `register_authoring_schema`, `get`, `batch` | `EvaluationRecipe`, `EvaluationBatchRecipe` | `EvaluationRecipeResult`, `EvaluationAuthoringSchema`, `EvaluationBatchItem`, `EvaluationBatchRowError`, `StagedExecutionContext`, `EMPTY_STAGED_EXECUTION_CONTEXT` | `feedbax.analysis.evaluation:compile_evaluation_run_matrix`, `feedbax.analysis.evaluation:execute_evaluation_run_matrix`, `feedbax.analysis.evaluation:execute_evaluation_run_spec` |
| `evaluation_batch_consumers` | `EvaluationBatchConsumerRegistry` | `register`, `get` | `EvaluationBatchConsumer` | `EvaluationBatchConsumerInput`, `EvaluationBatchMergeInput`, `EvaluationBatchMergeState`, `EvaluationBatchFinalizeInput`, `EvaluationBatchFragment` | `feedbax.analysis.evaluation_compaction:compact_evaluation_batch`, `feedbax.analysis.evaluation_compaction:merge_evaluation_batch_fragment`, `feedbax.analysis.evaluation_compaction:reclaim_evaluation_batch_caches`, `feedbax.analysis.evaluation_compaction:publish_evaluation_compaction_products` |
| `evaluation_product_union_finalizers` | `EvaluationCompactProductUnionFinalizerRegistry` | `register`, `get` | none | `EvaluationCompactProductUnionInput` | `feedbax.analysis.evaluation_product_union:finalize_evaluation_compact_product_union` |
<!-- plugin-api-inventory:end -->

It is the machine-readable mapping of these row IDs to schemas and real case
IDs. V13 preserves the exact twelve-case v12 inventory and order, then appends
the figure-composition case. The v12 inventory already preserves the v10
dynamic-component and v11 external-driver evidence, adds the custody/persistence
case, and binds both numeric protocol roles.
The terminal-certification row remains explicitly non-external-covered rather
than claiming a false stability/conformance cell for its focused in-repo tests.

## Pins and migration rules

Allowed semantic pins identify deliberately versioned meaning: schema IDs and
versions, numeric protocol versions, canonical authored/semantic identities,
artifact content hashes, exact-parent identities, and golden hashes for a
documented canonicalization rule. Source hashes, whole-module hashes,
incidental JSON formatting, hidden string keys in `manifest.metadata`,
unlisted runtime imports, and literals encoding Feedbax commit IDs are not
guarantees.

Supported numeric protocols coexist or migrate only at the unified bootstrap
boundary. Feedbax never infers a protocol, catches a rejection and retries a
legacy path, discovers a second compatibility registry, or publishes partial
process-global state. `feedbax.plugin.declaration.v1` used only the string
`feedbax.plugin.v1`; because that string did not declare the numeric downstream
window, v1 declarations reject after adoption rather than being silently
reinterpreted. V2 requires `downstream_protocol_version` explicitly.

## Console entry points: the unified `feedbax` command

Feedbax installs one unified console entry point, `feedbax`, which routes every
`python -m feedbax` engine subcommand plus the six absorbed subcommands `run`,
`analysis`, `figure`, `train`, `provider`, and `orchestrate`. The unified script
is the entry point new work should target.

The per-command scripts `feedbax-run`, `feedbax-analysis`, `feedbax-figure`,
`feedbax-train`, `feedbax-provider`, and `feedbax-orchestrate` remain installed
and behaviorally unchanged. Two of them are enumerated as stable behavior in the
guarantee table above — `feedbax-analysis` (`report-surface`,
`evaluation-surface`, `analysis-authoring`) and `feedbax-figure`
(`figure-composition`) — so they are guaranteed for every supported numeric
protocol and cannot be removed by an implementation change.

Deprecation path, recorded now so no later change has to reconstruct it:

- Successor: `feedbax <command>` replaces `feedbax-<command>` one-for-one, with
  identical arguments and identical exit codes, because the unified script
  delegates to the same main.
- The per-command scripts are superseded, not deprecated-in-effect: they are
  still supported and still tested. Nothing downstream must migrate today.
- Removing any of them follows this policy's change procedure in full — owner
  ratification, an earlier release that declares the deprecation, the stated
  compatibility duration, a green external fixture across the window, the
  required release class, release notes naming the replacement, and a focused
  rejection test. For the two enumerated scripts this additionally requires
  editing their guarantee rows.
- Until that procedure completes, both spellings are correct and neither is a
  transitional shim.

## Change procedure

A breaking proposal opens an issue naming affected protocols, imports,
schemas, fixtures, and migrate/reject behavior; introduces the successor
without prematurely lowering the minimum; updates policy constants, release
floor, fixture manifest, CI artifact, and release notes together; and removes
an old version only after every duration, deprecation, fixture, release, and
owner gate is satisfied.

The existing clean-wheel runner remains the only runner. CI invokes
`scripts/run_external_conformance.py`, persists its validated machine-readable
result, validates installed dependency metadata with `uv pip check`, and uploads
that validated v13 result as an artifact. V13 is the single post-composition
ratification transition; v12 remains explicit historical evidence and rejects
rather than acquiring the new case synthetically.
