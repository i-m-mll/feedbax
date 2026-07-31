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
| `result-role-binding` | `feedbax_external_conformance` | `RESULT_SCHEMA_ID`, `RESULT_SCHEMA_VERSION`, `RESULT_SCHEMA_VERSION_V11`, `REQUIRED_CASE_IDS`, `V10_REQUIRED_CASE_IDS`, `V11_REQUIRED_CASE_IDS`, `ConformanceResult`, `ProtocolRoleSlots`, `load_result` | `feedbax.external_conformance.result.v12` preserves the exact v10 and v11 inventories in order, appends `custody_persistence_recovery`, and requires explicit numeric roles `current = 1` and `minimum = 1`. V11 rejects for missing custody and bound-role evidence; it never migrates synthetically. V1 retains its v2 normalization-before-rejection behavior, and all other earlier decisions remain explicit. | `custody_persistence_recovery` |
| `array-values` | `feedbax.contracts.array_values` and the same names at `feedbax.contracts` | `ARRAY_VALUE_SCHEMA_ID`, `ARRAY_VALUE_SCHEMA_VERSION`, `ArrayValueSpec`, `ConstantArrayValueSpec`, `SparseCooArrayValueSpec`, `SparseCooEntrySpec`, `materialize_array_value` | `feedbax.spec.component_param.array_value.v1` is current. Partial tags and unknown versions reject; canonical sparse COO and constant declarations materialize deterministically. | `component_param_array_values` |
| `dynamic-component-definition` | `feedbax.contracts.component` | `COMPONENT_DEFINITION_SCHEMA_ID`, `COMPONENT_DEFINITION_SCHEMA_VERSION`, `COMPONENT_DEFINITION_SCHEMA_VERSION_V1`, `COMPONENT_DEFINITION_SCHEMA_VERSION_V2`, `ComponentDefinition`, `DynamicPortLayout`, `DynamicPortPolicy`, `DynamicPortPolicyError`, `derive_dynamic_port_count`, `derive_dynamic_port_layout`, `validate_dynamic_port_layout`, `migrate_component_definition_payload`, `migrate_component_definition_v1_to_v2_payload`, `migrate_component_definition_v2_to_v3_payload` | `feedbax.spec.component_definition.v3` is current. v1 migrates to v2 port kinds; v2 migrates to v3 dynamic-port policy. Dynamic layouts derive only from declared policy and parameters and reject mismatches. | `dynamic_component_ports` |
<!-- policy-guarantees:end -->

The versioned fixture-policy manifest is
`external/feedbax_conformance_fixture/src/feedbax_external_conformance/policy_manifest.v1.json`.
Its `plugin-bootstrap.plugin_api` value is the sole structured authority. This
rendering reproduces that value in exact order:

<!-- plugin-api-inventory:start -->
Namespace `feedbax.plugins` public names (ordered): `DOWNSTREAM_INTERFACE_POLICY_ID`, `DOWNSTREAM_PROTOCOL_CURRENT`, `DOWNSTREAM_PROTOCOL_MINIMUM`, `DOWNSTREAM_POLICY_EFFECTIVE_RELEASE`, `UnsupportedDownstreamProtocolVersion`, `validate_downstream_protocol_version`, `BootstrapError`, `BootstrapErrorCode`, `BootstrapState`, `FamilyRequirement`, `PluginDeclaration`, `PluginDependency`, `PluginProvenance`, `PluginRegistration`, `RegistrationContext`, `RegistryKey`, `bootstrap_application`, `discover_plugin_registrations`, `new_registration_context`, `DRIVERS`, `APPLICATION_REGISTRY_KEYS`, `ApplicationRegistryBundle`, `COMPONENTS`, `TRAINING_METHODS`, `ROW_LOWERERS`, `EXECUTION_PREPARATIONS`, `ANALYSIS_RECIPES`, `EVALUATION_RECIPES`, `EVALUATION_BATCH_CONSUMERS`, `EVALUATION_PRODUCT_UNION_FINALIZERS`, `TrainingMethodDescriptor`, `TrainingMethodRegistry`, `TrainingRowLowererRegistration`, `TrainingRowLowererRegistry`, `TrainingRowLoweringContext`, `TrainingRowLoweringResult`, `ExecutionPreparationProvider`, `ExecutionPreparationProviderRegistry`, `ExecutionPreparationRegistration`, `ExecutionPreparationRequest`, `ExecutionPreparationResult`, `ExecutionPreparationPlan`, `compile_training_method_authoring`, `training_method_row_lowerer_registration`, `AnalysisRecipe`, `AnalysisRecipeRegistry`, `AnalysisRecipeResult`, `EvaluationRecipe`, `EvaluationRecipeRegistry`, `EvaluationRecipeResult`, `EvaluationBatchRecipe`, `EvaluationAuthoringSchema`, `EvaluationBatchItem`, `EvaluationBatchRowError`, `EvaluationStatesStructureProviderProtocol`, `StagedExecutionContext`, `EMPTY_STAGED_EXECUTION_CONTEXT`, `EvaluationBatchConsumer`, `EvaluationBatchConsumerRegistry`, `EvaluationBatchConsumerInput`, `EvaluationBatchMergeInput`, `EvaluationBatchMergeState`, `EvaluationBatchFinalizeInput`, `EvaluationBatchFragment`, `EvaluationCompactProductUnionFinalizerRegistry`, `EvaluationCompactProductUnionInput`.

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
IDs. V12 preserves the v10 dynamic-component and v11 external-driver evidence,
then adds the custody/persistence case and binds both numeric protocol roles.
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
that validated v12 result as an artifact. V12 is the single ratification
transition: no policy-only intermediate result or later v13 is created.
