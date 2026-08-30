# Downstream interface stability policy

| Field | Value |
|---|---|
| Policy identity | `feedbax.downstream-interface-stability.v1` |
| Status | Owner-ratified |
| Effective release | Feedbax `0.2.0` |
| Extension protocol | current `1`, minimum supported `1` |
| Ratification evidence | Base policy: protected `develop` merge `b6697280324b3a675cf1de5fbca25b42a0f56795`; envelope-layer prerequisite rows: protected `develop` merge `798c085268119074f0522e3a2313a1722dfaedc8` |
| Policy source head | Protected `develop` merge `bc254ce60f8ce26640794788f8df9a236423052f` |
| Result schema identity | `feedbax.external_conformance.result.v14` |
| Runtime result evidence | No concrete conformance result artifact or execution receipt is pinned in this policy |
| Decision owner | Feedbax owner |

The Feedbax owner ratified this policy through protected `develop` delivery.
Protected merge `b6697280324b3a675cf1de5fbca25b42a0f56795` ratified the base
policy, and protected merge `798c085268119074f0522e3a2313a1722dfaedc8`
ratified the envelope-layer prerequisite rows below. Later protected amendments
through the policy source head preserve that ratified status. The result schema
identity names the required shape of a conformance result; it is not evidence
that the fixture ran. CI is configured to create and upload a result, but this
policy does not currently pin a concrete uploaded artifact or execution receipt.

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

Schema identity is guaranteed as a value, not as a symbol name. What a consumer
actually binds is the string a schema-identity constant holds — `feedbax.spec.graph.v5`,
`feedbax.orchestration.run_set_state.v5` — because that string is what is written
into saved documents and what admission, migration, and rejection are decided on.
Those strings, and the accept/migrate/reject behavior attached to them, are stated
in the "Durable schemas and behavior" column and are fully guaranteed. The names of
the Python constants that carry them (`*_SCHEMA_ID`, `*_SCHEMA_VERSION`,
`*_SCHEMA_VERSION_V<n>`) are not guaranteed. Those constants remain exported, but
renaming or reorganizing them is an ordinary internal change and does not go through
this policy's change procedure. Removing or changing the string values they hold does.

One exception is enumerated because a downstream consumer imports it by name:
`ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_ID` and
`ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_VERSION` are imported from `feedbax.analysis`
and stay listed in the `report-surface` row. The exception retires when that import
does.

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
| `orchestration-driver` | `feedbax.orchestration.drivers`; `feedbax.plugins`; `feedbax.orchestration.bundle`; `feedbax.orchestration.assembly` | `DriverAuthority`, `DriverCapabilityEnvelope`, `DriverCapabilityFacts`, `DriverConstructionContext`, `DriverHook`, `DriverRegistration`, `DriverRegistry`, `DriverStage`, `DriverVenue`, `RealizedDriverCapabilities`; `DRIVERS`, `ApplicationRegistryBundle`; `DeploymentPolicy`, `RunBundle`; `RunAssemblyRequest` | Driver capability envelopes and realized facts accept only schema ID `feedbax.orchestration.driver-capabilities` at version `3`; capability versions `1` and `2` reject rather than being reinterpreted. The unified application bootstrap owns the injected, sealed `DRIVERS` registry and has no process-global or compatibility registry. `feedbax.spec.deployment_policy.v1` migrates to v2; `feedbax.orchestration.run_bundle.v11` migrates to v12 while migrating its nested policy. `feedbax.spec.run_assembly_request.v7` requires an authored `feedbax_revision` authority that is verified against imported-package provenance before assembly compiles or writes anything; no earlier request version migrates, because a migrator cannot invent the revision a request was authored against, so v1-v6 reject with a re-authoring instruction. Older unsupported request and bundle versions reject. Registry construction validates the realized variant and every declared required callable before publication. | `external_driver_plugin` |
| `orchestration-lifecycle` | `feedbax.orchestration` | `RunIntent`, `EffectReservation`, `ControllerEvent`, `ControllerProjection`, `ControllerEventStore`, `ProviderInventoryObservation`, `OrphanHandlingPolicy`, `DurableController`, `RunSetState`, `RunSetStateStore`, `StageEngine` | Controller intent and reservations retain their explicit v1 schemas; events and projections use v2, with event v1 migrated and projection v1 explicitly rejected because projections rebuild from events. Provider inventory and orphan policy use explicit v1 schemas. The append-only controller event stream is authoritative: effects are durably reserved before adapter calls, paid acquisition requires authentication of one exact unexpired reservation, retry admission enforces the exact Invocation policy and preserves effect identity, ambiguous dispatch remains unknown until reconciliation or complete provider inventory proves absence, unmatched inventory resources receive a durable require-operator decision, and Studio status is reconstructed rather than inferred from process state. `feedbax.orchestration.run_set_state.v5` remains the finite StageEngine execution projection; versions v0 through v4 reject. | `public_lifecycle_recovery`, `custody_persistence_recovery` |
| `ordered-lowering` | `feedbax.lowering` and the same root exports at `feedbax` | `LowererRegistration`, `LoweredContribution`, `LowererExecutionError`, `OrderedLowererRegistry` | Duplicate IDs reject; order is `(order, lowerer_id)`; inactive contributions return `None`; failures identify lowerer and owner; no merge policy is invented. | `ordered_registration` |
| `component-registration` | `feedbax.component_registry` | `ComponentBuilder`, `DeclaredComponent`, `declare_component`, `ComponentResolution`, `ComponentRegistry`, `ComponentMigration`, `ComponentMigrationPack` | Neutral component declarations with layer-local facets, namespaced registration, owner-matched migration packs, deterministic resolution, and the stable methods `ComponentRegistry.register_component_type`, `register_migration`, `register_migration_pack`, and `resolve_component_spec`. | `component_registration_and_migration`, `dynamic_component_ports` |
| `structured-migrations` | `feedbax.contracts.migrations` | `SchemaMigration`, `SpecSchemaFamily`, `SpecFamilyMigrationPolicy`, `SpecMigrationResult`, `SpecSchemaRegistry`, `UnknownSpecFamily`, `UnsupportedMigrationPath`, `UnsupportedSpecVersion`, `MissingComponentOwner`, `UnsupportedComponentMigration`, `default_spec_registry`, `migrate_structured_spec_payload`, `migrate_graph_spec` | Versionless input rejects unless the caller deliberately selects `assume_current`; deterministic migration records retain source and target identities; unsupported paths fail closed. | `component_registration_and_migration`, `component_param_array_values` |
| `graph-spec` | `feedbax.contracts.graph` | `ComponentSpec`, `GraphProject`, `GraphSpec`, `ParamSchema`, `ParamValue`, `SemanticAnchor`, `StudioValueSpec`, `WireSpec`, `WorkspaceDocument` | `feedbax.spec.graph.v5` is current. The registered chain migrates legacy `1.0.0` to v2, v2 to v3, v3 to v4, and v4 to v5; unknown versions reject. `feedbax.workspace_document` version `1` owns graph and analysis presentation plus workspace-, stage-, and scenario-level view state separately from semantic graph identity. `feedbax.spec.studio.workspace.v1`, `feedbax.spec.studio.stage.v1`, and `feedbax.spec.studio.scenario.v2` migrate to presentation-free v2, v2, and v3 semantic records respectively; older mixed `GraphProject` payloads move their presentation fields into the workspace document. | `component_param_array_values` |
| `graph-compiler` | `feedbax.compiler` | `GraphDocument`, `DocumentRoot`, `ResolvedGraph`, `ExecutableGraph`, `CompilationRecord`, `GraphSourceMap`, `GraphSourceMapEntry`, `GraphKeySchedule`, `compile_graph` | `feedbax.graph_document` version `1`, `feedbax.resolved_graph` version `2`, and `feedbax.graph_compilation_record` version `2` are current and reject unknown versions. `compile_graph` accepts semantic documents only, applies the registered GraphSpec migration chain, emits source-map entries with revision-pinned semantic anchors, records truthful authored versus compiler-generated origins, and assigns node keys in deterministic runtime execution order. | `component_param_array_values`, `dynamic_component_ports` |
| `value-identity` | `feedbax.contracts.value_identity` and the same names at `feedbax.contracts` | `ValueIdentityRecord`, `authored_value_sha256`, `semantic_value_sha256`, `realization_value_sha256`, `value_identity_record` | `feedbax.value_identity.v1` is the only accepted record. Authored identity follows canonical declared encoding, semantic identity follows normalized exact numeric meaning, and realization identity additionally binds layout and backend. Other versions reject. | `value_identity` |
| `material-admission` | `feedbax.contracts.material_dependencies` and the same names at `feedbax.contracts` | `AdmissionWaiver`, `IncidentalAdmissionFailure`, `MaterialDependency`, `MaterialDependencyAdmission`, `MaterialDependencyObservation`, `MaterialDependencySet`, `MaterialDependencyValue`, `dependency_value_sha256`, `material_dependency_identity_sha256`, `validate_material_dependency_admission` | Material dependencies use `feedbax.spec.material_dependencies.v1`; the exact narrow waiver uses `feedbax.spec.admission_waiver.v1`. Versionless and other versions reject. Material identity excludes incidental provenance and declaration order; a waiver binds exactly one declared dependency and never bypasses material authenticity. | `material_dependencies` |
| `terminal-certification` | `feedbax.contracts.manifest` and the same names at `feedbax.contracts` | `TRAINING_RUN_CERTIFICATION_MIGRATION_TABLE`, `TrainingRunCertification`, `training_run_certification` | `feedbax.manifest.training_run_certification.v1` is current; v0 is rejected. Legacy completed/cancelled manifests project deterministically, legacy failed and nonterminal manifests reject because their terminal meaning is ambiguous. Focused in-repo migration tests cover this schema, but no external fixture case does, so this row does not claim a T cell. | No external case |
| `custody-persistence` | `feedbax.orchestration` | `PrimaryStatePersistenceError`, `CustodyPreservationRequired`, `RunSetState`, `RunSetStateStore`, `StageEngine` | `feedbax.orchestration.run_set_state.v5` is current and versions v0 through v4 reject. Primary persistence failure retains its typed `OSError` cause; destructive ephemeral teardown is blocked by the typed custody exception until collection is durably complete, then occurs exactly once. | `custody_persistence_recovery` |
| `emergency-persistence` | `feedbax.orchestration` | `ControlFilesystemPreflight`, `ControlFilesystemPreflightError`, `EmergencyProviderIdentity`, `EmergencyRunSetRecord`, `RunSetStateStore` | `feedbax.orchestration.emergency_run_set_record.v1` is current; v0, unknown, and malformed records reject. The bounded reserved channel publishes and reads back provider identity, preservation state, custody completion, spend boundary, primary failure, and next recovery action before destructive teardown. | `custody_persistence_recovery` |
| `result-role-binding` | `feedbax_external_conformance` | `REQUIRED_CASE_IDS`, `V10_REQUIRED_CASE_IDS`, `V11_REQUIRED_CASE_IDS`, `V12_REQUIRED_CASE_IDS`, `V13_REQUIRED_CASE_IDS`, `ConformanceResult`, `ProtocolRoleSlots`, `load_result` | `feedbax.external_conformance.result.v14` preserves the exact thirteen-case v13 inventory in order, appends `figure_role_reference_public_contract`, and requires explicit numeric roles `current = 1` and `minimum = 1`. V13 rejects for missing figure-role-reference evidence and V12 for missing figure-composition evidence; neither migrates synthetically. V1 retains its v2 normalization-before-rejection behavior, and all other earlier decisions remain explicit. | `custody_persistence_recovery` |
| `array-values` | `feedbax.contracts.array_values` and the same names at `feedbax.contracts` | `ArrayValueSpec`, `ConstantArrayValueSpec`, `SparseCooArrayValueSpec`, `SparseCooEntrySpec`, `materialize_array_value` | `feedbax.spec.component_param.array_value.v1` is current. Partial tags and unknown versions reject; canonical sparse COO and constant declarations materialize deterministically. | `component_param_array_values` |
| `dynamic-component-definition` | `feedbax.contracts.component` | `ComponentDefinition`, `DynamicPortLayout`, `DynamicPortPolicy`, `DynamicPortPolicyError`, `derive_dynamic_port_count`, `derive_dynamic_port_layout`, `validate_dynamic_port_layout`, `migrate_component_definition_payload`, `migrate_component_definition_v1_to_v2_payload`, `migrate_component_definition_v2_to_v3_payload` | `feedbax.spec.component_definition.v3` is current. v1 migrates to v2 port kinds; v2 migrates to v3 dynamic-port policy. Dynamic layouts derive only from declared policy and parameters and reject mismatches. | `dynamic_component_ports` |
| `figure-composition` | `feedbax.contracts.figures`; `feedbax.analysis.figures`; `feedbax.analysis.bundles`; `feedbax.contracts.matrix_core`; `feedbax.contracts.run_matrix` | Exact namespace and CLI inventory below. | Composition v2 resolves through one public resolver to ordinary `feedbax.spec.figure.v2` semantics while retaining exact authored identity and ordered full-chain provenance; v1 migrates to v2. Figure-only structural additions use `FigureCompositionDelta` and `SameSchemaStructuralAddition`; every added typed path must be declared exactly, with `feedbax.spec.figure_panel.v1` serving only as the declaration-side `PanelSpec` identity. Bundle execution owns a trusted repository root, Studio rejects unsupported composition with a typed error, runtime binding v2 separates authored and resolved identities, inheritance uses canonical absent-only list-index grafting, and shared `MatrixCompositionDelta` overrides retain prefix-aware acknowledgement with qualified layer attribution. | `figure_composition_public_contract` |
| `figure-role-references` | `feedbax.contracts.row_index`; `feedbax.contracts.figure_roles`; `feedbax.contracts.experiment_envelope` | `migrate_row_index_custody_payload`, `AuthenticatedRowIndex`, `RowIndexEntry`, `RowIndexCustodyBindings`, `RowCustodyBinding`, `ResolvedRowSet`, `RowSetSelector`, `AllRowsSelector`, `TagRowsSelector`, `RowSelectionError`, `RowSelectionErrorCode`, `expand_row_selector`, `normalize_row_tags`, `derive_row_label`; `FigureRowExpansionRequest`, `FigureRoleBindingContract`, `FigureRoleReferenceError`, `PerRowInputReference`, `SharedInputReference`, `ResolvedFigureInput`, `ResolvedFigureInputs`, `resolve_figure_input_roles`, `expand_figure_rows`, `row_namespace`; `ExperimentEnvelopeCompileRequest`, `ExperimentEnvelopeCompileResult`, `ExperimentEnvelopeRejection`, `ExperimentEnvelopeRejectionCategory`, `dispatch_experiment_envelope`, `require_builtin_envelope_schema` | The row-set selector is the closed tagged union `{"mode":"all"}` and `{"mode":"tag","tag":...}`; conjunction, negation, regular expressions, ordering, comparison, and nesting are absent and cannot be added without a new schema version. A selector expands exactly once into an explicit ordered row-id list pinned to the digest of the index it was expanded against; empty selections, duplicate row ids, and ambiguous custody bindings fail closed with stable codes. The index splits at the custody boundary: identity, deterministic order, and bounded normalized tags are compile-time, while authenticated artifact custody attaches post-run, so a first-time figure resolves its roles with no production record. `feedbax.spec.row_index_custody_bindings.v2` is current and adds the required `index_sha256`, which pins the exact index cut the bindings were produced against; `require_index` checks the digest as well as the id, so custody from another cut of one index refuses instead of binding its artifacts. `feedbax.spec.row_index_custody_bindings.v1` states no digest and is rejected rather than migrated in place, because a digest supplied by the reader would be the false authentication the field exists to prevent; `migrate_row_index_custody_payload` is the explicit upgrade and requires the caller to hand over the authoritative index. Row-expanded figure inputs are role references, per-row or shared, and every one of them states the closed artifact contract its fill must satisfy; an authored authority block is rejected with the offending field named, and resolved manifest digests stay in the compile lock rather than the figure's scientific identity. `FigureRoleBindingContract` carries `payload_schema_id` and `payload_schema_version` all-or-none, because half a payload schema identity admits every version of that schema. Row-index order alone derives the `row_{n}__` namespace, panel placement and titles, legend ownership, colorbar placement, and assembler height, producing ordinary `feedbax.spec.figure.v2` semantics. `python -m feedbax preflight-experiment-envelope` compiles an envelope with the one built-in dialect compiler; there is no compiler registry and no registrable dialect, so an envelope declaring any other schema is rejected by name. The dialect's supported versions are enumerated: `feedbax.experiment_envelope.v5` is current and v1/v2/v3/v4 remain accepted, each compiled as the grammar it names. Declared v1/v2 use `feedbax.experiment_envelope.compiler.v1`, declared v3 uses compiler.v2, declared v4 uses compiler.v3, and declared v5 uses compiler.v4; every prior grammar preserves its compiled document and lock bytes apart from the lock's own schema version. v2 adds evaluation `prerequisites` (a mapping from staged binding name to reference, mirroring the compiled `staged_parents` block), an analysis bundle's exact root set, a row-expanded figure's row-custody locator, and the checkpoint-only training layer. v3 adds `training.root`, a closed union over content-pinned `feedbax.spec.training_run_composition.v1` and `feedbax.spec.training_run.v4` parents with explicit row ids and existing typed matrix fields. Root training compiles to `feedbax.spec.training_run_matrix.v6`; Root training may name one whole-document-pinned `feedbax.spec.root_training_authority.v1`; its selected closed object contains only ordered `SourceBinding` and `RowDerivation` lists, which compile authority-first into matrix v6 while the lock retains the authored authority reference and every imported source pin; no runtime include remains. non-root v1/v2 compilation and durable matrix-v5 bytes remain unchanged. Matrix v6 preserves resolved-output row and checkpoint-transaction selectors, adds a closed fork source authority (`execution_hash` or `resolved_output_root`), and admits root-relative selected-checkpoint authority with an explicit source barrier. Target-only durable transforms bind the exact target method and state-slot identity, while optional finite nonnegative absolute LR tolerance is signed fork authority only and does not add a realized-LR reporting interface. V5 migrates to v6 only when an authentic execution hash already exists; Feedbax never synthesizes or relabels one. Resolved-output parents require governed custody at materialization and are never loaded during envelope compilation. Prior-version documents stating later grammar refuse by version. v4 adds the content-pinned root analysis, figure, and comparison-policy authorities and the root comparison-policy layer. v5 makes a root figure input state the closed artifact contract its bound manifest must satisfy — artifact role, provider, media type, decoded payload schema identity, and an explicit `payload_name` — and admits that payload schema pair on any figure input contract; a v5 root figure input stating no contract, or a contract with no explicit `payload_name`, is refused, and an artifact-free v5 root figure states no inputs at all. `feedbax.spec.experiment_compile_lock.v2` is current and carries that contract on the `figure_runtime_input` binding, which is what the runtime `FigureInputRoleAuthority` is built from; `feedbax.spec.experiment_compile_lock.v1` remains readable as its own grammar and a v1 document stating a contract is refused by version. `feedbax.spec.experiment_layer_root_authority.v1` and `feedbax.spec.figure.v2` do not move: runtime inputs and authorities are an execution overlay revalidated before any render effect, never part of a figure's authored identity. `migrate_experiment_envelope_payload` explicitly restates schema v1 to v2, v2 to v3, v3 to v4, and v4 to v5, except that a v4 *root figure* envelope has no v5 migration at all (`migration_intentionally_absent`): the contract v5 requires is not stated anywhere in a v4 root envelope, so such a document is re-authored rather than restated. No compile migrates silently because authored bytes are lock identity. Exit 0 is accepted, 2 is rejected from a closed category set, and 1 is infrastructure failure. | `figure_role_reference_public_contract` |
| `report-surface` | `feedbax.contracts.manifest`; `feedbax.analysis.reports` | `ReportSpec`, `ReportManifest`, `report_manifest_id`, `spec_identity_preimage`; `ORDERED_FIGURE_REPORT_TYPE`, `ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_ID`, `ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_VERSION`, `OrderedFigureReportParams`, `OrderedFigureReportSection`, `OrderedFigureReportFigure`, `ReportRecipeRegistry`, `ReportRecipeResult`, `ReportRecipeExecutionError`, `coerce_report_spec`, `execute_report_spec`, `execute_authored_report_spec`; CLI `feedbax-analysis report` | `feedbax.spec.report.v1` is the current `ReportSpec` version and the document carries its own `schema_id`/`schema_version`. An unversioned historical document is admitted as that one named v1 baseline; `feedbax.spec.report.v0` and unknown versions reject with family, schema ID, source version, current version, and `migration_intentionally_absent`. `feedbax.manifest.v1` is the current `ReportManifest` version and unknown manifest versions reject at load instead of validating. `report_manifest_id` mints identity from the semantic preimage with the schema-identity fields excluded, so a semantics-preserving schema migration never forks the identity of already-produced reports. Ordered-figure params accept only `feedbax.spec.report.ordered_figure.v3`; every included figure requires an authored caption plus a lowercase `figure_spec_sha256`, and figure and scalar-projection input roles stay disjoint. `execute_authored_report_spec` requires authoritative `StagedExactParents`, replaces authored inputs with the authenticated exact refs, refuses a parent carrying `material_dependencies`, and refuses any unauthenticated exact parent. It also accepts an optional already-resolved `execution_context`, mutually exclusive with the raw descriptor and root bindings, which is used as given rather than rebuilt from the exact-parent locators. `REPORT_RECIPES` is deliberately not guaranteed. | No external case |
| `evaluation-surface` | `feedbax.contracts.manifest`; `feedbax.analysis.evaluation` | `EvaluationRunSpec`, `EvaluationRunManifest`, `evaluation_run_manifest_id`; `EvaluationRunMatrixSpec`, `EvaluationRecipeRegistry`, `EvaluationRecipeResult`, `EvaluationBatchExecution`, `coerce_evaluation_run_spec`, `resolve_evaluation_matrix_authoring`, `compile_evaluation_run_matrix`, `materialize_evaluation_run_matrix`, `execute_evaluation_run_matrix`, `execute_evaluation_run_spec`; CLI `feedbax-analysis evaluate` | `feedbax.spec.evaluation_run.v1` is the current `EvaluationRunSpec` version and the document carries its own `schema_id`/`schema_version`. An unversioned historical document is admitted as that one named v1 baseline; `feedbax.spec.evaluation_run.v0`, a foreign `schema_id`, and unknown versions reject. The matrix family coexists across versions: `feedbax.spec.evaluation_run_matrix.v1` migrates to v2 by adding the empty staged-parent map and v2 migrates to v3 combined authoring, while v0 and unknown versions reject. `evaluation_run_manifest_id` excludes the schema-identity fields from its preimage, and switches to the dependency-scoped preimage only when every input carries an exact `material_dependency_identity_sha256` with its retained `material_dependencies` declaration. `feedbax-analysis evaluate` executes a matrix, a matrix delta, or a flat `EvaluationRunSpec` through the same compile, materialize, and execute path; a flat spec requires a stated non-empty escape-hatch reason, and staged parents resolve only against an explicit parent manifest root. | No external case |
| `analysis-authoring` | `feedbax.contracts.manifest`; `feedbax.analysis.specs`; `feedbax.analysis.bundles` | `AnalysisRunSpec`, `AnalysisRunManifest`, `analysis_run_manifest_id`; `execute_analysis_run_spec`; `AnalysisBundleSpec`, `AnalysisBundleDeltaSpec`, `authored_analysis_bundle_from_payload`, `resolve_analysis_bundle_authoring`, `execute_analysis_bundle`, `execute_staged_analysis_bundle`, `dry_run_staged_analysis_bundle`; CLI `feedbax-analysis run`, `feedbax-analysis bundle` | `feedbax.spec.analysis_run.v2` is current and `feedbax.spec.analysis_run.v1` migrates by making the historical implicit `recompute` evaluation-states policy explicit; v0 and unknown versions reject. `feedbax.spec.analysis_bundle.v6` is current and the registered chain migrates v2 through v5; older and unknown versions reject. `feedbax-analysis run` executes one serialized `AnalysisRunSpec` or `AnalysisRunDeltaSpec` with explicit run-alias catalogs and staged execution bindings, and `feedbax-analysis bundle` executes one file-authored bundle without a registered experiment package. A bundle must declare exactly one non-empty execution shape, `--exact-parents` documents must declare `schema_id` and `schema_version` explicitly, and staged bindings require an explicit execution descriptor. `execute_analysis_bundle` and `execute_staged_analysis_bundle` also accept an optional already-resolved `execution_context`, mutually exclusive with the raw descriptor and root bindings, and locate their own selected root refs beneath the manifest root they were selected from. | No external case |
<!-- policy-guarantees:end -->

## Proposed scientific compiler contract amendment

This section describes candidate bytes only. It is not part of the owner-ratified
guarantee block above and becomes stable downstream policy only through the parent-owned
protected delivery and owner-ratification step.

- Amend `graph-compiler` so `feedbax.compiler` additionally guarantees
  `ExperimentDocument`, `CampaignDocument`, `DeclarationRef`, `ScientificSeedDomain`,
  `ResolvedExperiment`, `CompilationFailureRecord`, `CompilerDiagnostic`, `CompilerPhase`,
  `DiagnosticSeverity`, `GraphCompilationError`, and `resolve_experiment`. The candidate
  schemas are `feedbax.experiment_document` v1, `feedbax.campaign_document` v1,
  `feedbax.resolved_experiment` v1, `feedbax.graph_compilation_record` v3, and
  `feedbax.graph_compilation_failure` v1. Earlier and unknown root or record versions
  reject rather than being restamped.
- Add `scientific-declarations` for `feedbax.declarations`, guaranteeing `Declaration`,
  `DeclarationCatalog`, `DeclarationCompositionError`, `DeclarationDocument`, the public
  facet types and constructors, `serialize_declaration`, and `load_declaration`.
  `feedbax.declaration_document` v1 is the sole durable neutral declaration form;
  serialization and loading require explicit protocol identity/registry authority, and
  unsupported versions, duplicate JSON authorities, and unknown protocols reject.

## Owner-ratified envelope-layer prerequisite rows

The rows `report-surface`, `evaluation-surface`, and `analysis-authoring` were
raised by Feedbax issue `88d021d` and ratified by protected `develop` merge
`798c085268119074f0522e3a2313a1722dfaedc8`. They are accepted stability
promises for the surfaces consumed by downstream envelope authoring layers.

Three deliberate limits apply to these guarantees:

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
Namespace `feedbax.plugins` public names (ordered): `DOWNSTREAM_INTERFACE_POLICY_ID`, `DOWNSTREAM_PROTOCOL_CURRENT`, `DOWNSTREAM_PROTOCOL_MINIMUM`, `DOWNSTREAM_POLICY_EFFECTIVE_RELEASE`, `UnsupportedDownstreamProtocolVersion`, `validate_downstream_protocol_version`, `BootstrapError`, `BootstrapErrorCode`, `BootstrapState`, `FamilyRequirement`, `PluginDeclaration`, `PluginDependency`, `PluginProvenance`, `PluginRegistration`, `RegistrationContext`, `RegistryKey`, `RegistryFamilyRegistration`, `bootstrap_application`, `discover_plugin_registrations`, `new_registration_context`, `DRIVERS`, `APPLICATION_REGISTRY_KEYS`, `ApplicationRegistryBundle`, `TrainingProgramCatalog`, `COMPONENTS`, `TRAINING_PROGRAMS`, `ANALYSIS_RECIPES`, `EVALUATION_RECIPES`, `EVALUATION_BATCH_CONSUMERS`, `EVALUATION_PRODUCT_UNION_FINALIZERS`, `DeclaredTrainingProgram`, `TrainingProgramDeclaration`, `TrainingProgramRuntimeFacet`, `TrainingProgramAuthoringFacet`, `TrainingProgramPreparationFacet`, `TrainingProgramProjectionFacet`, `TrainingProgramRowLoweringFacet`, `declare_training_program`, `TrainingMethodAuthoringContribution`, `TrainingMethodAuthoringHook`, `TrainingMethodMetadataProjector`, `TrainingMethodScheduleProjector`, `TrainingProgramRegistry`, `ExecutionPreparationProvider`, `ExecutionPreparationRequest`, `ExecutionPreparationResult`, `ExecutionPreparationPlan`, `compile_training_method_authoring`, `AnalysisRecipe`, `AnalysisRecipeRegistry`, `AnalysisRecipeResult`, `EvaluationRecipe`, `EvaluationRecipeRegistry`, `EvaluationRecipeResult`, `EvaluationBatchRecipe`, `EvaluationAuthoringSchema`, `EvaluationBatchItem`, `EvaluationBatchRowError`, `EvaluationStatesStructureProviderProtocol`, `StagedExecutionContext`, `EMPTY_STAGED_EXECUTION_CONTEXT`, `EvaluationBatchConsumer`, `EvaluationBatchConsumerRegistry`, `EvaluationBatchConsumerInput`, `EvaluationBatchMergeInput`, `EvaluationBatchMergeState`, `EvaluationBatchFinalizeInput`, `EvaluationBatchFragment`, `EvaluationCompactProductUnionFinalizerRegistry`, `EvaluationCompactProductUnionInput`.

Direct downstream entrypoint imports (ordered): `EVALUATION_RECIPES`, `FamilyRequirement`, `PluginDeclaration`, `PluginDependency`, `PluginRegistration`, `RegistrationContext`, `ANALYSIS_RECIPES`, `EVALUATION_BATCH_CONSUMERS`, `EVALUATION_PRODUCT_UNION_FINALIZERS`, `TRAINING_PROGRAMS`.

| Family key | Registry type | Registry methods (ordered) | Callback types (ordered) | Support types (ordered) | Public consumers (ordered) |
|---|---|---|---|---|---|
| `training_programs` | `TrainingProgramCatalog` | `register_program`, `resolve`, `validate_payload`, `resolve_execution` | `DeclaredTrainingProgram` | `TrainingProgramDeclaration`, `TrainingProgramRuntimeFacet`, `TrainingProgramAuthoringFacet`, `TrainingProgramPreparationFacet`, `TrainingProgramProjectionFacet`, `TrainingProgramRowLoweringFacet`, `ExecutionPreparationRequest`, `ExecutionPreparationResult`, `ExecutionPreparationPlan` | `feedbax.training.authoring:compile_training_method_authoring` |
| `analysis_recipes` | `AnalysisRecipeRegistry` | `register`, `get`, `structure_provider` | `AnalysisRecipe` | `AnalysisRecipeResult`, `EvaluationStatesStructureProviderProtocol` | `feedbax.analysis:resolve_analysis_inputs`, `feedbax.analysis:execute_analysis_run_spec` |
| `evaluation_recipes` | `EvaluationRecipeRegistry` | `register`, `register_authoring_schema`, `get`, `batch` | `EvaluationRecipe`, `EvaluationBatchRecipe` | `EvaluationRecipeResult`, `EvaluationAuthoringSchema`, `EvaluationBatchItem`, `EvaluationBatchRowError`, `StagedExecutionContext`, `EMPTY_STAGED_EXECUTION_CONTEXT` | `feedbax.analysis.evaluation:compile_evaluation_run_matrix`, `feedbax.analysis.evaluation:execute_evaluation_run_matrix`, `feedbax.analysis.evaluation:execute_evaluation_run_spec` |
| `evaluation_batch_consumers` | `EvaluationBatchConsumerRegistry` | `register`, `get` | `EvaluationBatchConsumer` | `EvaluationBatchConsumerInput`, `EvaluationBatchMergeInput`, `EvaluationBatchMergeState`, `EvaluationBatchFinalizeInput`, `EvaluationBatchFragment` | `feedbax.analysis.evaluation_compaction:compact_evaluation_batch`, `feedbax.analysis.evaluation_compaction:merge_evaluation_batch_fragment`, `feedbax.analysis.evaluation_compaction:reclaim_evaluation_batch_caches`, `feedbax.analysis.evaluation_compaction:publish_evaluation_compaction_products` |
| `evaluation_product_union_finalizers` | `EvaluationCompactProductUnionFinalizerRegistry` | `register`, `get` | none | `EvaluationCompactProductUnionInput` | `feedbax.analysis.evaluation_product_union:finalize_evaluation_compact_product_union` |
<!-- plugin-api-inventory:end -->

It is the machine-readable mapping of these row IDs to schemas and real case
IDs. V14 preserves the exact thirteen-case v13 inventory and order, then appends
the figure-role-reference case. V13 in turn preserved the twelve-case v12
inventory and appended the figure-composition case, and v12 already carried the
v10 dynamic-component and v11 external-driver evidence, added the
custody/persistence case, and bound both numeric protocol roles.
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
that validated v14 result as an artifact. V14 is the current result and adds
figure-role-reference evidence to the exact v13 inventory; v13 and v12 remain
explicit historical evidence and reject rather than acquiring later cases
synthetically.
