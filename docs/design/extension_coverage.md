# Extension capability lifecycle coverage

This is the baseline snapshot for Feedbax issue `e4df04e`. All Feedbax source
and test evidence below was audited at commit
`4876398564463dc83ab13bbd96807415fc3426eb`. The snapshot describes what an
installed external package can do through Feedbax's public mechanisms; an
in-repository built-in is not evidence that the corresponding extension seam is
open. Incremental witnesses `E17`, `E18`, and `E19` were reconciled on the umbrella
integration branch after the baseline audit; they do not silently reclassify
unrelated cells.

## Cell semantics

- **Open (`O`)**: an external package can use the capability at that lifecycle
  stage without editing Feedbax core. Every open cell names an evidence witness.
- **Partial (`P`)**: the cited witness works, but the named issue or deferral
  owns a remaining closed edge.
- **Closed (`C`)**: no external-package path exists at that stage. The named
  issue or explicit deferral owns the edge.
- **Not applicable (`—`)**: the stage has no capability-specific behavior. It
  is not a claim that a missing implementation is harmless.

Evidence keys (`E1`, and so on) resolve to durable `file:symbol` references
below. Owner keys resolve to existing issues or explicit deferrals. Thus, for
example, `P E1 / 9ac21f2` means “the E1 path works except for the edge owned by
issue 9ac21f2.”

Lifecycle abbreviations:

| Key | Stage | Key | Stage |
|---|---|---|---|
| `I` | installation / bootstrap | `P` | preflight |
| `A` | authoring | `G` | authorization / governance |
| `M` | schema / migration | `X` | execution |
| `D` | discovery | `R` | progress |
| `V` | validation | `C` | custody |
| `L` | lowering | `U` | resume / recovery |
| `Q` | reporting | `S` | Studio rendering |
| `T` | stability / conformance | | |

## Evidence catalog

| Key | Source witness | Test witness | What it proves |
|---|---|---|---|
| `E1` | `feedbax/component_registry/registry.py:ComponentRegistry`, `feedbax/plugins/application.py:COMPONENTS`, `feedbax/compiler/graph.py:compile_graph` | `tests/test_dynamic_port_policy.py`, `tests/test_graph_normalization.py:test_normalization_materializes_external_dynamic_ports_from_explicit_registry`, `tests/test_graphspec_builtins.py:test_graphspec_build_materializes_omitted_dynamic_ports_from_policy` | External components register into a fresh application bundle with v3 dynamic-port policy metadata; explicit registry injection derives deterministic ports from isolated effective parameters, validates namespaces, materializes GraphSpec nodes, and executes them. |
| `E2` | `feedbax/plugins/bootstrap.py:discover_plugin_registrations`, `feedbax/plugins/bootstrap.py:bootstrap_application`, `feedbax/plugins/application.py:ApplicationRegistryBundle`, `external/feedbax_conformance_fixture/src/feedbax_external_conformance/family.py:FIXTURE_RECORDS` | `tests/test_plugin_discovery_import_order.py:test_importing_plugins_does_not_eagerly_discover`, `tests/test_plugin_bootstrap.py`, `external/feedbax_conformance_fixture/src/feedbax_external_conformance/cases.py:check_unified_plugin_bootstrap` | One typed entry-point group discovers installed declarations without import-time side effects, accepts a typed registry family added outside Feedbax, validates protocols and dependencies, registers transactionally in deterministic order, and publishes isolated sealed caller-owned bundles with attributable provenance. |
| `E3` | `feedbax/contracts/training.py:TaskSpec`, `feedbax/contracts/graphs/builders.py:_build_task_component`, `feedbax/integrations/provider.py:task_registry_snapshot`, `feedbax/analysis/rollout.py:compiled_trial_rollout`, `feedbax/web/api/execution.py:sample_task_trials`, `web/src/components/panels/TaskScenarioPanel.tsx:TASK_CATALOG` | `tests/test_analysis_compiled_rollout.py:test_compiled_rollout_is_bit_identical_to_python_loop`, `tests/test_execution_task_sampling_api.py:test_sample_task_trials_rejects_unsupported_task_type` | Built-in tasks are authorable, compilable, executable, and previewable, but construction, provider validation, and the Studio catalog remain closed sets. |
| `E4` | `feedbax/analysis/specs.py:AnalysisRecipeRegistry`, `feedbax/analysis/validation.py:AnalysisRecipeProtocol`, `feedbax/analysis/specs.py:execute_analysis_run_spec` | `tests/test_analysis_registration_validation.py:test_valid_analysis_recipe_registers`, `tests/test_analysis_run_cli.py:test_run_subcommand_loads_installed_plugin_before_recipe_execution` | External analysis recipes register through the typed bootstrap, validate their callable contract, resolve authenticated inputs, execute, and emit manifests. Recipe params remain an untyped mapping. |
| `E5` | `feedbax/plot/constructors.py:FigureRegistry`, `feedbax/contracts/figures.py:FigureSpec`, `feedbax/analysis/figures.py:execute_figure_spec`, `feedbax/bin/figure.py:main` | `tests/test_declarative_figures.py:test_constructor_registry_validates_tiers_and_duplicates`, `tests/test_declarative_figures.py:test_execute_figure_spec_records_optional_omission_and_custody` | Figure constructors, templates, and pieces register through the typed bootstrap; CLI and web dispatch consume the injected sealed figure registry. |
| `E6` | `feedbax/contracts/array_values.py:ArrayValueSpec`, `feedbax/contracts/array_values.py:materialize_array_value`, `feedbax/contracts/graph.py:ComponentSpec`, `feedbax/compiler/graph.py:compile_graph` | `tests/test_array_value_specs.py:test_dense_sparse_and_dense_constant_share_semantic_identity`, `tests/test_structural_linear_state_space.py:test_nested_v4_sparse_migration_materializes_losslessly_and_preserves_envelope`, `external/feedbax_conformance_fixture/src/feedbax_external_conformance/cases.py:check_component_param_array_values` | Raw dense component params remain unchanged. Versioned sparse COO and constant declarations validate, migrate, preserve authored envelopes through read-only round trips, and materialize before component migration/build. Other spec surfaces and content-pinned references remain outside this seam. |
| `E7` | `feedbax/orchestration/bundle.py:ExecutionFamily`, `feedbax/orchestration/assembly.py:AssemblyCompilerRegistry`, `feedbax/orchestration/assembly.py:build_default_assembly_registry`, `feedbax/orchestration/executor_family.py:executor_family_adapter` | `tests/test_evaluation_orchestration.py:test_provider_free_cli_shadow_reaches_terminal_collection_in_fresh_process`, `tests/test_orchestration_core.py:test_stage_engine_resumes_from_every_stage_boundary` | Native training and evaluation matrices traverse the governed lifecycle. Compiler construction and lifecycle dispatch still enumerate the two core families. |
| `E8` | `feedbax/orchestration/drivers/capabilities.py:DriverRegistry`, `feedbax/orchestration/drivers/capabilities.py:DriverCapabilityEnvelope`, `feedbax/plugins/application.py:DRIVERS`, `feedbax/orchestration/stages.py:StageEngine`, `feedbax/orchestration/bundle.py:DeploymentPolicy` | `tests/test_driver_capabilities.py`, `tests/test_orchestration_cli.py:test_two_row_local_driver_demo_through_cli`, `tests/test_orchestration_core.py:test_stage_engine_resumes_from_every_stage_boundary` | External drivers register through the unified injected bootstrap, select context-sensitive realized facts, validate their declared callable groups, and traverse preflight and lifecycle construction without CLI edits or driver-name dispatch. Open-string policy and its nested request/bundle schemas migrate explicitly. |
| `E9` | `feedbax/contracts/artifact_custody.py:ImmutableArtifactBlobProviderSpec`, `feedbax/persistence/artifact_custody.py:ImmutableArtifactBlobProvider`, `feedbax/persistence/artifact_custody.py:open_immutable_artifact_blob_provider`, `feedbax/orchestration/staged_root_custody.py:StagedRootKind` | `tests/test_artifact_custody.py:test_portable_provider_spec_has_exact_root_free_json_and_round_trips`, `tests/test_artifact_custody.py:test_custody_survives_source_directory_deletion_and_materializes_copy` | The built-in immutable provider has portable schema, validation, storage, and recovery, but provider kinds and factories are fixed. |
| `E10` | `feedbax/contracts/schema_namespace.py:validate_schema_identity`, `feedbax/contracts/migrations.py:SpecSchemaFamily`, `feedbax/contracts/migrations.py:SpecSchemaRegistry` | `tests/test_structured_spec_migrations.py:test_structured_spec_registry_applies_registered_family_migration`, `tests/test_structured_spec_migrations.py:test_structured_spec_registry_reports_unknown_family` | Schema families have a public version/migration/rejection mechanism. The public authoring/provider projection is still separately curated. |
| `E11` | `feedbax/analysis/reports.py:ReportRecipeRegistry`, `feedbax/analysis/reports.py:execute_report_spec`, `feedbax/analysis/reports.py:OrderedFigureReportParams` | `tests/test_report_execution.py:test_report_spec_executes_registered_recipe_and_writes_markdown_render`, `tests/test_ordered_figure_report.py:test_ordered_figure_report_html_is_self_contained_interactive_and_deterministic` | Report recipes register through the typed bootstrap and emit governed artifacts; the built-in ordered report supports Markdown and self-contained HTML. |
| `E12` | `feedbax/contracts/training.py:TrainingMethodDescriptor`, `feedbax/contracts/training.py:TrainingMethodRegistry`, `feedbax/training/authoring.py:compile_training_method_authoring`, `feedbax/training/row_lowering.py:TrainingRowLowererRegistry`, `feedbax/training/executor.py:execute_training_run_spec` | `tests/test_training_authoring.py:test_default_matrix_compiler_reaches_descriptor_authoring`, `tests/test_training_method_plugin_cli.py:test_entry_point_descriptor_derives_method_and_preparation_from_one_hook`, `tests/test_training_row_lowering.py:test_orchestration_cli_discovers_and_lowers_downstream_rows` | A downstream training method can register, author, lower, prepare, execute, report progress, checkpoint, and resume through the governed CLI path. |
| `E13` | `feedbax/studio/schema.py:enumerate_studio_schema_registry`, `feedbax/studio/schema.py:validate_graph_connection_schema`, `feedbax/web/api/analysis.py:_run_analysis_sync` | `tests/test_provider_contract.py:test_studio_schema_materializes_external_dynamic_policy_without_type_branching`, `tests/test_studio_workspace.py:test_studio_save_load_materializes_dynamic_ports_with_explicit_registry`, `tests/test_graph_compiler.py:test_workspace_view_edits_cannot_change_semantic_or_runtime_identity`, `tests/test_worker_execution.py:test_worker_request_materializes_omitted_dynamic_ports_from_bootstrap_registry` | Studio renders registered component metadata, persists policy-materialized dynamic ports, and passes them through worker validation without component-type branching. Studio presentation persists in a separate `WorkspaceDocument`; its revision-pinned anchors consume compiler source maps, and view edits cannot change semantic or runtime identity. Not every extension family is projected into authoring UI. |
| `E14` | `feedbax/testing/suite.py:load_suite_manifest`, `feedbax/orchestration/conformance.py:CheckRegistry`, `external/feedbax_conformance_fixture/src/feedbax_external_conformance/result.py:ConformanceResult` | `tests/test_external_conformance_fixture.py`, `external/feedbax_conformance_fixture/src/feedbax_external_conformance/cases.py:check_dynamic_component_ports`, `external/feedbax_conformance_fixture/src/feedbax_external_conformance/cases.py:check_external_driver_plugin`, `external/feedbax_conformance_fixture/src/feedbax_external_conformance/cases.py:check_figure_composition_public_contract`, `external/feedbax_conformance_fixture/src/feedbax_external_conformance/cases.py:check_figure_role_reference_public_contract` | The v14 clean-installed fixture preserves the exact thirteen-case v13 inventory and order, appends figure-role-reference evidence, binds numeric current/minimum protocol roles to 1, and rejects v13 and v12 rather than synthesizing missing evidence. Exact case order, clean-install isolation, private-import and network gates, and explicit old-version decisions remain enforced. |
| `E15` | `feedbax/orchestration/state.py:RunSetStateStore`, `feedbax/orchestration/stages.py:StageEngine`, `feedbax/orchestration/collection_recovery.py:recover_collected_outputs` | `tests/test_orchestration_cli.py:test_watch_exits_after_all_rows_terminal`, `tests/test_orchestration_cli.py:test_collect_and_teardown_are_idempotent_after_completed_run` | The governed run lifecycle supplies progress, persisted state, collection recovery, and idempotent teardown for admitted families/drivers. |
| `E16` | `feedbax/analysis/specs.py:resolve_analysis_inputs`, `feedbax/analysis/specs.py:ResolvedEvaluationStateHandle`, `feedbax/analysis/evaluation_rows.py:project_evaluation_rows` | `tests/test_evaluation_row_projection.py:test_projects_all_resolver_source_kinds`, `tests/test_evaluation_row_projection.py:test_manifest_facts_come_from_authenticated_raw_bytes`, `tests/test_evaluation_row_projection.py:test_forged_exact_type_handle_is_rejected`, `tests/test_evaluation_row_projection.py:test_two_genuine_rows_cannot_splice_manifest_authority`, `tests/test_evaluation_row_projection.py:test_cross_row_source_retarget_is_rejected`, `external/feedbax_conformance_fixture/src/feedbax_external_conformance/cases.py:check_resolved_evaluation_row_projection` | The resolver issues one exact runtime handle with a private issuance sentinel and canonical immutable snapshots of the complete typed source and portable requested-manifest authority. Projection compares those small facts to the current source and raw-byte-authenticated authority, exposes the resolver-supplied state, and invokes one downstream cross-field callback with compact stable error codes. Durable content authentication remains with the durable loader; cache/recompute bytes and post-resolution state mutation are outside the handle guarantee. Coverage, duplicates, and scientific verdicts remain downstream. |
| `E17` | `feedbax/contracts/value_identity.py:ValueIdentityRecord`, `feedbax/contracts/value_identity.py:authored_value_sha256`, `feedbax/contracts/value_identity.py:semantic_value_sha256`, `feedbax/contracts/value_identity.py:realization_value_sha256` | `tests/test_value_identity.py:test_cross_encoding_semantic_equality_and_authored_inequality`, `tests/test_value_identity.py:test_expected_semantic_mismatch_fails_closed_and_chain_is_preserved`, `tests/test_value_identity.py:test_realization_uses_only_explicit_fingerprints` | The public v1 contract separates authored, exact normalized semantic, and explicit runtime realization identity. Existing durable envelopes do not embed it yet; their owning consumers retain their migration responsibility. |
| `E18` | `feedbax/contracts/material_dependencies.py:MaterialDependencySet`, `feedbax/contracts/material_dependencies.py:validate_material_dependency_admission`, `feedbax/contracts/manifest.py:TrainingRunCertification`, `feedbax/analysis/bundles.py:_preflight_staged_exact_parents`, `feedbax/testing/material_dependencies.py:check_material_dependency_contract` | `tests/test_material_dependencies.py:test_identity_projects_refs_and_values_to_material_identity`, `tests/test_staged_exact_parents.py:test_diverged_run_admits_certified_checkpoint_and_scopes_evaluation_identity`, `tests/test_staged_exact_parents.py:test_material_dependency_admission_rejects_before_outputs`, `tests/test_report_execution.py:test_authored_report_rejects_unhandled_material_dependencies_before_outputs` | Versioned declarations scope identity to material content or semantic value identity, authenticate certified dependency bytes through existing custody/provider authority, and permit only an exact authored incidental-check waiver. Ambiguous legacy failed manifests reject; authored reports fail closed unless parents first pass the shared bundle preflight. |
| `E19` | `feedbax/orchestration/state.py:RunSetStateStore`, `feedbax/orchestration/state.py:EmergencyRunSetRecord`, `feedbax/orchestration/stages.py:StageEngine`, `external/feedbax_conformance_fixture/src/feedbax_external_conformance/lifecycle.py:check_custody_persistence_recovery` | `tests/test_orchestration_state_persistence.py`, `tests/test_orchestration_core.py`, `tests/test_external_conformance_fixture.py:test_custody_persistence_case_uses_public_installed_contract` | The public lifecycle reserves bounded control capacity before driver actions, retains typed primary persistence failure, publishes and reads back a versioned emergency recovery record, blocks destructive ephemeral teardown until custody is durable, and permits exactly one deletion after recovered collection. This witness does not open the separately deferred custody-provider extension family or claim terminal-certification external coverage. |
| `E20` | `feedbax/contracts/figures.py:FigureCompositionSpec`, `feedbax/contracts/figures.py:FigureCompositionDelta`, `feedbax/contracts/figures.py:SameSchemaStructuralAddition`, `feedbax/analysis/figures.py:resolve_figure_spec`, `feedbax/analysis/bundles.py:execute_staged_analysis_bundle`, `feedbax/contracts/matrix_core.py:materialize_inherited_document`, `feedbax/contracts/run_matrix.py:apply_composition_deltas` | `external/feedbax_conformance_fixture/src/feedbax_external_conformance/cases.py:check_figure_composition_public_contract`, `tests/test_figure_composition.py`, `tests/test_source_document_inheritance.py` | One public resolver and coercer produce ordinary FigureSpec v2 semantics with separate raw-authored/resolved identity and ordered full-chain custody. FigureCompositionSpec v2 owns the figure-only delta and migrates v1 without semantic change. Same-schema additions require an exact complete typed-path declaration; `feedbax.spec.figure_panel.v1` identifies PanelSpec only in that declaration and never changes resolved panel bytes. The clean-installed witness exercises CLI parity, trusted bundle-root admission, runtime binding v2 identities, canonical absent-only list-index inheritance, and the unchanged shared MatrixCompositionDelta prefix-aware acknowledgement guarantee without a second composition or path language. |
| `E21` | `feedbax/contracts/row_index.py:AuthenticatedRowIndex`, `feedbax/contracts/row_index.py:RowIndexCustodyBindings`, `feedbax/contracts/row_index.py:expand_row_selector`, `feedbax/contracts/figure_roles.py:FigureRowExpansionRequest`, `feedbax/contracts/figure_roles.py:resolve_figure_input_roles`, `feedbax/contracts/figure_roles.py:expand_figure_rows`, `feedbax/contracts/experiment_envelope.py:dispatch_experiment_envelope`, `feedbax/__main__.py:preflight-experiment-envelope` | `external/feedbax_conformance_fixture/src/feedbax_external_conformance/cases.py:check_figure_role_reference_public_contract`, `tests/test_figure_role_references.py`, `tests/test_experiment_envelope_dispatch.py` | Row-set selection is a closed two-member tagged union expanded once into an explicit ordered row-id list pinned to its source index digest; empty, duplicate, and ambiguous selections fail closed with stable codes. The authenticated row index splits at the custody boundary, so compile-time identity/order/tags resolve without any post-run production record and authenticated artifact custody attaches afterwards. Row-expanded figure inputs are per-row or shared role references; a root figure input names one manifest and states the closed artifact contract it is read under — artifact role, provider, media type, decoded payload schema identity, and an explicit payload name — which the compile lock carries and the runtime input authority is built from. Authored authority blocks are rejected by field, half a payload schema identity is rejected as neither stated nor omitted, and resolved digests stay in the compile lock. Row-index order alone derives the row namespace, panel placement and titles, legend ownership, colorbar placement, and assembler height, producing unchanged FigureSpec v2 semantics. One entrypoint compiles an authored envelope with the single built-in dialect compiler; no registry mediates dispatch and an envelope declaring any other schema is rejected by name. |

## Coverage matrix

### Bootstrap through lowering

| Capability | `I` | `A` | `M` | `D` | `V` | `L` |
|---|---|---|---|---|---|---|
| Component | `P E1 / 301dce2` | `O E1` | `O E1` | `P E1 / 301dce2` | `O E1` | `O E1` |
| Task | `C D-TASK` | `P E3 / D-TASK` | `O E10` | `C D-TASK` | `P E3 / D-TASK` | `C D-TASK` |
| Analysis recipe | `P E2 / 301dce2` | `O E4` | `O E10` | `P E4 / 301dce2` | `P E4 / 46aeab1` | `O E4` |
| Figure | `C 301dce2` | `O E5, E20` | `O E10, E20` | `P E5 / 301dce2` | `O E5, E20` | `O E5, E20` |
| Value encoding | `—` | `O E6` | `O E6, E17` | `C c50193e` | `O E6` | `O E6` |
| Run kind | `C D-RUN` | `P E7 / D-RUN` | `P E7 / D-RUN` | `C D-RUN` | `C D-RUN` | `P E7 / D-RUN` |
| Driver | `O E8` | `O E8` | `O E8` | `O E8` | `O E8` | `—` |
| Custody provider | `C D-CUSTODY` | `P E9 / D-CUSTODY` | `O E9` | `C D-CUSTODY` | `O E9` | `P E9 / D-CUSTODY` |
| Schema family | `—` | `P E10 / 6b6a44b` | `O E10` | `P E10 / 6b6a44b` | `O E10` | `—` |
| Report format | `P E2 / 301dce2` | `O E11` | `O E10` | `P E11 / 301dce2` | `O E11` | `O E11` |
| Training method | `P E2 / 301dce2` | `O E12` | `O E10` | `P E12 / 301dce2` | `O E12` | `O E12` |

### Preflight through custody

| Capability | `P` | `G` | `X` | `R` | `C` |
|---|---|---|---|---|---|
| Component | `O E1` | `—` | `O E1` | `—` | `—` |
| Task | `P E3 / D-TASK` | `—` | `P E3 / D-TASK` | `—` | `—` |
| Analysis recipe | `O E4` | `O E16, E18` | `O E4` | `—` | `O E4` |
| Figure | `O E5, E20` | `O E16, E20` | `P E5 / 301dce2` | `—` | `O E5, E20` |
| Value encoding | `O E6` | `O E17` | `O E6` | `—` | `O E6` |
| Run kind | `P E7 / D-RUN` | `P E7 / D-RUN` | `P E7 / D-RUN` | `P E15 / D-RUN` | `P E7 / D-RUN` |
| Driver | `O E8` | `O E8` | `O E8` | `O E8, E15` | `O E8` |
| Custody provider | `P E9 / D-CUSTODY` | `P E9 / D-CUSTODY` | `P E9 / D-CUSTODY` | `—` | `O E9` |
| Schema family | `O E10` | `O E10` | `—` | `—` | `O E10` |
| Report format | `O E11` | `O E16` | `P E11 / 301dce2` | `—` | `O E11` |
| Training method | `O E12` | `O E16` | `O E12` | `O E12` | `O E12` |

### Recovery through conformance

| Capability | `U` | `Q` | `S` | `T` |
|---|---|---|---|---|
| Component | `—` | `—` | `P E13 / 301dce2` | `P E14 / f4476ae, 380f897` |
| Task | `—` | `—` | `P E3 / D-TASK` | `C D-TASK` |
| Analysis recipe | `—` | `O E4` | `P E13 / 301dce2` | `P E14, E16 / f4476ae` |
| Figure | `—` | `O E5, E20` | `P E13 / 301dce2` | `O E14, E20` |
| Value encoding | `—` | `—` | `O E6` | `O E6, E14` |
| Run kind | `P E15 / D-RUN` | `P E7 / D-RUN` | `C D-RUN` | `C D-RUN` |
| Driver | `O E8, E15, E19` | `O E8, E15, E19` | `—` | `O E14, E19` |
| Custody provider | `O E9` | `O E9` | `—` | `C D-CUSTODY` |
| Schema family | `O E10` | `P E10 / 6b6a44b` | `P E13 / 6b6a44b` | `P E14 / f4476ae, 380f897` |
| Report format | `—` | `O E11` | `P E13 / 301dce2` | `P E14 / f4476ae, 380f897` |
| Training method | `O E12` | `O E12` | `C 0491f60` | `P E14 / f4476ae, 380f897` |

## Owners and explicit deferrals

The issue owner keys above are deliberately existing issues. They do not widen
those issues beyond their recorded bodies:

- `301dce2`: unified extension discovery/bootstrap; also owns the figure CLI
  plugin-loading gap.
- `f4476ae`: guaranteed-surface and compatibility-window policy.
- `380f897`: clean-wheel external conformance fixture foundation.
- `cd43b83`: authored, semantic, and realization value identity.
- `9757814`: built-in array value encodings and their materialization boundary.
- `c50193e`: deferred registered-constructor escape hatch.
- `69034e6`: delivered orchestration-driver capability contract, injected registry,
  lifecycle construction, schema migrations, and external conformance.
- `9ac21f2`: component dynamic-port definition, validation, and Studio authoring.
- `8ca2ade`: typed authenticated evaluation-row projection.
- `43891d0`: delivered material-dependency-scoped identity, byte-authenticated
  admission, exact waiver semantics, and terminal/certified-prefix separation.
- `46aeab1`: typed analysis-recipe parameter model.
- `6b6a44b`: public authoring/provider projection from `SpecSchemaRegistry`.
- `0491f60`: deferred Studio-launchable training-method extension point.

The following are explicit deferrals rather than executable placeholder
surfaces:

- `D-RUN`: run-kind capability contract, recorded as deferred issue `fb58eff`.
  The current `ExecutionFamily` axis may or may not be identical to “run kind”;
  that decision stays inside the deferral. If executor family is a distinct
  axis, the parent must route it separately rather than widening `fb58eff`.
- `D-CUSTODY`: custody-provider capability contract, explicitly deferred by
  umbrella `f8a5183`. The built-in immutable artifact and checkpoint providers
  remain usable; no provider registry or generic capability protocol is implied.
- `D-TASK`: external task construction/provider-validation/Studio-preview
  capability is unmapped in the current umbrella. Existing issue `38d5c7e`
  supplied the component-aware compiled execution boundary and `79dc3d1`
  supplied built-in sampled preview, but neither owns an external task
  definition/catalog contract. No new issue is created here; the parent must
  decide whether to defer or route this edge.

## Prioritized seam sequence

The baseline preserves the ratified umbrella order:

1. Keep `5262573` as the standing downstream-intake gate. `4fa5721` has already
   landed and established explicit evaluation-output adequacy before this
   snapshot.
2. Land this baseline (`e4df04e`).
3. Decide `f4476ae`, `cd43b83`, and `43891d0` together: stability, value
   identity, and material-dependency admission constrain one another.
4. Establish the external canary in `380f897`.
5. Implement `8ca2ade`, migrate two materially different downstream consumers,
   and measure deleted compensation code before expanding the toolkit.
6. Queue `9757814` after those foundations, limited to its named GraphSpec
   integration sites.
7. Treat `301dce2`, `69034e6`, and any distinct executor-family work as
   unscheduled quiet-window work. They require a single execution path,
   plan-level comparison, downstream pins, and a passing external fixture on
   the new revision; this matrix does not authorize immediate construction.
8. Consume landed seams downstream through `rlrmp2/2440ab1`. Keep `c50193e`,
   `fb58eff`, and `D-CUSTODY` deferred until their activation conditions are
   met.

## Downstream candidate classification

The ca09544 assessment and current downstream candidates use four bins:

| Bin | Routing rule | Current examples |
|---|---|---|
| Generic mechanism | Move upstream only after two independent consumers demonstrate the same need, unless the evidence is a clear boundary defect. | Typed authenticated row projection (`8ca2ade`); built-in value encodings (`9757814`); unified discovery (`301dce2`). |
| Downstream scientific semantics | Remain downstream by design. | Conditioning membership, target geometry/support, reach constraints, replicate coverage, comparator meaning, and the row-specific scientific delta in the unmerged `rlrmp2/e786cb7` lane. |
| Legitimate explicit evidence | Keep deliberately; it is evidence, not framework duplication. | Golden semantic identities, telemetry/parity checks, physics replay, authored exact-parent pins, and explicit conformance certificates when they test the promised relationship. |
| Avoidable duplication | Delete or replace through the named upstream seam and `rlrmp2/2440ab1`. | Untyped `manifest.metadata` parsing, substring-matched errors, duplicated staged-context/provider scaffolds, repeated literal arrays where a governed encoding exists, and per-consumer generic authority rechecks. |

The two-independent-consumer rule applies to general abstractions. A clear
boundary defect does not wait for a second consumer; untyped authenticated
metadata access and admission/identity keyed to non-dependencies are such
defects. The active `rlrmp2/e786cb7` work is provisional and unmerged. Its
failed-manifest/checkpoint compensation now has the integrated upstream `E18`
replacement and routes to `rlrmp2/2440ab1` for adoption and deletion; its
comparator-authority reconciliation and row-specific scientific delta remain
downstream absent a second independent consumer. This does not advance the
ca09544 baseline recorded by `5262573`.

## Maintenance

Each seam child updates only the cells it changes, replaces evidence keys with
the landed `file:symbol` and focused test witnesses, and adds its external
conformance slice in the same change. A cell becomes open only after that
external package path works without core edits. Changes to the capability list,
lifecycle list, or an unmapped owner return to umbrella `f8a5183`; they are not
silently absorbed by the nearest implementation child.
