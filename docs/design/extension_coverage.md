# Extension capability lifecycle coverage

This is the baseline snapshot for Feedbax issue `e4df04e`. All Feedbax source
and test evidence below was audited at commit
`4876398564463dc83ab13bbd96807415fc3426eb`. The snapshot describes what an
installed external package can do through Feedbax's public mechanisms; an
in-repository built-in is not evidence that the corresponding extension seam is
open. Incremental witnesses `E17` and `E18` were reconciled on the umbrella
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
| `E1` | `feedbax/component_registry/registry.py:ComponentRegistry`, `feedbax/component_registry/registry.py:register_component_type`, `feedbax/contracts/graphs/serialization.py:spec_to_graph` | `tests/test_component_registration.py:test_entry_point_component_registration_records_package_provenance`, `tests/test_component_registration.py:test_downstream_migration_pack_can_migrate_owned_component_id` | Static external components can register, migrate, materialize, and execute. Component discovery is a separate fail-open loader, and dynamic-port policy is not part of the definition. |
| `E2` | `feedbax/plugins/discovery.py:feedbax_plugin_entry_points`, `feedbax/plugins/discovery.py:load_training_method_plugins`, `feedbax/component_registry/registry.py:ComponentRegistry.discover_entry_point_components` | `tests/test_plugin_discovery_import_order.py:test_importing_plugins_does_not_eagerly_discover`, `tests/test_orchestration_cli.py:test_broken_installed_plugin_fails_before_builtin_matrix_engine_or_provider` | Installed plugins are loadable, but family-specific loaders, registrar-name allowlists, and different failure policies remain. |
| `E3` | `feedbax/contracts/training.py:TaskSpec`, `feedbax/contracts/graphs/builders.py:_build_task_component`, `feedbax/integrations/provider.py:task_registry_snapshot`, `feedbax/analysis/rollout.py:compiled_trial_rollout`, `feedbax/web/api/execution.py:sample_task_trials`, `web/src/components/panels/TaskScenarioPanel.tsx:TASK_CATALOG` | `tests/test_analysis_compiled_rollout.py:test_compiled_rollout_is_bit_identical_to_python_loop`, `tests/test_execution_task_sampling_api.py:test_sample_task_trials_rejects_unsupported_task_type` | Built-in tasks are authorable, compilable, executable, and previewable, but construction, provider validation, and the Studio catalog remain closed sets. |
| `E4` | `feedbax/analysis/specs.py:register_analysis_recipe`, `feedbax/analysis/validation.py:AnalysisRecipeProtocol`, `feedbax/analysis/specs.py:execute_analysis_run_spec` | `tests/test_analysis_registration_validation.py:test_valid_analysis_recipe_registers`, `tests/test_analysis_run_cli.py:test_run_subcommand_loads_installed_plugin_before_recipe_execution` | External analysis recipes can register, validate their callable contract, resolve authenticated inputs, execute, and emit manifests. Recipe params remain an untyped mapping. |
| `E5` | `feedbax/plot/constructors.py:register_figure_constructor`, `feedbax/contracts/figures.py:FigureSpec`, `feedbax/analysis/figures.py:execute_figure_spec`, `feedbax/bin/figure.py:main` | `tests/test_declarative_figures.py:test_constructor_registry_validates_tiers_and_duplicates`, `tests/test_declarative_figures.py:test_execute_figure_spec_records_optional_omission_and_custody` | Figure constructors/specs are open in-process and execution is governed, but the figure CLI consumes registries without loading installed plugins. |
| `E6` | `feedbax/contracts/graph.py:ParamValue`, `feedbax/contracts/graph.py:ComponentSpec`, `feedbax/contracts/graph.py:StudioValueSpec` | `tests/test_studio_value_spec_contract.py:test_component_params_normalize_typed_value_specs`, `tests/test_graphspec_schema_migrations.py:test_graph_spec_v2_migration_adds_derived_dimensions_field` | Literal values and existing value specs validate and migrate. Array-valued component params still lack the planned sparse, constant/broadcast, and content-pinned encodings and their identity contract. |
| `E7` | `feedbax/orchestration/bundle.py:ExecutionFamily`, `feedbax/orchestration/assembly.py:AssemblyCompilerRegistry`, `feedbax/orchestration/assembly.py:build_default_assembly_registry`, `feedbax/orchestration/executor_family.py:executor_family_adapter` | `tests/test_evaluation_orchestration.py:test_provider_free_cli_shadow_reaches_terminal_collection_in_fresh_process`, `tests/test_orchestration_core.py:test_stage_engine_resumes_from_every_stage_boundary` | Native training and evaluation matrices traverse the governed lifecycle. Compiler construction and lifecycle dispatch still enumerate the two core families. |
| `E8` | `feedbax/orchestration/drivers/base.py:OrchestrationDriver`, `feedbax/orchestration/bundle.py:DeploymentPolicy`, `feedbax/bin/orchestrate.py:_driver_for_bundle` | `tests/test_orchestration_cli.py:test_two_row_local_driver_demo_through_cli`, `tests/test_runpod_orchestration_driver.py:test_stage_engine_governs_runpod_provisioning` | Local and RunPod drivers implement the lifecycle, but policy, CLI choices, construction, and extra hooks are closed or duck-typed. |
| `E9` | `feedbax/contracts/artifact_custody.py:ImmutableArtifactBlobProviderSpec`, `feedbax/persistence/artifact_custody.py:ImmutableArtifactBlobProvider`, `feedbax/persistence/artifact_custody.py:open_immutable_artifact_blob_provider`, `feedbax/orchestration/staged_root_custody.py:StagedRootKind` | `tests/test_artifact_custody.py:test_portable_provider_spec_has_exact_root_free_json_and_round_trips`, `tests/test_artifact_custody.py:test_custody_survives_source_directory_deletion_and_materializes_copy` | The built-in immutable provider has portable schema, validation, storage, and recovery, but provider kinds and factories are fixed. |
| `E10` | `feedbax/contracts/schema_namespace.py:validate_schema_identity`, `feedbax/contracts/migrations.py:SpecSchemaFamily`, `feedbax/contracts/migrations.py:SpecSchemaRegistry` | `tests/test_structured_spec_migrations.py:test_structured_spec_registry_applies_registered_family_migration`, `tests/test_structured_spec_migrations.py:test_structured_spec_registry_reports_unknown_family` | Schema families have a public version/migration/rejection mechanism. The public authoring/provider projection is still separately curated. |
| `E11` | `feedbax/analysis/reports.py:register_report_recipe`, `feedbax/analysis/reports.py:execute_report_spec`, `feedbax/analysis/reports.py:OrderedFigureReportParams` | `tests/test_report_execution.py:test_report_spec_executes_registered_recipe_and_writes_markdown_render`, `tests/test_ordered_figure_report.py:test_ordered_figure_report_html_is_self_contained_interactive_and_deterministic` | Registered report recipes can emit governed artifacts; the built-in ordered report supports Markdown and self-contained HTML. Installed report-recipe bootstrap is not uniform. |
| `E12` | `feedbax/contracts/training.py:TrainingMethodDescriptor`, `feedbax/contracts/training.py:TrainingMethodRegistry`, `feedbax/training/authoring.py:compile_training_method_authoring`, `feedbax/training/row_lowering.py:TrainingRowLowererRegistry`, `feedbax/training/executor.py:execute_training_run_spec` | `tests/test_training_authoring.py:test_default_matrix_compiler_reaches_descriptor_authoring`, `tests/test_training_method_plugin_cli.py:test_entry_point_descriptor_derives_method_and_preparation_from_one_hook`, `tests/test_training_row_lowering.py:test_orchestration_cli_discovers_and_lowers_downstream_rows` | A downstream training method can register, author, lower, prepare, execute, report progress, checkpoint, and resume through the governed CLI path. |
| `E13` | `feedbax/studio/schema.py:enumerate_studio_schema_registry`, `feedbax/studio/schema.py:validate_graph_connection_schema`, `feedbax/web/api/analysis.py:_run_analysis_sync` | `tests/test_studio_api_contracts.py:test_component_api_serves_representation_contract`, `tests/test_studio_analysis_jobs.py:test_studio_analysis_job_routes_eval_run_through_executable_spec` | Studio renders registered component metadata and runs analysis contracts, but not every extension family is bootstrapped or projected into authoring UI. |
| `E14` | `feedbax/testing/suite.py:load_suite_manifest`, `feedbax/orchestration/conformance.py:CheckRegistry`, `feedbax/orchestration/conformance.py:build_default_check_registry` | `tests/test_run_conformance.py:test_plugin_check_discovery_and_failure_propagation`, `tests/test_component_registration.py:test_absent_downstream_owner_fails_with_actionable_message` | In-repo and downstream-kit checks exist, but there is no clean-installed external fixture proving each seam and supported version window. |
| `E15` | `feedbax/orchestration/state.py:RunSetStateStore`, `feedbax/orchestration/stages.py:StageEngine`, `feedbax/orchestration/collection_recovery.py:recover_collected_outputs` | `tests/test_orchestration_cli.py:test_watch_exits_after_all_rows_terminal`, `tests/test_orchestration_cli.py:test_collect_and_teardown_are_idempotent_after_completed_run` | The governed run lifecycle supplies progress, persisted state, collection recovery, and idempotent teardown for admitted families/drivers. |
| `E16` | `feedbax/analysis/specs.py:resolve_analysis_inputs`, `feedbax/analysis/specs.py:EvaluationStateMaterializationReceipt`, `feedbax/analysis/evaluation_rows.py:project_verified_evaluation_rows`, `feedbax/analysis/evaluation_rows.py:require_exact_authored_cartesian_coverage` | `tests/test_evaluation_row_projection.py:test_projects_all_resolver_source_kinds_with_truthful_receipts`, `tests/test_evaluation_row_projection.py:test_duck_typed_fake_receipt_is_rejected`, `tests/test_evaluation_row_projection.py:test_in_place_state_mutation_breaks_snapshotted_value_identity`, `tests/test_evaluation_row_projection.py:test_manifest_alias_mutation_projects_only_authenticated_raw_byte_values`, `external/feedbax_conformance_fixture/src/feedbax_external_conformance/cases.py:check_typed_evaluation_row_projection` | The exact resolver-issued capability snapshots existing canonical value identities for the state PyTree and complete typed durable, manifest-keyed-cache, or authenticated-recompute source. Projection recomputes those identities and derives manifest, run-spec, metadata, and producer provenance from authenticated raw bytes before one downstream-owned cross-field callback. Stable reason codes and exact authored-Cartesian deltas are public; scientific membership and verdict policy stay downstream. |
| `E17` | `feedbax/contracts/value_identity.py:ValueIdentityRecord`, `feedbax/contracts/value_identity.py:authored_value_sha256`, `feedbax/contracts/value_identity.py:semantic_value_sha256`, `feedbax/contracts/value_identity.py:realization_value_sha256` | `tests/test_value_identity.py:test_cross_encoding_semantic_equality_and_authored_inequality`, `tests/test_value_identity.py:test_expected_semantic_mismatch_fails_closed_and_chain_is_preserved`, `tests/test_value_identity.py:test_realization_uses_only_explicit_fingerprints` | The public v1 contract separates authored, exact normalized semantic, and explicit runtime realization identity. Existing durable envelopes do not embed it yet; their owning consumers retain their migration responsibility. |
| `E18` | `feedbax/contracts/material_dependencies.py:MaterialDependencySet`, `feedbax/contracts/material_dependencies.py:validate_material_dependency_admission`, `feedbax/contracts/manifest.py:TrainingRunCertification`, `feedbax/analysis/bundles.py:_preflight_staged_exact_parents`, `feedbax/testing/material_dependencies.py:check_material_dependency_contract` | `tests/test_material_dependencies.py:test_identity_projects_refs_and_values_to_material_identity`, `tests/test_staged_exact_parents.py:test_diverged_run_admits_certified_checkpoint_and_scopes_evaluation_identity`, `tests/test_staged_exact_parents.py:test_material_dependency_admission_rejects_before_outputs`, `tests/test_report_execution.py:test_authored_report_rejects_unhandled_material_dependencies_before_outputs` | Versioned declarations scope identity to material content or semantic value identity, authenticate certified dependency bytes through existing custody/provider authority, and permit only an exact authored incidental-check waiver. Ambiguous legacy failed manifests reject; authored reports fail closed unless parents first pass the shared bundle preflight. |

## Coverage matrix

### Bootstrap through lowering

| Capability | `I` | `A` | `M` | `D` | `V` | `L` |
|---|---|---|---|---|---|---|
| Component | `P E1 / 301dce2` | `P E1 / 9ac21f2` | `O E1` | `P E1 / 301dce2` | `P E1 / 9ac21f2` | `O E1` |
| Task | `C D-TASK` | `P E3 / D-TASK` | `O E10` | `C D-TASK` | `P E3 / D-TASK` | `C D-TASK` |
| Analysis recipe | `P E2 / 301dce2` | `O E4` | `O E10` | `P E4 / 301dce2` | `P E4 / 46aeab1` | `O E4` |
| Figure | `C 301dce2` | `O E5` | `O E10` | `P E5 / 301dce2` | `O E5` | `O E5` |
| Value encoding | `—` | `P E6 / 9757814` | `P E6, E17 / 9757814` | `C c50193e` | `P E6 / 9757814` | `P E6 / 9757814` |
| Run kind | `C D-RUN` | `P E7 / D-RUN` | `P E7 / D-RUN` | `C D-RUN` | `C D-RUN` | `P E7 / D-RUN` |
| Driver | `C 301dce2, 69034e6` | `P E8 / 69034e6` | `P E8 / 69034e6` | `C 69034e6` | `C 69034e6` | `C 69034e6` |
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
| Figure | `O E5` | `O E16` | `P E5 / 301dce2` | `—` | `O E5` |
| Value encoding | `P E6 / 9757814` | `P E17, E18 / 9757814` | `P E6 / 9757814` | `—` | `P E6 / 9757814` |
| Run kind | `P E7 / D-RUN` | `P E7 / D-RUN` | `P E7 / D-RUN` | `P E15 / D-RUN` | `P E7 / D-RUN` |
| Driver | `P E8 / 69034e6` | `P E8 / 69034e6` | `P E8 / 69034e6` | `P E15 / 69034e6` | `P E8 / 69034e6` |
| Custody provider | `P E9 / D-CUSTODY` | `P E9 / D-CUSTODY` | `P E9 / D-CUSTODY` | `—` | `O E9` |
| Schema family | `O E10` | `O E10` | `—` | `—` | `O E10` |
| Report format | `O E11` | `O E16` | `P E11 / 301dce2` | `—` | `O E11` |
| Training method | `O E12` | `O E16` | `O E12` | `O E12` | `O E12` |

### Recovery through conformance

| Capability | `U` | `Q` | `S` | `T` |
|---|---|---|---|---|
| Component | `—` | `—` | `P E13 / 9ac21f2, 301dce2` | `P E14 / f4476ae, 380f897` |
| Task | `—` | `—` | `P E3 / D-TASK` | `C D-TASK` |
| Analysis recipe | `—` | `O E4` | `P E13 / 301dce2` | `P E14, E16 / f4476ae` |
| Figure | `—` | `O E5` | `P E13 / 301dce2` | `P E14 / f4476ae, 380f897` |
| Value encoding | `—` | `—` | `P E6 / 9757814` | `C f4476ae, 380f897` |
| Run kind | `P E15 / D-RUN` | `P E7 / D-RUN` | `C D-RUN` | `C D-RUN` |
| Driver | `P E15 / 69034e6` | `P E15 / 69034e6` | `C 69034e6` | `C 69034e6, 380f897` |
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
- `69034e6`: orchestration-driver capability contract and registry.
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
