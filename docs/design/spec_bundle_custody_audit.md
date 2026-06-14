# Spec Bundle Schema and Custody Final Audit

Umbrella: `588483d` (`Spec-based analysis bundles and artifact pipeline generalization`)
Final audit issue: `e8662b2`
Date: 2026-06-14

This audit was run after the Feedbax primitive branch landed on `develop` and after the
RLRMP integration branch `integration/588483d-rlrmp-spec-bundles` integrated the consumer
lanes through commit `b7c5a59`.

## Execution Note

The assigned Codex worker thread reached ledger setup and branch creation, then the Codex
app reported `systemError`. A replacement worker thread failed the same way before agent
work began. The audit was completed from the parent thread in the existing audit worktree
so the umbrella could reach protected-branch auth readiness.

## Feedbax Coverage

| Surface | Custody result | Evidence |
|---|---|---|
| Schema namespace and structured specs | Covered by `SpecSchemaRegistry`, `SpecSchemaFamily`, namespace validation, family policy rows, explicit migration edges, and explicit old-version rejection diagnostics. | `feedbax/migrations.py`; `tests/test_manifest_spec_payloads.py`; `tests/test_provider_contract.py`; `tests/test_analysis_registration_validation.py` |
| Graph specs and component parameter payloads | Covered by graph schema identity, recursive migration records, component migration packs, owner-qualified component IDs, and governed nested parameter payloads. | `feedbax/contracts/graph.py`; `feedbax/migrations.py`; `tests/test_graphspec_schema_migrations.py`; `tests/test_parameter_constraints.py` |
| Reusable nested `PopulationStructure` payloads | Covered as a governed nested schema with explicit identity/version rejection for unsupported or wrong identities. | `feedbax/dimred.py`; `tests/test_parameter_constraints.py` |
| Regeneration and replay records | Covered by Feedbax-owned `RegenerationSpec`, command/parameter/source/input/output provenance, file/tree hashing, manifest embedding, and explicit old-version rejection. | `feedbax/manifest.py`; `tests/test_regeneration_spec.py` |
| Analysis materialization and artifact custody | Covered by `ContextMaterializer`, opaque downstream payload boundaries, recursive artifact-ref recording, existing artifact adoption, artifact groups, and regeneration-spec recording. | `feedbax/analysis/materialization.py`; `feedbax/analysis/context.py`; `tests/test_analysis_context.py` |
| Analysis bundle plans | Covered by schema-bearing `AnalysisBundleSpec` v2, staged evaluation/analysis/materialization/report plans, lineage through stage dependencies, optional output statuses, regeneration records, and explicit rejection of unsupported v1 payloads. | `feedbax/analysis/bundles.py`; `tests/test_analysis_spec_bundles.py` |
| Checkpoint selection records | Covered by Feedbax-owned `CheckpointSelectionSpec` and `CheckpointSelectionManifest`, scorer identity, candidate refs, bank availability/missing status, selected refs, lineage normalization, and old-version rejection. | `feedbax/manifest.py`; `tests/test_checkpoint_selection_manifest.py` |
| Training retention artifacts | Covered by retention plan/artifact schema identities and the Feedbax structured-spec registry policy introduced under the umbrella. | `feedbax/retention_artifact_schema.py`; `feedbax/training/train.py`; `tests/test_retained_observables.py` |
| Provider-visible schemas and manifest mappings | Covered by provider manifest schema exports, neutral contract names, eval/analysis/report action depth, manifest mapping contracts, and validation/migration status reporting. | `feedbax/provider.py`; `tests/test_provider_contract.py` |
| Role-addressed array artifacts | Covered as Feedbax-owned manifest schemas for array stores and semantic role addresses, with deterministic role validation and digest checks. | `feedbax/artifact_schema.py`; `tests/test_artifact_materialize.py`; `tests/test_provider_contract.py` |

Feedbax now owns the reusable mechanics this umbrella targeted: structured schema
identity, migration or intentional rejection policy, regeneration custody, context-bound
materialization, artifact grouping, output-status recording, staged bundle execution,
checkpoint-selection manifests, provider/action manifests, and retention artifact schemas.

## RLRMP Coverage

| Active RLRMP surface | Custody result | Evidence |
|---|---|---|
| RLRMP scientific sidecar schema families | Covered downstream by `rlrmp.spec_migrations`, layered on Feedbax structured-spec policy. Current versions are accepted; historical active versions are explicitly rejected unless a deterministic migration is added. | `src/rlrmp/spec_migrations.py`; `tests/test_rlrmp_spec_migrations.py` |
| Generic materialization and artifact custody | Covered through Feedbax `ContextMaterializer` and `AnalysisRunManifest` artifact records; RLRMP recipes retain only scientific payload construction. | `src/rlrmp/analysis/declarative_materialization.py`; `tests/analysis/test_declarative_materialization.py` |
| GRU post-run diagnostics | Covered as a Feedbax bundle/resource and declarative materialization recipe with RLRMP-owned payload semantics. | `src/rlrmp/config/analysis_bundles/gru_postrun.yml`; `tests/analysis/pipelines/test_gru_postrun_materialization.py`; `tests/analysis/test_declarative_materialization.py` |
| Feedback-control quality diagnostics | Covered as a reusable bundle/lens with Feedbax artifact custody, grouped bulk artifacts, skipped/unavailable/not_applicable statuses, and an RLRMP-owned lens schema. | `src/rlrmp/config/analysis_bundles/feedback_quality_lens.yml`; `tests/analysis/test_declarative_materialization.py`; `tests/test_rlrmp_spec_migrations.py` |
| Robustness phenotype sidecars | Covered as a bundle/materializer output with lower-layer role inventory and an RLRMP-owned phenotype schema. | `src/rlrmp/analysis/declarative_materialization.py`; `tests/analysis/test_robustness_phenotype_bundle.py`; `tests/analysis/pipelines/test_hinf_phenotype_sidecar.py` |
| Output-feedback bridge diagnostics | Covered by declarative specs and Feedbax-backed output-feedback bridge bundle plumbing while preserving bridge-specific certificate semantics downstream. | `src/rlrmp/config/analysis_bundles/output_feedback_bridge.yml`; `tests/analysis/pipelines/test_output_feedback_rollout_recovery.py`; `tests/test_output_feedback_materializer_adapters.py` |
| Training diagnostics | Covered as Feedbax bundle contracts and registered RLRMP analysis recipes without moving training-diagnostic interpretation into Feedbax. | `src/rlrmp/analysis/training_diagnostics.py`; `tests/analysis/test_training_diagnostics_bundle.py`; `tests/test_monitor_training_diagnostics.py` |
| Standard matrix / certificate materialization | Covered by RLRMP-owned scientific schemas and Feedbax manifest/artifact custody; generic bundle execution remains Feedbax-owned. | `src/rlrmp/analysis/matrix/standard_matrix.py`; `tests/analysis/test_standard_matrix_bundle.py`; `tests/test_rlrmp_spec_migrations.py` |
| Legacy training configs and historical JSON | Intentionally archive-only. They are not promoted into active Feedbax custody because they predate durable run-spec provenance and are retained for provenance, not recurring execution. | `src/rlrmp/spec_migrations.py`; `tests/test_rlrmp_spec_migrations.py` |

RLRMP now keeps project-owned scientific meaning downstream: bridge certificate
interpretation, C&S perturbation taxonomy, objective-comparator semantics, robustness
phenotype interpretation, feedback-quality lens semantics, and historical/archive run
records. Generic replay, materialization, output status, grouping, bundle execution,
schema-policy, and manifest custody live in Feedbax.

## Remaining Findings

No new blocking implementation issue was found in the active recurring surfaces covered by
this umbrella. The open RLRMP child issues remain lifecycle-open only because their
integration branch still needs protected-branch auth and closure staging; their worker
status is done and their commits are integrated on
`integration/588483d-rlrmp-spec-bundles`.

The only anomaly observed by this audit is operational: two Codex worker threads for
`e8662b2` failed with app-level `systemError` before audit completion. That is recorded in
the issue closeout as a delegation/process anomaly, not as a Feedbax/RLRMP custody blocker.

## Verification

The focused verification for this audit should cover both repos:

- Feedbax: `uv run --no-sync pytest tests/test_analysis_spec_bundles.py tests/test_analysis_context.py tests/test_regeneration_spec.py tests/test_checkpoint_selection_manifest.py tests/test_manifest_spec_payloads.py tests/test_parameter_constraints.py tests/test_provider_contract.py tests/test_artifact_materialize.py tests/test_retained_observables.py -q`
- RLRMP: `uv run --no-sync pytest tests/analysis/test_declarative_materialization.py tests/analysis/test_robustness_phenotype_bundle.py tests/analysis/pipelines/test_hinf_phenotype_sidecar.py tests/analysis/pipelines/test_output_feedback_rollout_recovery.py tests/test_output_feedback_materializer_adapters.py tests/analysis/test_standard_matrix_bundle.py tests/analysis/pipelines/test_gru_postrun_materialization.py tests/analysis/test_training_diagnostics_bundle.py tests/test_monitor_training_diagnostics.py tests/test_rlrmp_spec_migrations.py -q`
- RLRMP lint: `uv run --no-sync ruff check src/rlrmp/__init__.py src/rlrmp/spec_migrations.py src/rlrmp/analysis/declarative_materialization.py src/rlrmp/analysis/training_diagnostics.py tests/analysis/test_declarative_materialization.py tests/analysis/test_robustness_phenotype_bundle.py tests/test_rlrmp_spec_migrations.py`
- Both repos: `git diff --check`
