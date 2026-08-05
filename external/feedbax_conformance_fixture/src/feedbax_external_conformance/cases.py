"""Fixture-owned builders and clean-installed public contract cases."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import jax
import jax.numpy as jnp
from pydantic import ValidationError

from feedbax import LowererRegistration, OrderedLowererRegistry, init_state_from_component
from feedbax.analysis import (
    EvaluationRowProjectionError,
    EvaluationRowProjectionErrorCode,
    ResolvedManifestInput,
    authenticated_manifest_ref,
    execute_analysis_run_spec,
    project_evaluation_rows,
    resolve_analysis_inputs,
)
from feedbax.analysis.bundles import (
    AnalysisBundleSpec,
    BundleStageSpec,
    execute_staged_analysis_bundle,
)
from feedbax.analysis.figures import coerce_figure_spec, resolve_figure_spec
from feedbax.analysis.evaluation import (
    EvaluationBatchExecution,
    EvaluationRunMatrixSpec,
    compile_evaluation_run_matrix,
    execute_evaluation_run_matrix,
    execute_evaluation_run_spec,
)
from feedbax.analysis.evaluation_compaction import (
    EvaluationBatchConsumerInput,
    compact_evaluation_batch,
    merge_evaluation_batch_fragment,
    publish_evaluation_compaction_products,
    reclaim_evaluation_batch_caches,
)
from feedbax.analysis.evaluation_product_union import (
    EvaluationCompactProductUnionBinding,
    finalize_evaluation_compact_product_union,
)
from feedbax.analysis.execution_context import EMPTY_STAGED_EXECUTION_CONTEXT
from feedbax.analysis.exact_parents import (
    STAGED_EXACT_PARENTS_SCHEMA_ID,
    STAGED_EXACT_PARENTS_SCHEMA_VERSION,
    STAGED_EXACT_PARENTS_SCHEMA_VERSION_V1,
    StagedExactParentEntry,
    StagedExactParents,
    migrate_staged_exact_parents,
)
from feedbax.component_registry import (
    ComponentMigration,
    ComponentMigrationPack,
    ComponentRegistry,
)
from feedbax.contracts import (
    ARRAY_VALUE_SCHEMA_ID,
    ARRAY_VALUE_SCHEMA_VERSION,
    MATERIAL_DEPENDENCIES_SCHEMA_ID,
    MATERIAL_DEPENDENCIES_SCHEMA_VERSION,
    MaterialDependency,
    MaterialDependencyObservation,
    MaterialDependencySet,
    ComponentSpec,
    ConstantArrayValueSpec,
    GraphSpec,
    SparseCooArrayValueSpec,
    SparseCooEntrySpec,
    ValueIdentityRecord,
    authored_value_sha256,
    materialize_array_value,
    semantic_value_sha256,
    value_identity_record,
)
from feedbax.contracts.figures import (
    FigureCompositionProvenance,
    FigureCompositionSourceRecord,
    FigureCompositionSpec,
    FigureRuntimeBindingSpec,
    FigureSpec,
    PanelSpec,
    ResolvedFigureSpec,
    TraceBinding,
    TraceFamily,
    TraceFamilyIndex,
)
from feedbax.contracts.figure_roles import (
    FigureRoleBindingContract,
    FigureRowExpansionRequest,
    expand_figure_rows,
    resolve_figure_input_roles,
)
from feedbax.contracts.row_index import (
    AuthenticatedRowIndex,
    RowIndexCustodyBindings,
    RowSelectionError,
    RowSelectionErrorCode,
    expand_row_selector,
)
from feedbax.contracts.graphs.serialization import graph_to_spec, spec_to_graph
from feedbax.contracts.graphs.normalization import normalize_graph_for_studio_authoring
from feedbax.contracts.manifest import (
    AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
    AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
    AnalysisRunSpec,
    EntrypointRef,
    EvaluationRunManifest,
    EvaluationRunSpec,
    ParentRef,
    Provenance,
    SpecPayload,
    canonical_json_bytes,
    load_manifest,
    sha256_bytes,
    write_manifest,
    OverridePatch,
)
from feedbax.contracts.matrix_core import (
    SOURCE_DOCUMENT_INHERITANCE_KEY,
    ContentPinnedJsonBase,
    SourceDocumentInheritance,
    load_content_pinned_json_base,
    materialize_inherited_document,
)
from feedbax.contracts.evaluation_lifecycle import (
    EVALUATION_BATCH_MERGE_CHECKPOINT_SCHEMA_ID,
    EVALUATION_BATCH_MERGE_CHECKPOINT_SCHEMA_VERSION,
    EvaluationBatchCompactionEvidence,
    EvaluationBatchConsumerDeclaration,
    EvaluationLifecycleRowOutcome,
    EvaluationMatrixBatchUnit,
)
from feedbax.contracts.evaluation_product_union import (
    EvaluationCompactProductUnion,
    EvaluationCompactProductUnionSource,
)
from feedbax.contracts.run_matrix import (
    AuthoredTrainingRow,
    MatrixCompositionDelta,
    TRAINING_ROW_LOWERER_REF_FIELD,
    TrainingRowLowererRef,
    apply_composition_deltas,
)
from feedbax.contracts.spec_storage import training_spec_sha256
from feedbax.contracts.training import MethodPayloadEnvelope, MethodRefSpec
from feedbax.training.authoring import compile_training_method_authoring
from feedbax.training.preparation import ExecutionPreparationRequest
from feedbax.training.row_lowering import TrainingRowLoweringContext
from feedbax.contracts.evaluation_states import store_evaluation_states_artifact
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.plugins import (
    COMPONENTS,
    DRIVERS,
    ANALYSIS_RECIPES,
    EVALUATION_BATCH_CONSUMERS,
    EVALUATION_PRODUCT_UNION_FINALIZERS,
    EVALUATION_RECIPES,
    EXECUTION_PREPARATIONS,
    ROW_LOWERERS,
    TRAINING_METHODS,
    BootstrapError,
    BootstrapErrorCode,
    FamilyRequirement,
    PluginDeclaration,
    PluginDependency,
    PluginRegistration,
    RegistrationContext,
    RegistryFamilyRegistration,
    bootstrap_application,
    discover_plugin_registrations,
    new_registration_context,
)
from feedbax.plugins.composition import compose_application
from feedbax.orchestration.drivers import (
    DriverConstructionContext,
    ResourceSemantics,
    TeardownSemantics,
)
from feedbax.testing import check_material_dependency_contract

from .family import (
    EXTERNAL_DYNAMIC_COMPONENT,
    FIXTURE_RECORDS,
    FixtureRecordRegistry,
)


_CUSTOM_COMPONENT = "fixture.CurrentScale"
_LEGACY_COMPONENT = "fixture.LegacyScale"


@dataclass(frozen=True)
class _LoweringContext:
    enabled: frozenset[str]


def _lowerer(name: str, order: int) -> LowererRegistration[_LoweringContext, str]:
    return LowererRegistration(
        lowerer_id=name,
        order=order,
        owner=f"feedbax-external-conformance.{name}",
        lowerer=lambda context: name if name in context.enabled else None,
    )


def check_ordered_registration() -> bool:
    """Prove shuffled registration is deterministic and duplicates fail closed."""
    registrations = (
        _lowerer("last", 20),
        _lowerer("beta", 10),
        _lowerer("alpha", 10),
    )
    expected = ("alpha", "beta", "last")
    for shuffled in (registrations, tuple(reversed(registrations))):
        registry = OrderedLowererRegistry[_LoweringContext, str](shuffled)
        if registry.available_ids() != expected:
            raise AssertionError("ordered lowerer result depends on registration order")
        lowered = registry.lower(_LoweringContext(enabled=frozenset(expected)))
        if tuple(item.fragment for item in lowered) != expected:
            raise AssertionError("ordered lowerer execution drifted")
    duplicate = OrderedLowererRegistry[_LoweringContext, str]([registrations[0]])
    try:
        duplicate.register(_lowerer("last", 0))
    except ValueError as exc:
        if "already registered" not in str(exc):
            raise
    else:
        raise AssertionError("duplicate ordered lowerer registration was accepted")
    return True


def _fixture_plugin(
    plugin_id: str,
    register,
    *,
    dependencies: tuple[PluginDependency, ...] = (),
    registry_families: tuple[RegistryFamilyRegistration[object], ...] = (),
) -> PluginRegistration:
    return PluginRegistration(
        declaration=PluginDeclaration(
            plugin_id=plugin_id,
            version="1",
            downstream_protocol_version=1,
            dependencies=dependencies,
            families=(FamilyRequirement(FIXTURE_RECORDS.family),),
        ),
        register=register,
        registry_families=registry_families,
    )


class _RegistrationEntryPoint:
    dist = None

    def __init__(self, registration: PluginRegistration) -> None:
        self.name = registration.declaration.plugin_id
        self.value = f"{self.name}:PLUGIN_REGISTRATION"
        self._registration = registration

    def load(self) -> PluginRegistration:
        return self._registration


def check_unified_plugin_bootstrap(*, entry_points: Iterable[object] | None = None) -> bool:
    """Prove installed typed discovery and the transactional generic-family contract."""

    expected_plugins = (
        "feedbax_external_conformance.foundation",
        "feedbax_external_conformance.dependent",
    )
    state = asyncio.run(
        compose_application(
            entry_points=entry_points,
            local_component_source=None,
        )
    )
    if tuple(sorted(item.plugin_id for item in state.provenance)) != tuple(
        sorted(expected_plugins)
    ):
        raise AssertionError("installed feedbax.plugins discovery inventory drifted")
    temporary_parent = "/private/tmp" if Path("/private/tmp").is_dir() else None
    with TemporaryDirectory(dir=temporary_parent) as temporary:
        root = Path(temporary)
        registry = state.registry(FIXTURE_RECORDS)
        # Static availability is intentionally shallow here. The installed
        # package lifecycle below proves the callback behavior through public
        # consumers rather than by calling resolved callbacks directly.
        if (
            state.registry(TRAINING_METHODS).descriptor("feedbax_external_conformance/training/v1")
            is None
        ):
            raise AssertionError("external training descriptor was not registered")
        if not state.registry(ROW_LOWERERS).available_keys():
            raise AssertionError("external row lowerer was not registered")
        if (
            state.registry(EXECUTION_PREPARATIONS).get("feedbax_external_conformance/training/v1")
            is None
        ):
            raise AssertionError("external execution preparation was not registered")
        if not callable(
            state.registry(ANALYSIS_RECIPES).get("feedbax_external_conformance.analysis")
        ):
            raise AssertionError("external analysis recipe was not resolved")
        if not callable(
            state.registry(EVALUATION_RECIPES).get("feedbax_external_conformance.evaluation")
        ):
            raise AssertionError("external evaluation recipe was not resolved")
        if not callable(
            state.registry(EVALUATION_RECIPES).batch("feedbax_external_conformance.evaluation")
        ):
            raise AssertionError("external evaluation batch recipe was not resolved")
        if (
            state.registry(EVALUATION_BATCH_CONSUMERS)
            .get("feedbax_external_conformance.consumer", "v1")
            .compact
            is None
        ):
            raise AssertionError("external batch consumer was not resolved")
        if not callable(
            state.registry(EVALUATION_PRODUCT_UNION_FINALIZERS).get(
                "feedbax_external_conformance.consumer", "v1"
            )
        ):
            raise AssertionError("external product-union finalizer was not resolved")
        method_ref = MethodRefSpec(
            package="feedbax_external_conformance", name="training", version="v1"
        )
        payload = MethodPayloadEnvelope(
            schema_id="feedbax_external_conformance.training",
            schema_version="feedbax_external_conformance.training.v1",
            payload={"gain": 3},
        )
        resolved = state.registry(TRAINING_METHODS).resolve_execution(method_ref, payload)
        if resolved.contract.method_ref != "feedbax_external_conformance/training/v1":
            raise AssertionError("external training resolution lost its method authority")
        authored_payload = {"gain": 3}
        row = AuthoredTrainingRow(
            row_id="fixture",
            row_index=0,
            payload=authored_payload,
            payload_hash=training_spec_sha256(authored_payload),
            axis_coordinates={},
        )
        compiled = compile_training_method_authoring(
            row, method_ref=method_ref, registry=state.registry(TRAINING_METHODS)
        )
        if compiled.run_spec.metadata != {"fixture_gain": 3}:
            raise AssertionError("external training authoring did not invoke its typed hook")
        lowerer = state.registry(ROW_LOWERERS)
        from .plugin import FIXTURE_LOWERER_IMPLEMENTATION_SHA256

        registration = next(iter(lowerer.available_keys()))
        lowerer_payload = {
            "gain": 4,
            "schema_id": registration[0],
            "schema_version": registration[1],
            TRAINING_ROW_LOWERER_REF_FIELD: TrainingRowLowererRef(
                lowerer_id=registration[2],
                lowerer_version=registration[3],
                implementation_sha256=FIXTURE_LOWERER_IMPLEMENTATION_SHA256,
            ).model_dump(mode="json"),
        }
        lowered = lowerer.lower(
            AuthoredTrainingRow(
                row_id="lower",
                row_index=0,
                payload=lowerer_payload,
                payload_hash=training_spec_sha256(lowerer_payload),
                axis_coordinates={},
            ),
            TrainingRowLoweringContext(),
        )
        if lowered is None or lowered.execution_payload != {"fixture_lowered_gain": 4}:
            raise AssertionError("external row lowerer was not invoked")
        prepared = state.registry(EXECUTION_PREPARATIONS).prepare(
            ExecutionPreparationRequest(
                run_spec=compiled.run_spec,
                method_payload=resolved.payload,
                method_contract=resolved.contract,
                effective_phase=resolved.effective_phase,
            )
        )
        if prepared.kernel_context != {"fixture": True}:
            raise AssertionError("external execution preparation was not invoked")

        authored_evaluation = compile_evaluation_run_matrix(
            {
                "base": {
                    "ref": "fixture_evaluation_base.json",
                    "sha256": "f65f9ae128d0b8361e5064b729c9078dec516d5cf7ca47e2aa65eab9c71a7195",
                },
                "axes": [{"id": "fixture", "values": [{"id": "one", "deltas": []}]}],
            },
            repo_root=Path(__file__).parent,
            registry=state.registry(EVALUATION_RECIPES),
        )
        if authored_evaluation.base.params != {"gain": 3} or [
            row.row_id for row in authored_evaluation.rows
        ] != ["fixture-one"]:
            raise AssertionError("external evaluation authoring schema was not consumed")

        scalar, scalar_path = execute_evaluation_run_spec(
            EvaluationRunSpec(
                evaluation_type="feedbax_external_conformance.evaluation",
                params={"gain": 3, "states_custody": "durable"},
            ),
            registry=state.registry(EVALUATION_RECIPES),
            root=root / "scalar",
        )
        if scalar.status != "completed" or load_manifest(scalar_path) != scalar:
            raise AssertionError("public scalar evaluation did not publish its manifest")

        matrix = EvaluationRunMatrixSpec.model_validate(
            {
                "base": {
                    "ref": "fixture_evaluation_base.json",
                    "sha256": "f65f9ae128d0b8361e5064b729c9078dec516d5cf7ca47e2aa65eab9c71a7195",
                },
                "axes": [
                    {
                        "id": "fixture",
                        "values": [
                            {
                                "id": cohort,
                                "deltas": [
                                    {"path": "params.gain", "value": index + 1},
                                    {
                                        "path": "params.states_custody",
                                        "value": "durable",
                                        "op": "add",
                                    },
                                ],
                            }
                            for index, cohort in enumerate(("left", "right"))
                        ],
                    }
                ],
            }
        )
        batch_root = root / "batch"
        executed = execute_evaluation_run_matrix(
            matrix,
            root=batch_root,
            repo_root=Path(__file__).parent,
            batch=EvaluationBatchExecution(),
            registry=state.registry(EVALUATION_RECIPES),
        )
        if tuple(row.row_id for row in executed.rows) != ("fixture-left", "fixture-right"):
            raise AssertionError("public batch evaluation lost authored row order")

        authority = authenticated_manifest_ref(
            executed.rows[0].result,
            executed.rows[0].manifest_path,
            "evaluation_run",
        )
        analysis_spec = AnalysisRunSpec(
            analysis_type="feedbax_external_conformance.analysis",
            inputs=[authority],
            evaluation_states_policy="require_durable",
            params={"requested_outputs": ["fixture"]},
        )
        resolved_inputs = resolve_analysis_inputs(
            analysis_spec,
            root=batch_root,
            evaluation_registry=state.registry(EVALUATION_RECIPES),
            registry=state.registry(ANALYSIS_RECIPES),
        )
        if resolved_inputs[0].states.value.tolist() != [1, 2]:
            raise AssertionError("typed durable evaluation states did not round-trip")

        analysis_root = root / "analysis"
        relocated_states = store_evaluation_states_artifact(
            resolved_inputs[0].states,
            root=analysis_root,
            manifest_id=executed.rows[0].result.id,
        )
        original_states = next(
            artifact
            for artifact in executed.rows[0].result.artifacts
            if artifact.role == "evaluation_states"
        )
        if relocated_states.sha256 != original_states.sha256:
            raise AssertionError("public state custody relocation changed authenticated bytes")
        relocated_manifest_path = write_manifest(
            executed.rows[0].result,
            root=analysis_root,
            index=False,
        )
        analysis_authority = authenticated_manifest_ref(
            executed.rows[0].result,
            relocated_manifest_path,
            "evaluation_run",
        )
        analysis_manifest, analysis_path = execute_analysis_run_spec(
            analysis_spec.model_copy(update={"inputs": [analysis_authority]}),
            registry=state.registry(ANALYSIS_RECIPES),
            evaluation_registry=state.registry(EVALUATION_RECIPES),
            experiment_registry=state.bundle.experiment_packages,
            root=analysis_root,
            fig_dump_formats=(),
        )
        if analysis_manifest.status != "completed" or load_manifest(analysis_path) != (
            analysis_manifest
        ):
            raise AssertionError("public analysis execution did not publish its manifest")

        declaration = EvaluationBatchConsumerDeclaration(
            leaf_id="fixture",
            consumer_id="feedbax_external_conformance.consumer",
            consumer_version="v1",
            terminal_analysis_type="feedbax_external_conformance.analysis",
            accepted_evaluation_state_schema_ids=("feedbax_external_conformance.states.v1",),
            compact_product_schema_id="feedbax_external_conformance.batch",
            compact_product_schema_version="feedbax_external_conformance.batch.v1",
            compact_product_role="compact",
            merge_state_schema_id="feedbax_external_conformance.merge",
            merge_state_schema_version="feedbax_external_conformance.merge.v1",
        )
        union_sources = []
        union_bindings = []
        for cohort_key, executed_row in zip(("left", "right"), executed.rows, strict=True):
            outcome = EvaluationLifecycleRowOutcome(
                row_id=executed_row.row_id,
                manifest_id=executed_row.result.id,
                manifest_path=str(executed_row.manifest_path),
                diagnostic_schema_ids=(executed_row.result.metadata["states_schema"],),
            )
            cohort_batch = EvaluationMatrixBatchUnit(
                batch_id=f"{cohort_key}-batch",
                ordered_row_ids=(executed_row.row_id,),
                required_leaf_ids=(declaration.leaf_id,),
            )
            cohort_hash = sha256_bytes(
                canonical_json_bytes(
                    {
                        "matrix": matrix.model_dump(mode="json"),
                        "cohort_row_id": executed_row.row_id,
                    }
                )
            )
            cohort_authority = authenticated_manifest_ref(
                executed_row.result,
                executed_row.manifest_path,
                "evaluation_run",
            )
            cohort_root = batch_root / "cohorts" / cohort_key
            custody_root = cohort_root / "custody"
            cohort_states = resolve_analysis_inputs(
                AnalysisRunSpec(
                    analysis_type="feedbax_external_conformance.analysis",
                    inputs=[cohort_authority],
                    evaluation_states_policy="require_durable",
                ),
                root=batch_root,
                evaluation_registry=state.registry(EVALUATION_RECIPES),
                registry=state.registry(ANALYSIS_RECIPES),
            )[0].states
            fragment = compact_evaluation_batch(
                declaration,
                EvaluationBatchConsumerInput(
                    matrix_intent_hash=cohort_hash,
                    batch=cohort_batch,
                    outcomes=(outcome,),
                    manifests=(executed_row.result.model_dump(mode="json"),),
                    states=(cohort_states,),
                    parent_authorities=(cohort_authority,),
                    parameters=declaration.parameters,
                    execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
                ),
                registry=state.registry(EVALUATION_BATCH_CONSUMERS),
                custody_root=custody_root,
            )
            acknowledgement = merge_evaluation_batch_fragment(
                declaration,
                registry=state.registry(EVALUATION_BATCH_CONSUMERS),
                matrix_intent_hash=cohort_hash,
                batch=cohort_batch,
                parent_authorities=(cohort_authority,),
                fragment=fragment,
                prior_merge_state=None,
                custody_root=custody_root,
                execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
            )
            reclamation = reclaim_evaluation_batch_caches(
                cohort_batch,
                registry=state.registry(EVALUATION_BATCH_CONSUMERS),
                matrix_intent_hash=cohort_hash,
                batch_index=0,
                outcomes=(outcome,),
                acknowledgements=(acknowledgement,),
                required_declarations=(declaration,),
                custody_root=custody_root,
                execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
            )
            (terminal_manifest,) = publish_evaluation_compaction_products(
                (declaration,),
                {declaration.leaf_id: acknowledgement.merge_state},
                (outcome,),
                registry=state.registry(EVALUATION_BATCH_CONSUMERS),
                custody_root=custody_root,
                execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
            )
            compaction = EvaluationBatchCompactionEvidence(
                matrix_intent_hash=cohort_hash,
                ordered_batch_ids=(cohort_batch.batch_id,),
                declared_leaf_ids=(declaration.leaf_id,),
                required_leaf_ids_by_batch={cohort_batch.batch_id: (declaration.leaf_id,)},
                reclamations=(reclamation,),
                terminal_products=(terminal_manifest,),
            )
            compaction_path = cohort_root / "evaluation-batch-compaction.json"
            compaction_path.parent.mkdir(parents=True, exist_ok=True)
            compaction_path.write_bytes(canonical_json_bytes(compaction.model_dump(mode="json")))
            checkpoint_path = (
                custody_root
                / "merge-checkpoints"
                / declaration.leaf_id
                / f"{cohort_batch.batch_id}.json"
            )
            terminal_manifest_value = load_manifest(
                ImmutableArtifactBlobProvider(custody_root).materialize(
                    terminal_manifest, cohort_root / "terminal-manifest.json"
                )
            )
            terminal_product = terminal_manifest_value.produced_data[0].artifacts[0]
            union_sources.append(
                EvaluationCompactProductUnionSource(
                    cohort_key=cohort_key,
                    matrix_intent_hash=cohort_hash,
                    consumer_id=declaration.consumer_id,
                    consumer_version=declaration.consumer_version,
                    leaf_id=declaration.leaf_id,
                    compact_product_schema_id=declaration.compact_product_schema_id,
                    compact_product_schema_version=declaration.compact_product_schema_version,
                    compact_product_role=declaration.compact_product_role,
                    ordered_row_ids=(executed_row.row_id,),
                    compaction_evidence_sha256=sha256_bytes(compaction_path.read_bytes()),
                    terminal_checkpoint_schema_id=EVALUATION_BATCH_MERGE_CHECKPOINT_SCHEMA_ID,
                    terminal_checkpoint_schema_version=(
                        EVALUATION_BATCH_MERGE_CHECKPOINT_SCHEMA_VERSION
                    ),
                    terminal_checkpoint_sha256=sha256_bytes(checkpoint_path.read_bytes()),
                    terminal_manifest_sha256=terminal_manifest.sha256,
                    terminal_product_sha256=terminal_product.sha256,
                )
            )
            union_bindings.append(
                EvaluationCompactProductUnionBinding(
                    cohort_key=cohort_key,
                    custody_root=custody_root,
                    compaction_evidence_path=compaction_path,
                    terminal_checkpoint_path=checkpoint_path,
                    terminal_manifest=terminal_manifest,
                )
            )

        union_declaration = EvaluationCompactProductUnion(
            consumer_id="feedbax_external_conformance.consumer",
            consumer_version="v1",
            output_schema_id="feedbax_external_conformance.union",
            output_schema_version="feedbax_external_conformance.union.v1",
            output_role="union",
            output_logical_name="fixture-union",
            sources=tuple(union_sources),
        )
        union_result = finalize_evaluation_compact_product_union(
            union_declaration,
            tuple(union_bindings),
            custody_root=root / "union",
            finalizer_registry=state.registry(EVALUATION_PRODUCT_UNION_FINALIZERS),
        )
        union_payload = json.loads(
            ImmutableArtifactBlobProvider(root / "union" / "analysis").get_bytes(
                union_result.terminal_product
            )
        )
        if [item["cohort_key"] for item in union_payload["cohorts"]] != ["left", "right"]:
            raise AssertionError("public compact-product union lost authored cohort order")
        if union_result.completed_stages != ("UNION", "COLLECT", "CERTIFY", "TEARDOWN"):
            raise AssertionError("public compact-product union omitted terminal evidence")

        if registry.keys() != ("foundation", "dependent"):
            raise AssertionError("plugin dependency result depends on discovery order")
        provenance = state.provenance
        if tuple(item.plugin_id for item in provenance) != expected_plugins:
            raise AssertionError("plugin provenance order drifted")
        if tuple(item.registration_order for item in provenance) != (0, 1):
            raise AssertionError("plugin provenance registration order drifted")
        if tuple(item.registered_keys for item in provenance) != (
            {
                COMPONENTS.family: (EXTERNAL_DYNAMIC_COMPONENT,),
                DRIVERS.family: ("fixture:driver",),
                FIXTURE_RECORDS.family: ("foundation",),
                TRAINING_METHODS.family: ("feedbax_external_conformance/training/v1",),
                ROW_LOWERERS.family: (
                    "('feedbax_external_conformance.training', 'v1', "
                    "'feedbax_external_conformance.lowerer', 'v1')",
                ),
                EXECUTION_PREPARATIONS.family: ("feedbax_external_conformance/training/v1",),
                ANALYSIS_RECIPES.family: ("feedbax_external_conformance.analysis",),
                EVALUATION_RECIPES.family: ("feedbax_external_conformance.evaluation",),
                EVALUATION_BATCH_CONSUMERS.family: ("feedbax_external_conformance.consumer@v1",),
                EVALUATION_PRODUCT_UNION_FINALIZERS.family: (
                    "feedbax_external_conformance.consumer@v1",
                ),
            },
            {FIXTURE_RECORDS.family: ("dependent",)},
        ):
            raise AssertionError("plugin provenance registered-key attribution drifted")
        expected_family_protocols = (
            {
                COMPONENTS.family: "1",
                DRIVERS.family: "1",
                FIXTURE_RECORDS.family: "1",
                TRAINING_METHODS.family: "1",
                ROW_LOWERERS.family: "1",
                EXECUTION_PREPARATIONS.family: "1",
                ANALYSIS_RECIPES.family: "1",
                EVALUATION_RECIPES.family: "1",
                EVALUATION_BATCH_CONSUMERS.family: "1",
                EVALUATION_PRODUCT_UNION_FINALIZERS.family: "1",
            },
            {FIXTURE_RECORDS.family: "1"},
        )
        for item, family_protocols in zip(
            provenance,
            expected_family_protocols,
            strict=True,
        ):
            if (
                item.distribution != "feedbax-external-conformance"
                or item.distribution_version != "0.1.0"
                or len(item.fingerprint) != 64
                or item.family_protocols != family_protocols
            ):
                raise AssertionError("installed plugin provenance is incomplete")
        try:
            registry.register("late")
        except RuntimeError as exc:
            if "sealed" not in str(exc):
                raise
        else:
            raise AssertionError("published external registry remained mutable")

    retained: list[FixtureRecordRegistry] = []

    def retained_factory() -> FixtureRecordRegistry:
        registry = FixtureRecordRegistry()
        retained.append(registry)
        return registry

    def fail_after_partial(context: RegistrationContext) -> None:
        context.registry(FIXTURE_RECORDS).register("partial")
        raise RuntimeError("fixture failure")

    failure = _fixture_plugin(
        "fixture.failure",
        fail_after_partial,
        registry_families=(
            RegistryFamilyRegistration(
                FIXTURE_RECORDS,
                retained_factory,
                lambda registry: registry.seal(),
            ),
        ),
    )
    try:
        asyncio.run(
            compose_application(
                entry_points=(_RegistrationEntryPoint(failure),),
                local_component_source=None,
            )
        )
    except BootstrapError as exc:
        if exc.code is not BootstrapErrorCode.REGISTRATION_FAILURE or exc.plugin_id != (
            "fixture.failure"
        ):
            raise
    else:
        raise AssertionError("partial plugin failure published bootstrap state")
    try:
        retained[0].register("escaped")
    except RuntimeError as exc:
        if "sealed" not in str(exc):
            raise
    else:
        raise AssertionError("failed bootstrap leaked a mutable retained registry")
    empty_provider = _fixture_plugin(
        "fixture.empty",
        lambda _context: None,
        registry_families=(
            RegistryFamilyRegistration(
                FIXTURE_RECORDS,
                FixtureRecordRegistry,
                lambda registry: registry.seal(),
            ),
        ),
    )
    empty = asyncio.run(
        compose_application(
            entry_points=(_RegistrationEntryPoint(empty_provider),),
            local_component_source=None,
        )
    )
    if empty.registry(FIXTURE_RECORDS).keys():
        raise AssertionError("failed registration contaminated an isolated context")

    first = _fixture_plugin(
        "fixture.conflict.first",
        lambda context: context.registry(FIXTURE_RECORDS).register("collision"),
        registry_families=(
            RegistryFamilyRegistration(
                FIXTURE_RECORDS,
                FixtureRecordRegistry,
                lambda registry: registry.seal(),
            ),
        ),
    )
    second = _fixture_plugin(
        "fixture.conflict.second",
        lambda context: context.registry(FIXTURE_RECORDS).register("collision"),
        dependencies=(PluginDependency("fixture.conflict.first", "1"),),
    )
    try:
        asyncio.run(
            compose_application(
                entry_points=(
                    _RegistrationEntryPoint(second),
                    _RegistrationEntryPoint(first),
                ),
                local_component_source=None,
            )
        )
    except BootstrapError as exc:
        if exc.code is not BootstrapErrorCode.NAMESPACE_COLLISION or exc.plugin_id != (
            "fixture.conflict.second"
        ):
            raise
    else:
        raise AssertionError("namespace collision published bootstrap state")

    missing = _fixture_plugin(
        "fixture.missing",
        lambda _context: None,
        dependencies=(PluginDependency("fixture.absent", "1"),),
        registry_families=(
            RegistryFamilyRegistration(
                FIXTURE_RECORDS,
                FixtureRecordRegistry,
                lambda registry: registry.seal(),
            ),
        ),
    )
    try:
        asyncio.run(
            compose_application(
                entry_points=(_RegistrationEntryPoint(missing),),
                local_component_source=None,
            )
        )
    except BootstrapError as exc:
        if exc.code is not BootstrapErrorCode.MISSING_DEPENDENCY or exc.plugin_id != (
            "fixture.missing"
        ):
            raise
    else:
        raise AssertionError("missing plugin dependency was accepted")

    class LegacyRegistrarPoint:
        name = "fixture-legacy"
        value = "fixture_legacy:register"
        dist = None

        @staticmethod
        def load():
            return lambda _registry: None

    try:
        discover_plugin_registrations(entry_points=(LegacyRegistrarPoint(),))
    except BootstrapError as exc:
        if exc.code is not BootstrapErrorCode.INVALID_REGISTRATION:
            raise
    else:
        raise AssertionError("legacy registrar-only entry point was accepted")
    return True


def check_dynamic_component_ports(*, entry_points: Iterable[object] | None = None) -> bool:
    """Prove an external dynamic component across bootstrap, schema, build, and runtime."""

    state = asyncio.run(
        compose_application(
            entry_points=entry_points,
            local_component_source=None,
        )
    )
    registry = state.registry(COMPONENTS)
    meta = registry.get(EXTERNAL_DYNAMIC_COMPONENT)
    if meta is None or meta.dynamic_port_policy is None:
        raise AssertionError("external dynamic component policy was not bootstrapped")
    definition = next(
        item for item in registry.list_all() if item.name == EXTERNAL_DYNAMIC_COMPONENT
    )
    if definition.schema_version != "feedbax.spec.component_definition.v3":
        raise AssertionError("external component definition did not retain v3 identity")

    graph_spec = GraphSpec(
        nodes={
            "external": ComponentSpec(
                type=EXTERNAL_DYNAMIC_COMPONENT,
                params={"channels": ["left", "middle", "right"]},
            )
        },
        input_ports=["left", "middle", "right"],
        output_ports=["output"],
        input_bindings={
            "left": ("external", "source_0"),
            "middle": ("external", "source_1"),
            "right": ("external", "source_2"),
        },
        output_bindings={"output": ("external", "output")},
    )
    materialized = normalize_graph_for_studio_authoring(
        graph_spec,
        component_registry=registry,
    )
    node = materialized.nodes["external"]
    if node.input_ports != ["source_0", "source_1", "source_2"]:
        raise AssertionError("external dynamic inputs were not deterministically materialized")
    if node.output_ports != ["output"]:
        raise AssertionError("external fixed output was not materialized")

    graph = spec_to_graph(graph_spec, component_registry=registry)
    runtime_node = graph.nodes["external"]
    if tuple(runtime_node.input_ports) != tuple(node.input_ports):
        raise AssertionError("runtime dynamic port order drifted from the materialized schema")
    component_state = init_state_from_component(graph)
    outputs, _ = graph(
        {
            "left": jnp.array([1.0]),
            "middle": jnp.array([2.0, 3.0]),
            "right": jnp.array([4.0]),
        },
        component_state,
        key=jax.random.PRNGKey(0),
    )
    if not np.array_equal(np.asarray(outputs["output"]), np.array([1.0, 2.0, 3.0, 4.0])):
        raise AssertionError("external dynamic component runtime output drifted")

    invalid = graph_spec.model_copy(
        update={
            "nodes": {
                "external": node.model_copy(update={"input_ports": ["source_0"]}),
            }
        }
    )
    try:
        spec_to_graph(invalid, component_registry=registry)
    except ValueError as exc:
        if "dynamic port layout mismatch" not in str(exc):
            raise
    else:
        raise AssertionError("external dynamic namespace mismatch was accepted")
    return True


def check_external_driver_plugin() -> bool:
    """Construct an installed external driver through unified plugin bootstrap."""
    state = asyncio.run(compose_application(local_component_source=None))
    driver = state.registry(DRIVERS).construct(
        "fixture:driver",
        DriverConstructionContext(configuration={"nested": {"source": "external-wheel"}}),
    )
    facts = driver.realized_capabilities.facts
    if facts.resources is not ResourceSemantics.EXTERNALLY_MANAGED:
        raise AssertionError("external driver resource ownership facts drifted")
    if facts.teardown is not TeardownSemantics.RESOURCES_PRESERVED:
        raise AssertionError("external driver teardown preservation facts drifted")
    return True


def _component_registry(*, migration_first: bool) -> ComponentRegistry:
    registry = ComponentRegistry(load_user_components=False)
    if registry.get(_CUSTOM_COMPONENT) is not None:
        raise AssertionError("fixture component appeared through import-time discovery")
    migration = ComponentMigration(
        source_type=_LEGACY_COMPONENT,
        target_type=_CUSTOM_COMPONENT,
        owner="feedbax-external-conformance",
        migration_id="feedbax-external-conformance.LegacyScale-to-CurrentScale.v1",
        source_param_schema_version="legacy",
        target_param_schema_version="1",
        migrate_params=lambda params: {"scale": params["factor"]},
    )
    pack = ComponentMigrationPack(
        owner="feedbax-external-conformance",
        package="feedbax-external-conformance",
        migrations=(migration,),
    )
    if migration_first:
        registry.register_migration_pack(pack)
    registry.register_component_type(
        _CUSTOM_COMPONENT,
        lambda params: dict(params),
        owner="feedbax-external-conformance",
        provenance="package:feedbax-external-conformance",
        param_schema=[
            {"name": "scale", "type": "float", "required": True},
        ],
        param_schema_version="1",
    )
    if not migration_first:
        registry.register_migration_pack(pack)
    return registry


def check_component_registration_and_migration() -> bool:
    """Prove explicit component registration and owner migration-pack behavior."""
    for migration_first in (False, True):
        registry = _component_registry(migration_first=migration_first)
        resolved = registry.resolve_component_spec(
            _LEGACY_COMPONENT,
            {"factor": 3.0},
            param_schema_version="legacy",
        )
        if (
            resolved.type_id != _CUSTOM_COMPONENT
            or resolved.params != {"scale": 3.0}
            or resolved.param_schema_version != "1"
        ):
            raise AssertionError("component migration depends on registration order")
        try:
            registry.register_migration(
                ComponentMigration(
                    source_type=_LEGACY_COMPONENT,
                    target_type=_CUSTOM_COMPONENT,
                    owner="conflicting-owner",
                    migration_id="conflicting.edge.v1",
                    source_param_schema_version="legacy",
                    target_param_schema_version="1",
                )
            )
        except ValueError as exc:
            if "already registered" not in str(exc):
                raise
        else:
            raise AssertionError("conflicting component migration was accepted")
    return True


def check_value_identity() -> bool:
    """Exercise authored, semantic, realization, and fail-closed schema identity."""
    authored = authored_value_sha256(
        encoding_kind="fixture.literal",
        encoding_schema_id="fixture.value",
        encoding_schema_version="fixture.value.v1",
        arguments={"values": [0.0, 1.0]},
        movable_locators=("one/location",),
    )
    relocated = authored_value_sha256(
        encoding_kind="fixture.literal",
        encoding_schema_id="fixture.value",
        encoding_schema_version="fixture.value.v1",
        arguments={"values": [0.0, 1.0]},
        movable_locators=("another/location",),
    )
    if authored != relocated:
        raise AssertionError("movable locator changed authored value identity")
    if semantic_value_sha256([-0.0], dtype="float64") != semantic_value_sha256(
        [0.0], dtype="float64"
    ):
        raise AssertionError("signed zero was not normalized")
    record = value_identity_record(
        authored_sha256=authored,
        value=[0.0, 1.0],
        dtype="float64",
        layout_fingerprint="fixture-c-order",
        backend_fingerprint="fixture-cpu",
    )
    if record.realization_sha256 is None:
        raise AssertionError("requested realization identity was absent")
    old = record.model_dump(mode="json")
    old["schema_version"] = "feedbax.value_identity.v0"
    try:
        ValueIdentityRecord.model_validate(old)
    except ValidationError:
        pass
    else:
        raise AssertionError("old value-identity schema was accepted")
    return True


def check_component_param_array_values() -> bool:
    """Exercise the public typed array contract through GraphSpec execution."""
    sparse = SparseCooArrayValueSpec(
        schema_id=ARRAY_VALUE_SCHEMA_ID,
        schema_version=ARRAY_VALUE_SCHEMA_VERSION,
        encoding="sparse_coo",
        shape=(2, 2),
        dtype="float32",
        nonfinite="forbid",
        fill=0.0,
        entries=(SparseCooEntrySpec(coordinate=(0, 1), value=0.5),),
    )
    constant = ConstantArrayValueSpec(
        schema_id=ARRAY_VALUE_SCHEMA_ID,
        schema_version=ARRAY_VALUE_SCHEMA_VERSION,
        encoding="constant",
        shape=(2, 2),
        dtype="float32",
        nonfinite="forbid",
        value=0.5,
    )
    dense_sparse = np.asarray([[0.0, 0.5], [0.0, 0.0]], dtype=np.float32)
    dense_constant = np.full((2, 2), 0.5, dtype=np.float32)
    if semantic_value_sha256(
        materialize_array_value(sparse), dtype="float32"
    ) != semantic_value_sha256(dense_sparse, dtype="float32"):
        raise AssertionError("sparse component-param materialization changed semantics")
    if semantic_value_sha256(
        materialize_array_value(constant), dtype="float32"
    ) != semantic_value_sha256(dense_constant, dtype="float32"):
        raise AssertionError("constant component-param materialization changed semantics")

    graph_spec = GraphSpec(
        nodes={
            "plant": ComponentSpec(
                type="StructuralLinearStateSpace",
                params={
                    "A": [[1.0, 0.0], [0.0, 1.0]],
                    "B": [[0.0], [0.0]],
                    "B_w": [[0.0], [0.0]],
                    "delta_A": sparse.model_dump(mode="json"),
                    "initial_state": [0.0, 0.0],
                    "pos_slice": [0, 1],
                    "vel_slice": [1, 2],
                },
                param_schema_version="feedbax.component.structural_linear_state_space.v1",
                input_ports=["force", "epsilon"],
                output_ports=["effector", "state"],
            )
        }
    )
    runtime = spec_to_graph(graph_spec, ComponentRegistry(load_user_components=False))
    if runtime.nodes["plant"].initial_delta_A != ((0.0, 0.5), (0.0, 0.0)):
        raise AssertionError("GraphSpec did not materialize sparse component params")
    if graph_to_spec(runtime).nodes["plant"].params["delta_A"] != sparse.model_dump(mode="json"):
        raise AssertionError("runtime round-trip lost authored sparse array identity")

    try:
        ComponentSpec.model_validate(
            {
                "type": "fixture.Component",
                "params": {"value": {"schema_id": ARRAY_VALUE_SCHEMA_ID}},
            }
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("partial component-param array tags were accepted")
    return True


def _parent(digest: str, *, role: str) -> ParentRef:
    return ParentRef(
        kind="TrainingRunManifest",
        id=f"feedbax-training-run:{role}",
        role=role,
        uri=f"artifact://sha256/{digest}",
        metadata={"manifest_sha256": digest, "size_bytes": 8},
    )


def _material_dependencies() -> MaterialDependencySet:
    manifest = _parent("a" * 64, role="training_run")
    checkpoint = _parent("b" * 64, role="training_checkpoint_custody")
    return MaterialDependencySet(
        schema_id=MATERIAL_DEPENDENCIES_SCHEMA_ID,
        schema_version=MATERIAL_DEPENDENCIES_SCHEMA_VERSION,
        dependencies=[
            MaterialDependency(name="manifest", value=manifest),
            MaterialDependency(
                name="checkpoint",
                value=checkpoint,
                depends_on=["manifest"],
            ),
        ],
        identity_inputs=["checkpoint"],
        provenance_metadata={"fixture": "external"},
    )


def check_material_dependencies() -> bool:
    """Use the public testing helper for positive and negative admission cases."""
    spec = _material_dependencies()
    observations = [
        MaterialDependencyObservation(
            name=dependency.name,
            value=dependency.value,
            available=True,
            authentic=True,
        )
        for dependency in reversed(spec.dependencies)
    ]
    report = check_material_dependency_contract(spec, observations)
    if report.dependency_count != 2 or not report.missing_canary or not report.unauthentic_canary:
        raise AssertionError("material-dependency conformance report was incomplete")
    old = spec.model_dump(mode="json")
    old["schema_version"] = "feedbax.spec.material_dependencies.v0"
    try:
        MaterialDependencySet.model_validate(old)
    except ValidationError:
        pass
    else:
        raise AssertionError("old material-dependency schema was accepted")
    return True


def check_exact_parent_migration() -> bool:
    """Prove v1 migration, v2 material binding, and unknown-version rejection."""
    parent = _parent("c" * 64, role="training_run")
    legacy = {
        "schema_id": STAGED_EXACT_PARENTS_SCHEMA_ID,
        "schema_version": STAGED_EXACT_PARENTS_SCHEMA_VERSION_V1,
        "parents": [{"parent": parent.model_dump(mode="json"), "execution_uri": "run-a"}],
        "metadata": {"fixture": True},
    }
    migrated = migrate_staged_exact_parents(legacy)
    expected_migrated = {
        "schema_id": STAGED_EXACT_PARENTS_SCHEMA_ID,
        "schema_version": STAGED_EXACT_PARENTS_SCHEMA_VERSION,
        "parents": [
            {
                "parent": parent.model_dump(mode="json"),
                "execution_uri": "run-a",
                "material_dependencies": None,
            }
        ],
        "metadata": {"fixture": True},
    }
    if migrated.model_dump(mode="json") != expected_migrated:
        raise AssertionError("StagedExactParents v1 migration drifted")
    current = StagedExactParents(
        schema_id=STAGED_EXACT_PARENTS_SCHEMA_ID,
        schema_version=STAGED_EXACT_PARENTS_SCHEMA_VERSION,
        parents=[
            StagedExactParentEntry(
                parent=parent,
                execution_uri="run-a",
                material_dependencies=_material_dependencies(),
            )
        ],
    )
    if current.parents[0].parent != parent:
        raise AssertionError("exact parent did not round-trip byte-identically")
    unknown = dict(legacy)
    unknown["schema_version"] = "feedbax.spec.staged_exact_parents.v0"
    try:
        migrate_staged_exact_parents(unknown)
    except ValueError as exc:
        if "unsupported StagedExactParents schema_version" not in str(exc):
            raise
    else:
        raise AssertionError("unsupported StagedExactParents version was accepted")
    return True


@dataclass(frozen=True)
class _ProjectedParameters:
    arm: str
    target: int


@dataclass(frozen=True)
class _ProjectedMetadata:
    states_schema: str


def _authenticated_ref(
    kind: str,
    id_: str,
    role: str,
    raw_bytes: bytes,
) -> ParentRef:
    return ParentRef(
        kind=kind,
        id=id_,
        role=role,
        metadata={
            "ref_schema_id": AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
            "ref_schema_version": AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
            "manifest_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "size_bytes": len(raw_bytes),
        },
    )


def _projection_input(root: Path, target: int) -> ResolvedManifestInput:
    training = _authenticated_ref(
        "TrainingRunManifest",
        "fixture-training",
        "training_run",
        b"fixture-training",
    )
    run_spec = EvaluationRunSpec(
        evaluation_type="fixture.row_projection",
        inputs=[training],
        params={"arm": "trained", "target": target},
    )
    states = {"sample": np.asarray(target)}
    artifact = store_evaluation_states_artifact(
        states,
        root=root,
        manifest_id=f"fixture-evaluation-{target}",
    )
    manifest = EvaluationRunManifest(
        id=f"fixture-evaluation-{target}",
        status="completed",
        evaluation_spec=SpecPayload(
            kind="EvaluationRunSpec",
            schema_id="feedbax.spec.evaluation_run",
            schema_version="feedbax.spec.evaluation_run.v1",
            inline=run_spec.model_dump(mode="json"),
        ),
        input_training_runs=[training],
        artifacts=[artifact],
        metadata={"states_schema": "fixture.states.v1"},
        provenance=Provenance(
            entrypoint=EntrypointRef(
                kind="feedbax-evaluation-recipe",
                name=run_spec.evaluation_type,
            ),
            parents=[training],
        ),
    )
    raw_bytes = (manifest.model_dump_json(indent=2, exclude_none=True) + "\n").encode()
    authority = _authenticated_ref(
        "EvaluationRunManifest",
        manifest.id,
        "evaluation_run",
        raw_bytes,
    )
    return ResolvedManifestInput(
        ref=authority,
        manifest=manifest,
        path=Path(f"/fixture/{manifest.id}.json"),
        raw_bytes=raw_bytes,
    )


def check_resolved_evaluation_row_projection() -> bool:
    """Exercise the narrow resolver-handle projection boundary from a clean wheel."""

    def project(facts):
        params = _ProjectedParameters(**facts.parameters)
        metadata = _ProjectedMetadata(**facts.metadata)
        if metadata.states_schema != "fixture.states.v1":
            raise ValueError("unexpected state schema")
        return (
            (params.arm, params.target),
            int(facts.states["sample"]),
            metadata,
        )

    with TemporaryDirectory() as directory:
        root = Path(directory)
        bootstrap_state = asyncio.run(
            bootstrap_application(
                new_registration_context(local_component_source=None), registrations=()
            )
        )
        manifest_inputs = [_projection_input(root, target) for target in (0, 1)]
        inputs = resolve_analysis_inputs(
            AnalysisRunSpec(
                analysis_type="fixture.row_projection.analysis",
                inputs=[item.ref for item in manifest_inputs],
                evaluation_states_policy="require_durable",
            ),
            registry=bootstrap_state.bundle.analysis_recipes,
            evaluation_registry=bootstrap_state.bundle.evaluation_recipes,
            root=root,
            authenticated_inputs=dict(enumerate(manifest_inputs)),
        )
        projected = project_evaluation_rows(inputs, project=project)
        if tuple((key, state) for key, state, _metadata in projected) != (
            (("trained", 0), 0),
            (("trained", 1), 1),
        ):
            raise AssertionError("resolved evaluation row projection drifted")
        spliced = replace(
            inputs[0],
            ref=inputs[1].ref,
            manifest_input=inputs[1].manifest_input,
        )
        try:
            project_evaluation_rows([spliced], project=project)
        except EvaluationRowProjectionError as exc:
            if exc.code is not EvaluationRowProjectionErrorCode.STATE_HANDLE_MISMATCH:
                raise AssertionError("row projection returned the wrong splice reason") from exc
        else:
            raise AssertionError("row projection accepted a cross-authority splice")
    return True


def check_figure_composition_public_contract() -> bool:
    """Prove the installed public figure composition and display contract."""

    def write_json(root: Path, name: str, payload: dict[str, object]) -> ContentPinnedJsonBase:
        path = root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return ContentPinnedJsonBase(
            ref=name,
            sha256=sha256_bytes(canonical_json_bytes(payload)),
        )

    def run_figure_cli(figure_cli: Path, *arguments: str) -> str:
        try:
            return subprocess.run(
                [str(figure_cli), *arguments],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except subprocess.CalledProcessError as exc:
            raise AssertionError(
                "installed feedbax-figure failed: "
                f"returncode={exc.returncode}, stderr={exc.stderr!r}"
            ) from exc

    with TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        base = FigureSpec(name="base", assembler="feedbax.grid_figure")
        base_ref = write_json(root, "base.json", base.model_dump(mode="json", exclude_none=True))
        middle = FigureCompositionSpec(
            parent=base_ref,
            deltas=[
                MatrixCompositionDelta(
                    layer_id="middle",
                    patches=[OverridePatch(op="replace", path="name", value="middle")],
                )
            ],
        )
        middle_ref = write_json(
            root,
            "middle.json",
            middle.model_dump(mode="json", exclude_none=True),
        )
        leaf = FigureCompositionSpec(
            parent=middle_ref,
            deltas=[
                MatrixCompositionDelta(
                    layer_id="leaf",
                    patches=[OverridePatch(op="replace", path="name", value="resolved")],
                    acknowledges_ancestor_paths=["name"],
                )
            ],
        )
        resolved = resolve_figure_spec(leaf, repo_root=root)
        if not isinstance(resolved, ResolvedFigureSpec):
            raise AssertionError("public resolver returned the wrong result type")
        if resolved.figure_spec.schema_version != "feedbax.spec.figure.v2":
            raise AssertionError("composition did not resolve to ordinary FigureSpec v2")
        if resolved.authored_identity_sha256 == resolved.resolved_identity_sha256:
            raise AssertionError("authored and resolved figure identities collapsed")
        if not isinstance(resolved.composition, FigureCompositionProvenance):
            raise AssertionError("composition provenance is missing")
        documents = resolved.composition.documents
        if [
            (record.order, record.role, record.ref, record.payload_path) for record in documents
        ] != [
            (0, "root_figure", "base.json", None),
            (1, "composition_envelope", "middle.json", None),
            (2, "authored_leaf", "<inline>", None),
        ] or not all(isinstance(record, FigureCompositionSourceRecord) for record in documents):
            raise AssertionError("full-chain source custody drifted")
        qualified = [
            layer_id
            for layer in resolved.composition.layers
            for layer_id in layer.qualified_layer_ids
        ]
        if len(qualified) != len(set(qualified)) or not all(":" in item for item in qualified):
            raise AssertionError("composition layer attribution is not qualified and unique")
        if coerce_figure_spec(leaf, repo_root=root) != resolved.figure_spec:
            raise AssertionError("coercer and resolver semantics diverged")

        leaf_path = root / "leaf.json"
        leaf_path.write_text(leaf.model_dump_json(), encoding="utf-8")
        figure_cli = Path(sys.executable).with_name("feedbax-figure")
        if not figure_cli.is_file():
            raise AssertionError("installed feedbax-figure console entrypoint is unavailable")
        displayed_spec = run_figure_cli(
            figure_cli, "resolve", str(leaf_path), "--repo-root", str(root)
        )
        if json.loads(displayed_spec) != resolved.figure_spec.model_dump(
            mode="json", exclude_none=True
        ):
            raise AssertionError("CLI display differs from public resolver semantics")
        displayed = json.loads(
            run_figure_cli(
                figure_cli,
                "resolve",
                str(leaf_path),
                "--repo-root",
                str(root),
                "--with-lineage",
            )
        )
        if displayed["resolved_identity_sha256"] != resolved.resolved_identity_sha256:
            raise AssertionError("CLI lineage identity differs from resolver identity")

        shared = {"index": {"values": [0, 1]}}
        shared_ref = write_json(root, "shared.json", shared).model_copy(
            update={"payload_path": ("index",)}
        )
        inherited = {
            "families": [{}],
            SOURCE_DOCUMENT_INHERITANCE_KEY: SourceDocumentInheritance(
                inherit=[{"target": "families.0.index", "parent": shared_ref}]
            ).model_dump(mode="json", exclude_none=True),
        }
        effective = materialize_inherited_document(inherited, repo_root=root)
        if effective["families"][0]["index"] != load_content_pinned_json_base(
            shared_ref, repo_root=root
        ):
            raise AssertionError("canonical list-index graft differs from selected payload")
        collided = json.loads(json.dumps(inherited))
        collided["families"][0]["index"] = {"local": True}
        try:
            materialize_inherited_document(collided, repo_root=root)
        except ValueError as exc:
            if "collides with a locally-present key" not in str(exc):
                raise
        else:
            raise AssertionError("source inheritance overwrote a present target")

        payload = {"metadata": {"variant": "base"}}
        ancestor = MatrixCompositionDelta(
            layer_id="ancestor",
            patches=[OverridePatch(op="replace", path="metadata", value={"variant": "a"})],
        )
        sibling = MatrixCompositionDelta(
            layer_id="child",
            patches=[OverridePatch(op="replace", path="metadata.variant", value="b")],
            acknowledges_ancestor_paths=["metadata.other"],
        )
        try:
            apply_composition_deltas(payload, [ancestor, sibling])
        except ValueError as exc:
            if "without explicit acknowledgement" not in str(exc):
                raise
        else:
            raise AssertionError("sibling path acknowledged an overlapping composition write")

        stage = BundleStageSpec(name="figure", kind="figure", figure=leaf)
        bundle = AnalysisBundleSpec(name="figure-bundle", stages=[stage])
        registries = asyncio.run(
            bootstrap_application(
                new_registration_context(local_component_source=None), registrations=()
            )
        ).bundle
        execution_root = root / "bundle-execution"
        execution = execute_staged_analysis_bundle(
            bundle,
            root=execution_root,
            repo_root=root,
            registries=registries,
        )
        if len(execution.stages) != 1 or execution.stages[0].status != "materialized":
            raise AssertionError("composed staged figure did not complete")
        manifest_paths = list((execution_root / "manifests" / "FigureManifest").glob("*.json"))
        if len(manifest_paths) != 1:
            raise AssertionError("composed staged figure did not emit one manifest")
        manifest = load_manifest(manifest_paths[0])
        if manifest.kind != "FigureManifest" or manifest.figure_spec.inline != (
            resolved.figure_spec.model_dump(mode="json", exclude_none=True)
        ):
            raise AssertionError("staged execution semantics differ from public resolution")
        for untrusted_root in (None, root / "wrong-repo-root"):
            if untrusted_root is not None:
                untrusted_root.mkdir()
            try:
                execute_staged_analysis_bundle(
                    bundle,
                    root=root / f"rejected-{untrusted_root is None}",
                    repo_root=untrusted_root,
                    registries=registries,
                )
            except (FileNotFoundError, ValueError):
                pass
            else:
                raise AssertionError("staged composition accepted an untrusted repository root")
        runtime = FigureRuntimeBindingSpec(
            authored_figure_source_sha256=resolved.authored_identity_sha256,
            resolved_figure_spec_sha256=resolved.resolved_identity_sha256,
            inputs=[],
            input_authorities=[],
            artifact_provider_bindings=[],
        )
        if runtime.schema_version != "feedbax.spec.figure_runtime_binding.v2":
            raise AssertionError("runtime binding v2 identity contract drifted")
    return True


def check_figure_role_reference_public_contract() -> bool:
    """Prove the installed public row index, selector, and figure role contract."""

    index = AuthenticatedRowIndex(
        index_id="external-fixture-rows",
        rows=[
            {"row_id": "row-a", "label": "row a", "tags": ["baseline", "external"]},
            {"row_id": "row-b", "label": "row b", "tags": ["external", "variant"]},
        ],
    )
    resolved_rows = expand_row_selector({"mode": "all"}, index)
    if resolved_rows.row_ids != ["row-a", "row-b"]:
        raise AssertionError("row-set expansion lost the index order")
    if resolved_rows.index_sha256 != index.canonical_sha256():
        raise AssertionError("expanded row set did not pin its source index digest")
    if expand_row_selector({"mode": "tag", "tag": "variant"}, index).row_ids != ["row-b"]:
        raise AssertionError("tag selection did not resolve the declared subset")
    for rejected in ({"mode": "any"}, {"mode": "all", "tag": "external"}, {"mode": "tag"}):
        try:
            expand_row_selector(rejected, index)
        except ValueError:
            continue
        raise AssertionError(f"row-set selector union admitted {rejected!r}")
    try:
        expand_row_selector({"mode": "tag", "tag": "absent"}, index)
    except RowSelectionError as exc:
        if exc.code is not RowSelectionErrorCode.EMPTY_SELECTION:
            raise AssertionError("empty selection did not fail with its stable code") from exc
    else:
        raise AssertionError("empty selection did not fail closed")

    contracts = [
        FigureRoleBindingContract(
            input_role="measured",
            artifact_role="external_result",
            artifact_provider="results",
        ),
        FigureRoleBindingContract(
            input_role="reference",
            artifact_role="external_result",
            artifact_provider="results",
        ),
    ]
    # A contract may pin the identity of the payload the selected artifact
    # decodes to, and does so whole or not at all: an id without a version admits
    # every version of that schema, which is not a smaller claim but an empty one.
    pinned = FigureRoleBindingContract(
        input_role="measured",
        artifact_role="external_result",
        artifact_provider="results",
        payload_name="measured",
        payload_schema_id="external.fixture_result",
        payload_schema_version="external.fixture_result.v1",
    )
    payload_selector = pinned.artifact_payload("row_1__measured")
    if payload_selector["name"] != "measured":
        raise AssertionError("an explicit payload name was not carried into the selector")
    if (
        payload_selector["payload_schema_id"] != "external.fixture_result"
        or payload_selector["payload_schema_version"] != "external.fixture_result.v1"
    ):
        raise AssertionError("the pinned payload schema identity was not emitted")
    for half in ("payload_schema_id", "payload_schema_version"):
        try:
            FigureRoleBindingContract(
                input_role="measured",
                artifact_role="external_result",
                artifact_provider="results",
                **{half: "external.fixture_result"},
            )
        except ValidationError:
            continue
        raise AssertionError(f"a contract accepted {half!r} without its counterpart")

    request = FigureRowExpansionRequest(
        figure_name="external-fixture-figure",
        rows={"mode": "all"},
        inputs={"measured": {"per_row": "primary"}, "reference": {"shared": "run:reference"}},
        role_contracts=contracts,
        assembler_title="External fixture rows",
    )
    try:
        FigureRowExpansionRequest(
            figure_name="external-fixture-figure",
            rows={"mode": "all"},
            inputs={
                "measured": {
                    "shared": "run:measured",
                    "manifest_sha256": "0" * 64,
                    "size_bytes": 1,
                }
            },
            role_contracts=contracts,
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("authored figure input accepted an embedded authority block")

    pending = resolve_figure_input_roles(request, resolved_rows)
    if pending.fully_bound or pending.pending_roles != ("row_1__measured", "row_2__measured"):
        raise AssertionError("a first-time figure did not resolve to pending per-row roles")

    produced = {row_id: json.dumps({"row": row_id}).encode() for row_id in ("row-a", "row-b")}
    bindings = RowIndexCustodyBindings(
        index_id=index.index_id,
        # v2 pins the index cut, not just the index: an id is stable across cuts.
        index_sha256=index.canonical_sha256(),
        bindings=[
            {
                "row_id": row_id,
                "binding_key": "primary",
                "parent": {
                    "kind": "AnalysisRunManifest",
                    "id": f"run:{row_id}",
                    "role": "analysis_run",
                    "metadata": {
                        "ref_schema_id": "feedbax.ref.authenticated_manifest",
                        "ref_schema_version": "feedbax.ref.authenticated_manifest.v1",
                        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                        "size_bytes": len(raw),
                    },
                },
            }
            for row_id, raw in produced.items()
        ],
    )
    resolved_inputs = resolve_figure_input_roles(request, resolved_rows, bindings)
    if not resolved_inputs.fully_bound or len(resolved_inputs.inputs) != 4:
        raise AssertionError("bound role resolution did not expand row-major over the row set")
    if [item.role for item in resolved_inputs.inputs] != [
        "row_1__measured",
        "row_1__reference",
        "row_2__measured",
        "row_2__reference",
    ]:
        raise AssertionError("figure role names drifted from the row namespace")
    if {item.parent.id for item in resolved_inputs.inputs if item.binding == "shared"} != {
        "run:reference"
    }:
        raise AssertionError("a shared role did not stay experiment-invariant")

    base = FigureSpec(
        name="external-fixture-base",
        assembler="feedbax.grid_figure",
        assembler_params={"height": 300, "title": "base"},
        panels=[
            PanelSpec(name="left", title="left panel", row=1, col=1),
            PanelSpec(name="right", title="right panel", row=1, col=2),
        ],
        trace_families=[
            TraceFamily(
                name="measured-left",
                index=TraceFamilyIndex(values=[0, 1]),
                legend_index=0,
                trace=TraceBinding(
                    name="measured-left-{index}",
                    constructor="feedbax.line",
                    panel="left",
                    data={"y": {"item": "measured", "path": "values.{index}"}},
                ),
            )
        ],
        metadata={"caption": "base caption"},
    )
    expanded = expand_figure_rows(base, request, resolved_rows, resolved_inputs)
    if len(expanded.panels) != 4 or len(expanded.trace_families or []) != 2:
        raise AssertionError("row expansion did not derive one base block per resolved row")
    if expanded.assembler_params["height"] != 600:
        raise AssertionError("assembler height was not derived from the base and row count")
    if expanded.assembler_params["title"] != "External fixture rows":
        raise AssertionError("the one authored assembler fact was not applied")
    if [panel.name for panel in expanded.panels] != [
        "row_1__left",
        "row_1__right",
        "row_2__left",
        "row_2__right",
    ]:
        raise AssertionError("panel namespace drifted from row-index order")
    if [panel.title for panel in expanded.panels][2] != "row b \u2014 left panel":
        raise AssertionError("panel titles are not derived from the row label")
    if [panel.row for panel in expanded.panels] != [1, 1, 2, 2]:
        raise AssertionError("panel grid placement did not follow row-index order")
    families = expanded.trace_families or []
    if families[0].legend_index != 0 or families[1].legend_index is not None:
        raise AssertionError("legend ownership did not stay with the first expanded row")
    if families[1].trace.params.get("showlegend") is not False:
        raise AssertionError("a later expanded row kept a duplicate legend entry")
    payload = expanded.model_dump(mode="json", exclude_none=True)
    if payload["trace_families"][1]["trace"]["data"]["y"]["item"] != "row_2__measured":
        raise AssertionError("trace data items were not bound to the expanded row role")
    if any(parent.get("metadata") for parent in payload["inputs"]):
        raise AssertionError("resolved figure inputs carried authority metadata")
    if len(payload["input_authorities"]) != 4:
        raise AssertionError("input authorities were not derived one per resolved input")
    if payload["metadata"] != base.metadata:
        raise AssertionError("row expansion restated derived facts into figure metadata")
    if expanded.schema_version != "feedbax.spec.figure.v2":
        raise AssertionError("row expansion did not produce ordinary FigureSpec v2 semantics")
    return True


__all__ = [
    "check_component_registration_and_migration",
    "check_component_param_array_values",
    "check_exact_parent_migration",
    "check_figure_composition_public_contract",
    "check_figure_role_reference_public_contract",
    "check_material_dependencies",
    "check_ordered_registration",
    "check_resolved_evaluation_row_projection",
    "check_value_identity",
]
