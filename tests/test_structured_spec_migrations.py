from __future__ import annotations

import pytest

from feedbax.contracts.artifact_custody import (
    IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_ID,
    IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_VERSION,
)
from feedbax.contracts.migrations import (
    SchemaMigration,
    STUDIO_TASK_BINDING_LEGACY_V1,
    SpecSchemaFamily,
    SpecSchemaRegistry,
    UnknownSpecFamily,
    UnsupportedSpecVersion,
    default_spec_registry,
    migrate_studio_workspace_spec,
    migrate_studio_scenario_spec,
    migrate_studio_stage_spec,
    migrate_studio_task_binding_spec,
    migrate_structured_spec_payload,
)
from feedbax.contracts.training import (
    LR_SCHEDULE_SPEC_SCHEMA_ID,
    LR_SCHEDULE_SPEC_SCHEMA_VERSION,
    LOSS_TERM_SPEC_SCHEMA_VERSION,
    LOSS_TERM_SPEC_SCHEMA_VERSION_V1,
    TRAINING_RUN_SPEC_SCHEMA_VERSION,
    TRAINING_RUN_SPEC_SCHEMA_VERSION_V1,
)
from feedbax.contracts.run_matrix import (
    TRAINING_ROW_PLANNING_PROVENANCE_SCHEMA_VERSION,
    TRAINING_ROW_PLANNING_PROVENANCE_SCHEMA_VERSION_V1,
    TRAINING_ROW_PROVENANCE_SCHEMA_VERSION,
    TRAINING_ROW_PROVENANCE_SCHEMA_VERSION_V1,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
)
from feedbax.contracts.descriptors import (
    COMPONENT_DESCRIPTOR_SCHEMA_VERSION,
    COMPONENT_SELECTOR_SYNTAX_SCHEMA_VERSION,
    DESCRIPTOR_BASIS_SCHEMA_VERSION,
    SELECTOR_FALLBACK_POLICY_SCHEMA_VERSION,
    SELECTOR_ROLE_IDENTITY_SCHEMA_VERSION,
    VARIABLE_DESCRIPTOR_SCHEMA_VERSION,
)
from feedbax.contracts.schema_namespace import SchemaNamespaceError, SchemaNamespaceKind
from feedbax.contracts.graph import (
    GRAPH_SPEC_SCHEMA_VERSION,
    GRAPH_SPEC_SCHEMA_VERSION_V2,
    GRAPH_SPEC_SCHEMA_VERSION_V3,
    LEGACY_STUDIO_SCENARIO_SCHEMA_VERSION,
    LEGACY_GRAPH_SPEC_SCHEMA_VERSION,
    STUDIO_BIOMECHANICS_SCHEMA_ID,
    STUDIO_BIOMECHANICS_SCHEMA_VERSION,
    STUDIO_SCENARIO_SCHEMA_VERSION,
    STUDIO_SCENARIO_SCHEMA_VERSION_V1,
    StudioScenarioSpec,
)
from feedbax.contracts.component import COMPONENT_DEFINITION_SCHEMA_VERSION_V1
from feedbax.contracts.expressions import (
    PATH_EXPRESSION_SCHEMA_ID,
    PATH_EXPRESSION_SCHEMA_VERSION,
)
from feedbax.contracts.extraction import (
    EXTRACTION_PRODUCT_SPEC_SCHEMA_ID,
    EXTRACTION_PRODUCT_SPEC_SCHEMA_VERSION,
)
from feedbax.contracts.manifest import (
    EVALUATION_STATES_CONTAINER_SCHEMA_VERSION,
    FIGURE_MANIFEST_SCHEMA_ID,
    FIGURE_MANIFEST_SCHEMA_VERSION,
)
from feedbax.contracts.studio_api import (
    STUDIO_API_TRANSPORT_SCHEMA_ID,
    STUDIO_API_TRANSPORT_SCHEMA_VERSION,
)
from feedbax.contracts.representation import (
    REPRESENTATION_SCHEMA_ID,
    REPRESENTATION_SCHEMA_VERSION,
    REPRESENTATION_SCHEMA_VERSION_V4,
    REPRESENTATION_SCHEMA_VERSION_V3,
    REPRESENTATION_SCHEMA_VERSION_V2,
    REPRESENTATION_SCHEMA_VERSION_V1,
    REPRESENTATION_SCHEMA_VERSION_V0,
    RepresentationSpec,
)
from feedbax.contracts.workspace_replay import (
    WORKSPACE_REPLAY_SCHEMA_ID,
    WORKSPACE_REPLAY_SCHEMA_VERSION,
)
from feedbax.contracts.checkpoints import (
    TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V1,
    TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V2,
    TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V3,
    TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V4,
    TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V5,
    TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V6,
)
from feedbax.contracts.value_schema import ValueSchema
from feedbax.execution.models import (
    EXECUTION_CLOUD_PAYLOAD_SCHEMA_ID,
    EXECUTION_CLOUD_PAYLOAD_SCHEMA_VERSION,
    EXECUTION_REPRODUCIBILITY_SCHEMA_ID,
    EXECUTION_REPRODUCIBILITY_SCHEMA_VERSION,
)
from feedbax.objectives.spec import validate_objective_spec
from feedbax.orchestration.events import RUN_EVENT_SCHEMA_ID, RUN_EVENT_SCHEMA_VERSION
from feedbax.training.diagnostics import (
    NATIVE_EXECUTION_PRODUCER_CONTEXT_SCHEMA_ID,
    NATIVE_EXECUTION_PRODUCER_CONTEXT_SCHEMA_VERSION,
    TRAINING_DIAGNOSTICS_SCHEMA_ID,
    TRAINING_DIAGNOSTICS_SCHEMA_VERSION,
)
from feedbax.orchestration.bundle import (
    EXECUTION_IDENTITY_ENVELOPE_SCHEMA_ID,
    EXECUTION_IDENTITY_ENVELOPE_SCHEMA_VERSION,
    EXECUTION_IDENTITY_ENVELOPE_SCHEMA_VERSION_V1,
    RUN_BUNDLE_SCHEMA_VERSION,
    RUN_BUNDLE_SCHEMA_VERSION_V1,
    RUN_BUNDLE_SCHEMA_VERSION_V2,
    RUN_BUNDLE_SCHEMA_VERSION_V3,
    RUN_BUNDLE_SCHEMA_VERSION_V4,
    RunBundle,
)
from feedbax.contracts.spec_storage import training_run_execution_hash

pytestmark = [pytest.mark.feedbax_contract, pytest.mark.migration_contract]


def _registry() -> SpecSchemaRegistry:
    registry = SpecSchemaRegistry()
    registry.register_family(
        SpecSchemaFamily(
            kind="DemoSpec",
            schema_id="feedbax.spec.demo",
            current_version="feedbax.spec.demo.v2",
        )
    )
    return registry


def test_structured_spec_registry_rejects_flat_feedbax_schema_identity() -> None:
    registry = SpecSchemaRegistry()

    with pytest.raises(SchemaNamespaceError) as excinfo:
        registry.register_family(
            SpecSchemaFamily(
                kind="DemoSpec",
                schema_id="feedbax.demo",
                current_version="feedbax.demo.v1",
            )
        )

    message = str(excinfo.value)
    assert "DemoSpec" in message
    assert "feedbax.demo" in message
    assert "feedbax.spec." in message


def test_structured_spec_registry_accepts_current_version_without_migration() -> None:
    registry = _registry()
    payload = {"schema_version": "feedbax.spec.demo.v2", "value": 3}

    result = registry.migrate("DemoSpec", payload)

    assert result.kind == "DemoSpec"
    assert result.schema_id == "feedbax.spec.demo"
    assert result.source_version == "feedbax.spec.demo.v2"
    assert result.target_version == "feedbax.spec.demo.v2"
    assert result.payload == payload
    assert result.migration_records == []
    assert not result.migrated


def test_structured_spec_registry_rejects_versionless_payload_without_opt_in() -> None:
    registry = _registry()
    payload = {"value": 3}

    with pytest.raises(UnsupportedSpecVersion, match="missing schema_version"):
        registry.migrate("DemoSpec", payload)


def test_structured_spec_registry_requires_explicit_current_version_opt_in() -> None:
    registry = _registry()
    payload = {"value": 3}

    result = registry.migrate("DemoSpec", payload, assume_current=True)

    assert result.source_version == "feedbax.spec.demo.v2"
    assert result.target_version == "feedbax.spec.demo.v2"
    assert result.payload == payload
    assert "schema_version" not in result.payload


def test_structured_spec_registry_applies_registered_family_migration() -> None:
    registry = _registry()

    def migrate_v1_to_v2(payload: dict[str, object]) -> dict[str, object]:
        payload["renamed"] = payload.pop("old")
        return payload

    registry.register_migration(
        "DemoSpec",
        SchemaMigration(
            source_version="demo.v1",
            target_version="feedbax.spec.demo.v2",
            migration_id="demo-spec-v1-to-v2",
            migrate=migrate_v1_to_v2,
            description="Rename old to renamed.",
        ),
    )

    result = registry.migrate(
        "DemoSpec",
        {"schema_version": "demo.v1", "old": 7},
    )

    assert result.payload == {"schema_version": "feedbax.spec.demo.v2", "renamed": 7}
    assert result.migrated
    assert [record.migration_id for record in result.migration_records] == ["demo-spec-v1-to-v2"]
    assert result.migration_records[0].source_schema_version == "demo.v1"
    assert result.migration_records[0].target_schema_version == "feedbax.spec.demo.v2"


def test_structured_spec_registry_rejects_explicit_unsupported_old_version() -> None:
    registry = _registry()
    registry.reject_version(
        "DemoSpec",
        "demo.v0",
        reason="pre-release payloads were never durable",
    )

    with pytest.raises(UnsupportedSpecVersion) as excinfo:
        registry.migrate("DemoSpec", {"schema_version": "demo.v0"})

    message = str(excinfo.value)
    assert "family='DemoSpec'" in message
    assert "schema_id='feedbax.spec.demo'" in message
    assert "source_version='demo.v0'" in message
    assert "current_version='feedbax.spec.demo.v2'" in message
    assert "migration_intentionally_absent=yes" in message
    assert "pre-release payloads were never durable" in message


@pytest.mark.parametrize(
    ("kind", "schema_id", "current_version"),
    [
        (
            "RunAssemblyRequest",
            "feedbax.spec.run_assembly_request",
            "feedbax.spec.run_assembly_request.v1",
        ),
        (
            "StudioTrainingAssemblySpec",
            "feedbax.spec.studio.training_assembly",
            "feedbax.spec.studio.training_assembly.v1",
        ),
    ],
)
def test_default_registry_registers_assemble_contract_families(
    kind: str,
    schema_id: str,
    current_version: str,
) -> None:
    family = default_spec_registry.resolve(kind)

    assert family.identity == schema_id
    assert family.current_version == current_version
    assert family.policy is not None
    assert family.policy.stance == "reject"
    accepted = default_spec_registry.migrate(
        kind,
        {"schema_id": schema_id, "schema_version": current_version},
    )
    assert accepted.target_version == current_version

    with pytest.raises(UnsupportedSpecVersion, match="migration_intentionally_absent=yes"):
        default_spec_registry.migrate(
            kind,
            {"schema_id": schema_id, "schema_version": f"{schema_id}.v0"},
        )


def test_execution_identity_envelope_v1_migrates_with_unavailable_provenance() -> None:
    family = default_spec_registry.resolve("ExecutionIdentityEnvelope")
    assert family.identity == EXECUTION_IDENTITY_ENVELOPE_SCHEMA_ID
    assert family.current_version == EXECUTION_IDENTITY_ENVELOPE_SCHEMA_VERSION
    assert family.policy is not None
    assert family.policy.stance == "migrate"

    migrated = default_spec_registry.migrate(
        "ExecutionIdentityEnvelope",
        {
            "schema_id": EXECUTION_IDENTITY_ENVELOPE_SCHEMA_ID,
            "schema_version": EXECUTION_IDENTITY_ENVELOPE_SCHEMA_VERSION_V1,
        },
    )
    assert migrated.payload["schema_version"] == EXECUTION_IDENTITY_ENVELOPE_SCHEMA_VERSION
    assert migrated.payload["row_provenance"] is None
    assert [record.migration_id for record in migrated.migration_records] == [
        "execution-identity-envelope-v1-to-v2-row-provenance-unavailable"
    ]


@pytest.mark.parametrize(
    ("kind", "schema_id", "current_version"),
    [
        (
            "NativeExecutionProducerContext",
            NATIVE_EXECUTION_PRODUCER_CONTEXT_SCHEMA_ID,
            NATIVE_EXECUTION_PRODUCER_CONTEXT_SCHEMA_VERSION,
        ),
        (
            "TrainingDiagnostics",
            TRAINING_DIAGNOSTICS_SCHEMA_ID,
            TRAINING_DIAGNOSTICS_SCHEMA_VERSION,
        ),
    ],
)
def test_native_execution_documents_have_explicit_rejection_policy(
    kind: str,
    schema_id: str,
    current_version: str,
) -> None:
    family = default_spec_registry.resolve(kind)
    assert family.identity == schema_id
    assert family.current_version == current_version
    assert family.policy is not None
    assert family.policy.owner_module.startswith("feedbax.training.diagnostics.")
    assert family.policy.stance == "reject"
    assert family.policy.emitted_by
    assert family.policy.consumed_by

    accepted = default_spec_registry.migrate(
        kind,
        {"schema_id": schema_id, "schema_version": current_version},
    )
    assert accepted.target_version == current_version
    with pytest.raises(UnsupportedSpecVersion, match="migration_intentionally_absent=yes"):
        default_spec_registry.migrate(
            kind,
            {"schema_id": schema_id, "schema_version": f"{schema_id}.v0"},
        )


@pytest.mark.parametrize("old_version", [RUN_BUNDLE_SCHEMA_VERSION_V1, RUN_BUNDLE_SCHEMA_VERSION_V2])
def test_run_bundle_old_versions_require_reassembly(old_version: str) -> None:
    family = default_spec_registry.resolve("RunBundle")
    assert family.current_version == RUN_BUNDLE_SCHEMA_VERSION
    assert family.policy is not None
    assert family.policy.stance == "migrate"

    with pytest.raises(UnsupportedSpecVersion) as excinfo:
        default_spec_registry.migrate("RunBundle", {"schema_version": old_version})

    message = str(excinfo.value)
    assert old_version in message
    assert "migration_intentionally_absent=yes" in message


def test_run_bundle_v3_migrates_with_explicitly_unavailable_row_provenance() -> None:
    migrated = default_spec_registry.migrate(
        "RunBundle",
        {
            "schema_id": "feedbax.orchestration.run_bundle",
            "schema_version": RUN_BUNDLE_SCHEMA_VERSION_V3,
            "rows": [{"row_id": "legacy"}],
        },
    )

    assert migrated.target_version == RUN_BUNDLE_SCHEMA_VERSION
    assert migrated.payload["schema_version"] == RUN_BUNDLE_SCHEMA_VERSION
    assert "provenance" not in migrated.payload["rows"][0]
    assert [record.migration_id for record in migrated.migration_records] == [
        "run-bundle-v3-to-v4-training-row-provenance",
        "run-bundle-v4-to-v5-envelope-row-provenance",
    ]


def test_run_bundle_v4_moves_row_provenance_into_execution_envelope() -> None:
    payload_sha256 = "a" * 64
    resolved_root_hash = "b" * 64
    execution_hash = training_run_execution_hash(resolved_root_hash, [])
    provenance = {
        "schema_id": "feedbax.spec.training_row_provenance",
        "schema_version": TRAINING_ROW_PROVENANCE_SCHEMA_VERSION_V1,
        "row_id": "row",
        "row_index": 0,
        "planned_run_id": "feedbax-training-run:row",
        "authored_payload_hash": "c" * 64,
        "seed": 7,
        "axis_coordinates": {"learning_rate": 0.001},
        "overrides": [{"path": "training.learning_rate", "value": 0.001}],
        "lowerer_identities": [
            {"lowerer_id": "feedbax.tests.lowerer", "lowerer_version": "v1"}
        ],
    }
    migrated = default_spec_registry.migrate(
        "RunBundle",
        {
            "schema_id": "feedbax.orchestration.run_bundle",
            "schema_version": RUN_BUNDLE_SCHEMA_VERSION_V4,
            "run_set_id": "2026-07-13-deadbeef",
            "driver": "local",
            "rows": [
                {
                    "row_id": "row",
                    "provenance": provenance,
                    "execution": {
                        "schema_id": EXECUTION_IDENTITY_ENVELOPE_SCHEMA_ID,
                        "schema_version": EXECUTION_IDENTITY_ENVELOPE_SCHEMA_VERSION_V1,
                        "payload": {
                            "schema_id": "feedbax.spec.training_run",
                            "schema_version": "feedbax.spec.training_run.v1",
                            "artifact_id": f"artifact://sha256/{payload_sha256}",
                            "sha256": payload_sha256,
                        },
                        "authored_intent": {
                            "schema_id": "feedbax.spec.authored_training",
                            "schema_version": "feedbax.spec.authored_training.v1",
                            "artifact_id": "artifact://authored/row",
                            "sha256": "d" * 64,
                            "intent_hash": "e" * 64,
                        },
                        "resolved_snapshot": {
                            "schema_id": "feedbax.spec.resolved_training",
                            "schema_version": "feedbax.spec.resolved_training.v1",
                            "artifact_id": "artifact://resolved/row",
                            "sha256": "f" * 64,
                            "root_hash": resolved_root_hash,
                        },
                        "execution_capsule": {
                            "schema_id": "feedbax.spec.execution_capsule",
                            "schema_version": "feedbax.spec.execution_capsule.v1",
                            "artifact_id": "artifact://execution/row",
                            "sha256": "1" * 64,
                            "execution_hash": execution_hash,
                        },
                        "immutable_inputs": [],
                    },
                    "launch": {"command": ["python", "-m", "feedbax"]},
                }
            ],
            "environment": {"python_version": "3.12"},
            "budget": {"max_wall_clock_seconds": 60.0},
        },
    )
    row = migrated.payload["rows"][0]
    assert "provenance" not in row
    assert row["execution"]["schema_version"] == EXECUTION_IDENTITY_ENVELOPE_SCHEMA_VERSION
    assert row["execution"]["row_provenance"] == {
        **provenance,
        "schema_version": TRAINING_ROW_PROVENANCE_SCHEMA_VERSION,
        "lowered_execution_payload_hash": payload_sha256,
    }
    bundle = RunBundle.model_validate(migrated.payload)
    assert bundle.rows[0].execution.row_provenance is not None
    assert (
        bundle.rows[0].execution.row_provenance.lowered_execution_payload_hash
        == bundle.rows[0].execution.payload.sha256
    )


@pytest.mark.parametrize(
    ("kind", "schema_id"),
    [
        ("AuthoredTrainingRow", "feedbax.spec.authored_training_row"),
        ("TrainingRowLoweringResult", "feedbax.spec.training_row_lowering_result"),
        (
            "TrainingRowPlanningProvenance",
            "feedbax.spec.training_row_planning_provenance",
        ),
        ("TrainingRowProvenance", "feedbax.spec.training_row_provenance"),
    ],
)
def test_row_lowering_contracts_have_explicit_schema_policy(
    kind: str,
    schema_id: str,
) -> None:
    family = default_spec_registry.resolve(kind)
    assert family.identity == schema_id
    expected_version = {
        "TrainingRowPlanningProvenance": TRAINING_ROW_PLANNING_PROVENANCE_SCHEMA_VERSION,
        "TrainingRowProvenance": TRAINING_ROW_PROVENANCE_SCHEMA_VERSION,
    }.get(kind, f"{schema_id}.v1")
    assert family.current_version == expected_version
    assert family.policy is not None
    assert family.policy.stance == "reject"
    with pytest.raises(UnsupportedSpecVersion, match="migration_intentionally_absent=yes"):
        default_spec_registry.migrate(
            kind,
            {"schema_id": schema_id, "schema_version": f"{schema_id}.v0"},
        )

    rejected_v1 = {
        "TrainingRowPlanningProvenance": (
            TRAINING_ROW_PLANNING_PROVENANCE_SCHEMA_VERSION_V1
        ),
        "TrainingRowProvenance": TRAINING_ROW_PROVENANCE_SCHEMA_VERSION_V1,
    }.get(kind)
    if rejected_v1 is not None:
        with pytest.raises(UnsupportedSpecVersion, match="migration_intentionally_absent=yes"):
            default_spec_registry.migrate(
                kind,
                {"schema_id": schema_id, "schema_version": rejected_v1},
            )


def test_structured_spec_registry_reports_unknown_family() -> None:
    registry = _registry()

    with pytest.raises(UnknownSpecFamily) as excinfo:
        registry.migrate("MissingSpec", {"schema_version": "missing.v1"})

    message = str(excinfo.value)
    assert "Unknown Feedbax structured spec family 'MissingSpec'" in message
    assert "known families: DemoSpec" in message


def test_default_registry_registers_evaluation_states_container_family() -> None:
    family = default_spec_registry.resolve("EvaluationStatesContainer")

    assert family.identity == "feedbax.manifest.evaluation_states_container"
    assert family.current_version == EVALUATION_STATES_CONTAINER_SCHEMA_VERSION
    assert family.policy is not None
    assert family.policy.stance == "reject"
    migrations = default_spec_registry.available_migrations("EvaluationStatesContainer")
    assert [migration.migration_id for migration in migrations] == [
        "evaluation-states-container-v1-to-v2"
    ]


def test_default_registry_registers_figure_manifest_family() -> None:
    family = default_spec_registry.resolve("FigureManifest")

    assert family.identity == FIGURE_MANIFEST_SCHEMA_ID
    assert family.current_version == FIGURE_MANIFEST_SCHEMA_VERSION
    assert family.policy is not None
    assert family.policy.owner_module == "feedbax.contracts.manifest"
    assert family.policy.stance == "reject"


def test_structured_spec_registry_reports_missing_migration_path() -> None:
    registry = _registry()

    with pytest.raises(UnsupportedSpecVersion) as excinfo:
        registry.migrate("DemoSpec", {"schema_version": "demo.v1"})

    message = str(excinfo.value)
    assert "No Feedbax structured spec migration path registered" in message
    assert "family='DemoSpec'" in message
    assert "source_version='demo.v1'" in message
    assert "current_version='feedbax.spec.demo.v2'" in message
    assert "no explicit unsupported-version policy" in message


def test_default_structured_spec_registry_exposes_foundation_families() -> None:
    families = {family.kind: family for family in default_spec_registry.families()}

    assert families["GraphSpec"].identity == "feedbax.spec.graph"
    assert families["GraphSpec"].namespace == SchemaNamespaceKind.SPEC
    assert families["GraphSpec"].current_version == GRAPH_SPEC_SCHEMA_VERSION
    assert families["PopulationStructureSpec"].identity == "feedbax.spec.population_structure"
    assert families["PopulationStructureSpec"].namespace == SchemaNamespaceKind.SPEC
    assert families["TrainingRunSpec"].identity == "feedbax.spec.training_run"
    assert families["TrainingRunSpec"].current_version == TRAINING_RUN_SPEC_SCHEMA_VERSION
    assert families["TrainingRunMatrixSpec"].identity == TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID
    assert (
        families["TrainingRunMatrixSpec"].current_version == TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION
    )
    assert families["RunEvent"].identity == RUN_EVENT_SCHEMA_ID
    assert families["RunEvent"].namespace == SchemaNamespaceKind.RUN_EVENT
    assert families["RunEvent"].current_version == RUN_EVENT_SCHEMA_VERSION
    assert families["TrainingSpec"].identity == "feedbax.spec.training"
    assert families["LrScheduleSpec"].identity == LR_SCHEDULE_SPEC_SCHEMA_ID
    assert families["LrScheduleSpec"].current_version == LR_SCHEDULE_SPEC_SCHEMA_VERSION
    assert families["LossTermSpec"].identity == "feedbax.spec.training.loss_term"
    assert families["LossTermSpec"].current_version == LOSS_TERM_SPEC_SCHEMA_VERSION
    assert (
        families["StandardSupervisedMethodPayload"].identity
        == "feedbax.spec.training_method.standard_supervised_payload"
    )
    assert families["AnalysisBundleSpec"].identity == "feedbax.spec.analysis_bundle"
    assert families["AnalysisBundleSpec"].current_version == "feedbax.spec.analysis_bundle.v4"
    assert families["PathExpression"].identity == PATH_EXPRESSION_SCHEMA_ID
    assert families["PathExpression"].current_version == PATH_EXPRESSION_SCHEMA_VERSION
    assert families["ExtractionProductSpec"].identity == EXTRACTION_PRODUCT_SPEC_SCHEMA_ID
    assert (
        families["ExtractionProductSpec"].current_version == EXTRACTION_PRODUCT_SPEC_SCHEMA_VERSION
    )
    assert families["ReportSpec"].identity == "feedbax.spec.report"
    assert families["ReportSpec"].current_version == "feedbax.spec.report.v1"
    assert (
        families["AnalysisDataProductRequirement"].identity
        == "feedbax.spec.analysis_data_product_requirement"
    )
    assert families["AnalysisDataProduct"].identity == "feedbax.manifest.analysis_data_product"
    assert families["ExecutionSpec"].identity == "feedbax.spec.execution"
    assert families["ExecutionSpec"].current_version == "feedbax.spec.execution.v2"
    assert families["ExecutionPlan"].identity == "feedbax.manifest.execution_plan"
    assert families["ExecutionPlan"].current_version == "feedbax.manifest.execution.v3"
    assert families["ExecutionCloudPayload"].identity == EXECUTION_CLOUD_PAYLOAD_SCHEMA_ID
    assert (
        families["ExecutionCloudPayload"].current_version == EXECUTION_CLOUD_PAYLOAD_SCHEMA_VERSION
    )
    assert families["ExecutionReproducibility"].identity == EXECUTION_REPRODUCIBILITY_SCHEMA_ID
    assert (
        families["ExecutionReproducibility"].current_version
        == EXECUTION_REPRODUCIBILITY_SCHEMA_VERSION
    )
    assert families["LocalExecutionResult"].identity == "feedbax.manifest.local_execution_result"
    assert families["LocalExecutionResult"].current_version == "feedbax.manifest.execution.v3"
    assert (
        families["StagedAnalysisBundleExecution"].identity
        == "feedbax.manifest.analysis_bundle_execution"
    )
    assert families["ValueSchema"].identity == "feedbax.spec.studio.schema.value"
    assert families["StudioApiTransport"].identity == STUDIO_API_TRANSPORT_SCHEMA_ID
    assert families["StudioApiTransport"].current_version == STUDIO_API_TRANSPORT_SCHEMA_VERSION
    assert families["RepresentationSpec"].identity == REPRESENTATION_SCHEMA_ID
    assert families["RepresentationSpec"].current_version == REPRESENTATION_SCHEMA_VERSION
    assert families["VariableDescriptor"].identity == "feedbax.spec.descriptor.variable"
    assert families["ComponentDescriptor"].identity == "feedbax.spec.descriptor.component"
    assert families["DescriptorBasisIdentity"].identity == "feedbax.spec.descriptor.basis"
    assert (
        families["TrainingCheckpointTransactionManifest"].identity
        == "feedbax.manifest.training_checkpoint_transaction"
    )
    assert (
        families["TrainingCheckpointLatestPointer"].identity
        == "feedbax.manifest.training_checkpoint_latest_pointer"
    )
    assert families["TrainingRunManifest"].identity == "feedbax.manifest.training_run"
    assert families["RegenerationSpec"].identity == "feedbax.spec.regeneration"
    assert families["ProviderManifest"].current_version == "feedbax.manifest.v1"
    assert families["ModelArtifactManifest"].identity == "feedbax.manifest.model_artifact"
    assert families["SpecPayload"].identity == "feedbax.manifest.spec_payload"
    assert families["SpecPayload"].namespace == SchemaNamespaceKind.MANIFEST
    assert not families["RegistryEntry"].durable
    assert not families["StudioSchemaRegistry"].durable


def test_loss_term_spec_v1_migrates_to_v2_schema_identity() -> None:
    result = migrate_structured_spec_payload(
        "LossTermSpec",
        {
            "schema_version": LOSS_TERM_SPEC_SCHEMA_VERSION_V1,
            "type": "TargetStateLoss",
            "label": "position",
            "selector": "state.output",
            "target_value": [1.0, 0.0],
            "norm": "huber",
            "time_agg": {"mode": "final"},
        },
        path="loss",
    )

    assert result.target_version == LOSS_TERM_SPEC_SCHEMA_VERSION
    assert result.payload["schema_id"] == "feedbax.spec.training.loss_term"
    assert result.payload["schema_version"] == LOSS_TERM_SPEC_SCHEMA_VERSION
    assert result.payload["type"] == "TargetStateLoss"
    assert [record.migration_id for record in result.migration_records] == [
        "loss-term-spec-v1-to-v2-objective-adapter"
    ]


def test_loss_term_spec_current_version_accepts_without_migration() -> None:
    payload = {
        "schema_id": "feedbax.spec.training.loss_term",
        "schema_version": LOSS_TERM_SPEC_SCHEMA_VERSION,
        "type": "TargetStateLoss",
        "label": "position",
        "selector": "state.output",
        "target_value": [1.0, 0.0],
    }

    result = migrate_structured_spec_payload("LossTermSpec", payload, path="loss")

    assert result.payload == payload
    assert not result.migrated


def test_loss_term_spec_v1_migration_rejects_unmapped_range_mode() -> None:
    with pytest.raises(ValueError, match="no ObjectiveSpec equivalent"):
        migrate_structured_spec_payload(
            "LossTermSpec",
            {
                "schema_version": LOSS_TERM_SPEC_SCHEMA_VERSION_V1,
                "type": "TargetStateLoss",
                "label": "position",
                "selector": "state.output",
                "target_value": [1.0, 0.0],
                "time_agg": {"mode": "range", "start": 0, "end": 2},
            },
            path="loss",
        )


def test_manifest_schema_identities_survive_contract_package_move() -> None:
    families = {family.kind: family for family in default_spec_registry.families()}

    expected_manifest_identities = {
        "ArrayRecord": "feedbax.manifest.array_record",
        "ArrayStorePayload": "feedbax.manifest.array_store",
        "GraphSpecManifest": "feedbax.manifest.graph_spec",
        "ModelArtifactManifest": "feedbax.manifest.model_artifact",
        "ProviderManifest": "feedbax.manifest.provider",
        "RegistrySnapshot": "feedbax.manifest.registry_snapshot",
        "SpecPayload": "feedbax.manifest.spec_payload",
        "StagedAnalysisBundleExecution": "feedbax.manifest.analysis_bundle_execution",
        "TrainingCheckpointTransactionManifest": (
            "feedbax.manifest.training_checkpoint_transaction"
        ),
        "TrainingCheckpointLatestPointer": "feedbax.manifest.training_checkpoint_latest_pointer",
        "TrainingRunManifest": "feedbax.manifest.training_run",
        "ExecutionCloudPayload": EXECUTION_CLOUD_PAYLOAD_SCHEMA_ID,
        "ExecutionReproducibility": EXECUTION_REPRODUCIBILITY_SCHEMA_ID,
        "StudioPipelineMaterializationResult": (
            "feedbax.manifest.studio.pipeline_materialization_result"
        ),
        "StudioSchemaRegistry": "feedbax.manifest.studio.schema_registry",
        "StudioTrainingLocalRunResult": "feedbax.manifest.studio.training_local_run_result",
        "WorkspaceReplayProduct": WORKSPACE_REPLAY_SCHEMA_ID,
    }

    for kind, identity in expected_manifest_identities.items():
        assert families[kind].identity == identity
        assert families[kind].namespace == SchemaNamespaceKind.MANIFEST
        assert families[kind].current_version.startswith("feedbax.manifest.")


def test_policy_matrix_uses_canonical_owner_and_emitter_modules() -> None:
    families = {family.kind: family for family in default_spec_registry.families()}

    expected_policy_paths = {
        "VariableDescriptor": (
            "feedbax.contracts.descriptors",
            ("GraphSpec/training/run metadata", "provider_manifest.schemas"),
        ),
        "ComponentDescriptor": (
            "feedbax.contracts.descriptors",
            ("VariableDescriptor components", "provider_manifest.schemas"),
        ),
        "DescriptorBasisIdentity": (
            "feedbax.contracts.descriptors",
            ("descriptor-bearing specs", "provider_manifest.schemas"),
        ),
        "AnalysisDataProductRequirement": (
            "feedbax.contracts.graph",
            ("AnalysisRunSpec.input_requirements", "provider_manifest.schemas"),
        ),
        "AnalysisDataProduct": (
            "feedbax.contracts.manifest",
            ("AnalysisRunManifest.produced_data", "provider_manifest.schemas"),
        ),
        "TrainingRunSpec": (
            "feedbax.contracts.training",
            ("TrainingRunManifest.training_spec", "provider_manifest.schemas"),
        ),
        "TrainingRunMatrixSpec": (
            "feedbax.contracts.run_matrix",
            ("feedbax.training.run_matrix", "provider_manifest.schemas"),
        ),
        "LrScheduleSpec": (
            "feedbax.contracts.training",
            ("OptimizerSpec.lr_schedule", "provider_manifest.schemas"),
        ),
        "TrainingRunManifest": (
            "feedbax.contracts.manifest",
            ("feedbax.contracts.manifest", "feedbax.integrations.provider"),
        ),
        "TrainingCheckpointTransactionManifest": (
            "feedbax.contracts.checkpoints",
            ("feedbax.training.checkpoint_custody",),
        ),
        "TrainingCheckpointLatestPointer": (
            "feedbax.contracts.checkpoints",
            ("feedbax.training.checkpoint_custody",),
        ),
        "ExecutionSpec": (
            "feedbax.execution.models",
            ("feedbax.execution.models", "feedbax.integrations.provider"),
        ),
        "ExecutionPlan": (
            "feedbax.execution.models",
            ("feedbax.execution.models", "feedbax.integrations.provider"),
        ),
        "ExecutionCloudPayload": (
            "feedbax.execution.models",
            ("feedbax.execution.models", "feedbax.integrations.provider"),
        ),
        "ExecutionReproducibility": (
            "feedbax.execution.models",
            ("feedbax.execution.models", "feedbax.integrations.provider"),
        ),
        "LocalExecutionResult": (
            "feedbax.execution.models",
            ("feedbax.execution.models", "feedbax.integrations.provider"),
        ),
        "ArrayStorePayload": (
            "feedbax.contracts.artifact_schema",
            ("feedbax.contracts.artifact_schema", "provider_manifest.schemas"),
        ),
        "ModelArtifactManifest": (
            "feedbax.contracts.manifest",
            ("feedbax.contracts.manifest", "feedbax.integrations.provider"),
        ),
        "ProviderManifest": (
            "feedbax.integrations.provider",
            ("feedbax.integrations.provider.provider_manifest",),
        ),
        "ValueSchema": (
            "feedbax.contracts.value_schema",
            ("feedbax.studio.schema", "feedbax.integrations.provider"),
        ),
        "StudioSchemaRegistry": (
            "feedbax.studio.schema",
            ("feedbax.studio.schema", "feedbax.integrations.provider"),
        ),
        "StudioTrainingExecutionRequest": (
            "feedbax.studio.execution",
            ("feedbax.studio.execution", "feedbax.integrations.provider"),
        ),
        "WorkspaceReplayProduct": (
            "feedbax.contracts.workspace_replay",
            ("eval/validation replay materialization", "provider_manifest.schemas"),
        ),
        "StudioApiTransport": (
            "feedbax.contracts.studio_api",
            ("feedbax.contracts.studio_api", "scripts.generate_studio_contracts"),
        ),
    }

    for kind, (owner_module, emitted_by) in expected_policy_paths.items():
        policy = families[kind].policy
        assert policy is not None
        assert policy.owner_module == owner_module
        assert policy.emitted_by == emitted_by


def test_contract_value_schema_round_trips_without_payload_shape_change() -> None:
    from feedbax.studio.schema import ValueSchema as StudioValueSchema

    payload = {
        "id": "node.output",
        "label": "Node output",
        "kind": "array",
        "dtype": "float32",
        "shape": [None, 2],
        "rank": 2,
        "units": "m",
        "frame": "world",
        "origin": "declared",
        "metadata": {"source": "test"},
    }

    schema = ValueSchema.model_validate(payload)
    dumped = schema.model_dump()

    assert dumped == payload
    assert ValueSchema.__module__ == "feedbax.contracts.value_schema"
    assert StudioValueSchema is ValueSchema


def test_default_registry_enforces_spec_and_manifest_namespace_categories() -> None:
    spec_kinds = {
        "GraphSpec",
        "PopulationStructureSpec",
        "TrainingRunSpec",
        "TrainingRunMatrixSpec",
        "TrainingSpec",
        "LrScheduleSpec",
        "TaskSpec",
        "StandardSupervisedMethodPayload",
        "AnalysisDataProductRequirement",
        "VariableDescriptor",
        "ComponentDescriptor",
        "DescriptorBasisIdentity",
        "ObjectiveSpec",
        "EvaluationRunSpec",
        "AnalysisRunSpec",
        "AnalysisBundleSpec",
        "PathExpression",
        "ExtractionProductSpec",
        "ReportSpec",
        "RegenerationSpec",
        "ExecutionSpec",
        "StudioApiTransport",
        "StudioWorkspaceSpec",
        "StudioTaskBindingSpec",
        "StudioPipelineMaterializationRequest",
    }
    manifest_kinds = {
        "SpecPayload",
        "GraphSpecManifest",
        "ModelArtifactManifest",
        "ArrayStorePayload",
        "ArrayRecord",
        "TrainingCheckpointTransactionManifest",
        "TrainingCheckpointLatestPointer",
        "TrainingRunManifest",
        "AnalysisDataProduct",
        "ExecutionPlan",
        "ExecutionCloudPayload",
        "ExecutionReproducibility",
        "LocalExecutionResult",
        "StagedAnalysisBundleExecution",
        "ProviderManifest",
        "RegistrySnapshot",
        "StudioPipelineMaterializationResult",
        "WorkspaceReplayProduct",
    }

    families = {family.kind: family for family in default_spec_registry.families()}

    assert {families[kind].namespace for kind in spec_kinds} == {SchemaNamespaceKind.SPEC}
    assert {families[kind].namespace for kind in manifest_kinds} == {SchemaNamespaceKind.MANIFEST}
    assert families["WorkspaceReplayProduct"].current_version == WORKSPACE_REPLAY_SCHEMA_VERSION
    assert not any(
        family.namespace == SchemaNamespaceKind.COMPONENT_PARAMS
        for family in default_spec_registry.families()
    )


def test_default_policy_matrix_covers_registered_emitted_families() -> None:
    missing = default_spec_registry.families_missing_policy()

    assert missing == ()

    for family in default_spec_registry.families():
        assert family.policy is not None, family.kind
        policy = family.policy
        assert policy.owner_module
        assert policy.emitted_by
        assert policy.consumed_by
        assert policy.required_tests
        assert policy.rejection_message
        assert policy.stance in {"migrate", "reject"}
        if policy.stance == "migrate":
            assert policy.supported_old_versions, family.kind
        else:
            assert policy.rejected_old_versions, family.kind


def test_immutable_blob_provider_family_has_canonical_reject_policy() -> None:
    family = default_spec_registry.resolve("ImmutableArtifactBlobProviderSpec")
    policy = family.policy
    config_family = default_spec_registry.resolve("ImmutableArtifactBlobProviderConfig")

    assert family.identity == IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_ID
    assert family.current_version == IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_VERSION
    assert family.namespace == SchemaNamespaceKind.SPEC
    assert policy is not None
    assert policy.owner_module == "feedbax.contracts.artifact_custody"
    assert policy.emitted_by == (
        "feedbax.persistence.artifact_custody",
        "feedbax.integrations.provider",
    )
    assert policy.consumed_by == (
        "feedbax.persistence.open_immutable_artifact_blob_provider",
        "downstream staged execution",
    )
    assert policy.stance == "reject"
    assert policy.supported_old_versions == ()
    assert policy.rejected_old_versions == ("feedbax.spec.immutable_artifact_blob_provider.v0",)
    assert default_spec_registry.available_migrations(family.kind) == ()
    assert config_family.identity == family.identity
    assert config_family.current_version == family.current_version
    assert not config_family.durable
    assert config_family.policy is not None
    assert config_family.policy.covers == family.kind

    with pytest.raises(UnsupportedSpecVersion, match="current_version"):
        default_spec_registry.migrate(
            family.kind,
            {"schema_version": "feedbax.spec.immutable_artifact_blob_provider.v0"},
        )


def test_default_policy_matrix_covers_provider_schema_exports_and_capability_refs() -> None:
    from feedbax.integrations.provider import provider_manifest

    manifest = provider_manifest()
    schema_refs = set(manifest.schemas)
    for capability in manifest.capabilities.values():
        if capability.input_schema is not None:
            schema_refs.add(capability.input_schema)
        if capability.output_schema is not None:
            schema_refs.add(capability.output_schema)

    missing = schema_refs - set(default_spec_registry.policy_matrix())

    assert missing == set()
    assert default_spec_registry.resolve("ComponentRegistrySnapshot").policy is not None
    assert default_spec_registry.resolve("ComponentRegistrySnapshot").policy.covers == (
        "RegistrySnapshot"
    )


def test_default_policy_matrix_exercises_accept_migrate_or_reject_behavior() -> None:
    for family in default_spec_registry.families():
        current = default_spec_registry.migrate(
            family.kind,
            {"schema_version": family.current_version},
        )
        assert current.source_version == family.current_version
        assert current.target_version == family.current_version
        assert not current.migrated

        policy = family.policy
        assert policy is not None
        if policy.stance == "migrate":
            for old_version in policy.supported_old_versions:
                payload = {"schema_version": old_version}
                if family.kind == "StudioValueSpec":
                    payload.update({"mode": "constant", "value": 1, "metadata": {}})
                if family.kind == "TrainingRunMatrixSpec":
                    payload["base"] = {"inline": {}}
                migrated = default_spec_registry.migrate(
                    family.kind,
                    payload,
                )
                assert migrated.source_version == old_version
                assert migrated.target_version == family.current_version
                assert migrated.migrated

        for old_version in policy.rejected_old_versions:
            with pytest.raises(UnsupportedSpecVersion) as excinfo:
                default_spec_registry.migrate(
                    family.kind,
                    {"schema_version": old_version},
                )

            message = str(excinfo.value)
            assert f"family='{family.kind}'" in message
            assert f"schema_id='{family.identity}'" in message
            assert f"source_version='{old_version}'" in message
            assert f"current_version='{family.current_version}'" in message
            assert "migration_intentionally_absent=yes" in message


def test_default_policy_matrix_distinguishes_graph_and_studio_old_versions() -> None:
    graph_policy = default_spec_registry.resolve("GraphSpec").policy
    task_binding_policy = default_spec_registry.resolve("StudioTaskBindingSpec").policy
    objective_policy = default_spec_registry.resolve("ObjectiveSpec").policy
    population_policy = default_spec_registry.resolve("PopulationStructureSpec").policy
    execution_policy = default_spec_registry.resolve("ExecutionSpec").policy
    execution_plan_policy = default_spec_registry.resolve("ExecutionPlan").policy
    checkpoint_policy = default_spec_registry.resolve(
        "TrainingCheckpointTransactionManifest"
    ).policy
    cloud_payload_policy = default_spec_registry.resolve("ExecutionCloudPayload").policy
    reproducibility_policy = default_spec_registry.resolve("ExecutionReproducibility").policy
    local_execution_result_policy = default_spec_registry.resolve("LocalExecutionResult").policy
    studio_api_policy = default_spec_registry.resolve("StudioApiTransport").policy
    report_policy = default_spec_registry.resolve("ReportSpec").policy
    extraction_policy = default_spec_registry.resolve("ExtractionProductSpec").policy
    component_definition_policy = default_spec_registry.resolve("ComponentDefinition").policy

    assert graph_policy is not None
    assert graph_policy.stance == "migrate"
    assert graph_policy.supported_old_versions == (
        LEGACY_GRAPH_SPEC_SCHEMA_VERSION,
        GRAPH_SPEC_SCHEMA_VERSION_V2,
        GRAPH_SPEC_SCHEMA_VERSION_V3,
    )
    assert component_definition_policy is not None
    assert component_definition_policy.stance == "migrate"
    assert component_definition_policy.supported_old_versions == (
        COMPONENT_DEFINITION_SCHEMA_VERSION_V1,
    )
    assert task_binding_policy is not None
    assert task_binding_policy.stance == "migrate"
    assert task_binding_policy.supported_old_versions == (STUDIO_TASK_BINDING_LEGACY_V1,)
    assert task_binding_policy.rejected_old_versions == ("feedbax.studio.task_bindings.v0",)
    representation_policy = default_spec_registry.resolve("RepresentationSpec").policy
    assert representation_policy is not None
    assert representation_policy.stance == "migrate"
    assert representation_policy.supported_old_versions == (
        REPRESENTATION_SCHEMA_VERSION_V1,
        REPRESENTATION_SCHEMA_VERSION_V2,
        REPRESENTATION_SCHEMA_VERSION_V3,
        REPRESENTATION_SCHEMA_VERSION_V4,
    )
    assert representation_policy.rejected_old_versions == (REPRESENTATION_SCHEMA_VERSION_V0,)
    assert objective_policy is not None
    assert objective_policy.stance == "reject"
    assert objective_policy.rejected_old_versions == ("feedbax.objective.v0",)
    assert population_policy is not None
    training_run_policy = default_spec_registry.resolve("TrainingRunSpec").policy
    lr_schedule_policy = default_spec_registry.resolve("LrScheduleSpec").policy
    assert training_run_policy is not None
    assert training_run_policy.stance == "migrate"
    assert training_run_policy.supported_old_versions == (TRAINING_RUN_SPEC_SCHEMA_VERSION_V1,)
    assert lr_schedule_policy is not None
    assert lr_schedule_policy.stance == "migrate"
    assert lr_schedule_policy.supported_old_versions == (
        "feedbax.spec.training.lr_schedule.v1",
    )
    assert lr_schedule_policy.rejected_old_versions == ("feedbax.spec.training.lr_schedule.v0",)
    assert checkpoint_policy is not None
    assert checkpoint_policy.stance == "migrate"
    assert checkpoint_policy.supported_old_versions == (
        TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V1,
        TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V2,
        TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V3,
        TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V4,
            TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V5,
            TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V6,
    )
    assert execution_policy is not None
    assert execution_policy.rejected_old_versions == ("feedbax.spec.execution.v1",)
    assert report_policy is not None
    assert report_policy.stance == "reject"
    assert report_policy.rejected_old_versions == ("feedbax.spec.report.v0",)
    assert extraction_policy is not None
    assert extraction_policy.stance == "reject"
    assert extraction_policy.rejected_old_versions == ("feedbax.spec.extraction_product.v0",)
    assert execution_plan_policy is not None
    assert execution_plan_policy.rejected_old_versions == (
        "feedbax.manifest.execution.v2",
        "feedbax.manifest.execution.v1",
    )
    assert cloud_payload_policy is not None
    assert cloud_payload_policy.rejected_old_versions == (
        "feedbax.manifest.execution_cloud_payload.v0",
    )
    assert reproducibility_policy is not None
    assert reproducibility_policy.rejected_old_versions == (
        "feedbax.manifest.execution_reproducibility.v0",
    )
    assert local_execution_result_policy is not None
    assert local_execution_result_policy.rejected_old_versions == (
        "feedbax.manifest.execution.v2",
        "feedbax.manifest.execution.v1",
    )
    assert studio_api_policy is not None
    assert studio_api_policy.rejected_old_versions == ("feedbax.spec.studio.api_transport.v0",)
    assert population_policy.stance == "reject"
    assert population_policy.rejected_old_versions == ("feedbax.population_structure.v1",)

    migrated = default_spec_registry.migrate(
        "StudioTaskBindingSpec",
        {"schema_version": STUDIO_TASK_BINDING_LEGACY_V1},
    )
    assert migrated.migrated
    assert migrated.target_version == "feedbax.spec.studio.task_bindings.v2"

    with pytest.raises(UnsupportedSpecVersion) as excinfo:
        default_spec_registry.migrate(
            "StudioTaskBindingSpec",
            {"schema_version": "feedbax.studio.task_bindings.v0"},
        )

    message = str(excinfo.value)
    assert "family='StudioTaskBindingSpec'" in message
    assert "feedbax.studio.task_bindings.v0" in message
    assert "migration_intentionally_absent=yes" in message


def test_representation_v1_migrates_to_capability_aware_v2() -> None:
    result = default_spec_registry.migrate(
        "RepresentationSpec",
        {
            "schema_id": REPRESENTATION_SCHEMA_ID,
            "schema_version": REPRESENTATION_SCHEMA_VERSION_V1,
            "anchors": [
                {
                    "id": "origin",
                    "semantic_role": "origin",
                    "binding": {"kind": "literal", "value": [0.0, 0.0]},
                }
            ],
        },
    )

    assert result.source_version == REPRESENTATION_SCHEMA_VERSION_V1
    assert result.target_version == REPRESENTATION_SCHEMA_VERSION
    assert result.migrated
    representation = RepresentationSpec.model_validate(result.payload)
    assert representation.reachability is None
    assert representation.muscle_path_geometry is None


def test_representation_v2_migrates_to_muscle_path_aware_v3() -> None:
    result = default_spec_registry.migrate(
        "RepresentationSpec",
        {
            "schema_id": REPRESENTATION_SCHEMA_ID,
            "schema_version": REPRESENTATION_SCHEMA_VERSION_V2,
            "elements": [{"id": "paths", "archetype": "muscle_path"}],
            "muscle_path_geometry": {
                "frames": [{"id": "world", "origin": [0.0, 0.0]}],
                "paths": [
                    {
                        "id": "path",
                        "points": [
                            {"frame": "world", "position": [0.0, 0.0]},
                            {"frame": "link0", "position": [0.1, 0.0]},
                        ],
                    }
                ],
            },
        },
    )

    assert result.source_version == REPRESENTATION_SCHEMA_VERSION_V2
    assert result.target_version == REPRESENTATION_SCHEMA_VERSION
    assert result.migrated
    representation = RepresentationSpec.model_validate(result.payload)
    assert representation.muscle_path_geometry is not None
    assert len(representation.muscle_path_geometry.paths) == 1
    assert "frames" not in result.payload["muscle_path_geometry"]


def test_representation_v3_migrates_to_same_entity_frame_provider_v4() -> None:
    result = default_spec_registry.migrate(
        "RepresentationSpec",
        {
            "schema_id": REPRESENTATION_SCHEMA_ID,
            "schema_version": REPRESENTATION_SCHEMA_VERSION_V3,
            "elements": [{"id": "paths", "archetype": "muscle_path"}],
        },
    )

    assert result.source_version == REPRESENTATION_SCHEMA_VERSION_V3
    assert result.target_version == REPRESENTATION_SCHEMA_VERSION
    assert result.migrated
    representation = RepresentationSpec.model_validate(result.payload)
    assert representation.elements[0].frame_provider is None


def test_representation_v4_migrates_to_reference_pose_aware_v5() -> None:
    result = default_spec_registry.migrate(
        "RepresentationSpec",
        {
            "schema_id": REPRESENTATION_SCHEMA_ID,
            "schema_version": REPRESENTATION_SCHEMA_VERSION_V4,
            "elements": [
                {
                    "id": "links",
                    "archetype": "planar_chain",
                    "planar_chain": {"frame_ids": ["world", "link0", "link1"]},
                }
            ],
        },
    )

    assert result.source_version == REPRESENTATION_SCHEMA_VERSION_V4
    assert result.target_version == REPRESENTATION_SCHEMA_VERSION
    assert result.migrated
    representation = RepresentationSpec.model_validate(result.payload)
    assert representation.elements[0].planar_chain is not None
    assert representation.elements[0].planar_chain.reference_pose is None


@pytest.mark.parametrize(
    "source_version",
    [LEGACY_STUDIO_SCENARIO_SCHEMA_VERSION, STUDIO_SCENARIO_SCHEMA_VERSION_V1],
)
def test_studio_scenario_v1_migrates_empty_biomechanics_to_typed_v2(
    source_version: str,
) -> None:
    result = migrate_studio_scenario_spec(
        {
            "id": "scenario:train",
            "schema_version": source_version,
            "label": "Train",
            "biomechanics_spec": {},
        }
    )

    assert result.source_version == source_version
    assert result.target_version == STUDIO_SCENARIO_SCHEMA_VERSION
    assert result.migrated
    assert result.payload["biomechanics_spec"] == {
        "schema_version": STUDIO_BIOMECHANICS_SCHEMA_VERSION,
    }
    scenario = StudioScenarioSpec.model_validate(result.payload)
    assert scenario.biomechanics_spec is not None
    assert scenario.biomechanics_spec.schema_id == STUDIO_BIOMECHANICS_SCHEMA_ID


def test_studio_scenario_v1_does_not_silently_accept_unknown_biomechanics_fields() -> None:
    result = migrate_studio_scenario_spec(
        {
            "id": "scenario:train",
            "schema_version": STUDIO_SCENARIO_SCHEMA_VERSION_V1,
            "label": "Train",
            "biomechanics_spec": {"rest_pose": [0.0, 0.0]},
        }
    )

    with pytest.raises(ValueError, match="rest_pose"):
        StudioScenarioSpec.model_validate(result.payload)


def test_studio_biomechanics_rejects_unmigratable_old_versions() -> None:
    with pytest.raises(UnsupportedSpecVersion, match="studio.biomechanics.v0"):
        default_spec_registry.migrate(
            "StudioBiomechanicsSpec",
            {
                "schema_id": STUDIO_BIOMECHANICS_SCHEMA_ID,
                "schema_version": "feedbax.spec.studio.biomechanics.v0",
            },
        )


def test_studio_task_binding_entrypoint_migrates_v1_payload() -> None:
    result = migrate_studio_task_binding_spec(
        {
            "schema_version": STUDIO_TASK_BINDING_LEGACY_V1,
            "exposed_outputs": [
                {
                    "id": "inputs",
                    "label": "Inputs",
                    "kind": "signal",
                    "path": "inputs",
                    "bindable": True,
                    "metadata": {},
                }
            ],
            "bindings": [
                {
                    "id": "task:inputs->network:input",
                    "source_output_id": "inputs",
                    "target_node_id": "network",
                    "target_port": "input",
                    "role": "model_input",
                    "metadata": {},
                }
            ],
            "metadata": {},
        }
    )

    assert result.source_version == STUDIO_TASK_BINDING_LEGACY_V1
    assert result.target_version == "feedbax.spec.studio.task_bindings.v2"
    assert result.payload["schema_version"] == "feedbax.spec.studio.task_bindings.v2"
    assert result.payload["exposed_data"][0]["id"] == "inputs"
    assert "exposed_outputs" not in result.payload
    assert result.payload["bindings"][0]["source_data_id"] == "inputs"
    assert "source_output_id" not in result.payload["bindings"][0]
    assert [record.migration_id for record in result.migration_records] == [
        "studio-task-bindings-v1-to-v2"
    ]
    assert result.migration_records[0].metadata["spec_path"] == "task_binding_spec"


def test_studio_task_binding_entrypoint_rejects_explicit_unsupported_version() -> None:
    with pytest.raises(UnsupportedSpecVersion, match="task_bindings.v0"):
        migrate_studio_task_binding_spec({"schema_version": "feedbax.studio.task_bindings.v0"})


def test_fresh_studio_scenario_can_opt_into_current_versionless_specs() -> None:
    payload = {
        "id": "scenario:train",
        "label": "Train",
        "objective_spec": {
            "terms": [
                {
                    "selector": "task_data:targets",
                    "label": "Target tracking",
                }
            ]
        },
    }

    result = migrate_studio_scenario_spec(payload, assume_current=True)

    objective_spec = result.payload["objective_spec"]
    assert objective_spec == payload["objective_spec"]
    assert "schema_version" not in objective_spec


def test_studio_workspace_stamps_versionless_nested_scenario_specs() -> None:
    payload = {
        "id": "workspace:durable",
        "label": "Durable",
        "schema_version": default_spec_registry.resolve("StudioWorkspaceSpec").current_version,
        "scenarios": {
            "scenario:train": {
                "id": "scenario:train",
                "label": "Train",
            }
        },
    }

    result = migrate_studio_workspace_spec(payload)

    assert result.payload["scenarios"]["scenario:train"]["schema_version"] == (
        default_spec_registry.resolve("StudioScenarioSpec").current_version
    )
    assert "schema_version" not in payload["scenarios"]["scenario:train"]


def test_studio_workspace_stamps_versionless_scenario_owned_structured_specs() -> None:
    payload = {
        "id": "workspace:durable",
        "label": "Durable",
        "schema_version": default_spec_registry.resolve("StudioWorkspaceSpec").current_version,
        "scenarios": {
            "scenario:train": {
                "id": "scenario:train",
                "schema_version": default_spec_registry.resolve(
                    "StudioScenarioSpec"
                ).current_version,
                "label": "Train",
                "objective_spec": {
                    "terms": [
                        {
                            "selector": "task_data:targets",
                            "label": "Target tracking",
                        }
                    ]
                },
            }
        },
    }

    result = migrate_studio_workspace_spec(payload)

    objective_spec = result.payload["scenarios"]["scenario:train"]["objective_spec"]
    assert objective_spec["schema_version"] == (
        default_spec_registry.resolve("ObjectiveSpec").current_version
    )
    assert "schema_version" not in payload["scenarios"]["scenario:train"]["objective_spec"]


def test_studio_workspace_stamps_versionless_nested_stage_specs() -> None:
    payload = {
        "id": "workspace:durable",
        "label": "Durable",
        "schema_version": default_spec_registry.resolve("StudioWorkspaceSpec").current_version,
        "stages": [
            {
                "id": "stage:train",
                "kind": "train",
                "label": "Train",
            }
        ],
    }

    result = migrate_studio_workspace_spec(payload)

    assert result.payload["stages"][0]["schema_version"] == (
        default_spec_registry.resolve("StudioStageSpec").current_version
    )
    assert "schema_version" not in payload["stages"][0]


def test_fresh_studio_stage_can_opt_into_current_versionless_spec() -> None:
    payload = {
        "id": "stage:train",
        "kind": "train",
        "label": "Train",
    }

    result = migrate_studio_stage_spec(payload, assume_current=True)

    assert result.payload == payload
    assert result.source_version == default_spec_registry.resolve("StudioStageSpec").current_version


def test_objective_entrypoint_rejects_explicit_unsupported_version() -> None:
    with pytest.raises(UnsupportedSpecVersion, match="feedbax.objective.v0"):
        validate_objective_spec({"schema_version": "feedbax.objective.v0"})


def test_descriptor_schema_families_reject_old_versions() -> None:
    descriptor_versions = {
        "VariableDescriptor": VARIABLE_DESCRIPTOR_SCHEMA_VERSION,
        "ComponentDescriptor": COMPONENT_DESCRIPTOR_SCHEMA_VERSION,
        "DescriptorBasisIdentity": DESCRIPTOR_BASIS_SCHEMA_VERSION,
        "SelectorRoleIdentity": SELECTOR_ROLE_IDENTITY_SCHEMA_VERSION,
        "ComponentSelectorSyntax": COMPONENT_SELECTOR_SYNTAX_SCHEMA_VERSION,
        "SelectorFallbackPolicyIdentity": SELECTOR_FALLBACK_POLICY_SCHEMA_VERSION,
    }

    for kind, current_version in descriptor_versions.items():
        family = default_spec_registry.resolve(kind)
        assert family.policy is not None
        assert family.policy.stance == "reject"
        assert "tests/test_descriptor_schema.py" in family.policy.required_tests

        old_version = current_version.removesuffix(".v1") + ".v0"
        with pytest.raises(UnsupportedSpecVersion) as excinfo:
            default_spec_registry.migrate(kind, {"schema_version": old_version})

        message = str(excinfo.value)
        assert f"family='{kind}'" in message
        assert f"source_version='{old_version}'" in message
        assert "migration_intentionally_absent=yes" in message
