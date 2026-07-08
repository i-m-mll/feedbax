from __future__ import annotations

import pytest

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
    LEGACY_GRAPH_SPEC_SCHEMA_VERSION,
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
from feedbax.contracts.manifest import EVALUATION_STATES_CONTAINER_SCHEMA_VERSION
from feedbax.contracts.studio_api import (
    STUDIO_API_TRANSPORT_SCHEMA_ID,
    STUDIO_API_TRANSPORT_SCHEMA_VERSION,
)
from feedbax.contracts.representation import (
    REPRESENTATION_SCHEMA_ID,
    REPRESENTATION_SCHEMA_VERSION,
    REPRESENTATION_SCHEMA_VERSION_V0,
)
from feedbax.contracts.workspace_replay import (
    WORKSPACE_REPLAY_SCHEMA_ID,
    WORKSPACE_REPLAY_SCHEMA_VERSION,
)
from feedbax.contracts.checkpoints import (
    TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V1,
    TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V2,
)
from feedbax.contracts.value_schema import ValueSchema
from feedbax.execution.models import (
    EXECUTION_CLOUD_PAYLOAD_SCHEMA_ID,
    EXECUTION_CLOUD_PAYLOAD_SCHEMA_VERSION,
    EXECUTION_REPRODUCIBILITY_SCHEMA_ID,
    EXECUTION_REPRODUCIBILITY_SCHEMA_VERSION,
)
from feedbax.objectives.spec import validate_objective_spec

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
    assert [record.migration_id for record in result.migration_records] == [
        "demo-spec-v1-to-v2"
    ]
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
    assert families["AnalysisBundleSpec"].current_version == "feedbax.spec.analysis_bundle.v2"
    assert families["PathExpression"].identity == PATH_EXPRESSION_SCHEMA_ID
    assert families["PathExpression"].current_version == PATH_EXPRESSION_SCHEMA_VERSION
    assert families["ExtractionProductSpec"].identity == EXTRACTION_PRODUCT_SPEC_SCHEMA_ID
    assert (
        families["ExtractionProductSpec"].current_version
        == EXTRACTION_PRODUCT_SPEC_SCHEMA_VERSION
    )
    assert families["ReportSpec"].identity == "feedbax.spec.report"
    assert families["ReportSpec"].current_version == "feedbax.spec.report.v1"
    assert (
        families["AnalysisDataProductRequirement"].identity
        == "feedbax.spec.analysis_data_product_requirement"
    )
    assert (
        families["AnalysisDataProduct"].identity
        == "feedbax.manifest.analysis_data_product"
    )
    assert families["ExecutionSpec"].identity == "feedbax.spec.execution"
    assert families["ExecutionSpec"].current_version == "feedbax.spec.execution.v2"
    assert families["ExecutionPlan"].identity == "feedbax.manifest.execution_plan"
    assert families["ExecutionPlan"].current_version == "feedbax.manifest.execution.v3"
    assert families["ExecutionCloudPayload"].identity == EXECUTION_CLOUD_PAYLOAD_SCHEMA_ID
    assert (
        families["ExecutionCloudPayload"].current_version
        == EXECUTION_CLOUD_PAYLOAD_SCHEMA_VERSION
    )
    assert (
        families["ExecutionReproducibility"].identity
        == EXECUTION_REPRODUCIBILITY_SCHEMA_ID
    )
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
        families["TrainingRunManifest"].identity
        == "feedbax.manifest.training_run"
    )
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
    assert {families[kind].namespace for kind in manifest_kinds} == {
        SchemaNamespaceKind.MANIFEST
    }
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
    assert representation_policy.stance == "reject"
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
    assert lr_schedule_policy.stance == "reject"
    assert lr_schedule_policy.rejected_old_versions == ("feedbax.spec.training.lr_schedule.v0",)
    assert checkpoint_policy is not None
    assert checkpoint_policy.stance == "migrate"
    assert checkpoint_policy.supported_old_versions == (
        TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V1,
        TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V2,
    )
    assert execution_policy is not None
    assert execution_policy.rejected_old_versions == ("feedbax.spec.execution.v1",)
    assert report_policy is not None
    assert report_policy.stance == "reject"
    assert report_policy.rejected_old_versions == ("feedbax.spec.report.v0",)
    assert extraction_policy is not None
    assert extraction_policy.stance == "reject"
    assert extraction_policy.rejected_old_versions == (
        "feedbax.spec.extraction_product.v0",
    )
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
    assert studio_api_policy.rejected_old_versions == (
        "feedbax.spec.studio.api_transport.v0",
    )
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
