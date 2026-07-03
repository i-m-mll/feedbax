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
    migrate_studio_task_binding_spec,
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
    LEGACY_GRAPH_SPEC_SCHEMA_VERSION,
)
from feedbax.contracts.manifest import EVALUATION_STATES_CONTAINER_SCHEMA_VERSION
from feedbax.contracts.value_schema import ValueSchema
from feedbax.objectives.spec import validate_objective_spec


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


def test_structured_spec_registry_treats_versionless_payload_as_current() -> None:
    registry = _registry()
    payload = {"value": 3}

    result = registry.migrate("DemoSpec", payload)

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
    assert families["TrainingRunSpec"].current_version == "feedbax.spec.training_run.v1"
    assert families["TrainingSpec"].identity == "feedbax.spec.training"
    assert (
        families["StandardSupervisedMethodPayload"].identity
        == "feedbax.spec.training_method.standard_supervised_payload"
    )
    assert families["AnalysisBundleSpec"].identity == "feedbax.spec.analysis_bundle"
    assert families["AnalysisBundleSpec"].current_version == "feedbax.spec.analysis_bundle.v2"
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
    assert families["LocalExecutionResult"].identity == "feedbax.manifest.local_execution_result"
    assert families["LocalExecutionResult"].current_version == "feedbax.manifest.execution.v3"
    assert (
        families["StagedAnalysisBundleExecution"].identity
        == "feedbax.manifest.analysis_bundle_execution"
    )
    assert families["ValueSchema"].identity == "feedbax.spec.studio.schema.value"
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
        "StudioPipelineMaterializationResult": (
            "feedbax.manifest.studio.pipeline_materialization_result"
        ),
        "StudioSchemaRegistry": "feedbax.manifest.studio.schema_registry",
        "StudioTrainingLocalRunResult": "feedbax.manifest.studio.training_local_run_result",
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
        "ReportSpec",
        "RegenerationSpec",
        "ExecutionSpec",
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
        "LocalExecutionResult",
        "StagedAnalysisBundleExecution",
        "ProviderManifest",
        "RegistrySnapshot",
        "StudioPipelineMaterializationResult",
    }

    families = {family.kind: family for family in default_spec_registry.families()}

    assert {families[kind].namespace for kind in spec_kinds} == {SchemaNamespaceKind.SPEC}
    assert {families[kind].namespace for kind in manifest_kinds} == {
        SchemaNamespaceKind.MANIFEST
    }
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
                migrated = default_spec_registry.migrate(
                    family.kind,
                    {"schema_version": old_version},
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
    local_execution_result_policy = default_spec_registry.resolve("LocalExecutionResult").policy
    report_policy = default_spec_registry.resolve("ReportSpec").policy

    assert graph_policy is not None
    assert graph_policy.stance == "migrate"
    assert graph_policy.supported_old_versions == (
        LEGACY_GRAPH_SPEC_SCHEMA_VERSION,
        GRAPH_SPEC_SCHEMA_VERSION_V2,
    )
    assert task_binding_policy is not None
    assert task_binding_policy.stance == "migrate"
    assert task_binding_policy.supported_old_versions == (STUDIO_TASK_BINDING_LEGACY_V1,)
    assert task_binding_policy.rejected_old_versions == ("feedbax.studio.task_bindings.v0",)
    assert objective_policy is not None
    assert objective_policy.stance == "reject"
    assert objective_policy.rejected_old_versions == ("feedbax.objective.v0",)
    assert population_policy is not None
    assert execution_policy is not None
    assert execution_policy.rejected_old_versions == ("feedbax.spec.execution.v1",)
    assert report_policy is not None
    assert report_policy.stance == "reject"
    assert report_policy.rejected_old_versions == ("feedbax.spec.report.v0",)
    assert execution_plan_policy is not None
    assert execution_plan_policy.rejected_old_versions == (
        "feedbax.manifest.execution.v2",
        "feedbax.manifest.execution.v1",
    )
    assert local_execution_result_policy is not None
    assert local_execution_result_policy.rejected_old_versions == (
        "feedbax.manifest.execution.v2",
        "feedbax.manifest.execution.v1",
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
