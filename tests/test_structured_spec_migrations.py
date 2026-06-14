from __future__ import annotations

import pytest

from feedbax.migrations import (
    SchemaMigration,
    STUDIO_TASK_BINDING_LEGACY_V1,
    SpecSchemaFamily,
    SpecSchemaRegistry,
    UnknownSpecFamily,
    UnsupportedSpecVersion,
    default_spec_registry,
    migrate_studio_task_binding_spec,
)
from feedbax.schema_namespace import SchemaNamespaceError, SchemaNamespaceKind
from feedbax.contracts.graph import GRAPH_SPEC_SCHEMA_VERSION, LEGACY_GRAPH_SPEC_SCHEMA_VERSION
from feedbax.objective_spec import validate_objective_spec


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
    assert families["TrainingSpec"].identity == "feedbax.spec.training"
    assert families["ProviderManifest"].current_version == "feedbax.manifest.v1"
    assert families["ModelArtifactManifest"].identity == "feedbax.manifest.model_artifact"
    assert families["SpecPayload"].identity == "feedbax.manifest.spec_payload"
    assert families["SpecPayload"].namespace == SchemaNamespaceKind.MANIFEST
    assert not families["RegistryEntry"].durable
    assert not families["StudioSchemaRegistry"].durable


def test_default_registry_enforces_spec_and_manifest_namespace_categories() -> None:
    spec_kinds = {
        "GraphSpec",
        "PopulationStructureSpec",
        "TrainingSpec",
        "TaskSpec",
        "ObjectiveSpec",
        "EvaluationRunSpec",
        "AnalysisRunSpec",
        "ReportSpec",
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
        "ExecutionPlan",
        "LocalExecutionResult",
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
    from feedbax.provider import provider_manifest

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

    assert graph_policy is not None
    assert graph_policy.stance == "migrate"
    assert graph_policy.supported_old_versions == (LEGACY_GRAPH_SPEC_SCHEMA_VERSION,)
    assert task_binding_policy is not None
    assert task_binding_policy.stance == "migrate"
    assert task_binding_policy.supported_old_versions == (STUDIO_TASK_BINDING_LEGACY_V1,)
    assert task_binding_policy.rejected_old_versions == ("feedbax.studio.task_bindings.v0",)
    assert objective_policy is not None
    assert objective_policy.stance == "reject"
    assert objective_policy.rejected_old_versions == ("feedbax.objective.v0",)
    assert population_policy is not None
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
