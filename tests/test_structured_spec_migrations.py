from __future__ import annotations

import pytest

from feedbax.contracts.artifact_custody import (
    CONTENT_LEASE_RECORD_SCHEMA_ID,
    CONTENT_LEASE_RECORD_SCHEMA_VERSION_V1,
    CONTENT_RETENTION_INDEX_SCHEMA_ID,
    CONTENT_RETENTION_INDEX_SCHEMA_VERSION_V1,
    ContentLeaseRecord,
    ContentRetentionIndex,
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
    STUDIO_BIOMECHANICS_SCHEMA_ID,
)
from feedbax.contracts.representation import (
    REPRESENTATION_SCHEMA_ID,
    REPRESENTATION_SCHEMA_VERSION,
    REPRESENTATION_SCHEMA_VERSION_V4,
    REPRESENTATION_SCHEMA_VERSION_V3,
    REPRESENTATION_SCHEMA_VERSION_V2,
    REPRESENTATION_SCHEMA_VERSION_V1,
    RepresentationSpec,
)
from feedbax.contracts.value_schema import ValueSchema

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
            record_metadata=lambda source: {
                "source_old": source["old"],
                "source_had_renamed": "renamed" in source,
            },
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
    assert result.migration_records[0].metadata == {
        "description": "Rename old to renamed.",
        "source_old": 7,
        "source_had_renamed": False,
    }


def test_schema_migration_callbacks_receive_independent_nested_source_copies() -> None:
    source = {"schema_version": "demo.v1", "nested": {"values": [1]}}

    def record_metadata(payload):
        payload["nested"]["values"].append(2)
        return {"recorded_values": payload["nested"]["values"]}

    def migrate(payload):
        assert payload["nested"]["values"] == [1]
        payload["nested"]["values"].append(3)
        return payload

    migrated, record = SchemaMigration(
        source_version="demo.v1",
        target_version="demo.v2",
        migration_id="isolated-callbacks",
        migrate=migrate,
        record_metadata=record_metadata,
    ).apply(source)

    assert source["nested"]["values"] == [1]
    assert migrated["nested"]["values"] == [1, 3]
    assert record.metadata["recorded_values"] == [1, 2]


def test_schema_migration_record_metadata_cannot_overwrite_reserved_keys() -> None:
    migration = SchemaMigration(
        source_version="demo.v1",
        target_version="demo.v2",
        migration_id="reserved-record-metadata",
        migrate=lambda payload: payload,
        description="Authoritative description.",
        record_metadata=lambda _source: {"description": "replacement"},
    )

    with pytest.raises(ValueError, match="reserved metadata keys.*description"):
        migration.apply({"schema_version": "demo.v1"})


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


def test_retention_v1_migration_preserves_expiry_as_fail_closed() -> None:
    digest = "a" * 64
    lease = {
        "schema_id": CONTENT_LEASE_RECORD_SCHEMA_ID,
        "schema_version": CONTENT_LEASE_RECORD_SCHEMA_VERSION_V1,
        "lease_id": "legacy",
        "owner": "worker",
        "expires_at": "2026-09-15T00:00:00Z",
        "targets": [
            {
                "ref_kind": "object",
                "object": {
                    "object_id": f"content-object://sha256/{digest}",
                    "sha256": digest,
                    "size_bytes": 1,
                },
            }
        ],
    }
    migrated_lease = default_spec_registry.migrate("ContentLeaseRecord", lease).payload
    assert ContentLeaseRecord.model_validate(migrated_lease).expiry_disposition == "block"

    migrated_index = default_spec_registry.migrate(
        "ContentRetentionIndex",
        {
            "schema_id": CONTENT_RETENTION_INDEX_SCHEMA_ID,
            "schema_version": CONTENT_RETENTION_INDEX_SCHEMA_VERSION_V1,
            "generation": 0,
            "leases": [lease],
        },
    ).payload
    assert (
        ContentRetentionIndex.model_validate(migrated_index).leases[0].expiry_disposition == "block"
    )


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


def test_manifest_schema_identities_survive_contract_package_move() -> None:
    families = {family.kind: family for family in default_spec_registry.families()}

    expected_manifest_identities = {
        "ArrayRecord": "feedbax.manifest.array_record",
        "ArrayStorePayload": "feedbax.manifest.array_store",
        "GraphSpecManifest": "feedbax.manifest.graph_spec",
        "ModelArtifactManifest": "feedbax.manifest.model_artifact",
        "SpecPayload": "feedbax.manifest.spec_payload",
        "TrainingRunManifest": "feedbax.manifest.training_run",
        "EvaluationRunManifest": "feedbax.manifest.evaluation_run",
    }

    for kind, identity in expected_manifest_identities.items():
        assert families[kind].identity == identity
        assert families[kind].namespace == SchemaNamespaceKind.MANIFEST
        assert families[kind].current_version.startswith("feedbax.manifest.")


def test_contract_value_schema_round_trips_without_payload_shape_change() -> None:
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


def test_default_registry_enforces_spec_and_manifest_namespace_categories() -> None:
    spec_kinds = {
        "GraphSpec",
        "PopulationStructureSpec",
        "VariableDescriptor",
        "ComponentDescriptor",
        "DescriptorBasisIdentity",
        "EvaluationRunSpec",
        "PathExpression",
        "StudioApiTransport",
        "StudioTaskBindingSpec",
    }
    manifest_kinds = {
        "SpecPayload",
        "GraphSpecManifest",
        "ModelArtifactManifest",
        "ArrayStorePayload",
        "ArrayRecord",
        "TrainingRunManifest",
    }

    families = {family.kind: family for family in default_spec_registry.families()}

    assert {families[kind].namespace for kind in spec_kinds} == {SchemaNamespaceKind.SPEC}
    assert {families[kind].namespace for kind in manifest_kinds} == {SchemaNamespaceKind.MANIFEST}
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
    assert policy.emitted_by == ("feedbax.persistence.artifact_custody",)
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
    assert result.target_version == "feedbax.spec.studio.task_bindings.v3"
    assert result.payload["schema_version"] == "feedbax.spec.studio.task_bindings.v3"
    assert result.payload["exposed_data"][0]["id"] == "inputs"
    assert "exposed_outputs" not in result.payload
    assert result.payload["bindings"][0]["source_data_id"] == "inputs"
    assert "source_output_id" not in result.payload["bindings"][0]
    assert [record.migration_id for record in result.migration_records] == [
        "studio-task-bindings-v1-to-v2",
        "studio-task-bindings-v2-to-v3",
    ]
    assert result.migration_records[0].metadata["spec_path"] == "task_binding_spec"


@pytest.mark.parametrize("binding_role", ["", "component_parameter"])
@pytest.mark.parametrize("temporality", ["constant", "trajectory"])
def test_v2_task_binding_migrates_only_direct_input_routes(temporality, binding_role) -> None:
    payload = {
        "schema_version": "feedbax.spec.studio.task_bindings.v2",
        "exposed_data": [
            {
                "id": "gain",
                "label": "Gain",
                "kind": "intervention",
                "role": "component_parameter",
                "path": "intervene.visible.gain",
                "bindable": True,
                "metadata": {"temporality": temporality},
            }
        ],
        "bindings": [
            {
                "id": "task:gain->visible:gain",
                "source_data_id": "gain",
                "target_node_id": "visible",
                "target_port": "gain",
                "role": binding_role,
                "metadata": {"task_parameter_label": "visible.gain"},
            }
        ],
    }
    if temporality == "constant":
        with pytest.raises(ValueError, match="original producer"):
            migrate_studio_task_binding_spec(payload)
        return

    result = migrate_studio_task_binding_spec(payload)
    (data,) = result.payload["exposed_data"]
    (binding,) = result.payload["bindings"]
    assert data["path"] == "inputs.gain"
    assert binding["role"] == data["role"] == "graph_input"
    assert (binding["target_node_id"], binding["target_port"]) == ("visible", "gain")
    assert binding["metadata"] == {}


def test_studio_task_binding_entrypoint_rejects_explicit_unsupported_version() -> None:
    with pytest.raises(UnsupportedSpecVersion, match="task_bindings.v0"):
        migrate_studio_task_binding_spec({"schema_version": "feedbax.studio.task_bindings.v0"})


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
