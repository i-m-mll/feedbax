"""Fixture-owned builders and clean-installed public contract cases."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from feedbax import LowererRegistration, OrderedLowererRegistry
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
    MATERIAL_DEPENDENCIES_SCHEMA_ID,
    MATERIAL_DEPENDENCIES_SCHEMA_VERSION,
    MaterialDependency,
    MaterialDependencyObservation,
    MaterialDependencySet,
    ValueIdentityRecord,
    authored_value_sha256,
    semantic_value_sha256,
    value_identity_record,
)
from feedbax.contracts.manifest import ParentRef
from feedbax.testing import check_material_dependency_contract


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


def _component_registry(*, migration_first: bool) -> ComponentRegistry:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
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


__all__ = [
    "check_component_registration_and_migration",
    "check_exact_parent_migration",
    "check_material_dependencies",
    "check_ordered_registration",
    "check_value_identity",
]
