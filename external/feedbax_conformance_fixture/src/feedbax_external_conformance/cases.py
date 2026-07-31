"""Fixture-owned builders and clean-installed public contract cases."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
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
    project_evaluation_rows,
    resolve_analysis_inputs,
)
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
)
from feedbax.contracts.evaluation_states import store_evaluation_states_artifact
from feedbax.plugins import (
    COMPONENTS,
    DRIVERS,
    BootstrapError,
    BootstrapErrorCode,
    FamilyRequirement,
    PluginDeclaration,
    PluginDependency,
    PluginRegistration,
    RegistrationContext,
    bootstrap_application,
    discover_plugin_registrations,
)
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
    new_fixture_registration_context,
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
    )


def check_unified_plugin_bootstrap(*, entry_points: Iterable[object] | None = None) -> bool:
    """Prove installed typed discovery and the transactional generic-family contract."""

    sources = discover_plugin_registrations(entry_points=entry_points)
    expected_plugins = (
        "feedbax_external_conformance.foundation",
        "feedbax_external_conformance.dependent",
    )
    if tuple(sorted(source.registration.declaration.plugin_id for source in sources)) != tuple(
        sorted(expected_plugins)
    ):
        raise AssertionError("installed feedbax.plugins discovery inventory drifted")

    states = []
    for shuffled in (sources, tuple(reversed(sources))):
        states.append(
            asyncio.run(
                bootstrap_application(
                    new_fixture_registration_context(),
                    registrations=shuffled,
                )
            )
        )
    if entry_points is None:
        states.append(asyncio.run(bootstrap_application(new_fixture_registration_context())))

    for state in states:
        registry = state.registry(FIXTURE_RECORDS)
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
            },
            {FIXTURE_RECORDS.family: ("dependent",)},
        ):
            raise AssertionError("plugin provenance registered-key attribution drifted")
        expected_family_protocols = (
            {
                COMPONENTS.family: "1",
                DRIVERS.family: "1",
                FIXTURE_RECORDS.family: "1",
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

    if any(
        left.registry(FIXTURE_RECORDS) is right.registry(FIXTURE_RECORDS)
        for index, left in enumerate(states)
        for right in states[index + 1 :]
    ):
        raise AssertionError("fresh bootstrap contexts shared an external registry")

    retained: list[FixtureRecordRegistry] = []

    def fail_after_partial(context: RegistrationContext) -> None:
        context.registry(FIXTURE_RECORDS).register("partial")
        raise RuntimeError("fixture failure")

    failure_context = new_fixture_registration_context(registry_sink=retained)
    try:
        asyncio.run(
            bootstrap_application(
                failure_context,
                registrations=(_fixture_plugin("fixture.failure", fail_after_partial),),
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
    empty = asyncio.run(bootstrap_application(new_fixture_registration_context(), registrations=()))
    if empty.registry(FIXTURE_RECORDS).keys():
        raise AssertionError("failed registration contaminated an isolated context")

    first = _fixture_plugin(
        "fixture.conflict.first",
        lambda context: context.registry(FIXTURE_RECORDS).register("collision"),
    )
    second = _fixture_plugin(
        "fixture.conflict.second",
        lambda context: context.registry(FIXTURE_RECORDS).register("collision"),
        dependencies=(PluginDependency("fixture.conflict.first", "1"),),
    )
    try:
        asyncio.run(
            bootstrap_application(new_fixture_registration_context(), registrations=(second, first))
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
    )
    try:
        asyncio.run(
            bootstrap_application(new_fixture_registration_context(), registrations=(missing,))
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

    registrations = (
        discover_plugin_registrations(entry_points=entry_points)
        if entry_points is not None
        else None
    )
    state = asyncio.run(
        bootstrap_application(
            new_fixture_registration_context(),
            registrations=registrations,
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
    state = asyncio.run(bootstrap_application(new_fixture_registration_context()))
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
            bootstrap_application(new_fixture_registration_context(), registrations=())
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


__all__ = [
    "check_component_registration_and_migration",
    "check_component_param_array_values",
    "check_exact_parent_migration",
    "check_material_dependencies",
    "check_ordered_registration",
    "check_resolved_evaluation_row_projection",
    "check_value_identity",
]
