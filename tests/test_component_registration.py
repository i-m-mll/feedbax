from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from feedbax.component_registry import (
    ComponentRegistry,
    ComponentMigration,
    ComponentMigrationPack,
    get_component_registry,
    register_cde_templates,
    register_component_type,
)
from feedbax.runtime.channel import Channel
from feedbax.runtime.components import Gain
from feedbax.contracts.graph import ComponentSpec, GraphSpec, WireSpec
from feedbax.contracts.graphs.builders import build_component
from feedbax.models.cde import CDENetwork
from feedbax.runtime.graph import Component
from feedbax.contracts.graphs.serialization import spec_to_graph
from feedbax.contracts.representation import REPRESENTATION_SCHEMA_VERSION, RepresentationSpec


class _PrototypeSource(Component):
    input_ports = ()
    output_ports = ("signal",)

    def __call__(self, inputs, state, *, key):
        return {"signal": jnp.ones((3,))}, state


def _single_node_spec(
    component_type: str,
    params: dict[str, Any] | None = None,
    *,
    param_schema_version: str | None = None,
) -> GraphSpec:
    return GraphSpec(
        nodes={
            "component": ComponentSpec(
                type=component_type,
                params=params or {},
                param_schema_version=param_schema_version,
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        input_ports=["input"],
        output_ports=["output"],
        input_bindings={"input": ("component", "input")},
        output_bindings={"output": ("component", "output")},
    )


def test_programmatic_component_registration_materializes_via_spec_to_graph() -> None:
    register_component_type(
        "TestProgrammaticGain",
        lambda params: Gain(gain=float(params["gain"])),
        category="Test",
        description="Test-only gain.",
        param_schema=[{"name": "gain", "type": "float", "default": 3.0}],
        input_ports=["input"],
        output_ports=["output"],
        provenance="test-suite",
    )

    registry = get_component_registry()
    meta = registry.get("TestProgrammaticGain")
    assert meta is not None
    assert meta.default_params == {"gain": 3.0}
    assert meta.input_ports == ["input"]
    assert meta.output_ports == ["output"]
    assert meta.provenance == "test-suite"

    definition = next(item for item in registry.list_all() if item.name == "TestProgrammaticGain")
    assert definition.default_params == {"gain": 3.0}
    assert definition.input_ports == ["input"]
    assert definition.output_ports == ["output"]
    assert definition.provenance == "test-suite"

    graph = spec_to_graph(_single_node_spec("TestProgrammaticGain", {"gain": 4.0}))

    component = graph.nodes["component"]
    assert isinstance(component, Gain)
    assert component.gain == 4.0

    default_graph = spec_to_graph(_single_node_spec("TestProgrammaticGain"))
    default_component = default_graph.nodes["component"]
    assert isinstance(default_component, Gain)
    assert default_component.gain == 3.0


def test_registered_component_output_prototype_feeds_stateful_materialization() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    registry.register_component_type(
        "TestPrototypeSource",
        lambda params: _PrototypeSource(),
        category="Test",
        description="Source with registry-owned output prototype.",
        param_schema=[{"name": "width", "type": "int", "default": 3}],
        input_ports=[],
        output_ports=["signal"],
        output_prototype_fn=lambda params, inputs: {
            "signal": jnp.zeros((int(params.get("width", 3)),))
        },
    )
    spec = GraphSpec(
        nodes={
            "source": ComponentSpec(
                type="TestPrototypeSource",
                params={},
                input_ports=[],
                output_ports=["signal"],
            ),
            "delay": ComponentSpec(
                type="Channel",
                params={"delay": 1, "add_noise": False},
                input_ports=["input"],
                output_ports=["output"],
            ),
        },
        wires=[
            WireSpec(
                source_node="source",
                source_port="signal",
                target_node="delay",
                target_port="input",
            )
        ],
        output_ports=["output"],
        output_bindings={"output": ("delay", "output")},
    )

    graph = spec_to_graph(spec, registry)

    assert isinstance(graph.nodes["delay"], Channel)
    assert graph.nodes["delay"].input_proto.shape == (3,)


def test_feedbax_component_meta_rejects_output_prototype_mutation() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    meta = registry.get("FixedField")
    assert meta is not None
    assert meta.provenance == "feedbax"
    original = meta.output_prototype_fn

    with pytest.raises(AttributeError, match="Feedbax-owned ComponentMeta.output_prototype_fn"):
        meta.output_prototype_fn = lambda params, inputs: {"force": inputs["force"]}

    assert meta.output_prototype_fn is original


def test_builtin_registry_has_explicit_builders_and_consistent_port_metadata() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)

    for name in registry.names():
        meta = registry.get(name)
        assert meta is not None
        assert callable(meta.builder), f"{name} has no explicit builder contract"
        if meta.port_types is not None:
            assert set(meta.port_types.inputs) == set(meta.input_ports), name
            assert set(meta.port_types.outputs) == set(meta.output_ports), name


def test_unsupported_builtin_builder_contract_fails_clearly() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    meta = registry.get("MomentArmProjection")
    assert meta is not None
    assert callable(meta.builder)
    assert getattr(meta.builder, "_feedbax_unsupported_builder", False)
    assert "MomentArmProjection" not in registry.executable_names()

    with pytest.raises(NotImplementedError, match="display-only abstraction"):
        build_component("projection", "MomentArmProjection", {}, component_registry=registry)


def test_entry_point_component_registration_records_package_provenance() -> None:
    def registrar(component_registry: ComponentRegistry) -> None:
        component_registry.register_component_type(
            "TestEntryPointGain",
            lambda params: Gain(gain=float(params.get("gain", 1.0))),
            category="Test",
            param_schema=[{"name": "gain", "type": "float", "default": 2.0}],
            input_ports=["input"],
            output_ports=["output"],
        )

    class FakeDist:
        metadata = {"Name": "feedbax-test-plugin"}

    class FakeEntryPoint:
        name = "feedbax_test_plugin"
        dist = FakeDist()

        def load(self) -> Callable[[ComponentRegistry], None]:
            return registrar

    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    registry.discover_entry_point_components(entry_points=[FakeEntryPoint()])

    meta = registry.get("TestEntryPointGain")
    assert meta is not None
    assert meta.provenance == "package:feedbax-test-plugin"

    definition = next(item for item in registry.list_all() if item.name == "TestEntryPointGain")
    assert definition.default_params == {"gain": 2.0}
    assert definition.input_ports == ["input"]
    assert definition.output_ports == ["output"]
    assert definition.provenance == "package:feedbax-test-plugin"

    graph = spec_to_graph(_single_node_spec("TestEntryPointGain", {"gain": 5.0}), registry)
    assert isinstance(graph.nodes["component"], Gain)
    assert graph.nodes["component"].gain == 5.0


def test_user_component_file_registers_palette_metadata_and_builder(
    tmp_path: Path,
) -> None:
    component_file = tmp_path / "custom_gain.py"
    component_file.write_text(
        textwrap.dedent(
            """
            from feedbax.runtime.components import Gain


            class UserGain(Gain):
                pass


            UserGain._feedbax_component_meta = {
                "name": "TestUserFileGain",
                "category": "User",
                "description": "Loaded from a user component file.",
                "param_schema": [{"name": "gain", "type": "float", "default": 6.0}],
                "input_ports": ["input"],
                "output_ports": ["output"],
            }
            """
        )
    )

    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    registry.load_user_components(tmp_path)

    definition = next(item for item in registry.list_all() if item.name == "TestUserFileGain")
    assert definition.default_params == {"gain": 6.0}
    assert definition.input_ports == ["input"]
    assert definition.output_ports == ["output"]
    assert definition.provenance == f"file:{component_file}"

    graph = spec_to_graph(_single_node_spec("TestUserFileGain", {"gain": 7.0}), registry)
    assert isinstance(graph.nodes["component"], Gain)
    assert graph.nodes["component"].gain == 7.0


def test_unknown_component_error_names_type_and_known_registry_contents() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)

    with pytest.raises(ValueError) as exc_info:
        spec_to_graph(_single_node_spec("DefinitelyUnknownComponent"), registry)

    message = str(exc_info.value)
    assert "DefinitelyUnknownComponent" in message
    assert "Known component types:" in message
    assert "Gain" in message


def test_cde_primitives_build_and_execute_with_expected_semantics() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    state_key = jax.random.PRNGKey(0)

    input_node = build_component(
        "obs_in",
        "Input",
        {"output_port": "obs"},
        component_registry=registry,
    )
    outputs, _ = input_node(
        {"input": jnp.array([1.0, 2.0])},
        input_node.init_state(key=state_key),
        key=state_key,
    )
    assert jnp.array_equal(outputs["obs"], jnp.array([1.0, 2.0]))

    subtract = build_component("subtract", "Subtract", {}, component_registry=registry)
    outputs, _ = subtract(
        {"a": jnp.array([3.0, 5.0]), "b": jnp.array([1.0, 2.0])},
        subtract.init_state(key=state_key),
        key=state_key,
    )
    assert jnp.array_equal(outputs["out"], jnp.array([2.0, 3.0]))

    reshape = build_component("reshape", "Reshape", {"shape": [2, 2]}, component_registry=registry)
    outputs, _ = reshape(
        {"input": jnp.arange(4.0)},
        reshape.init_state(key=state_key),
        key=state_key,
    )
    assert outputs["output"].shape == (2, 2)
    assert jnp.array_equal(outputs["output"], jnp.array([[0.0, 1.0], [2.0, 3.0]]))

    matmul = build_component("matmul", "MatMul", {}, component_registry=registry)
    outputs, _ = matmul(
        {"a": jnp.array([[1.0, 2.0], [3.0, 4.0]]), "b": jnp.array([0.5, 1.0])},
        matmul.init_state(key=state_key),
        key=state_key,
    )
    assert jnp.array_equal(outputs["out"], jnp.array([2.5, 5.5]))

    scale = build_component("scale", "Scale", {"scale": -0.5}, component_registry=registry)
    outputs, _ = scale(
        {"input": jnp.array([2.0, -4.0])},
        scale.init_state(key=state_key),
        key=state_key,
    )
    assert jnp.array_equal(outputs["output"], jnp.array([-1.0, 2.0]))

    sigmoid = build_component("sigmoid", "Sigmoid", {}, component_registry=registry)
    outputs, _ = sigmoid(
        {"input": jnp.array([-jnp.inf, 0.0, jnp.inf])},
        sigmoid.init_state(key=state_key),
        key=state_key,
    )
    assert jnp.array_equal(outputs["output"], jnp.array([0.0, 0.5, 1.0]))


def test_cde_templates_are_not_builtin_defaults() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)

    assert registry.get("CDE Standard") is None


def test_caller_registered_cde_templates_are_executable_and_run_one_step() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    registry.register_template_pack(register_cde_templates, provenance="test-cde-pack")

    for template_name in ("CDE Standard", "CDE + Decay", "CDE + Anti-NF", "CDE Hybrid v9b"):
        meta = registry.get(template_name)
        assert meta is not None
        assert meta.provenance == "test-cde-pack"
        assert meta.template_kind == "executable"
        assert meta.template_id is not None
        assert meta.template_id.startswith("feedbax.templates.cde_")
        assert registry.template_builder_issues(meta) == []

        graph = spec_to_graph(meta.template_graph, registry)
        state = graph.init_state(key=jax.random.PRNGKey(0))
        outputs, _ = graph(
            {
                "obs": jnp.array([0.2]),
                "obs_prev": jnp.array([0.1]),
                "h_prev": jnp.array([0.3]),
            },
            state,
            key=jax.random.PRNGKey(1),
        )
        assert outputs["h_new"].shape == (1,)
        assert outputs["action"].shape == (1,)


def test_executable_builtin_templates_have_complete_builders() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    for template_name in ("Recurrent Controller", "Simple Feedback Loop"):
        meta = registry.get(template_name)
        assert meta is not None
        assert registry.template_builder_issues(meta) == []


def test_cde_standard_template_matches_standard_cde_network_step() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    registry.register_template_pack(register_cde_templates, provenance="test-cde-pack")
    meta = registry.get("CDE Standard")
    assert meta is not None
    graph = spec_to_graph(meta.template_graph, registry)
    net = CDENetwork(
        obs_dim=1,
        hidden_dim=1,
        out_size=1,
        vf_width=128,
        vf_depth=1,
        decay=0.0,
        use_anti_nf=False,
        key=jax.random.PRNGKey(42),
    )
    graph = eqx.tree_at(lambda g: g.nodes["vf"].linears, graph, net.vector_field.layers)
    graph = eqx.tree_at(lambda g: g.nodes["linear"].layer, graph, net.readout)

    net_state = net.init_state(key=jax.random.PRNGKey(0))
    cde_state = net_state.get(net.state_index)
    graph_state = graph.init_state(key=jax.random.PRNGKey(0))
    obs = jnp.array([0.25])

    net_outputs, _ = net({"input": obs}, net_state, key=jax.random.PRNGKey(1))
    graph_outputs, _ = graph(
        {"obs": obs, "obs_prev": cde_state.obs_prev, "h_prev": cde_state.hidden},
        graph_state,
        key=jax.random.PRNGKey(1),
    )

    assert jnp.allclose(graph_outputs["h_new"], net_outputs["hidden"])
    assert jnp.allclose(graph_outputs["action"], net_outputs["output"])


def test_builtin_component_rename_migration_materializes_registered_target() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    registry.register_migration(
        ComponentMigration(
            source_type="LegacyGain",
            target_type="Gain",
            owner="feedbax",
            migration_id="feedbax.component.LegacyGain-to-Gain.v1",
            target_param_schema_version="1",
            description="Test-only built-in rename edge.",
        )
    )

    graph = spec_to_graph(_single_node_spec("LegacyGain", {"gain": 4.0}), registry)

    component = graph.nodes["component"]
    assert isinstance(component, Gain)
    assert component.gain == 4.0
    gain = next(item for item in registry.list_all() if item.name == "Gain")
    assert [migration.source_type for migration in gain.migrations] == ["LegacyGain"]
    assert gain.migrations[0].owner == "feedbax"


def test_component_parameter_schema_migration_renames_required_parameter() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    registry.register_migration(
        ComponentMigration(
            source_type="Gain",
            target_type="Gain",
            owner="feedbax",
            source_param_schema_version="legacy-scale",
            target_param_schema_version="1",
            migration_id="feedbax.component.Gain.params.legacy-scale-to-1",
            migrate_params=lambda params: {"gain": params["scale"]},
        )
    )

    graph = spec_to_graph(
        _single_node_spec("Gain", {"scale": 8.0}, param_schema_version="legacy-scale"),
        registry,
    )

    component = graph.nodes["component"]
    assert isinstance(component, Gain)
    assert component.gain == 8.0


def test_downstream_migration_pack_can_migrate_owned_component_id() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    registry.register_component_type(
        "rlrmp.CurrentGain",
        lambda params: Gain(gain=float(params["gain"])),
        owner="rlrmp",
        provenance="package:rlrmp",
        param_schema=[{"name": "gain", "type": "float", "default": 1.0, "required": True}],
        input_ports=["input"],
        output_ports=["output"],
    )
    registry.register_migration_pack(
        ComponentMigrationPack(
            owner="rlrmp",
            package="rlrmp",
            migrations=(
                ComponentMigration(
                    source_type="rlrmp.LegacyGain",
                    target_type="rlrmp.CurrentGain",
                    owner="rlrmp",
                    migration_id="rlrmp.component.LegacyGain-to-CurrentGain.v1",
                ),
            ),
        )
    )

    graph = spec_to_graph(_single_node_spec("rlrmp.LegacyGain", {"gain": 9.0}), registry)

    component = graph.nodes["component"]
    assert isinstance(component, Gain)
    assert component.gain == 9.0
    definition = next(item for item in registry.list_all() if item.name == "rlrmp.CurrentGain")
    assert definition.owner == "rlrmp"
    assert definition.identity is not None
    assert definition.identity.provenance_kind == "package"
    assert definition.migrations[0].source_type == "rlrmp.LegacyGain"


def test_component_registry_round_trips_representation_contract() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    representation = RepresentationSpec.model_validate(
        {
            "anchors": [
                {
                    "id": "origin",
                    "semantic_role": "origin",
                    "interaction_roles": ["selectable"],
                    "binding": {"kind": "literal", "value": [0.0, 0.0], "dim": 2},
                },
                {
                    "id": "endpoint",
                    "semantic_role": "endpoint",
                    "interaction_roles": ["draggable", "editable"],
                    "binding": {
                        "kind": "param_path",
                        "path": "gain",
                        "expected_type": "float",
                    },
                },
            ],
            "elements": [
                {
                    "id": "gain-vector",
                    "archetype": "vector",
                    "anchors": ["endpoint", "origin"],
                    "frame_provider": {"kind": "from_input_port", "input_port": "input"},
                    "style": [{"channel": "stroke", "value": "currentColor"}],
                    "dim": 2,
                    "scale_invariant": True,
                }
            ],
            "style": [{"channel": "visibility", "value": True}],
        }
    )

    registry.register_component_type(
        "RepresentedGain",
        lambda params: Gain(gain=float(params["gain"])),
        category="Test",
        description="Test-only represented gain.",
        param_schema=[{"name": "gain", "type": "float", "default": 1.0}],
        input_ports=["input"],
        output_ports=["output"],
        representation=representation,
    )

    definition = next(item for item in registry.list_all() if item.name == "RepresentedGain")
    assert definition.representation is not None
    assert definition.representation.schema_id == "feedbax.spec.studio.representation"
    assert definition.representation.schema_version == REPRESENTATION_SCHEMA_VERSION
    assert [anchor.id for anchor in definition.representation.anchors] == ["endpoint", "origin"]
    assert definition.representation.elements[0].archetype == "vector"
    assert definition.representation.elements[0].frame_provider is not None
    assert definition.representation.elements[0].frame_provider.input_port == "input"


def test_builtin_mechanics_expose_workspace_representations() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    definitions = {item.name: item for item in registry.list_all()}

    point_mass = definitions["PointMass"]
    assert point_mass.representation is not None
    assert [element.archetype for element in point_mass.representation.elements] == [
        "point_body"
    ]
    assert point_mass.representation.anchors[0].id == "center"
    assert point_mass.representation.anchors[0].binding is not None

    two_link = definitions["TwoLinkArm"]
    assert two_link.default_params["link_lengths"] == [0.30, 0.33]
    assert two_link.representation is not None
    assert two_link.representation.reachability is not None
    assert two_link.representation.reachability.origin_anchor == "shoulder"
    assert two_link.representation.reachability.radius_transform == "sum_abs"
    links = next(
        element
        for element in two_link.representation.elements
        if element.archetype == "planar_chain"
    )
    assert links.anchors == ["shoulder", "elbow", "effector"]
    assert links.bindings["link_lengths"].kind == "param_path"
    assert links.bindings["link_lengths"].path == "link_lengths"
    assert links.planar_chain is not None
    assert links.planar_chain.reference_pose is not None
    assert links.planar_chain.reference_pose.coordinate_space == "configuration"
    assert links.planar_chain.reference_pose.values == [0.0, 0.0]


def test_two_link_arm_builder_uses_representation_link_lengths_param() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)

    component = build_component(
        "arm",
        "TwoLinkArm",
        {"dt": 0.01, "link_lengths": [0.5, 0.25]},
        component_registry=registry,
    )

    assert bool(jnp.allclose(component.plant.skeleton.l, jnp.array([0.5, 0.25])))


def test_builtin_muscle_representations_declare_consolidated_geometry_sources() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    definitions = {item.name: item for item in registry.list_all()}

    arm_template = definitions["Arm6MuscleRigidTendon"]
    assert arm_template.representation is not None
    arm_elements = {element.id: element for element in arm_template.representation.elements}
    assert arm_elements["links"].archetype == "planar_chain"
    assert arm_elements["muscle-paths"].archetype == "muscle_path"
    assert arm_elements["muscle-paths"].frame_provider is not None
    assert arm_elements["muscle-paths"].frame_provider.kind == "from_input_port"
    assert arm_elements["muscle-paths"].frame_provider.input_port == "angles"
    assert arm_template.representation.muscle_path_geometry is not None
    assert (
        arm_template.representation.metadata["geometry_source"]
        == "feedbax.mechanics.geometry.TwoLinkArmMuscleGeometry.default_six_muscle"
    )
    assert arm_template.representation.metadata["composition_rule"] == {
        "kind": "subgraph_children",
        "allow_outer_geometry_fallback": False,
    }
    opacity = next(
        style for style in arm_elements["muscle-paths"].style if style.channel == "opacity"
    )
    assert opacity.binding is not None
    assert opacity.binding.selector.compact == "output:activations"

    analytical = definitions["AnalyticalMusculoskeletalPlant"]
    assert analytical.representation is not None
    assert (
        analytical.representation.metadata["geometry_source"]
        == "feedbax.mechanics.muscle_config.default_6muscle_2link_muscled_arm_parameters"
    )
    muscle_geometry = analytical.representation.muscle_path_geometry
    assert muscle_geometry is not None
    assert muscle_geometry.schema_id == "feedbax.spec.studio.muscle_path_geometry"
    assert muscle_geometry.schema_version == "feedbax.spec.studio.muscle_path_geometry.v1"
    assert len(muscle_geometry.paths) == 6
    assert all(len(path.points) == 2 for path in muscle_geometry.paths)
    assert {point.frame for path in muscle_geometry.paths for point in path.points} <= {
        "world",
        "link0",
        "link1",
    }
    analytical_elements = {element.id: element for element in analytical.representation.elements}
    assert analytical_elements["muscle-paths"].frame_provider is not None
    assert (
        analytical_elements["muscle-paths"].frame_provider.kind
        == "from_representation_element"
    )
    assert analytical_elements["muscle-paths"].frame_provider.element_id == "links"
    assert analytical_elements["links"].planar_chain is not None
    assert analytical_elements["links"].planar_chain.pose_fallback == "zero"
    assert analytical_elements["links"].bindings["link_lengths"].kind == "literal"

    serialized = analytical.representation.model_dump(mode="json", exclude_none=True)
    assert serialized["schema_version"] == REPRESENTATION_SCHEMA_VERSION
    assert "frames" not in serialized["muscle_path_geometry"]
    assert serialized["muscle_path_geometry"]["paths"][0]["id"] == "muscle-0"

    point_mass_template = definitions["PointMass8MuscleRelu"]
    assert point_mass_template.representation is not None
    assert any(
        element.archetype == "muscle_path"
        for element in point_mass_template.representation.elements
    )
    assert (
        point_mass_template.representation.metadata["geometry_source"]
        == "feedbax.mechanics.geometry.PointMassRadialGeometry"
    )
    assert point_mass_template.representation.muscle_path_geometry is None


def test_builtin_reach_tasks_expose_schematic_objective_representations() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    definitions = {item.name: item for item in registry.list_all()}

    simple = definitions["SimpleReaches"]
    assert simple.representation is not None
    simple_elements = {element.id: element for element in simple.representation.elements}
    assert simple.representation.metadata["canonical_goal_anchor"] == "goal"
    assert simple_elements["workspace"].archetype == "region"
    assert simple_elements["reach-distribution"].archetype == "distribution_glyph"
    assert simple_elements["goal-marker"].metadata["canonical_goal"] is True
    assert simple_elements["objective"].archetype == "objective_link"

    delayed = definitions["DelayedReaches"]
    assert delayed.representation is not None
    delayed_elements = {element.id: element for element in delayed.representation.elements}
    assert delayed.representation.metadata["temporality"]["kind"] == "scheduled"
    assert (
        delayed.representation.metadata["temporality"]["target_on_epochs_param"]
        == "target_on_epochs"
    )
    assert delayed_elements["goal-marker"].metadata["canonical_goal"] is True
    assert delayed_elements["reach-distribution"].metadata["distribution"] == "center_out"


def test_component_registry_rejects_representation_unknown_param_path() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)

    with pytest.raises(ValueError, match="declared parameter"):
        registry.register_component_type(
            "BadRepresentation",
            lambda params: Gain(gain=1.0),
            category="Test",
            param_schema=[{"name": "gain", "type": "float", "default": 1.0}],
            input_ports=["input"],
            output_ports=["output"],
            representation={
                "elements": [
                    {
                        "id": "bad",
                        "archetype": "vector",
                        "bindings": {
                            "length": {
                                "kind": "param_path",
                                "path": "missing",
                            }
                        },
                    }
                ]
            },
        )


def test_representation_contract_rejects_non_current_schema_version() -> None:
    with pytest.raises(ValueError, match="literal_error"):
        RepresentationSpec.model_validate(
            {
                "schema_id": "feedbax.spec.studio.representation",
                "schema_version": "feedbax.spec.studio.representation.v0",
            }
        )


def test_representation_reachability_requires_known_origin_anchor() -> None:
    with pytest.raises(ValueError, match="unknown anchor"):
        RepresentationSpec.model_validate(
            {
                "reachability": {
                    "origin_anchor": "missing",
                    "radius_binding": {"kind": "literal", "value": 1.0},
                }
            }
        )


def test_planar_chain_reference_pose_uses_configuration_coordinates() -> None:
    representation = RepresentationSpec.model_validate(
        {
            "elements": [
                {
                    "id": "links",
                    "archetype": "planar_chain",
                    "planar_chain": {
                        "frame_ids": ["world", "link0", "link1"],
                        "reference_pose": {
                            "coordinate_space": "configuration",
                            "values": [-1.2, 1.7],
                        },
                    },
                }
            ]
        }
    )

    planar_chain = representation.elements[0].planar_chain
    assert planar_chain is not None
    assert planar_chain.reference_pose is not None
    assert planar_chain.reference_pose.values == [-1.2, 1.7]

    with pytest.raises(ValueError, match="configuration coordinate"):
        RepresentationSpec.model_validate(
            {
                "elements": [
                    {
                        "id": "links",
                        "archetype": "planar_chain",
                        "planar_chain": {
                            "frame_ids": ["world", "link0", "link1"],
                            "reference_pose": {"values": [0.0]},
                        },
                    }
                ]
            }
        )

    with pytest.raises(ValueError, match="literal_error"):
        RepresentationSpec.model_validate(
            {
                "elements": [
                    {
                        "id": "links",
                        "archetype": "planar_chain",
                        "planar_chain": {
                            "frame_ids": ["world", "link0", "link1"],
                            "reference_pose": {
                                "coordinate_space": "workspace",
                                "values": [0.0, 0.0],
                            },
                        },
                    }
                ]
            }
        )


def test_representation_selector_anchor_subpaths_are_enumerated() -> None:
    valid = RepresentationSpec.model_validate(
        {
            "anchors": [
                {
                    "id": "effector",
                    "semantic_role": "endpoint",
                    "binding": {
                        "kind": "selector",
                        "selector": {
                            "namespace": "mechanics_object",
                            "compact": "mechanics:scenario:train.mechanics",
                            "target_id": "scenario:train:mechanics",
                        },
                        "anchor_subpath": "position",
                    },
                }
            ]
        }
    )
    assert valid.anchors[0].binding is not None

    with pytest.raises(ValueError, match="anchor_subpath"):
        RepresentationSpec.model_validate(
            {
                "anchors": [
                    {
                        "id": "port-anchor",
                        "semantic_role": "endpoint",
                        "binding": {
                            "kind": "selector",
                            "selector": {
                                "namespace": "graph_port",
                                "compact": "port:node.output",
                            },
                            "anchor_subpath": "position",
                        },
                    }
                ]
            }
        )

    with pytest.raises(ValueError, match="literal_error"):
        RepresentationSpec.model_validate(
            {
                "anchors": [
                    {
                        "id": "bad-subpath",
                        "semantic_role": "endpoint",
                        "binding": {
                            "kind": "selector",
                            "selector": {
                                "namespace": "task_object",
                                "compact": "task:scenario:train",
                                "target_id": "scenario:train",
                            },
                            "anchor_subpath": "freeform",
                        },
                    }
                ]
            }
        )


def test_absent_downstream_owner_fails_with_actionable_message() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)

    with pytest.raises(ValueError) as exc_info:
        spec_to_graph(_single_node_spec("rlrmp.LegacyGain", {"gain": 1.0}), registry)

    message = str(exc_info.value)
    assert "owner='rlrmp'" in message
    assert "migration pack" in message
    assert "rlrmp.LegacyGain" in message


def test_loaded_owner_without_migration_edge_fails_with_version_context() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    registry.register_component_type(
        "rlrmp.CurrentGain",
        lambda params: Gain(gain=float(params["gain"])),
        owner="rlrmp",
        provenance="package:rlrmp",
        param_schema=[{"name": "gain", "type": "float", "default": 1.0}],
        input_ports=["input"],
        output_ports=["output"],
    )

    with pytest.raises(ValueError) as exc_info:
        spec_to_graph(_single_node_spec("rlrmp.UnsupportedGain", {"gain": 1.0}), registry)

    message = str(exc_info.value)
    assert "No component migration registered" in message
    assert "owner='rlrmp'" in message
    assert "rlrmp.UnsupportedGain" in message


def test_unsupported_component_parameter_schema_fails_with_current_version() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)

    with pytest.raises(ValueError) as exc_info:
        spec_to_graph(
            _single_node_spec("Gain", {"scale": 1.0}, param_schema_version="unsupported"),
            registry,
        )

    message = str(exc_info.value)
    assert "No component migration registered" in message
    assert "source_param_schema_version='unsupported'" in message
    assert "current_param_schema_version='1'" in message


def test_elementwise_affine_modulator_is_builtin_component() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)

    meta = registry.get("ElementwiseAffineModulator")

    assert meta is not None
    assert meta.default_params["signal_shape"] == [1]
    assert meta.default_params["baseline"] == 1.0
    assert meta.default_params["gain_init"] == 0.0
    assert meta.default_params["bias_init"] == 0.0
    assert meta.input_ports == ["signal", "modulator", "scale", "bias"]
    assert meta.output_ports == ["output"]
