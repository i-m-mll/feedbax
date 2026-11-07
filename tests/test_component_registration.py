from __future__ import annotations

from feedbax.component_registry import ComponentPack
from feedbax.contracts.representation import RepresentationSpec

import asyncio
import sys
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from pydantic import BaseModel, Field, ConfigDict

from feedbax.component_registry import (
    ComponentRegistry,
)
from feedbax.component_registry.cde_templates import cde_template_pack
from feedbax.runtime.channel import Channel
from feedbax.runtime.components import Gain
from feedbax.contracts.graph import ComponentSpec, GraphSpec, WireSpec
from feedbax.compiler.builders import build_component
from feedbax.compiler.serialization import graph_to_spec
from feedbax.models.cde import cde_controller
from feedbax.runtime.graph import Component
from feedbax.contracts.portable import Ports, ComponentOptions
from feedbax.contracts.component import PortType
from feedbax.studio.component_presentation import PresentationOverride
from tests.support.graph_compiler_test_support import spec_to_graph
from feedbax.contracts.representation import REPRESENTATION_SCHEMA_VERSION
from feedbax.plugins.application import COMPONENTS, new_registration_context
from feedbax.plugins.bootstrap import (
    FamilyRequirement,
    PluginDeclaration,
    PluginRegistration,
    bootstrap_application,
)


class _PrototypeSource(Component):
    ports = Ports({}, {"signal": PortType(dtype="any")})
    component_options = ComponentOptions(
        output_prototype_fn=lambda params, inputs: {"signal": jnp.zeros((params["width"],))}
    )
    width: int = 3

    def __call__(self, inputs, state, *, key):
        return {"signal": jnp.ones((self.width,))}, state


def _single_node_spec(
    component_type: str,
    params: dict[str, Any] | None = None,
    *,
    revision: int | None = None,
) -> GraphSpec:
    return GraphSpec(
        nodes={
            "component": ComponentSpec(
                type=component_type,
                params=params or {},
                revision=revision,
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
    class TestProgrammaticGain(Gain):
        def __init__(self, gain: float = 3.0):
            super().__init__(gain=gain)

    registry = ComponentRegistry()
    registry.ingest(ComponentPack(namespace="test", components=(TestProgrammaticGain,)))
    meta = registry.get("test.TestProgrammaticGain")
    assert meta.default_params == {"gain": 3.0}
    graph = spec_to_graph(_single_node_spec("test.TestProgrammaticGain", {"gain": 4.0}), registry)
    assert isinstance(graph.nodes["component"], TestProgrammaticGain)
    assert graph.nodes["component"].gain == 4.0
    default = spec_to_graph(_single_node_spec("test.TestProgrammaticGain"), registry)
    assert default.nodes["component"].gain == 3.0


def test_registered_component_output_prototype_feeds_stateful_materialization() -> None:
    registry = ComponentRegistry()
    registry.ingest(ComponentPack(namespace="test", components=(_PrototypeSource,)))
    spec = GraphSpec(
        nodes={
            "source": ComponentSpec(
                type="test._PrototypeSource",
                params={},
                input_ports=[],
                output_ports=["signal"],
            ),
            "delay": ComponentSpec(
                type="feedbax.Channel",
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


def test_declared_component_is_immutable() -> None:
    registry = ComponentRegistry()
    meta = registry.get("feedbax.FixedField")
    assert meta is not None
    assert meta.provenance == "feedbax"
    original = meta.output_prototype_fn

    with pytest.raises(AttributeError):
        meta.output_prototype_fn = lambda params, inputs: {"force": inputs["force"]}

    assert meta.output_prototype_fn is original


def test_builtin_registry_discovers_history_free_owners_with_consistent_contracts(
    monkeypatch,
) -> None:
    from feedbax.component_registry import builtins

    monkeypatch.setattr(builtins, "_PORTABLE_MODULES", (*builtins._PORTABLE_MODULES, __name__))
    registry = ComponentRegistry()
    owner = registry.get("feedbax._PluginGain")
    assert owner.owner_class is _PluginGain
    assert owner.revision == 1

    for name in registry.names():
        meta = registry.get(name)
        assert meta is not None
        assert callable(meta.builder), f"{name} has no explicit builder contract"
        if meta.port_types is not None:
            assert set(meta.port_types.inputs) == set(meta.input_ports), name
            assert set(meta.port_types.outputs) == set(meta.output_ports), name


def test_unimplemented_projection_placeholders_are_not_registered() -> None:
    registry = ComponentRegistry()
    assert registry.get("MomentArmProjection") is None
    assert registry.get("RadialForceProjection") is None
    with pytest.raises(ValueError, match="Unsupported component type 'MomentArmProjection'"):
        build_component("projection", "MomentArmProjection", {}, component_registry=registry)


class _PluginGain(Component):
    ports = Ports({"input": PortType(dtype="any")}, {"output": PortType(dtype="any")})
    gain: float

    def __init__(self, gain: Annotated[float, Field(gt=0)] = 2.0):
        self.gain = gain

    def __call__(self, inputs, state, *, key):
        return {"output": self.gain * inputs["input"]}, state


class _UnmarkedPluginGain(_PluginGain):
    pass


def test_entry_point_component_registration_records_package_provenance() -> None:
    def registrar(context) -> None:
        component_registry = context.registry(COMPONENTS)
        reexport = ModuleType("feedbax_test_plugin.reexport")
        reexport.PluginGain = _PluginGain
        assert not component_registry.discover_modules(reexport, namespace="test").components
        pack = component_registry.discover_modules(sys.modules[__name__], namespace="test")
        component_registry.ingest(pack, provenance="package:feedbax-test-plugin")

    class FakeDist:
        metadata = {"Name": "feedbax-test-plugin"}

    class FakeEntryPoint:
        name = "feedbax_test_plugin"
        dist = FakeDist()

        value = "feedbax_test_plugin:PLUGIN_REGISTRATION"

        def load(self) -> PluginRegistration:
            return PluginRegistration(
                PluginDeclaration(
                    "feedbax.test.components",
                    "1",
                    2,
                    families=(FamilyRequirement(COMPONENTS.family),),
                ),
                registrar,
            )

    state = asyncio.run(
        bootstrap_application(new_registration_context(), entry_points=[FakeEntryPoint()])
    )
    registry = state.bundle.components

    meta = registry.get("test._PluginGain")
    assert meta is not None
    assert meta.provenance == "package:feedbax-test-plugin"

    definition = next(item for item in registry.list_all() if item.name == "test._PluginGain")
    assert definition.default_params == {"gain": 2.0}
    assert definition.input_ports == ["input"]
    assert definition.output_ports == ["output"]
    assert definition.provenance == "package:feedbax-test-plugin"

    graph = spec_to_graph(_single_node_spec("test._PluginGain", {"gain": 5.0}), registry)
    leaf = graph.nodes["component"]
    assert leaf({"input": 3.0}, None, key=None)[0] == {"output": 15.0}
    assert graph_to_spec(graph, registry).nodes["component"].params == {"gain": 5.0}
    assert registry.serialize_component(_PluginGain())[0] == "test._PluginGain"
    root = Path(__file__).resolve().parents[1]
    assert (
        "tests/test_component_registration.py"
        in registry.component_source_paths(root)["test._PluginGain"]
    )
    with pytest.raises(ValueError, match="greater than 0"):
        registry.resolve_component_spec("test._PluginGain", {"gain": 0.0})


def test_component_pack_collision_leaves_no_partial_registration() -> None:
    class PluginGain(_PluginGain):
        pass

    registry = ComponentRegistry()
    registry.ingest(ComponentPack(namespace="test", components=(PluginGain,)))
    with pytest.raises(ValueError, match="already registered"):
        registry.ingest(ComponentPack(namespace="test", components=(_PluginGain, PluginGain)))
    assert registry.get("test._PluginGain") is None
    registry.ingest(ComponentPack(namespace="test", components=(_PluginGain,)))
    assert registry.get("test._PluginGain") is not None


def test_unknown_component_error_names_type_and_known_registry_contents() -> None:
    registry = ComponentRegistry()

    with pytest.raises(ValueError) as exc_info:
        spec_to_graph(_single_node_spec("DefinitelyUnknownComponent"), registry)

    message = str(exc_info.value)
    assert "DefinitelyUnknownComponent" in message
    assert "Known component types:" in message
    assert "feedbax.Gain" in message


def test_cde_primitives_build_and_execute_with_expected_semantics() -> None:
    registry = ComponentRegistry()
    state_key = jax.random.PRNGKey(0)

    input_node = build_component(
        "obs_in",
        "feedbax.Input",
        {"output_port": "obs"},
        component_registry=registry,
    )
    outputs, _ = input_node(
        {"input": jnp.array([1.0, 2.0])},
        input_node.init_state(key=state_key),
        key=state_key,
    )
    assert jnp.array_equal(outputs["obs"], jnp.array([1.0, 2.0]))

    subtract = build_component("subtract", "feedbax.Subtract", {}, component_registry=registry)
    outputs, _ = subtract(
        {"a": jnp.array([3.0, 5.0]), "b": jnp.array([1.0, 2.0])},
        subtract.init_state(key=state_key),
        key=state_key,
    )
    assert jnp.array_equal(outputs["out"], jnp.array([2.0, 3.0]))

    reshape = build_component(
        "reshape", "feedbax.Reshape", {"shape": [2, 2]}, component_registry=registry
    )
    outputs, _ = reshape(
        {"input": jnp.arange(4.0)},
        reshape.init_state(key=state_key),
        key=state_key,
    )
    assert outputs["output"].shape == (2, 2)
    assert jnp.array_equal(outputs["output"], jnp.array([[0.0, 1.0], [2.0, 3.0]]))

    matmul = build_component("matmul", "feedbax.MatMul", {}, component_registry=registry)
    outputs, _ = matmul(
        {"a": jnp.array([[1.0, 2.0], [3.0, 4.0]]), "b": jnp.array([0.5, 1.0])},
        matmul.init_state(key=state_key),
        key=state_key,
    )
    assert jnp.array_equal(outputs["out"], jnp.array([2.5, 5.5]))

    scale = build_component("scale", "feedbax.Scale", {"scale": -0.5}, component_registry=registry)
    outputs, _ = scale(
        {"input": jnp.array([2.0, -4.0])},
        scale.init_state(key=state_key),
        key=state_key,
    )
    assert jnp.array_equal(outputs["output"], jnp.array([-1.0, 2.0]))

    sigmoid = build_component("sigmoid", "feedbax.Sigmoid", {}, component_registry=registry)
    outputs, _ = sigmoid(
        {"input": jnp.array([-jnp.inf, 0.0, jnp.inf])},
        sigmoid.init_state(key=state_key),
        key=state_key,
    )
    assert jnp.array_equal(outputs["output"], jnp.array([0.0, 0.5, 1.0]))


def test_cde_templates_are_not_builtin_defaults() -> None:
    registry = ComponentRegistry()

    assert registry.get("feedbax.CDEStandard") is None


def test_caller_registered_cde_templates_are_executable_and_run_one_step() -> None:
    registry = ComponentRegistry()
    registry.ingest(cde_template_pack(), provenance="test-cde-pack")

    for template_name in (
        "feedbax.CDEStandard",
        "feedbax.CDEDecay",
        "feedbax.CDEAntiNF",
        "feedbax.CDEHybridV9b",
    ):
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
    registry = ComponentRegistry()
    for template_name in ("feedbax.RecurrentController", "feedbax.SimpleFeedbackLoop"):
        meta = registry.get(template_name)
        assert meta is not None
        assert registry.template_builder_issues(meta) == []


def test_cde_standard_template_matches_recurrent_cde_first_step() -> None:
    registry = ComponentRegistry()
    registry.ingest(cde_template_pack(), provenance="test-cde-pack")
    meta = registry.get("feedbax.CDEStandard")
    assert meta is not None
    graph = spec_to_graph(meta.template_graph, registry)
    net = cde_controller(
        obs_dim=1,
        hidden_dim=1,
        out_size=1,
        vf_width=128,
        vf_depth=1,
        decay=0.0,
        use_anti_nf=False,
        key=jax.random.PRNGKey(42),
    )
    graph = eqx.tree_at(lambda g: g.nodes["vf"].linears, graph, net.nodes["vf"].linears)
    graph = eqx.tree_at(lambda g: g.nodes["linear"].layer, graph, net.nodes["linear"].layer)

    net_state = net.init_state(key=jax.random.PRNGKey(0))
    carry = net.initial_cycle_port_values(net_state)
    graph_state = graph.init_state(key=jax.random.PRNGKey(0))
    obs = jnp.array([0.25])

    net_outputs, _, _ = net.step({"input": obs}, net_state, key=jax.random.PRNGKey(1))
    graph_outputs, _ = graph(
        {
            "obs": obs,
            "obs_prev": carry[("obs_prev_in", "input")],
            "h_prev": carry[("h_prev_in", "input")],
        },
        graph_state,
        key=jax.random.PRNGKey(1),
    )

    assert jnp.allclose(graph_outputs["h_new"], net_outputs["hidden"])
    assert jnp.allclose(graph_outputs["action"], net_outputs["output"])


def test_component_registry_round_trips_representation_contract() -> None:
    class RepresentedGain(Gain):
        def __init__(self, gain: float = 1.0):
            super().__init__(gain=gain)

    registry = ComponentRegistry()
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

    registry.ingest(
        ComponentPack(
            namespace="test",
            components=(RepresentedGain,),
            presentation={RepresentedGain: PresentationOverride(representation=representation)},
        )
    )

    definition = next(item for item in registry.list_all() if item.name == "test.RepresentedGain")
    assert definition.representation is not None
    assert definition.representation.schema_id == "feedbax.spec.studio.representation"
    assert definition.representation.schema_version == REPRESENTATION_SCHEMA_VERSION
    assert [anchor.id for anchor in definition.representation.anchors] == ["endpoint", "origin"]
    assert definition.representation.elements[0].archetype == "vector"
    assert definition.representation.elements[0].frame_provider is not None
    assert definition.representation.elements[0].frame_provider.input_port == "input"


def test_builtin_mechanics_expose_workspace_representations() -> None:
    registry = ComponentRegistry()
    definitions = {item.name: item for item in registry.list_all()}

    point_mass = definitions["feedbax.PointMass"]
    assert point_mass.representation is not None
    assert [element.archetype for element in point_mass.representation.elements] == ["point_body"]
    assert point_mass.representation.anchors[0].id == "center"
    assert point_mass.representation.anchors[0].binding is not None

    two_link = definitions["feedbax.TwoLinkArm"]
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
    registry = ComponentRegistry()

    component = build_component(
        "arm",
        "feedbax.TwoLinkArm",
        {"dt": 0.01, "link_lengths": [0.5, 0.25]},
        component_registry=registry,
    )

    assert bool(jnp.allclose(component.plant.skeleton.l, jnp.array([0.5, 0.25])))


def test_builtin_muscle_representations_declare_consolidated_geometry_sources() -> None:
    registry = ComponentRegistry()
    definitions = {item.name: item for item in registry.list_all()}

    arm_template = definitions["feedbax.Arm6MuscleRigidTendon"]
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

    analytical = definitions["feedbax.AnalyticalMusculoskeletalPlant"]
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
    assert analytical_elements["muscle-paths"].frame_provider.kind == "from_representation_element"
    assert analytical_elements["muscle-paths"].frame_provider.element_id == "links"
    assert analytical_elements["links"].planar_chain is not None
    assert analytical_elements["links"].planar_chain.pose_fallback == "zero"
    assert analytical_elements["links"].bindings["link_lengths"].kind == "literal"

    serialized = analytical.representation.model_dump(mode="json", exclude_none=True)
    assert serialized["schema_version"] == REPRESENTATION_SCHEMA_VERSION
    assert "frames" not in serialized["muscle_path_geometry"]
    assert serialized["muscle_path_geometry"]["paths"][0]["id"] == "muscle-0"

    point_mass_template = definitions["feedbax.PointMass8MuscleRelu"]
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
    registry = ComponentRegistry()
    definitions = {item.name: item for item in registry.list_all()}

    simple = definitions["feedbax.SimpleReaches"]
    assert simple.representation is not None
    simple_elements = {element.id: element for element in simple.representation.elements}
    assert simple.representation.metadata["canonical_goal_anchor"] == "goal"
    assert simple_elements["workspace"].archetype == "region"
    assert simple_elements["reach-distribution"].archetype == "distribution_glyph"
    assert simple_elements["goal-marker"].metadata["canonical_goal"] is True
    assert simple_elements["objective"].archetype == "objective_link"

    delayed = definitions["feedbax.DelayedReaches"]
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
    class BadRepresentation(Gain):
        def __init__(self, gain: float = 1.0):
            super().__init__(gain=gain)

    registry = ComponentRegistry()

    with pytest.raises(ValueError, match="declared parameter"):
        registry.ingest(
            ComponentPack(
                namespace="test",
                components=(BadRepresentation,),
                presentation={
                    BadRepresentation: PresentationOverride(
                        representation=RepresentationSpec.model_validate(
                            {
                                "elements": [
                                    {
                                        "id": "bad",
                                        "archetype": "vector",
                                        "bindings": {
                                            "length": {"kind": "param_path", "path": "missing"}
                                        },
                                    }
                                ]
                            }
                        )
                    )
                },
            )
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
    registry = ComponentRegistry()

    with pytest.raises(ValueError) as exc_info:
        spec_to_graph(_single_node_spec("rlrmp.LegacyGain", {"gain": 1.0}), registry)

    message = str(exc_info.value)
    assert "owner='rlrmp'" in message
    assert "migration pack" in message
    assert "rlrmp.LegacyGain" in message


def test_loaded_owner_without_migration_edge_fails_with_version_context() -> None:
    class CurrentGain(Gain):
        pass

    registry = ComponentRegistry()
    registry.ingest(ComponentPack(namespace="rlrmp", components=(CurrentGain,)))
    with pytest.raises(ValueError) as exc_info:
        spec_to_graph(_single_node_spec("rlrmp.UnsupportedGain", {"gain": 1.0}), registry)
    message = str(exc_info.value)
    assert "No component migration registered" in message
    assert "owner='rlrmp'" in message
    assert "rlrmp.UnsupportedGain" in message


def test_unsupported_component_parameter_schema_fails_with_current_version() -> None:
    registry = ComponentRegistry()

    with pytest.raises(ValueError) as exc_info:
        registry.resolve_component_spec("feedbax.Gain", {"scale": 1.0}, legacy_token="unsupported")
    message = str(exc_info.value)
    assert "No component migration registered" in message
    assert "source_legacy_token='unsupported'" in message
    assert "current_revision=1" in message


def test_elementwise_affine_modulator_is_builtin_component() -> None:
    registry = ComponentRegistry()

    meta = registry.get("feedbax.ElementwiseAffineModulator")

    assert meta is not None
    assert meta.default_params["signal_shape"] == [1]
    assert meta.default_params["baseline"] == 1.0
    assert meta.default_params["gain_init"] == 0.0
    assert meta.default_params["bias_init"] == 0.0
    assert meta.input_ports == ["signal", "modulator", "scale", "bias"]
    assert meta.output_ports == ["output"]


class _TypedGainParams(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    gain: float = 1.0


class _TypedGain(Gain):
    config = _TypedGainParams


def test_downstream_typed_model_is_the_only_schema_source() -> None:
    registry = ComponentRegistry()
    registry.ingest(ComponentPack(namespace="downstream", components=(_TypedGain,)))
    graph = spec_to_graph(_single_node_spec("downstream._TypedGain", {"gain": 4.5}), registry)
    serialized = graph_to_spec(graph, registry)
    assert serialized.nodes["component"].type == "downstream._TypedGain"
    assert serialized.nodes["component"].revision == 1
    assert serialized.nodes["component"].params == {"gain": 4.5}
    with pytest.raises(ValueError, match="valid number"):
        registry.resolve_component_spec("downstream._TypedGain", {"gain": "4.5"})


def test_effective_tuple_from_array_exports_without_authoring_normalizers():
    from pydantic import BaseModel, field_validator

    calls = []

    class ShapeConfig(BaseModel):
        shape: tuple[int, ...]

        @field_validator("shape")
        @classmethod
        def canonical_shape(cls, shape):
            calls.append(shape)
            return shape

    class ArrayShape(Component):
        config = ShapeConfig
        ports = Ports({}, {})
        shape: jax.Array

        def __init__(self, shape):
            self.shape = jnp.asarray(shape)

        def __call__(self, inputs, state, *, key):
            return {}, state

    component = ArrayShape.from_config(ShapeConfig(shape=(2, 3)))
    assert component.configuration().shape == (2, 3)
    assert component.configuration().model_dump(mode="json", warnings="error") == {"shape": [2, 3]}
    assert calls == [(2, 3)]


def test_component_pack_rejects_ambiguous_runtime_owner() -> None:
    registry = ComponentRegistry()
    registry.ingest(ComponentPack(namespace="downstream", components=(_TypedGain,)))
    with pytest.raises(ValueError, match="runtime type already registered"):
        registry.ingest(ComponentPack(namespace="other", components=(_TypedGain,)))


def test_declared_parameter_model_owns_defaults_required_fields_and_constraints() -> None:
    registry = ComponentRegistry()

    assert registry.resolve_component_spec("feedbax.MatMul", {}).params == {}
    selector = registry.resolve_component_spec("feedbax.StateFeedbackSelector", {})
    assert selector.params["expected_state_dim"] is None
    assert selector.params["state_slices"] is None

    assert registry.resolve_component_spec("feedbax.Gain", {}).params == {"gain": 1.0}
    with pytest.raises(ValueError, match="extra_forbidden"):
        registry.resolve_component_spec("feedbax.Gain", {"gain": 1.0, "scale": 2.0})
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        registry.resolve_component_spec("feedbax.GRU", {"input_size": 0, "hidden_size": 4})
