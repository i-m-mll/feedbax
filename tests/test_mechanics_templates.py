from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from feedbax.acausal import (
    AcausalConnection,
    AcausalSystem,
    ForceSource,
    Ground,
    LinearDamper,
    LinearSpring,
    Mass,
    PositionSensor,
    VelocitySensor,
)
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.domain import MECHANICS_DOMAIN_ID
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.contracts.graphs.builders import build_component
from feedbax.contracts.graphs.acausal_compiler import compile_acausal_graph
from feedbax.contracts.graphs.serialization import graph_to_spec
from tests.graph_compiler_test_support import spec_to_graph
from feedbax.mechanics.muscle_config import default_6muscle_2link_moment_arms
from feedbax.mechanics.analytical_plant import AnalyticalMusculoskeletalPlant


pytestmark = [pytest.mark.usefixtures("enable_jax_x64"), pytest.mark.feedbax_contract]


def _registry() -> ComponentRegistry:
    return ComponentRegistry(load_user_components=False)


def _rollout(system: AcausalSystem, inputs: dict[str, jax.Array], steps: int = 8) -> jax.Array:
    state = system.init_state(key=jax.random.PRNGKey(0))
    values = []
    for step in range(steps):
        outputs, state = system(inputs, state, key=jax.random.PRNGKey(step + 1))
        values.append(jnp.asarray([outputs["position"], outputs["velocity"]]))
    return jnp.asarray(values)


def test_mechanics_templates_are_real_executable_acausal_graphs() -> None:
    registry = _registry()

    for name, inputs, outputs in (
        ("PointMass", ("force",), ("position", "velocity")),
        ("MassSpringDamper", ("force",), ("position", "velocity")),
        ("PointMassWithMuscles", ("flexor", "extensor"), ("position", "velocity")),
        ("TwoLinkArm6Muscle", ("excitation",), ("effector", "state")),
    ):
        meta = registry.get(name)
        assert meta is not None
        assert meta.template_kind == "executable"
        assert isinstance(meta.template_graph, AcausalGraphSpec)
        assert registry.template_builder_issues(meta) == []

        system = compile_acausal_graph(meta.template_graph, name, registry)
        assert system.input_ports == inputs
        assert system.output_ports == outputs


def test_two_link_arm_6muscle_template_uses_canonical_moment_arms() -> None:
    registry = _registry()
    template = registry.get("TwoLinkArm6Muscle")
    assert template is not None
    graph = template.template_graph
    assert isinstance(graph, AcausalGraphSpec)
    assert graph.physical_domain == "planar_multibody"

    moment_arms = default_6muscle_2link_moment_arms()
    muscle_nodes = [graph.nodes[f"muscle_{index}"] for index in range(6)]
    assert jnp.allclose(
        jnp.asarray([node.params["moment_arm"] for node in muscle_nodes]),
        moment_arms,
    )

    compiled = compile_acausal_graph(graph, "arm", registry)
    direct = build_component(
        "direct",
        "AnalyticalMusculoskeletalPlant",
        {"dt": graph.solver.dt, "n_steps": 1},
        component_registry=registry,
    )
    assert jnp.allclose(compiled.plant.moment_arms, direct.plant.moment_arms)

    compiled_state = compiled.init_state(key=jax.random.PRNGKey(0))
    direct_state = direct.init_state(key=jax.random.PRNGKey(0))
    inputs = {"excitation": jnp.ones(6) * 0.25}
    compiled_outputs, _ = compiled(inputs, compiled_state, key=jax.random.PRNGKey(1))
    direct_outputs, _ = direct(inputs, direct_state, key=jax.random.PRNGKey(1))
    assert jnp.allclose(
        compiled_outputs["state"].plant.skeleton.angle,
        direct_outputs["state"].plant.skeleton.angle,
    )


def test_multibody_registry_entries_declare_workspace_representations() -> None:
    registry = _registry()
    expected = {
        "PlanarLink": "planar_chain",
        "MusclePath": "muscle_path",
        "PointMarker": "marker",
    }
    for name, archetype in expected.items():
        meta = registry.get(name)
        assert meta is not None
        assert meta.representation is not None
        assert {element.archetype for element in meta.representation.elements} == {archetype}


def test_mass_spring_damper_template_matches_hand_built_system() -> None:
    registry = _registry()
    template = registry.get("MassSpringDamper")
    assert template is not None
    compiled = compile_acausal_graph(template.template_graph, "msd", registry)
    direct = AcausalSystem(
        elements={
            "wall": Ground("wall"),
            "mass": Mass("mass", mass=1.0),
            "spring": LinearSpring("spring", stiffness=10.0),
            "damper": LinearDamper("damper", damping=0.5),
            "force": ForceSource("force"),
            "position": PositionSensor("position"),
            "velocity": VelocitySensor("velocity"),
        },
        connections=[
            AcausalConnection(("wall", "flange"), ("spring", "flange_a")),
            AcausalConnection(("wall", "flange"), ("damper", "flange_a")),
            AcausalConnection(("spring", "flange_b"), ("mass", "flange")),
            AcausalConnection(("damper", "flange_b"), ("mass", "flange")),
            AcausalConnection(("force", "flange"), ("mass", "flange")),
            AcausalConnection(("position", "flange"), ("mass", "flange")),
            AcausalConnection(("velocity", "flange"), ("mass", "flange")),
        ],
        dt=0.001,
        input_bindings={"force": "force"},
        output_bindings={"position": "position", "velocity": "velocity"},
    )

    assert jnp.allclose(
        _rollout(compiled, {"force": jnp.asarray(1.0)}),
        _rollout(direct, {"force": jnp.asarray(1.0)}),
    )


def test_preformed_muscle_leaf_activation_drives_conserving_force() -> None:
    registry = _registry()
    template = registry.get("PointMassWithMuscles")
    assert template is not None
    system = compile_acausal_graph(template.template_graph, "muscled_point_mass", registry)

    flexor = _rollout(system, {"flexor": jnp.asarray(1.0), "extensor": jnp.asarray(0.0)})
    extensor = _rollout(system, {"flexor": jnp.asarray(0.0), "extensor": jnp.asarray(1.0)})

    assert flexor[-1, 1] > 0.0
    assert extensor[-1, 1] < 0.0


def test_analytical_musculoskeletal_plant_builds_and_steps_from_graph_spec() -> None:
    registry = _registry()
    spec = GraphSpec(
        nodes={
            "plant": ComponentSpec(
                type="AnalyticalMusculoskeletalPlant",
                params={"dt": 0.01, "n_steps": 2},
                input_ports=["excitation"],
                output_ports=["effector", "state"],
            )
        },
        input_ports=["excitation"],
        output_ports=["effector"],
        input_bindings={"excitation": ("plant", "excitation")},
        output_bindings={"effector": ("plant", "effector")},
    )

    graph = spec_to_graph(spec, registry)
    plant = graph.nodes["plant"]
    assert isinstance(plant.plant, AnalyticalMusculoskeletalPlant)
    assert plant.input_ports == ("excitation",)
    assert plant.backend is not None
    assert plant.backend.sub_dt == pytest.approx(0.005)

    state = graph.init_state(key=jax.random.PRNGKey(0))
    outputs, _ = graph(
        {"excitation": jnp.zeros(plant.plant.input_size)},
        state,
        key=jax.random.PRNGKey(1),
    )
    assert "effector" in outputs
    assert graph_to_spec(graph).nodes["plant"].type == "AnalyticalMusculoskeletalPlant"


def test_mechanics_domain_registry_entries_compile_or_template_build() -> None:
    registry = _registry()
    checked = []
    for definition in registry.list_all():
        if definition.domain != MECHANICS_DOMAIN_ID:
            continue
        meta = registry.get(definition.name)
        assert meta is not None
        checked.append(definition.name)
        if isinstance(meta.template_graph, AcausalGraphSpec):
            assert registry.template_builder_issues(meta) == []
            compile_acausal_graph(meta.template_graph, definition.name, registry)
            continue

        graph = AcausalGraphSpec(
            physical_domain="translational",
            nodes={definition.name: ComponentSpec(type=definition.name)},
            connections=[],
            solver={"solver_type": "euler", "dt": 0.001},
        )
        report = registry.template_builder_issues(
            type(
                "Template",
                (),
                {
                    "name": definition.name,
                    "template_id": None,
                    "template_graph": graph,
                },
            )()
        )
        assert not any("unknown element type" in issue.reason for issue in report)

    assert {"MassSpringDamper", "PointMassWithMuscles", "RigidTendonHillMuscle"} <= set(checked)
