from __future__ import annotations

import diffrax as dfx
import jax
import jax.numpy as jnp
import optimistix as optx
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
)
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.compiler.acausal_compiler import compile_acausal_graph
from tests.support.graph_compiler_test_support import spec_to_graph


pytestmark = [pytest.mark.usefixtures("enable_jax_x64"), pytest.mark.feedbax_contract]


def _registry() -> ComponentRegistry:
    return ComponentRegistry(load_user_components=False)


def _msd_interior(*, solver_type: str = "euler") -> AcausalGraphSpec:
    return AcausalGraphSpec(
        physical_domain="translational",
        nodes={
            "wall": ComponentSpec(type="Ground"),
            "mass": ComponentSpec(type="Mass", params={"mass": 1.0}),
            "spring": ComponentSpec(type="LinearSpring", params={"stiffness": 10.0}),
            "damper": ComponentSpec(type="LinearDamper", params={"damping": 0.5}),
            "act": ComponentSpec(
                type="ActuationInput",
                params={"port_name": "u", "source_kind": "force"},
            ),
            "sense": ComponentSpec(
                type="SensorOutput",
                params={"port_name": "pos", "quantity": "position"},
            ),
        },
        connections=[
            {"a": ("wall", "flange"), "b": ("spring", "flange_a")},
            {"a": ("wall", "flange"), "b": ("damper", "flange_a")},
            {"a": ("spring", "flange_b"), "b": ("mass", "flange")},
            {"a": ("damper", "flange_b"), "b": ("mass", "flange")},
            {"a": ("act", "flange"), "b": ("mass", "flange")},
            {"a": ("sense", "flange"), "b": ("mass", "flange")},
        ],
        solver={"solver_type": solver_type, "dt": 0.001},
    )


def _acausal_graph_spec(interior: AcausalGraphSpec) -> GraphSpec:
    return GraphSpec(
        nodes={
            "plant": ComponentSpec(
                type="AcausalSystem",
                input_ports=["u"],
                output_ports=["pos"],
            )
        },
        input_ports=["u"],
        output_ports=["pos"],
        input_bindings={"u": ("plant", "u")},
        output_bindings={"pos": ("plant", "pos")},
        subgraphs={"plant": interior},
    )


def _rollout_component(component: AcausalSystem, *, port: str, steps: int = 8) -> jax.Array:
    state = component.init_state(key=jax.random.PRNGKey(0))
    values = []
    for step in range(steps):
        outputs, state = component(
            {port: jnp.asarray(1.0)},
            state,
            key=jax.random.PRNGKey(step + 1),
        )
        values.append(outputs["pos"])
    return jnp.asarray(values)


def test_spec_to_graph_compiles_acausal_msd_and_responds_to_input() -> None:
    graph = spec_to_graph(_acausal_graph_spec(_msd_interior()), _registry())
    plant = graph.nodes["plant"]

    assert isinstance(plant, AcausalSystem)
    assert plant.input_ports == ("u",)
    assert plant.output_ports == ("pos",)

    state = graph.init_state(key=jax.random.PRNGKey(0))
    outputs = []
    for step in range(8):
        out, state = graph({"u": jnp.asarray(1.0)}, state, key=jax.random.PRNGKey(step + 1))
        outputs.append(out["pos"])

    trajectory = jnp.asarray(outputs)
    assert trajectory[-1] > trajectory[0]


def test_compiled_acausal_spec_matches_direct_system_trajectory() -> None:
    compiled = compile_acausal_graph(_msd_interior(), "plant", _registry())
    direct = AcausalSystem(
        elements={
            "wall": Ground("wall"),
            "mass": Mass("mass", mass=1.0),
            "spring": LinearSpring("spring", stiffness=10.0),
            "damper": LinearDamper("damper", damping=0.5),
            "act": ForceSource("act"),
            "sense": PositionSensor("sense"),
        },
        connections=[
            AcausalConnection(("wall", "flange"), ("spring", "flange_a")),
            AcausalConnection(("wall", "flange"), ("damper", "flange_a")),
            AcausalConnection(("spring", "flange_b"), ("mass", "flange")),
            AcausalConnection(("damper", "flange_b"), ("mass", "flange")),
            AcausalConnection(("act", "flange"), ("mass", "flange")),
            AcausalConnection(("sense", "flange"), ("mass", "flange")),
        ],
        dt=0.001,
        input_bindings={"u": "act"},
        output_bindings={"pos": "sense"},
    )

    assert jnp.allclose(
        _rollout_component(compiled, port="u"),
        _rollout_component(direct, port="u"),
    )


def test_nested_acausal_boundary_ports_flatten_like_flat_spec() -> None:
    nested = AcausalGraphSpec(
        physical_domain="translational",
        nodes={
            "wall": ComponentSpec(type="Ground"),
            "spring": ComponentSpec(type="LinearSpring", params={"stiffness": 10.0}),
            "damper": ComponentSpec(type="LinearDamper", params={"damping": 0.5}),
            "free": ComponentSpec(type="BoundaryPort", params={"port_name": "flange"}),
        },
        connections=[
            {"a": ("wall", "flange"), "b": ("spring", "flange_a")},
            {"a": ("wall", "flange"), "b": ("damper", "flange_a")},
            {"a": ("free", "flange"), "b": ("spring", "flange_b")},
            {"a": ("free", "flange"), "b": ("damper", "flange_b")},
        ],
        solver={"solver_type": "euler", "dt": 0.001},
    )
    nested_interior = AcausalGraphSpec(
        physical_domain="translational",
        nodes={
            "actuator": ComponentSpec(type="AcausalSystem", input_ports=["flange"]),
            "mass": ComponentSpec(type="Mass", params={"mass": 1.0}),
            "act": ComponentSpec(
                type="ActuationInput",
                params={"port_name": "u", "source_kind": "force"},
            ),
            "sense": ComponentSpec(
                type="SensorOutput",
                params={"port_name": "pos", "quantity": "position"},
            ),
        },
        connections=[
            {"a": ("actuator", "flange"), "b": ("mass", "flange")},
            {"a": ("act", "flange"), "b": ("mass", "flange")},
            {"a": ("sense", "flange"), "b": ("mass", "flange")},
        ],
        solver={"solver_type": "euler", "dt": 0.001},
        subgraphs={"actuator": nested},
    )

    flat = compile_acausal_graph(_msd_interior(), "plant", _registry())
    flattened = compile_acausal_graph(nested_interior, "plant", _registry())

    assert jnp.allclose(
        _rollout_component(flat, port="u"),
        _rollout_component(flattened, port="u"),
    )


def test_missing_acausal_interior_raises_domain_named_error() -> None:
    spec = GraphSpec(
        nodes={"plant": ComponentSpec(type="AcausalSystem")},
    )

    with pytest.raises(ValueError, match="feedbax.domain.acausal"):
        spec_to_graph(spec, _registry())


@pytest.mark.parametrize(
    ("solver_name", "solver_cls", "implicit"),
    [
        ("euler", dfx.Euler, False),
        ("implicit_euler", dfx.ImplicitEuler, True),
        ("kvaerno5", dfx.Kvaerno5, True),
        ("tsit5", dfx.Tsit5, False),
    ],
)
def test_acausal_solver_mapping(solver_name: str, solver_cls: type, implicit: bool) -> None:
    system = compile_acausal_graph(_msd_interior(solver_type=solver_name), "plant", _registry())

    assert isinstance(system.solver, solver_cls)
    if implicit:
        assert isinstance(system.root_finder, optx.Newton)
    else:
        assert system.root_finder is None


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("explicit_order", [False, True])
def test_persisted_v1_gear_has_reduced_report_and_physical_bound_outputs(nested, explicit_order):
    from feedbax.compiler.acausal_compiler import compile_acausal_authoring_report
    from feedbax.contracts.migrations import migrate_acausal_graph_spec

    spec = AcausalGraphSpec(
        schema_id="feedbax.spec.acausal_graph",
        schema_version="feedbax.spec.acausal_graph.v1",
        physical_domain="rotational",
        nodes={
            "a": ComponentSpec(type="Inertia", params={"inertia": 2.0}),
            "b": ComponentSpec(type="Inertia", params={"inertia": 3.0}),
            "gear": ComponentSpec(type="GearRatio", params={"ratio": 2.0}),
            "wall": ComponentSpec(type="RotationalGround"),
            "spring": ComponentSpec(type="TorsionalSpring", params={"stiffness": 4.0}),
            "drive": ComponentSpec(
                type="ActuationInput", params={"port_name": "u", "source_kind": "torque"}
            ),
            "angle": ComponentSpec(
                type="SensorOutput", params={"port_name": "qb", "quantity": "angle"}
            ),
            "velocity": ComponentSpec(
                type="SensorOutput", params={"port_name": "vb", "quantity": "angular_velocity"}
            ),
            "torque": ComponentSpec(
                type="SensorOutput", params={"port_name": "tb", "quantity": "torque"}
            ),
        },
        connections=[
            {"a": a, "b": b}
            for a, b in [
                (("a", "flange"), ("gear", "flange_a")),
                (("b", "flange"), ("gear", "flange_b")),
                (("wall", "flange"), ("spring", "flange_a")),
                (("spring", "flange_b"), ("a", "flange")),
                (("drive", "flange"), ("a", "flange")),
                (("angle", "flange"), ("b", "flange")),
                (("velocity", "flange"), ("b", "flange")),
                (("torque", "flange"), ("b", "flange")),
            ]
        ],
        solver={"solver_type": "euler", "dt": 0.01},
    )
    if explicit_order:
        spec = spec.model_copy(
            update={
                "nodes": {
                    **spec.nodes,
                    **{
                        name: spec.nodes[name].model_copy(
                            update={"params": {**spec.nodes[name].params, "order": order}}
                        )
                        for name, order in (("velocity", 0), ("torque", 1), ("angle", 2))
                    },
                }
            }
        )
    if nested:
        inner = AcausalGraphSpec(
            physical_domain="rotational",
            nodes={
                "gear": spec.nodes["gear"],
                "left": ComponentSpec(type="BoundaryPort", params={"port_name": "flange_a"}),
                "right": ComponentSpec(type="BoundaryPort", params={"port_name": "flange_b"}),
            },
            connections=[
                {"a": ("left", "flange"), "b": ("gear", "flange_a")},
                {"a": ("right", "flange"), "b": ("gear", "flange_b")},
            ],
            solver=spec.solver,
        )
        spec = spec.model_copy(
            update={
                "nodes": {
                    **spec.nodes,
                    "gear": ComponentSpec(
                        type="AcausalSystem", input_ports=["flange_a", "flange_b"]
                    ),
                },
                "subgraphs": {"gear": inner},
            }
        )
    import json

    payload = json.loads(spec.model_dump_json())
    loaded = AcausalGraphSpec.model_validate(migrate_acausal_graph_spec(payload).payload)
    report = compile_acausal_authoring_report(
        loaded, node_path=["plant"], component_registry=_registry()
    )
    assert report.status == "ok", report.diagnostics
    assert report.summary["n_equations"] == 2
    assert report.summary["n_states"] == 2
    system = compile_acausal_graph(loaded, "plant", _registry())
    assert system.output_ports == (("vb", "tb", "qb") if explicit_order else ("qb", "tb", "vb"))
    state = system.init_state(key=jax.random.PRNGKey(0))
    outputs, state = system({"u": jnp.array([14.0])}, state, key=jax.random.PRNGKey(1))
    assert outputs["vb"] == pytest.approx(0.02)
    assert outputs["tb"] == pytest.approx(6.0)
    outputs, _ = system({"u": jnp.array([14.0])}, state, key=jax.random.PRNGKey(2))
    assert outputs["qb"] == pytest.approx(0.0002)

    # A boundary-bearing child does not excuse an enclosing massless free coordinate.
    unsupported = loaded.model_copy(
        update={
            "nodes": {
                **loaded.nodes,
                "a": ComponentSpec(type="AngleSensor"),
                "b": ComponentSpec(type="AngleSensor"),
            }
        }
    )
    invalid_report = compile_acausal_authoring_report(
        unsupported, node_path=["plant"], component_registry=_registry()
    )
    assert invalid_report.status == "error"
    assert any(d.code == "acausal.unbalanced" for d in invalid_report.diagnostics)


def test_invalid_persisted_gear_has_actionable_authoring_diagnostic():
    from feedbax.compiler.acausal_compiler import compile_acausal_authoring_report

    spec = AcausalGraphSpec(
        physical_domain="rotational",
        nodes={"gear": ComponentSpec(type="GearRatio", params={"ratio": 0.0})},
        solver={"solver_type": "euler", "dt": 0.01},
    )
    report = compile_acausal_authoring_report(
        spec, node_path=["plant"], component_registry=_registry()
    )
    assert report.status == "error"
    assert any(
        d.code == "acausal.gear_constraint" and "finite and nonzero" in d.message
        for d in report.diagnostics
    )
