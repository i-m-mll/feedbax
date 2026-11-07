from __future__ import annotations

import diffrax as dfx
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
    PrescribedMotion,
)
from feedbax.component_registry import ComponentPack, ComponentRegistry
from feedbax.contracts.acausal import AcausalGraphSpec, SolverConfigSpec
from feedbax.contracts.domain import ACAUSAL_DOMAIN_ID
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.compiler import GraphCompilationError, GraphDocument, compile_graph
from feedbax.compiler.acausal_compiler import (
    compile_acausal_authoring_report,
    compile_acausal_graph,
)
from feedbax.runtime.graph import Graph, RolloutStepContext, RolloutStepHookResult
from tests.support.graph_compiler_test_support import spec_to_graph


pytestmark = [pytest.mark.usefixtures("enable_jax_x64"), pytest.mark.feedbax_contract]


def _registry() -> ComponentRegistry:
    return ComponentRegistry()


@pytest.mark.parametrize("grounded", [True, False], ids=["ground", "second-motion-source"])
def test_conflicting_coordinate_owners_refuse_report_and_execution(grounded: bool) -> None:
    motion = ComponentSpec(
        type="feedbax.ActuationInput", params={"source_kind": "prescribed_motion"}
    )
    other = (
        ComponentSpec(type="feedbax.Ground")
        if grounded
        else ComponentSpec(
            type="feedbax.ActuationInput",
            params={"source_kind": "prescribed_motion", "port_name": "other"},
        )
    )
    connections = [
        (("motion", "flange"), ("sense", "flange")),
        (("other", "flange"), ("sense", "flange")),
    ]
    spec = AcausalGraphSpec(
        physical_domain="translational",
        nodes={
            "motion": motion,
            "other": other,
            "sense": ComponentSpec(type="feedbax.SensorOutput", params={"quantity": "position"}),
        },
        connections=[{"a": a, "b": b} for a, b in connections],
        solver={"solver_type": "euler", "dt": 0.01},
    )
    registry = _registry()
    report = compile_acausal_authoring_report(
        spec, node_path=["plant"], component_registry=registry
    )
    assert report.status == "error"
    code = (
        "acausal.grounded_prescribed_coordinate" if grounded else "acausal.parallel_across_sources"
    )
    diagnostic = next(d for d in report.diagnostics if d.code == code)
    assert diagnostic.severity == "error"
    assert set(diagnostic.node_ids) == {"motion", "other"}

    message = "grounded by" if grounded else "multiple prescribed motion sources"
    with pytest.raises(ValueError, match=message):
        compile_acausal_graph(spec, "plant", registry)
    with pytest.raises(ValueError, match=message):
        AcausalSystem(
            elements={
                "motion": PrescribedMotion("motion"),
                "other": Ground("other") if grounded else PrescribedMotion("other"),
                "sense": PositionSensor("sense"),
            },
            connections=[AcausalConnection(a, b) for a, b in connections],
        )


def test_registered_custom_mass_preserves_independent_prescribed_inputs() -> None:
    class CustomMass(Mass):
        pass

    nodes = {
        "mass": ComponentSpec(type="test.CustomMass", params={"mass": 2.0}),
        "velocity": ComponentSpec(type="feedbax.SensorOutput", params={"quantity": "velocity"}),
    }
    connections = [{"a": ("velocity", "flange"), "b": ("mass", "flange")}]
    for side, stiffness in (("left", 2.0), ("right", 3.0)):
        spring = f"{side}_spring"
        nodes[side] = ComponentSpec(
            type="feedbax.ActuationInput",
            params={"source_kind": "prescribed_motion", "port_name": side},
        )
        nodes[spring] = ComponentSpec(type="feedbax.LinearSpring", params={"stiffness": stiffness})
        connections.extend(
            [
                {"a": (side, "flange"), "b": (spring, "flange_a")},
                {"a": (spring, "flange_b"), "b": ("mass", "flange")},
            ]
        )
    spec = AcausalGraphSpec(
        physical_domain="translational",
        nodes=nodes,
        connections=connections,
        solver={"solver_type": "euler", "dt": 0.1},
    )
    registry = _registry()
    registry.ingest(ComponentPack(namespace="test", components=(CustomMass,)))
    report = compile_acausal_authoring_report(
        spec, node_path=["plant"], component_registry=registry
    )
    assert report.status == "ok", report.diagnostics
    system = compile_acausal_graph(spec, "plant", registry)
    outputs, _ = system(
        {"left": jnp.array([1.0, 0.0]), "right": jnp.array([-2.0, 0.0])},
        system.init_state(key=jax.random.PRNGKey(0)),
        key=jax.random.PRNGKey(1),
    )
    # From rest, Euler advances velocity by dt * (2 * 1 + 3 * -2) / 2.
    assert outputs["y"] == pytest.approx(-0.2)


def _msd_interior() -> AcausalGraphSpec:
    return AcausalGraphSpec(
        physical_domain="translational",
        nodes={
            "wall": ComponentSpec(type="feedbax.Ground"),
            "mass": ComponentSpec(type="feedbax.Mass", params={"mass": 1.0}),
            "spring": ComponentSpec(type="feedbax.LinearSpring", params={"stiffness": 10.0}),
            "damper": ComponentSpec(type="feedbax.LinearDamper", params={"damping": 0.5}),
            "act": ComponentSpec(
                type="feedbax.ActuationInput",
                params={"port_name": "u", "source_kind": "force"},
            ),
            "sense": ComponentSpec(
                type="feedbax.SensorOutput",
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
        solver={"solver_type": "euler", "dt": 0.001},
    )


def _acausal_graph_spec(interior: AcausalGraphSpec) -> GraphSpec:
    return GraphSpec(
        nodes={
            "plant": ComponentSpec(
                type="feedbax.AcausalSystem",
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


def test_driven_massless_spring_endpoint_is_rejected_before_stepping() -> None:
    connections = [
        (("wall", "flange"), ("support", "flange_a")),
        (("mass", "flange"), ("support", "flange_b")),
        (("mass", "flange"), ("spring", "flange_a")),
        (("spring", "flange_b"), ("drive", "flange")),
        (("spring", "flange_b"), ("sense", "flange")),
    ]
    spec = AcausalGraphSpec(
        physical_domain="translational",
        nodes={
            "wall": ComponentSpec(type="feedbax.Ground"),
            "mass": ComponentSpec(type="feedbax.Mass"),
            "support": ComponentSpec(type="feedbax.LinearSpring"),
            "spring": ComponentSpec(type="feedbax.LinearSpring"),
            "drive": ComponentSpec(
                type="feedbax.ActuationInput", params={"port_name": "u", "source_kind": "force"}
            ),
            "sense": ComponentSpec(
                type="feedbax.SensorOutput", params={"port_name": "pos", "quantity": "position"}
            ),
        },
        connections=[{"a": a, "b": b} for a, b in connections],
        solver={"solver_type": "euler", "dt": 0.01},
    )
    registry = _registry()
    with pytest.raises(GraphCompilationError, match="require inertia"):
        spec_to_graph(
            _acausal_graph_spec(spec),
            registry,
            input_prototypes={("__graph__", "u"): jnp.zeros(1)},
        )
    with pytest.raises(ValueError, match="require inertia"):
        AcausalSystem(
            elements={
                "wall": Ground("wall"),
                "mass": Mass("mass"),
                "support": LinearSpring("support"),
                "spring": LinearSpring("spring"),
                "drive": ForceSource("drive"),
                "sense": PositionSensor("sense"),
            },
            connections=[AcausalConnection(a, b) for a, b in connections],
        )
    report = compile_acausal_authoring_report(
        spec, node_path=["plant"], component_registry=registry
    )
    assert report.status == "error"
    diagnostic = next(d for d in report.diagnostics if d.code == "acausal.unbalanced")
    assert len(diagnostic.variables) == 2
    assert all(
        name.rsplit(".", 1)[0] in {"drive.flange", "sense.flange", "spring.flange_b"}
        for name in diagnostic.variables
    )


def _prescribed_motion_graph(*, bind_input: bool = True) -> GraphSpec:
    return GraphSpec(
        nodes={
            "plant": ComponentSpec(
                type="feedbax.AcausalSystem",
                input_ports=["motion"],
                output_ports=["velocity"],
            ),
        },
        input_ports=["command"] if bind_input else [],
        input_bindings={"command": ("plant", "motion")} if bind_input else {},
        output_ports=["velocity"],
        output_bindings={"velocity": ("plant", "velocity")},
        subgraphs={
            "plant": AcausalGraphSpec(
                physical_domain="translational",
                nodes={
                    "wall": ComponentSpec(type="feedbax.Ground"),
                    "spring": ComponentSpec(
                        type="feedbax.LinearSpring", params={"stiffness": 10.0}
                    ),
                    "drive": ComponentSpec(
                        type="feedbax.ActuationInput",
                        params={"port_name": "motion", "source_kind": "prescribed_motion"},
                    ),
                    "sense": ComponentSpec(
                        type="feedbax.SensorOutput",
                        params={"port_name": "velocity", "quantity": "velocity"},
                    ),
                },
                connections=[
                    {"a": ("drive", "flange"), "b": ("sense", "flange")},
                    {"a": ("drive", "flange"), "b": ("spring", "flange_a")},
                    {"a": ("spring", "flange_b"), "b": ("wall", "flange")},
                ],
                solver={"solver_type": "euler", "dt": 0.001},
            ),
        },
    )


def test_compiled_prescribed_motion_uses_velocity_from_each_pair() -> None:
    spec = _prescribed_motion_graph()
    graph = compile_graph(
        GraphDocument(graph=spec),
        _registry(),
        input_prototypes={("__graph__", "command"): jnp.zeros(2)},
    ).graph
    key = jax.random.PRNGKey(0)

    outputs, _, _ = graph.rollout_with_trace(
        {"command": jnp.array([[2.0, 0.0], [2.0, -1.0], [2.0, 0.0]])},
        graph.init_state(key=key),
        key=key,
        n_steps=3,
        trace=(),
    )

    assert jnp.array_equal(outputs["velocity"], jnp.array([0.0, -1.0, 0.0]))


@pytest.mark.parametrize(
    "command",
    [2.0, [[[2.0, 0.0]]] * 3, None],
    ids=["scalar", "extra-axis", "missing"],
)
def test_compiled_prescribed_motion_rejects_invalid_rollout_input(
    command: float | list[list[list[float]]] | None,
) -> None:
    graph = compile_graph(GraphDocument(graph=_prescribed_motion_graph()), _registry()).graph
    key = jax.random.PRNGKey(0)
    inputs = {} if command is None else {"command": jnp.asarray(command)}
    state = graph.init_state(key=key)

    with pytest.raises(ValueError, match="motion"):
        graph.rollout_with_trace(inputs, state, key=key, n_steps=3, trace=())


def test_named_effort_source_requires_its_own_input() -> None:
    plant = compile_acausal_graph(_msd_interior(), "plant", _registry())
    key = jax.random.PRNGKey(0)
    state = plant.init_state(key=key)

    with pytest.raises(ValueError, match="Missing required acausal source input 'u'"):
        plant({"input": jnp.asarray(2.0)}, state, key=key)


def test_compile_graph_rejects_scalar_prescribed_motion_prototype() -> None:
    spec = _prescribed_motion_graph()

    with pytest.raises(GraphCompilationError, match="motion"):
        compile_graph(
            GraphDocument(graph=spec),
            _registry(),
            input_prototypes={("__graph__", "command"): jnp.asarray(2.0)},
        )


def test_rollout_hook_can_supply_unbound_prescribed_motion() -> None:
    def supply_motion(context: RolloutStepContext) -> RolloutStepHookResult:
        return RolloutStepHookResult(
            port_values={**context.port_values, ("plant", "motion"): jnp.array([2.0, 1.0])},
        )

    spec = _prescribed_motion_graph(bind_input=False)
    graph = compile_graph(GraphDocument(graph=spec), _registry()).graph
    key = jax.random.PRNGKey(0)

    outputs, _, _ = graph.rollout_with_trace(
        {},
        graph.init_state(key=key),
        key=key,
        n_steps=3,
        trace=(),
        rollout_step_hook=supply_motion,
    )

    assert jnp.array_equal(outputs["velocity"], jnp.ones(3))


def _planar_arm_spec(
    registry: ComponentRegistry,
    *,
    solver: SolverConfigSpec | None = None,
    bind_input: bool = False,
) -> GraphSpec:
    template = registry.get("feedbax.TwoLinkArm6Muscle")
    assert template is not None
    interior = template.template_graph
    assert isinstance(interior, AcausalGraphSpec)
    if solver is not None:
        interior = interior.model_copy(update={"solver": solver})
    return GraphSpec(
        nodes={
            "arm": ComponentSpec(
                type="feedbax.AcausalSystem",
                input_ports=["excitation"],
                output_ports=["state"],
            ),
        },
        input_ports=["excitation"] if bind_input else [],
        input_bindings={"excitation": ("arm", "excitation")} if bind_input else {},
        output_ports=["state"],
        output_bindings={"state": ("arm", "state")},
        subgraphs={"arm": interior},
    )


@pytest.fixture
def planar_arm_graph() -> Graph:
    registry = _registry()
    return spec_to_graph(_planar_arm_spec(registry), registry)


def test_planar_tsit5_step_matches_direct_diffrax_after_document_roundtrip() -> None:
    registry = _registry()
    dt = 0.01
    solver = SolverConfigSpec(solver_type="tsit5", dt=dt)
    spec = _planar_arm_spec(registry, solver=solver, bind_input=True)
    document = GraphDocument.model_validate_json(GraphDocument(graph=spec).model_dump_json())
    executable = compile_graph(document, registry)
    resolved_interior = executable.resolved.graph.subgraphs["arm"]
    assert resolved_interior.solver == solver
    assert (
        compile_acausal_authoring_report(
            resolved_interior, node_path=["arm"], component_registry=registry
        ).status
        == "ok"
    )

    graph = executable.graph
    arm = graph.nodes["arm"]
    key = jax.random.PRNGKey(0)
    state = graph.init_state(key=key)
    initial = state.get(arm.state_index).plant
    excitation = jnp.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    outputs, _, _ = graph.step({"excitation": excitation}, state, key=key)
    actual = outputs["state"].plant.muscles.activations

    term = dfx.ODETerm(arm.plant.vector_field)
    tsit5 = dfx.Tsit5()
    solver_state = tsit5.init(term, 0.0, dt, initial, excitation)
    expected, _, _, _, result = tsit5.step(
        term, 0.0, dt, initial, excitation, solver_state, made_jump=True
    )
    assert result == dfx.RESULTS.successful
    assert jnp.allclose(actual, expected.muscles.activations)


@pytest.mark.parametrize(
    ("domain", "solver", "message"),
    [
        (
            "scalar",
            {"solver_type": "euler", "root_finder": {}},
            r"euler.*root_finder|root_finder.*euler",
        ),
        (
            "planar",
            {"solver_type": "implicit_euler"},
            r"[Pp]lanar.*implicit_euler",
        ),
    ],
)
def test_unsupported_solver_is_rejected_by_report_and_public_compile(
    domain: str, solver: dict, message: str
) -> None:
    registry = _registry()
    config = SolverConfigSpec.model_validate({"dt": 0.01, **solver})
    if domain == "planar":
        spec = _planar_arm_spec(registry, solver=config)
        node = "arm"
    else:
        interior = _msd_interior().model_copy(update={"solver": config})
        spec = _acausal_graph_spec(interior)
        node = "plant"

    report = compile_acausal_authoring_report(
        spec.subgraphs[node], node_path=[node], component_registry=registry
    )
    assert report.status == "error"
    assert any(diagnostic.code == "acausal.invalid_solver" for diagnostic in report.diagnostics)
    with pytest.raises(GraphCompilationError, match=message):
        compile_graph(GraphDocument(graph=spec), registry)


@pytest.mark.parametrize("shape", [(), (1, 6)], ids=["scalar", "extra-axis"])
def test_planar_arm_rejects_invalid_late_excitation(
    planar_arm_graph: Graph, shape: tuple[int, ...]
) -> None:
    def supply_excitation(context: RolloutStepContext) -> RolloutStepHookResult:
        return RolloutStepHookResult(
            port_values={**context.port_values, ("arm", "excitation"): jnp.full(shape, 0.5)},
        )

    key = jax.random.PRNGKey(0)
    state = planar_arm_graph.init_state(key=key)
    with pytest.raises(ValueError, match=r"(?=.*excitation)(?=.*shape)(?=.*\(6,\))"):
        planar_arm_graph.step({}, state, key=key, rollout_step_hook=supply_excitation)


def test_planar_arm_accepts_late_excitation(planar_arm_graph: Graph) -> None:
    def supply_excitation(context: RolloutStepContext) -> RolloutStepHookResult:
        return RolloutStepHookResult(
            port_values={
                **context.port_values,
                ("arm", "excitation"): jnp.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
            },
        )

    key = jax.random.PRNGKey(0)
    state = planar_arm_graph.init_state(key=key)
    driven, _, _ = planar_arm_graph.step({}, state, key=key, rollout_step_hook=supply_excitation)

    activations = driven["state"].plant.muscles.activations
    assert activations[0] > 0
    assert jnp.array_equal(activations[1:], jnp.zeros(5))


def test_authoring_report_exposes_prescribed_motion_pair() -> None:
    spec = _prescribed_motion_graph()
    report = compile_acausal_authoring_report(
        spec.subgraphs["plant"],
        node_path=["plant"],
        component_registry=_registry(),
    )

    assert report.status == "ok"
    assert report.derived_interface is not None
    assert report.derived_interface["inputs"]["motion"]["shape"] == [2]


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
            "wall": ComponentSpec(type="feedbax.Ground"),
            "spring": ComponentSpec(type="feedbax.LinearSpring", params={"stiffness": 10.0}),
            "damper": ComponentSpec(type="feedbax.LinearDamper", params={"damping": 0.5}),
            "free": ComponentSpec(type="feedbax.BoundaryPort", params={"port_name": "flange"}),
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
            "actuator": ComponentSpec(type="feedbax.AcausalSystem", input_ports=["flange"]),
            "mass": ComponentSpec(type="feedbax.Mass", params={"mass": 1.0}),
            "act": ComponentSpec(
                type="feedbax.ActuationInput",
                params={"port_name": "u", "source_kind": "force"},
            ),
            "sense": ComponentSpec(
                type="feedbax.SensorOutput",
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
        nodes={"plant": ComponentSpec(type="feedbax.AcausalSystem")},
    )

    with pytest.raises(ValueError, match="feedbax.domain.acausal"):
        spec_to_graph(spec, _registry())


@pytest.mark.parametrize(
    ("solver_name", "expected_position"),
    [
        # Backward Euler from rest: x = dt*v and v = dt*(1 - 10*x - 0.5*v).
        ("implicit_euler", 0.01 / 1.15),
        # Exact underdamped step response at t=0.1 for m=1, k=10, c=0.5, force=1.
        ("kvaerno5", 0.00487699),
    ],
)
def test_scalar_implicit_rollout_respects_authored_root_budget(
    solver_name: str, expected_position: float
) -> None:
    key = jax.random.PRNGKey(0)
    registry = _registry()
    dt = 0.1

    def rollout(max_steps: int) -> jax.Array:
        solver = SolverConfigSpec.model_validate(
            {
                "solver_type": solver_name,
                "dt": dt,
                "root_finder": {"rtol": 0, "max_steps": max_steps},
            }
        )
        interior = _msd_interior().model_copy(update={"solver": solver})
        graph = spec_to_graph(_acausal_graph_spec(interior), registry)
        outputs, _, _ = graph.rollout_with_trace(
            {"u": jnp.ones(1)}, graph.init_state(key=key), key=key, n_steps=1, trace=()
        )
        return jax.block_until_ready(outputs["pos"])

    with pytest.raises(RuntimeError, match=r"maximum number of steps.*nonlinear solver"):
        rollout(max_steps=1)

    assert rollout(max_steps=8)[0].item() == pytest.approx(expected_position, abs=1e-7)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("explicit_order", [False, True])
def test_persisted_v1_gear_has_reduced_report_and_physical_bound_outputs(nested, explicit_order):
    from feedbax.contracts.migrations import migrate_acausal_graph_spec

    spec = AcausalGraphSpec(
        schema_id="feedbax.spec.acausal_graph",
        schema_version="feedbax.spec.acausal_graph.v1",
        physical_domain="rotational",
        nodes={
            "a": ComponentSpec(type="feedbax.Inertia", params={"inertia": 2.0}),
            "b": ComponentSpec(type="feedbax.Inertia", params={"inertia": 3.0}),
            "gear": ComponentSpec(type="feedbax.GearRatio", params={"ratio": 2.0}),
            "wall": ComponentSpec(type="feedbax.RotationalGround"),
            "spring": ComponentSpec(type="feedbax.TorsionalSpring", params={"stiffness": 4.0}),
            "drive": ComponentSpec(
                type="feedbax.ActuationInput", params={"port_name": "u", "source_kind": "torque"}
            ),
            "angle": ComponentSpec(
                type="feedbax.SensorOutput", params={"port_name": "qb", "quantity": "angle"}
            ),
            "velocity": ComponentSpec(
                type="feedbax.SensorOutput",
                params={"port_name": "vb", "quantity": "angular_velocity"},
            ),
            "torque": ComponentSpec(
                type="feedbax.SensorOutput", params={"port_name": "tb", "quantity": "torque"}
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
                "left": ComponentSpec(
                    type="feedbax.BoundaryPort", params={"port_name": "flange_a"}
                ),
                "right": ComponentSpec(
                    type="feedbax.BoundaryPort", params={"port_name": "flange_b"}
                ),
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
                        type="feedbax.AcausalSystem", input_ports=["flange_a", "flange_b"]
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
                "a": ComponentSpec(type="feedbax.AngleSensor"),
                "b": ComponentSpec(type="feedbax.AngleSensor"),
            }
        }
    )
    invalid_report = compile_acausal_authoring_report(
        unsupported, node_path=["plant"], component_registry=_registry()
    )
    assert invalid_report.status == "error"
    assert any(d.code == "acausal.unbalanced" for d in invalid_report.diagnostics)


def test_invalid_persisted_gear_has_actionable_authoring_diagnostic():
    spec = AcausalGraphSpec(
        physical_domain="rotational",
        nodes={"gear": ComponentSpec(type="feedbax.GearRatio", params={"ratio": 0.0})},
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


@pytest.mark.parametrize("with_adapter", [False, True])
def test_persisted_direct_element_signal_input_binds_with_adapters(with_adapter):
    from feedbax.contracts.migrations import migrate_acausal_graph_spec

    nodes = {
        "mass": ComponentSpec(type="feedbax.Mass", params={"mass": 2.0}),
        "wall": ComponentSpec(type="feedbax.Ground"),
        "spring": ComponentSpec(type="feedbax.LinearSpring", params={"stiffness": 1.0}),
        "muscle": ComponentSpec(
            type="feedbax.RigidTendonHillMuscle", params={"max_isometric_force": 10.0}
        ),
        "velocity": ComponentSpec(
            type="feedbax.SensorOutput", params={"port_name": "v", "quantity": "velocity"}
        ),
    }
    connections = [
        {"a": a, "b": b}
        for a, b in [
            (("wall", "flange"), ("spring", "flange_a")),
            (("spring", "flange_b"), ("mass", "flange")),
            (("muscle", "flange"), ("mass", "flange")),
            (("velocity", "flange"), ("mass", "flange")),
        ]
    ]
    if with_adapter:
        # Authored adapters precede direct signal inputs, even against lexical order.
        nodes["adapter"] = ComponentSpec(
            type="feedbax.ActuationInput", params={"port_name": "z_drive", "source_kind": "force"}
        )
        connections.append({"a": ("adapter", "flange"), "b": ("mass", "flange")})
    spec = AcausalGraphSpec(
        physical_domain="translational",
        nodes=nodes,
        connections=connections,
        solver={"solver_type": "euler", "dt": 0.01},
    )
    import json

    payload = json.loads(spec.model_dump_json())
    loaded = AcausalGraphSpec.model_validate(migrate_acausal_graph_spec(payload).payload)
    report = compile_acausal_authoring_report(
        loaded, node_path=["plant"], component_registry=_registry()
    )
    assert report.status == "ok", report.diagnostics
    system = compile_acausal_graph(loaded, "plant", _registry())
    assert system.input_ports == (("z_drive", "muscle") if with_adapter else ("muscle",))
    state = system.init_state(key=jax.random.PRNGKey(0))
    inputs = {"muscle": jnp.array([1.0])}
    if with_adapter:
        inputs["z_drive"] = jnp.array([4.0])
    outputs, _ = system(inputs, state, key=jax.random.PRNGKey(1))
    assert outputs["v"] == pytest.approx(0.07 if with_adapter else 0.05)
