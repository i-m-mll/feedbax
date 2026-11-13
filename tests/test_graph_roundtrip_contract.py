from __future__ import annotations

from copy import deepcopy
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from feedbax.component_registry import ComponentRegistry, required_interior_domain
from feedbax.component_registry.retained import retained_component_registry
from feedbax.components import penzai
from feedbax.contracts.domain import CAUSAL_DOMAIN_ID
from feedbax.contracts.graph import ComponentSpec, GraphSpec, ParamSchema
from feedbax.compiler.serialization import graph_to_spec
from feedbax.mechanics.analytical_plant import AnalyticalMusculoskeletalPlant
from feedbax.mechanics.skeleton.arm import TwoLinkArm
from feedbax.models.networks import PopulationStructure, recurrent_controller
from tests.support.graph_compiler_test_support import spec_to_graph


def _registry_cases() -> list[str]:
    registry = retained_component_registry()
    return [
        component_type
        for component_type in registry.executable_names()
        if required_interior_domain(component_type, registry) is None
        and registry.get(component_type).domain == CAUSAL_DOMAIN_ID
        and not component_type.startswith("feedbax.equinox.")
    ]


def _map_value(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.125
    if isinstance(value, str):
        return f"{value}_nondefault"
    if isinstance(value, (list, tuple)):
        return [_map_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _map_value(item) for key, item in value.items()}
    return value


_SCALAR_CONTROL_TYPES = (
    "feedbax.Derivative",
    "feedbax.Integrator",
    "feedbax.IntegratorDiscrete",
    "feedbax.PID",
    "feedbax.PIDDiscrete",
    "feedbax.UnitDelay",
    "feedbax.ZeroOrderHold",
)

_SHAPED_SIGNAL_TYPES = (
    "feedbax.BandPassFilter",
    "feedbax.Channel",
    "feedbax.DelayLine",
    "feedbax.FirstOrderFilter",
    "feedbax.HighPassFilter",
    "feedbax.RateLimiter",
)


_NONE_VALUES: dict[tuple[str, str], Any] = {
    **{
        (kind, "input_dtype"): "float16"
        for kind in (*_SHAPED_SIGNAL_TYPES, *_SCALAR_CONTROL_TYPES)
    },
    **{(kind, "input_shape"): [3] for kind in _SHAPED_SIGNAL_TYPES},
    ("feedbax.PID", "integral_limit"): 2.0,
    ("feedbax.PIDDiscrete", "output_limit"): 2.0,
    ("feedbax.LinearStateSpace", "initial_state"): [0.1, 0.2, 0.3, 0.4],
    ("feedbax.StructuralLinearStateSpace", "initial_state"): [0.1, 0.2, 0.3, 0.4],
    ("feedbax.AssembleFeedback", "layout"): {"list": [None]},
    ("feedbax.AffineFeedbackController", "bias"): [0.25],
    ("feedbax.AffineFeedbackController", "feedforward"): [0.5],
    ("feedbax.Channel", "noise_role"): "motor_command",
    ("feedbax.Channel", "noise_timing"): "post_controller",
    ("feedbax.Linear", "bias_init"): 0.125,
    ("feedbax.DelayedReaches", "n_control_stages"): 49,
    ("feedbax.DelayedReaches", "go_cue_event_name"): "go",
    ("feedbax.DynamicsMatrixPerturb", "mass"): 1.25,
    ("feedbax.LinearStateSpace", "B_w"): [[0.1], [0.2], [0.3], [0.4]],
    ("feedbax.StructuralLinearStateSpace", "B_w"): [[0.1], [0.2], [0.3], [0.4]],
    ("feedbax.StateFeedbackSelector", "expected_state_dim"): 4,
    ("feedbax.StateFeedbackSelector", "output_size"): 4,
    ("feedbax.ConstantInput", "arrays"): [0.25, -0.5],
    ("feedbax.NetworkClamp", "unit_spec"): [0.25, None],
    ("feedbax.NetworkConstantInput", "unit_spec"): [0.25, None],
}

_INTERVENTION_PAYLOAD_FIELDS = {
    "feedbax.ConstantInput": "arrays",
    "feedbax.NetworkClamp": "unit_spec",
    "feedbax.NetworkConstantInput": "unit_spec",
}


_VALUES: dict[tuple[str, str], Any] = {
    ("feedbax.PenzaiAdapter", "builder_name"): "test_roundtrip_sigmoid",
    ("feedbax.StateSpace", "A"): [[-0.5]],
    ("feedbax.StateSpace", "B"): [[0.5]],
    ("feedbax.StateSpace", "C"): [[0.5]],
    ("feedbax.StateSpace", "D"): [[0.5]],
    ("feedbax.TransferFunction", "num"): [0.5],
    ("feedbax.TransferFunction", "den"): [1.0, 0.5],
    ("feedbax.AffineValueComposer", "feature_rules"): [{"kind": "identity", "state_slice": [1, 2]}],
    ("feedbax.Channel", "add_noise"): False,
    ("feedbax.Channel", "include_zero_noise_terms"): True,
    ("feedbax.DelayedReaches", "hold_epochs"): [0],
    ("feedbax.DelayedReaches", "move_epochs"): [1, 2],
    ("feedbax.DelayedReaches", "target_visible_from_start"): True,
    ("feedbax.DelayedReaches", "target_on_epochs"): [0, 2],
    ("feedbax.SelectState", "paths"): ["effector.force"],
    ("feedbax.StateFeedbackSelector", "state_slices"): {
        "position": {"start": 1, "stop": 3},
        "velocity": {"start": 3, "stop": 5},
    },
    ("feedbax.StateFeedbackSelector", "channels"): [
        {"slice": "position", "transform": "negate"},
        {"slice": "velocity", "transform": "identity"},
    ],
    ("feedbax.StructuralLinearStateSpace", "delta_A"): {
        "schema_id": "feedbax.spec.component_param.array_value",
        "schema_version": "feedbax.spec.component_param.array_value.v1",
        "shape": [4, 4],
        "dtype": "float32",
        "nonfinite": "forbid",
        "encoding": "sparse_coo",
        "fill": 0.0,
        "entries": [{"coordinate": [0, 1], "value": 0.25}],
    },
    ("feedbax.ThresholdLatchedForce", "state_selector"): {"kind": "fixed", "path": ["pos", 1]},
}


def _non_default_value(component_type: str, schema: ParamSchema) -> Any:
    key = (component_type, schema.name)
    if key in _VALUES:
        return deepcopy(_VALUES[key])
    if key == ("feedbax.JointMuscleGeometry", "paths"):
        paths = deepcopy(schema.default)
        paths[0]["params"]["reference_length"] *= 1.1
        return paths
    if schema.name == "dtype":
        return "float16"
    if schema.default is None:
        return deepcopy(_NONE_VALUES[key])
    if schema.type == "bool":
        return not bool(schema.default)
    if schema.type == "enum":
        return next(option for option in schema.options or [] if option != schema.default)
    if schema.type == "int":
        candidate = int(schema.default) + 1
        if schema.max is not None and candidate > schema.max:
            candidate = int(schema.default) - 1
        return candidate
    if schema.type == "float":
        candidate = float(schema.default) + 0.125
        if schema.max is not None and candidate > schema.max:
            candidate = float(schema.default) - 0.125
        return candidate
    if schema.type == "str":
        return "nondefault"
    if schema.type == "bounds2d":
        return [[-0.75, -0.5], [0.75, 0.5]]
    return _map_value(deepcopy(schema.default))


def _ports(registry: ComponentRegistry, component_type: str, params: dict[str, Any]):
    if component_type == "feedbax.PenzaiAdapter":
        return [params["input_port"]], [params["output_port"]]
    meta = registry.get(component_type)
    assert meta is not None
    layout = registry.dynamic_port_layout(component_type, params)
    if layout is None:
        return list(meta.input_ports), list(meta.output_ports)
    return list(layout.input_ports), list(layout.output_ports)


def _contextualize_params(
    component_type: str,
    param_name: str | None,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Keep one-field registry probes semantically valid and observable."""
    params = {name: value for name, value in params.items() if value is not None}
    if component_type == "feedbax.StateSpace":
        for name, value in {"A": [[-1.0]], "B": [[1.0]], "C": [[1.0]], "D": [[0.0]]}.items():
            params.setdefault(name, value)
    elif component_type == "feedbax.PenzaiAdapter":
        params.setdefault("builder_name", "test_roundtrip_relu")
    elif component_type == "feedbax.TransferFunction":
        params.setdefault("num", [1.0])
        params.setdefault("den", [1.0, 1.0])
    elif component_type == "feedbax.Channel":
        params.setdefault("input_shape", [2])
        if param_name in {"additive_noise_std", "signal_dependent_noise_std"}:
            params["noise_model"] = "signal_dependent_plus_additive"
        if param_name == "include_zero_noise_terms":
            params["noise_std"] = 0.0
    elif component_type == "feedbax.AssembleFeedback" and param_name == "n_inputs":
        params["layout"] = {"tuple": [None] * params["n_inputs"]}
    elif component_type == "feedbax.AffineValueComposer" and param_name == "output_block_size":
        size = int(params["output_block_size"])
        params["gain_init"] = [[0.0] for _ in range(size)]
        params["bias_init"] = [0.0 for _ in range(size)]
    elif component_type == "feedbax.DelayedReaches" and param_name == "target_visible_from_start":
        params["target_on_epochs"] = [0, 1, 2]
    elif component_type == "feedbax.DelayedReaches" and param_name == "n_control_stages":
        params.pop("n_steps", None)
    elif component_type == "feedbax.DelayedReaches" and param_name == "move_epochs":
        params["hold_epochs"] = [0]
    elif component_type == "feedbax.DynamicsMatrixPerturb":
        params.setdefault("mass", 1.0)
    elif (
        param_name in {"active", "payload_dtype"} and component_type in _INTERVENTION_PAYLOAD_FIELDS
    ):
        payload_field = _INTERVENTION_PAYLOAD_FIELDS[component_type]
        params[payload_field] = deepcopy(_NONE_VALUES[component_type, payload_field])
    elif component_type == "feedbax.StateFeedbackSelector" and param_name == "output_size":
        params["channels"] = [{"slice": "position", "transform": "identity"}]
        params["output_size"] = 2
    return params


def _single_node_spec(
    registry: ComponentRegistry,
    component_type: str,
    params: dict[str, Any],
) -> tuple[GraphSpec, dict[tuple[str, str], Any]]:
    input_ports, output_ports = _ports(registry, component_type, params)
    spec = GraphSpec(
        nodes={
            "node": ComponentSpec(
                type=component_type,
                params=params,
                input_ports=input_ports,
                output_ports=output_ports,
            )
        }
    )
    if component_type == "feedbax.AffineFeedbackController":
        feedback_shape = jnp.asarray(params["gain"]).shape[-1:]
        return spec, {("node", "feedback"): jnp.zeros(feedback_shape)}
    prototype_shape = (2,)
    if component_type in _SHAPED_SIGNAL_TYPES and isinstance(
        params.get("input_shape"), list
    ):
        prototype_shape = tuple(int(dim) for dim in params["input_shape"])
    elif component_type in _SCALAR_CONTROL_TYPES:
        prototype_shape = (params["n_dims"],)
    elif component_type == "feedbax.VectorElements":
        prototype_shape = (params["size"],)
    elif component_type == "feedbax.RigidTendonHillMuscleThelen":
        count = params["n_muscles"]
        prototype_shape = () if count == 1 else (count,)
    prototypes = {
        ("node", port): jnp.zeros(prototype_shape, dtype=params.get("input_dtype"))
        for port in input_ports
    }
    if component_type == "feedbax.SelectState":
        point_mass = registry.get("feedbax.PointMass")
        assert point_mass is not None and point_mass.builder is not None
        prototypes[("node", "state")] = point_mass.builder({"dt": 0.01}).state_index.init
    elif component_type == "feedbax.Demux":
        prototypes[("node", "input")] = jnp.zeros(sum(params["sizes"]))
    elif component_type == "feedbax.AffineValueComposer":
        block_size = int(params["output_block_size"])
        prototypes[("node", "base")] = jnp.zeros(block_size)
        prototypes[("node", "gain")] = jnp.zeros_like(jnp.asarray(params["gain_init"]))
        prototypes[("node", "bias")] = jnp.zeros(block_size)
    return spec, prototypes


@pytest.fixture
def roundtrip_registry(component_type: str, monkeypatch: pytest.MonkeyPatch) -> ComponentRegistry:
    if component_type == "feedbax.PenzaiAdapter":
        pytest.importorskip("penzai")
        from penzai import pz

        monkeypatch.setattr(penzai, "_PENZAI_MODEL_BUILDERS", {})
        penzai.register_penzai_builder(
            "test_roundtrip_relu", lambda _: pz.nn.Elementwise(jax.nn.relu)
        )
        penzai.register_penzai_builder(
            "test_roundtrip_sigmoid", lambda _: pz.nn.Elementwise(jax.nn.sigmoid)
        )
    return retained_component_registry()


@pytest.mark.parametrize("component_type", _registry_cases())
def test_registered_buildable_component_param_round_trips(
    component_type: str,
    roundtrip_registry: ComponentRegistry,
) -> None:
    """Every declared leaf parameter is exercised non-default and must survive."""
    registry = roundtrip_registry
    restore_registry = retained_component_registry()
    meta = registry.get(component_type)
    assert meta is not None
    baseline_params = _contextualize_params(component_type, None, deepcopy(meta.default_params))
    baseline_spec, baseline_prototypes = _single_node_spec(
        registry, component_type, baseline_params
    )
    baseline = graph_to_spec(
        spec_to_graph(baseline_spec, registry, baseline_prototypes), registry
    )

    probes: list[ParamSchema | None] = list(meta.param_schema)
    if not probes or component_type == "feedbax.PenzaiAdapter":
        probes.insert(0, None)

    for schema in probes:
        param_name = schema.name if schema is not None else None
        witness = f"{component_type}.{param_name}"
        if schema is None:
            params = baseline_params
            serialized, prototypes = baseline, baseline_prototypes
        else:
            params = deepcopy(baseline_params)
            params[schema.name] = _non_default_value(component_type, schema)
            params = _contextualize_params(component_type, schema.name, params)
            spec, prototypes = _single_node_spec(registry, component_type, params)
            serialized = graph_to_spec(spec_to_graph(spec, registry, prototypes), registry)
        emitted = serialized.nodes["node"]
        assert emitted.type == component_type, witness

        if param_name is not None and (
            (component_type, param_name) == ("feedbax.DelayedReaches", "move_epochs")
            or (
                component_type in _SHAPED_SIGNAL_TYPES
                and param_name in {"input_shape", "input_dtype"}
            )
            or component_type in _INTERVENTION_PAYLOAD_FIELDS
        ):
            assert emitted.params[param_name] == params[param_name], witness
        if param_name is not None and param_name != "schema_version":
            assert emitted.params != baseline.nodes["node"].params, (
                f"{witness} did not affect the canonical GraphSpec"
            )

        second = spec_to_graph(serialized, restore_registry, prototypes)
        restored_params = graph_to_spec(second, restore_registry).nodes["node"].params
        assert restored_params == emitted.params, witness


def test_two_link_arm_nondefault_lengths_round_trip() -> None:
    registry = ComponentRegistry()
    spec = GraphSpec(
        nodes={
            "arm": ComponentSpec(
                type="feedbax.TwoLinkArm",
                params={"dt": 0.02, "link_lengths": [0.41, 0.29]},
                input_ports=["force"],
                output_ports=["effector", "state"],
            )
        }
    )

    restored = spec_to_graph(graph_to_spec(spec_to_graph(spec, registry)), registry)
    arm = restored.nodes["arm"].plant.skeleton
    assert isinstance(arm, TwoLinkArm)
    assert jnp.allclose(arm.l, jnp.asarray([0.41, 0.29]))


@pytest.mark.parametrize("solver_params", [{}, {"solver_type": "tsit5"}], ids=["default", "tsit5"])
def test_analytical_plant_authored_contract_round_trips(solver_params: dict[str, str]) -> None:
    registry = ComponentRegistry()
    params = {
        "dt": 0.02,
        "n_steps": 4,
        "tau_act": 0.015,
        "tau_deact": 0.055,
        "clip_states": False,
        "solver_type": "euler",
        **solver_params,
    }
    spec = GraphSpec(
        nodes={
            "plant": ComponentSpec(
                type="feedbax.AnalyticalMusculoskeletalPlant",
                params=params,
                input_ports=["excitation"],
                output_ports=["effector", "state"],
            )
        }
    )

    first = spec_to_graph(spec, registry)
    serialized = graph_to_spec(first)
    assert serialized.nodes["plant"].params == params
    restored = spec_to_graph(serialized, registry).nodes["plant"]
    assert isinstance(restored.plant, AnalyticalMusculoskeletalPlant)
    assert restored.backend.n_substeps == 4
    assert restored.plant.to_params() == {
        "tau_act": 0.015,
        "tau_deact": 0.055,
        "clip_states": False,
    }


@pytest.mark.parametrize(
    "params",
    [
        {"noise": {"model": "additive_gaussian", "std": 0.2}},
        {"noise_model": "multiplicative_gaussian", "signal_dependent_noise_std": 0.2},
        {"noise_model": "additive_gaussian", "additive_std": 0.2},
    ],
)
def test_channel_rejects_noncanonical_noise_spellings(params: dict[str, Any]) -> None:
    registry = ComponentRegistry()
    spec = GraphSpec(
        nodes={
            "channel": ComponentSpec(
                type="feedbax.Channel",
                params={"delay": 0, "input_shape": [2], **params},
                input_ports=["input"],
                output_ports=["output"],
            )
        }
    )
    with pytest.raises(
        Exception,
        match="canonical Channel noise parameters|Unsupported Channel noise_model",
    ):
        spec_to_graph(spec, registry)


def test_programmatic_recurrent_controller_graph_round_trips() -> None:
    population = PopulationStructure.from_indices(
        input_only_indices=[0],
        readout_only_indices=[1],
        recurrent_only_indices=[2],
        input_readout_indices=[3],
    )
    network = recurrent_controller(
        input_size=5,
        hidden_size=4,
        out_size=2,
        encoding_size=3,
        hidden_nonlinearity=jax.nn.relu,
        out_nonlinearity=jax.nn.sigmoid,
        hidden_noise_std=0.2,
        population_structure=population,
        dtype=jnp.float16,
        key=jax.random.PRNGKey(0),
    )
    spec = graph_to_spec(network)
    roundtrip = graph_to_spec(spec_to_graph(spec, ComponentRegistry()))

    assert roundtrip == spec
    assert roundtrip.nodes["cell"].params["dtype"] == "float16"
    assert roundtrip.nodes["readout"].params["initialization"] == "zero_bias"
    assert roundtrip.nodes["hidden_noise"].params["input_dtype"] == "float16"
