from __future__ import annotations

from copy import deepcopy
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from feedbax.component_registry import ComponentRegistry, required_interior_domain
from feedbax.contracts.graph import ComponentSpec, GraphSpec, ParamSchema
from feedbax.contracts.graphs.serialization import graph_to_spec
from feedbax.mechanics.analytical_plant import AnalyticalMusculoskeletalPlant
from feedbax.mechanics.skeleton.arm import TwoLinkArm
from feedbax.models.networks import PopulationStructure, SimpleStagedNetwork
from feedbax.runtime.channel import Channel
from feedbax.runtime.components import Activation, Linear
from feedbax.runtime.graph import Graph
from tests.graph_compiler_test_support import spec_to_graph


def _registry_cases() -> list[pytest.ParameterSet]:
    registry = ComponentRegistry(load_user_components=False)
    cases = []
    for component_type in registry.executable_names():
        if required_interior_domain(component_type, registry) is not None:
            continue
        meta = registry.get(component_type)
        assert meta is not None
        params = meta.param_schema or [None]
        for schema in params:
            param_name = None if schema is None else schema.name
            marks = []
            if component_type == "Stabilization":
                marks.append(
                    pytest.mark.xfail(
                        strict=True,
                        reason=(
                            "fbed0f2: registered Stabilization remains abstract and cannot build"
                        ),
                    )
                )
            cases.append(
                pytest.param(
                    component_type,
                    param_name,
                    id=f"{component_type}-{param_name or 'no-params'}",
                    marks=marks,
                )
            )
    return cases


def _map_value(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.125
    if isinstance(value, str):
        return f"{value}_nondefault"
    if isinstance(value, list):
        return [_map_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _map_value(item) for key, item in value.items()}
    return value


_NONE_VALUES: dict[tuple[str, str], Any] = {
    ("AffineFeedbackController", "bias"): [0.25],
    ("AffineFeedbackController", "feedforward"): [0.5],
    ("Channel", "input_shape"): [3],
    ("Channel", "noise_role"): "motor_command",
    ("Channel", "noise_timing"): "post_controller",
    ("DelayedReaches", "n_control_stages"): 49,
    ("DelayedReaches", "go_cue_event_name"): "go",
    ("DynamicsMatrixPerturb", "mass"): 1.25,
    ("FeedbackChannels", "input_shape"): [[3], [3]],
    ("LinearStateSpace", "B_w"): [[0.1], [0.2], [0.3], [0.4]],
    ("StructuralLinearStateSpace", "B_w"): [[0.1], [0.2], [0.3], [0.4]],
    ("StateFeedbackSelector", "expected_state_dim"): 4,
    ("StateFeedbackSelector", "output_size"): 4,
}


_VALUES: dict[tuple[str, str], Any] = {
    ("AffineValueComposer", "schema_version"): "feedbax.component.affine_value_composer.v1",
    ("AffineValueComposer", "feature_rules"): [
        {"kind": "identity", "state_slice": [1, 2]}
    ],
    ("Channel", "add_noise"): False,
    ("DelayedReaches", "target_visible_from_start"): True,
    ("DelayedReaches", "target_on_epochs"): [0, 2],
    ("StateFeedbackSelector", "state_slices"): {
        "position": {"start": 1, "stop": 3},
        "velocity": {"start": 3, "stop": 5},
    },
    ("StateFeedbackSelector", "channels"): [
        {"slice": "position", "transform": "negate"},
        {"slice": "velocity", "transform": "identity"},
    ],
    ("StructuralLinearStateSpace", "delta_A"): {
        "schema_id": "feedbax.spec.component_param.array_value",
        "schema_version": "feedbax.spec.component_param.array_value.v1",
        "shape": [4, 4],
        "dtype": "float32",
        "nonfinite": "forbid",
        "encoding": "sparse_coo",
        "fill": 0.0,
        "entries": [{"coordinate": [0, 1], "value": 0.25}],
    },
    ("ThresholdLatchedForce", "state_selector"): {"kind": "fixed", "path": ["pos", 1]},
}


def _non_default_value(component_type: str, schema: ParamSchema) -> Any:
    key = (component_type, schema.name)
    if key in _VALUES:
        return deepcopy(_VALUES[key])
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
    if component_type == "Channel":
        params.setdefault("input_shape", [2])
        if param_name in {"additive_noise_std", "signal_dependent_noise_std"}:
            params["noise_model"] = "signal_dependent_plus_additive"
    elif component_type == "FeedbackChannels":
        if param_name == "paths":
            params["selector"] = "paths"
        if param_name == "noise_std":
            params["add_noise"] = True
    elif component_type == "AffineValueComposer" and param_name == "output_block_size":
        size = int(params["output_block_size"])
        params["gain_init"] = [[0.0] for _ in range(size)]
        params["bias_init"] = [0.0 for _ in range(size)]
    elif component_type == "DelayedReaches" and param_name == "target_visible_from_start":
        params["target_on_epochs"] = [0, 1, 2]
    elif component_type == "DelayedReaches" and param_name == "n_control_stages":
        params.pop("n_steps", None)
    elif component_type == "DynamicsMatrixPerturb":
        params.setdefault("mass", 1.0)
    elif component_type == "StateFeedbackSelector" and param_name == "output_size":
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
    prototype_shape = (2,)
    if component_type == "Channel" and isinstance(params.get("input_shape"), list):
        prototype_shape = tuple(int(dim) for dim in params["input_shape"])
    prototypes = {("node", port): jnp.zeros(prototype_shape) for port in input_ports}
    return spec, prototypes


@pytest.mark.parametrize(("component_type", "param_name"), _registry_cases())
def test_registered_buildable_component_param_round_trips(
    component_type: str,
    param_name: str | None,
) -> None:
    """Every declared leaf parameter is exercised non-default and must survive."""
    registry = ComponentRegistry(load_user_components=False)
    meta = registry.get(component_type)
    assert meta is not None
    baseline_params = _contextualize_params(component_type, None, deepcopy(meta.default_params))
    params = deepcopy(baseline_params)
    if param_name is not None:
        schema = next(item for item in meta.param_schema if item.name == param_name)
        params[param_name] = _non_default_value(component_type, schema)
        params = _contextualize_params(component_type, param_name, params)

    spec, prototypes = _single_node_spec(registry, component_type, params)
    first = spec_to_graph(spec, registry, prototypes)
    serialized = graph_to_spec(first)
    emitted = serialized.nodes["node"]
    assert emitted.type == component_type

    if param_name is not None and param_name != "schema_version":
        baseline_spec, baseline_prototypes = _single_node_spec(
            registry, component_type, baseline_params
        )
        baseline = graph_to_spec(
            spec_to_graph(baseline_spec, registry, baseline_prototypes)
        )
        assert emitted.params != baseline.nodes["node"].params, (
            f"{component_type}.{param_name} did not affect the canonical GraphSpec"
        )

    second = spec_to_graph(
        serialized,
        ComponentRegistry(load_user_components=False),
        prototypes,
    )
    assert graph_to_spec(second).nodes["node"].params == emitted.params


def test_two_link_arm_nondefault_lengths_round_trip() -> None:
    registry = ComponentRegistry(load_user_components=False)
    spec = GraphSpec(
        nodes={
            "arm": ComponentSpec(
                type="TwoLinkArm",
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


def test_analytical_plant_authored_contract_round_trips() -> None:
    registry = ComponentRegistry(load_user_components=False)
    params = {
        "dt": 0.02,
        "n_steps": 4,
        "tau_act": 0.015,
        "tau_deact": 0.055,
        "clip_states": False,
    }
    spec = GraphSpec(
        nodes={
            "plant": ComponentSpec(
                type="AnalyticalMusculoskeletalPlant",
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
    registry = ComponentRegistry(load_user_components=False)
    spec = GraphSpec(
        nodes={
            "channel": ComponentSpec(
                type="Channel",
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


def test_simple_staged_network_expansion_preserves_authored_semantics() -> None:
    population = PopulationStructure.from_indices(
        input_only_indices=[0],
        readout_only_indices=[1],
        recurrent_only_indices=[2],
        input_readout_indices=[3],
    )
    network = SimpleStagedNetwork(
        input_size=5,
        hidden_size=4,
        out_size=2,
        encoding_size=3,
        hidden_nonlinearity=jax.nn.relu,
        out_nonlinearity=jax.nn.sigmoid,
        hidden_noise_std=0.2,
        population_structure=population,
        population_mask_mode="plain_all_ones",
        dtype=jnp.float16,
        key=jax.random.PRNGKey(0),
    )
    graph = Graph(nodes={"net": network})

    serialized = graph_to_spec(graph)
    assert set(serialized.nodes) == {
        "net_input_mux",
        "net_encoder",
        "net_cell",
        "net_hidden_activation",
        "net_hidden_noise",
        "net_readout",
    }
    restored = spec_to_graph(serialized, ComponentRegistry(load_user_components=False))
    encoder = restored.nodes["net_encoder"]
    activation = restored.nodes["net_hidden_activation"]
    noise = restored.nodes["net_hidden_noise"]
    readout = restored.nodes["net_readout"]
    assert isinstance(encoder, Linear)
    assert (encoder.input_size, encoder.output_size) == (5, 3)
    assert encoder.layer.weight.dtype == jnp.float16
    assert isinstance(activation, Activation)
    assert activation.activation_name == "relu"
    assert isinstance(noise, Channel)
    assert float(noise.noise_func.std) == pytest.approx(0.2)
    assert isinstance(readout, Linear)
    assert readout.activation_name == "sigmoid"
    assert readout.layer.weight.dtype == jnp.float16
    assert len(restored.parameter_constraints) == 2
