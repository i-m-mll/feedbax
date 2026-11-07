import jax.numpy as jnp
import jax.random as jr
import pytest
from jax.experimental import disable_x64, enable_x64

from feedbax.compiler import GraphCompilationError, GraphDocument, compile_graph
from feedbax.compiler.serialization import graph_to_spec
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.runtime.signal_transforms import (
    ConstantInput,
    ConstantInputParams,
    NetworkClamp,
    NetworkConstantInput,
    NetworkValueParams,
)
from feedbax.runtime.graph import Graph, init_state_from_component
from feedbax.runtime.iteration import run_component


_INPUT = jnp.array([2.0, 3.0], dtype=jnp.float32)
_INPUT_PORT = {"input": ("intervention", "input")}
_OUTPUT_PORT = {"output": ("intervention", "output")}


@pytest.fixture(scope="module")
def registry():
    return ComponentRegistry()


def _compile(registry, spec, signal=_INPUT):
    document = GraphDocument(graph=spec)
    document = GraphDocument.model_validate_json(document.model_dump_json())
    return compile_graph(document, registry, input_prototypes={("intervention", "input"): signal})


def _assert_rollout(graph, expected, signal=_INPUT):
    outputs, _, _ = run_component(
        graph,
        {"input": jnp.stack((signal, signal))},
        init_state_from_component(graph),
        n_steps=2,
        key=jr.key(0),
        state_filter=False,
    )
    assert outputs["output"].tolist() == [expected, expected]


def _authored_spec(component_type, params):
    return GraphSpec(
        nodes={
            "intervention": ComponentSpec(
                type=f"feedbax.{component_type}",
                params=params,
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        input_ports=["input"],
        output_ports=["output"],
        input_bindings=_INPUT_PORT,
        output_bindings=_OUTPUT_PORT,
    )


def _legacy_spec(component_type, params, version):
    payload = _authored_spec(component_type, params).model_dump(mode="json")
    payload["schema_version"] = "feedbax.spec.graph.v5"
    node = payload["nodes"]["intervention"]
    node["type"] = component_type
    node.pop("revision", None)
    if version is not None:
        node["param_schema_version"] = version
    return GraphSpec.model_validate(payload)


@pytest.mark.parametrize(
    ("component_cls", "params_cls", "field", "values", "expected"),
    [
        (NetworkClamp, NetworkValueParams, "unit_spec", [8.0, None], [8.0, 3.0]),
        (NetworkConstantInput, NetworkValueParams, "unit_spec", [5.0, None], [12.0, 3.0]),
        (ConstantInput, ConstantInputParams, "arrays", [5.0, 7.0], [12.0, 17.0]),
    ],
    ids=["clamp", "selected-addition", "dense-addition"],
)
def test_intervention_payload_survives_public_graph_round_trip(
    registry, component_cls, params_cls, field, values, expected
):
    payload = jnp.array([jnp.nan if value is None else value for value in values])
    graph = Graph(
        nodes={
            "intervention": component_cls(params_cls(active=True, scale=2.0, **{field: payload}))
        },
        input_ports=("input",),
        output_ports=("output",),
        input_bindings=_INPUT_PORT,
        output_bindings=_OUTPUT_PORT,
    )
    _assert_rollout(graph, expected)
    exported = graph_to_spec(graph, registry)
    rebuilt = _compile(registry, exported)
    _assert_rollout(rebuilt.graph, expected)

    params = {field: values, "active": True}
    if component_cls is not NetworkClamp:
        params["scale"] = 2.0
    authored = _compile(registry, _authored_spec(component_cls.__name__, params))
    _assert_rollout(authored.graph, expected)

    disabled = _compile(
        registry, _authored_spec(component_cls.__name__, {**params, "active": False})
    )
    _assert_rollout(disabled.graph, [2.0, 3.0])
    assert graph_to_spec(disabled.graph, registry).nodes["intervention"].params[field] == values


@pytest.mark.parametrize(
    ("component_type", "field"),
    [
        ("NetworkClamp", "unit_spec"),
        ("NetworkConstantInput", "unit_spec"),
        ("ConstantInput", "arrays"),
    ],
)
def test_legacy_intervention_requires_reauthoring_only_when_enabled(
    registry, component_type, field
):
    for version in ("1", None):
        disabled = _legacy_spec(component_type, {"scale": 2.0, "active": False}, version)
        _assert_rollout(_compile(registry, disabled).graph, [2.0, 3.0])

        enabled = _legacy_spec(component_type, {"scale": 2.0, "active": True}, version)
        with pytest.raises(GraphCompilationError, match=field):
            _compile(registry, enabled)

    if component_type == "NetworkClamp":
        repaired = _authored_spec(
            component_type, {"scale": 2.0, "active": True, field: [8.0, None]}
        )
        compiled = _compile(registry, repaired)
        _assert_rollout(compiled.graph, [8.0, 3.0])
        assert "scale" not in graph_to_spec(compiled.graph, registry).nodes["intervention"].params


def test_intervention_payload_must_fit_the_signal(registry):
    spec = _authored_spec("NetworkClamp", {"active": True, "unit_spec": [1.0, None, 3.0]})
    with pytest.raises(GraphCompilationError, match="unit_spec.*(shape|length|width)"):
        _compile(registry, spec)


def test_intervention_payload_preserves_state_dtype_and_precision(registry):
    with enable_x64():
        signal = jnp.array([2.0, 3.0], dtype=jnp.float64)
        value = 8.00000001
        graph = Graph(
            nodes={
                "intervention": NetworkClamp(
                    NetworkValueParams(
                        unit_spec=jnp.array([value, jnp.nan], dtype=jnp.float64)
                    )
                )
            },
            input_ports=("input",),
            output_ports=("output",),
            input_bindings=_INPUT_PORT,
            output_bindings=_OUTPUT_PORT,
        )
        exported = graph_to_spec(graph, registry)
        rebuilt = _compile(registry, exported, signal).graph
        _assert_rollout(rebuilt, [value, 3.0], signal)
        state = init_state_from_component(rebuilt).get(rebuilt.nodes["intervention"].params_index)
        assert state.unit_spec.dtype == jnp.float64

    with disable_x64(), pytest.raises(GraphCompilationError, match="float64"):
        _compile(registry, exported)
