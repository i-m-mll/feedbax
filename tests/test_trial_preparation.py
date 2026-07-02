import equinox as eqx
import jax.numpy as jnp
from equinox.nn import StateIndex

from feedbax.config.mapping import WhereDict
from feedbax.contracts.graph import StudioTaskBindingSpec
from feedbax.runtime.graph import Component, Graph, init_state_from_component
from feedbax.runtime.task_bindings import (
    apply_task_parameter_state_inits,
    binding_spec_from_legacy_extra_inputs,
    expose_task_bindings,
)
from feedbax.intervene import TimeSeriesParam
from feedbax.tasks import (
    TaskTrialSpec,
    TrialTimeline,
    extract_timeseries_params,
    infer_n_steps,
    merge_intervene_inputs,
    prepare_inputs,
    prepare_trial,
    safe_state_set,
    set_state_by_path,
    set_state_matching_dtypes,
    where_key_to_path,
)


class _NodeState(eqx.Module):
    hidden: jnp.ndarray
    output: jnp.ndarray


class _Params(eqx.Module):
    static_gain: jnp.ndarray | None
    dynamic_gain: jnp.ndarray


class _InterventionNode(Component):
    input_ports = ("input", "params_override")
    output_ports = ("output",)

    state_index: StateIndex
    params_index: StateIndex
    hidden: jnp.ndarray
    readout: jnp.ndarray

    def __init__(self):
        self.state_index = StateIndex(
            _NodeState(
                hidden=jnp.zeros((2,), dtype=jnp.float32),
                output=jnp.zeros((2,), dtype=jnp.float32),
            )
        )
        self.params_index = StateIndex(
            _Params(
                static_gain=jnp.asarray(1.0, dtype=jnp.float32),
                dynamic_gain=jnp.asarray(0.0, dtype=jnp.float32),
            )
        )
        self.hidden = jnp.asarray([1.0], dtype=jnp.float32)
        self.readout = jnp.asarray([2.0], dtype=jnp.float32)

    def __call__(self, inputs, state, *, key):
        return {"output": inputs["input"]}, state

    def task_parameter_state_indices(self):
        return {"foo": self.params_index}


def _graph() -> Graph:
    return Graph(
        nodes={"net": _InterventionNode()},
        wires=(),
        input_ports=("input", "intervene:foo"),
        output_ports=("output",),
        input_bindings={
            "input": ("net", "input"),
            "intervene:foo": ("net", "params_override"),
        },
        output_bindings={"output": ("net", "output")},
    )


def test_where_key_to_path_strips_unique_suffix() -> None:
    assert where_key_to_path("net.hidden#initial") == "net.hidden"
    assert where_key_to_path(lambda state: state.net.hidden) == "net.hidden"


def test_prepare_inputs_wraps_single_required_port() -> None:
    graph = _graph()
    inputs = jnp.zeros((3, 2), dtype=jnp.float32)

    prepared = prepare_inputs(graph, inputs)

    assert set(prepared) == {"input"}
    assert jnp.array_equal(prepared["input"], inputs)


def test_intervention_inputs_are_prefixed_when_merged() -> None:
    inputs = {"input": jnp.zeros((3, 2), dtype=jnp.float32)}
    intervene_inputs = {"foo": _Params(None, jnp.arange(3.0))}

    merged = merge_intervene_inputs(inputs, intervene_inputs)

    assert set(merged) == {"input", "intervene:foo"}
    assert merged["intervene:foo"].dynamic_gain.shape == (3,)


def test_extract_timeseries_params_broadcasts_static_leaves_and_defaults_for_none() -> None:
    params = _Params(
        static_gain=None,
        dynamic_gain=TimeSeriesParam(jnp.asarray([2.0, 3.0, 4.0], dtype=jnp.float32)),
    )
    defaults = _Params(
        static_gain=jnp.asarray(1.0, dtype=jnp.float32),
        dynamic_gain=jnp.asarray(0.0, dtype=jnp.float32),
    )

    extracted = extract_timeseries_params(params, defaults)

    assert extracted.static_gain.shape == (3,)
    assert jnp.array_equal(extracted.static_gain, jnp.ones((3,), dtype=jnp.float32))
    assert jnp.array_equal(extracted.dynamic_gain, jnp.asarray([2.0, 3.0, 4.0]))


def test_set_state_by_path_updates_graph_node_state_attribute() -> None:
    graph = _graph()
    state = init_state_from_component(graph)

    updated = set_state_by_path(graph, state, "net.hidden", jnp.asarray([5.0, 6.0]))

    assert jnp.array_equal(graph.state_view(updated).net.hidden, jnp.asarray([5.0, 6.0]))


def test_set_state_matching_dtypes_casts_replacement_to_existing_dtype() -> None:
    graph = _graph()
    state = init_state_from_component(graph)
    node = graph.get_node("net")
    replacement = _Params(
        static_gain=jnp.asarray(2, dtype=jnp.int32),
        dynamic_gain=jnp.asarray(3, dtype=jnp.int32),
    )

    updated = safe_state_set(state, node.params_index, replacement)

    assert updated.get(node.params_index).static_gain.dtype == jnp.float32
    assert updated.get(node.params_index).dynamic_gain.dtype == jnp.float32
    assert safe_state_set is set_state_matching_dtypes


def test_prepare_trial_applies_inits_interventions_and_infers_timeline_steps() -> None:
    graph = _graph()
    trial_spec = TaskTrialSpec(
        inits=WhereDict({"net.hidden": jnp.asarray([7.0, 8.0], dtype=jnp.float32)}),
        targets=WhereDict(),
        inputs=jnp.zeros((5, 2), dtype=jnp.float32),
        intervene={
            "foo": _Params(
                static_gain=jnp.asarray(4.0, dtype=jnp.float32),
                dynamic_gain=TimeSeriesParam(jnp.asarray([1.0, 2.0, 3.0])),
            )
        },
        timeline=TrialTimeline(n_steps=3),
    )

    prepared = prepare_trial(graph, trial_spec)

    assert prepared.n_steps == 3
    assert set(prepared.inputs) == {"input", "intervene:foo"}
    assert jnp.array_equal(
        graph.state_view(prepared.init_state).net.hidden, jnp.asarray([7.0, 8.0])
    )
    node = graph.get_node("net")
    assert prepared.init_state.get(node.params_index).static_gain == jnp.asarray(4.0)
    assert prepared.inputs["intervene:foo"].dynamic_gain.shape == (3,)


def test_task_parameter_binding_exposes_constant_state_init_plan() -> None:
    graph = _graph()
    binding_spec = {
        "schema_version": "feedbax.spec.studio.task_bindings.v2",
        "exposed_data": [
            {
                "id": "foo",
                "label": "Foo params",
                "kind": "intervention",
                "role": "component_parameter",
                "path": "intervene.foo",
                "bindable": True,
                "dtype": "object",
                "metadata": {"temporal_support": "constant"},
            }
        ],
        "bindings": [
            {
                "id": "task:foo->net:params_override",
                "source_data_id": "foo",
                "target_node_id": "net",
                "target_port": "params_override",
                "role": "component_parameter",
                "metadata": {"task_parameter_label": "foo"},
            }
        ],
        "metadata": {},
    }

    exposure = expose_task_bindings(graph, StudioTaskBindingSpec.model_validate(binding_spec))

    assert exposure.input_plans == ()
    assert exposure.state_init_plans[0].target_label == "foo"
    state = init_state_from_component(exposure.graph)
    replacement = _Params(
        static_gain=jnp.asarray(9.0, dtype=jnp.float32),
        dynamic_gain=jnp.asarray(0.5, dtype=jnp.float32),
    )
    updated = apply_task_parameter_state_inits(
        state,
        exposure.graph,
        exposure.state_init_plans,
        {"foo": replacement},
    )

    node = exposure.graph.get_node("net")
    assert updated.get(node.params_index).static_gain == jnp.asarray(9.0)


def test_legacy_extra_inputs_map_intervene_prefix_to_component_parameter_binding() -> None:
    graph = _graph()
    binding_spec = binding_spec_from_legacy_extra_inputs(graph, ["intervene:foo"])

    assert binding_spec.exposed_data[0].role == "component_parameter"
    assert binding_spec.exposed_data[0].metadata["legacy_input"] == "intervene:foo"
    exposure = expose_task_bindings(graph, binding_spec)

    assert exposure.state_init_plans == ()
    assert exposure.input_plans[0].graph_input == "intervene:foo"
    assert exposure.input_plans[0].target_port == "params_override"


def test_infer_n_steps_prefers_timeline_over_input_length() -> None:
    timeline = TrialTimeline(n_steps=2)

    assert infer_n_steps({"input": jnp.zeros((5, 1))}, timeline) == 2


def test_graph_get_node_attrs_returns_public_named_node_params() -> None:
    graph = _graph()

    hidden, readout = graph.get_node_attrs("net", "hidden", "readout")

    assert jnp.array_equal(hidden, jnp.asarray([1.0], dtype=jnp.float32))
    assert jnp.array_equal(readout, jnp.asarray([2.0], dtype=jnp.float32))
