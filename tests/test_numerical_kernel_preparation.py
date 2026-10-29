from copy import deepcopy
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.contracts.training import standard_supervised_method_contract
from feedbax.contracts.worker import (
    AxisSpec,
    MappingLevelSpec,
    ProgressCoordinate,
    SlotAxisBindingSpec,
)
from feedbax.training.phase_executor import NumericalUpdateKernel, PhaseProgramExecutor
from feedbax.training.worker_validation import (
    resolve_execution_mapping,
    WorkerContractValidationError,
)
from tests.test_graph import _make_node_output_initialized_cycle


def runtime(update, *, mapped=False):
    contract = standard_supervised_method_contract()
    levels, bindings = (), {}
    if mapped:
        contract.axes.append(AxisSpec(name="replica", role="replicate", size=3))
        for slot in contract.state_slots:
            slot.axis_bindings = [SlotAxisBindingSpec(axis="replica", mode="mapped", array_axis=0)]
        contract.phase_program.update_steps[0].axes.append("replica")
        levels, bindings = resolve_execution_mapping(
            SimpleNamespace(
                method_contract=contract,
                mapping_levels=[MappingLevelSpec(axis="replica")],
            )
        )
    contract.phase_program.phases[0].max_steps = 3
    kernel = NumericalUpdateKernel(
        update, ("rate", "schedule_origin_step"), excluded_slots=("objective",)
    )
    executor = PhaseProgramExecutor(
        contract.phase_program,
        {contract.phase_program.update_steps[0].kernel.kernel_ref: kernel},
        mapping_levels=levels,
        slot_axis_bindings=bindings,
    )
    slots = {
        "model": jnp.zeros((3, 2) if mapped else (2,)),
        "optimizer": jnp.zeros((3,) if mapped else (), dtype=jnp.int32),
        "prng": jax.random.split(jax.random.key(0), 3) if mapped else jax.random.key(0),
        "train_loss": jnp.zeros((3,) if mapped else ()),
        "batch_counter": jnp.zeros((3,) if mapped else (), dtype=jnp.int32),
        "objective": {"host": object()},
    }
    return executor, kernel, slots


@pytest.mark.parametrize("mapped", [False, True])
def test_preparation_is_pure_and_execution_matches_science_without_retracing(mapped):
    traces = []

    def update(slots, context):
        traces.append(1)
        key, _ = jax.random.split(slots["prng"])
        return {
            "model": slots["model"] + context["rate"],
            "optimizer": slots["optimizer"] + 1,
            "prng": key,
            "train_loss": jnp.sum(slots["model"] ** 2),
            "batch_counter": slots["batch_counter"] + 1,
        }

    executor, kernel, slots = runtime(update, mapped=mapped)
    context = {"rate": jnp.asarray(0.5), "schedule_origin_step": 7}
    before = jax.tree.map(
        lambda x: np.asarray(x)
        if eqx.is_array(x) and not jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key)
        else x,
        slots,
    )
    executor.preflight(slots, run_id="prepare", context=context)
    assert len(traces) == 1
    assert executor.checkpoint_store.visits() == ()
    np.testing.assert_array_equal(slots["model"], before["model"])
    np.testing.assert_array_equal(slots["batch_counter"], before["batch_counter"])
    prepared = executor._prepared_numerical[executor.program.update_steps[0].name]
    current = slots
    for i in range(3):
        context["rate"] = jnp.asarray(0.5 + i)
        context["unrelated_host_metadata"] = object()
        current = {
            **current,
            **prepared(
                current,
                ProgressCoordinate(run_id="run", phase=f"phase{i}", program_step=i),
                context,
            ),
        }
    jax.block_until_ready(current)
    assert len(traces) == 1
    np.testing.assert_array_equal(current["model"], np.full_like(np.asarray(current["model"]), 4.5))
    np.testing.assert_array_equal(current["optimizer"], 3)
    scientific = {k: v for k, v in slots.items() if k != "objective"}

    def scalar(values, ctx):
        return update(values, ctx)

    direct = eqx.filter_vmap(scalar, in_axes=(eqx.if_array(0), None)) if mapped else scalar
    for i in range(3):
        scientific = {**scientific, **direct(scientific, {**context, "rate": jnp.asarray(0.5 + i)})}
    for name in ("model", "optimizer", "train_loss", "batch_counter"):
        np.testing.assert_array_equal(current[name], scientific[name])
    np.testing.assert_array_equal(
        jax.random.key_data(current["prng"]), jax.random.key_data(scientific["prng"])
    )
    assert executor.preparation_telemetry["kernel_preparation_seconds"] >= 0
    for changed, match in [
        ({**context, "schedule_origin_step": 8}, "static inputs"),
        ({"rate": context["rate"]}, "schedule_origin_step"),
    ]:
        with pytest.raises((ValueError, KeyError), match=match):
            prepared(current, None, changed)
    with pytest.raises((ValueError, TypeError), match="shape|Shape|types"):
        prepared({**current, "model": jnp.zeros((4,))}, None, context)
    del current["optimizer"]
    with pytest.raises(ValueError, match="disappeared"):
        prepared(current, None, context)


def test_prepared_run_keeps_host_coordinate_semantics_and_no_warmup():
    seen = []

    def numerical(slots, context):
        return {
            "model": slots["model"] + context["rate"],
            "optimizer": slots["optimizer"] + 1,
            "prng": slots["prng"],
            "train_loss": jnp.asarray(0.0),
            "batch_counter": slots["batch_counter"] + 1,
        }

    executor, _, slots = runtime(numerical)
    result = executor.run(
        slots, run_id="run", context={"rate": jnp.asarray(1.0), "schedule_origin_step": 0}
    )
    np.testing.assert_array_equal(result.slots["model"], [3, 3])
    assert result.end_completed_batches == 3

    def host(slots, coordinate, context):
        seen.append(coordinate.program_step)
        return numerical(slots, {"rate": coordinate.program_step})

    executor.kernels = {next(iter(executor.kernels)): host}
    executor.run(slots, run_id="host")
    assert seen == [0, 1, 2]


def test_preparation_timer_excludes_host_shape_preflight(monkeypatch):
    import feedbax.training.phase_executor as module

    times = iter([10.0, 14.0])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(times))
    executor, _, slots = runtime(lambda slots, context: {"model": slots["model"] + context["rate"]})
    executor.preflight(
        slots, run_id="time", context={"rate": jnp.asarray(1.0), "schedule_origin_step": 0}
    )
    assert executor.preparation_telemetry["kernel_preparation_seconds"] == 4


@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("invalid", ["shape", "dtype", "tree", "category", "static", "unknown"])
def test_invalid_numerical_output_fails_before_update_or_checkpoint(mapped, invalid):
    def update(slots, context):
        if invalid == "unknown":
            return {"unknown": slots["model"]}
        if invalid == "shape":
            value = jnp.zeros((9,), dtype=jnp.float32)
        elif invalid == "dtype":
            value = slots["model"].astype(jnp.int32)
        elif invalid == "tree":
            value = {"parameters": slots["model"]}
        elif invalid == "category":
            value = 0.0
        else:
            value = {**slots["model"], "mode": "eval"}
        return {"model": value, "batch_counter": slots["batch_counter"] + 1}

    executor, _, slots = runtime(update, mapped=mapped)
    executor.program.phases[0].max_steps = 1
    if invalid == "static":
        slots["model"] = {"parameters": slots["model"], "mode": "train"}
    before = deepcopy({name: value for name, value in slots.items() if name != "objective"})
    before["objective"] = dict(slots["objective"])
    progress = []
    with pytest.raises(WorkerContractValidationError) as error:
        executor.run(
            slots,
            run_id="invalid",
            context={"rate": jnp.asarray(1.0), "schedule_origin_step": 0},
            progress_callback=progress.append,
        )
    assert f"/update_steps/{executor.program.update_steps[0].name}/writes" in str(error.value)
    if invalid != "unknown":
        assert "/model" in str(error.value)
    assert executor.checkpoint_store.visits() == ()
    assert progress == []
    assert eqx.tree_equal(slots, before)
    np.testing.assert_array_equal(slots["batch_counter"], 0)


def test_one_kernel_prepares_distinct_scalar_and_mapped_steps():
    traces = []

    def update(slots, context):
        traces.append(slots["model"].shape)
        return {"model": slots["model"] + context["rate"]}

    executor, _, slots = runtime(update, mapped=True)
    mapped = executor.program.update_steps[0]
    executor.program.update_steps.append(mapped.model_copy(update={"name": "scalar", "axes": []}))
    executor.program.phases[0].update_steps.append("scalar")
    executor.preflight(
        slots, run_id="both", context={"rate": jnp.asarray(1.0), "schedule_origin_step": 0}
    )
    assert traces == [(2,), (3, 2)]
    assert len(executor._prepared_numerical) == 2


@pytest.mark.parametrize("mapped", [False, True])
def test_recurrent_graph_update_retains_abi_and_learns_trial_initializer(mapped):
    def loss(model):
        outputs, _ = model(
            {"context": jnp.asarray(3.0)},
            model.init_state(key=jax.random.key(0)),
            key=jax.random.key(0),
            n_steps=2,
        )
        return jnp.sum(outputs["out"])

    def update(slots, context):
        value, gradient = eqx.filter_value_and_grad(loss)(slots["model"])
        model = eqx.tree_at(
            lambda graph: graph.nodes["encoder"].factor,
            slots["model"],
            slots["model"].nodes["encoder"].factor
            - context["rate"] * gradient.nodes["encoder"].factor,
        )
        return {
            "model": model,
            "optimizer": slots["optimizer"],
            "prng": slots["prng"],
            "train_loss": value,
            "batch_counter": slots["batch_counter"] + 1,
        }

    executor, _, slots = runtime(update, mapped=mapped)
    executor.program.phases[0].max_steps = 1
    graph = _make_node_output_initialized_cycle(encoder_factor=2.0).prepare()
    slots["model"] = eqx.filter_vmap(lambda _: graph)(jnp.arange(3)) if mapped else graph
    structure = jax.tree.structure(slots["model"])
    static = eqx.filter(slots["model"], lambda value: not eqx.is_array(value))
    context = {"rate": jnp.asarray(0.1), "schedule_origin_step": 0}
    executor.preflight(slots, run_id="graph", context=context)
    assert jax.tree.structure(slots["model"]) == structure
    np.testing.assert_array_equal(slots["batch_counter"], 0)
    result = executor.run(slots, run_id="graph", context=context)
    learned = result.slots["model"]
    assert jax.tree.structure(learned) == structure
    assert eqx.tree_equal(eqx.filter(learned, lambda value: not eqx.is_array(value)), static)
    np.testing.assert_allclose(result.slots["train_loss"], 7.0)
    np.testing.assert_allclose(learned.nodes["encoder"].factor, 1.775)
    evaluate_loss = eqx.filter_vmap(loss) if mapped else loss
    np.testing.assert_allclose(evaluate_loss(learned), 6.49375)
    np.testing.assert_array_equal(result.slots["batch_counter"], 1)
