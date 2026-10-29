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


@pytest.mark.parametrize("invalid", ["shape", "dtype", "unknown"])
def test_invalid_mapped_output_fails_before_update_or_checkpoint(invalid):
    def update(slots, context):
        if invalid == "unknown":
            return {"unknown": slots["model"]}
        value = jnp.zeros((4,)) if invalid == "shape" else slots["model"].astype(jnp.int32)
        return {"model": value}

    executor, _, slots = runtime(update, mapped=True)
    with pytest.raises(WorkerContractValidationError, match="kernel preparation failed"):
        executor.preflight(
            slots, run_id="invalid", context={"rate": jnp.asarray(1.0), "schedule_origin_step": 0}
        )
    assert executor.checkpoint_store.visits() == ()
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
