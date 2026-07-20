from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree as jt
import optax
import pytest

from feedbax.contracts.training import (
    LR_SCHEDULE_SPEC_SCHEMA_ID,
    LR_SCHEDULE_SPEC_SCHEMA_VERSION,
    LrScheduleSpec,
    LossTermSpec,
    ObjectiveSlotSpec,
    OptimizerSpec,
    TaskSpec,
    TrainingConfig,
    TrainingRunSpec,
    WorkerExecutionSpec,
    standard_supervised_effective_phase_spec,
    standard_supervised_method_contract,
    standard_supervised_method_payload,
    standard_supervised_method_ref,
)
from feedbax.contracts.checkpoints import CheckpointSegmentLineage
from feedbax.training.schedule_clocks import (
    lr_origin_for_continuation,
    resolve_schedule_window,
    schedule_progress,
)
from feedbax.integrations.provider import provider_manifest
from feedbax.training.optimizers import (
    OptimizerStateCompatibilityError,
    build_optimizer,
    learning_rate_at_step,
    validate_optimizer_state_for_spec,
)


pytestmark = pytest.mark.feedbax_contract


def _minimal_graph() -> dict[str, object]:
    return {
        "nodes": {
            "gain": {
                "type": "Gain",
                "params": {"gain": 1.0},
                "input_ports": ["input"],
                "output_ports": ["output"],
            }
        },
        "wires": [],
        "input_ports": ["input"],
        "output_ports": ["output"],
        "input_bindings": {"input": ("gain", "input")},
        "output_bindings": {"output": ("gain", "output")},
    }


def _training_run_payload_without_lr_schedule() -> dict[str, object]:
    worker = WorkerExecutionSpec(
        method_contract=standard_supervised_method_contract(),
        effective_phase=standard_supervised_effective_phase_spec(),
    )
    spec = TrainingRunSpec(
        graph={"inline": _minimal_graph()},
        task=TaskSpec(type="ReachingTask", params={"n_steps": 4}),
        training_config=TrainingConfig(n_batches=2, batch_size=3),
        objective=ObjectiveSlotSpec(
            loss=LossTermSpec(type="target_state", label="target", selector="output")
        ),
        method_ref=standard_supervised_method_ref(),
        method_payload=standard_supervised_method_payload(),
        worker_execution=worker,
    )
    payload = spec.model_dump(mode="json")
    method_optimizer = payload["method_payload"]["payload"]["optimizer"]
    method_optimizer.pop("lr_schedule", None)
    return payload


def _expected_cosine(peak: float, *, alpha: float, progress: float) -> float:
    terminal = peak * alpha
    return terminal + 0.5 * (1.0 + math.cos(math.pi * progress)) * (peak - terminal)


def _scheduled_learning_rate(opt_state: Any) -> float:
    leaves = jt.leaves(opt_state, is_leaf=_is_injected_hyperparams_state)
    for leaf in leaves:
        if _is_injected_hyperparams_state(leaf):
            return float(leaf.hyperparams["learning_rate"])
    raise AssertionError("scheduled optimizer state not found")


def _is_injected_hyperparams_state(value: Any) -> bool:
    fields = getattr(value, "_fields", ())
    return {"hyperparams", "inner_state"}.issubset(set(fields))


def _with_restored_injected_count(opt_state: Any, count: int) -> Any:
    """Return ``opt_state`` with injected-hyperparameter counters patched."""
    if _is_injected_hyperparams_state(opt_state):
        hyperparams_states = dict(opt_state.hyperparams_states)
        lr_state = hyperparams_states["learning_rate"]
        hyperparams_states["learning_rate"] = lr_state._replace(
            count=jnp.asarray(count, dtype=jnp.int32)
        )
        return opt_state._replace(
            count=jnp.asarray(count, dtype=jnp.int32),
            hyperparams_states=hyperparams_states,
        )
    if isinstance(opt_state, tuple):
        return type(opt_state)(_with_restored_injected_count(value, count) for value in opt_state)
    if isinstance(opt_state, list):
        return [_with_restored_injected_count(value, count) for value in opt_state]
    if isinstance(opt_state, dict):
        return {
            key: _with_restored_injected_count(value, count)
            for key, value in opt_state.items()
        }
    return opt_state


def _state_after_zero_update(
    optimizer: optax.GradientTransformationExtraArgs,
    opt_state: Any,
    params: dict[str, jax.Array],
) -> Any:
    grads = jt.map(jnp.zeros_like, params)
    _updates, next_state = optimizer.update(grads, opt_state, params)
    return next_state


@pytest.mark.parametrize("step", [0, 5, 12])
def test_constant_lr_schedule_values_are_origin_relative(step: int) -> None:
    schedule = LrScheduleSpec(kind="constant", learning_rate_0=0.03)

    assert float(
        learning_rate_at_step(schedule, current_step=100 + step, schedule_origin_step=100)
    ) == pytest.approx(0.03)


def test_warmup_cosine_schedule_pins_representative_values() -> None:
    schedule = LrScheduleSpec(
        kind="warmup_cosine",
        learning_rate_0=0.1,
        total_steps=10,
        constant_lr_iterations=4,
        warmup_init_fraction=0.1,
        cosine_annealing_alpha=0.2,
    )

    samples = {
        0: 0.01,
        4: 0.1,
        7: _expected_cosine(0.1, alpha=0.2, progress=0.5),
        10: 0.02,
    }
    for step, expected in samples.items():
        assert float(
            learning_rate_at_step(schedule, current_step=step, schedule_origin_step=0)
        ) == pytest.approx(expected)


def test_warmup_cosine_holds_terminal_rate_after_authored_endpoint() -> None:
    schedule = LrScheduleSpec(
        kind="warmup_cosine",
        learning_rate_0=0.03,
        total_steps=3500,
        constant_lr_iterations=1000,
        warmup_init_fraction=0.1,
        cosine_annealing_alpha=0.001,
    )

    for step in (3500, 3501, 4499, 4500):
        assert float(
            learning_rate_at_step(schedule, current_step=step, schedule_origin_step=0)
        ) == pytest.approx(3e-5)


def test_delayed_cosine_schedule_pins_representative_values() -> None:
    schedule = LrScheduleSpec(
        kind="delayed_cosine",
        learning_rate_0=0.1,
        total_steps=8,
        constant_lr_iterations=3,
        cosine_annealing_alpha=0.2,
    )

    samples = {
        0: 0.1,
        3: 0.1,
        5: _expected_cosine(0.1, alpha=0.2, progress=2 / 5),
        8: 0.02,
    }
    for step, expected in samples.items():
        assert float(
            learning_rate_at_step(schedule, current_step=step, schedule_origin_step=0)
        ) == pytest.approx(expected)


def test_optimizer_builder_uses_explicit_resume_barrier_origin() -> None:
    schedule = LrScheduleSpec(
        kind="warmup_cosine",
        learning_rate_0=0.1,
        total_steps=10,
        constant_lr_iterations=4,
        warmup_init_fraction=0.1,
        cosine_annealing_alpha=0.2,
    )
    spec = OptimizerSpec(type="adamw", params={"weight_decay": 0.0}, lr_schedule=schedule)
    params = {"w": jnp.asarray(1.0)}

    optimizer = build_optimizer(spec, schedule_origin_step=12_000, current_step=12_000)
    opt_state = optimizer.init(params)

    assert _scheduled_learning_rate(opt_state) == pytest.approx(0.01)
    validate_optimizer_state_for_spec(spec, opt_state)


def test_optimizer_builder_resume_barrier_decouples_restored_count() -> None:
    schedule = LrScheduleSpec(
        kind="warmup_cosine",
        learning_rate_0=0.1,
        total_steps=10,
        constant_lr_iterations=4,
        warmup_init_fraction=0.1,
        cosine_annealing_alpha=0.2,
    )
    spec = OptimizerSpec(type="adamw", params={"weight_decay": 0.0}, lr_schedule=schedule)
    params = {"w": jnp.asarray(1.0)}
    optimizer = build_optimizer(
        spec,
        schedule_origin_step=12_000,
        current_step=12_000,
        optimizer_count_at_current_step=12_000,
    )
    restored_state = _with_restored_injected_count(optimizer.init(params), 12_000)

    next_state = _state_after_zero_update(optimizer, restored_state, params)

    assert _scheduled_learning_rate(next_state) == pytest.approx(0.01)


def test_optimizer_builder_run_start_origin_reproduces_program_step_schedule() -> None:
    schedule = LrScheduleSpec(
        kind="warmup_cosine",
        learning_rate_0=0.1,
        total_steps=10,
        constant_lr_iterations=4,
        warmup_init_fraction=0.1,
        cosine_annealing_alpha=0.2,
    )
    spec = OptimizerSpec(type="adamw", params={"weight_decay": 0.0}, lr_schedule=schedule)

    optimizer = build_optimizer(spec, schedule_origin_step=0, current_step=12_004)
    opt_state = optimizer.init({"w": jnp.asarray(1.0)})

    assert _scheduled_learning_rate(opt_state) == pytest.approx(0.02)


def test_optimizer_builder_run_start_keeps_restored_count_on_global_schedule() -> None:
    schedule = LrScheduleSpec(
        kind="warmup_cosine",
        learning_rate_0=0.1,
        total_steps=10,
        constant_lr_iterations=4,
        warmup_init_fraction=0.1,
        cosine_annealing_alpha=0.2,
    )
    spec = OptimizerSpec(type="adamw", params={"weight_decay": 0.0}, lr_schedule=schedule)
    params = {"w": jnp.asarray(1.0)}
    optimizer = build_optimizer(
        spec,
        schedule_origin_step=0,
        current_step=12_000,
        optimizer_count_at_current_step=12_000,
    )
    restored_state = _with_restored_injected_count(optimizer.init(params), 12_000)

    next_state = _state_after_zero_update(optimizer, restored_state, params)

    assert _scheduled_learning_rate(next_state) == pytest.approx(0.02)


def test_declared_schedule_rejects_plain_legacy_optimizer_state() -> None:
    spec = OptimizerSpec(
        type="adamw",
        params={"weight_decay": 0.0},
        lr_schedule=LrScheduleSpec(kind="constant", learning_rate_0=0.1),
    )
    old_state = optax.adamw(learning_rate=0.1, weight_decay=0.0).init(
        {"w": jnp.asarray(1.0)}
    )

    with pytest.raises(OptimizerStateCompatibilityError, match="declares lr_schedule"):
        validate_optimizer_state_for_spec(spec, old_state)


def test_legacy_training_run_spec_without_lr_schedule_still_loads() -> None:
    payload = _training_run_payload_without_lr_schedule()

    spec = TrainingRunSpec.model_validate(payload)

    optimizer = spec.method_payload.payload["optimizer"]
    assert "lr_schedule" not in payload["method_payload"]["payload"]["optimizer"]
    assert OptimizerSpec.model_validate(optimizer).lr_schedule is None


def test_lr_schedule_schema_identity_is_publicly_discoverable() -> None:
    manifest = provider_manifest()

    schema = manifest.schemas["LrScheduleSpec"]
    assert schema["properties"]["schema_id"]["const"] == LR_SCHEDULE_SPEC_SCHEMA_ID
    assert (
        LrScheduleSpec.model_fields["schema_version"].default
        == LR_SCHEDULE_SPEC_SCHEMA_VERSION
    )


def test_segment_start_clock_resolves_incident_ramp_values() -> None:
    schedule = LrScheduleSpec(
        kind="warmup_cosine",
        learning_rate_0=0.1,
        total_steps=1000,
        constant_lr_iterations=500,
        origin={"kind": "segment_start"},
    )
    lineage = CheckpointSegmentLineage(
        start_batch=12_000,
        segment_batch_count=1_000,
        parent_transaction_id="parent",
    )
    window = resolve_schedule_window(
        schedule.origin,
        lineage=lineage,
        duration=schedule.total_steps,
    )

    assert [schedule_progress(batch, window=window) for batch in (12_000, 12_500, 13_000)] == [
        0.0,
        0.5,
        1.0,
    ]


def test_run_start_clock_is_continuous_across_seam() -> None:
    root = CheckpointSegmentLineage(start_batch=0, segment_batch_count=20_000)
    continuation = CheckpointSegmentLineage(
        start_batch=12_000,
        segment_batch_count=8_000,
        parent_transaction_id="parent",
    )
    origin = {"kind": "run_start"}
    schedule = LrScheduleSpec(
        kind="delayed_cosine",
        learning_rate_0=0.1,
        total_steps=20_000,
        constant_lr_iterations=1_000,
        origin=origin,
    )
    root_window = resolve_schedule_window(schedule.origin, lineage=root, duration=20_000)
    resumed_window = resolve_schedule_window(
        schedule.origin, lineage=continuation, duration=20_000
    )
    assert root_window == resumed_window
    assert float(
        learning_rate_at_step(schedule, current_step=12_500, schedule_origin_step=root_window.start_batch)
    ) == pytest.approx(
        float(
            learning_rate_at_step(
                schedule,
                current_step=12_500,
                schedule_origin_step=resumed_window.start_batch,
            )
        )
    )


def test_absolute_past_window_requires_explicit_inert_override() -> None:
    schedule = LrScheduleSpec(
        kind="warmup_cosine",
        learning_rate_0=0.1,
        total_steps=1_000,
        constant_lr_iterations=500,
        origin={"kind": "absolute", "batch": 1_000},
    )
    lineage = CheckpointSegmentLineage(
        start_batch=12_000,
        segment_batch_count=1_000,
        parent_transaction_id="parent",
    )
    with pytest.raises(ValueError, match="allow_inert=true"):
        resolve_schedule_window(schedule.origin, lineage=lineage, duration=schedule.total_steps)
    assert resolve_schedule_window(
        schedule.origin,
        lineage=lineage,
        duration=schedule.total_steps,
        allow_inert=True,
    ).start_batch == 1_000


def test_lr_schedule_v1_migrates_origin_and_v2_requires_it() -> None:
    migrated = LrScheduleSpec.model_validate(
        {
            "schema_version": "feedbax.spec.training.lr_schedule.v1",
            "kind": "constant",
            "learning_rate_0": 0.1,
        }
    )
    assert migrated.schema_version == LR_SCHEDULE_SPEC_SCHEMA_VERSION
    assert migrated.origin.kind == "run_start"

    with pytest.raises(ValueError, match="origin"):
        LrScheduleSpec.model_validate(
            {
                "schema_version": LR_SCHEDULE_SPEC_SCHEMA_VERSION,
                "kind": "constant",
                "learning_rate_0": 0.1,
            }
        )


def test_lr_continuation_modes_map_to_typed_clocks() -> None:
    assert lr_origin_for_continuation("continue").kind == "run_start"
    assert lr_origin_for_continuation("restart").kind == "segment_start"
