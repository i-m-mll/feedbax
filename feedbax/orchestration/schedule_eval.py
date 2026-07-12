"""Shared learning-rate schedule evaluation for orchestration checks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from feedbax.contracts.training import OptimizerSpec


@dataclass(frozen=True)
class ScheduleEvalContext:
    """Concrete inputs passed to ``build_optimizer`` schedule construction."""

    schedule_origin_step: int
    current_step: int
    optimizer_count_at_current_step: int

    def model_dump(self) -> dict[str, int]:
        return {
            "schedule_origin_step": self.schedule_origin_step,
            "current_step": self.current_step,
            "optimizer_count_at_current_step": self.optimizer_count_at_current_step,
        }


class MissingScheduleContext(ValueError):
    """Raised when scheduled optimizer evaluation lacks resume/build context."""


def learning_rate_from_build_optimizer(
    optimizer_spec: OptimizerSpec,
    *,
    sample_step: int,
    schedule_origin_step: int,
    current_step: int,
    optimizer_count_at_current_step: int,
) -> float:
    """Evaluate the live scheduled LR through the same optimizer builder used by runs."""
    import jax.numpy as jnp
    import jax.tree as jt

    from feedbax.training.optimizers import build_optimizer

    if sample_step < current_step:
        raise ValueError(f"lr_trace sample_step={sample_step} precedes current_step={current_step}")
    optimizer = build_optimizer(
        optimizer_spec,
        schedule_origin_step=schedule_origin_step,
        current_step=current_step,
        optimizer_count_at_current_step=optimizer_count_at_current_step,
    )
    params = {"w": jnp.asarray(1.0)}
    state = optimizer.init(params)
    target_count = optimizer_count_at_current_step + (sample_step - current_step)
    state = _with_injected_count(state, target_count)
    grads = jt.map(jnp.zeros_like, params)
    _updates, next_state = optimizer.update(grads, state, params)
    return _scheduled_learning_rate(next_state)


def extract_resume_context(
    bundle_row_spec: Mapping[str, Any] | None,
    training_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract declared resume/fork schedule context from row artifacts."""
    raw = _first_present(
        _path(bundle_row_spec, "resume_context"),
        _path(training_diagnostics, "resume_context"),
        {},
    )
    return {
        "schedule_origin_step": _first_present(
            _path(raw, "schedule_origin_step"),
            _path(bundle_row_spec, "schedule_origin_step"),
            _path(training_diagnostics, "schedule_origin_step"),
        ),
        "current_step": _first_present(
            _path(raw, "current_step"),
            _path(bundle_row_spec, "current_step"),
            _path(training_diagnostics, "current_step"),
        ),
        "optimizer_count_at_current_step": _first_present(
            _path(raw, "optimizer_count_at_current_step"),
            _path(bundle_row_spec, "optimizer_count_at_current_step"),
            _path(training_diagnostics, "optimizer_count_at_current_step"),
        ),
    }


def extract_optimizer_build_context(
    bundle_row_spec: Mapping[str, Any] | None,
    training_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract the explicit context the run path declares for optimizer construction.

    The executor's construction context must be recorded separately from the
    declared resume context. Falling back to the declaration would turn this
    check into a self-comparison and conceal dropped or mis-wired context.
    """
    raw = _first_present(
        _path(bundle_row_spec, "optimizer_build_context"),
        _path(bundle_row_spec, "build_optimizer_context"),
        _path(bundle_row_spec, "executor_optimizer_context"),
        _path(bundle_row_spec, "training", "optimizer_build_context"),
        _path(bundle_row_spec, "method_payload", "payload", "optimizer_build_context"),
        _path(training_diagnostics, "optimizer_build_context"),
        _MISSING,
    )
    if raw is _MISSING:
        return {
            "schedule_origin_step": _MISSING,
            "current_step": _MISSING,
            "optimizer_count_at_current_step": _MISSING,
        }
    return {
        "schedule_origin_step": _path(raw, "schedule_origin_step"),
        "current_step": _path(raw, "current_step"),
        "optimizer_count_at_current_step": _path(raw, "optimizer_count_at_current_step"),
    }


def require_schedule_context(context: Mapping[str, Any], *, label: str) -> ScheduleEvalContext:
    """Validate and coerce schedule context for optimizer schedule evaluation."""
    missing = [
        key
        for key in ("schedule_origin_step", "current_step", "optimizer_count_at_current_step")
        if context.get(key) is _MISSING
    ]
    if missing:
        raise MissingScheduleContext(f"{label} missing {', '.join(missing)}")
    return ScheduleEvalContext(
        schedule_origin_step=int(context["schedule_origin_step"]),
        current_step=int(context["current_step"]),
        optimizer_count_at_current_step=int(context["optimizer_count_at_current_step"]),
    )


def schedule_sample_steps(
    optimizer_spec: OptimizerSpec,
    context: ScheduleEvalContext,
    *,
    minimum: int = 4,
) -> tuple[int, ...]:
    """Choose schedule-relevant global sample steps at or after the current step."""
    schedule = optimizer_spec.lr_schedule
    local_positions = {0, 1, 2, 3}
    if schedule is not None and schedule.total_steps is not None:
        terminal = int(schedule.total_steps)
        warmup_or_hold = int(schedule.constant_lr_iterations)
        mid_decay = warmup_or_hold + max((terminal - warmup_or_hold) // 2, 1)
        local_positions = {0, warmup_or_hold, mid_decay, terminal}
        cursor = terminal - 1
        while len(local_positions) < minimum and cursor > 0:
            local_positions.add(cursor)
            cursor -= 1
    cursor = 0
    while len(local_positions) < minimum:
        local_positions.add(cursor)
        cursor += 1
    return tuple(context.current_step + position for position in sorted(local_positions))


def evaluate_schedule_samples(
    optimizer_spec: OptimizerSpec,
    context: ScheduleEvalContext,
    sample_steps: Sequence[int],
) -> dict[int, float]:
    """Evaluate scheduled LR values for sample steps using ``build_optimizer``."""
    return {
        int(step): learning_rate_from_build_optimizer(
            optimizer_spec,
            sample_step=int(step),
            schedule_origin_step=context.schedule_origin_step,
            current_step=context.current_step,
            optimizer_count_at_current_step=context.optimizer_count_at_current_step,
        )
        for step in sample_steps
    }


def compare_schedule_samples(
    optimizer_spec: OptimizerSpec,
    *,
    expected_context: ScheduleEvalContext,
    observed_context: ScheduleEvalContext,
    rel_tol: float = 1e-9,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return sampled LR triples and the subset whose values differ."""
    sample_steps = schedule_sample_steps(optimizer_spec, expected_context)
    expected = evaluate_schedule_samples(optimizer_spec, expected_context, sample_steps)
    observed = evaluate_schedule_samples(optimizer_spec, observed_context, sample_steps)
    samples = [
        {
            "sample_step": step,
            "expected": expected[step],
            "observed": observed[step],
        }
        for step in sample_steps
    ]
    mismatches = [
        sample
        for sample in samples
        if not math.isclose(
            float(sample["observed"]),
            float(sample["expected"]),
            rel_tol=rel_tol,
            abs_tol=0.0,
        )
    ]
    return samples, mismatches


def _with_injected_count(value: Any, count: int) -> Any:
    import jax.numpy as jnp

    if _is_injected_hyperparams_state(value):
        hyperparams_states = dict(value.hyperparams_states)
        patched_states = {
            key: state._replace(count=jnp.asarray(count, dtype=jnp.int32))
            for key, state in hyperparams_states.items()
            if hasattr(state, "_replace") and hasattr(state, "count")
        }
        hyperparams_states.update(patched_states)
        return value._replace(
            count=jnp.asarray(count, dtype=jnp.int32),
            hyperparams_states=hyperparams_states,
        )
    if isinstance(value, tuple):
        return type(value)(_with_injected_count(item, count) for item in value)
    if isinstance(value, list):
        return [_with_injected_count(item, count) for item in value]
    if isinstance(value, dict):
        return {key: _with_injected_count(item, count) for key, item in value.items()}
    return value


def _scheduled_learning_rate(value: Any) -> float:
    import jax.tree as jt

    leaves = jt.leaves(value, is_leaf=_is_injected_hyperparams_state)
    for leaf in leaves:
        if _is_injected_hyperparams_state(leaf):
            return float(leaf.hyperparams["learning_rate"])
    raise ValueError("scheduled optimizer state not found")


def _is_injected_hyperparams_state(value: Any) -> bool:
    fields = getattr(value, "_fields", ())
    return {"hyperparams", "inner_state"}.issubset(set(fields))


class _Missing:
    pass


_MISSING = _Missing()


def _path(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if current is None:
            return _MISSING
        if isinstance(current, Mapping):
            if key not in current:
                return _MISSING
            current = current[key]
            continue
        current = getattr(current, key, _MISSING)
        if current is _MISSING:
            return _MISSING
    return current


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not _MISSING:
            return value
    return _MISSING
