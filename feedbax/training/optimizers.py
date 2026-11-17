"""Public optimizer construction from Feedbax training contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import inspect
from math import isfinite
from typing import Any

import jax
import jax.numpy as jnp
import optax

from feedbax.contracts.training import LrScheduleSpec, OptimizerSpec


class OptimizerStateCompatibilityError(ValueError):
    """Raised when a restored optimizer state does not match the declared spec."""


def learning_rate_schedule(schedule_spec: LrScheduleSpec) -> optax.Schedule:
    """Return an Optax schedule for origin-relative learning-rate positions.

    Args:
        schedule_spec: Declarative schedule contract.

    Returns:
        Callable mapping an origin-relative step to a learning rate.
    """
    spec = LrScheduleSpec.model_validate(schedule_spec)
    learning_rate = float(spec.learning_rate_0)
    if spec.kind == "constant":
        return optax.constant_schedule(learning_rate)
    if spec.total_steps is None:
        raise ValueError(f"{spec.kind} schedules require total_steps")
    if spec.kind == "warmup_cosine":
        return optax.warmup_cosine_decay_schedule(
            init_value=learning_rate * float(spec.warmup_init_fraction),
            peak_value=learning_rate,
            warmup_steps=int(spec.constant_lr_iterations),
            decay_steps=int(spec.total_steps),
            end_value=learning_rate * float(spec.cosine_annealing_alpha),
        )
    if spec.kind == "delayed_cosine":
        return optax.join_schedules(
            schedules=[
                optax.constant_schedule(learning_rate),
                optax.cosine_decay_schedule(
                    init_value=learning_rate,
                    decay_steps=int(spec.total_steps) - int(spec.constant_lr_iterations),
                    alpha=float(spec.cosine_annealing_alpha),
                ),
            ],
            boundaries=[int(spec.constant_lr_iterations)],
        )
    raise ValueError(f"Unsupported learning-rate schedule kind {spec.kind!r}")


def learning_rate_at_step(
    schedule_spec: LrScheduleSpec,
    *,
    current_step: int,
    schedule_origin_step: int,
) -> jax.Array:
    """Evaluate a schedule using an explicit origin-relative clock.

    Args:
        schedule_spec: Declarative schedule contract.
        current_step: Global step at which the learning rate is needed.
        schedule_origin_step: Global step that defines schedule position zero.

    Returns:
        Learning rate at ``current_step - schedule_origin_step``.

    Raises:
        ValueError: If ``current_step`` is before ``schedule_origin_step``.
    """
    schedule_position = int(current_step) - int(schedule_origin_step)
    if schedule_position < 0:
        raise ValueError(
            "current_step must be greater than or equal to schedule_origin_step; "
            f"got current_step={current_step}, schedule_origin_step={schedule_origin_step}"
        )
    return learning_rate_schedule(schedule_spec)(jnp.asarray(schedule_position, dtype=jnp.int32))


def build_optimizer(
    optimizer_spec: OptimizerSpec,
    *,
    schedule_origin_step: int,
    current_step: int,
    optimizer_count_at_current_step: int = 0,
    gradient_clip: float | None = None,
) -> optax.GradientTransformationExtraArgs:
    """Build an Optax optimizer from an ``OptimizerSpec``.

    Args:
        optimizer_spec: Declarative optimizer contract. When ``lr_schedule`` is
            present, ``params`` must not also declare ``learning_rate`` because
            the schedule owns that hyperparameter.
        schedule_origin_step: Global step that maps to schedule position zero.
            ``resume_barrier`` callers pass the resume step for continuation-
            local schedules; ``run_start`` callers pass zero.
        current_step: Global step at which the returned optimizer starts. The
            schedule offset is computed as ``current_step - schedule_origin_step``
            before correcting for ``optimizer_count_at_current_step``.
        optimizer_count_at_current_step: Optax injected-hyperparameter schedule
            count stored in the optimizer state at ``current_step``. Freshly
            initialized states use the default ``0``. Restored states should pass
            their restored count at the resume/current step so schedule position
            is ``(count - optimizer_count_at_current_step) + (current_step -
            schedule_origin_step)`` instead of implicitly following the restored
            Optax count.
        gradient_clip: Optional global-norm clipping threshold prepended to the
            optimizer.

    Returns:
        Optax gradient transformation. Scheduled optimizers use
        ``inject_hyperparams`` so the live learning rate is visible in optimizer
        state diagnostics.
    """
    spec = OptimizerSpec.model_validate(optimizer_spec)
    params = dict(spec.params)
    if spec.lr_schedule is None:
        rate = _fixed_rate(params)
        return _chain_optimizer(_optimizer_factory(spec.type)(rate, **params), gradient_clip)

    _require_scheduled_rate(params)
    schedule_position = int(current_step) - int(schedule_origin_step)
    if schedule_position < 0:
        raise ValueError(
            "current_step must be greater than or equal to schedule_origin_step; "
            f"got current_step={current_step}, schedule_origin_step={schedule_origin_step}"
        )
    base_schedule = learning_rate_schedule(spec.lr_schedule)
    count_origin = int(optimizer_count_at_current_step)

    def shifted_schedule(count: Any) -> jax.Array:
        return base_schedule(
            (jnp.asarray(count, dtype=jnp.int32) - count_origin) + schedule_position
        )

    return _schedule_optimizer(spec.type, params, shifted_schedule, gradient_clip)


@dataclass(frozen=True)
class _StepSchedule:
    rate: optax.Schedule

    def init(self) -> tuple[()]:
        return ()

    def update(self, state: tuple[()], **_extra_args) -> tuple[()]:
        return state

    def __call__(self, _state: tuple[()], *, schedule_step: int | jax.Array = 0, **_extra_args):
        return self.rate(schedule_step)


def build_step_optimizer(
    spec: OptimizerSpec, *, gradient_clip: float | None = None,
) -> optax.GradientTransformationExtraArgs:
    """Build an optimizer whose learning rate follows an explicit update clock."""
    params = dict(spec.params)
    if spec.lr_schedule is None:
        schedule = optax.constant_schedule(_fixed_rate(params))
    else:
        _require_scheduled_rate(params)
        schedule = learning_rate_schedule(spec.lr_schedule)
    return _schedule_optimizer(spec.type, params, _StepSchedule(schedule), gradient_clip)


def _fixed_rate(params: dict[str, Any]) -> float:
    rate = params.pop("learning_rate", None)
    if type(rate) not in (int, float) or not isfinite(rate) or rate <= 0:
        raise ValueError("/params/learning_rate requires a finite positive rate")
    return float(rate)


def _require_scheduled_rate(params: Mapping[str, Any]) -> None:
    if "learning_rate" in params:
        raise ValueError(
            "/params/learning_rate must be omitted when /lr_schedule declares the schedule"
        )


def _schedule_optimizer(kind, params, schedule, gradient_clip):
    factory = _optimizer_factory(kind)
    static_args = ("mask",) if "mask" in inspect.signature(factory).parameters else ()
    optimizer = optax.inject_hyperparams(factory, static_args=static_args)(learning_rate=schedule, **params)
    return _chain_optimizer(optimizer, gradient_clip)


def _chain_optimizer(optimizer, gradient_clip):
    if gradient_clip is None:
        return optax.chain(optimizer)
    return optax.chain(optax.clip_by_global_norm(float(gradient_clip)), optimizer)


def validate_optimizer_state_for_spec(optimizer_spec: OptimizerSpec, opt_state: Any) -> None:
    """Reject incompatible restored optimizer state for a declared schedule.

    Args:
        optimizer_spec: Optimizer contract associated with the run.
        opt_state: Restored Optax state tree.

    Raises:
        OptimizerStateCompatibilityError: If ``optimizer_spec`` declares an LR
            schedule but ``opt_state`` does not contain Optax injected-
            hyperparameter state.
    """
    spec = OptimizerSpec.model_validate(optimizer_spec)
    if spec.lr_schedule is None:
        return
    if _contains_injected_hyperparams_state(opt_state):
        return
    raise OptimizerStateCompatibilityError(
        "OptimizerSpec declares lr_schedule, but the restored optimizer state "
        "does not contain Optax injected-hyperparameter schedule state. "
        "Resume with a freshly initialized scheduled optimizer state or provide "
        "an explicit, tested optimizer-state conversion."
    )


_OPTIMIZER_FACTORIES: Mapping[str, Callable[..., optax.GradientTransformation]] = {
    "adamw": optax.adamw,
    "adam": optax.adam,
    "sgd": optax.sgd,
    "rmsprop": optax.rmsprop,
}


def optimizer_types() -> tuple[str, ...]:
    """Return the names accepted by the optimizer builder."""
    return tuple(_OPTIMIZER_FACTORIES)


def _optimizer_factory(optimizer_type: str) -> Callable[..., optax.GradientTransformation]:
    try:
        return _OPTIMIZER_FACTORIES[optimizer_type.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported optimizer type {optimizer_type!r}") from exc


def _contains_injected_hyperparams_state(tree: Any) -> bool:
    if _is_injected_hyperparams_state(tree):
        return True
    leaves = jax.tree.leaves(tree, is_leaf=_is_injected_hyperparams_state)
    return any(_is_injected_hyperparams_state(leaf) for leaf in leaves)


def _is_injected_hyperparams_state(value: Any) -> bool:
    fields = getattr(value, "_fields", ())
    return {"hyperparams", "inner_state"}.issubset(set(fields))
