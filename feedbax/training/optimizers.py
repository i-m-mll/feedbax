"""Public optimizer construction from Feedbax training contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import inspect
from math import isfinite
from typing import Any

import jax
import optax

from feedbax.contracts.training import LrScheduleSpec, OptimizerSpec


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
