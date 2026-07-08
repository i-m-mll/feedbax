"""Public optimizer construction from Feedbax training contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import inspect
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
            and does not depend on a restored optimizer state's ``count`` field.
        gradient_clip: Optional global-norm clipping threshold prepended to the
            optimizer.

    Returns:
        Optax gradient transformation. Scheduled optimizers use
        ``inject_hyperparams`` so the live learning rate is visible in optimizer
        state diagnostics.
    """
    spec = OptimizerSpec.model_validate(optimizer_spec)
    params = dict(spec.params)
    transforms: list[optax.GradientTransformationExtraArgs | optax.GradientTransformation] = []
    if gradient_clip is not None:
        transforms.append(optax.clip_by_global_norm(float(gradient_clip)))

    if spec.lr_schedule is None:
        learning_rate = params.pop("learning_rate", None)
        if learning_rate is None:
            raise ValueError("/params/learning_rate is required when lr_schedule is absent")
        transforms.append(_optimizer_factory(spec.type)(float(learning_rate), **params))
        return optax.chain(*transforms)

    if "learning_rate" in params:
        raise ValueError(
            "/params/learning_rate must be omitted when /lr_schedule declares the schedule"
        )
    schedule_position = int(current_step) - int(schedule_origin_step)
    if schedule_position < 0:
        raise ValueError(
            "current_step must be greater than or equal to schedule_origin_step; "
            f"got current_step={current_step}, schedule_origin_step={schedule_origin_step}"
        )
    base_schedule = learning_rate_schedule(spec.lr_schedule)

    def shifted_schedule(count: Any) -> jax.Array:
        return base_schedule(jnp.asarray(count, dtype=jnp.int32) + schedule_position)

    factory = _optimizer_factory(spec.type)
    static_args = ("mask",) if "mask" in inspect.signature(factory).parameters else ()
    transforms.append(
        optax.inject_hyperparams(factory, static_args=static_args)(
            learning_rate=shifted_schedule,
            **params,
        )
    )
    return optax.chain(*transforms)


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


def _optimizer_factory(
    optimizer_type: str,
) -> Callable[..., optax.GradientTransformation]:
    factories: Mapping[str, Callable[..., optax.GradientTransformation]] = {
        "adamw": optax.adamw,
        "adam": optax.adam,
        "sgd": optax.sgd,
        "rmsprop": optax.rmsprop,
    }
    try:
        return factories[optimizer_type.lower()]
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
