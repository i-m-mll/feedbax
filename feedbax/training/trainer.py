"""Small training helpers shared by analysis code and executor kernels.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from collections.abc import Callable, Mapping
from functools import wraps
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
import optax  # type: ignore
from equinox import field
from jax_cookbook.misc import mse
from jax_cookbook.progress import piter
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from feedbax.objectives.loss import AbstractLoss, TermTree
from feedbax.runtime.graph import Component, RolloutStepHook
from feedbax.runtime.iteration import run_component
from feedbax.runtime.parameter_constraints import project_component_parameters
from feedbax.runtime.state import StateT
from feedbax.tasks import (
    TaskTrialSpec,
    extract_timeseries_params,
    infer_n_steps,
    prepare_inputs,
)


def _cast_to_state_dtypes(new_value, current_value):
    """Cast a replacement StateIndex value to the stored State leaf dtypes."""

    def _cast_leaf(new_leaf, current_leaf):
        if hasattr(current_leaf, "dtype"):
            if getattr(current_leaf, "weak_type", False):
                raise ValueError(
                    "Cannot update a weakly typed StateIndex leaf with public "
                    "Equinox State.set(); initialize the component StateIndex "
                    "with an explicitly typed JAX array."
                )
            return jnp.asarray(new_leaf, dtype=current_leaf.dtype)
        return new_leaf

    return jt.map(_cast_leaf, new_value, current_value)


def _state_set_matching_dtypes(state, idx, new_value):
    """Set a StateIndex value via public Equinox API after dtype normalization."""

    current_value = state.get(idx)
    return state.set(idx, _cast_to_state_dtypes(new_value, current_value))


def grad_wrap_simple_loss_func(
    loss_func: Callable[[Array, Array], Float],
    nan_safe: bool = False,
):
    """Adapt an ``(output, target)`` loss to a ``jax.grad``-ready model loss."""

    if nan_safe:

        def _forward_pass(model, x):
            x_cleaned = jnp.nan_to_num(x, nan=0.0)
            return model(x_cleaned)

    else:

        def _forward_pass(model, x):
            return model(x)

    @wraps(loss_func)
    def wrapper(
        model,
        x,
        y,
    ) -> Tuple[float, TermTree[AbstractLoss]]:
        return loss_func(_forward_pass(model, x), y)

    return wrapper


class SimpleTrainer(eqx.Module):
    """Train a model on a whole in-memory dataset for a fixed iteration count."""

    loss_func: Callable[[eqx.Module, Array, Array], Float] = field(
        default=grad_wrap_simple_loss_func(mse),
    )
    optimizer: optax.GradientTransformation = field(
        default=optax.sgd(1e-2),
    )

    def __call__(self, model, x, y, n_iter=100, progress_bar=True):
        opt_state = self.optimizer.init(model)

        def train_step(current_model, current_opt_state, x_batch, y_batch):
            loss_value, grads = jax.value_and_grad(self.loss_func)(
                current_model,
                x_batch,
                y_batch,
            )
            updates, new_opt_state = self.optimizer.update(grads, current_opt_state)
            new_model = eqx.apply_updates(current_model, updates)
            return new_model, new_opt_state, loss_value

        if progress_bar:
            train_iter = piter(range(n_iter), description="Training")
        else:
            train_iter = range(n_iter)

        for _ in train_iter:
            model, opt_state, loss = train_step(model, opt_state, x, y)
            if progress_bar:
                train_iter.set_postfix(loss=f"{loss:.4f}")

        return model


def _extract_intervene_inputs(
    intervene: Mapping,
    model: Component,
) -> dict[str, PyTree]:
    """Extract time-varying intervention params from a trial spec."""

    indices = model.intervention_state_indices()
    result = {}
    for label, params in intervene.items():
        if label not in indices:
            continue
        idx = indices[label]
        default_params = idx.init
        tv_params = extract_timeseries_params(params, default_params)
        if tv_params is not None:
            result[f"intervene:{label}"] = tv_params
    return result


def grad_wrap_abstract_loss(
    loss_func: AbstractLoss,
    loss_reduction_fn: Optional[Callable] = None,
    rollout_step_hook: Optional[RolloutStepHook] = None,
):
    """Wrap a state-based task loss so gradients are taken over trainable parameters."""

    @wraps(loss_func)
    def wrapper(
        diff_model: Component,
        static_model: Component,
        trial_specs: TaskTrialSpec,
        init_states: StateT,
        keys: PRNGKeyArray,
    ) -> Tuple[Array, Tuple[TermTree[AbstractLoss], StateT]]:
        model = eqx.combine(diff_model, static_model)

        def _run_trial(trial_spec, init_state, key):
            inputs = prepare_inputs(model, trial_spec.inputs)
            n_steps = infer_n_steps(inputs, getattr(trial_spec, "timeline", None))
            if trial_spec.intervene:
                intervene_inputs = _extract_intervene_inputs(
                    trial_spec.intervene,
                    model,
                )
                if intervene_inputs:
                    inputs = {**inputs, **intervene_inputs}
            _, _, state_history = run_component(
                model,
                inputs,
                init_state,
                key=key,
                n_steps=n_steps,
                rollout_step_hook=rollout_step_hook,
            )
            return jt.map(lambda x: x[1:] if x is not None else x, state_history)

        states: StateT = eqx.filter_vmap(_run_trial)(trial_specs, init_states, keys)
        losses = loss_func(states, trial_specs, model)

        if loss_reduction_fn is not None:
            per_trial_total = losses.aggregate(leaf_fn=lambda x: x)
            total = loss_reduction_fn(per_trial_total)
        else:
            total = losses.total
        return total, (losses, states)

    return wrapper


def training_step_for_abstract_loss(
    optimizer: optax.GradientTransformation,
    model: Component,
    opt_state: optax.OptState,
    loss_func: AbstractLoss,
    where_train_spec,
    trial_specs: TaskTrialSpec,
    init_states: StateT,
    keys: PRNGKeyArray,
    *,
    loss_reduction_fn: Optional[Callable] = None,
    rollout_step_hook: Optional[RolloutStepHook] = None,
) -> tuple[Component, optax.OptState, TermTree[AbstractLoss], StateT]:
    """Run one supervised gradient update using the executor-native helper stack."""

    diff_model, static_model = eqx.partition(model, where_train_spec)
    (_, (losses, states)), grads = eqx.filter_value_and_grad(
        grad_wrap_abstract_loss(
            loss_func,
            loss_reduction_fn=loss_reduction_fn,
            rollout_step_hook=rollout_step_hook,
        ),
        has_aux=True,
    )(
        diff_model,
        static_model,
        trial_specs,
        init_states,
        keys,
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return project_component_parameters(model), opt_state, losses, states
