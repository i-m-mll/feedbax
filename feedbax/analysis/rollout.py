"""Compile-once batched execution of per-trial rollouts.

Evaluation recipes that call a per-trial rollout once per trial, with a fresh
Python closure each time, force JAX to trace and compile a fresh jaxpr on every
call. This module provides the opposite discipline as an opt-in facility: a pure
per-trial rollout function is mapped over a stacked-trials pytree with
`jax.lax.map` inside a single module-level `jax.jit`, so one executable is
compiled per (function, shape/dtype structure) per process and reused for every
subsequent batch.

The facility is controller-law agnostic. It knows nothing about plants,
controllers, tasks, or feedbax's `Component`/`Graph` model: the caller supplies
a pure callable `(context, trial) -> result` where `context` holds every
trial-invariant operand and `trial` is one slice of the stacked trials.

This is not a second way to run feedbax models, and it does not overlap the
`Component` execution stack. Models built as `Component`/`Graph` with
`equinox.nn.State` are batched over trials by `AbstractTask.eval_trials` and
`eval_ensemble_on_trials` (`feedbax/tasks/task.py`) using `filter_vmap`, and
their time-step iteration belongs to `run_component`
(`feedbax/runtime/iteration.py`) — use those for `Component` models. This module
performs no time iteration whatsoever: the time loop, when there is one, lives
inside the caller's `per_trial` callable. It exists only for pure non-`Component`
callables that need sequential `lax.map` batching over trials, with bounded
memory relative to `vmap` and a compile-once cache contract that prevents
per-closure retracing, as used by evaluation recipes that own their own
execution.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import jax.tree as jt
from jax.tree_util import keystr
from jaxtyping import PyTree

__all__ = [
    "TrialStructureError",
    "compiled_trial_rollout",
    "stack_trials",
]

_Context = TypeVar("_Context")
_Trial = TypeVar("_Trial")
_Result = TypeVar("_Result")


class TrialStructureError(ValueError):
    """Raised when trial operands violate the stacked-trials contract.

    Raised only outside the compiled region, by validation that inspects shapes,
    dtypes, and pytree structure without touching numeric values.
    """


@partial(jax.jit, static_argnums=(0,))
def _mapped_trial_rollout(
    per_trial: Callable[[Any, Any], Any],
    context: Any,
    trials: Any,
) -> Any:
    """Map `per_trial` over the leading trial axis of `trials` inside one jit.

    `per_trial` is a static argument, so `jax.jit`'s own cache is keyed on
    `(per_trial, avals(context), avals(trials))`: exactly one compilation per
    function and shape/dtype structure per process, with no additional cache.
    """
    return jax.lax.map(lambda trial: per_trial(context, trial), trials)


def _leaf_shape_dtype(leaf: Any) -> tuple[tuple[int, ...], Any]:
    """Return `(shape, dtype)` for an array-like leaf without converting it."""
    return jnp.shape(leaf), jnp.result_type(leaf)


def _validate_stacked_trials(
    trials: PyTree[Any],
    trial_structure: PyTree[jax.ShapeDtypeStruct] | None,
) -> int:
    """Validate stacked trial operands and return the trial-axis size.

    Fail-closed structural validation only; it reads shapes, dtypes, and pytree
    structure and never inspects or alters numeric values.

    Args:
        trials: Pytree whose leaves each carry a leading trial axis.
        trial_structure: Optional pytree of `jax.ShapeDtypeStruct` describing a
            *single* trial, i.e. without the leading trial axis.

    Returns:
        The common leading-axis size.

    Raises:
        TrialStructureError: If the pytree has no leaves, a leaf is scalar, the
            leading axis sizes disagree, the trial axis is empty, or a declared
            `trial_structure` does not match.
    """
    entries, treedef = jt.flatten_with_path(trials)
    if not entries:
        raise TrialStructureError("stacked trials pytree contains no array leaves")

    sizes: dict[int, list[str]] = {}
    for path, leaf in entries:
        shape, _ = _leaf_shape_dtype(leaf)
        if not shape:
            raise TrialStructureError(
                f"stacked trial leaf {keystr(path)} is a scalar; every leaf must carry a "
                "leading trial axis"
            )
        sizes.setdefault(shape[0], []).append(keystr(path))
    if len(sizes) > 1:
        detail = ", ".join(f"{size} at {paths[0]}" for size, paths in sorted(sizes.items()))
        raise TrialStructureError(
            f"stacked trial leaves disagree on the leading trial axis: {detail}"
        )
    (size,) = sizes
    if size == 0:
        raise TrialStructureError("stacked trials have an empty leading trial axis")

    if trial_structure is not None:
        declared_entries, declared_treedef = jt.flatten_with_path(trial_structure)
        if declared_treedef != treedef:
            raise TrialStructureError(
                f"stacked trials structure {treedef} does not match the declared per-trial "
                f"structure {declared_treedef}"
            )
        for (path, leaf), (_, declared) in zip(entries, declared_entries):
            shape, dtype = _leaf_shape_dtype(leaf)
            if shape[1:] != tuple(declared.shape):
                raise TrialStructureError(
                    f"stacked trial leaf {keystr(path)} has per-trial shape {shape[1:]}, "
                    f"declared {tuple(declared.shape)}"
                )
            if dtype != declared.dtype:
                raise TrialStructureError(
                    f"stacked trial leaf {keystr(path)} has dtype {dtype}, "
                    f"declared {declared.dtype}"
                )
    return size


def stack_trials(trials: Sequence[PyTree[Any]]) -> PyTree[Any]:
    """Stack per-trial pytrees into one pytree with a leading trial axis.

    Args:
        trials: Non-empty sequence of per-trial pytrees that share one pytree
            structure and, leafwise, one shape and dtype.

    Returns:
        A pytree of the same structure whose leaves are the stacked leaves, each
        of shape `(len(trials), *leaf_shape)`.

    Raises:
        TrialStructureError: If the sequence is empty, or the trials disagree on
            pytree structure, leaf shape, or leaf dtype.
    """
    if len(trials) == 0:
        raise TrialStructureError("stack_trials requires at least one trial")

    reference_entries, reference_treedef = jt.flatten_with_path(trials[0])
    for index, trial in enumerate(trials[1:], start=1):
        entries, treedef = jt.flatten_with_path(trial)
        if treedef != reference_treedef:
            raise TrialStructureError(
                f"trial {index} has pytree structure {treedef}, but trial 0 has {reference_treedef}"
            )
        for (path, leaf), (_, reference) in zip(entries, reference_entries):
            shape, dtype = _leaf_shape_dtype(leaf)
            reference_shape, reference_dtype = _leaf_shape_dtype(reference)
            if shape != reference_shape:
                raise TrialStructureError(
                    f"trial {index} leaf {keystr(path)} has shape {shape}, but trial 0 has "
                    f"{reference_shape}"
                )
            if dtype != reference_dtype:
                raise TrialStructureError(
                    f"trial {index} leaf {keystr(path)} has dtype {dtype}, but trial 0 has "
                    f"{reference_dtype}"
                )
    return jt.map(lambda *values: jnp.stack(values), *trials)


def compiled_trial_rollout(
    per_trial: Callable[[_Context, _Trial], _Result],
    *,
    trial_structure: PyTree[jax.ShapeDtypeStruct] | None = None,
) -> Callable[[_Context, PyTree[Any]], PyTree[Any]]:
    """Build a batch rollout callable that compiles once per shape structure.

    The returned callable takes `(context, trials)` — `context` being every
    trial-invariant operand and `trials` the stacked per-trial operands from
    [`stack_trials`][feedbax.analysis.rollout.stack_trials] — validates the
    stacked trials outside the compiled region, then runs `per_trial` over the
    leading trial axis with `jax.lax.map` inside one `jax.jit`. Results are
    returned stacked along the same leading axis, and are bit-identical to
    calling `per_trial` on each trial in a Python loop and stacking.

    Compile-once contract:

    - The only compilation cache is `jax.jit`'s own, on a module-level jitted
      driver that takes `per_trial` as a static argument. One executable is
      compiled per `(per_trial, shape/dtype structure of context and trials)`
      per process; no state is added by this module.
    - `per_trial` must therefore have stable identity across calls, since static
      arguments are cached by hash and equality and plain functions use object
      identity. Define it at module level, or hold onto one `functools.partial`
      of it. Building a fresh closure or lambda per call misses the cache and
      retraces every time — the defect this facility exists to prevent — and
      also pins every such function in the process-lifetime jit cache.
    - The wrapper returned here holds no cache of its own, so it is cheap to
      rebuild; keeping it at module level is a readability choice, not a
      correctness requirement.

    `context` and `trials` are ordinary (traced) `jax.jit` arguments, so their
    leaves must all be JAX-compatible arrays. Values that must be static, such
    as a mode flag that changes the traced program, belong in `per_trial`
    itself — for example as a module-level variant function per mode.

    Args:
        per_trial: Pure function `(context, trial) -> result`, where `trial` is
            one slice of `trials` along the leading axis.
        trial_structure: Optional pytree of `jax.ShapeDtypeStruct` describing a
            single trial, without the leading trial axis. When given, stacked
            trials are checked against it before compilation.

    Returns:
        A callable `(context, trials) -> results` mapping over the leading trial
        axis.

    Raises:
        TypeError: If `per_trial` is not callable.
    """
    if not callable(per_trial):
        raise TypeError(f"per_trial must be callable, got {type(per_trial).__name__}")

    def rollout(context: _Context, trials: PyTree[Any]) -> PyTree[Any]:
        _validate_stacked_trials(trials, trial_structure)
        return _mapped_trial_rollout(per_trial, context, trials)

    rollout.__name__ = f"compiled_trial_rollout[{getattr(per_trial, '__name__', 'per_trial')}]"
    rollout.__qualname__ = rollout.__name__
    rollout.__doc__ = (
        "Run stacked trials through the compiled rollout for "
        f"{getattr(per_trial, '__qualname__', per_trial)!s}."
    )
    return rollout
