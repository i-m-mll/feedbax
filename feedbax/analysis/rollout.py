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
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
import jax.tree as jt
from jax.tree_util import keystr
from jaxtyping import PyTree

from feedbax.contracts.manifest import (
    EvaluationRunSpec,
    canonical_json_bytes,
    sha256_bytes,
)

__all__ = [
    "EvaluationStateCapture",
    "EvaluationStateIdentity",
    "EvaluationStateIdentityMismatch",
    "EvaluationStateProvenance",
    "TrialStructureError",
    "capture_evaluation_state",
    "compiled_trial_rollout",
    "stack_trials",
]

_Context = TypeVar("_Context")
_Prefix = TypeVar("_Prefix")
_PrefixResult = TypeVar("_PrefixResult")
_State = TypeVar("_State")
_Trial = TypeVar("_Trial")
_Result = TypeVar("_Result")


class EvaluationStateIdentityMismatch(ValueError):
    """Raised when a captured state does not belong to the requested model/state contract."""


def _require_identity_token(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class EvaluationStateIdentity:
    """Stable caller-owned model and state-contract identities."""

    model: str
    state: str

    def __post_init__(self) -> None:
        _require_identity_token(self.model, name="model identity")
        _require_identity_token(self.state, name="state identity")


@dataclass(frozen=True)
class EvaluationStateProvenance:
    """Reference to the run spec that owns prefix/task provenance."""

    run_spec: EvaluationRunSpec
    prefix_steps_param: str
    task_pin_param: str
    _run_spec_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.run_spec, EvaluationRunSpec):
            raise TypeError(
                f"run_spec must be EvaluationRunSpec, got {type(self.run_spec).__name__}"
            )
        _require_identity_token(self.prefix_steps_param, name="prefix steps parameter")
        _require_identity_token(self.task_pin_param, name="task pin parameter")
        self._validate_params()
        object.__setattr__(
            self,
            "_run_spec_sha256",
            sha256_bytes(
                canonical_json_bytes(self.run_spec.model_dump(mode="json", exclude_none=True))
            ),
        )

    @property
    def prefix_steps(self) -> int:
        return self.run_spec.params[self.prefix_steps_param]

    @property
    def task_pin(self) -> str:
        return self.run_spec.params[self.task_pin_param]

    def validate(self) -> None:
        self._validate_params()
        actual_sha256 = sha256_bytes(
            canonical_json_bytes(self.run_spec.model_dump(mode="json", exclude_none=True))
        )
        if actual_sha256 != self._run_spec_sha256:
            raise EvaluationStateIdentityMismatch("evaluation run spec changed after state capture")

    def _validate_params(self) -> None:
        if self.prefix_steps_param not in self.run_spec.params:
            raise ValueError(
                f"evaluation run spec has no {self.prefix_steps_param!r} prefix parameter"
            )
        if self.task_pin_param not in self.run_spec.params:
            raise ValueError(
                f"evaluation run spec has no {self.task_pin_param!r} task-pin parameter"
            )
        prefix_steps = self.run_spec.params[self.prefix_steps_param]
        if not isinstance(prefix_steps, int) or isinstance(prefix_steps, bool) or prefix_steps < 0:
            raise ValueError(f"{self.prefix_steps_param!r} must be a nonnegative integer")
        _require_identity_token(
            self.run_spec.params[self.task_pin_param],
            name=f"{self.task_pin_param!r} task pin",
        )


_StateLeaves = tuple[tuple[str, tuple[int, ...], str], ...]
_StateStructure = tuple[Any, _StateLeaves]


def _state_structure(state: Any) -> _StateStructure:
    entries, treedef = jt.flatten_with_path(state)
    return (
        treedef,
        tuple(
            (keystr(path), tuple(jnp.shape(leaf)), str(jnp.result_type(leaf)))
            for path, leaf in entries
        ),
    )


@dataclass(frozen=True)
class EvaluationStateCapture(Generic[_State]):
    """Terminal state captured from one rollout prefix for reuse within an execution."""

    state: _State
    identity: EvaluationStateIdentity
    provenance: EvaluationStateProvenance
    _structure: _StateStructure = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EvaluationStateIdentity):
            raise TypeError("identity must be EvaluationStateIdentity")
        if not isinstance(self.provenance, EvaluationStateProvenance):
            raise TypeError("provenance must be EvaluationStateProvenance")
        object.__setattr__(self, "_structure", _state_structure(self.state))

    def resume(self, expected_identity: EvaluationStateIdentity) -> _State:
        """Return the state after fail-closed identity and provenance validation."""
        self.provenance.validate()
        if self.identity != expected_identity:
            raise EvaluationStateIdentityMismatch(
                "captured evaluation state identity mismatch: "
                f"captured model={self.identity.model!r}, state={self.identity.state!r}; "
                f"expected model={expected_identity.model!r}, state={expected_identity.state!r}"
            )
        actual_structure = _state_structure(self.state)
        if actual_structure != self._structure:
            raise EvaluationStateIdentityMismatch(
                "captured evaluation state structure changed after capture"
            )
        return self.state


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

    This is the simple case: per-trial pytrees that are already in their final
    layout and dtype, as in parity references and recipes whose trials need no
    preparation. Operand builders that assemble stacked trials field by field —
    casting dtypes, deriving fields, or drawing some leaves from outside the
    per-trial objects — should keep doing that and pass the result straight to
    the rollout callable.

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


def capture_evaluation_state(
    prefix_rollout: Callable[[_Context, _Prefix], _PrefixResult],
    context: _Context,
    prefix: _Prefix,
    *,
    terminal_state: Callable[[_PrefixResult], _State],
    identity: EvaluationStateIdentity,
    provenance: EvaluationStateProvenance,
) -> tuple[_PrefixResult, EvaluationStateCapture[_State]]:
    """Run one prefix and capture its terminal state for in-memory row reuse.

    Args:
        prefix_rollout: Pure or eager callable `(context, prefix) -> result`.
        context: Trial-invariant rollout operands.
        prefix: The single wash-in/prefix trial operands.
        terminal_state: Selector mapping the prefix result to the complete state
            needed to resume a row.
        identity: Stable model and state-contract identities.
        provenance: Reference to the run spec that records prefix/task facts.

    Returns:
        The prefix result and an in-memory typed state capture.
    """
    if not callable(prefix_rollout):
        raise TypeError(f"prefix_rollout must be callable, got {type(prefix_rollout).__name__}")
    if not callable(terminal_state):
        raise TypeError(f"terminal_state must be callable, got {type(terminal_state).__name__}")
    result = prefix_rollout(context, prefix)
    state = terminal_state(result)
    capture = EvaluationStateCapture(
        state=state,
        identity=identity,
        provenance=provenance,
    )
    return result, capture


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
    returned stacked along the same leading axis. Feedbax's parity tests show
    that they match an eager per-trial Python loop byte for byte on the tested
    backend, dtypes, and shapes; XLA guarantees no such thing in general, since
    a `scan` of a function may fuse differently from eager execution of it, so
    those parity tests are the canary rather than a standing promise.

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
    - Array operands belong in `context`, never in `per_trial`'s closure or
      `functools.partial` arguments: a captured array is baked into the
      executable as a compile-time constant and, because each fresh capture is a
      fresh static function, silently retraces on every call without raising.
      Adapting a callable that takes more than `(context, trial)` means widening
      `context`, not partially applying the extra operands.
    - The wrapper returned here holds no cache of its own, so it is cheap to
      rebuild; keeping it at module level is a readability choice, not a
      correctness requirement.

    `context` and `trials` are ordinary (traced) `jax.jit` arguments, so their
    leaves must all be JAX-compatible arrays. Values that must be static, such
    as a mode flag that changes the traced program, belong in `per_trial`
    itself — for example as a module-level variant function per mode.

    Only `trials` is validated here, so keeping `context` stable across calls is
    the caller's responsibility: build its leaves with explicit dtypes rather
    than letting Python scalars in as weakly typed values, since a leaf that
    arrives as `float32` on one call and weak-typed or `float64` on the next is
    a different aval and costs another compilation — and, where the rollout
    carries mixed precision, a different numeric result.

    Args:
        per_trial: Pure function `(context, trial) -> result`, where `trial` is
            one slice of `trials` along the leading axis.
        trial_structure: Optional pytree of `jax.ShapeDtypeStruct` describing a
            single trial, without the leading trial axis. When given, stacked
            trials are checked against it before compilation. The declaration
            pins pytree structure, per-trial shape, and dtype; it does not pin
            weak typing.

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
