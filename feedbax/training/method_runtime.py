"""Compile and commit blocks of one-replica method updates."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from feedbax.runtime._graph_key_schedule import derive_key
from feedbax.runtime.graph import Graph
from feedbax.runtime.parameter_constraints import apply_parameter_constraints
from feedbax.tasks.document import TaskParameters
from feedbax.training.methods import TrainingState

if TYPE_CHECKING:
    from jaxtyping import PyTree
    from pydantic import BaseModel

    from feedbax.contracts.training import OptimizerSpec
    from feedbax.objectives.document import ObjectiveExecutor, ObjectiveResult
    from feedbax.runtime.trial_execution import TrialExecution
    from feedbax.tasks.document import Trial
    from feedbax.training.methods import RegisteredTrainingMethod, TrainingMetrics


def _replica_key(seed: int, replica: str, stream: str) -> jax.Array:
    root = derive_key(jr.key(seed, impl="threefry2x32"), "studio_training", stream)
    return root if replica == "0" else derive_key(root, "replica", replica)


def stack_replicas[T](values: Sequence[T]) -> T:
    """Stack array leaves on a leading replica axis and retain one static tree."""
    if not values:
        raise ValueError("At least one replica is required")
    arrays, static = eqx.partition(values[0], eqx.is_array)
    leaves, structure = jax.tree.flatten(arrays)
    inventory = [_array_signature(leaf) for leaf in leaves]
    dynamic = [arrays]
    for value in values[1:]:
        arrays, replica_static = eqx.partition(value, eqx.is_array)
        leaves, replica_structure = jax.tree.flatten(arrays)
        if not eqx.tree_equal(static, replica_static) or replica_structure != structure:
            raise ValueError("Replica PyTrees must have identical structure and static metadata")
        if [_array_signature(leaf) for leaf in leaves] != inventory:
            raise ValueError("Replica arrays must have identical shapes and dtypes")
        dynamic.append(arrays)
    return eqx.combine(jax.tree.map(lambda *items: jnp.stack(items), *dynamic), static)


def _array_signature(leaf):
    return leaf.shape, leaf.dtype, getattr(leaf, "weak_type", False)


def replica_value[T](tree: T, index: int) -> T:
    """Read one position from a tree with a leading replica axis."""
    return jax.tree.map(lambda leaf: leaf[index] if eqx.is_array(leaf) else leaf, tree)


def _check_replicas(replica_ids: tuple[str, ...]) -> None:
    if not replica_ids or not all(replica_ids) or len(set(replica_ids)) != len(replica_ids):
        raise ValueError("Replica identifiers must be nonempty and unique")


def _check_state(state: TrainingState) -> None:
    if not isinstance(state, TrainingState) or not isinstance(state.model, Graph):
        raise TypeError("A training method must return TrainingState with a Graph model")
    if not all(eqx.is_array(leaf) for leaf in jax.tree.leaves(state.extra)):
        raise TypeError("TrainingState.extra must be a PyTree of arrays")


def _all_finite(arrays: PyTree[jax.Array]) -> jax.Array:
    finite = [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(arrays)
              if jnp.issubdtype(leaf.dtype, jnp.inexact)]
    return jnp.all(jnp.stack(finite)) if finite else jnp.array(True)


def initialize_method(
    method: RegisteredTrainingMethod, settings: BaseModel, *, models: Sequence[Graph],
    seed: int, replica_ids: tuple[str, ...],
) -> TrainingState:
    """Initialize independent replica states against their respective models."""
    _check_replicas(replica_ids)
    if len(models) != len(replica_ids):
        raise ValueError("The model sequence and replica identifiers must correspond")
    states = []
    for model, replica in zip(models, replica_ids, strict=True):
        state = method.init(settings, model, _replica_key(seed, replica, "method-init"))
        _check_state(state)
        states.append(state)
    stacked = stack_replicas(states)
    if not bool(_all_finite(eqx.filter(stacked, eqx.is_array))):
        raise ValueError("Training method initialized a nonfinite state")
    return stacked


class OptimizerStep[AuxT](NamedTuple):
    model: Graph
    state: optax.OptState
    loss: jax.Array
    aux: AuxT
    grad_norm: jax.Array


@dataclass(frozen=True)
class TrainingContext[S: BaseModel]:
    """Trial, objective and optimizer primitives for one replica and update."""

    settings: S
    step: jax.Array
    start_step: int
    key: jax.Array
    run_key: jax.Array
    _execution: TrialExecution
    _objective: ObjectiveExecutor
    _parameters: TaskParameters
    _batch_size: int

    def sample_trials(
        self, count: int | None = None, *, key: jax.Array | None = None,
    ) -> tuple[Trial, jax.Array]:
        """Sample a batch whose trials and keys can be reused in several rollouts."""
        return self._execution.sample_trials(
            self._parameters, run_key=self.run_key,
            update_key=derive_key(self.key, "task-trials") if key is None else key,
            count=self._batch_size if count is None else count,
        )

    def rollout(
        self, model: Graph, trials: Trial, trial_keys: jax.Array, *,
        input_overrides: Mapping[str, jax.Array] | None = None,
    ) -> Mapping[str, jax.Array]:
        """Return retained rollouts, replacing any supplied per-trial graph inputs."""
        outputs, _ = self._execution.rollout_batch(
            model, trials, trial_keys, input_overrides=input_overrides,
        )
        return outputs

    def objective(
        self, outputs: Mapping[str, jax.Array], trials: Trial, trial_keys: jax.Array,
    ) -> ObjectiveResult:
        return self._objective.evaluate(outputs, trials, run_key=self.run_key, trial_keys=trial_keys)

    @staticmethod
    def optimizer(
        spec: OptimizerSpec, *, gradient_clip: float | None = None,
    ) -> optax.GradientTransformationExtraArgs:
        from feedbax.training.optimizers import build_step_optimizer

        return build_step_optimizer(spec, gradient_clip=gradient_clip)

    def optimizer_step[AuxT](
        self, optimizer: optax.GradientTransformation, model: Graph, state: optax.OptState,
        loss_fn: Callable[[Graph], tuple[jax.Array, AuxT]],
    ) -> OptimizerStep[AuxT]:
        """Differentiate the loss on learned graph parameters, then project constraints."""
        from feedbax.compiler.trainability import model_trainability

        _, trainable_filter = model_trainability(model)
        trainable, frozen = eqx.partition(model, trainable_filter)

        def loss_and_aux(values):
            return loss_fn(eqx.combine(values, frozen))

        (loss, aux), grads = eqx.filter_value_and_grad(loss_and_aux, has_aux=True)(trainable)
        update = optax.with_extra_args_support(optimizer).update
        updates, state = update(grads, state, trainable, schedule_step=self.step - self.start_step)
        model = apply_parameter_constraints(eqx.combine(eqx.apply_updates(trainable, updates), frozen))
        return OptimizerStep(model, state, loss, aux, optax.global_norm(grads))


class BlockResult(NamedTuple):
    state: TrainingState
    completed: int
    metrics: TrainingMetrics
    rejected: bool
    sampled_model: Graph | None = None


@dataclass(frozen=True)
class LoweredBlock:
    traced: jax.stages.Traced
    execution: jax.stages.Lowered
    platform: str


def _metric_array(value):
    array = jnp.asarray(value)
    if not any(jnp.issubdtype(array.dtype, dtype) for dtype in (jnp.floating, jnp.integer, jnp.bool_)):
        raise TypeError("Training metrics must be real numerical or boolean arrays")
    return jnp.asarray(array, dtype=array.dtype)


class TrainingLoop:
    """Own the logical update count, replica keys and atomic state adoption."""

    def __init__(
        self, method: RegisteredTrainingMethod, settings: BaseModel, *, execution: TrialExecution,
        objective: ObjectiveExecutor, batch_size: int, seed: int, replica_ids: tuple[str, ...],
        start_step: int = 0, parameters: TaskParameters = TaskParameters(), sample_at: int = 0,
    ) -> None:
        _check_replicas(replica_ids)
        if batch_size < 1 or not 0 <= start_step < 2**32:
            raise ValueError("Invalid training batch size or schedule origin")
        self._replica_count = len(replica_ids)
        self._start_step = start_step
        step_keys = jnp.stack([_replica_key(seed, replica, "update") for replica in replica_ids])
        run_keys = jnp.stack([_replica_key(seed, replica, "task-run") for replica in replica_ids])

        def block(static, arrays, completed, update_keys, task_keys, sample_update, active_count, count):
            @jax.vmap
            def single_update(replica_arrays, key_root, run_key, step):
                key = derive_key(key_root, "step", index=step)
                ctx = TrainingContext(settings, step, start_step, key, run_key,
                                      execution, objective, parameters, batch_size)
                candidate, metrics = method.update(eqx.combine(replica_arrays, static), ctx)
                _check_state(candidate)
                dynamic, new_static = eqx.partition(candidate, eqx.is_array)
                if not eqx.tree_equal(static, new_static):
                    raise ValueError("Training update changed the fixed structure or metadata of method state")
                return dynamic, jax.tree.map(_metric_array, dict(metrics))

            @jax.jit
            def propose(current, step):
                return single_update(current, update_keys, task_keys, jnp.broadcast_to(step, (len(replica_ids),)))

            _, metrics_shape = jax.eval_shape(propose, arrays, completed)
            # Rejected and unattempted metric rows are sliced away at the host boundary.
            no_metrics = jax.tree.map(lambda shape: jnp.zeros(shape.shape, shape.dtype), metrics_shape)

            def advance(carry, offset):
                current, accepted, active, witness = carry

                def attempt(value):
                    candidate, metrics = propose(value, completed + offset)
                    finite = _all_finite(candidate)
                    next_value = jax.lax.cond(finite, lambda: candidate, lambda: value)
                    capture = finite & (completed + offset + jnp.uint32(1) == sample_update)
                    next_witness = jax.lax.cond(capture, lambda: candidate.model, lambda: witness)
                    return (next_value, accepted + finite, finite, next_witness), metrics

                return jax.lax.cond(
                    active & (offset < active_count), attempt,
                    lambda value: ((value, accepted, active, witness), no_metrics), current,
                )
            carry, metrics = jax.lax.scan(
                advance, (arrays, jnp.int32(0), jnp.array(True), arrays.model), jnp.arange(count, dtype=jnp.uint32),
            )
            arrays, accepted, _, witness = carry
            return arrays, accepted, metrics, witness

        self._block = block
        self._keys = step_keys, run_keys
        self._sample_at = sample_at
        self._static: TrainingState | None = None
        self._kernel: jax.stages.Wrapped | None = None
        self._lowered: dict[tuple, LoweredBlock] = {}
        self._compiled: dict[tuple, jax.stages.Compiled] = {}
        self.observe_lowering: Callable[[TrainingState, tuple[jax.Array, ...], int, LoweredBlock], None] | None = None

    def _operands(self, state: TrainingState, completed: int, count: int, limit: int):
        if not self._start_step <= completed < completed + count <= 2**32:
            raise ValueError("The block must contain updates within the training clock")
        if not 0 < limit <= count:
            raise ValueError("Train replay limit must be inside the actual block")
        _check_state(state)
        arrays, static = eqx.partition(state, eqx.is_array)
        leaves = jax.tree.leaves(arrays)
        if any(leaf.ndim < 1 or leaf.shape[0] != self._replica_count for leaf in leaves):
            raise ValueError("Training state arrays must share the declared leading replica axis")
        if self._static is None:
            self._static = static
            self._kernel = jax.jit(partial(self._block, static), static_argnums=6)
        elif not eqx.tree_equal(self._static, static):
            raise ValueError("Training state changed its fixed structure or metadata between blocks")
        controls = self.block_arguments(completed, limit)
        signature = count, tuple(
            (_array_signature(leaf), leaf.sharding, leaf.committed) for leaf in (*leaves, *controls)
        )
        return signature, (arrays, *controls)

    def _lower(self, signature: tuple, operands, count: int) -> LoweredBlock:
        from jax.export import default_export_platform

        if signature not in self._lowered:
            assert self._kernel is not None
            traced = self._kernel.trace(*operands, count)
            self._lowered[signature] = LoweredBlock(traced, traced.lower(), default_export_platform())
        return self._lowered[signature]

    def lower_block(self, state: TrainingState, *, completed: int, count: int) -> LoweredBlock:
        """Lower the same block and operand types that execution would compile."""
        signature, operands = self._operands(state, completed, count, count)
        return self._lower(signature, operands, count)

    def block_arguments(self, completed: int, limit: int) -> tuple[jax.Array, ...]:
        return jnp.uint32(completed), *self._keys, jnp.uint32(self._sample_at), jnp.uint32(limit)

    def run_block(self, state: TrainingState, *, completed: int, count: int, limit: int | None = None) -> BlockResult:
        """Run a bounded compiled block and return the contiguous committed prefix."""
        limit = count if limit is None else limit
        signature, operands = self._operands(state, completed, count, limit)
        if signature not in self._compiled:
            lowered = self._lower(signature, operands, count)
            if self.observe_lowering is not None:
                self.observe_lowering(state, operands[1:], count, lowered)
            self._compiled[signature] = lowered.execution.compile()
        updated, accepted, metrics, witness = self._compiled[signature](*operands)
        accepted, metrics = jax.device_get((accepted, metrics))
        n = int(accepted)
        updated = eqx.combine(updated, self._static)
        sampled = eqx.combine(witness, self._static.model) if completed < self._sample_at <= completed + n else None
        return BlockResult(updated, completed + n, jax.tree.map(lambda values: values[:n], metrics), n < limit, sampled)
