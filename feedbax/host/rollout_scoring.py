"""Score resolved Evaluate trials with the ordinary paired key policy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import equinox as eqx
import jax

from feedbax.contracts.computation_validity import ArgumentSignature, ComputationStart, ProgramIdentity
from feedbax.contracts.rollout import trial_indices
from feedbax.execution.trial_places import TrialPlacePlan
from feedbax.tasks.document import TaskParameters
from feedbax.validity.computation import (
    Operand, UnrecordableArguments, capture_arguments, capture_program, pytree_operands,
)

if TYPE_CHECKING:
    from feedbax.objectives.document import ObjectiveExecutor
    from feedbax.runtime.graph import Graph
    from feedbax.execution.trial_execution import TrialEvaluation, TrialExecution


@dataclass(frozen=True)
class RolloutScorer:
    """One resolved cohort and Objective, independent of its attempt placement."""

    execution: TrialExecution
    objective: ObjectiveExecutor | None
    seed: int
    trials: tuple[str, ...]
    places: TrialPlacePlan = field(default_factory=TrialPlacePlan)

    def __post_init__(self) -> None:
        trial_indices(self.trials, self.execution.task.condition_ids)

    def evaluate(self, graph: Graph, *, capture_state: Callable | None = None) -> TrialEvaluation:
        call = self.kernel(capture_state=capture_state).lower(graph, slot="rollout", seed=self.seed)
        return call.lowered.compile()(*call.operands)

    def kernel(self, *, capture_state: Callable | None) -> RolloutKernel:
        return RolloutKernel(self, capture_state)


@dataclass(frozen=True)
class LoweredRollout:
    lowered: jax.stages.Lowered
    operands: tuple[Operand, ...]
    slot: str
    evidence: tuple[ComputationStart, ProgramIdentity] | UnrecordableArguments


class RolloutKernel:
    """Lower all the numerics for one resolved scorer and raw-state layout."""

    def __init__(self, scorer: RolloutScorer, capture_state: Callable | None):
        self.scorer = scorer
        self.capture_state = capture_state
        self._keys: dict[int, jax.Array] = {}
        self._lowered: dict[ArgumentSignature, tuple[object, jax.stages.Lowered, ProgramIdentity]] = {}

    def lower(self, graph: Graph, *, slot: str, seed: int) -> LoweredRollout:
        if (key := self._keys.get(seed)) is None:
            self._keys[seed] = key = jax.random.PRNGKey(seed)
        values = graph, self.scorer.objective, TaskParameters(), key
        dynamic, static = eqx.partition(values, eqx.is_array)
        leaves, structure = jax.tree.flatten(dynamic)
        operands = tuple(leaves)

        def score(*leaves):
            model, objective, parameters, key = eqx.combine(structure.unflatten(leaves), static)
            return self.scorer.execution.evaluate(
                model, objective, parameters,
                key=key, trial_ids=self.scorer.trials, capture_state=self.capture_state,
                observed_points=self.scorer.places.observations,
            )

        try:
            _, metadata = pytree_operands(values)
            arguments = capture_arguments(operands, static=(self.scorer.trials, metadata))
        except UnrecordableArguments as error:
            return LoweredRollout(jax.jit(score).lower(*operands), operands, slot, error)

        signature = arguments.signature
        cached = self._lowered.get(signature)
        if cached is None or not eqx.tree_equal(cached[0], static):
            lowered = jax.jit(score).lower(*operands)
            program = capture_program(
                lowered, slot=slot, specialization=signature, execution_platform=jax.default_backend(),
            )
            self._lowered[signature] = static, lowered, program
        else:
            _, lowered, cached_program = cached
            program = cached_program.model_copy(update={"slot": slot})
        return LoweredRollout(
            lowered, operands, slot, (ComputationStart(slot=slot, arguments=arguments), program),
        )
