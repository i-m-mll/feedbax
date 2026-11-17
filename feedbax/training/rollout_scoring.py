"""Score resolved Evaluate trials with the ordinary paired key policy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import equinox as eqx
import jax

from feedbax.contracts.rollout import trial_indices
from feedbax.tasks.document import TaskParameters

if TYPE_CHECKING:
    from feedbax.objectives.document import ObjectiveExecutor
    from feedbax.runtime.graph import Graph
    from feedbax.runtime.trial_execution import TrialEvaluation, TrialExecution


@dataclass(frozen=True)
class RolloutScorer:
    """One resolved cohort and Objective, independent of its attempt placement."""

    execution: TrialExecution
    objective: ObjectiveExecutor
    seed: int
    trials: tuple[str, ...]

    def __post_init__(self) -> None:
        trial_indices(self.trials, self.execution.task.condition_ids)

    def evaluate(self, graph: Graph, *, capture_state: Callable | None = None) -> TrialEvaluation:
        return self.execution.evaluate(
            graph, self.objective, TaskParameters(),
            key=jax.random.PRNGKey(self.seed), trial_ids=self.trials, capture_state=capture_state,
        )

    def lower(self, graph: Graph, *, capture_state: Callable | None = None):
        def score(model):
            return self.evaluate(model, capture_state=capture_state)

        return eqx.filter_jit(score).lower(graph)
