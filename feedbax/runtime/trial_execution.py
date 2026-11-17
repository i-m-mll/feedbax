"""Run a paired trial source and score retained graph outputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from feedbax.objectives.document import ObjectiveExecutor, ObjectiveResult
from feedbax.runtime._graph_key_schedule import (
    derive_key,
    map_trial_samples,
    sample_training_trials,
    validation_trial_keys,
)
from feedbax.runtime.graph import Graph, GraphTraceRequest
from feedbax.runtime.model_reset import ModelResetPlan
from feedbax.tasks.document import TaskParameters, Trial, TrialSource


class TrialEvaluation(NamedTuple):
    objective: ObjectiveResult
    trials: Trial
    retained: Mapping[str, jax.Array]
    history: Mapping[str, jax.Array] | None


@dataclass(frozen=True)
class TrialExecution:
    """Bind the rollout and observation plan independently of the objective."""

    task: TrialSource
    reset: ModelResetPlan
    input_bindings: tuple[tuple[str, str], ...]
    observations: tuple[GraphTraceRequest, ...]

    def rollout(self, graph: Graph, trial: Trial, key: jax.Array, *, capture_state=None):
        inputs = {port: trial.inputs[signal] for port, signal in self.input_bindings}
        reset = self.reset.prepare(graph, trial.inits, key=key, inputs=inputs)
        _, _, retained, history = graph.rollout_with_trace_and_history(
            inputs,
            reset.state,
            key=key,
            n_steps=self.task.n_steps,
            cycle_init=reset.cycle_init,
            trace=self.observations,
            capture_state=capture_state,
        )
        return retained, history

    def rollouts(self, graph: Graph, trials: Trial, keys: jax.Array) -> dict[str, jax.Array]:
        return jax.vmap(lambda trial, key: self.rollout(graph, trial, key)[0])(trials, keys)

    def score(
        self,
        graph: Graph,
        objective: ObjectiveExecutor,
        trials: Trial,
        *,
        run_key: jax.Array,
        trial_keys: jax.Array,
        capture_state: Callable | None = None,
    ) -> TrialEvaluation:
        keys = map_trial_samples(lambda key: derive_key(key, "trial-model"), trial_keys, trial_keys)
        retained, history = map_trial_samples(
            lambda item: self.rollout(graph, item[0], item[1], capture_state=capture_state), (trials, keys), keys,
        )
        return TrialEvaluation(
            objective.evaluate(retained, trials, run_key=run_key, trial_keys=trial_keys),
            trials,
            retained,
            history,
        )

    def training_batch(
        self,
        graph: Graph,
        objective: ObjectiveExecutor,
        parameters: TaskParameters,
        *,
        run_key: jax.Array,
        update_key: jax.Array,
        count: int,
    ) -> ObjectiveResult:
        def sample(key):
            return self.task.ahead(parameters, run_key, key), key

        trials, keys = sample_training_trials(sample, update_key, count)
        return self.score(graph, objective, trials, run_key=run_key, trial_keys=keys).objective

    def validation(
        self,
        graph: Graph,
        objective: ObjectiveExecutor,
        parameters: TaskParameters,
        *,
        key: jax.Array,
        capture_state: Callable | None = None,
    ) -> TrialEvaluation:
        trials = self.task.validation(parameters, key)
        keys = validation_trial_keys(key, trials.epochs.starts.shape[0])
        return self.score(
            graph, objective, trials, run_key=key, trial_keys=keys, capture_state=capture_state
        )

    def evaluate(
        self, graph: Graph, objective: ObjectiveExecutor, parameters: TaskParameters, *,
        key: jax.Array, trial_ids: tuple[str, ...], capture_state: Callable | None = None,
    ) -> TrialEvaluation:
        keys = jnp.stack([derive_key(key, "evaluate-trial", trial) for trial in trial_ids])
        if self.task.condition_ids:
            trials = self.task.evaluation(parameters, key, trial_ids, keys)
        else:
            trials = map_trial_samples(lambda trial_key: self.task.ahead(parameters, key, trial_key), keys, keys)
        return self.score(
            graph, objective, trials, run_key=key, trial_keys=keys, capture_state=capture_state,
        )
