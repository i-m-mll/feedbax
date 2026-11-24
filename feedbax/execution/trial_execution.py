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
from feedbax.compiler.model_reset import ModelResetPlan
from feedbax.tasks.document import TaskParameters, Trial, TrialSource


class TrialEvaluation(NamedTuple):
    objective: ObjectiveResult | None
    trials: Trial
    retained: Mapping[str, jax.Array]
    history: Mapping[str, jax.Array] | None


def sample_evaluation_trials(
    task: TrialSource, parameters: TaskParameters, *, key: jax.Array, trial_ids: tuple[str, ...],
) -> tuple[Trial, jax.Array]:
    """Prepare the same trials for a preview or a rollout of the given Evaluate rows."""
    keys = jnp.stack([derive_key(key, "evaluate-trial", trial) for trial in trial_ids])
    if task.condition_ids:
        trials = task.evaluation(parameters, key, trial_ids, keys)
    else:
        trials = map_trial_samples(lambda trial_key: task.ahead(parameters, key, trial_key), keys, keys)
    return trials, keys


@dataclass(frozen=True)
class TrialExecution:
    """Bind the rollout and observation plan independently of the objective."""

    task: TrialSource
    reset: ModelResetPlan
    input_bindings: tuple[tuple[str, str], ...]
    observations: tuple[GraphTraceRequest, ...]

    def rollout(
        self, graph: Graph, trial: Trial, key: jax.Array, *,
        capture_state=None, input_overrides: Mapping[str, jax.Array] | None = None,
    ):
        inputs = {port: trial.inputs[signal] for port, signal in self.input_bindings}
        if input_overrides:
            if unknown := input_overrides.keys() - set(graph.input_ports):
                raise ValueError(f"Unknown graph input overrides: {', '.join(sorted(unknown))}")
            inputs.update(input_overrides)
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

    def sample_trials(
        self, parameters: TaskParameters, *, run_key: jax.Array, update_key: jax.Array, count: int,
    ) -> tuple[Trial, jax.Array]:
        if count < 1:
            raise ValueError("A training batch must contain at least one trial")

        def sample(key):
            return self.task.ahead(parameters, run_key, key), key

        return sample_training_trials(sample, update_key, count)

    def rollout_batch(
        self, graph: Graph, trials: Trial, trial_keys: jax.Array, *,
        capture_state: Callable | None = None,
        input_overrides: Mapping[str, jax.Array] | None = None,
    ) -> tuple[Mapping[str, jax.Array], Mapping[str, jax.Array] | None]:
        keys = map_trial_samples(lambda key: derive_key(key, "trial-model"), trial_keys, trial_keys)

        def rollout(item):
            trial, key, overrides = item
            return self.rollout(graph, trial, key, capture_state=capture_state, input_overrides=overrides)

        return map_trial_samples(rollout, (trials, keys, input_overrides), keys)

    def score(
        self,
        graph: Graph,
        objective: ObjectiveExecutor | None,
        trials: Trial,
        *,
        run_key: jax.Array,
        trial_keys: jax.Array,
        capture_state: Callable | None = None,
    ) -> TrialEvaluation:
        retained, history = self.rollout_batch(
            graph, trials, trial_keys, capture_state=capture_state,
        )
        return TrialEvaluation(
            None if objective is None else objective.evaluate(retained, trials, run_key=run_key, trial_keys=trial_keys),
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
        trials, keys = self.sample_trials(
            parameters, run_key=run_key, update_key=update_key, count=count,
        )
        scored = self.score(graph, objective, trials, run_key=run_key, trial_keys=keys)
        assert scored.objective is not None
        return scored.objective

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
        self, graph: Graph, objective: ObjectiveExecutor | None, parameters: TaskParameters, *,
        key: jax.Array, trial_ids: tuple[str, ...], capture_state: Callable | None = None,
    ) -> TrialEvaluation:
        trials, keys = sample_evaluation_trials(self.task, parameters, key=key, trial_ids=trial_ids)
        return self.score(
            graph, objective, trials, run_key=key, trial_keys=keys, capture_state=capture_state,
        )
