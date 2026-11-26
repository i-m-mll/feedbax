"""Run a paired trial source and score retained graph outputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from feedbax.objectives.document import ObjectiveExecutor, ObjectiveResult, TermTrace
from feedbax.runtime._graph_key_schedule import (
    derive_key,
    map_trial_samples,
    sample_training_trials,
    validation_trial_keys,
)
from feedbax.runtime._rollout import scan_rollout, select_step_inputs
from feedbax.runtime.graph import Graph, GraphTraceRequest, signal_leaf_value
from feedbax.compiler.model_reset import ModelResetPlan
from feedbax.contracts.expression_registry import Evaluation
from feedbax.contracts.expression_values import NodePath, ValueBase, ValueType
from feedbax.contracts.model_values import SignalLeafPath
from feedbax.tasks.document import DocumentTask, PendingTrial, TaskParameters, Trial, TrialSource


class TrialEvaluation(NamedTuple):
    objective: ObjectiveResult | None
    trials: Trial
    retained: Mapping[str, jax.Array]
    history: Mapping[str, jax.Array] | None
    term_traces: Mapping[str, TermTrace] | None


def sample_trial(task: TrialSource, parameters: TaskParameters, run_key, trial_key) -> Trial | PendingTrial:
    if isinstance(task, DocumentTask):
        return task.sample(parameters, run_key, trial_key)
    return task.ahead(parameters, run_key, trial_key)


def sample_evaluation_trials(
    task: TrialSource, parameters: TaskParameters, *, key: jax.Array, trial_ids: tuple[str, ...],
) -> tuple[Trial | PendingTrial, jax.Array]:
    """Prepare the same trials for a preview or a rollout of the given Evaluate rows."""
    keys = jnp.stack([derive_key(key, "evaluate-trial", trial) for trial in trial_ids])
    if task.condition_ids:
        trials = task.evaluation(parameters, key, trial_ids, keys)
    else:
        trials = map_trial_samples(lambda trial_key: sample_trial(task, parameters, key, trial_key), keys, keys)
    return trials, keys


_DTYPES = {ValueBase.BOOLEAN: bool, ValueBase.INTEGER: int, ValueBase.REAL: float}


@dataclass(frozen=True)
class TrialModelOutput:
    reference: str
    port: str
    leaf: SignalLeafPath
    type: ValueType

    def read(self, outputs) -> Evaluation:
        dtype = _DTYPES[self.type.base]
        if self.port not in outputs:
            return Evaluation(jnp.zeros(self.type.shape, dtype=dtype), jnp.asarray(False))
        return Evaluation(
            jnp.asarray(signal_leaf_value(outputs[self.port], self.leaf), dtype=dtype), jnp.asarray(True),
        )


@dataclass(frozen=True)
class TrialExecution:
    """Bind the rollout and observation plan independently of the objective."""

    task: TrialSource
    reset: ModelResetPlan
    input_bindings: tuple[tuple[str, str], ...]
    observations: tuple[GraphTraceRequest, ...]
    observation_origins: tuple[tuple[str, str], ...]
    task_reads: tuple[TrialModelOutput, ...] = ()

    def _drive(self, graph, pending, key, capture_state, input_overrides):
        task = self.task
        assert isinstance(task, DocumentTask)
        parameters, prepared, task_state = pending

        def reads(outputs):
            return {read.reference: read.read(outputs) for read in self.task_reads}

        def task_step(state, k, outputs, overrides):
            state, inputs, *traces = task.step(parameters, prepared, state, k, outputs)
            model_inputs = {port: inputs[signal] for port, signal in self.input_bindings}
            return state, model_inputs | overrides, (inputs, *traces)

        def seed_inputs(outputs):
            overrides = jax.tree.map(lambda values: values[0], input_overrides) if self.task.n_steps else {}
            return task_step(task_state, jnp.asarray(0), reads(outputs), overrides)[1]

        reset = self.reset.prepare(
            graph, task.inits(prepared), key=key, inputs=seed_inputs, reset_values=prepared.values,
        )

        def step(overrides, model_state, carry, _key, k):
            cycle, state, previous = carry
            state, model_inputs, samples = task_step(state, k, previous, overrides)
            outputs, model_state, cycle, retained = graph.step_with_trace(
                model_inputs, model_state, cycle, key=key, trace=self.observations, t=k,
            )
            return (retained, samples), model_state, (cycle, state, reads(outputs))

        def finish(stacked, carry):
            _, state, outputs = carry
            retained, samples = stacked
            return retained, task.finish(parameters, prepared, state, samples, outputs)

        def view(model_state, carry):
            return capture_state(model_state, carry[0])

        (retained, trial), _, history = scan_rollout(
            step, reset.state, select_step_inputs(input_overrides, task.n_steps), jnp.arange(task.n_steps), task.n_steps,
            init_step_carry=(reset.cycle_init, task_state, reads(reset.outputs)),
            state_view_fn=view, finish_outputs=finish, checkpoint=graph.checkpoint, save_history=capture_state is not None,
        )
        return {operand: retained[origin] for operand, origin in self.observation_origins}, history, trial

    def _run(
        self, graph, trial, key, *, capture_state=None, input_overrides=None,
    ):
        input_overrides = input_overrides or {}
        if unknown := input_overrides.keys() - set(graph.input_ports):
            raise ValueError(f"Unknown graph input overrides: {', '.join(sorted(unknown))}")
        if isinstance(trial, PendingTrial):
            return self._drive(graph, trial, key, capture_state, input_overrides)
        inputs = {port: trial.inputs[signal] for port, signal in self.input_bindings}
        inputs.update(input_overrides)
        reset = self.reset.prepare(
            graph, trial.inits, key=key, inputs=inputs, input_series=True, reset_values=trial.variables,
        )
        _, _, retained, history = graph.rollout_with_trace_and_history(
            inputs,
            reset.state,
            key=key,
            n_steps=self.task.n_steps,
            cycle_init=reset.cycle_init,
            trace=self.observations,
            capture_state=capture_state,
        )
        return {operand: retained[origin] for operand, origin in self.observation_origins}, history, trial

    def rollout(
        self, graph: Graph, trial: Trial | PendingTrial, key: jax.Array, *,
        capture_state=None, input_overrides: Mapping[str, jax.Array] | None = None,
    ):
        retained, history, _ = self._run(
            graph, trial, key, capture_state=capture_state, input_overrides=input_overrides,
        )
        return retained, history

    def rollouts(self, graph: Graph, trials: Trial, keys: jax.Array) -> dict[str, jax.Array]:
        return jax.vmap(lambda trial, key: self.rollout(graph, trial, key)[0])(trials, keys)

    def sample_trials(
        self, parameters: TaskParameters, *, run_key: jax.Array, update_key: jax.Array, count: int,
    ) -> tuple[Trial | PendingTrial, jax.Array]:
        if count < 1:
            raise ValueError("A training batch must contain at least one trial")

        def sample(key):
            return sample_trial(self.task, parameters, run_key, key), key

        return sample_training_trials(sample, update_key, count)

    def rollout_batch(
        self, graph: Graph, trials: Trial | PendingTrial, trial_keys: jax.Array, *,
        capture_state: Callable | None = None,
        input_overrides: Mapping[str, jax.Array] | None = None,
    ) -> tuple[Mapping[str, jax.Array], Mapping[str, jax.Array] | None]:
        retained, history, _ = self.execute_batch(
            graph, trials, trial_keys, capture_state=capture_state, input_overrides=input_overrides,
        )
        return retained, history

    def execute_batch(
        self, graph: Graph, trials: Trial | PendingTrial, trial_keys: jax.Array, *,
        capture_state: Callable | None = None,
        input_overrides: Mapping[str, jax.Array] | None = None,
    ) -> tuple[Mapping[str, jax.Array], Mapping[str, jax.Array] | None, Trial]:
        keys = map_trial_samples(lambda key: derive_key(key, "trial-model"), trial_keys, trial_keys)

        def rollout(item):
            trial, key, overrides = item
            return self._run(graph, trial, key, capture_state=capture_state, input_overrides=overrides)

        batch = (trials, keys, input_overrides)
        if isinstance(trials, PendingTrial):
            return map_trial_samples(rollout, batch, keys)
        retained, history = map_trial_samples(lambda item: rollout(item)[:2], batch, keys)
        return retained, history, trials

    def score(
        self,
        graph: Graph,
        objective: ObjectiveExecutor | None,
        trials: Trial | PendingTrial,
        *,
        run_key: jax.Array,
        trial_keys: jax.Array,
        capture_state: Callable | None = None,
        observed_points: Mapping[str, tuple[NodePath, ...]] | None = None,
    ) -> TrialEvaluation:
        retained, history, trials = self.execute_batch(
            graph, trials, trial_keys, capture_state=capture_state,
        )
        score, traces = None, None
        if objective is not None:
            if capture_state is None:
                score = objective.evaluate(retained, trials, run_key=run_key, trial_keys=trial_keys)
            else:
                score, traces = objective.evaluate_with_traces(
                    retained, trials, run_key=run_key, trial_keys=trial_keys, observed_points=observed_points,
                )
        return TrialEvaluation(score, trials, retained, history, traces)

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
        epochs = trials.prepared.epochs if isinstance(trials, PendingTrial) else trials.epochs
        ids = self.task.condition_ids if isinstance(self.task, DocumentTask) else ()
        keys = validation_trial_keys(key, epochs.starts.shape[0], ids=ids)
        return self.score(
            graph, objective, trials, run_key=key, trial_keys=keys, capture_state=capture_state
        )

    def evaluate(
        self, graph: Graph, objective: ObjectiveExecutor | None, parameters: TaskParameters, *,
        key: jax.Array, trial_ids: tuple[str, ...], capture_state: Callable | None = None,
        observed_points: Mapping[str, tuple[NodePath, ...]] | None = None,
    ) -> TrialEvaluation:
        trials, keys = sample_evaluation_trials(self.task, parameters, key=key, trial_ids=trial_ids)
        return self.score(
            graph, objective, trials, run_key=key, trial_keys=keys, capture_state=capture_state,
            observed_points=observed_points,
        )
