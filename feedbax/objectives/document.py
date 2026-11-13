"""Evaluate objective terms over retained post-step values and a task trial."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from feedbax.contracts.expression_values import (
    TIME,
    Level,
    Problem,
    ProblemCode,
    ScopedValue,
    Site,
    ValueBase,
    ValueType,
)
from feedbax.contracts.objective_document import (
    ObjectiveDocument as ObjectiveDocument,
    ObjectiveTerm as ObjectiveTerm,
    Reduction as Reduction,
)
from feedbax.expression_check import references
from feedbax.expression_compile import Compiled, Inputs, compile_expression
from feedbax.expression_registry import FunctionRegistry
from feedbax.objectives.loss import AbstractLoss, TermTree
from feedbax.runtime.epoch_windows import whole_window
from feedbax.tasks.document import DocumentTask, Trial


class ObjectiveInputs(NamedTuple):
    run_key: jax.Array
    trial_keys: jax.Array
    parameters: Mapping[str, Mapping[str, tuple]] | None = None


class TermValue(NamedTuple):
    value: jax.Array
    covered: jax.Array


class ObjectiveResult(NamedTuple):
    loss: jax.Array
    terms: Mapping[str, TermValue]


@dataclass(frozen=True)
class _Term:
    id: str
    measure: Compiled
    weight: Compiled
    reduction: Reduction
    epochs: tuple[int, ...]
    references: frozenset[str]


class ObjectiveExecutor(AbstractLoss):
    """One implementation for per-trial reduction and covered-trial means."""

    label: str
    terms: tuple[_Term, ...] = ()
    names: tuple[ScopedValue, ...] = ()
    period: float = 1.0
    n_steps: int = 0

    def __call__(self, states, trial_specs: Trial, model: ObjectiveInputs):
        """Expose the same batch values as a loss tree."""
        return TermTree.branch(
            self.label,
            {
                key: TermTree.leaf(key, term.value)
                for key, term in self.evaluate(
                    states,
                    trial_specs,
                    run_key=model.run_key,
                    trial_keys=model.trial_keys,
                    parameters=model.parameters,
                ).terms.items()
            },
        )

    def evaluate(self, retained, trials, *, run_key, trial_keys, parameters=None):
        values = jax.vmap(
            lambda kept, trial, key: self.evaluate_trial(
                kept,
                trial,
                run_key,
                key,
                parameters=parameters,
            )
        )(retained, trials, trial_keys)
        reduced = {}
        for name, (value, covered) in values.items():
            count = jnp.count_nonzero(covered)
            reduced[name] = TermValue(
                jnp.sum(jnp.where(covered, value, 0)) / jnp.maximum(count, 1), count
            )
        return ObjectiveResult(
            sum((term.value for term in reduced.values()), jnp.asarray(0.0)), reduced
        )

    def evaluate_trial(
        self,
        retained,
        trial: Trial,
        run_key,
        trial_key,
        *,
        parameters=None,
    ) -> Mapping[str, TermValue]:
        """Evaluate every measure over the full trial before selecting its window."""
        n_steps = self.n_steps
        k = jnp.arange(n_steps)
        results = {}
        fixed = {}
        traces = {}
        available = {}
        for name in self.names:
            if name.id in retained:
                source = retained[name.id]
            elif name.name in trial.variables:
                source = trial.variables[name.name]
                if trial.variable_defined is not None and name.name in trial.variable_defined:
                    available[name.id] = trial.variable_defined[name.name]
            elif name.name in trial.inputs:
                source = trial.inputs[name.name]
                available[name.id] = trial.present[name.name]
            else:
                continue
            (traces if name.level == Level.STEP else fixed)[name.id] = source

        supplied = fixed.keys() | traces.keys()
        for term in self.terms:
            if term.references - supplied:
                results[term.id] = TermValue(jnp.asarray(0.0), jnp.asarray(False))
                continue
            selected = whole_window(trial.epochs, n_steps, term.epochs)
            weight_is_step = term.weight.checked.root.level == Level.STEP
            measure_is_step = term.measure.checked.root.level == Level.STEP
            authored = {} if parameters is None else parameters.get(term.id, {})
            measure_args = authored.get("measure", term.measure.parameters)
            weight_args = authored.get("weight", term.weight.parameters)
            measure_inputs = Inputs(measure_args, fixed, run_key, trial_key)
            weight_inputs = Inputs(weight_args, fixed, run_key, trial_key)
            measure_state = term.measure.with_trial(
                measure_inputs, term.measure.evaluate_run(measure_inputs)
            )
            weight_state = term.weight.with_trial(
                weight_inputs, term.weight.evaluate_run(weight_inputs)
            )
            if not measure_is_step:
                measure, _ = term.measure.evaluate_step(measure_inputs, measure_state)
                weight, _ = term.weight.evaluate_step(weight_inputs, weight_state)
                results[term.id] = TermValue(
                    measure.value * weight.value, measure.defined & weight.defined
                )
                continue
            if n_steps == 0:
                results[term.id] = TermValue(jnp.asarray(0.0), jnp.asarray(False))
                continue
            start = jnp.min(jnp.where(selected, k, n_steps))
            last = jnp.max(jnp.where(selected, k, -1))
            count = jnp.count_nonzero(selected)
            rank = jnp.cumsum(selected)
            u = (rank - 1) / jnp.maximum(count, 1)
            t = (k - start) * self.period
            duration = (last - start + 1) * self.period

            def step(states, data):
                measure_state, weight_state = states
                sample, environment, mask, local, fraction = data
                values = fixed | environment
                measure, measure_state = term.measure.evaluate_step(
                    measure_inputs._replace(values=values, step=sample, defined=mask), measure_state
                )
                if weight_is_step:
                    weight, weight_state = term.weight.evaluate_step(
                        weight_inputs._replace(
                            values=values | {"u": fraction, "t": local, "T": duration},
                            step=sample,
                            defined=mask,
                        ),
                        weight_state,
                    )
                    measure = measure._replace(
                        value=measure.value * weight.value,
                        defined=measure.defined & weight.defined,
                    )
                return (measure_state, weight_state), measure

            _, samples = jax.lax.scan(
                step, (measure_state, weight_state), (k, traces, available, t, u)
            )
            reduced = _reduce(samples.value, samples.defined & selected, term.reduction)
            if not weight_is_step:
                weight, _ = term.weight.evaluate_step(weight_inputs, weight_state)
                reduced = TermValue(reduced.value * weight.value, reduced.covered & weight.defined)
            results[term.id] = reduced
        return results


def _reduce(values, mask, reduction):
    n = jnp.count_nonzero(mask)
    match reduction:
        case Reduction.SUM:
            value = jnp.sum(jnp.where(mask, values, 0))
        case Reduction.MEAN:
            value = jnp.sum(jnp.where(mask, values, 0)) / jnp.maximum(n, 1)
        case Reduction.LAST:
            last = jnp.max(jnp.where(mask, jnp.arange(values.size), -1))
            value = values[jnp.maximum(last, 0)]
        case Reduction.MAX:
            value = jnp.max(jnp.where(mask, values, -jnp.inf))
    return TermValue(jnp.where(n > 0, value, 0), n > 0)


def compile_objective(
    document: ObjectiveDocument,
    names: tuple[ScopedValue, ...],
    functions: FunctionRegistry,
    *,
    task: DocumentTask,
) -> ObjectiveExecutor | tuple[Problem, ...]:
    """Compile objective expressions and whole-epoch windows against a paired task."""
    period = task.period
    indexes = {epoch: index for index, epoch in enumerate(task.epochs)}
    problems = []
    terms = []
    seen = set()
    scalar = ValueType(base=ValueBase.REAL)
    time = ValueType(base=ValueBase.REAL, dimensions=TIME)
    weight_names = (
        *names,
        *(
            ScopedValue(id=name, name=name, type=kind, level=Level.STEP)
            for name, kind in (("t", time), ("T", time), ("u", scalar))
        ),
    )
    referenced = set()
    for term in document.terms:
        path = ("terms", term.id)
        if term.id in seen:
            problems.append(
                Problem(
                    code=ProblemCode.DUPLICATE_NAME,
                    path=path,
                    reason=f"Duplicate term {term.id!r}.",
                )
            )
            continue
        seen.add(term.id)
        if not term.enabled:
            continue
        if term.measure_type.shape or term.measure_type.base != ValueBase.REAL:
            problems.append(
                Problem(
                    code=ProblemCode.TYPE_MISMATCH,
                    path=(*path, "measure"),
                    reason="An objective measure must be a real scalar.",
                )
            )
            continue
        measure = compile_expression(
            term.measure,
            Site(expected=term.measure_type, names=names, finest=Level.STEP),
            functions,
            field_address=(document.id, *path, "measure"),
            period_seconds=period,
            path=(*path, "measure"),
        )
        if not isinstance(measure, Compiled):
            problems.extend(measure)
            continue
        absent = set(term.epochs) - indexes.keys()
        if absent or term.epochs and measure.checked.root.level != Level.STEP:
            reason = (
                f"Unknown epochs {sorted(absent)}." if absent else "This measure has no time axis."
            )
            problems.append(
                Problem(code=ProblemCode.OUT_OF_SCOPE, path=(*path, "epochs"), reason=reason)
            )
            continue
        limit = Level.STEP if measure.checked.root.level == Level.STEP else Level.TRIAL
        weight = compile_expression(
            term.weight,
            Site(expected=scalar, names=weight_names, finest=limit),
            functions,
            field_address=(document.id, *path, "weight"),
            period_seconds=period,
            path=(*path, "weight"),
        )
        if not isinstance(weight, Compiled):
            problems.extend(weight)
            continue
        used = (references(term.measure) | references(term.weight)) - {"t", "T", "u"}
        terms.append(
            _Term(
                term.id,
                measure,
                weight,
                term.reduction,
                tuple(indexes[epoch] for epoch in term.epochs),
                used,
            )
        )
        referenced.update(used)
    if problems:
        return tuple(problems)
    return ObjectiveExecutor(
        document.id,
        tuple(terms),
        tuple(name for name in names if name.id in referenced),
        period,
        task.n_steps,
    )
