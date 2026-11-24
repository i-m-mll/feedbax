"""Evaluate objective terms over retained post-step values and a task trial."""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import equinox as eqx
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
from feedbax.contracts.expression_check import references
from feedbax.expression_compile import Compiled, Inputs, compile_expression
from feedbax.contracts.expression_registry import Evaluation, FunctionRegistry
from feedbax.objectives.base import TermTree
from feedbax.runtime.epoch_windows import whole_window
from feedbax.tasks.document import Trial, TrialSource


class ObjectiveInputs(NamedTuple):
    run_key: jax.Array
    trial_keys: jax.Array
    parameters: Mapping[str, Mapping[str, tuple]] | None = None


class TermValue(NamedTuple):
    value: jax.Array
    covered: jax.Array


class TermTrace(NamedTuple):
    values: jax.Array | None
    steps: jax.Array
    reduced: TermValue
    weight: Evaluation | None

    @property
    def covered(self) -> jax.Array:
        return self.reduced.covered


class ObjectiveResult(NamedTuple):
    loss: jax.Array
    terms: Mapping[str, TermValue]


class ObjectiveUse(Enum):
    """Compile strictly, rebind an inherited Objective, or show the terms that resolve."""

    TRAIN = "train"
    INHERITED = "inherited"
    PREVIEW = "preview"


@dataclass(frozen=True)
class _Term:
    id: str
    measure: Compiled
    weight: Compiled
    reduction: Reduction
    epochs: tuple[int, ...]
    references: frozenset[str]
    weight_references: frozenset[str]


@dataclass(frozen=True)
class _UncoveredTerm:
    id: str


class ObjectiveExecutor(eqx.Module):
    """One implementation for per-trial reduction and covered-trial means."""

    label: str
    terms: tuple[_Term | _UncoveredTerm, ...] = ()
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
        return _reduce_trials(values)

    def evaluate_with_traces(self, retained, trials, *, run_key, trial_keys, parameters=None):
        traces = jax.vmap(
            lambda kept, trial, key: self.trial_terms(
                kept, trial, run_key, key, parameters=parameters,
            )
        )(retained, trials, trial_keys)
        return _reduce_trials({term: trace.reduced for term, trace in traces.items()}), traces

    def evaluate_trial(
        self,
        retained,
        trial: Trial,
        run_key,
        trial_key,
        *,
        parameters=None,
    ) -> Mapping[str, TermValue]:
        return {
            term: trace.reduced for term, trace in
            self.trial_terms(retained, trial, run_key, trial_key, parameters=parameters).items()
        }

    def trial_terms(
        self,
        retained,
        trial: Trial,
        run_key,
        trial_key,
        *,
        parameters=None,
        unobserved: Set[str] = frozenset(),
    ) -> Mapping[str, TermTrace]:
        """The weight, weighted measure, covered steps and reduction of each term."""
        n_steps = self.n_steps
        k = jnp.arange(n_steps)
        results = {}
        uncovered = TermTrace(
            None, jnp.zeros(n_steps, dtype=bool),
            TermValue(jnp.asarray(0.0), jnp.asarray(False)), None,
        )
        fixed = {}
        traces = {}
        available = {}
        for name in self.names:
            if name.id in retained:
                source = retained[name.id]
            elif name.id in trial.variables:
                source = trial.variables[name.id]
                if trial.variable_defined is not None and name.id in trial.variable_defined:
                    available[name.id] = trial.variable_defined[name.id]
            elif name.id in trial.inputs:
                source = trial.inputs[name.id]
                available[name.id] = trial.present[name.id]
            else:
                continue
            (traces if name.level == Level.STEP else fixed)[name.id] = source

        supplied = fixed.keys() | traces.keys()
        for term in self.terms:
            if isinstance(term, _UncoveredTerm) or term.references - supplied:
                results[term.id] = uncovered
                continue
            selected = whole_window(trial.epochs, n_steps, term.epochs)
            show_weight = not term.weight_references & unobserved
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
                covered = jnp.asarray(measure.defined & weight.defined)
                results[term.id] = TermTrace(
                    None, jnp.broadcast_to(covered, (n_steps,)),
                    TermValue(measure.value * weight.value, covered),
                    _weight_trace(weight, selected) if show_weight else None,
                )
                continue
            if n_steps == 0:
                results[term.id] = uncovered
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
                weight = None
                if weight_is_step:
                    weight, weight_state = term.weight.evaluate_step(
                        weight_inputs._replace(
                            values=values | {"u": fraction, "t": local, "T": duration},
                            step=sample,
                            defined=mask,
                        ),
                        weight_state,
                    )
                return (measure_state, weight_state), (measure, weight)

            _, (samples, weight) = jax.lax.scan(
                step, (measure_state, weight_state), (k, traces, available, t, u)
            )
            if weight is None:
                weight, _ = term.weight.evaluate_step(weight_inputs, weight_state)
            weights = _weight_trace(weight, selected)
            values = samples.value * weights.value
            steps = samples.defined & weights.defined
            reduced = _reduce(values, steps, term.reduction)
            results[term.id] = TermTrace(values, steps, reduced, weights if show_weight else None)
        return results


def _weight_trace(weight: Evaluation, selected: jax.Array) -> Evaluation:
    return Evaluation(jnp.broadcast_to(weight.value, selected.shape), selected & weight.defined)


def _reduce_trials(values):
    reduced = {}
    for name, (value, covered) in values.items():
        count = jnp.count_nonzero(covered)
        reduced[name] = TermValue(
            jnp.sum(jnp.where(covered, value, 0)) / jnp.maximum(count, 1), count,
        )
    return ObjectiveResult(
        sum((term.value for term in reduced.values()), jnp.asarray(0.0)), reduced,
    )


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
    task: TrialSource,
    sites: dict[tuple[str | int, ...], Site] | None = None,
    use: ObjectiveUse = ObjectiveUse.TRAIN,
    nominal: bool = False,
) -> ObjectiveExecutor | tuple[Problem, ...]:
    """Compile an authored objective or rebind the validated Train terms to Evaluate."""
    period = task.period
    epoch_ids = set(task.epochs)
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
    measure_ids = {name.id for name in names}
    weight_ids = {name.id for name in weight_names}
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
        measure_site = Site(expected=term.measure_type, names=names, finest=Level.STEP)
        if sites is not None:
            sites[(*path, "measure")] = measure_site
            sites[(*path, "weight")] = Site(expected=scalar, names=weight_names, finest=Level.STEP)
        if not term.enabled:
            continue
        if term.measure_type.shape or term.measure_type.base != ValueBase.REAL:
            problems.append(
                Problem(
                    code=ProblemCode.TYPE_MISMATCH,
                    path=(*path, "measure"),
                    reason="This measure is not a real scalar.",
                )
            )
            continue
        measure_refs = references(term.measure)
        weight_refs = references(term.weight)
        absent = set(term.epochs) - epoch_ids
        if use in (ObjectiveUse.INHERITED, ObjectiveUse.PREVIEW) and (
            absent or measure_refs - measure_ids or weight_refs - weight_ids
        ):
            terms.append(_UncoveredTerm(term.id))
            continue
        measure = compile_expression(
            term.measure,
            measure_site,
            functions,
            field_address=(document.id, *path, "measure"),
            period_seconds=period,
            path=(*path, "measure"),
            nominal=nominal,
        )
        if not isinstance(measure, Compiled):
            problems.extend(measure)
            continue
        if absent or term.epochs and measure.checked.root.level != Level.STEP:
            reason = (
                f"Unknown epochs {sorted(absent)}." if absent else "This measure has no time axis."
            )
            problems.append(
                Problem(code=ProblemCode.OUT_OF_SCOPE, path=(*path, "epochs"), reason=reason)
            )
            continue
        limit = Level.STEP if measure.checked.root.level == Level.STEP else Level.TRIAL
        weight_site = Site(expected=scalar, names=weight_names, finest=limit)
        if sites is not None:
            sites[(*path, "weight")] = weight_site
        weight = compile_expression(
            term.weight,
            weight_site,
            functions,
            field_address=(document.id, *path, "weight"),
            period_seconds=period,
            path=(*path, "weight"),
            nominal=nominal,
        )
        if not isinstance(weight, Compiled):
            problems.extend(weight)
            continue
        used = (references(measure.checked.tree) | references(weight.checked.tree)) - {"t", "T", "u"}
        terms.append(
            _Term(
                term.id,
                measure,
                weight,
                term.reduction,
                tuple(index for index, epoch in enumerate(task.epochs) if epoch in term.epochs),
                used,
                weight_refs,
            )
        )
        referenced.update(used)
    if problems and use is not ObjectiveUse.PREVIEW:
        return tuple(problems)
    if problems:
        compiled_ids = {term.id for term in terms}
        missing = dict.fromkeys(term.id for term in document.terms if term.enabled and term.id not in compiled_ids)
        terms.extend(map(_UncoveredTerm, missing))
    return ObjectiveExecutor(
        document.id,
        tuple(terms),
        tuple(name for name in names if name.id in referenced),
        period,
        task.n_steps,
    )
