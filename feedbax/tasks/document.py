"""Objective-free task documents and their ahead-of-rollout runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import groupby
from math import isfinite
from typing import Any, NamedTuple, Protocol

from feedbax.compiler.model_document import ResolvedModelInterface

import jax
import jax.numpy as jnp

from feedbax.contracts.expression_values import (
    RESERVED_NAMES,
    TIME,
    DrawValue,
    Expression,
    Level,
    CallValue,
    LiteralValue,
    NameValue,
    Problem,
    ProblemCode,
    ScopedValue,
    Site,
    UnresolvedValue,
    ValueBase,
    ValueType,
    walk,
)
from feedbax.contracts.task_document import (
    Cell as Cell,
    Epoch as Epoch,
    EpochEnd as EpochEnd,
    EpochSteps,
    Rest,
    SetVariableChoices,
    Until,
    Link as Link,
    Signal as Signal,
    TaskDocument as TaskDocument,
    ValidationTrial as ValidationTrial,
    ValidationValue,
    Variable as Variable,
)
from feedbax.contracts.expression_check import Checked, check, dependency_order, references
from feedbax.contracts.expression_text import resolve_expression
from feedbax.expression_compile import Compiled, Inputs, compile_expression
from feedbax.contracts.expression_registry import FunctionRegistry
from feedbax.contracts.rollout import trial_indices
from feedbax.contracts.trial_bindings import selected_signal_id
from feedbax.contracts.time_units import length_steps, literal_time
from feedbax.runtime._graph_key_schedule import map_trial_samples, validation_trial_keys
from feedbax.runtime.epoch_windows import CompiledWindow, EpochRecord, compile_window, in_epoch
from feedbax.runtime.timing import runtime_time_steps


@dataclass(frozen=True)
class TaskInterface:
    variables: tuple[ScopedValue, ...]
    signals: tuple[ScopedValue, ...]
    links: tuple[tuple[str, str], ...]
    model_outputs: tuple[ScopedValue, ...] = ()
    epoch_conditions: tuple[tuple[str, str], ...] = ()
    set_choices: tuple[tuple[str, SetVariableChoices], ...] = ()


class Trial(NamedTuple):
    inits: Mapping[str, Any]
    inputs: Mapping[str, jax.Array]
    present: Mapping[str, jax.Array]
    variables: Mapping[str, jax.Array]
    epochs: EpochRecord
    variable_defined: Mapping[str, jax.Array] | None = None


class TaskParameters(NamedTuple):
    fields: Mapping[tuple[str, ...], tuple] | None = None
    variables: Mapping[str, jax.Array] | None = None
    epoch_steps: Mapping[str, jax.Array] | None = None


class PendingTrial(NamedTuple):
    """A trial whose Task must receive outputs from the model before it can emit signals."""

    parameters: TaskParameters
    prepared: _Prepared
    state: _TaskState


class TrialSource(Protocol):
    @property
    def interface(self) -> TaskInterface: ...
    @property
    def n_steps(self) -> int: ...
    @property
    def period(self) -> float: ...
    @property
    def epochs(self) -> tuple[str, ...]: ...
    @property
    def step_counts(self) -> tuple[EpochSteps, ...]: ...
    @property
    def reads_model(self) -> bool: ...
    @property
    def condition_ids(self) -> tuple[str, ...]: ...
    def init(self, parameters: TaskParameters, run_key: jax.Array, trial_key: jax.Array): ...
    def step(self, parameters: TaskParameters, variables, state, k: jax.Array, model_outputs): ...
    def ahead(
        self, parameters: TaskParameters, run_key: jax.Array, trial_key: jax.Array
    ) -> Trial: ...
    def validation(self, parameters: TaskParameters, key: jax.Array) -> Trial | PendingTrial: ...
    def evaluation(
        self, parameters: TaskParameters, key: jax.Array, trial_ids: tuple[str, ...], trial_keys: jax.Array,
    ) -> Trial | PendingTrial: ...


class _Prepared(NamedTuple):
    values: Mapping[str, Any]
    epochs: EpochRecord
    absent: Mapping[str, Any]
    run_key: jax.Array
    trial_key: jax.Array
    lengths: jax.Array | None = None
    windows: Mapping[str, jax.Array] | None = None


class _Timeline(NamedTuple):
    index: jax.Array
    record: EpochRecord
    conditions: tuple[Any, ...]


class _TaskState(NamedTuple):
    variables: Mapping[str, Any]
    cells: tuple
    timeline: _Timeline | None


@dataclass(frozen=True)
class _Field:
    address: tuple[str, ...]
    code: Compiled
    list_count: int | None = None
    length_in_steps: bool = False

    def inputs(self, parameters, prepared, values, k=0, *, epoch_entry=False, repeat=None):
        args = (
            parameters.fields.get(self.address, self.code.parameters)
            if parameters.fields
            else self.code.parameters
        )
        return Inputs(
            args, values, prepared.run_key, prepared.trial_key, repeat=repeat, step=k, epoch_entry=epoch_entry
        )

    def prepare(self, parameters, prepared, values, *, repeat=None):
        def prepare(index):
            inputs = self.inputs(parameters, prepared, values, repeat=index)
            return self.code.with_trial(inputs, self.code.evaluate_run(inputs))

        if self.list_count is None:
            return prepare(repeat)
        return jax.vmap(prepare)(jnp.arange(self.list_count))

    def evaluate(self, parameters, prepared, values, *, repeat=None):
        def evaluate(index):
            inputs = self.inputs(parameters, prepared, values, repeat=index)
            state = self.code.with_trial(inputs, self.code.evaluate_run(inputs))
            return self.code.evaluate_step(inputs, state)[0].value

        if self.list_count is None:
            return evaluate(repeat)
        return jax.vmap(evaluate)(jnp.arange(self.list_count))

    def step(self, inputs, state):
        if self.list_count is None:
            return self.code.evaluate_step(inputs, state)
        return jax.vmap(
            lambda index, item_state: self.code.evaluate_step(inputs._replace(repeat=index), item_state)
        )(jnp.arange(self.list_count), state)


@dataclass(frozen=True)
class _Cell:
    first: int
    last: int
    field: _Field
    repeat: int | None


@dataclass(frozen=True)
class _Signal:
    id: str
    absent: _Field
    cells: tuple[_Cell, ...]
    window: CompiledWindow | None = None


@dataclass(frozen=True)
class DocumentTask:
    interface: TaskInterface
    n_steps: int
    period: float
    epochs: tuple[str, ...]
    variable_fields: tuple[tuple[ScopedValue, _Field], ...]
    length_fields: tuple[_Field | None, ...]
    bounds: tuple[int, ...]
    signals: tuple[_Signal, ...]
    validation_rows: tuple[TaskParameters, ...]
    condition_ids: tuple[str, ...]
    input_bindings: tuple[tuple[str, str], ...]
    step_counts: tuple[EpochSteps, ...]
    epoch_repeats: tuple[int | None, ...]
    conditions: tuple[_Field | None, ...]
    reads_model: bool = False

    def init(self, parameters, run_key, trial_key):
        values = {}
        prepared = _Prepared(values, None, None, run_key, trial_key)
        for variable, field in self.variable_fields:
            if variable.level < Level.STEP:
                given = parameters.variables
                values[variable.id] = (
                    given[variable.id]
                    if given and variable.id in given
                    else field.evaluate(parameters, prepared, values)
                )
        lengths = []
        for epoch, field, bound, repeat in zip(
            self.epochs, self.length_fields, self.bounds, self.epoch_repeats, strict=True,
        ):
            given = parameters.epoch_steps
            if given and epoch in given:
                length = given[epoch]
            elif field is None:
                length = bound  # Until has a fixed limit; Rest contributes zero here.
            else:
                value = field.evaluate(parameters, prepared, values | {"i": repeat}, repeat=repeat)
                length = value if field.length_in_steps else runtime_time_steps(value, self.period)
            lengths.append(jnp.clip(length, 0, bound))
        rest = next(
            (index for index, (field, condition) in enumerate(zip(self.length_fields, self.conditions, strict=True))
             if field is None and condition is None),
            None,
        )
        if rest is not None:
            lengths[rest] = jnp.maximum(0, self.n_steps - sum(lengths))
        raw = jnp.asarray(lengths, dtype=int)
        ends = jnp.cumsum(raw)
        starts = ends - raw
        reached = starts <= self.n_steps
        epochs = EpochRecord(
            jnp.where(reached, starts, -1),
            jnp.where(reached, jnp.minimum(ends, self.n_steps), -1),
            jnp.where(reached, int(EpochEnd.LENGTH), int(EpochEnd.UNREACHED)),
        )
        absent = {
            signal.id: signal.absent.evaluate(parameters, prepared, values)
            for signal in self.signals
        }
        inputs = Inputs((), values, run_key, trial_key)
        windows = {
            signal.id: signal.window.mask(epochs, self.n_steps, inputs)
            for signal in self.signals if signal.window is not None
        }
        prepared = prepared._replace(epochs=epochs, absent=absent, lengths=raw, windows=windows)
        timeline = None
        if any(self.conditions):
            unreachable = jnp.full(raw.shape, -1, dtype=raw.dtype)
            timeline = _Timeline(
                jnp.asarray(0), EpochRecord(unreachable.at[0].set(0), unreachable, jnp.zeros_like(raw)),
                tuple(
                    condition.prepare(parameters, prepared, values | {"i": repeat}, repeat=repeat)
                    if condition is not None else None
                    for condition, repeat in zip(self.conditions, self.epoch_repeats, strict=True)
                ),
            )
        variable_state = {
            variable.id: field.prepare(parameters, prepared, values)
            for variable, field in self.variable_fields
            if variable.level == Level.STEP
        }
        cell_state = tuple(
            tuple(
                cell.field.prepare(parameters, prepared, values | {"i": cell.repeat}, repeat=cell.repeat)
                for cell in signal.cells
            )
            for signal in self.signals
        )
        return prepared, _TaskState(variable_state, cell_state, timeline)

    def _advance(self, parameters, trial, timeline, values, defined, k):
        """Take every transition due at this boundary, including level tests on entry."""
        lengths = trial.lengths
        for index, (condition, repeat, length_field) in enumerate(zip(
            self.conditions, self.epoch_repeats, self.length_fields, strict=True,
        )):
            def test(current):
                record = current.record
                start = record.starts[index]
                duration = lengths[index]
                if condition is None and length_field is None:
                    duration = self.n_steps - start - jnp.sum(lengths[index + 1:])
                met = False
                states = current.conditions
                if condition is not None:
                    t = (k - start) * self.period
                    T = duration * self.period
                    environment = values | {"t": t, "T": T, "u": t / jnp.where(T > 0, T, 1), "i": repeat}
                    inputs = condition.inputs(
                        parameters, trial, environment, k, epoch_entry=k == start, repeat=repeat,
                    )._replace(defined=defined)
                    result, state = condition.step(inputs, states[index])
                    states = (*states[:index], state, *states[index + 1:])
                    met = result.value & result.defined
                done = met | (k - start >= duration)
                outcome = jnp.where(
                    met, int(EpochEnd.CONDITION),
                    int(EpochEnd.LIMIT if condition is not None else EpochEnd.LENGTH),
                )
                starts = record.starts
                if index + 1 < len(self.epochs):
                    starts = starts.at[index + 1].set(jnp.where(done, k, starts[index + 1]))
                return _Timeline(
                    current.index + done.astype(int),
                    EpochRecord(
                        starts,
                        record.ends.at[index].set(jnp.where(done, k, -1)),
                        record.outcomes.at[index].set(jnp.where(done, outcome, int(EpochEnd.UNREACHED))),
                    ),
                    states,
                )

            timeline = jax.lax.cond(timeline.index == index, test, lambda current: current, timeline)
        return timeline

    def step(self, parameters, variables, state, k, model_outputs):
        """Emit the sample for step k using the same transition as ahead()."""
        values = dict(variables.values)
        values.update((name, sample.value) for name, sample in model_outputs.items())
        values["time"] = k * self.period
        defined = {name: sample.defined for name, sample in model_outputs.items()}
        dynamic = {}
        dynamic_defined = {}
        next_variables = {}
        for variable, field in self.variable_fields:
            if variable.level == Level.STEP:
                inputs = field.inputs(parameters, variables, values, k)._replace(defined=defined)
                result, next_variables[variable.id] = field.step(
                    inputs, state.variables[variable.id]
                )
                values[variable.id] = result.value
                defined[variable.id] = result.defined
                dynamic[variable.id] = result.value
                dynamic_defined[variable.id] = result.defined

        timeline = state.timeline
        if timeline is not None:
            timeline = self._advance(parameters, variables, timeline, values, defined, k)

        outputs = {}
        present = {}
        next_cells = []
        for signal, cell_states in zip(self.signals, state.cells, strict=True):
            output = variables.absent[signal.id]
            available = False
            next_signal = []
            within = True
            if signal.window is not None:
                within = variables.windows[signal.id][k] if self.n_steps else False
            for cell, previous in zip(signal.cells, cell_states, strict=True):
                if timeline is None:
                    start = variables.epochs.starts[cell.first]
                    end = variables.epochs.ends[cell.last]
                    active = in_epoch(k, start, end)
                    length = end - start
                else:
                    start = timeline.record.starts[cell.first]
                    active = (cell.first <= timeline.index) & (timeline.index <= cell.last)
                    length = jnp.sum(variables.lengths[cell.first:cell.last + 1])
                duration = length * self.period
                t = (k - start) * self.period
                environment = values | {
                    "t": t,
                    "T": duration,
                    "u": t / jnp.where(duration > 0, duration, 1),
                    "i": cell.repeat,
                }
                inputs = cell.field.inputs(
                    parameters, variables, environment, k, epoch_entry=k == start, repeat=cell.repeat
                )._replace(defined=defined)
                result, next_state = cell.field.code.evaluate_step(inputs, previous)
                next_signal.append(next_state)
                supplied = within & active & result.defined
                output = jnp.where(supplied, result.value, output)
                available = available | supplied
            outputs[signal.id] = output
            present[signal.id] = jnp.asarray(available)
            next_cells.append(tuple(next_signal))
        return _TaskState(next_variables, tuple(next_cells), timeline), outputs, present, dynamic, dynamic_defined

    def finish(self, parameters, variables, state, samples, model_outputs):
        inputs, present, traces, defined = samples
        epochs = variables.epochs
        if state.timeline is not None:
            state, *_ = self.step(parameters, variables, state, jnp.asarray(self.n_steps), model_outputs)
            epochs = state.timeline.record
        fixed = {
            variable.id: variables.values[variable.id]
            for variable in self.interface.variables
            if variable.level < Level.STEP
        }
        return Trial(self.inits(variables), inputs, present, fixed | traces, epochs, defined)

    def inits(self, variables):
        return {address: variables.values[variable] for address, variable in self.interface.links}

    def sample(self, parameters, run_key, trial_key) -> Trial | PendingTrial:
        if self.reads_model:
            return PendingTrial(parameters, *self.init(parameters, run_key, trial_key))
        return self.ahead(parameters, run_key, trial_key)

    def ahead(self, parameters, run_key, trial_key) -> Trial:
        if self.reads_model:
            raise ValueError("This Task needs Model outputs to produce a trial.")
        variables, state = self.init(parameters, run_key, trial_key)

        def step(carry, k):
            carry, inputs, present, traces, defined = self.step(parameters, variables, carry, k, {})
            return carry, (inputs, present, traces, defined)

        final, samples = jax.lax.scan(step, state, jnp.arange(self.n_steps))
        return self.finish(parameters, variables, final, samples, {})

    def validation(self, parameters, key) -> Trial | PendingTrial:
        keys = validation_trial_keys(key, len(self.validation_rows))
        return self._sample_rows(self.validation_rows, parameters, key, keys)

    def evaluation(self, parameters, key, trial_ids, trial_keys) -> Trial | PendingTrial:
        indices = trial_indices(trial_ids, self.condition_ids)
        rows = tuple(self.validation_rows[i] for i in indices)
        return self._sample_rows(rows, parameters, key, trial_keys)

    def _sample_rows(self, rows, parameters, key, trial_keys) -> Trial:
        if not rows[0].variables and not rows[0].epoch_steps:
            return map_trial_samples(lambda trial_key: self.sample(parameters, key, trial_key), trial_keys, trial_keys)
        batch = jax.tree.map(lambda *samples: jnp.stack(samples), *rows)
        return map_trial_samples(
            lambda item: self.sample(item[0]._replace(fields=parameters.fields), key, item[1]),
            (batch, trial_keys), trial_keys,
        )

    def model_inputs(self, trial: Trial) -> dict[str, jax.Array]:
        return {port: trial.inputs[signal] for port, signal in self.input_bindings}


def _validation_expression(value: ValidationValue) -> Expression:
    if isinstance(value, tuple):
        return CallValue(function="vector", args=tuple(map(_validation_expression, value)))
    return LiteralValue(value=value)


def zero(base, shape) -> Expression:
    """The value of a type that is nothing: false, or zero in every entry."""
    if not shape:
        return LiteralValue(value=False if base == ValueBase.BOOLEAN else 0)
    child = zero(base, shape[1:])
    return CallValue(function="vector", args=(child,) * shape[0])


def _compatible_task_signal(model: ResolvedModelInterface, port, offered: ValueType) -> bool:
    leaf = model.signals.at(port.address)
    return leaf.accepts(offered) if leaf is not None else port.type.accepts(offered)


def bind_task_inputs(
    model: ResolvedModelInterface,
    signals: tuple[ScopedValue, ...],
    input_bindings: Mapping[str, str | None] | None,
    *, input_labels: Mapping[str, str] | None = None,
) -> tuple[tuple[tuple[str, str], ...], tuple[Problem, ...]]:
    """Resolve runtime input names to stable signal IDs and report invalid bindings."""
    problems = []
    bindings = []
    by_id = {value.id: value for value in signals}
    authored = input_bindings or {}
    for port in model.inputs:
        label = (input_labels or {}).get(port.root_id, port.name)
        signal_id = selected_signal_id(port.root_id, by_id, authored)
        signal = by_id.get(signal_id) if signal_id is not None else None
        path = ("model", "inputs", port.id)
        if signal is None:
            if signal_id is not None:
                reason = f"Model input {label!r} reads undeclared signal ID {signal_id!r}."
            elif port.required:
                reason = f"Model input {label!r} needs a Task signal."
            else:
                continue
            problems.append(Problem(
                code=ProblemCode.UNKNOWN_NAME, reason=reason, path=path, model_address=port.address,
            ))
        elif not _compatible_task_signal(model, port, signal.type):
            problems.append(Problem(
                code=ProblemCode.TYPE_MISMATCH,
                reason=f"Signal {signal.name!r} does not fit model input {label!r}.",
                path=path, model_address=port.address,
            ))
        else:
            bindings.append((port.name, signal.id))
    for port in sorted(authored.keys() - {item.root_id for item in model.inputs}):
        problems.append(Problem(
            code=ProblemCode.UNKNOWN_NAME, reason=f"{port!r} is not an input of the Model.",
            path=("model", "inputs", port),
        ))

    return tuple(bindings), tuple(problems)


def delay_initial_problems(
    model: ResolvedModelInterface, variables: Mapping[str, ScopedValue], broken: set[str] | frozenset[str] = frozenset(),
) -> tuple[Problem, ...]:
    failures = []
    for link in model.links:
        variable = variables.get(link.variable)
        place = {"path": ("model", *link.address.stored_path()), "variable": link.variable, "model_address": link.address}
        if variable is None or link.variable in broken:
            failures.append(Problem(
                code=ProblemCode.UNKNOWN_NAME, reason=f"Delay Initial Task variable {link.variable!r} is unavailable.", **place,
            ))
        elif variable.list_count is not None or variable.level > Level.TRIAL or not link.type.accepts(variable.type):
            failures.append(Problem(
                code=ProblemCode.TYPE_MISMATCH, reason=f"{variable.name} does not fit a trial-start Delay Initial.", **place,
            ))
    return tuple(failures)


def compile_task(
    document: TaskDocument,
    functions: FunctionRegistry,
    *,
    model: ResolvedModelInterface,
    input_bindings: Mapping[str, str | None] | None = None,
    sites: dict[tuple[str | int, ...], Site] | None = None,
    link_problems: list[Problem] | None = None,
    nominal: bool = False,
) -> DocumentTask | tuple[Problem, ...]:
    """Compile a bounded ahead-of-rollout task from resolved model field types, or every independent Problem.

    A variable with problems keeps its declared type in scope, so each field that reads it is marked where it does, and
    fields that do not read it report only their own problems.
    """
    problems = []
    period = model.period_seconds
    if not isfinite(period) or period <= 0:
        return (Problem(code=ProblemCode.STRUCTURE, reason="The model period must be positive."),)

    def problem(code, reason, path, variable=None, model_address=None):
        target = link_problems if link_problems is not None and path[:1] == ("links",) else problems
        target.append(Problem(code=code, reason=reason, path=path, variable=variable, model_address=model_address))

    def unique(items, section):
        result, names = {}, set()
        for item in items:
            if (
                item.id in result
                or item.name in names
                or item.name in RESERVED_NAMES
                or item.id in RESERVED_NAMES
            ):
                problem(
                    ProblemCode.DUPLICATE_NAME,
                    f"Repeated or reserved name {item.name!r}.",
                    (section, item.id),
                )
                continue
            result[item.id] = item
            names.add(item.name)
        return result

    variables = unique(document.variables, "variables")
    epochs = unique(document.epochs, "epochs")
    kept_signals = unique(document.signals, "signals")
    broken: set[str] = set()
    """The variables with problems of their own or read from another that has them."""

    def lower(tree, expected, finest, scope, address, *, epoch=False, discrete=False,
              list_count=None, repeat_count=None):
        site = Site(
            expected=expected, finest=finest, names=tuple(scope),
            list_count=list_count, repeat_count=repeat_count,
        )
        if sites is not None:
            sites[address] = site
        code = compile_expression(
            tree,
            site,
            functions,
            field_address=(document.id, *map(str, address)),
            period_seconds=period,
            reset_on_epoch=epoch,
            discrete_uniform=discrete,
            path=address,
            nominal=nominal,
        )
        marks = [
            Problem(
                code=ProblemCode.UNKNOWN_NAME,
                reason=f"{variables[node.reference].name!r} has problems.",
                path=(*address, *at),
            )
            for at, node in walk(code.checked.tree if isinstance(code, Compiled) else tree)
            if isinstance(node, NameValue) and node.reference in broken
        ]
        problems.extend(marks if isinstance(code, Compiled) else (*code, *marks))
        return _Field(address, code, list_count) if isinstance(code, Compiled) and not marks else None

    time_type = ValueType(base=ValueBase.REAL, dimensions=TIME)
    time = ScopedValue(id="time", name="time", type=time_type, level=Level.STEP)
    integer = ValueType(base=ValueBase.INTEGER)
    repeat_index = ScopedValue(id="i", name="i", type=integer, level=Level.TRIAL)
    count_scope = tuple(
        ScopedValue(id=key, name=item.name, type=item.type, level=Level.RUN)
        for key, item in variables.items()
        if item.type == integer and item.count is None
    )
    count_site = Site(expected=integer, finest=Level.RUN, names=count_scope, structural=True)

    def dependency_tree(value, site):
        parsed = resolve_expression(value, site, functions)
        return value if isinstance(parsed, tuple) else parsed.value

    values = {key: item.value for key, item in variables.items()}
    if any(isinstance(value, UnresolvedValue) for value in values.values()):
        names = (*(
            ScopedValue(id=key, name=item.name, type=item.type, level=Level.STEP)
            for key, item in variables.items()
        ), time)
        for key, value in values.items():
            if isinstance(value, UnresolvedValue):
                site = Site(expected=variables[key].type, finest=Level.STEP, names=(*model.outputs, *names))
                values[key] = dependency_tree(value, site)
    count_expressions = {
        key: dependency_tree(item.count, count_site)
        for key, item in variables.items() if item.count is not None
    }
    count_references = set().union(*(
        references(item) for item in (
            *count_expressions.values(), *(dependency_tree(repeat.count, count_site) for repeat in document.repeats),
        )
    ))
    pending = list(count_references)
    for key in pending:
        if key in variables:
            for dependency in references(values[key]) - count_references:
                count_references.add(dependency)
                pending.append(dependency)
    order, cycles = dependency_order(values, counts=count_expressions)
    problems.extend(cycles)
    scope = {}
    model_scope = {output.id: output for output in model.outputs}
    for cycle in cycles:
        (_, key, _) = cycle.path
        broken.add(key)
        scope[key] = ScopedValue(id=key, name=variables[key].name, type=variables[key].type, level=Level.RUN)
    fields = []
    fixed_values = {}
    fixed_key = jax.random.key(0) if count_references else None
    fixed = _Prepared(fixed_values, None, None, fixed_key, fixed_key)
    variable_overrides = set().union(*(row.variables for row in document.validation_trials))
    def resolve_count(tree, path):
        if sites is not None:
            sites[path] = count_site
        parsed = resolve_expression(tree, count_site, functions)
        if isinstance(parsed, tuple):
            problems.extend(item.model_copy(update={"path": (*path, *item.path)}) for item in parsed)
            return None
        tree = parsed.value
        if isinstance(tree, LiteralValue) and type(tree.value) is int and tree.unit is None:
            count = tree.value
        elif isinstance(tree, NameValue) and tree.reference in fixed_values and any(
            item.id == tree.reference for item in count_scope
        ):
            count = int(fixed_values[tree.reference])
        else:
            problem(ProblemCode.LIST_COUNT, "Count needs an integer or a fixed Integer variable.", path)
            return None
        if not 0 < count < 2**31:
            problem(ProblemCode.LIST_COUNT, "Count must be a positive integer.", path)
            return None
        return count

    for key in order:
        variable = variables[key]
        count = (
            resolve_count(variable.count, ("variables", key, "count")) if variable.count is not None else None
        )
        field = lower(
            variable.value,
            variable.type,
            Level.TRIAL if variable.set_choices is not None else Level.STEP,
            (*model.outputs, *scope.values(), time),
            ("variables", key, "value"),
            list_count=count,
        )
        level = Level.RUN if field is None else field.code.checked.root.level
        scope[key] = ScopedValue(id=key, name=variable.name, type=variable.type, level=level, list_count=count)
        if field is None or variable.count is not None and count is None:
            broken.add(key)
        else:
            fields.append((scope[key], field))
            if (
                key in count_references and key not in variable_overrides and level == Level.RUN
                and references(values[key]) <= fixed_values.keys()
                and all(
                    not isinstance(node, DrawValue)
                    and not (
                        isinstance(node, CallValue) and (function := functions.resolve(node.function))
                        and function.info.random
                    )
                    for _, node in walk(values[key])
                )
            ):
                fixed_values[key] = field.evaluate(TaskParameters(), fixed, fixed_values)

    if sites is not None:
        for key, variable in variables.items():
            sites[("variables", key, "value")] = Site(
                expected=variable.type, finest=Level.STEP,
                names=(*model.outputs, *scope.values(), time), list_count=scope[key].list_count,
            )

    indexes = {key: index for index, key in enumerate(epochs)}
    repeat_spans = {}
    repeat_ids = set()
    for repeat in document.repeats:
        path = ("repeats", repeat.id)
        count = resolve_count(repeat.count, (*path, "count"))
        first, last = indexes.get(repeat.first_epoch), indexes.get(repeat.last_epoch)
        if repeat.id in repeat_ids or first is None or last is None or first > last:
            problem(ProblemCode.STRUCTURE, "A repeat needs a unique id and an ordered epoch span.", path)
            continue
        repeat_ids.add(repeat.id)
        span = set(range(first, last + 1))
        if span & repeat_spans.keys():
            problem(ProblemCode.STRUCTURE, "Repeats cannot overlap.", path)
            continue
        repeat_spans.update({index: (first, last, count) for index in span})

    axes = tuple(
        ScopedValue(id=name, name=name, type=value_type, level=Level.STEP)
        for name, value_type in (
            ("t", time_type),
            ("T", time_type),
            ("u", ValueType(base=ValueBase.REAL)),
        )
    )
    lengths = []
    conditions = []
    unread = set()
    """The epochs whose length has problems."""
    bounds = []
    counts: list[EpochSteps | None] = []
    rest = []
    for epoch in document.epochs:
        path = ("epochs", epoch.id, "end", "length")
        end = epoch.end
        repeat = repeat_spans.get(indexes.get(epoch.id))
        condition_path = ("epochs", epoch.id, "end", "condition")
        condition_names = (
            *model.outputs, *scope.values(), time, *axes, *((repeat_index,) if repeat else ()),
        )
        if sites is not None:
            sites[condition_path] = Site(
                expected=ValueType(base=ValueBase.BOOLEAN), finest=Level.STEP, names=condition_names,
                repeat_count=repeat[2] if repeat else None,
            )
            if isinstance(end, Until):
                sites[path] = Site(expected=time_type, finest=Level.TRIAL, names=tuple(scope.values()))
        if isinstance(end, Until):
            if rest:
                problem(
                    ProblemCode.STRUCTURE, "An Until epoch must precede the Rest epoch.",
                    ("epochs", epoch.id, "end"),
                )
            condition = lower(
                end.condition, ValueType(base=ValueBase.BOOLEAN), Level.STEP, condition_names, condition_path,
                epoch=True, repeat_count=repeat[2] if repeat else None,
            )
            limit_path = ("epochs", epoch.id, "end", "limit")
            if sites is not None:
                sites[limit_path] = Site(expected=time_type, finest=Level.RUN, structural=True)
            bound = 0
            try:
                bound = length_steps(end.limit, period)
            except ValueError as error:
                problem(ProblemCode.STRUCTURE, str(error), limit_path)
            if condition is not None:
                for at, node in walk(condition.code.checked.tree):
                    if (isinstance(node, CallValue) and node.function == "held"
                        and isinstance(duration := node.args[1], LiteralValue)):
                        try:
                            length_steps(literal_time(duration), period)
                        except ValueError as error:
                            problem(ProblemCode.STRUCTURE, str(error), (*condition_path, *at, "args", 1))
            lengths.append(None)
            conditions.append(condition)
            bounds.append(bound)
            counts.append(EpochSteps(least=0, nominal=bound, greatest=bound))
            unread.add(epoch.id)
            continue
        conditions.append(None)
        if isinstance(end, Rest):
            if sites is not None:
                # A Rest End is still the site of the length it can be given instead.
                sites[path] = Site(expected=time_type, finest=Level.TRIAL, names=tuple(scope.values()))
            rest.append(epoch.id)
            lengths.append(None)
            bounds.append(0)
            counts.append(None)
            continue
        length_scope = (*scope.values(), repeat_index) if repeat is not None else tuple(scope.values())
        length = dependency_tree(end.length, Site(expected=time_type, finest=Level.TRIAL, names=length_scope))
        discrete = isinstance(length, DrawValue) and length.family == "uniform"
        field = lower(
            end.length, time_type, Level.TRIAL, length_scope, path, discrete=discrete,
            repeat_count=repeat[2] if repeat is not None else None,
        )
        bound = end.max_steps
        if field is None:
            unread.add(epoch.id)
            lengths.append(None)
            bounds.append(bound or 0)
            counts.append(EpochSteps(least=0, nominal=bound or 0, greatest=bound or 0))
            continue
        count = None
        if isinstance(length, LiteralValue):
            try:
                steps = length_steps(literal_time(length), period)
            except ValueError as error:
                problem(ProblemCode.STRUCTURE, str(error), path)
                steps = 0
            field = replace(field, code=replace(field.code, parameters=(jnp.asarray(steps),)), length_in_steps=True)
            if bound is None:
                bound = steps
            elif steps > bound:
                problem(ProblemCode.LENGTH_BUDGET, "The length exceeds its step bound.", path)
            count = EpochSteps(least=max(steps, 0), nominal=max(steps, 0), greatest=max(steps, 0))
        if discrete:
            field = replace(field, length_in_steps=True)
        if discrete and all(isinstance(arg, LiteralValue) for arg in length.args):
            low, high = map(int, field.code.parameters)
            if not 0 <= low <= high:
                problem(
                    ProblemCode.STRUCTURE, "Length draw needs ordered nonnegative bounds.", path
                )
            elif bound is None:
                bound = high
            elif high > bound:
                problem(ProblemCode.LENGTH_BUDGET, "The draw exceeds the epoch bound.", path)
            if 0 <= low <= high:
                count = EpochSteps(least=low, nominal=(low + high + 1) // 2, greatest=high)
        if bound is None:
            problem(ProblemCode.UNBOUNDED_LENGTH, "A variable length needs a step bound.", path)
        lengths.append(field)
        bounds.append(bound)
        counts.append(count or EpochSteps(least=0, nominal=bound or 0, greatest=bound or 0))
    if any((repeat_spans.get(indexes[epoch], (0, 0, 1))[2] or 1) > 1 for epoch in rest):
        problem(ProblemCode.STRUCTURE, "A rest epoch cannot repeat.", ("epochs",))
    if len(rest) > 1 or rest and document.fixed_steps is None:
        problem(ProblemCode.STRUCTURE, "One rest epoch requires a fixed trial length.", ("epochs",))
    max_steps = sum(
        (bound or 0) * (repeat_spans.get(index, (0, 0, 1))[2] or 1) for index, bound in enumerate(bounds)
    )
    if document.fixed_steps is not None and max_steps > document.fixed_steps:
        problem(
            ProblemCode.LENGTH_BUDGET,
            "The length bounds exceed the fixed trial length.",
            ("epochs",),
        )

    signals = []
    declared_signals = []
    for signal in document.signals:
        path = ("signals", signal.id)
        absent_value = signal.absent or zero(signal.type.base, signal.type.shape)
        absent = lower(absent_value, signal.type, Level.TRIAL, scope.values(), (*path, "absent"))
        occupied = set()
        cell_ids = set()
        cells = []
        for cell in signal.cells:
            cell_path = (*path, "cells", cell.id)
            if cell.id in cell_ids:
                problem(ProblemCode.DUPLICATE_NAME, f"Repeated cell id {cell.id!r}.", cell_path)
                continue
            cell_ids.add(cell.id)
            first, last = indexes.get(cell.first_epoch), indexes.get(cell.last_epoch)
            if first is None or last is None or first > last:
                problem(
                    ProblemCode.OUT_OF_SCOPE,
                    "Cell endpoints must identify an ordered epoch span.",
                    cell_path,
                )
                continue
            span = set(range(first, last + 1))
            if occupied & span:
                problem(ProblemCode.STRUCTURE, "Cells for one signal overlap.", cell_path)
                continue
            occupied.update(span)
            brackets = {repeat_spans.get(index) for index in span}
            repeated = {bracket[2] for bracket in brackets if bracket is not None and bracket[2] is not None}
            inside = None not in brackets
            repeat_count = min(repeated) if inside and repeated else None
            cell_names = (*scope.values(), time, *axes, *((repeat_index,) if inside else ()))
            value_path = (*cell_path, "value")
            code = lower(
                cell.value, signal.type, Level.STEP, cell_names, value_path,
                epoch=True, repeat_count=repeat_count,
            )
            if code is not None:
                cells.append(_Cell(first, last, code, None))
                if inside and len(repeated) > 1:
                    other = check(
                        cell.value, Site(expected=signal.type, finest=Level.STEP, names=cell_names,
                                         repeat_count=max(repeated)),
                        functions, path=value_path,
                    )
                    if not isinstance(other, Checked):
                        problems.extend(other)
        if absent is not None:
            signals.append(_Signal(signal.id, absent, tuple(cells)))
        if kept_signals.get(signal.id) is signal:
            declared_signals.append(
                ScopedValue(id=signal.id, name=signal.name, type=signal.type, level=Level.STEP)
            )

    bindings, input_problems = bind_task_inputs(model, tuple(declared_signals), input_bindings)
    problems.extend(input_problems)

    writable = {field.address.to_text(): field for field in model.reset.fields}
    links = {}
    for link in document.links:
        target = writable.get(link.model_field)
        variable = scope.get(link.variable)
        path = ("links", link.model_field)
        if target is None:
            name = variable.name if variable is not None else link.variable
            problem(
                ProblemCode.UNKNOWN_NAME, f"{name} is linked to a field the Model does not have.", path,
                link.variable if variable is not None else None,
            )
        elif variable is None:
            problem(
                ProblemCode.UNKNOWN_NAME, f"{link.variable} is not a variable of the Task.", path,
                model_address=target.address,
            )
        elif link.variable in broken:
            problem(ProblemCode.UNKNOWN_NAME, f"{variable.name!r} has problems.", path, link.variable)
        elif variable.list_count is not None or variable.level > target.finest or not target.accepts(variable.type):
            problem(ProblemCode.TYPE_MISMATCH, f"{variable.name} does not fit its Model field.", path, link.variable)
        elif link.model_field in links:
            problem(
                ProblemCode.STRUCTURE, f"{variable.name} links a Model field that is already linked.", path, link.variable,
            )
        else:
            links[link.model_field] = link.variable
    problems.extend(delay_initial_problems(model, scope, broken))
    for address, field in writable.items():
        if field.required and all(link.model_field != address for link in document.links):
            problem(
                ProblemCode.STRUCTURE, "The Model needs this value at the start of each trial.", ("links", address),
                model_address=field.address,
            )
    rows = []
    first = document.validation_trials[0]
    for index, row in enumerate(document.validation_trials):
        path = ("validation_trials", index)
        if (
            row.variables.keys() != first.variables.keys()
            or row.epoch_steps.keys() != first.epoch_steps.keys()
        ):
            problem(ProblemCode.STRUCTURE, "Validation rows must assign the same fields.", path)
            continue
        row_values = {}
        for key, value in row.variables.items():
            variable = scope.get(key)
            if key in broken:
                continue
            if variable is None or variable.level == Level.STEP:
                problem(
                    ProblemCode.OUT_OF_SCOPE,
                    "Validation needs a run or trial variable.",
                    (*path, "variables", key),
                )
                continue
            code = lower(
                _validation_expression(value),
                variable.type if variable.list_count is None else variable.type.model_copy(
                    update={"shape": (variable.list_count, *variable.type.shape)}
                ),
                Level.RUN,
                (),
                (*path, "variables", key),
            )
            if code is not None:
                literal_inputs = Inputs(code.code.parameters, {}, None, None)
                run = code.code.evaluate_run(literal_inputs)
                row_values[key] = code.code.evaluate_step(
                    literal_inputs, code.code.with_trial(literal_inputs, run)
                )[0].value
        for key, value in row.epoch_steps.items():
            epoch_index = indexes.get(key)
            if epoch_index is not None and isinstance(document.epochs[epoch_index].end, Until):
                problem(
                    ProblemCode.STRUCTURE, "Validation cannot replace a condition-ended epoch with a fixed length.",
                    (*path, "epoch_steps", key),
                )
                continue
            if key in unread:
                continue
            if (
                epoch_index is None
                or lengths[epoch_index] is None
                or not 0 <= value <= bounds[epoch_index]
            ):
                problem(
                    ProblemCode.LENGTH_BUDGET,
                    "Validation epoch step count must fit a bounded length.",
                    (*path, "epoch_steps", key),
                )
        rows.append(
            TaskParameters(
                variables=row_values,
                epoch_steps={key: jnp.asarray(value) for key, value in row.epoch_steps.items()},
            )
        )
    if problems:
        return tuple(problems)
    groups = []
    for span, members in groupby(range(len(epochs)), key=repeat_spans.get):
        members = tuple(members)
        if span is None:
            groups.append((members, None))
        else:
            assert span[2] is not None
            groups.extend((members, index) for index in range(span[2]))
    instances = tuple(index for members, _ in groups for index in members)
    repeat_indices = tuple(repeat for members, repeat in groups for _ in members)
    n_steps = max_steps if document.fixed_steps is None else document.fixed_steps
    if rest:
        others = [counts[index] for index in instances if counts[index] is not None]
        index = counts.index(None)
        counts[index] = EpochSteps(
            least=max(0, n_steps - sum(count.greatest for count in others)),
            nominal=max(0, n_steps - sum(count.nominal for count in others)),
            greatest=max(0, n_steps - sum(count.least for count in others)),
        )
    if document.repeats:
        def instantiate(signal):
            cells = []
            offset = 0
            for members, repeat in groups:
                for cell in signal.cells:
                    first = max(members[0], cell.first)
                    last = min(members[-1], cell.last)
                    if first <= last:
                        cells.append(_Cell(offset + first - members[0], offset + last - members[0], cell.field, repeat))
                offset += len(members)
            return replace(signal, cells=tuple(cells))

        signals = [instantiate(signal) for signal in signals]
    epoch_ids = tuple(epochs)
    instance_ids = tuple(epoch_ids[index] for index in instances)
    first_until = next(
        (slot for slot, index in enumerate(instances) if conditions[index] is not None),
        len(instances),
    )
    windows = {signal.id: signal.window for signal in document.signals if signal.window is not None}
    for index, signal in enumerate(signals):
        if signal.id not in windows:
            continue
        path = ("signals", signal.id, "window")
        compiled = compile_window(
            windows[signal.id], epochs=instance_ids, period=period, names=tuple(scope.values()),
            functions=functions, field_address=(document.id, *path), path=path, nominal=nominal,
        )
        if isinstance(compiled, tuple):
            problems.extend(compiled)
        elif compiled.reads_runtime_epoch(first_until):
            problem(
                ProblemCode.OUT_OF_SCOPE,
                "This signal window needs an epoch boundary that is only known after Until.",
                path,
            )
        else:
            signals[index] = replace(signal, window=compiled)
    if problems:
        return tuple(problems)
    used_outputs = frozenset().union(*(
        references(field.code.checked.tree)
        for field in (*(field for _, field in fields), *conditions)
        if field is not None
    )) & model_scope.keys()
    outputs = tuple(output for output in model.outputs if output.id in used_outputs)
    epoch_conditions = tuple((epoch.id, epoch.name) for epoch in document.epochs if isinstance(epoch.end, Until))
    set_choices = tuple((item.id, item.set_choices) for item in document.variables if item.set_choices is not None)
    return DocumentTask(
        TaskInterface(
            tuple(scope.values()), tuple(declared_signals), tuple(links.items()), outputs, epoch_conditions, set_choices,
        ),
        n_steps,
        period,
        instance_ids,
        tuple(fields),
        tuple(lengths[index] for index in instances),
        tuple(bounds[index] for index in instances),
        tuple(signals),
        tuple(rows),
        document.condition_ids,
        tuple(bindings),
        tuple(counts[index] for index in instances),
        repeat_indices,
        tuple(conditions[index] for index in instances),
        bool(outputs),
    )
