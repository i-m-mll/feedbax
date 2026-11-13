"""Objective-free task documents and their ahead-of-rollout runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from math import isfinite
from typing import Any, NamedTuple, Protocol

from feedbax.compiler.model_document import ResolvedModelInterface
from feedbax.contracts.model_address import ModelAddress

import jax
import jax.numpy as jnp
from pydantic import Field

from feedbax.contracts.expression_values import (
    RESERVED_NAMES,
    Scalar,
    TIME,
    DrawValue,
    Expression,
    Level,
    CallValue,
    LiteralValue,
    Problem,
    ProblemCode,
    ScopedValue,
    Site,
    ValueBase,
    ValueRecord,
    ValueType,
)
from feedbax.expression_check import dependency_order
from feedbax.expression_compile import Compiled, Inputs, compile_expression
from feedbax.expression_registry import FunctionRegistry
from feedbax.runtime._graph_key_schedule import derive_key
from feedbax.runtime.epoch_windows import EpochRecord, in_epoch


class Variable(ValueRecord):
    id: str
    name: str
    type: ValueType
    value: Expression


class Epoch(ValueRecord):
    id: str
    name: str
    length: Expression | None
    max_steps: int | None = Field(default=None, ge=0)


class Cell(ValueRecord):
    first_epoch: str
    last_epoch: str
    value: Expression


class Signal(ValueRecord):
    id: str
    name: str
    type: ValueType
    cells: tuple[Cell, ...]
    absent: Expression | None = None


class Link(ValueRecord):
    model_field: str
    variable: str


type ValidationValue = Scalar | tuple[ValidationValue, ...]


class ValidationTrial(ValueRecord):
    variables: dict[str, ValidationValue] = Field(default_factory=dict)
    epoch_steps: dict[str, int] = Field(default_factory=dict)


class TaskDocument(ValueRecord):
    id: str
    variables: tuple[Variable, ...]
    epochs: tuple[Epoch, ...] = Field(min_length=1)
    signals: tuple[Signal, ...]
    links: tuple[Link, ...] = ()
    fixed_steps: int | None = Field(default=None, ge=0)
    validation_trials: tuple[ValidationTrial, ...] = Field(
        default=(ValidationTrial(),), min_length=1
    )


@dataclass(frozen=True)
class ModelField:
    address: ModelAddress
    type: ValueType


@dataclass(frozen=True)
class TaskInterface:
    variables: tuple[ScopedValue, ...]
    signals: tuple[ScopedValue, ...]
    links: tuple[tuple[str, str], ...]
    model_outputs: tuple[ScopedValue, ...] = ()


class EpochEnd(IntEnum):
    UNREACHED = 0
    LENGTH = 1


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


class TrialSource(Protocol):
    @property
    def interface(self) -> TaskInterface: ...
    @property
    def n_steps(self) -> int: ...
    @property
    def reads_model(self) -> bool: ...
    def init(self, parameters: TaskParameters, run_key: jax.Array, trial_key: jax.Array): ...
    def step(self, parameters: TaskParameters, variables, state, k: jax.Array, model_outputs): ...
    def ahead(
        self, parameters: TaskParameters, run_key: jax.Array, trial_key: jax.Array
    ) -> Trial: ...
    def validation(self, parameters: TaskParameters, key: jax.Array) -> Trial: ...


class _Prepared(NamedTuple):
    values: Mapping[str, Any]
    epochs: EpochRecord
    absent: Mapping[str, Any]
    run_key: jax.Array
    trial_key: jax.Array


@dataclass(frozen=True)
class _Field:
    address: tuple[str, ...]
    code: Compiled

    def inputs(self, parameters, prepared, values, k=0, *, epoch_entry=False):
        args = (
            parameters.fields.get(self.address, self.code.parameters)
            if parameters.fields
            else self.code.parameters
        )
        return Inputs(
            args, values, prepared.run_key, prepared.trial_key, step=k, epoch_entry=epoch_entry
        )

    def prepare(self, parameters, prepared, values):
        inputs = self.inputs(parameters, prepared, values)
        return self.code.with_trial(inputs, self.code.evaluate_run(inputs))

    def evaluate(self, parameters, prepared, values):
        inputs = self.inputs(parameters, prepared, values)
        state = self.code.with_trial(inputs, self.code.evaluate_run(inputs))
        return self.code.evaluate_step(inputs, state)[0].value


@dataclass(frozen=True)
class _Cell:
    first: int
    last: int
    field: _Field


@dataclass(frozen=True)
class _Signal:
    name: str
    absent: _Field
    cells: tuple[_Cell, ...]


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
    input_bindings: tuple[tuple[str, str], ...]
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
        for epoch, field, bound in zip(self.epochs, self.length_fields, self.bounds, strict=True):
            given = parameters.epoch_steps
            if given and epoch in given:
                length = given[epoch]
            elif field is None:
                length = 0
            else:
                length = jnp.rint(
                    field.evaluate(parameters, prepared, values) / self.period
                ).astype(int)
            lengths.append(jnp.clip(length, 0, bound))
        if None in self.length_fields:
            lengths[self.length_fields.index(None)] = jnp.maximum(0, self.n_steps - sum(lengths))
        raw = jnp.asarray(lengths)
        ends = jnp.cumsum(raw)
        starts = ends - raw
        reached = starts <= self.n_steps
        epochs = EpochRecord(
            jnp.where(reached, starts, -1),
            jnp.where(reached, jnp.minimum(ends, self.n_steps), -1),
            jnp.where(reached, int(EpochEnd.LENGTH), int(EpochEnd.UNREACHED)),
        )
        absent = {
            signal.name: signal.absent.evaluate(parameters, prepared, values)
            for signal in self.signals
        }
        prepared = prepared._replace(epochs=epochs, absent=absent)
        variable_state = {
            variable.id: field.prepare(parameters, prepared, values)
            for variable, field in self.variable_fields
            if variable.level == Level.STEP
        }
        cell_state = tuple(
            tuple(cell.field.prepare(parameters, prepared, values) for cell in signal.cells)
            for signal in self.signals
        )
        return prepared, (variable_state, cell_state)

    def step(self, parameters, variables, state, k, model_outputs):
        """Emit the sample for step k using the same transition as ahead()."""
        previous_variables, previous_cells = state
        values = dict(variables.values)
        values["time"] = k * self.period
        defined = {}
        dynamic = {}
        dynamic_defined = {}
        next_variables = {}
        for variable, field in self.variable_fields:
            if variable.level == Level.STEP:
                inputs = field.inputs(parameters, variables, values, k)._replace(defined=defined)
                result, next_variables[variable.id] = field.code.evaluate_step(
                    inputs, previous_variables[variable.id]
                )
                values[variable.id] = result.value
                defined[variable.id] = result.defined
                dynamic[variable.name] = result.value
                dynamic_defined[variable.name] = result.defined

        outputs = {}
        present = {}
        next_cells = []
        for signal, cell_states in zip(self.signals, previous_cells, strict=True):
            output = variables.absent[signal.name]
            available = False
            next_signal = []
            for cell, previous in zip(signal.cells, cell_states, strict=True):
                start = variables.epochs.starts[cell.first]
                end = variables.epochs.ends[cell.last]
                active = in_epoch(k, start, end)
                duration = (end - start) * self.period
                t = (k - start) * self.period
                environment = values | {
                    "t": t,
                    "T": duration,
                    "u": t / jnp.where(duration > 0, duration, 1),
                }
                inputs = cell.field.inputs(
                    parameters, variables, environment, k, epoch_entry=k == start
                )._replace(defined=defined)
                result, next_state = cell.field.code.evaluate_step(inputs, previous)
                next_signal.append(next_state)
                supplied = active & result.defined
                output = jnp.where(supplied, result.value, output)
                available = available | supplied
            outputs[signal.name] = output
            present[signal.name] = jnp.asarray(available)
            next_cells.append(tuple(next_signal))
        return (next_variables, tuple(next_cells)), outputs, present, dynamic, dynamic_defined

    def ahead(self, parameters, run_key, trial_key) -> Trial:
        variables, state = self.init(parameters, run_key, trial_key)

        def step(carry, k):
            carry, inputs, present, traces, defined = self.step(parameters, variables, carry, k, {})
            return carry, (inputs, present, traces, defined)

        _, (inputs, present, traces, defined) = jax.lax.scan(step, state, jnp.arange(self.n_steps))
        fixed = {
            variable.name: variables.values[variable.id]
            for variable in self.interface.variables
            if variable.level < Level.STEP
        }
        return Trial(
            {address: variables.values[variable] for address, variable in self.interface.links},
            inputs,
            present,
            fixed | traces,
            variables.epochs,
            defined,
        )

    def validation(self, parameters, key) -> Trial:
        rows = self.validation_rows
        keys = jax.vmap(lambda i: derive_key(key, "task", "validation", index=i))(
            jnp.arange(len(rows))
        )
        if not rows[0].variables and not rows[0].epoch_steps:
            return jax.vmap(self.ahead, in_axes=(None, None, 0))(parameters, key, keys)
        batch = jax.tree.map(lambda *samples: jnp.stack(samples), *rows)
        return jax.vmap(
            lambda row, trial_key: self.ahead(
                row._replace(fields=parameters.fields), key, trial_key
            )
        )(batch, keys)

    def model_inputs(self, trial: Trial) -> dict[str, jax.Array]:
        return {port: trial.inputs[signal] for port, signal in self.input_bindings}


def _validation_expression(value: ValidationValue) -> Expression:
    if isinstance(value, tuple):
        return CallValue(function="vector", args=tuple(map(_validation_expression, value)))
    return LiteralValue(value=value)


def _zero(base, shape):
    if not shape:
        return LiteralValue(value=False if base == ValueBase.BOOLEAN else 0)
    child = _zero(base, shape[1:])
    return CallValue(function="vector", args=(child,) * shape[0])


def compile_task(
    document: TaskDocument,
    functions: FunctionRegistry,
    *,
    model: ResolvedModelInterface,
    reset_fields: Mapping[str, ModelField] | None = None,
    input_overrides: Mapping[str, str] | None = None,
) -> DocumentTask | tuple[Problem, ...]:
    """Compile a bounded ahead-of-rollout task from resolved model field types."""
    problems = []
    period = model.period_seconds
    if not isfinite(period) or period <= 0:
        return (Problem(code=ProblemCode.STRUCTURE, reason="The model period must be positive."),)

    def problem(code, reason, path):
        problems.append(Problem(code=code, reason=reason, path=path))

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
            result[item.id] = item
            names.add(item.name)
        return result

    variables = unique(document.variables, "variables")
    epochs = unique(document.epochs, "epochs")
    unique(document.signals, "signals")
    if problems:
        return tuple(problems)

    def lower(tree, expected, finest, scope, address, *, epoch=False, discrete=False):
        code = compile_expression(
            tree,
            Site(expected=expected, finest=finest, names=tuple(scope)),
            functions,
            field_address=(document.id, *map(str, address)),
            period_seconds=period,
            reset_on_epoch=epoch,
            discrete_uniform=discrete,
            path=address,
        )
        if isinstance(code, Compiled):
            return _Field(address, code)
        problems.extend(code)
        return None

    time_type = ValueType(base=ValueBase.REAL, dimensions=TIME)
    time = ScopedValue(id="time", name="time", type=time_type, level=Level.STEP)
    order = dependency_order({key: item.value for key, item in variables.items()})
    if order and isinstance(order[0], Problem):
        return order
    scope = {}
    fields = []
    for key in order:
        variable = variables[key]
        field = lower(
            variable.value,
            variable.type,
            Level.STEP,
            (*scope.values(), time),
            ("variables", key, "value"),
        )
        if field is not None:
            declared = ScopedValue(
                id=key, name=variable.name, type=variable.type, level=field.code.checked.root.level
            )
            scope[key] = declared
            fields.append((declared, field))
    if problems:
        return tuple(problems)

    lengths = []
    bounds = []
    rest = []
    for epoch in document.epochs:
        path = ("epochs", epoch.id, "length")
        if epoch.length is None:
            rest.append(epoch.id)
            lengths.append(None)
            bounds.append(0)
            continue
        discrete = isinstance(epoch.length, DrawValue) and epoch.length.family == "uniform"
        field = lower(epoch.length, time_type, Level.TRIAL, scope.values(), path, discrete=discrete)
        bound = epoch.max_steps
        if isinstance(epoch.length, LiteralValue) and field is not None:
            (seconds,) = field.code.parameters
            steps = round(float(seconds) / period)
            if seconds < 0 or (seconds > 0 and steps == 0):
                problem(
                    ProblemCode.STRUCTURE,
                    "Length must be zero or round to at least one step.",
                    path,
                )
            if bound is None:
                bound = steps
            elif steps > bound:
                problem(ProblemCode.LENGTH_BUDGET, "The length exceeds its step bound.", path)
        if (
            discrete
            and field is not None
            and all(isinstance(arg, LiteralValue) for arg in epoch.length.args)
        ):
            low, high = (round(float(seconds) / period) for seconds in field.code.parameters)
            if not 0 <= low <= high:
                problem(
                    ProblemCode.STRUCTURE, "Length draw needs ordered nonnegative bounds.", path
                )
            elif bound is None:
                bound = high
            elif high > bound:
                problem(ProblemCode.LENGTH_BUDGET, "The draw exceeds the epoch bound.", path)
        if bound is None:
            problem(ProblemCode.UNBOUNDED_LENGTH, "A variable length needs a step bound.", path)
        lengths.append(field)
        bounds.append(bound)
    if len(rest) > 1 or rest and document.fixed_steps is None:
        problem(ProblemCode.STRUCTURE, "One rest epoch requires a fixed trial length.", ("epochs",))
    if (
        document.fixed_steps is not None
        and sum(bound or 0 for bound in bounds) > document.fixed_steps
    ):
        problem(
            ProblemCode.LENGTH_BUDGET,
            "The length bounds exceed the fixed trial length.",
            ("epochs",),
        )
    if problems:
        return tuple(problems)

    axes = tuple(
        ScopedValue(id=name, name=name, type=value_type, level=Level.STEP)
        for name, value_type in (
            ("t", time_type),
            ("T", time_type),
            ("u", ValueType(base=ValueBase.REAL)),
        )
    )
    indexes = {key: index for index, key in enumerate(epochs)}
    signals = []
    declared_signals = []
    for signal in document.signals:
        path = ("signals", signal.id)
        absent_value = signal.absent or _zero(signal.type.base, signal.type.shape)
        absent = lower(absent_value, signal.type, Level.TRIAL, scope.values(), (*path, "absent"))
        occupied = set()
        cells = []
        for i, cell in enumerate(signal.cells):
            cell_path = (*path, "cells", i)
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
            code = lower(
                cell.value,
                signal.type,
                Level.STEP,
                (*scope.values(), time, *axes),
                (*path, "cells", cell.first_epoch, cell.last_epoch, "value"),
                epoch=True,
            )
            if code is not None:
                cells.append(_Cell(first, last, code))
        if absent is not None:
            signals.append(_Signal(signal.name, absent, tuple(cells)))
            declared_signals.append(
                ScopedValue(id=signal.id, name=signal.name, type=signal.type, level=Level.STEP)
            )

    bindings = []
    signal_names = {value.name: value for value in declared_signals}
    overrides = input_overrides or {}
    for port in model.inputs:
        signal_name = overrides.get(port.name, port.name)
        signal = signal_names.get(signal_name)
        path = ("model", "inputs", port.id)
        if signal is None:
            if port.required or port.name in overrides:
                problem(
                    ProblemCode.UNKNOWN_NAME,
                    f"Model input {port.name!r} needs signal {signal_name!r}.",
                    path,
                )
        elif not port.type.accepts(signal.type):
            problem(
                ProblemCode.TYPE_MISMATCH,
                f"Signal {signal_name!r} has an incompatible sample type.",
                path,
            )
        else:
            bindings.append((port.name, signal.name))
    for port in sorted(overrides.keys() - {item.name for item in model.inputs}):
        problem(
            ProblemCode.UNKNOWN_NAME, f"Unknown model input {port!r}.", ("model", "inputs", port)
        )

    # In the signed model interface, delay Initial links are reset-output
    # references. A trial cannot write those computed values.
    writable = reset_fields or {}
    links = {}
    for link in document.links:
        target = writable.get(link.model_field)
        variable = scope.get(link.variable)
        path = ("links", link.model_field)
        if target is None or variable is None:
            problem(
                ProblemCode.UNKNOWN_NAME,
                "Link must resolve to a model field and a task variable.",
                path,
            )
        elif variable.level == Level.STEP or not target.type.accepts(variable.type):
            problem(ProblemCode.TYPE_MISMATCH, "Link needs a compatible run or trial value.", path)
        elif target.address.to_text() in links:
            problem(ProblemCode.STRUCTURE, "A model field may be linked once.", path)
        else:
            links[target.address.to_text()] = link.variable
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
            if variable is None or variable.level == Level.STEP:
                problem(
                    ProblemCode.OUT_OF_SCOPE,
                    "Validation needs a run or trial variable.",
                    (*path, "variables", key),
                )
                continue
            code = lower(
                _validation_expression(value),
                variable.type,
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
    n_steps = sum(bounds) if document.fixed_steps is None else document.fixed_steps
    return DocumentTask(
        TaskInterface(tuple(scope.values()), tuple(declared_signals), tuple(links.items())),
        n_steps,
        period,
        tuple(epochs),
        tuple(fields),
        tuple(lengths),
        tuple(bounds),
        tuple(signals),
        tuple(rows),
        tuple(bindings),
    )
