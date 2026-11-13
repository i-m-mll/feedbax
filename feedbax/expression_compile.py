"""Lower checked expressions to JAX closures and explicit phase values."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
from math import isfinite
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as random

from feedbax.contracts.expression_units import UNITS
from feedbax.contracts.expression_values import (
    CallValue,
    DrawValue,
    Expression,
    Level,
    LiteralValue,
    NameValue,
    NodePath,
    Problem,
    ProblemCode,
    Site,
    ValueBase,
)
from feedbax.expression_builtins import CallContext
from feedbax.expression_check import Checked, check
from feedbax.expression_registry import Evaluation, FunctionRegistry, StateReset
from feedbax.runtime._graph_key_schedule import derive_key


_DTYPES = {ValueBase.REAL: float, ValueBase.INTEGER: int, ValueBase.BOOLEAN: bool}


class Inputs(NamedTuple):
    parameters: tuple
    values: Mapping[str, Any]
    run_key: jax.Array
    trial_key: jax.Array
    repeat: int | jax.Array | None = None
    step: int | jax.Array = 0
    epoch_entry: bool | jax.Array = False
    defined: Mapping[str, Any] | None = None


class State(NamedTuple):
    values: dict[str, Any]
    slots: dict[str, Any]


@dataclass(frozen=True)
class Compiled:
    checked: Checked
    parameters: tuple[jax.Array, ...]
    initial_state: State
    evaluate_run: Callable[[Inputs], dict[str, Any]]
    evaluate_trial: Callable[[Inputs, dict[str, Any]], dict[str, Any]]
    evaluate_step: Callable[[Inputs, State], tuple[Evaluation, State]]

    def with_trial(self, inputs: Inputs, run_values: dict[str, Any]) -> State:
        return State(self.evaluate_trial(inputs, run_values), self.initial_state.slots)


def _path_key(path: NodePath) -> str:
    return json.dumps(path, ensure_ascii=False, separators=(",", ":"))


def compile_expression(
    tree: Expression,
    site: Site,
    functions: FunctionRegistry,
    *,
    field_address: tuple[str, ...],
    period_seconds: float | None = None,
    reset_on_epoch: bool = False,
    path: NodePath = (),
) -> Compiled | tuple[Problem, ...]:
    """Compile with dynamic literal operands, independently of the authored spelling."""
    checked = check(tree, site, functions, path=path)
    if not isinstance(checked, Checked):
        return checked
    if period_seconds is not None and (not isfinite(period_seconds) or period_seconds <= 0):
        return (
            Problem(
                code=ProblemCode.STRUCTURE, reason="The model period must be positive.", path=path
            ),
        )
    if not field_address or any(not part for part in field_address):
        return (
            Problem(code=ProblemCode.STRUCTURE, reason="A field address is required.", path=path),
        )

    parameters: list[jax.Array] = []
    slots: dict[str, Any] = {}
    problems: list[Problem] = []
    run_evaluators: list[tuple[str, Callable]] = []
    trial_evaluators: list[tuple[str, Callable]] = []
    names = {value.name: value.id for value in site.names}
    draw_ordinal = 0
    random_call_ordinal = 0

    def visit(node: Expression, at: NodePath) -> Callable:
        nonlocal draw_ordinal, random_call_ordinal
        node_type = checked.nodes[at]
        slot = _path_key(at)
        if isinstance(node, LiteralValue):
            value = node.value
            if node.unit == "st":
                if period_seconds is None:
                    problems.append(
                        Problem(
                            code=ProblemCode.OUT_OF_SCOPE,
                            reason="A value in steps needs a model period.",
                            path=(*path, *at),
                        )
                    )
                else:
                    value *= period_seconds
            elif node.unit is not None and (scale := UNITS[node.unit].scale) != 1:
                value *= scale
            dtype = _DTYPES[node_type.type.base]
            if dtype is int and (int(value) != value or not -(2**31) <= value < 2**31):
                problems.append(
                    Problem(
                        code=ProblemCode.TYPE_MISMATCH,
                        reason="Integer literal is outside the supported range.",
                        path=(*path, *at),
                    )
                )
                value = 0
            index = len(parameters)
            parameters.append(jnp.asarray(value, dtype=dtype))

            def evaluate(inputs, _values, _slots):
                return Evaluation(inputs.parameters[index], True)
        elif isinstance(node, NameValue):

            def evaluate(inputs, _values, _slots):
                defined = inputs.defined.get(node.reference, True) if inputs.defined else True
                return Evaluation(inputs.values[node.reference], defined)
        else:
            key = node.family if isinstance(node, DrawValue) else node.function
            function = functions.resolve(key)
            assert function is not None
            if at in checked.calls:
                function, node = checked.calls[at]
            if function.info.requires_period and period_seconds is None:
                problems.append(
                    Problem(
                        code=ProblemCode.OUT_OF_SCOPE,
                        reason=f"{key} needs a model period.",
                        path=(*path, *at),
                    )
                )
            stream = None
            if isinstance(node, DrawValue):
                stream = ("draw", str(draw_ordinal))
                draw_ordinal += 1
            elif function.info.random:
                stream = ("call", str(random_call_ordinal))
                random_call_ordinal += 1
            args = tuple(visit(child, (*at, "args", i)) for i, child in enumerate(node.args))
            kwargs = {
                name: visit(child, (*at, "kwargs", name))
                for name, child in (
                    sorted(node.kwargs.items()) if isinstance(node, CallValue) else ()
                )
            }
            implicit = {
                name: names[name]
                for name in function.info.implicit_names
                if name != "T" or "over" not in kwargs
            }
            implementation = function.evaluate
            stateful = function.initial_state is not None
            if function.initial_state is not None:
                slots[slot] = function.initial_state(node_type.type)
            resets = stateful and reset_on_epoch and function.info.reset == StateReset.EPOCH
            initial = slots.get(slot)

            def evaluate(inputs, values, state):
                source = inputs.run_key if node_type.level == Level.RUN else inputs.trial_key
                key_value = None
                if stream is not None:
                    key_value = derive_key(
                        source, "expression", *field_address, *stream, index=inputs.repeat
                    )
                    if not isinstance(node, DrawValue) and node_type.level == Level.STEP:
                        key_value = random.fold_in(key_value, inputs.step)
                previous = state.get(slot)
                if resets:
                    previous = jax.tree.map(
                        lambda start, old: jnp.where(inputs.epoch_entry, start, old),
                        initial,
                        previous,
                    )
                positional = tuple(argument(inputs, values, state) for argument in args)
                named = {name: arg(inputs, values, state) for name, arg in kwargs.items()}
                defined = tuple(arg.defined for arg in positional)
                context = CallContext(
                    {name: inputs.values[ref] for name, ref in implicit.items()},
                    key_value,
                    previous,
                    period_seconds,
                    defined,
                    inputs.step,
                )
                result = implementation(
                    context,
                    *(arg.value for arg in positional),
                    **{name: arg.value for name, arg in named.items()},
                )
                if stateful:
                    result, state[slot] = result
                valid = True
                for arg in named.values():
                    valid = valid & arg.defined
                if inputs.defined:
                    for ref in implicit.values():
                        valid = valid & inputs.defined.get(ref, True)
                if isinstance(result, Evaluation):
                    return Evaluation(result.value, valid & result.defined)
                for available in defined:
                    valid = valid & available
                return Evaluation(result, valid)

        if isinstance(node, (LiteralValue, NameValue)) or node_type.level == Level.STEP:
            return evaluate
        phase = run_evaluators if node_type.level == Level.RUN else trial_evaluators
        phase.append((slot, evaluate))
        return lambda _inputs, values, _state: values[slot]

    evaluate = visit(tree, ())
    if problems:
        return tuple(problems)

    def run(inputs: Inputs) -> dict[str, Any]:
        values = {}
        for slot, evaluate_node in run_evaluators:
            values[slot] = evaluate_node(inputs, values, {})
        return values

    def trial(inputs: Inputs, run_values: dict[str, Any]) -> dict[str, Any]:
        values = dict(run_values)
        for slot, evaluate_node in trial_evaluators:
            values[slot] = evaluate_node(inputs, values, {})
        return values

    def step(inputs: Inputs, state: State):
        next_slots = dict(state.slots)
        value, defined = evaluate(inputs, state.values, next_slots)
        return Evaluation(jnp.asarray(value, dtype=_DTYPES[site.expected.base]), defined), State(
            state.values, next_slots
        )

    return Compiled(checked, tuple(parameters), State({}, slots), run, trial, step)
