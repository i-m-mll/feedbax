"""Discrete trajectory linearization of the authored graph transition."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import prod

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import DictKey, GetAttrKey, SequenceKey

from feedbax.builtins.runtime.channel import toggle_channel_noise
from feedbax.builtins.runtime.components import Noise
from feedbax.compiler.model_reset import ModelResetError
from feedbax.contracts.expression_values import Problem, ProblemCode
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.model_address import (
    ComponentPortAddress, FieldElement, InputBoundaryEnd, ModelAddress, OutputBoundaryEnd, PortElement, WireElement,
)
from feedbax.contracts.model_values import SignalLeafPath
from feedbax.execution.trial_execution import sample_trial
from feedbax.execution.trial_resolution import PreparedTrialExecution
from feedbax.objectives.document import ObjectiveExecutor
from feedbax.runtime._signal_schedule import ScopedPort, SignalScheduleError, resolve_signal_schedule
from feedbax.runtime.graph import Graph, GraphSignalEdits, GraphTraceRequest, _nested_cycle_key, signal_leaf_value
from feedbax.runtime.raw_state import StateLayoutError, bind_raw_state
from feedbax.tasks.document import PendingTrial, TaskParameters, Trial


@dataclass(frozen=True)
class StateCoordinates:
    """The flat coordinates belonging to one declared Model state or delay leaf."""

    key: str
    shape: tuple[int, ...]
    start: int
    stop: int


@dataclass(frozen=True)
class Linearization:
    """A perturbation model along a trajectory in the graph's actual step convention.

    The matrices have a leading time axis only when the requested samples differ.
    States are pre-step and outputs are observed during their corresponding step.
    """

    A: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    layout: tuple[StateCoordinates, ...]
    states: jax.Array
    outputs: jax.Array
    steps: tuple[int, ...]
    dt: float


@dataclass(frozen=True)
class QuadraticCost:
    """Stage blocks and terminal block of the compiled Objective."""

    Q: jax.Array
    R: jax.Array
    N: jax.Array
    Qf: jax.Array
    state_gradient: jax.Array
    command_gradient: jax.Array
    constant: jax.Array

    def evaluate(self, states: jax.Array, commands: jax.Array) -> jax.Array:
        x, terminal = states[:-1], states[-1]
        return (
            jnp.einsum("ti,tij,tj->", x, self.Q, x)
            + jnp.einsum("ti,tij,tj->", commands, self.R, commands)
            + 2 * jnp.einsum("ti,tij,tj->", x, self.N, commands)
            + terminal @ self.Qf @ terminal
            + jnp.vdot(states, self.state_gradient)
            + jnp.vdot(commands, self.command_gradient)
            + self.constant
        )


def quadratic_cost(
    objective: ObjectiveExecutor,
    task: Trial,
    observe: Callable[[jax.Array, jax.Array], Mapping[str, jax.Array]],
    *,
    state_width: int,
    command_width: int,
) -> QuadraticCost | tuple[Problem, ...]:
    """Lower the actual B3 evaluator on independent state and command samples.

    The observation function receives T+1 states and T commands and produces
    the retained signals at the evaluator's usual step alignment.
    """
    steps = objective.n_steps
    width = state_width + command_width
    key = jax.random.key(0)
    base = jnp.zeros((steps + 1, width), dtype=float)
    seeds = jnp.broadcast_to(jnp.eye(width, dtype=base.dtype)[:, None, :], (width, *base.shape))

    def score(values):
        states, commands = values[:, :state_width], values[:-1, state_width:]
        terms = objective.evaluate_trial(observe(states, commands), task, key, key)
        return sum(
            (jnp.where(term.covered, term.value, 0) for term in terms.values()),
            jnp.asarray(0.0, dtype=values.dtype),
        )

    scalar_and_gradient = jax.jit(jax.value_and_grad(score))
    constant, gradient = scalar_and_gradient(base)

    @jax.jit
    def half_blocks(directions):
        # For a stage-separable Objective each time block responds independently.
        # Sharing the basis across time avoids materializing a dense T² Hessian.
        hv = jax.vmap(lambda v: jax.jvp(jax.grad(score), (base,), (v,))[1])(directions)
        h = jnp.moveaxis(hv, 0, -1)
        return (h + h.swapaxes(-1, -2)) / 4

    blocks = half_blocks(seeds)
    tolerance = 128 * np.finfo(base.dtype).eps
    for index in range(3):
        probe = jax.random.normal(jax.random.fold_in(key, index), base.shape, dtype=base.dtype)
        value, slope = scalar_and_gradient(probe)
        change = jnp.einsum("tij,tj->ti", blocks, probe)
        predicted = constant + jnp.vdot(gradient + change, probe)
        if not (
            np.allclose(value, predicted, rtol=tolerance, atol=tolerance)
            and np.allclose(slope, gradient + 2 * change, rtol=tolerance, atol=tolerance)
        ):
            return (Problem(
                code=ProblemCode.STRUCTURE, path=("objective",),
                reason="The Objective is not a stage-separable quadratic in state and command.",
            ),)
    return QuadraticCost(
        blocks[:-1, :state_width, :state_width], blocks[:-1, state_width:, state_width:],
        blocks[:-1, :state_width, state_width:], blocks[-1, :state_width, :state_width],
        gradient[:, :state_width], gradient[:-1, state_width:], constant,
    )


class _Refused(ValueError):
    def __init__(self, reason: str, address: ModelAddress | None = None):
        super().__init__(reason)
        self.address = address

    def problem(self) -> Problem:
        return Problem(
            code=ProblemCode.STRUCTURE, reason=str(self),
            path=self.address.stored_path() if self.address is not None else (),
            model_address=self.address,
        )


@dataclass(frozen=True)
class _Signal:
    port: ScopedPort
    direction: str
    leaf: SignalLeafPath = ()


def _graph_at(spec: GraphSpec, scope: tuple[str, ...]) -> GraphSpec:
    for name in scope:
        child = (spec.subgraphs or {}).get(name)
        if not isinstance(child, GraphSpec):
            raise _Refused(f"No causal graph at {scope!r}.")
        spec = child
    return spec


def _scope_path(scope: tuple[str, ...]) -> tuple[str, ...]:
    return ("graph", *(token for name in scope for token in ("subgraphs", name)))


def _signal(model: PreparedTrialExecution, address: ModelAddress) -> _Signal:
    element, leaf = address.element, ()
    if isinstance(element, FieldElement) and isinstance(element.parent, PortElement):
        element, leaf = element.parent, element.path
    if not isinstance(element, PortElement):
        raise _Refused("Select a Model signal port or signal leaf.", address)
    end, scope = element.end, address.composites
    spec = _graph_at(model.resolution.graph, scope)
    if isinstance(end, ComponentPortAddress) and end.direction in {"input", "output"}:
        if end.component in spec.nodes:
            return _Signal(ScopedPort((*scope, end.component), end.port), end.direction, leaf)
    if isinstance(end, InputBoundaryEnd):
        if scope and end.port in spec.input_bindings:
            return _Signal(ScopedPort(scope, end.port), "input", leaf)
        prefix = _scope_path(scope)
        for name in spec.nodes:
            if model.lowered.origins.get((*prefix, "nodes", name)) == ModelAddress(composites=scope, element=element):
                return _Signal(ScopedPort((*scope, name), "output"), "output", leaf)
    if isinstance(end, OutputBoundaryEnd):
        if scope and end.port in spec.output_bindings:
            return _Signal(ScopedPort(scope, end.port), "output", leaf)
        resolved = model.signal_scope.leaves.get(address.to_text())
        name = (resolved.name if resolved is not None else end.port)
        if name in spec.output_bindings:
            component, port = spec.output_bindings[name]
            return _Signal(ScopedPort((*scope, component), port), "output", leaf)
    raise _Refused("The Model signal has no executable port.", address)


def _opened_port(model: PreparedTrialExecution, address: ModelAddress) -> ScopedPort:
    element, scope = address.element, address.composites
    if not isinstance(element, WireElement):
        raise _Refused("An opening must name a Model wire.", address)
    spec = _graph_at(model.resolution.graph, scope)
    for wire in spec.wires:
        if wire.id == element.id:
            return ScopedPort((*scope, wire.target_node), wire.target_port)
    raise _Refused("The wire has no independently driven input to open.", address)


def _leaf_path(path):
    return tuple(
        key.key if isinstance(key, DictKey) else key.name if isinstance(key, GetAttrKey) else key.idx
        for key in path
        if isinstance(key, (DictKey, GetAttrKey, SequenceKey))
    )


def _adjust(value, leaf, change):
    def requested(part):
        return part + jnp.asarray(change, dtype=part.dtype)

    if not leaf:
        return requested(value)
    return jax.tree.map_with_path(lambda path, part: requested(part) if _leaf_path(path) == leaf else part, value)


def _state_samples(graph: Graph, state, carry):
    """State-only reads are aliases of component storage, not independent dynamics."""
    result = dict(carry)
    for wire in graph.wires:
        initializer = wire.recurrent_initializer or {}
        if initializer.get("model_state_sample") is not True:
            continue
        node = graph.nodes[initializer["source_node"]]
        output = node.initial_outputs(node.state_view(state))[initializer["source_port"]]
        result[wire.target_node, wire.target_port] = output
    for name, node in graph.nodes.items():
        nested = _nested_cycle_key(name)
        if isinstance(node, Graph) and nested in result:
            result[nested] = _state_samples(node, state, result[nested])
    return result


def _state_layout(model, graph, state, carry):
    raw = bind_raw_state(graph, model.executable.resolved.graph)
    values = tuple(item for item in raw.values if not item.field.configuration)

    # Distinct leaf objects identify storage slots without depending on private
    # Equinox State keys or a second declaration of component/Delay layouts.
    leaves, structure = jax.tree.flatten((state, carry))
    tagged = [jnp.array(leaf, copy=True) for leaf in leaves]
    locations = {id(leaf): index for index, leaf in enumerate(tagged)}
    tagged_state, tagged_carry = structure.unflatten(tagged)
    parts, indexes, size = [], [], 0
    for item in values:
        value = item.read(tagged_state, tagged_carry)
        if not jnp.issubdtype(value.dtype, jnp.floating):
            continue
        if id(value) not in locations:
            raise _Refused(f"State {item.field.key!r} is not a stored coordinate.")
        width = value.size
        parts.append(StateCoordinates(item.field.key, value.shape, size, size + width))
        indexes.append((item, locations[id(value)]))
        size += width

    def capture(state, carry):
        if not indexes:
            return jnp.zeros((0,), dtype=float)
        return jnp.concatenate([item.read(state, carry).reshape(-1) for item, _ in indexes])

    def replace(state, carry, vector):
        leaves, structure = jax.tree.flatten((state, carry))
        for part, (_, index) in zip(parts, indexes, strict=True):
            leaves[index] = jnp.asarray(vector[part.start:part.stop].reshape(part.shape), dtype=leaves[index].dtype)
        state, carry = structure.unflatten(leaves)
        return state, _state_samples(graph, state, carry)

    return tuple(parts), capture, replace


def _concrete_signal(parameters, signal: _Signal, address: ModelAddress):
    for name in signal.port.node[:-1]:
        if name not in parameters.subgraphs:
            raise _Refused("The signal has no executable prototype.", address)
        parameters = parameters.subgraphs[name]
    name = signal.port.node[-1]
    ports = parameters.node_inputs if signal.direction == "input" else parameters.node_outputs
    key = name, signal.port.port
    try:
        value = signal_leaf_value(ports[key], signal.leaf)
    except (KeyError, IndexError, AttributeError, TypeError) as error:
        raise _Refused("The signal has no executable prototype.", address) from error
    if not hasattr(value, "dtype") or not jnp.issubdtype(value.dtype, jnp.floating):
        raise _Refused("A linearization channel needs one real array leaf.", address)
    return value


def _constant_or_series(matrices):
    if all(np.array_equal(matrices[0], other) for other in matrices[1:]):
        return matrices[0]
    return jnp.stack(matrices)


def linearize(
    model: PreparedTrialExecution,
    task: Trial | None = None,
    *,
    inputs: Sequence[ModelAddress] = (),
    outputs: Sequence[ModelAddress] = (),
    openings: Sequence[ModelAddress] = (),
    about: int | None = None,
) -> Linearization | tuple[Problem, ...]:
    """Differentiate the paired Model's step at a nominal trajectory or one of its steps.

    An input adds a perturbation to its signal. Opening a wire sets its target's
    driving value to zero before applying any perturbation at that target.
    """
    try:
        return _linearize(model, task, tuple(inputs), tuple(outputs), tuple(openings), about)
    except _Refused as error:
        return (error.problem(),)
    except ModelResetError as error:
        return (error.problem,)
    except SignalScheduleError as error:
        return (_Refused(str(error), model.lowered.source_address(error.authored_path)).problem(),)
    except StateLayoutError as error:
        return (Problem(code=ProblemCode.STRUCTURE, reason=str(error), path=tuple(error.path.strip("/").split("/"))),)


def _linearize(model, task, inputs, outputs, openings, about):
    graph = toggle_channel_noise(model.executable.graph, False)
    key = jax.random.key(0)
    execution = model.execution
    task = sample_trial(execution.task, TaskParameters(), key, key) if task is None else task
    if isinstance(task, PendingTrial):
        raise _Refused("Linearization needs a materialized, model-independent trial.")
    n_steps = execution.task.n_steps
    if not n_steps or about is not None and not 0 <= about < n_steps:
        raise _Refused("The selected step is outside the nominal trial.")
    active = tuple(range(n_steps)) if about is None else (about,)
    external = {port: task.inputs[signal] for port, signal in execution.input_bindings}
    reset = execution.reset.prepare(
        graph, task.inits, key=key, inputs=external, input_series=True, reset_values=task.variables,
    )
    state, carry = reset.state, reset.cycle_init
    layout, capture, restore = _state_layout(model, graph, state, carry)

    schedule = resolve_signal_schedule(model.resolution.graph)
    input_aliases, output_aliases = dict(schedule.input_aliases), dict(schedule.output_aliases)
    scoped = graph._scoped_schedule is not None

    def actual(port, direction):
        aliases = input_aliases if direction == "input" else output_aliases
        return aliases.get(port, port) if scoped else port

    ports_in, ports_out = tuple(_signal(model, item) for item in inputs), tuple(_signal(model, item) for item in outputs)
    opened = {actual(_opened_port(model, item), "input") for item in openings}
    drives = {"input": {}, "output": {}}
    offset = 0
    parameters = model.resolution.parameters
    reported = set()
    for address, signal in zip(inputs, ports_in, strict=True):
        prototype = _concrete_signal(parameters, signal, address)
        port = actual(signal.port, signal.direction)
        identity = port, signal.direction, signal.leaf
        if identity in reported:
            raise _Refused("The same input signal is selected more than once.", address)
        reported.add(identity)
        width = prod(prototype.shape)
        drives[signal.direction].setdefault(port, []).append((signal.leaf, offset, width, prototype.shape))
        offset += width
    traces = []
    for index, (address, signal) in enumerate(zip(outputs, ports_out, strict=True)):
        _concrete_signal(parameters, signal, address)
        port = signal.port
        traces.append(GraphTraceRequest("port", str(index), node=port.node[-1], port=port.port, scope=port.node[:-1], leaf=signal.leaf))
    traces = tuple(traces)
    noise = {ScopedPort(path, "output") for path in schedule.order if isinstance(graph.get_node(path), Noise)}
    zero = jnp.zeros((offset,), dtype=float)

    def transition(base_state, base_carry, step_inputs, t, x, u):
        state, carry = restore(base_state, base_carry, x)

        def apply(ports, cleared, port, value):
            if port in cleared:
                value = jax.tree.map(jnp.zeros_like, value)
            for leaf, start, size, shape in ports.get(port, ()):
                value = _adjust(value, leaf, u[start:start + size].reshape(shape))
            return value

        _, state, carry, sampled = graph.step_with_trace(
            step_inputs, state, carry, key=key, t=t, trace=traces,
            signal_edits=GraphSignalEdits(
                lambda port, value: apply(drives["input"], opened, port, value),
                lambda port, value: apply(drives["output"], noise, port, value),
            ),
        )
        y = jnp.concatenate([sampled[trace.selector].reshape(-1) for trace in traces]) if traces else jnp.zeros((0,))
        return (capture(state, carry), y), (state, carry)

    def differentiate(state, carry, step_inputs, t, x, u):
        def step(x, u):
            return transition(state, carry, step_inputs, t, x, u)[0]

        return jax.jacrev(step, argnums=(0, 1))(x, u)

    differentiate = jax.jit(differentiate)
    advance = jax.jit(transition)
    As, Bs, Cs, Ds, xs, ys = [], [], [], [], [], []
    for step in range(active[-1] + 1):
        x = capture(state, carry)
        step_inputs = jax.tree.map(lambda value: value[step], external)
        t = jnp.asarray(step)
        if step in active:
            ((a, b), (c, d)) = differentiate(state, carry, step_inputs, t, x, zero)
            for series, matrix in zip((As, Bs, Cs, Ds), (a, b, c, d), strict=True):
                series.append(matrix)
            xs.append(x)
        (_, y), (state, carry) = advance(state, carry, step_inputs, t, x, zero)
        if step in active:
            ys.append(y)

    return Linearization(
        *map(_constant_or_series, (As, Bs, Cs, Ds)),
        layout, jnp.stack(xs), jnp.stack(ys), active, model.execution.task.period,
    )
