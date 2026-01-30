"""Basic components for math, sources, and signal processing."""

from __future__ import annotations

from typing import Callable, Sequence

import equinox as eqx
from equinox import Module, field
from equinox.nn import State, StateIndex
import jax
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import PRNGKeyArray, PyTree

from feedbax.graph import Component


def _activation_fn(name: str) -> Callable:
    mapping = {
        "relu": jax.nn.relu,
        "tanh": jax.nn.tanh,
        "identity": lambda x: x,
    }
    return mapping.get(name, jax.nn.relu)


class Gain(Component):
    """Multiply input by a constant gain."""

    input_ports = ("input",)
    output_ports = ("output",)

    gain: float

    def __init__(self, gain: float = 1.0):
        self.gain = float(gain)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        output = jt.map(lambda x: x * self.gain, inputs["input"])
        return {"output": output}, state


class Sum(Component):
    """Sum two inputs."""

    input_ports = ("a", "b")
    output_ports = ("output",)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        output = jt.map(lambda a, b: a + b, inputs["a"], inputs["b"])
        return {"output": output}, state


class Multiply(Component):
    """Element-wise product of two inputs."""

    input_ports = ("a", "b")
    output_ports = ("output",)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        output = jt.map(lambda a, b: a * b, inputs["a"], inputs["b"])
        return {"output": output}, state


class Saturation(Component):
    """Clamp input to a min/max range."""

    input_ports = ("input",)
    output_ports = ("output",)

    min_val: float
    max_val: float

    def __init__(self, min_val: float = -1.0, max_val: float = 1.0):
        self.min_val = float(min_val)
        self.max_val = float(max_val)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        output = jt.map(lambda x: jnp.clip(x, self.min_val, self.max_val), inputs["input"])
        return {"output": output}, state


class Constant(Component):
    """Constant value output."""

    output_ports = ("output",)

    value: PyTree = field(static=True)

    def __init__(self, value: PyTree = 0.0):
        self.value = jt.map(jnp.asarray, value)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        return {"output": self.value}, state


class _StepSource(Component):
    """Base class for sources driven by a step counter."""

    step_index: StateIndex
    _initial_state: int = field(static=True)
    dt: float

    def __init__(self, dt: float = 0.01):
        self.dt = float(dt)
        self._initial_state = 0
        self.step_index = StateIndex(self._initial_state)

    def _step_time(self, state: State) -> tuple[float, State]:
        step = state.get(self.step_index)
        next_state = state.set(self.step_index, step + 1)
        return float(step) * self.dt, next_state


class Ramp(_StepSource):
    """Linear ramp over time."""

    output_ports = ("output",)

    slope: PyTree = field(static=True)
    intercept: PyTree = field(static=True)

    def __init__(self, slope: PyTree = 1.0, intercept: PyTree = 0.0, dt: float = 0.01):
        super().__init__(dt=dt)
        self.slope = jt.map(jnp.asarray, slope)
        self.intercept = jt.map(jnp.asarray, intercept)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        t, state = self._step_time(state)
        output = jt.map(lambda s, b: b + s * t, self.slope, self.intercept)
        return {"output": output}, state


class Sine(_StepSource):
    """Sinusoidal signal."""

    output_ports = ("output",)

    amplitude: PyTree = field(static=True)
    frequency: float
    phase: float
    offset: PyTree = field(static=True)

    def __init__(
        self,
        amplitude: PyTree = 1.0,
        frequency: float = 1.0,
        phase: float = 0.0,
        offset: PyTree = 0.0,
        dt: float = 0.01,
    ):
        super().__init__(dt=dt)
        self.amplitude = jt.map(jnp.asarray, amplitude)
        self.frequency = float(frequency)
        self.phase = float(phase)
        self.offset = jt.map(jnp.asarray, offset)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        t, state = self._step_time(state)
        omega = 2 * jnp.pi * self.frequency
        output = jt.map(
            lambda a, b: b + a * jnp.sin(omega * t + self.phase),
            self.amplitude,
            self.offset,
        )
        return {"output": output}, state


class Pulse(_StepSource):
    """Pulse/square wave."""

    output_ports = ("output",)

    amplitude: PyTree = field(static=True)
    period: float
    duty_cycle: float
    offset: PyTree = field(static=True)

    def __init__(
        self,
        amplitude: PyTree = 1.0,
        period: float = 1.0,
        duty_cycle: float = 0.5,
        offset: PyTree = 0.0,
        dt: float = 0.01,
    ):
        super().__init__(dt=dt)
        self.amplitude = jt.map(jnp.asarray, amplitude)
        self.period = float(period)
        self.duty_cycle = float(duty_cycle)
        self.offset = jt.map(jnp.asarray, offset)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        t, state = self._step_time(state)
        phase = (t / self.period) % 1.0 if self.period > 0 else 0.0
        gate = jnp.where(phase < self.duty_cycle, 1.0, 0.0)
        output = jt.map(lambda a, b: b + a * gate, self.amplitude, self.offset)
        return {"output": output}, state


class Noise(Component):
    """Random noise source."""

    output_ports = ("output",)

    mean: float
    std: float
    shape: tuple[int, ...] = field(static=True)

    def __init__(self, mean: float = 0.0, std: float = 1.0, shape: Sequence[int] = (1,)):
        self.mean = float(mean)
        self.std = float(std)
        self.shape = tuple(int(x) for x in shape)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        sample = jax.random.normal(key, self.shape) * self.std + self.mean
        return {"output": sample}, state


class Spring(Component):
    """Linear spring: F = k * displacement."""

    input_ports = ("displacement",)
    output_ports = ("force",)

    stiffness: float

    def __init__(self, stiffness: float = 1.0):
        self.stiffness = float(stiffness)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        output = jt.map(lambda x: self.stiffness * x, inputs["displacement"])
        return {"force": output}, state


class Damper(Component):
    """Viscous damper: F = b * velocity."""

    input_ports = ("velocity",)
    output_ports = ("force",)

    damping: float

    def __init__(self, damping: float = 1.0):
        self.damping = float(damping)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        output = jt.map(lambda x: self.damping * x, inputs["velocity"])
        return {"force": output}, state


class DelayLine(Component):
    """Discrete delay buffer."""

    input_ports = ("input",)
    output_ports = ("output",)

    delay: int
    init_value: float
    state_index: StateIndex
    _initial_state: tuple[PyTree, tuple[PyTree, ...]] = field(static=True)

    def __init__(self, delay: int = 1, init_value: float = 0.0, input_proto: PyTree | None = None):
        self.delay = int(delay)
        self.init_value = float(init_value)
        if input_proto is None:
            input_proto = jnp.zeros(1)
        output = jt.map(lambda x: jnp.full_like(x, self.init_value), input_proto)
        queue = self.delay * (output,)
        self._initial_state = (output, queue)
        self.state_index = StateIndex(self._initial_state)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        output, queue = state.get(self.state_index)
        if self.delay > 0:
            next_output = queue[0]
            next_queue = queue[1:] + (inputs["input"],)
        else:
            next_output = inputs["input"]
            next_queue = queue
        state = state.set(self.state_index, (next_output, next_queue))
        return {"output": next_output}, state


class MLP(Component):
    """Multi-layer perceptron."""

    input_ports = ("input",)
    output_ports = ("output",)

    linears: tuple[eqx.nn.Linear, ...]
    activation: Callable = field(static=True)
    final_activation: Callable = field(static=True)
    input_size: int = field(static=True)
    output_size: int = field(static=True)
    hidden_sizes: tuple[int, ...] = field(static=True)
    activation_name: str = field(static=True)
    final_activation_name: str = field(static=True)

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: Sequence[int] = (64,),
        activation: str = "relu",
        final_activation: str = "identity",
        *,
        key: PRNGKeyArray,
    ):
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)
        self.activation_name = activation
        self.final_activation_name = final_activation
        sizes = [self.input_size, *self.hidden_sizes, self.output_size]
        keys = jax.random.split(key, len(sizes) - 1)
        self.linears = tuple(
            eqx.nn.Linear(sizes[i], sizes[i + 1], key=keys[i])
            for i in range(len(sizes) - 1)
        )
        self.activation = _activation_fn(self.activation_name)
        self.final_activation = _activation_fn(self.final_activation_name)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        x = inputs["input"]
        for idx, layer in enumerate(self.linears):
            x = layer(x)
            if idx < len(self.linears) - 1:
                x = self.activation(x)
            else:
                x = self.final_activation(x)
        return {"output": x}, state


class GRU(Component):
    """Standalone GRU cell."""

    input_ports = ("input", "hidden")
    output_ports = ("output", "hidden")

    cell: eqx.nn.GRUCell
    input_size: int = field(static=True)
    hidden_size: int = field(static=True)

    def __init__(self, input_size: int, hidden_size: int, *, key: PRNGKeyArray):
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.cell = eqx.nn.GRUCell(self.input_size, self.hidden_size, key=key)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        hidden = inputs["hidden"]
        new_hidden = self.cell(inputs["input"], hidden)
        return {"output": new_hidden, "hidden": new_hidden}, state


class LSTM(Component):
    """Standalone LSTM cell."""

    input_ports = ("input", "hidden", "cell")
    output_ports = ("output", "hidden", "cell")

    cell: eqx.nn.LSTMCell
    input_size: int = field(static=True)
    hidden_size: int = field(static=True)

    def __init__(self, input_size: int, hidden_size: int, *, key: PRNGKeyArray):
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.cell = eqx.nn.LSTMCell(self.input_size, self.hidden_size, key=key)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        hidden = inputs["hidden"]
        cell_state = inputs["cell"]
        new_hidden, new_cell = self.cell(inputs["input"], hidden, cell_state)
        return {"output": new_hidden, "hidden": new_hidden, "cell": new_cell}, state
