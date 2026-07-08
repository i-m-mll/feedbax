"""Basic components for math, sources, and signal processing."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import equinox as eqx
from equinox import Module, field
from equinox.nn import State, StateIndex
import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Array, PRNGKeyArray, PyTree

from feedbax.runtime.graph import Component


def identity_activation(x: Any) -> Any:
    return x


NAMED_ACTIVATIONS: dict[str, Callable] = {
    "relu": jax.nn.relu,
    "tanh": jax.nn.tanh,
    "sigmoid": jax.nn.sigmoid,
    "softmax": jax.nn.softmax,
    "identity": identity_activation,
}


def _callable_name(activation: Callable) -> str:
    return getattr(activation, "__name__", activation.__class__.__name__)


def resolve_activation(activation: str | Callable) -> tuple[str, Callable]:
    if isinstance(activation, str):
        return activation, NAMED_ACTIVATIONS.get(activation, jax.nn.relu)
    if callable(activation):
        return _callable_name(activation), activation
    raise TypeError("activation must be a string name or callable")


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


class Input(Component):
    """Pass a graph input through a source-like node."""

    input_ports: tuple[str, ...] = field(static=True)
    output_ports: tuple[str, ...] = field(static=True)
    output_port: str = field(static=True)

    def __init__(self, output_port: str = "output"):
        self.input_ports = ("input",)
        self.output_port = str(output_port)
        self.output_ports = (self.output_port,)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        return {self.output_port: inputs["input"]}, state


class Sum(Component):
    """Sum present inputs."""

    input_ports = ("a", "b", "c", "d")
    output_ports = ("output",)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        values = [inputs[port] for port in self.input_ports if port in inputs]
        if not values:
            raise ValueError("Sum requires at least one input")
        output = values[0]
        for value in values[1:]:
            output = jt.map(lambda a, b: a + b, output, value)
        return {"output": output}, state


class Subtract(Component):
    """Subtract input ``b`` from input ``a``."""

    input_ports = ("a", "b")
    output_ports = ("out",)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        output = jt.map(lambda a, b: a - b, inputs["a"], inputs["b"])
        return {"out": output}, state


class Multiply(Component):
    """Element-wise product of two inputs."""

    input_ports = ("a", "b")
    output_ports = ("output",)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        output = jt.map(lambda a, b: a * b, inputs["a"], inputs["b"])
        return {"output": output}, state


class Reshape(Component):
    """Reshape an array input to a configured shape."""

    input_ports = ("input",)
    output_ports = ("output",)

    shape: tuple[int, ...] = field(static=True)

    def __init__(self, shape: Sequence[int] = (1, 1)):
        parsed = tuple(int(dim) for dim in shape)
        if not parsed:
            raise ValueError("Reshape shape must contain at least one dimension")
        self.shape = parsed

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        return {"output": jnp.reshape(inputs["input"], self.shape)}, state


class MatMul(Component):
    """Matrix multiply inputs ``a`` and ``b``."""

    input_ports = ("a", "b")
    output_ports = ("out",)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        return {"out": jnp.matmul(inputs["a"], inputs["b"])}, state


class Scale(Component):
    """Multiply input by a scalar or broadcastable scale."""

    input_ports = ("input",)
    output_ports = ("output",)

    scale: PyTree

    def __init__(self, scale: PyTree = 1.0):
        self.scale = jnp.asarray(scale)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        scale = jnp.asarray(self.scale)
        output = jt.map(lambda x: jnp.asarray(x) * scale, inputs["input"])
        return {"output": output}, state


class Sigmoid(Component):
    """Apply a logistic sigmoid elementwise."""

    input_ports = ("input",)
    output_ports = ("output",)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        return {"output": jt.map(jax.nn.sigmoid, inputs["input"])}, state


def _affine_param(value: PyTree, shape: tuple[int, ...], name: str) -> PyTree:
    array = jnp.asarray(value)
    try:
        jnp.broadcast_shapes(array.shape, shape)
    except ValueError as exc:
        raise ValueError(
            f"ElementwiseAffineModulator {name} shape {array.shape} cannot broadcast "
            f"to signal_shape {shape}"
        ) from exc
    return jnp.broadcast_to(array, shape)


class ElementwiseAffineModulator(Component):
    """Apply per-element affine modulation to a signal.

    Computes ``signal * (baseline + scale * modulator) + bias * modulator``.
    By default ``scale`` and ``bias`` come from the trainable ``gain`` and
    ``bias`` fields, but either may be provided as an explicit input port.
    ``modulator`` may be scalar or broadcastable to the configured signal shape.
    """

    input_ports = ("signal", "modulator", "scale", "bias")
    output_ports = ("output",)

    baseline: Array
    gain: Array
    bias: Array
    signal_shape: tuple[int, ...] = field(static=True)

    def __init__(
        self,
        signal_shape: Sequence[int],
        baseline: PyTree = 1.0,
        gain_init: PyTree = 0.0,
        bias_init: PyTree = 0.0,
    ):
        shape = tuple(int(dim) for dim in signal_shape)
        if not shape:
            raise ValueError("ElementwiseAffineModulator signal_shape must be non-empty")
        if any(dim <= 0 for dim in shape):
            raise ValueError("ElementwiseAffineModulator signal_shape dimensions must be positive")
        self.signal_shape = shape
        self.baseline = _affine_param(baseline, shape, "baseline")
        self.gain = _affine_param(gain_init, shape, "gain_init")
        self.bias = _affine_param(bias_init, shape, "bias_init")

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        signal = jnp.asarray(inputs["signal"])
        modulator = jnp.asarray(inputs["modulator"])
        scale = jnp.asarray(inputs.get("scale", self.gain))
        bias = jnp.asarray(inputs.get("bias", self.bias))
        if signal.shape != self.signal_shape:
            raise ValueError(
                "ElementwiseAffineModulator signal shape "
                f"{signal.shape} does not match configured signal_shape {self.signal_shape}"
            )
        try:
            jnp.broadcast_shapes(signal.shape, modulator.shape, scale.shape, bias.shape)
        except ValueError as exc:
            raise ValueError(
                "ElementwiseAffineModulator input shapes cannot broadcast to signal shape "
                f"{signal.shape}: modulator={modulator.shape}, scale={scale.shape}, "
                f"bias={bias.shape}"
            ) from exc
        output = signal * (self.baseline + scale * modulator) + bias * modulator
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


class Ravel(Component):
    """Flatten a PyTree input to a vector."""

    input_ports = ("input",)
    output_ports = ("output",)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        output, _ = ravel_pytree(inputs["input"])
        return {"output": output}, state


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
    input_proto: PyTree
    state_index: StateIndex
    _initial_state: tuple[PyTree, tuple[PyTree, ...]] = field(static=True)

    def __init__(self, delay: int = 1, init_value: float = 0.0, input_proto: PyTree | None = None):
        self.delay = int(delay)
        self.init_value = float(init_value)
        if input_proto is None:
            input_proto = jnp.zeros(1)
        self.input_proto = input_proto
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


class Linear(Component):
    """Linear layer."""

    input_ports = ("input",)
    output_ports = ("output",)

    layer: eqx.nn.Linear
    activation: Callable = field(static=True)
    activation_name: str = field(static=True)
    input_size: int = field(static=True)
    output_size: int = field(static=True)
    use_bias: bool = field(static=True)
    dtype: Any = field(static=True)

    def __init__(
        self,
        input_size: int,
        output_size: int,
        use_bias: bool = True,
        activation: str | Callable = "identity",
        dtype: Any = jnp.float32,
        *,
        key: PRNGKeyArray,
    ):
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.use_bias = bool(use_bias)
        self.dtype = dtype
        self.activation_name, self.activation = resolve_activation(activation)
        self.layer = eqx.nn.Linear(
            self.input_size,
            self.output_size,
            use_bias=self.use_bias,
            dtype=dtype,
            key=key,
        )

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        output = self.activation(self.layer(inputs["input"]))
        return {"output": output}, state


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
    dtype: Any = field(static=True)

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: Sequence[int] = (64,),
        activation: str | Callable = "relu",
        final_activation: str | Callable = "identity",
        dtype: Any = jnp.float32,
        *,
        key: PRNGKeyArray,
    ):
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)
        self.dtype = dtype
        self.activation_name, self.activation = resolve_activation(activation)
        self.final_activation_name, self.final_activation = resolve_activation(final_activation)
        sizes = [self.input_size, *self.hidden_sizes, self.output_size]
        keys = jax.random.split(key, len(sizes) - 1)
        self.linears = tuple(
            eqx.nn.Linear(sizes[i], sizes[i + 1], dtype=dtype, key=keys[i])
            for i in range(len(sizes) - 1)
        )
    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        x = inputs["input"]
        for idx, layer in enumerate(self.linears):
            x = layer(x)
            if idx < len(self.linears) - 1:
                x = self.activation(x)
            else:
                x = self.final_activation(x)
        return {"output": x}, state


class GRUState(Module):
    """State view for a leaf GRU cell."""

    hidden: Array
    output: Array


class GRU(Component):
    """Standalone GRU cell."""

    input_ports = ("input", "hidden")
    output_ports = ("output", "hidden")

    cell: eqx.nn.GRUCell
    input_size: int = field(static=True)
    hidden_size: int = field(static=True)
    dtype: Any = field(static=True)
    state_index: StateIndex
    _initial_state: GRUState = field(static=True)

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dtype: Any = jnp.float32,
        *,
        key: PRNGKeyArray,
    ):
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.dtype = dtype
        self.cell = eqx.nn.GRUCell(self.input_size, self.hidden_size, dtype=dtype, key=key)
        hidden = jnp.zeros(self.hidden_size, dtype=dtype)
        self._initial_state = GRUState(hidden=hidden, output=hidden)
        self.state_index = StateIndex(self._initial_state)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        current_state: GRUState = state.get(self.state_index)
        hidden = inputs.get("hidden", current_state.hidden)
        new_hidden = self.cell(inputs["input"], hidden)
        state = state.set(self.state_index, GRUState(hidden=new_hidden, output=new_hidden))
        return {"output": new_hidden, "hidden": new_hidden}, state


class LSTMState(Module):
    """State view for a leaf LSTM cell."""

    hidden: Array
    cell: Array
    output: Array


class LSTM(Component):
    """Standalone LSTM cell."""

    input_ports = ("input", "hidden", "cell")
    output_ports = ("output", "hidden", "cell")

    cell: eqx.nn.LSTMCell
    input_size: int = field(static=True)
    hidden_size: int = field(static=True)
    dtype: Any = field(static=True)
    state_index: StateIndex
    _initial_state: LSTMState = field(static=True)

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dtype: Any = jnp.float32,
        *,
        key: PRNGKeyArray,
    ):
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.dtype = dtype
        self.cell = eqx.nn.LSTMCell(self.input_size, self.hidden_size, dtype=dtype, key=key)
        hidden = jnp.zeros(self.hidden_size, dtype=dtype)
        cell = jnp.zeros(self.hidden_size, dtype=dtype)
        self._initial_state = LSTMState(hidden=hidden, cell=cell, output=hidden)
        self.state_index = StateIndex(self._initial_state)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        current_state: LSTMState = state.get(self.state_index)
        hidden = inputs.get("hidden", current_state.hidden)
        cell_state = inputs.get("cell", current_state.cell)
        new_hidden, new_cell = self.cell(inputs["input"], (hidden, cell_state), key=key)
        state = state.set(
            self.state_index,
            LSTMState(hidden=new_hidden, cell=new_cell, output=new_hidden),
        )
        return {"output": new_hidden, "hidden": new_hidden, "cell": new_cell}, state


class Mux(Component):
    """Concatenate multiple inputs into a single vector.

    Creates ``n_inputs`` input ports named ``in_0``, ``in_1``, etc.,
    and concatenates them along the last axis.

    Args:
        n_inputs: Number of input ports to create.
    """

    output_ports = ("output",)

    n_inputs: int = field(static=True)
    input_ports: tuple[str, ...]

    def __init__(self, n_inputs: int = 2):
        self.n_inputs = int(n_inputs)
        self.input_ports = tuple(f"in_{i}" for i in range(self.n_inputs))

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        parts = []
        for i in range(self.n_inputs):
            port = f"in_{i}"
            if port not in inputs:
                continue
            flat, _ = ravel_pytree(inputs[port])
            parts.append(jnp.atleast_1d(flat))
        if not parts:
            raise ValueError("Mux requires at least one connected input")
        output = jnp.concatenate(parts, axis=-1)
        return {"output": output}, state


class Demux(Component):
    """Split a vector into multiple outputs by size.

    Creates ``len(sizes)`` output ports named ``out_0``, ``out_1``, etc.

    Args:
        sizes: Tuple of sizes for each output slice along the last axis.
    """

    input_ports = ("input",)

    sizes: tuple[int, ...] = field(static=True)
    output_ports: tuple[str, ...]

    def __init__(self, sizes: Sequence[int]):
        self.sizes = tuple(int(s) for s in sizes)
        if not self.sizes:
            raise ValueError("Demux sizes must contain at least one output size")
        if any(size <= 0 for size in self.sizes):
            raise ValueError("Demux sizes must be positive integers")
        self.output_ports = tuple(f"out_{i}" for i in range(len(self.sizes)))

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        x = inputs["input"]
        if x.shape[-1] != sum(self.sizes):
            raise ValueError(
                f"Demux input final dimension {x.shape[-1]} does not match "
                f"sum(sizes) {sum(self.sizes)}"
            )
        outputs: dict[str, PyTree] = {}
        start = 0
        for i, sz in enumerate(self.sizes):
            outputs[f"out_{i}"] = x[..., start : start + sz]
            start += sz
        return outputs, state


class Switch(Component):
    """Route signal based on a threshold condition.

    When ``condition > threshold``, outputs ``true_input``; otherwise
    outputs ``false_input``.

    Args:
        threshold: Comparison threshold for routing.
    """

    input_ports = ("condition", "true_input", "false_input")
    output_ports = ("output",)

    threshold: float

    def __init__(self, threshold: float = 0.0):
        self.threshold = float(threshold)

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        cond = inputs["condition"]
        output = jnp.where(
            cond > self.threshold,
            inputs["true_input"],
            inputs["false_input"],
        )
        return {"output": output}, state


class DeadZone(Component):
    """Dead-zone nonlinearity: zero output for small inputs.

    For |u| < threshold, output is zero. For |u| >= threshold,
    output is u - sign(u) * threshold (shifted toward zero).

    Args:
        threshold: Half-width of the dead band.
    """

    input_ports = ("input",)
    output_ports = ("output",)

    threshold: float

    def __init__(self, threshold: float = 0.1):
        self.threshold = float(threshold)

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        u = inputs["input"]
        output = jnp.where(
            jnp.abs(u) < self.threshold,
            0.0,
            u - jnp.sign(u) * self.threshold,
        )
        return {"output": output}, state


class RateLimiter(Component):
    """Limit the rate of change of a signal.

    Clamps the per-step change to ``[-max_rate * dt, max_rate * dt]``.

    Args:
        max_rate: Maximum allowed rate of change (units per second).
        dt: Time step.
        n_dims: Dimensionality of the signal.
        initial_value: Initial previous-output value.
    """

    input_ports = ("input",)
    output_ports = ("output",)

    max_rate: float
    dt: float
    n_dims: int = field(static=True)
    state_index: StateIndex
    _initial_state: Array = field(static=True)

    def __init__(
        self,
        max_rate: float = 1.0,
        dt: float = 0.01,
        n_dims: int = 1,
        initial_value: float = 0.0,
    ):
        self.max_rate = float(max_rate)
        self.dt = float(dt)
        self.n_dims = n_dims
        self._initial_state = jnp.zeros(n_dims) + initial_value
        self.state_index = StateIndex(self._initial_state)

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        prev = state.get(self.state_index)
        u = inputs["input"]
        max_delta = self.max_rate * self.dt
        delta = jnp.clip(u - prev, -max_delta, max_delta)
        output = prev + delta
        state = state.set(self.state_index, output)
        return {"output": output}, state
