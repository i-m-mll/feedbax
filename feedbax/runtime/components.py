"""Basic components for math, sources, and signal processing."""

from __future__ import annotations

from collections.abc import Mapping
from math import prod
from typing import Annotated, Any, Callable, Literal, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
from equinox import Module, field
from equinox.nn import State, StateIndex
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PRNGKeyArray, PyTree
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from feedbax.component_registry import component
from feedbax.component_registry.declarations import (
    ComponentConstructorPlan,
    MissingPrototypeInput,
    ParameterField,
)
from feedbax.component_registry.prototypes import (
    _missing_input,
    _required_input,
    array_prototype,
    input_passthrough_output_prototype,
    migrate_stateful_v1,
    stateful_passthrough_output_prototype,
)
from feedbax.contracts.component import DynamicPortPolicy, PortType, PortTypeSpec
from feedbax.contracts.migrations import ComponentMigration
from feedbax.runtime.graph import Component

LINEAR_PARAM_SCHEMA_VERSION_V1 = "1"
LINEAR_PARAM_SCHEMA_VERSION = "2"


def identity_activation(x: Any) -> Any:
    return x


# Keep wrapped JAX functions out of persisted models. Checkpoint custody
# resolves these module-level functions by name in a fresh process.
def relu(x: Array) -> Array:
    return jax.nn.relu(x)


def tanh(x: Array) -> Array:
    return jax.nn.tanh(x)


def sigmoid(x: Array) -> Array:
    return jax.nn.sigmoid(x)


def softmax(x: Array) -> Array:
    return jax.nn.softmax(x)


NAMED_ACTIVATIONS: dict[str, Callable] = {
    "relu": relu,
    "tanh": tanh,
    "sigmoid": sigmoid,
    "softmax": softmax,
    "identity": identity_activation,
}


def _callable_name(activation: Callable) -> str:
    return getattr(activation, "__name__", activation.__class__.__name__)


def resolve_activation(activation: str | Callable) -> tuple[str, Callable]:
    if isinstance(activation, str):
        try:
            return activation, NAMED_ACTIVATIONS[activation]
        except KeyError as exc:
            vocabulary = ", ".join(NAMED_ACTIVATIONS)
            raise ValueError(
                f"Unknown activation {activation!r}. Supported values: {vocabulary}"
            ) from exc
    if callable(activation):
        return _callable_name(activation), activation
    raise TypeError("activation must be a string name or callable")


def named_activation(activation: str | Callable) -> str:
    """Resolve a portable activation name from a built-in name or callable."""
    if isinstance(activation, str):
        return resolve_activation(activation)[0]
    for name, function in NAMED_ACTIVATIONS.items():
        if activation is function or activation is getattr(jax.nn, name, function):
            return name
    raise ValueError(f"Expected a built-in activation, got {activation!r}")


_RequiredFloat = Annotated[float, Field(json_schema_extra={"feedbax": {"required": True}})]


class _ConstantParams(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    dtype: str | None = None

    value: JsonValue = Field(
        default=0.0,
        json_schema_extra={"feedbax": {"type": "object", "required": True}},
    )


def _migrate_linear_v1(params: dict[str, Any]) -> dict[str, Any]:
    return {**params, "initialization": "random"}


def _shape_from_single_array(proto: Any, *, component: str, port: str) -> tuple[int, ...]:
    leaves = jt.leaves(proto)
    if len(leaves) != 1 or not hasattr(leaves[0], "shape"):
        raise ValueError(
            f"{component} output prototype requires {port!r} to be a single array prototype"
        )
    return tuple(int(dim) for dim in leaves[0].shape)


def _array_proto_from_shape(shape: Any) -> Any | None:
    if not isinstance(shape, (list, tuple)):
        return None
    try:
        return jnp.zeros(tuple(int(dim) for dim in shape))
    except (TypeError, ValueError):
        return None


def _proto_from_value(value: Any) -> Any:
    return jt.map(lambda x: jnp.zeros_like(jnp.asarray(x)), value)


def _positive_sizes(params: Mapping[str, Any], *, component: str) -> tuple[int, ...]:
    sizes = params.get("sizes")
    if not isinstance(sizes, (list, tuple)):
        raise ValueError(f"{component} requires 'sizes' as a list of positive integers")
    parsed = tuple(int(size) for size in sizes)
    if not parsed:
        raise ValueError(f"{component} sizes must contain at least one output size")
    if any(size <= 0 for size in parsed):
        raise ValueError(f"{component} sizes must be positive integers")
    return parsed


def _ravel_width(proto: Any, *, component: str, port: str) -> int:
    leaves = jt.leaves(proto)
    if not leaves or any(not hasattr(leaf, "shape") for leaf in leaves):
        raise ValueError(f"{component} output prototype requires {port!r} to be array-shaped")
    return sum(prod(tuple(int(dim) for dim in leaf.shape)) for leaf in leaves)


def source_value_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a source prototype from the configured scalar/array value."""

    value = params.get("value", 0.0)
    if params.get("dtype") is not None:
        value = jnp.asarray(value, dtype=params["dtype"])
    return {"output": _proto_from_value(value)}


def step_source_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a ramp/sine/pulse prototype from amplitude or slope params."""

    return {"output": _proto_from_value(params.get("amplitude", params.get("slope", 1.0)))}


def noise_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a random-noise sample prototype from the configured shape."""

    return {"output": jnp.zeros(tuple(int(dim) for dim in params.get("shape", [1])))}


def binary_elementwise_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the first available binary elementwise input prototype."""

    proto = inputs.get("a")
    if proto is None:
        proto = inputs.get("b")
    if proto is None:
        raise MissingPrototypeInput(
            "binary elementwise output prototype requires input prototype 'a' or 'b'"
        )
    return {"output": proto}


def input_node_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an input-node prototype under its configured output port."""

    output_port = str(params.get("output_port", "output"))
    return {output_port: _required_input(inputs, "input", component="Input")}


def subtract_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the first available subtract input prototype as ``out``."""

    del params
    proto = inputs.get("a")
    if proto is None:
        proto = inputs.get("b")
    if proto is None:
        raise MissingPrototypeInput("Subtract output prototype requires input prototype 'a' or 'b'")
    return {"out": proto}


def sum_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the first available sum input prototype as ``output``."""

    del params
    for port in ("a", "b", "c", "d"):
        proto = inputs.get(port)
        if proto is not None:
            return {"output": proto}
    raise MissingPrototypeInput("Sum output prototype requires at least one input prototype")


def reshape_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a reshaped array prototype."""

    proto = _required_input(inputs, "input", component="Reshape")
    leaves = jt.leaves(proto)
    if len(leaves) != 1:
        raise ValueError("Reshape output prototype requires a single array input")
    shape = params.get("shape", params.get("output_shape", [1, 1]))
    if not isinstance(shape, (list, tuple)):
        raise ValueError("Reshape output prototype requires 'shape' as a list")
    return {"output": jnp.zeros(tuple(int(dim) for dim in shape), dtype=leaves[0].dtype)}


def matmul_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a prototype for ``a @ b``."""

    del params
    a = _required_input(inputs, "a", component="MatMul")
    b = _required_input(inputs, "b", component="MatMul")
    return {"out": jnp.matmul(jnp.zeros_like(a), jnp.zeros_like(b))}


def ravel_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a one-dimensional flattened prototype for an arbitrary PyTree input."""

    proto = _required_input(inputs, "input", component="Ravel")
    return {"output": jnp.zeros((_ravel_width(proto, component="Ravel", port="input"),))}


def linear_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a dense-layer output prototype from ``output_size``."""

    return {"output": jnp.zeros((int(params.get("output_size", 1)),), dtype=params.get("dtype"))}


def recurrent_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return GRU output and hidden prototypes from ``hidden_size``."""

    hidden = jnp.zeros((int(params.get("hidden_size", 1)),), dtype=params.get("dtype"))
    return {"output": hidden, "hidden": hidden}


def lstm_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return LSTM output, hidden, and cell prototypes from ``hidden_size``."""

    hidden = jnp.zeros((int(params.get("hidden_size", 1)),), dtype=params.get("dtype"))
    return {"output": hidden, "hidden": hidden, "cell": hidden}


def mux_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a concatenated vector prototype for all declared mux inputs."""

    n_inputs = int(params.get("n_inputs", 2))
    ports = [f"in_{index}" for index in range(n_inputs)]
    extra_dynamic_ports = [
        port
        for port in inputs
        if port.startswith("in_") and port.removeprefix("in_").isdigit() and port not in ports
    ]
    if extra_dynamic_ports:
        raise ValueError(
            f"Mux output prototype received ports {extra_dynamic_ports!r} outside "
            f"declared n_inputs={n_inputs}"
        )
    missing = [port for port in ports if port not in inputs]
    if missing:
        raise MissingPrototypeInput(
            "Mux output prototype requires input prototypes for ports "
            + ", ".join(repr(port) for port in missing)
        )
    widths = [_ravel_width(inputs[port], component="Mux", port=port) for port in ports]
    return {"output": jnp.zeros((sum(widths),), dtype=params.get("dtype"))}


def demux_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return per-output prototypes by slicing the input final dimension."""

    proto = _required_input(inputs, "input", component="Demux")
    shape = _shape_from_single_array(proto, component="Demux", port="input")
    if not shape:
        raise ValueError("Demux input prototype must have at least one dimension")
    sizes = _positive_sizes(params, component="Demux")
    width = shape[-1]
    total = sum(sizes)
    if width != total:
        raise ValueError(f"Demux input final dimension {width} does not match sum(sizes) {total}")
    prefix = shape[:-1]
    return {f"out_{index}": jnp.zeros((*prefix, size)) for index, size in enumerate(sizes)}


def switch_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the routed-signal prototype when both switch arms agree."""

    true_proto = _required_input(inputs, "true_input", component="Switch")
    false_proto = _required_input(inputs, "false_input", component="Switch")
    true_shape = _shape_from_single_array(true_proto, component="Switch", port="true_input")
    false_shape = _shape_from_single_array(false_proto, component="Switch", port="false_input")
    if true_shape != false_shape:
        raise ValueError(
            "Switch output prototype requires true_input and false_input "
            f"to have matching shapes, got {true_shape!r} and {false_shape!r}"
        )
    return {"output": true_proto}


def elementwise_affine_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the signal prototype for elementwise affine modulation."""

    signal = inputs.get("signal")
    if signal is not None:
        return {"output": signal}
    proto = _array_proto_from_shape(params.get("signal_shape"))
    if proto is None:
        raise _missing_input("signal", component="ElementwiseAffineModulator")
    return {"output": proto}


def spring_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return spring force with displacement shape."""

    return {"force": _required_input(inputs, "displacement", component="Spring")}


def damper_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return damper force with velocity shape."""

    return {"force": _required_input(inputs, "velocity", component="Damper")}


def stateful_vector_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the vector prototype shared by state storage and output."""

    proto = inputs.get("input")
    if proto is None:
        proto = array_prototype(params)
    return {"output": stateful_vector_prototype(int(params.get("n_dims", 1)), proto)}


_STATEFUL_VECTOR_CONSTRUCTOR = ComponentConstructorPlan(
    context_adapters={"input_shape": ("input_proto", "array_prototype")}
)


_STATEFUL_VECTOR_PROTOTYPE_FIELDS = (
    ParameterField(name="input_shape", type="array", default=None, required=False),
    ParameterField(name="input_dtype", type="str", default=None, required=False),
)


@component(
    "Gain",
    category="Math",
    description="Multiply input by constant.",
    icon="SlidersHorizontal",
    output_prototype_fn=input_passthrough_output_prototype,
    port_types=PortTypeSpec(
        inputs={"input": PortType(dtype="any")},
        outputs={"output": PortType(dtype="any")},
    ),
)
class Gain(Component):
    """Multiply input by a constant gain."""

    input_ports = ("input",)
    output_ports = ("output",)

    gain: float

    def __init__(self, gain: _RequiredFloat = 1.0):
        self.gain = float(gain)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        output = jt.map(lambda x: x * self.gain, inputs["input"])
        return {"output": output}, state


@component(
    "Input",
    category="Sources",
    description="Pass an external graph input through a source-like node.",
    parameter_fields=[
        ParameterField(name="output_port", type="str", default="output", required=True),
    ],
    input_ports=["input"],
    output_ports=["output"],
    icon="LogIn",
    port_types=PortTypeSpec(
        inputs={"input": PortType(dtype="any")},
        outputs={"output": PortType(dtype="any")},
    ),
    output_prototype_fn=input_node_output_prototype,
)
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


@component(
    "Sum",
    category="Math",
    description="Add present inputs.",
    icon="Sigma",
    port_types=PortTypeSpec(
        inputs={
            "a": PortType(dtype="any"),
            "b": PortType(dtype="any"),
            "c": PortType(dtype="any"),
            "d": PortType(dtype="any"),
        },
        outputs={"output": PortType(dtype="any")},
    ),
    output_prototype_fn=sum_output_prototype,
)
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


@component(
    "Subtract",
    category="Math",
    description="Subtract input b from input a.",
    icon="Minus",
    port_types=PortTypeSpec(
        inputs={"a": PortType(dtype="any"), "b": PortType(dtype="any")},
        outputs={"out": PortType(dtype="any")},
    ),
    output_prototype_fn=subtract_output_prototype,
)
class Subtract(Component):
    """Subtract input ``b`` from input ``a``."""

    input_ports = ("a", "b")
    output_ports = ("out",)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        output = jt.map(lambda a, b: a - b, inputs["a"], inputs["b"])
        return {"out": output}, state


@component(
    "Multiply",
    category="Math",
    description="Element-wise product.",
    icon="X",
    port_types=PortTypeSpec(
        inputs={"a": PortType(dtype="any"), "b": PortType(dtype="any")},
        outputs={"output": PortType(dtype="any")},
    ),
    output_prototype_fn=binary_elementwise_output_prototype,
)
class Multiply(Component):
    """Element-wise product of two inputs."""

    input_ports = ("a", "b")
    output_ports = ("output",)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        output = jt.map(lambda a, b: a * b, inputs["a"], inputs["b"])
        return {"output": output}, state


@component(
    "Reshape",
    category="Math",
    description="Reshape an array to a fixed output shape.",
    parameter_fields=[
        ParameterField(name="shape", type="array", default=[1, 1], required=True),
    ],
    icon="Rows3",
    port_types=PortTypeSpec(
        inputs={"input": PortType(dtype="vector")},
        outputs={"output": PortType(dtype="array")},
    ),
    output_prototype_fn=reshape_output_prototype,
)
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


@component(
    "MatMul",
    category="Math",
    description="Matrix multiply a and b.",
    icon="Table2",
    port_types=PortTypeSpec(
        inputs={"a": PortType(dtype="array"), "b": PortType(dtype="array")},
        outputs={"out": PortType(dtype="array")},
    ),
    output_prototype_fn=matmul_output_prototype,
)
class MatMul(Component):
    """Matrix multiply inputs ``a`` and ``b``."""

    input_ports = ("a", "b")
    output_ports = ("out",)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        return {"out": jnp.matmul(inputs["a"], inputs["b"])}, state


@component(
    "Scale",
    category="Math",
    description="Multiply input by a scalar or broadcastable scale.",
    parameter_fields=[
        ParameterField(name="scale", type="float", default=1.0, required=True),
        ParameterField(name="fixed", type="bool", default=False, required=False),
    ],
    icon="Scaling",
    port_types=PortTypeSpec(
        inputs={"input": PortType(dtype="any")},
        outputs={"output": PortType(dtype="any")},
    ),
    output_prototype_fn=input_passthrough_output_prototype,
)
class Scale(Component):
    """Multiply input by a scalar or broadcastable scale."""

    input_ports = ("input",)
    output_ports = ("output",)

    scale: PyTree
    fixed: bool = field(static=True)

    def __init__(self, scale: PyTree = 1.0, fixed: bool = False):
        self.fixed = fixed
        self.scale = float(scale) if fixed else jnp.asarray(scale)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        scale = jnp.asarray(self.scale)
        output = jt.map(lambda x: jnp.asarray(x) * scale, inputs["input"])
        return {"output": output}, state


@component(
    "Sigmoid",
    category="Math",
    description="Apply a logistic sigmoid elementwise.",
    icon="Activity",
    port_types=PortTypeSpec(
        inputs={"input": PortType(dtype="any")},
        outputs={"output": PortType(dtype="any")},
    ),
    output_prototype_fn=input_passthrough_output_prototype,
)
class Sigmoid(Component):
    """Apply a logistic sigmoid elementwise."""

    input_ports = ("input",)
    output_ports = ("output",)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        return {"output": jt.map(jax.nn.sigmoid, inputs["input"])}, state


@component(
    "Activation",
    constructor=ComponentConstructorPlan(renames={"activation": "activation_name"}),
    attribute_paths={"activation": "activation_name"},
    category="Math",
    description="Apply a named elementwise activation.",
    parameter_fields=[
        ParameterField(
            name="activation",
            type="enum",
            options=["identity", "tanh", "relu", "sigmoid", "softmax"],
            default="identity",
            required=True,
        ),
    ],
    icon="Activity",
    port_types=PortTypeSpec(
        inputs={"input": PortType(dtype="any")},
        outputs={"output": PortType(dtype="any")},
    ),
    output_prototype_fn=input_passthrough_output_prototype,
)
class Activation(Component):
    """Apply one named activation elementwise without introducing parameters."""

    input_ports = ("input",)
    output_ports = ("output",)

    activation_name: str = field(static=True)
    activation: Callable = field(static=True)

    def __init__(self, activation_name: str, activation: Callable | None = None):
        if activation is None:
            try:
                activation = NAMED_ACTIVATIONS[activation_name]
            except KeyError as exc:
                raise ValueError(f"Unknown activation {activation_name!r}") from exc
        self.activation_name = activation_name
        self.activation = activation

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        return {"output": jt.map(self.activation, inputs["input"])}, state


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


@component(
    "ElementwiseAffineModulator",
    attribute_paths={"gain_init": "gain", "bias_init": "bias"},
    category="Math",
    description=(
        "Per-element affine modulation: signal * (baseline + gain * modulator) + bias * modulator."
    ),
    parameter_fields=[
        ParameterField(name="signal_shape", type="array", default=[1], required=True),
        ParameterField(name="baseline", type="array", default=1.0, required=False),
        ParameterField(name="gain_init", type="array", default=0.0, required=False),
        ParameterField(name="bias_init", type="array", default=0.0, required=False),
        ParameterField(name="trainable", type="bool", default=True, required=False),
    ],
    icon="SlidersHorizontal",
    port_types=PortTypeSpec(
        inputs={
            "signal": PortType(dtype="vector"),
            "modulator": PortType(dtype="vector"),
            "scale": PortType(dtype="vector"),
            "bias": PortType(dtype="vector"),
        },
        outputs={"output": PortType(dtype="vector")},
    ),
    output_prototype_fn=elementwise_affine_output_prototype,
)
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
    trainable: bool = field(static=True)

    def __init__(
        self,
        signal_shape: Sequence[int],
        baseline: PyTree = 1.0,
        gain_init: PyTree = 0.0,
        bias_init: PyTree = 0.0,
        trainable: bool = True,
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
        self.trainable = bool(trainable)

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


@component(
    "Saturation",
    category="Signal Processing",
    description="Clamp to min/max range.",
    icon="SlidersHorizontal",
    port_types=PortTypeSpec(
        inputs={"input": PortType(dtype="any")},
        outputs={"output": PortType(dtype="any")},
    ),
    output_prototype_fn=input_passthrough_output_prototype,
)
class Saturation(Component):
    """Clamp input to a min/max range."""

    input_ports = ("input",)
    output_ports = ("output",)

    min_val: float
    max_val: float

    def __init__(self, min_val: _RequiredFloat = -1.0, max_val: _RequiredFloat = 1.0):
        self.min_val = float(min_val)
        self.max_val = float(max_val)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        output = jt.map(lambda x: jnp.clip(x, self.min_val, self.max_val), inputs["input"])
        return {"output": output}, state


@component(
    "Constant",
    category="Sources",
    description="Constant value output.",
    param_model=_ConstantParams,
    input_ports=[],
    icon="Circle",
    port_types=PortTypeSpec(
        inputs={},
        outputs={"output": PortType(dtype="any")},
    ),
    output_prototype_fn=source_value_output_prototype,
)
class Constant(Component):
    """Constant value output."""

    output_ports = ("output",)

    value: PyTree
    dtype: str | None = field(static=True)

    def __init__(self, value: PyTree = 0.0, dtype: str | None = None):
        self.dtype = dtype
        if dtype is not None:
            value = jnp.asarray(value, dtype=dtype)
        self.value = jt.map(jnp.asarray, value)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        return {"output": self.value}, state


@component(
    "Ravel",
    category="Signal Processing",
    description="Flatten a PyTree value into a vector.",
    icon="Layers",
    port_types=PortTypeSpec(
        inputs={"input": PortType(dtype="any")},
        outputs={"output": PortType(dtype="vector")},
    ),
    output_prototype_fn=ravel_output_prototype,
)
class Ravel(Component):
    """Flatten a PyTree input to a vector."""

    input_ports = ("input",)
    output_ports = ("output",)

    @staticmethod
    def flatten(value: PyTree) -> Array:
        output, _ = ravel_pytree(value)
        return output

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        return {"output": self.flatten(inputs["input"])}, state


class _StepSource(Component):
    """Base class for sources driven by a step counter."""

    step_index: StateIndex
    dt: float

    def __init__(self, dt: float = 0.01):
        self.dt = float(dt)
        initial_state = 0
        self.step_index = StateIndex(initial_state)

    def _take_step(self, state: State) -> tuple[Array, State]:
        step = state.get(self.step_index)
        return step, state.set(self.step_index, step + 1)


@component(
    "Ramp",
    category="Sources",
    description="Linear ramp over time.",
    parameter_fields=[
        ParameterField(name="slope", type="float", default=1.0, required=True),
        ParameterField(name="intercept", type="float", default=0.0, required=True),
        ParameterField(name="dt", type="float", default=0.01, required=True),
    ],
    input_ports=[],
    icon="TrendingUp",
    port_types=PortTypeSpec(
        inputs={},
        outputs={"output": PortType(dtype="any")},
    ),
    output_prototype_fn=step_source_output_prototype,
)
class Ramp(_StepSource):
    """Linear ramp over time."""

    output_ports = ("output",)

    slope: PyTree
    intercept: PyTree

    def __init__(self, slope: PyTree = 1.0, intercept: PyTree = 0.0, dt: float = 0.01):
        super().__init__(dt=dt)
        self.slope = jt.map(jnp.asarray, slope)
        self.intercept = jt.map(jnp.asarray, intercept)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        step, state = self._take_step(state)
        t = step * self.dt
        output = jt.map(lambda s, b: b + s * t, self.slope, self.intercept)
        return {"output": output}, state


@component(
    "Sine",
    category="Sources",
    description="Sinusoidal signal.",
    parameter_fields=[
        ParameterField(name="amplitude", type="float", default=1.0, required=True),
        ParameterField(name="frequency", type="float", default=1.0, required=True),
        ParameterField(name="phase", type="float", default=0.0, required=False),
        ParameterField(name="offset", type="float", default=0.0, required=False),
        ParameterField(name="dt", type="float", default=0.01, required=True),
    ],
    input_ports=[],
    icon="AudioWaveform",
    port_types=PortTypeSpec(
        inputs={},
        outputs={"output": PortType(dtype="any")},
    ),
    output_prototype_fn=step_source_output_prototype,
)
class Sine(_StepSource):
    """Sinusoidal signal."""

    output_ports = ("output",)

    amplitude: PyTree
    frequency: float
    phase: float
    offset: PyTree

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
        step, state = self._take_step(state)
        t = step * self.dt
        omega = 2 * jnp.pi * self.frequency
        output = jt.map(
            lambda a, b: b + a * jnp.sin(omega * t + self.phase),
            self.amplitude,
            self.offset,
        )
        return {"output": output}, state


@component(
    "Pulse",
    category="Sources",
    description="Pulse/square wave.",
    parameter_fields=[
        ParameterField(name="amplitude", type="float", default=1.0, required=True),
        ParameterField(name="period", type="float", default=1.0, required=True),
        ParameterField(name="duty_cycle", type="float", default=0.5, required=True),
        ParameterField(name="offset", type="float", default=0.0, required=False),
        ParameterField(name="dt", type="float", default=0.01, required=True),
    ],
    input_ports=[],
    icon="Activity",
    port_types=PortTypeSpec(
        inputs={},
        outputs={"output": PortType(dtype="any")},
    ),
    output_prototype_fn=step_source_output_prototype,
)
class Pulse(_StepSource):
    """Pulse/square wave."""

    output_ports = ("output",)

    amplitude: PyTree
    period: float
    duty_cycle: float
    offset: PyTree

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
        step, state = self._take_step(state)
        phase = 0.0
        if self.period > 0 and self.dt != 0:
            # Wrap in step units to avoid rounding elapsed time before finding the phase.
            steps_per_period = self.period / self.dt
            phase = (step % steps_per_period) / steps_per_period
        gate = jnp.where(phase < self.duty_cycle, 1.0, 0.0)
        output = jt.map(lambda a, b: b + a * gate, self.amplitude, self.offset)
        return {"output": output}, state


@component(
    "Noise",
    category="Signal Processing",
    description="Random noise source.",
    parameter_fields=[
        ParameterField(name="mean", type="float", default=0.0, required=False),
        ParameterField(name="std", type="float", default=1.0, required=True),
        ParameterField(name="shape", type="array", default=[1], required=False),
    ],
    input_ports=[],
    icon="Sparkles",
    port_types=PortTypeSpec(
        inputs={},
        outputs={"output": PortType(dtype="any")},
    ),
    output_prototype_fn=noise_output_prototype,
)
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


@component(
    "Spring",
    category="Mechanics",
    description="Linear spring.",
    icon="Move",
    port_types=PortTypeSpec(
        inputs={"displacement": PortType(dtype="vector")},
        outputs={"force": PortType(dtype="vector")},
    ),
    output_prototype_fn=spring_output_prototype,
)
class Spring(Component):
    """Linear spring: F = k * displacement."""

    input_ports = ("displacement",)
    output_ports = ("force",)

    stiffness: float

    def __init__(self, stiffness: _RequiredFloat = 1.0):
        self.stiffness = float(stiffness)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        output = jt.map(lambda x: self.stiffness * x, inputs["displacement"])
        return {"force": output}, state


@component(
    "Damper",
    category="Mechanics",
    description="Viscous damper.",
    icon="Move",
    port_types=PortTypeSpec(
        inputs={"velocity": PortType(dtype="vector")},
        outputs={"force": PortType(dtype="vector")},
    ),
    output_prototype_fn=damper_output_prototype,
)
class Damper(Component):
    """Viscous damper: F = b * velocity."""

    input_ports = ("velocity",)
    output_ports = ("force",)

    damping: float

    def __init__(self, damping: _RequiredFloat = 1.0):
        self.damping = float(damping)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        output = jt.map(lambda x: self.damping * x, inputs["velocity"])
        return {"force": output}, state


@component(
    "DelayLine",
    constructor=ComponentConstructorPlan(
        context_adapters={"input_shape": ("input_proto", "array_prototype")}
    ),
    param_schema_version="2",
    category="Signal Processing",
    description="Discrete delay buffer.",
    parameter_fields=[
        ParameterField(name="delay", type="int", default=1, min=0, required=True),
        ParameterField(name="init_value", type="float", default=0.0, required=False),
        ParameterField(name="input_shape", type="array", default=None, required=False),
        ParameterField(name="input_dtype", type="str", default=None, required=False),
    ],
    icon="Clock",
    port_types=PortTypeSpec(
        inputs={"input": PortType(dtype="any")},
        outputs={"output": PortType(dtype="any")},
    ),
    output_prototype_fn=stateful_passthrough_output_prototype,
    migrations=(
        ComponentMigration(
            source_type="DelayLine",
            target_type="DelayLine",
            owner="feedbax",
            source_param_schema_version="1",
            target_param_schema_version="2",
            migration_id="feedbax.component.DelayLine.params.v1-to-v2",
            migrate_params=migrate_stateful_v1,
            description="Preserve stateful declarations with optional inferred input dtype.",
        ),
    ),
)
class DelayLine(Component):
    """Discrete delay buffer."""

    input_ports = ("input",)
    output_ports = ("output",)

    delay: int
    init_value: float
    input_proto: PyTree
    state_index: StateIndex

    def __init__(self, delay: int = 1, init_value: float = 0.0, input_proto: PyTree | None = None):
        self.delay = int(delay)
        self.init_value = float(init_value)
        if input_proto is None:
            input_proto = jnp.zeros(1)
        self.input_proto = input_proto
        output = jt.map(lambda x: jnp.full_like(x, self.init_value), input_proto)
        queue = self.delay * (output,)
        initial_state = (output, queue)
        self.state_index = StateIndex(initial_state)

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


@component(
    "Linear",
    build_context_fields=("key",),
    constructor=ComponentConstructorPlan(injections={"key": jr.PRNGKey(0)}),
    attribute_paths={"activation": "activation_name"},
    category="Neural Networks",
    description="Linear layer.",
    parameter_fields=[
        ParameterField(name="dtype", type="str", default="float32", required=False),
        ParameterField(name="input_size", type="int", default=1, min=1, required=True),
        ParameterField(name="output_size", type="int", default=1, min=1, required=True),
        ParameterField(name="use_bias", type="bool", default=True, required=False),
        ParameterField(name="bias_init", type="float", default=None, required=False),
        ParameterField(
            name="initialization",
            type="enum",
            options=["random", "zero_bias", "zeros"],
            default="random",
            required=False,
        ),
        ParameterField(
            name="activation",
            type="enum",
            options=["identity", "tanh", "relu", "sigmoid", "softmax"],
            default="identity",
            required=False,
        ),
    ],
    icon="Minus",
    port_types=PortTypeSpec(
        inputs={"input": PortType(dtype="vector")},
        outputs={"output": PortType(dtype="vector")},
    ),
    output_prototype_fn=linear_output_prototype,
    param_schema_version=LINEAR_PARAM_SCHEMA_VERSION,
    trainable_by_default=True,
    migrations=(
        ComponentMigration(
            source_type="Linear",
            target_type="Linear",
            owner="feedbax",
            source_param_schema_version=LINEAR_PARAM_SCHEMA_VERSION_V1,
            target_param_schema_version=LINEAR_PARAM_SCHEMA_VERSION,
            migration_id="feedbax.component.Linear.params.v1-to-v2",
            migrate_params=_migrate_linear_v1,
            description="Make the existing random Linear initialization explicit.",
        ),
    ),
)
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
    initialization: Literal["random", "zero_bias", "zeros"] = field(static=True)
    dtype: Any = field(static=True)
    bias_init: float | None = field(static=True)

    def __init__(
        self,
        input_size: int,
        output_size: int,
        use_bias: bool = True,
        activation: str | Callable = "identity",
        initialization: Literal["random", "zero_bias", "zeros"] = "random",
        dtype: Any = jnp.float32,
        bias_init: float | None = None,
        *,
        key: PRNGKeyArray,
    ):
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.use_bias = bool(use_bias)
        if initialization not in ("random", "zero_bias", "zeros"):
            raise ValueError(
                "Linear initialization must be 'random', 'zero_bias', or 'zeros'; "
                f"got {initialization!r}"
            )
        self.initialization = initialization
        self.dtype = jnp.dtype(dtype).name
        self.activation_name, self.activation = resolve_activation(activation)
        layer = eqx.nn.Linear(
            self.input_size,
            self.output_size,
            use_bias=self.use_bias,
            dtype=dtype,
            key=key,
        )
        if self.initialization == "zeros":
            layer = eqx.tree_at(lambda item: item.weight, layer, jnp.zeros_like(layer.weight))
        self.bias_init = bias_init
        if bias_init is not None and layer.bias is None:
            raise ValueError("Linear bias_init requires use_bias=True")
        if layer.bias is not None and (bias_init is not None or initialization != "random"):
            bias = jnp.full_like(layer.bias, 0.0 if bias_init is None else bias_init)
            layer = eqx.tree_at(lambda item: item.bias, layer, bias)
        self.layer = layer

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        output = self.activation(self.layer(inputs["input"]))
        return {"output": output}, state


@component(
    "MLP",
    build_context_fields=("key",),
    constructor=ComponentConstructorPlan(injections={"key": jr.PRNGKey(0)}),
    attribute_paths={
        "activation": "activation_name",
        "final_activation": "final_activation_name",
    },
    category="Neural Networks",
    description="Multi-layer perceptron.",
    parameter_fields=[
        ParameterField(name="dtype", type="str", default="float32", required=False),
        ParameterField(name="input_size", type="int", default=4, min=1, required=True),
        ParameterField(name="output_size", type="int", default=2, min=1, required=True),
        ParameterField(name="hidden_sizes", type="array", default=[64], required=False),
        ParameterField(
            name="activation",
            type="enum",
            options=["relu", "tanh", "identity"],
            default="relu",
            required=False,
        ),
        ParameterField(
            name="final_activation",
            type="enum",
            options=["identity", "tanh", "relu"],
            default="identity",
            required=False,
        ),
    ],
    icon="Brain",
    port_types=PortTypeSpec(
        inputs={"input": PortType(dtype="vector")},
        outputs={"output": PortType(dtype="vector")},
    ),
    output_prototype_fn=linear_output_prototype,
    trainable_by_default=True,
)
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
        self.dtype = jnp.dtype(dtype).name
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


@component(
    "GRU",
    build_context_fields=("key",),
    constructor=ComponentConstructorPlan(injections={"key": jr.PRNGKey(0)}),
    attribute_paths={"use_bias": "cell.use_bias"},
    category="Neural Networks",
    description="GRU cell.",
    parameter_fields=[
        ParameterField(name="dtype", type="str", default="float32", required=False),
        ParameterField(name="input_size", type="int", default=4, min=1, required=True),
        ParameterField(name="hidden_size", type="int", default=4, min=1, required=True),
        ParameterField(name="use_bias", type="bool", default=True, required=False),
        ParameterField(name="stateful", type="bool", default=True, required=False),
    ],
    icon="BrainCircuit",
    port_types=PortTypeSpec(
        inputs={"input": PortType(dtype="vector"), "hidden": PortType(dtype="vector")},
        outputs={"output": PortType(dtype="vector"), "hidden": PortType(dtype="vector")},
    ),
    output_prototype_fn=recurrent_output_prototype,
    trainable_by_default=True,
)
class GRU(Component):
    """Standalone GRU cell."""

    input_ports = ("input", "hidden")
    output_ports = ("output", "hidden")

    cell: eqx.nn.GRUCell
    input_size: int = field(static=True)
    hidden_size: int = field(static=True)
    dtype: Any = field(static=True)
    state_index: StateIndex | None

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dtype: Any = jnp.float32,
        stateful: bool = True,
        *,
        use_bias: bool = True,
        key: PRNGKeyArray,
    ):
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.dtype = jnp.dtype(dtype).name
        self.cell = eqx.nn.GRUCell(
            self.input_size, self.hidden_size, use_bias=use_bias, dtype=dtype, key=key
        )
        if stateful:
            hidden = jnp.zeros(self.hidden_size, dtype=dtype)
            self.state_index = StateIndex(GRUState(hidden=hidden, output=hidden))
        else:
            self.state_index = None

    @property
    def stateful(self) -> bool:
        return self.state_index is not None

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        hidden = inputs.get("hidden")
        if hidden is None:
            if self.state_index is None:
                raise ValueError("A stateless GRU requires an explicit hidden input")
            hidden = state.get(self.state_index).hidden
        new_hidden = self.cell(inputs["input"], hidden)
        if self.state_index is not None:
            state = state.set(self.state_index, GRUState(hidden=new_hidden, output=new_hidden))
        return {"output": new_hidden, "hidden": new_hidden}, state


class LSTMState(Module):
    """State view for a leaf LSTM cell."""

    hidden: Array
    cell: Array
    output: Array


@component(
    "LSTM",
    build_context_fields=("key",),
    constructor=ComponentConstructorPlan(injections={"key": jr.PRNGKey(0)}),
    attribute_paths={"use_bias": "cell.use_bias"},
    category="Neural Networks",
    description="LSTM cell.",
    parameter_fields=[
        ParameterField(name="dtype", type="str", default="float32", required=False),
        ParameterField(name="input_size", type="int", default=4, min=1, required=True),
        ParameterField(name="hidden_size", type="int", default=4, min=1, required=True),
        ParameterField(name="use_bias", type="bool", default=True, required=False),
    ],
    icon="BrainCircuit",
    port_types=PortTypeSpec(
        inputs={
            "input": PortType(dtype="vector"),
            "hidden": PortType(dtype="vector"),
            "cell": PortType(dtype="vector"),
        },
        outputs={
            "output": PortType(dtype="vector"),
            "hidden": PortType(dtype="vector"),
            "cell": PortType(dtype="vector"),
        },
    ),
    output_prototype_fn=lstm_output_prototype,
    trainable_by_default=True,
)
class LSTM(Component):
    """Standalone LSTM cell."""

    input_ports = ("input", "hidden", "cell")
    output_ports = ("output", "hidden", "cell")

    cell: eqx.nn.LSTMCell
    input_size: int = field(static=True)
    hidden_size: int = field(static=True)
    dtype: Any = field(static=True)
    state_index: StateIndex

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dtype: Any = jnp.float32,
        *,
        use_bias: bool = True,
        key: PRNGKeyArray,
    ):
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.dtype = jnp.dtype(dtype).name
        self.cell = eqx.nn.LSTMCell(
            self.input_size, self.hidden_size, use_bias=use_bias, dtype=dtype, key=key
        )
        hidden = jnp.zeros(self.hidden_size, dtype=dtype)
        cell = jnp.zeros(self.hidden_size, dtype=dtype)
        initial_state = LSTMState(hidden=hidden, cell=cell, output=hidden)
        self.state_index = StateIndex(initial_state)

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


@component(
    "Mux",
    category="Signal Processing",
    description="Concatenate inputs into single vector.",
    parameter_fields=[
        ParameterField(name="dtype", type="str", default=None, required=False),
        ParameterField(name="n_inputs", type="int", default=2, min=1, required=True),
    ],
    input_ports=["in_0", "in_1"],
    icon="GitMerge",
    port_types=PortTypeSpec(
        inputs={
            "in_0": PortType(dtype="vector"),
            "in_1": PortType(dtype="vector"),
        },
        outputs={"output": PortType(dtype="vector")},
    ),
    dynamic_port_policy=DynamicPortPolicy(
        count_param="n_inputs",
        count_mode="integer",
        direction="input",
        fixed_output_ports=("output",),
        generated_name_template="in_{index}",
        dynamic_port_type=PortType(dtype="vector"),
    ),
    output_prototype_fn=mux_output_prototype,
)
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
    dtype: Any = field(static=True)

    def __init__(self, n_inputs: int = 2, *, dtype: Any = None):
        self.dtype = jnp.dtype(dtype).name if dtype is not None else None
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
            if inputs.get(port) is None:
                continue
            flat, _ = ravel_pytree(inputs[port])
            parts.append(jnp.asarray(flat, dtype=self.dtype))
        if not parts:
            raise ValueError("Mux requires at least one connected input")
        output = jnp.concatenate(parts, axis=-1)
        return {"output": output}, state


@component(
    "Demux",
    category="Signal Processing",
    description="Split vector into multiple outputs.",
    parameter_fields=[
        ParameterField(name="sizes", type="array", default=[1, 1], required=True),
    ],
    output_ports=["out_0", "out_1"],
    icon="GitBranch",
    port_types=PortTypeSpec(
        inputs={"input": PortType(dtype="vector")},
        outputs={
            "out_0": PortType(dtype="vector"),
            "out_1": PortType(dtype="vector"),
        },
    ),
    dynamic_port_policy=DynamicPortPolicy(
        count_param="sizes",
        count_mode="sequence_length",
        direction="output",
        fixed_input_ports=("input",),
        generated_name_template="out_{index}",
        dynamic_port_type=PortType(dtype="vector"),
    ),
    output_prototype_fn=demux_output_prototype,
)
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


@component(
    "Switch",
    runtime_type=None,
    category="Signal Processing",
    description="Route signal by threshold condition.",
    parameter_fields=[
        ParameterField(name="threshold", type="float", default=0.0, required=True),
    ],
    icon="GitCompare",
    port_types=PortTypeSpec(
        inputs={
            "condition": PortType(dtype="scalar"),
            "true_input": PortType(dtype="any"),
            "false_input": PortType(dtype="any"),
        },
        outputs={"output": PortType(dtype="any")},
    ),
    output_prototype_fn=switch_output_prototype,
)
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


@component(
    "DeadZone",
    runtime_type=None,
    category="Signal Processing",
    description="Zero output for small inputs.",
    parameter_fields=[
        ParameterField(name="threshold", type="float", default=0.1, min=0.0, required=True),
    ],
    icon="MinusSquare",
    port_types=PortTypeSpec(
        inputs={"input": PortType(dtype="any")},
        outputs={"output": PortType(dtype="any")},
    ),
    output_prototype_fn=input_passthrough_output_prototype,
)
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


def stateful_vector_prototype(n_dims: int, input_proto: Array | None = None) -> Array:
    """Use a resolved floating vector, or allocate one from the declared dimension."""
    if input_proto is None:
        return jnp.zeros(n_dims)
    if getattr(input_proto, "ndim", None) != 1 or not jnp.issubdtype(
        input_proto.dtype, jnp.floating
    ):
        raise ValueError("Stateful signal blocks require a real floating vector input")
    return input_proto


@component(
    "RateLimiter",
    constructor=_STATEFUL_VECTOR_CONSTRUCTOR,
    category="Signal Processing",
    description="Limit rate of change of signal.",
    parameter_fields=[
        ParameterField(name="max_rate", type="float", default=1.0, min=0.0, required=True),
        ParameterField(name="dt", type="float", default=0.01, min=0.0, required=True),
        ParameterField(name="n_dims", type="int", default=1, min=1, required=False),
        ParameterField(name="initial_value", type="float", default=0.0, required=False),
        *_STATEFUL_VECTOR_PROTOTYPE_FIELDS,
    ],
    icon="Gauge",
    port_types=PortTypeSpec(
        inputs={"input": PortType(dtype="vector")},
        outputs={"output": PortType(dtype="vector")},
    ),
    output_prototype_fn=stateful_vector_output_prototype,
)
class RateLimiter(Component):
    """Limit the rate of change of a signal.

    Clamps the per-step change to ``[-max_rate * dt, max_rate * dt]``.

    Args:
        max_rate: Maximum allowed rate of change (units per second).
        dt: Time step.
        n_dims: Dimensionality used when no input prototype is supplied.
        initial_value: Initial previous-output value.
        input_proto: Shape and dtype of the signal.
    """

    input_ports = ("input",)
    output_ports = ("output",)

    max_rate: float
    dt: float
    n_dims: int = field(static=True)
    _initial_value_bits: Array
    state_index: StateIndex

    def __init__(
        self,
        max_rate: float = 1.0,
        dt: float = 0.01,
        n_dims: int = 1,
        initial_value: float = 0.0,
        *,
        input_proto: Array | None = None,
    ):
        self.max_rate = float(max_rate)
        self.dt = float(dt)
        self.n_dims = n_dims
        # Keep the authored precision in a dynamic, nontrainable leaf even without JAX x64.
        self._initial_value_bits = jnp.asarray(
            np.array([initial_value], dtype="<f8").view(np.uint8)
        )
        prototype = stateful_vector_prototype(n_dims, input_proto)
        initial_state = jnp.full_like(prototype, initial_value)
        self.state_index = StateIndex(initial_state)

    @property
    def initial_value(self) -> float:
        return np.asarray(self._initial_value_bits).view("<f8").item()

    @property
    def input_proto(self) -> Array:
        return self.state_index.init

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
