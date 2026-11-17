#!/usr/bin/env python3
"""Generate Component wrappers for Equinox neural network classes.

This script introspects equinox.nn classes and generates typed Component
wrappers that can be used as leaf nodes in feedbax computation graphs.

Usage:
    uv run scripts/generate_eqx_components.py

The generated file will be written to feedbax/components/equinox.py
"""

from __future__ import annotations

import inspect
import subprocess
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import UnionType
from typing import Literal, Union, get_args, get_origin

import equinox.nn as nn

TRAINABLE_DTYPE_PARAMS = {"dtype"}
DEFAULT_TRAINABLE_DTYPE = "float32"


@dataclass
class ComponentSpec:
    """Specification for generating a Component wrapper."""

    eqx_class_name: str
    input_ports: tuple[str, ...] = ("input",)
    output_ports: tuple[str, ...] = ("output",)
    # For RNN cells that need hidden state ports
    extra_input_ports: tuple[str, ...] = ()
    extra_output_ports: tuple[str, ...] = ()
    # Category for organization
    category: Literal["linear", "conv", "rnn", "norm", "pool", "attention", "other"] = "other"
    # Custom docstring
    docstring: str | None = None
    # Parameters to exclude from __init__ (handled specially)
    exclude_params: tuple[str, ...] = ()
    # Feedbax-owned defaults expressed in the generated module's namespace.
    default_expressions: dict[str, str] = field(default_factory=dict)
    # Constructor seeds exposed as live properties: public name -> (layer attribute, type).
    layer_properties: dict[str, tuple[str, str]] = field(default_factory=dict)
    # Constructor fields stored in a nested upstream layer.
    storage_paths: dict[str, str] = field(default_factory=dict)
    # Deprecated constructor aliases resolved into canonical upstream fields.
    resolved_aliases: tuple[str, ...] = ()
    # How the wrapper calls the underlying layer.
    call_kind: Literal[
        "single_input",
        "single_input_keyed",
        "batch_norm",
        "multihead_attention",
        "gru_cell",
        "lstm_cell",
    ] = "single_input"
    # Runtime contract metadata emitted on the generated class.
    state_handling: str = "stateless"
    key_handling: str = "ignored"


# Define specs for all Equinox classes we want to wrap
COMPONENT_SPECS: list[ComponentSpec] = [
    # === Linear layers ===
    ComponentSpec(
        "Linear",
        category="linear",
        docstring="Linear transformation layer.",
    ),
    ComponentSpec(
        "Identity",
        category="linear",
        docstring="Identity layer (passes input through unchanged).",
    ),
    ComponentSpec(
        "MLP",
        category="linear",
        docstring="""Multi-layer perceptron with configurable depth and activations.

    The constructor's `activation` and `final_activation` seed the layer's
    activations. The properties expose the live callables used by the layer,
    with learned parameters batched per hidden layer and neuron for
    `activation` and per output neuron for a vector-valued `final_activation`.
    """,
        layer_properties={
            "activation": ("activation", "Callable"),
            "final_activation": ("final_activation", "Callable"),
        },
        default_expressions={"activation": "relu", "final_activation": "identity_activation"},
    ),
    # === Convolutional layers ===
    ComponentSpec(
        "Conv1d",
        category="conv",
        docstring="1D convolution layer.",
    ),
    ComponentSpec(
        "Conv2d",
        category="conv",
        docstring="2D convolution layer.",
    ),
    ComponentSpec(
        "Conv3d",
        category="conv",
        docstring="3D convolution layer.",
    ),
    ComponentSpec(
        "ConvTranspose1d",
        category="conv",
        docstring="1D transposed convolution (deconvolution) layer.",
    ),
    ComponentSpec(
        "ConvTranspose2d",
        category="conv",
        docstring="2D transposed convolution (deconvolution) layer.",
    ),
    ComponentSpec(
        "ConvTranspose3d",
        category="conv",
        docstring="3D transposed convolution (deconvolution) layer.",
    ),
    # === RNN cells ===
    ComponentSpec(
        "GRUCell",
        input_ports=("input", "hidden"),
        output_ports=("output", "hidden"),
        call_kind="gru_cell",
        category="rnn",
        docstring="Gated Recurrent Unit cell.",
    ),
    ComponentSpec(
        "LSTMCell",
        input_ports=("input", "hidden", "cell"),
        output_ports=("output", "hidden", "cell"),
        call_kind="lstm_cell",
        key_handling="forwarded",
        category="rnn",
        docstring="Long Short-Term Memory cell.",
    ),
    # === Normalization layers ===
    ComponentSpec(
        "LayerNorm",
        resolved_aliases=("elementwise_affine",),
        category="norm",
        docstring="Layer normalization.",
    ),
    ComponentSpec(
        "RMSNorm",
        category="norm",
        docstring="Root Mean Square normalization.",
    ),
    ComponentSpec(
        "GroupNorm",
        category="norm",
        docstring="Group normalization.",
    ),
    ComponentSpec(
        "BatchNorm",
        category="norm",
        call_kind="batch_norm",
        state_handling="threads_eqx_state",
        key_handling="optional_forwarded",
        docstring="Batch normalization with explicit Equinox State threading.",
        exclude_params=("mode",),
    ),
    # === Pooling layers ===
    ComponentSpec(
        "MaxPool1d",
        category="pool",
        docstring="1D max pooling.",
    ),
    ComponentSpec(
        "MaxPool2d",
        category="pool",
        docstring="2D max pooling.",
    ),
    ComponentSpec(
        "MaxPool3d",
        category="pool",
        docstring="3D max pooling.",
    ),
    ComponentSpec(
        "AvgPool1d",
        category="pool",
        docstring="1D average pooling.",
    ),
    ComponentSpec(
        "AvgPool2d",
        category="pool",
        docstring="2D average pooling.",
    ),
    ComponentSpec(
        "AvgPool3d",
        category="pool",
        docstring="3D average pooling.",
    ),
    ComponentSpec(
        "AdaptiveMaxPool1d",
        category="pool",
        docstring="1D adaptive max pooling.",
    ),
    ComponentSpec(
        "AdaptiveMaxPool2d",
        category="pool",
        docstring="2D adaptive max pooling.",
    ),
    ComponentSpec(
        "AdaptiveMaxPool3d",
        category="pool",
        docstring="3D adaptive max pooling.",
    ),
    ComponentSpec(
        "AdaptiveAvgPool1d",
        category="pool",
        docstring="1D adaptive average pooling.",
    ),
    ComponentSpec(
        "AdaptiveAvgPool2d",
        category="pool",
        docstring="2D adaptive average pooling.",
    ),
    ComponentSpec(
        "AdaptiveAvgPool3d",
        category="pool",
        docstring="3D adaptive average pooling.",
    ),
    # === Attention ===
    ComponentSpec(
        "MultiheadAttention",
        storage_paths={"dropout_p": "dropout.p", "inference": "dropout.inference"},
        input_ports=("query", "key_", "value"),
        output_ports=("output",),
        call_kind="multihead_attention",
        key_handling="forwarded",
        category="attention",
        docstring="Multi-head attention layer.",
    ),
    ComponentSpec(
        "RotaryPositionalEmbedding",
        category="attention",
        docstring="Rotary positional embedding for attention.",
    ),
    # === Other ===
    ComponentSpec(
        "Embedding",
        category="other",
        docstring="""Embedding layer for discrete tokens.

    The optional constructor `weight` seeds the embedding table. The
    `weight` property exposes the live table, including when randomly initialized.
    """,
        layer_properties={"weight": ("weight", 'Float[Array, "num_embeddings embedding_size"]')},
    ),
    ComponentSpec(
        "Dropout",
        resolved_aliases=("deterministic",),
        call_kind="single_input_keyed",
        key_handling="forwarded",
        category="other",
        docstring="Dropout layer that honors its configured inference/deterministic settings.",
    ),
    ComponentSpec(
        "PReLU",
        category="other",
        docstring="""Parametric ReLU activation.

    The constructor `init_alpha` seeds the negative slope. The
    `init_alpha` property exposes the live slope as an array.
    """,
        layer_properties={"init_alpha": ("negative_slope", "Array")},
    ),
]


# Published revision-1 field evidence; independent of future constructor changes.
LEGACY_CONFIG_FIELDS = {
    "Linear": ("in_features", "out_features", "use_bias", "dtype"),
    "Identity": (),
    "Conv1d": (
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
        "use_bias",
        "padding_mode",
        "dtype",
    ),
    "Conv2d": (
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
        "use_bias",
        "padding_mode",
        "dtype",
    ),
    "Conv3d": (
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
        "use_bias",
        "padding_mode",
        "dtype",
    ),
    "ConvTranspose1d": (
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "output_padding",
        "padding",
        "dilation",
        "groups",
        "use_bias",
        "padding_mode",
        "dtype",
    ),
    "ConvTranspose2d": (
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "output_padding",
        "padding",
        "dilation",
        "groups",
        "use_bias",
        "padding_mode",
        "dtype",
    ),
    "ConvTranspose3d": (
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "output_padding",
        "padding",
        "dilation",
        "groups",
        "use_bias",
        "padding_mode",
        "dtype",
    ),
    "GRUCell": ("input_size", "hidden_size", "use_bias", "dtype"),
    "LSTMCell": ("input_size", "hidden_size", "use_bias", "dtype"),
    "LayerNorm": ("shape", "eps", "use_weight", "use_bias", "dtype", "elementwise_affine"),
    "RMSNorm": ("shape", "eps", "use_weight", "use_bias", "dtype"),
    "GroupNorm": ("groups", "channels", "eps", "channelwise_affine", "dtype"),
    "MaxPool1d": ("kernel_size", "stride", "padding", "use_ceil"),
    "MaxPool2d": ("kernel_size", "stride", "padding", "use_ceil"),
    "MaxPool3d": ("kernel_size", "stride", "padding", "use_ceil"),
    "AvgPool1d": ("kernel_size", "stride", "padding", "use_ceil"),
    "AvgPool2d": ("kernel_size", "stride", "padding", "use_ceil"),
    "AvgPool3d": ("kernel_size", "stride", "padding", "use_ceil"),
    "AdaptiveMaxPool1d": ("target_shape",),
    "AdaptiveMaxPool2d": ("target_shape",),
    "AdaptiveMaxPool3d": ("target_shape",),
    "AdaptiveAvgPool1d": ("target_shape",),
    "AdaptiveAvgPool2d": ("target_shape",),
    "AdaptiveAvgPool3d": ("target_shape",),
    "MultiheadAttention": (
        "num_heads",
        "query_size",
        "key_size",
        "value_size",
        "output_size",
        "qk_size",
        "vo_size",
        "use_query_bias",
        "use_key_bias",
        "use_value_bias",
        "use_output_bias",
        "dropout_p",
        "inference",
        "dtype",
    ),
    "RotaryPositionalEmbedding": ("embedding_size", "theta", "dtype"),
    "Dropout": ("p", "inference", "deterministic"),
}


def get_init_signature(eqx_class) -> inspect.Signature:
    """Get the __init__ signature of an Equinox class."""
    return inspect.signature(eqx_class.__init__)


def format_type_annotation(annotation) -> str:
    """Format a type annotation as a string."""
    if annotation is inspect.Parameter.empty:
        return ""

    # Handle common types
    if annotation is type(None):
        return "None"

    if get_origin(annotation) in (Union, UnionType):
        args = get_args(annotation)
        if len(args) == 2 and type(None) in args:
            arg = next(arg for arg in args if arg is not type(None))
            return f"Optional[{format_type_annotation(arg)}]"
        return f"Union[{', '.join(format_type_annotation(arg) for arg in args)}]"

    # Handle Literal types specially
    if hasattr(annotation, "__origin__"):
        origin = annotation.__origin__
        args = getattr(annotation, "__args__", ())

        if origin is type(None):
            return "None"

        # Check for Literal
        origin_name = getattr(origin, "__name__", str(origin))
        if origin_name == "Literal" or str(origin) == "typing.Literal":
            # Format Literal args as strings
            formatted_args = ", ".join(repr(a) for a in args)
            return f"Literal[{formatted_args}]"

        if args:
            formatted_args = ", ".join(format_type_annotation(a) for a in args)
            return f"{origin_name}[{formatted_args}]"
        return origin_name

    if hasattr(annotation, "__name__"):
        return annotation.__name__

    # Handle jaxtyping annotations like Float[Array, "..."]
    ann_str = str(annotation)
    if "Float[" in ann_str or "Array" in ann_str:
        return "Any"  # Simplify complex jaxtyping annotations

    return str(annotation)


def format_default(default) -> str:
    """Format a default value as a string."""
    if default is inspect.Parameter.empty:
        return ""
    if default is None:
        return "None"
    if isinstance(default, str):
        return repr(default)
    if isinstance(default, bool):
        return str(default)
    if isinstance(default, (int, float)):
        return str(default)
    if isinstance(default, tuple):
        if len(default) == 0:
            return "()"
        return repr(default)
    if callable(default):
        raise ValueError("Callable defaults require an explicit default expression")
    # Handle dataclass field defaults (MISSING, factory, etc.)
    default_str = str(default)
    if "MISSING" in default_str or "factory" in default_str.lower():
        return "None"
    return repr(default)


def default_for_param(spec: ComponentSpec, name: str, default) -> str:
    """Return the Feedbax override or render the upstream default."""

    if name in spec.default_expressions:
        return spec.default_expressions[name]
    if name in TRAINABLE_DTYPE_PARAMS and default is None:
        return repr(DEFAULT_TRAINABLE_DTYPE)
    try:
        return format_default(default)
    except ValueError as exc:
        raise ValueError(
            f"Cannot generate default for {spec.eqx_class_name}.{name}: {exc}"
        ) from exc


def is_portable(spec: ComponentSpec) -> bool:
    return not (spec.layer_properties or spec.call_kind == "batch_norm")


def parameter_annotation(spec: ComponentSpec, name: str, param: inspect.Parameter) -> str:
    if is_portable(spec) and name == "dtype":
        return "str | None"
    return format_type_annotation(param.annotation)


def generate_component_class(spec: ComponentSpec) -> str:
    """Generate the Component wrapper class for a given spec."""
    eqx_class = getattr(nn, spec.eqx_class_name)
    sig = get_init_signature(eqx_class)

    # Filter out 'self', excluded params, and VAR_POSITIONAL/VAR_KEYWORD
    params = [
        (name, param)
        for name, param in sig.parameters.items()
        if name != "self"
        and name not in spec.exclude_params
        and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]

    # Separate required (positional) and optional (keyword) params
    # Also separate out 'key' which we handle specially
    positional_params = []
    keyword_params = []
    has_key_param = False

    for name, param in params:
        if name == "key":
            has_key_param = True
            continue

        if param.default is inspect.Parameter.empty:
            positional_params.append((name, param))
        else:
            keyword_params.append((name, param))

    # Build __init__ signature
    init_params = ["self"]

    for name, param in positional_params:
        type_hint = parameter_annotation(spec, name, param)
        if type_hint:
            init_params.append(f"{name}: {type_hint}")
        else:
            init_params.append(name)

    for name, param in keyword_params:
        type_hint = parameter_annotation(spec, name, param)
        default = default_for_param(spec, name, param.default)
        if type_hint and default:
            init_params.append(f"{name}: {type_hint} = {default}")
        elif type_hint:
            init_params.append(f"{name}: {type_hint}")
        elif default:
            init_params.append(f"{name} = {default}")
        else:
            init_params.append(name)

    # Add key parameter
    if has_key_param:
        init_params.append("*")
        init_params.append("key: PRNGKeyArray")

    # Build the layer construction call
    layer_args = [name for name, _ in positional_params + keyword_params]
    layer_call_args = ", ".join(
        f"{name}=_spatial_tuple({name})"
        if "Sequence[" in format_type_annotation(dict(params)[name].annotation)
        else f"{name}={name}"
        for name in layer_args
    )
    if has_key_param:
        if layer_call_args:
            layer_call_args += ", key=key"
        else:
            layer_call_args = "key=key"

    call_helpers = {
        "single_input": "_call_single_input_layer",
        "single_input_keyed": "_call_single_input_keyed_layer",
        "batch_norm": "_call_batch_norm_layer",
        "multihead_attention": "_call_multihead_attention_layer",
        "gru_cell": "_call_gru_cell_layer",
        "lstm_cell": "_call_lstm_cell_layer",
    }
    call_body = f"""
        return {call_helpers[spec.call_kind]}(self, inputs, state, key=key)"""

    layer_properties = dict(spec.layer_properties)
    if is_portable(spec):
        upstream_fields = {item.name for item in fields(eqx_class)}
        for name, param in positional_params + keyword_params:
            if name in upstream_fields or name in spec.storage_paths:
                layer_properties[name] = (
                    spec.storage_paths.get(name, name),
                    format_type_annotation(param.annotation) or "Any",
                )

    # Generate fields only for parameters owned by the wrapper.
    attr_declarations = []
    init_body_assignments = []
    for name, param in positional_params + keyword_params:
        if name in layer_properties or name in spec.resolved_aliases:
            continue
        type_hint = parameter_annotation(spec, name, param) or "Any"
        attr_declarations.append(f"    {name}: {type_hint}")
        init_body_assignments.append(f"        self.{name} = {name}\n")

    properties = "".join(
        f"\n    @property\n"
        f"    def {name}(self) -> {type_hint}:\n"
        f"        return self.layer.{layer_attr}\n"
        for name, (layer_attr, type_hint) in layer_properties.items()
    )
    for name in spec.resolved_aliases:
        properties += f"\n    @property\n    def {name}(self) -> None:\n        return None\n"

    portable = is_portable(spec)
    dtype_property = ""
    if portable and any(name == "dtype" for name, _ in params):
        dtype_property = """
    config_bindings = (ConfigBinding(name="dtype", path="_graph_dtype"),)

    @property
    def _graph_dtype(self) -> str | None:
        return None if self.dtype is None else jnp.dtype(self.dtype).name
"""
    options = ""
    if portable:
        trainable = spec.category not in {"pool", "other"} and spec.eqx_class_name not in {
            "Identity",
            "RotaryPositionalEmbedding",
        }
        options = (
            f"\n{spec.eqx_class_name}.component_options = _owner_options("
            f"{spec.eqx_class_name}, has_key={has_key_param!r}, "
            f"trainable={trainable!r})\n"
        )

    # Build the class
    docstring = spec.docstring or f"Component wrapper for eqx.nn.{spec.eqx_class_name}."
    if portable:
        ports = (
            f"    ports = Ports(inputs={dict.fromkeys(spec.input_ports)!r}, "
            f"outputs={dict.fromkeys(spec.output_ports)!r})"
        ).replace(": None", ": PortType(dtype='array')")
    else:
        ports = f"    input_ports = {spec.input_ports!r}\n    output_ports = {spec.output_ports!r}"
    contract_expr = (
        "{"
        f"'call_kind': {spec.call_kind!r}, "
        f"'state_handling': {spec.state_handling!r}, "
        f"'key_handling': {spec.key_handling!r}"
        "}"
    )

    class_code = f'''
class {spec.eqx_class_name}({"Component" if portable else "GraphNode"}):
    """{docstring}"""

{ports}
    generated_wrapper_schema_version = EQUINOX_WRAPPER_SCHEMA_VERSION
    generated_for_equinox_version = EQUINOX_WRAPPER_EQUINOX_VERSION
    wrapper_contract = {contract_expr}

    layer: eqx.nn.{spec.eqx_class_name}
{chr(10).join(attr_declarations) if attr_declarations else ""}
    def __init__({", ".join(init_params)}):
{"".join(init_body_assignments)}        self.layer = eqx.nn.{spec.eqx_class_name}({layer_call_args})
{properties}{dtype_property}

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:{call_body}
'''
    return class_code + options


def generate_all_components() -> str:
    """Generate the complete module with all component wrappers."""
    # Group specs by category for organized output
    by_category: dict[str, list[ComponentSpec]] = {}
    for spec in COMPONENT_SPECS:
        by_category.setdefault(spec.category, []).append(spec)

    category_order = ["linear", "conv", "rnn", "norm", "pool", "attention", "other"]

    # Build the module
    header = '''"""Auto-generated Equinox component wrappers.

This file is generated by scripts/generate_eqx_components.py.
Do not edit manually - regenerate instead.

To regenerate:
    uv run scripts/generate_eqx_components.py
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any, Callable, Literal, Optional, Sequence, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.nn import State
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from feedbax.contracts.component import PortType
from feedbax.contracts.portable import ComponentOptions, ConfigBinding, MissingPrototypeInput, Ports
from feedbax.runtime.components import identity_activation, relu
from feedbax.runtime.graph import Component, GraphNode


EQUINOX_WRAPPER_SCHEMA_VERSION = "feedbax.equinox_wrappers.v3"
EQUINOX_WRAPPER_GENERATOR = "scripts/generate_eqx_components.py"
EQUINOX_WRAPPER_EQUINOX_VERSION = eqx.__version__


'''

    # Generate __all__
    all_names = [
        "EQUINOX_WRAPPER_SCHEMA_VERSION",
        "EQUINOX_WRAPPER_GENERATOR",
        "EQUINOX_WRAPPER_EQUINOX_VERSION",
        "EQUINOX_WRAPPER_CONTRACTS",
        *[spec.eqx_class_name for spec in COMPONENT_SPECS],
    ]
    all_str = "__all__ = [\n" + ",\n".join(f'    "{name}"' for name in all_names) + ",\n]\n"

    contracts = (
        "EQUINOX_WRAPPER_CONTRACTS = {\n"
        + "".join(
            "    "
            + repr(spec.eqx_class_name)
            + ": "
            + repr(
                {
                    "call_kind": spec.call_kind,
                    "state_handling": spec.state_handling,
                    "key_handling": spec.key_handling,
                }
            )
            + ",\n"
            for spec in COMPONENT_SPECS
        )
        + "}\n"
    )

    helpers = """

def _spatial_tuple(value: Any) -> Any:
    return tuple(map(_spatial_tuple, value)) if isinstance(value, list) else value


def _owner_options(
    cls: type[Component], *, has_key: bool = False, trainable: bool = False,
) -> ComponentOptions:
    def output_prototype(params: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
        missing = set(cls.input_ports) - inputs.keys()
        if missing:
            raise MissingPrototypeInput(f"Missing input prototypes for {sorted(missing)}")
        settings = {name: value for name, value in params.items() if name != "trainable"}
        key = jax.random.key(0)
        if has_key:
            settings["key"] = key

        def forward(values):
            layer, state = eqx.nn.make_with_state(cls)(**settings)
            return layer(values, state, key=key)[0]

        return jax.eval_shape(forward, dict(inputs))

    return ComponentOptions(
        output_prototype_fn=output_prototype,
        prototype_depends_on_connected_inputs=True,
        build_context_fields=("key",) if has_key else (),
        trainable_by_default=trainable,
    )


def _call_single_input_layer(
    component: GraphNode,
    inputs: dict[str, PyTree],
    state: State,
    *,
    key: PRNGKeyArray,
) -> tuple[dict[str, PyTree], State]:
    output = component.layer(inputs["input"])
    return {"output": output}, state


def _call_single_input_keyed_layer(
    component: GraphNode,
    inputs: dict[str, PyTree],
    state: State,
    *,
    key: PRNGKeyArray,
) -> tuple[dict[str, PyTree], State]:
    output = component.layer(inputs["input"], key=key)
    return {"output": output}, state


def _call_batch_norm_layer(
    component: GraphNode,
    inputs: dict[str, PyTree],
    state: State,
    *,
    key: PRNGKeyArray,
) -> tuple[dict[str, PyTree], State]:
    output, state = component.layer(inputs["input"], state, key=key)
    return {"output": output}, state


def _call_gru_cell_layer(
    component: GraphNode,
    inputs: dict[str, PyTree],
    state: State,
    *,
    key: PRNGKeyArray,
) -> tuple[dict[str, PyTree], State]:
    hidden = inputs["hidden"]
    new_hidden = component.layer(inputs["input"], hidden)
    return {"output": new_hidden, "hidden": new_hidden}, state


def _call_lstm_cell_layer(
    component: GraphNode,
    inputs: dict[str, PyTree],
    state: State,
    *,
    key: PRNGKeyArray,
) -> tuple[dict[str, PyTree], State]:
    hidden = inputs["hidden"]
    cell_state = inputs["cell"]
    new_hidden, new_cell = component.layer(inputs["input"], (hidden, cell_state), key=key)
    return {"output": new_hidden, "hidden": new_hidden, "cell": new_cell}, state


def _call_multihead_attention_layer(
    component: GraphNode,
    inputs: dict[str, PyTree],
    state: State,
    *,
    key: PRNGKeyArray,
) -> tuple[dict[str, PyTree], State]:
    output = component.layer(
        query=inputs["query"],
        key_=inputs["key_"],
        value=inputs["value"],
        key=key,
    )
    return {"output": output}, state
"""

    # Generate classes grouped by category
    body_parts = []
    for category in category_order:
        if category not in by_category:
            continue
        specs = by_category[category]
        category_header = f"\n# {'=' * 60}\n# {category.upper()} LAYERS\n# {'=' * 60}\n"
        body_parts.append(category_header)
        for spec in specs:
            body_parts.append(generate_component_class(spec))

    return header + all_str + contracts + helpers + "".join(body_parts)


def generate_histories() -> str:
    """Render frozen legacy evidence for the current wrapper owners."""
    imports = ", ".join(LEGACY_CONFIG_FIELDS)
    records = "\n".join(
        f"    {name}: History(legacy=(LegacyEndpoint('feedbax.equinox.{name}', '1', 1, "
        f"partial(_revision_one_fields, fields=frozenset({fields!r}))),)),"
        for name, fields in LEGACY_CONFIG_FIELDS.items()
    )
    return f'''"""Generated frozen Equinox wire evidence; regenerate with generate_eqx_components.py."""

from functools import partial

from feedbax.component_registry.history import History, LegacyEndpoint
from feedbax.components.equinox import {imports}


def _revision_one_fields(params, *, fields):
    return set(params) <= fields


HISTORIES = {{
{records}
}}
'''


def main():
    """Generate and write the components file."""
    output_path = Path(__file__).parent.parent / "feedbax" / "components" / "equinox.py"

    print("Generating Equinox component wrappers...")
    content = generate_all_components()

    output_path.write_text(content)
    history_path = output_path.parent.parent / "component_migrations" / "equinox.py"
    history_path.write_text(generate_histories())
    try:
        subprocess.run(["ruff", "format", str(output_path), str(history_path)], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"Warning: could not format generated file with ruff: {exc}")
    print(f"Written to: {output_path}")

    # Count generated classes
    n_classes = len(COMPONENT_SPECS)
    print(f"Generated {n_classes} component wrappers.")


if __name__ == "__main__":
    main()
