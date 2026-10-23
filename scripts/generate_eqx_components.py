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
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import equinox.nn as nn

TRAINABLE_DTYPE_PARAMS = {"dtype"}
DEFAULT_TRAINABLE_DTYPE = "jnp.float32"


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
        docstring="Multi-layer perceptron with configurable depth and activations.",
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
        docstring="Embedding layer for discrete tokens.",
    ),
    ComponentSpec(
        "Dropout",
        call_kind="single_input_keyed",
        key_handling="forwarded",
        category="other",
        docstring="Dropout layer that honors its configured inference/deterministic settings.",
    ),
    ComponentSpec(
        "PReLU",
        category="other",
        docstring="Parametric ReLU activation.",
    ),
]


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

        # Handle Optional (Union with None)
        if origin_name == "Union" and type(None) in args:
            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1:
                return f"Optional[{format_type_annotation(non_none_args[0])}]"

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
        # For callable defaults like activation functions
        if hasattr(default, "__module__") and hasattr(default, "__name__"):
            module = default.__module__
            name = default.__name__
            # Handle JAX functions
            if module.startswith("jax"):
                if "nn" in module:
                    return f"jax.nn.{name}"
                return f"jax.{name}"
            # Handle lambdas - use a sensible default
            if name == "<lambda>":
                return "lambda x: x"
            return name
        if hasattr(default, "__name__"):
            name = default.__name__
            if name == "<lambda>":
                return "lambda x: x"
            return name
        return "None"  # Can't represent this default
    # Handle dataclass field defaults (MISSING, factory, etc.)
    default_str = str(default)
    if "MISSING" in default_str or "factory" in default_str.lower():
        return "None"
    return repr(default)


def default_for_param(name: str, default) -> str:
    """Return generated defaults, overriding trainable dtype to float32."""

    if name in TRAINABLE_DTYPE_PARAMS and default is None:
        return DEFAULT_TRAINABLE_DTYPE
    return format_default(default)


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
    init_body_assignments = []
    stored_attrs = []

    for name, param in positional_params:
        type_hint = format_type_annotation(param.annotation)
        if type_hint:
            init_params.append(f"{name}: {type_hint}")
        else:
            init_params.append(name)
        stored_attrs.append(name)
        init_body_assignments.append(f"self.{name} = {name}")

    for name, param in keyword_params:
        type_hint = format_type_annotation(param.annotation)
        default = default_for_param(name, param.default)
        if type_hint and default:
            init_params.append(f"{name}: {type_hint} = {default}")
        elif type_hint:
            init_params.append(f"{name}: {type_hint}")
        elif default:
            init_params.append(f"{name} = {default}")
        else:
            init_params.append(name)
        stored_attrs.append(name)
        init_body_assignments.append(f"self.{name} = {name}")

    # Add key parameter
    if has_key_param:
        init_params.append("*")
        init_params.append("key: PRNGKeyArray")

    # Build the layer construction call
    layer_args = [name for name, _ in positional_params + keyword_params]
    layer_call_args = ", ".join(f"{name}={name}" for name in layer_args)
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

    # Generate attribute declarations for stored params
    attr_declarations = []
    for name, param in positional_params + keyword_params:
        type_hint = format_type_annotation(param.annotation)
        if type_hint:
            attr_declarations.append(f"    {name}: {type_hint}")
        else:
            attr_declarations.append(f"    {name}: Any")

    # Build the class
    docstring = spec.docstring or f"Component wrapper for eqx.nn.{spec.eqx_class_name}."
    input_ports_str = repr(spec.input_ports)
    output_ports_str = repr(spec.output_ports)
    contract_expr = (
        "{"
        f"'call_kind': {spec.call_kind!r}, "
        f"'state_handling': {spec.state_handling!r}, "
        f"'key_handling': {spec.key_handling!r}"
        "}"
    )

    class_code = f'''
class {spec.eqx_class_name}(Component):
    """{docstring}"""

    input_ports = {input_ports_str}
    output_ports = {output_ports_str}
    generated_wrapper_schema_version = EQUINOX_WRAPPER_SCHEMA_VERSION
    generated_for_equinox_version = EQUINOX_WRAPPER_EQUINOX_VERSION
    wrapper_contract = {contract_expr}

    layer: eqx.nn.{spec.eqx_class_name}
{chr(10).join(attr_declarations) if attr_declarations else ""}
    def __init__({", ".join(init_params)}):
{chr(10).join(f"        {line}" for line in init_body_assignments)}
        self.layer = eqx.nn.{spec.eqx_class_name}({layer_call_args})

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:{call_body}
'''
    return class_code


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

from collections.abc import Hashable
from typing import Any, Callable, Literal, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import equinox as eqx
from equinox.nn import State
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from feedbax.runtime.graph import Component


EQUINOX_WRAPPER_SCHEMA_VERSION = "feedbax.equinox_wrappers.v2"
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

def _call_single_input_layer(
    component: Component,
    inputs: dict[str, PyTree],
    state: State,
    *,
    key: PRNGKeyArray,
) -> tuple[dict[str, PyTree], State]:
    output = component.layer(inputs["input"])
    return {"output": output}, state


def _call_single_input_keyed_layer(
    component: Component,
    inputs: dict[str, PyTree],
    state: State,
    *,
    key: PRNGKeyArray,
) -> tuple[dict[str, PyTree], State]:
    output = component.layer(inputs["input"], key=key)
    return {"output": output}, state


def _call_batch_norm_layer(
    component: Component,
    inputs: dict[str, PyTree],
    state: State,
    *,
    key: PRNGKeyArray,
) -> tuple[dict[str, PyTree], State]:
    output, state = component.layer(inputs["input"], state, key=key)
    return {"output": output}, state


def _call_gru_cell_layer(
    component: Component,
    inputs: dict[str, PyTree],
    state: State,
    *,
    key: PRNGKeyArray,
) -> tuple[dict[str, PyTree], State]:
    hidden = inputs["hidden"]
    new_hidden = component.layer(inputs["input"], hidden)
    return {"output": new_hidden, "hidden": new_hidden}, state


def _call_lstm_cell_layer(
    component: Component,
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
    component: Component,
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
            try:
                class_code = generate_component_class(spec)
                body_parts.append(class_code)
            except Exception as e:
                print(f"Warning: Failed to generate {spec.eqx_class_name}: {e}")

    return header + all_str + contracts + helpers + "".join(body_parts)


def main():
    """Generate and write the components file."""
    output_path = Path(__file__).parent.parent / "feedbax" / "components" / "equinox.py"

    print("Generating Equinox component wrappers...")
    content = generate_all_components()

    output_path.write_text(content)
    try:
        subprocess.run(["ruff", "format", str(output_path)], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"Warning: could not format generated file with ruff: {exc}")
    print(f"Written to: {output_path}")

    # Count generated classes
    n_classes = len(COMPONENT_SPECS)
    print(f"Generated {n_classes} component wrappers.")


if __name__ == "__main__":
    main()
