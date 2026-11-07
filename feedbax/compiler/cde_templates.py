"""Native graph specifications for neural controlled differential equation controllers."""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp

from feedbax.contracts.graph import (
    CanvasPositionSpec,
    ComponentSpec,
    GraphSpec,
    GraphUIState,
    NodeUIState,
    WireSpec,
)



def _node(type_: str, inputs: list[str], outputs: list[str], **params) -> ComponentSpec:
    return ComponentSpec(type=f"feedbax.{type_}", params=params, input_ports=inputs, output_ports=outputs)


def _wire(src: str, output: str, dst: str, input_: str, **kwargs) -> WireSpec:
    return WireSpec(
        source_node=src, source_port=output, target_node=dst, target_port=input_, **kwargs
    )


def cde_step_spec(
    obs_dim: int = 1,
    hidden_dim: int = 1,
    action_dim: int = 1,
    *,
    hidden_sizes: tuple[int, ...] = (128,),
    decay: float | None = None,
    alpha: float | None = None,
    dtype: str = "float32",
    bias_init: float | None = None,
) -> GraphSpec:
    """Build the CDE Euler update with explicit prior observation and hidden inputs."""
    step_inputs = ["obs", "obs_prev", "h_prev"]
    nodes = {
        f"{name}_in": _node("Input", ["input"], [name], output_port=name) for name in step_inputs
    }
    nodes.update(
        subtract=_node("Subtract", ["a", "b"], ["out"]),
        vf=_node(
            "MLP",
            ["input"],
            ["output"],
            input_size=hidden_dim,
            output_size=hidden_dim * obs_dim,
            hidden_sizes=list(hidden_sizes),
            activation="relu",
            final_activation="tanh",
            dtype=dtype,
        ),
        reshape=_node("Reshape", ["input"], ["output"], shape=[hidden_dim, obs_dim]),
        matmul=_node("MatMul", ["a", "b"], ["out"]),
    )
    wires = [
        _wire("obs_in", "obs", "subtract", "a"),
        _wire("obs_prev_in", "obs_prev", "subtract", "b"),
        _wire("h_prev_in", "h_prev", "vf", "input"),
        _wire("vf", "output", "reshape", "input"),
        _wire("reshape", "output", "matmul", "a"),
        _wire("subtract", "out", "matmul", "b"),
        _wire("h_prev_in", "h_prev", "sum_h", "a"),
        _wire("matmul", "out", "sum_h", "b"),
    ]
    sum_ports = ["a", "b"]
    if decay is not None:
        nodes["decay"] = _node("Scale", ["input"], ["output"], scale=-decay, fixed=True)
        wires.extend(
            [
                _wire("h_prev_in", "h_prev", "decay", "input"),
                _wire("decay", "output", "sum_h", "c"),
            ]
        )
        sum_ports.append("c")
    if alpha is not None:
        nodes.update(
            negate_h=_node("Scale", ["input"], ["output"], scale=-1.0, fixed=True),
            gru_gate=_node(
                "GRU",
                ["input", "hidden"],
                ["output", "hidden"],
                input_size=obs_dim,
                hidden_size=hidden_dim,
                dtype=dtype,
                stateful=False,
            ),
            alpha=_node("Scale", ["input"], ["output"], scale=alpha, fixed=True),
        )
        sum_ports.append("c" if decay is None else "d")
        wires.extend(
            [
                _wire("h_prev_in", "h_prev", "negate_h", "input"),
                _wire("obs_in", "obs", "gru_gate", "input"),
                _wire("negate_h", "output", "gru_gate", "hidden"),
                _wire("gru_gate", "output", "alpha", "input"),
                _wire("alpha", "output", "sum_h", sum_ports[-1]),
            ]
        )
    nodes.update(
        sum_h=_node("Sum", sum_ports, ["output"]),
        linear=_node(
            "Linear",
            ["input"],
            ["output"],
            input_size=hidden_dim,
            output_size=action_dim,
            dtype=dtype,
            bias_init=bias_init,
        ),
        sigmoid=_node("Sigmoid", ["input"], ["output"]),
    )
    wires.extend(
        [
            _wire("sum_h", "output", "linear", "input"),
            _wire("linear", "output", "sigmoid", "input"),
        ]
    )
    return GraphSpec(
        nodes=nodes,
        wires=wires,
        input_ports=step_inputs,
        output_ports=["h_new", "action"],
        input_bindings={name: (f"{name}_in", "input") for name in step_inputs},
        output_bindings={"h_new": ("sum_h", "output"), "action": ("sigmoid", "output")},
    )


def cde_controller_spec(
    obs_dim: int,
    hidden_dim: int,
    out_size: int,
    vf_width: int = 64,
    vf_depth: int = 2,
    decay: float = 0.1,
    use_anti_nf: bool = True,
    alpha: float = 1.0,
    dtype: Any = jnp.float32,
) -> GraphSpec:
    """Author a recurrent CDE over concatenated task input and feedback.

    The hidden and observation cycle wires carry the preceding sample.
    On reset, the hidden wire reads the live learned `h0` source, and
    the observation wire starts at zero. Pass the returned carry into
    subsequent calls to :meth:`Graph.step` to continue the same trajectory.
    """
    if not math.isfinite(decay) or not 0.0 <= decay <= 1.0:
        raise ValueError(f"CDE decay must be finite and in the range [0, 1]; got {decay!r}.")
    if vf_depth < 0:
        raise ValueError(f"CDE vector field depth must be nonnegative; got {vf_depth!r}.")

    dtype_name = jnp.dtype(dtype).name
    spec = cde_step_spec(
        obs_dim,
        hidden_dim,
        out_size,
        hidden_sizes=(vf_width,) * vf_depth,
        decay=decay,
        alpha=alpha if use_anti_nf else None,
        dtype=dtype_name,
        bias_init=-5.0,
    )
    spec.nodes.update(
        observation=_node("Mux", ["in_0", "in_1"], ["output"], n_inputs=2, dtype=dtype_name),
        h0=_node(
            "Constant", [], ["output"], value=[0.0] * hidden_dim, dtype=dtype_name, trainable=True
        ),
    )
    spec.wires.extend(
        [
            _wire("observation", "output", "obs_in", "input"),
            _wire(
                "observation",
                "output",
                "obs_prev_in",
                "input",
                temporality="recurrent",
                recurrent_initializer={
                    "kind": "constant",
                    "value": [0.0] * obs_dim,
                    "dtype": dtype_name,
                },
            ),
            _wire(
                "sum_h",
                "output",
                "h_prev_in",
                "input",
                temporality="recurrent",
                recurrent_initializer={
                    "kind": "node-output",
                    "source_node": "h0",
                    "source_port": "output",
                    "state_slot": "hidden",
                },
            ),
        ]
    )
    return spec.model_copy(
        update={
            "input_ports": ["input", "feedback"],
            "input_bindings": {
                "input": ("observation", "in_0"),
                "feedback": ("observation", "in_1"),
            },
            "output_ports": ["output", "hidden"],
            "output_bindings": {
                "output": spec.output_bindings["action"],
                "hidden": spec.output_bindings["h_new"],
            },
        }
    )



from pydantic import Field
from feedbax.contracts.portable import GraphTemplate, ComponentOptions, expansion_configuration

_CDEConfig = expansion_configuration(cde_step_spec)


class _DecayConfig(_CDEConfig):
    decay: float | None = 0.1


class _AntiNFConfig(_CDEConfig):
    alpha: float | None = 0.1


class _HybridConfig(_DecayConfig):
    alpha: float | None = 0.1


class CDEStandard(GraphTemplate):
    """CDE update from observation increments with a sigmoid action readout."""

    config = _CDEConfig
    component_options = ComponentOptions(is_composite=True, template_id="feedbax.templates.cde_standard", template_kind="executable")

    @classmethod
    def expand(cls, config):
        return cde_step_spec(**config.model_dump())


class CDEDecay(CDEStandard):
    """CDE with fixed hidden-state decay."""

    config = _DecayConfig
    component_options = ComponentOptions(is_composite=True, template_id="feedbax.templates.cde_decay", template_kind="executable")


class CDEAntiNF(CDEStandard):
    """CDE with GRU-gated negative feedback."""

    config = _AntiNFConfig
    component_options = ComponentOptions(is_composite=True, template_id="feedbax.templates.cde_anti_nf", template_kind="executable")


class CDEHybridV9b(CDEStandard):
    """CDE with fixed decay and GRU-gated negative feedback."""

    config = _HybridConfig
    component_options = ComponentOptions(is_composite=True, template_id="feedbax.templates.cde_hybrid_v9b", template_kind="executable")
