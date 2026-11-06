from __future__ import annotations

import math
from typing import Any, Protocol

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray

from feedbax.contracts.component import PortType, PortTypeSpec
from feedbax.contracts.graph import ComponentSpec, GraphSpec, GraphUIState, NodeUIState, WireSpec

from .declarations import DeclaredComponent, declare_component


class _Registry(Protocol):
    def register(self, meta: DeclaredComponent) -> None: ...


def _node(type_: str, inputs: list[str], outputs: list[str], **params) -> ComponentSpec:
    return ComponentSpec(type=type_, params=params, input_ports=inputs, output_ports=outputs)


def _wire(src: str, output: str, dst: str, input_: str, **kwargs) -> WireSpec:
    return WireSpec(
        source_node=src, source_port=output, target_node=dst, target_port=input_, **kwargs
    )


def _cde_keys(key: PRNGKeyArray) -> dict[str, PRNGKeyArray]:
    return dict(zip(("vf", "linear", "h0", "gru_gate"), jr.split(key, 4), strict=True))


def cde_initialization_keys(
    key: PRNGKeyArray, *, use_anti_nf: bool, path: tuple[str, ...] = ()
) -> dict[tuple[str, ...], PRNGKeyArray]:
    """Supply the controller's neural initialization keys to the graph compiler."""
    keys = _cde_keys(key)
    neural_nodes = ("vf", "linear", "gru_gate") if use_anti_nf else ("vf", "linear")
    return {(*path, node): keys[node] for node in neural_nodes}


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
    *,
    key: PRNGKeyArray | None = None,
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
    h0_key = _cde_keys(jr.key(0) if key is None else key)["h0"]
    h0 = 0.01 * jr.normal(h0_key, (hidden_dim,), dtype=dtype)
    spec.nodes.update(
        observation=_node("Mux", ["in_0", "in_1"], ["output"], n_inputs=2, dtype=dtype_name),
        h0=_node("Constant", [], ["output"], value=h0.tolist(), dtype=dtype_name, trainable=True),
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


def _template_layout(spec: GraphSpec) -> GraphUIState:
    positions = {
        "obs_in": (20, 40),
        "obs_prev_in": (20, 160),
        "h_prev_in": (20, 280),
        "subtract": (160, 40),
        "vf": (160, 280),
        "reshape": (320, 280),
        "matmul": (480, 40),
        "decay": (320, 160),
        "negate_h": (320, 360),
        "gru_gate": (480, 360),
        "alpha": (640, 360),
        "sum_h": (640, 160),
        "linear": (800, 160),
        "sigmoid": (960, 160),
    }
    return GraphUIState(
        node_states={
            name: NodeUIState(position={"x": x, "y": y}, collapsed=False, selected=False)
            for name, (x, y) in positions.items()
            if name in spec.nodes
        }
    )


def register_cde_templates(registry: _Registry) -> None:
    variants = (
        (
            "CDE Standard",
            "cde_standard",
            "BrainCircuit",
            "CDE update from observation increments with a sigmoid action readout.",
            None,
            None,
        ),
        (
            "CDE + Decay",
            "cde_decay",
            "TrendingUp",
            "CDE with fixed hidden-state decay.",
            0.1,
            None,
        ),
        (
            "CDE + Anti-NF",
            "cde_anti_nf",
            "BrainCog",
            "CDE with GRU-gated negative feedback.",
            None,
            0.1,
        ),
        (
            "CDE Hybrid v9b",
            "cde_hybrid_v9b",
            "Sparkles",
            "CDE with fixed decay and GRU-gated negative feedback.",
            0.1,
            0.1,
        ),
    )
    for name, suffix, icon, description, decay, alpha in variants:
        spec = cde_step_spec(decay=decay, alpha=alpha)
        registry.register(
            declare_component(
                name=name,
                category="CDE Controllers",
                description=description,
                param_schema=[],
                input_ports=spec.input_ports,
                output_ports=spec.output_ports,
                icon=icon,
                is_composite=True,
                port_types=PortTypeSpec(
                    inputs={port: PortType(dtype="vector") for port in spec.input_ports},
                    outputs={port: PortType(dtype="vector") for port in spec.output_ports},
                ),
                template_graph=spec,
                template_ui_state=_template_layout(spec),
                template_id=f"feedbax.templates.{suffix}",
                template_kind="executable",
            )
        )
