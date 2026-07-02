"""Built-in graph template/spec factories.

Templates are the canonical representation for Feedbax-provided composite
structures. They produce portable ``GraphSpec`` payloads that can be stored,
opened in Studio, and instantiated by ``spec_to_graph`` without relying on a
separate backend-special model type.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from collections.abc import Mapping
from typing import Any, Literal

from feedbax.contracts.graph import ComponentSpec, GraphMetadata, GraphSpec, WireSpec
from feedbax.models.networks import (
    PopulationStructure,
    lower_population_constraints,
    population_structure_from_spec,
)


TemplateKind = Literal["executable", "display"]


@dataclass(frozen=True)
class GraphTemplateMetadata:
    """Metadata describing a built-in graph template."""

    id: str
    name: str
    category: str
    kind: TemplateKind
    description: str
    component_type: str | None = None


def recurrent_zero_initializer(width: int | None = None, *, state_slot: str) -> dict[str, Any]:
    """Return the canonical zero initializer for a recurrent edge."""

    initializer: dict[str, Any] = {
        "kind": "zeros",
        "scope": "trial",
        "source": "state_initializer",
        "state_slot": state_slot,
    }
    if width is not None:
        initializer["shape"] = [int(width)]
    return initializer


def recurrent_graph_input_initializer(source: str, *, state_slot: str) -> dict[str, Any]:
    """Return a recurrent initializer supplied by an external graph input."""

    return {
        "kind": "graph-input",
        "scope": "trial",
        "source": str(source),
        "state_slot": state_slot,
    }


def network_recurrent_wires(
    cell_type: str,
    hidden_size: int,
    *,
    hidden_source_node: str = "cell",
    hidden_source_port: str = "hidden",
) -> list[WireSpec]:
    """Return recurrent carry wires for a Network cell subgraph."""

    wires = [
        WireSpec(
            source_node=hidden_source_node,
            source_port=hidden_source_port,
            target_node="cell",
            target_port="hidden",
            temporality="recurrent",
            recurrent_initializer=recurrent_zero_initializer(
                hidden_size,
                state_slot="hidden",
            ),
        )
    ]
    if cell_type == "LSTM":
        wires.append(
            WireSpec(
                source_node="cell",
                source_port="cell",
                target_node="cell",
                target_port="cell",
                temporality="recurrent",
                recurrent_initializer=recurrent_zero_initializer(
                    hidden_size,
                    state_slot="cell",
                ),
            )
        )
    return wires


def recurrent_controller_template_graph(
    *,
    input_size: int,
    hidden_size: int,
    out_size: int,
    cell_type: str = "GRU",
    out_nonlinearity: str = "identity",
    population_structure: PopulationStructure | Mapping[str, object] | None = None,
    name: str = "Recurrent controller",
    description: str = "Explicit recurrent controller graph template",
) -> GraphSpec:
    """Build an explicit recurrent controller graph with ordinary nodes and wires."""

    normalized_cell_type = "LSTM" if cell_type in {"LSTM", "LSTMCell"} else "GRU"
    parameter_constraints = []
    if population_structure is not None:
        parameter_constraints = list(
            lower_population_constraints(
                population_structure,
                hidden_size=int(hidden_size),
                input_size=int(input_size),
                out_size=int(out_size),
                cell_type=normalized_cell_type,
            )
        )
    now = datetime.now().isoformat()
    nodes = {
        "input_mux": ComponentSpec(
            type="Mux",
            params={"n_inputs": 2},
            input_ports=["in_0", "in_1"],
            output_ports=["output"],
        ),
        "cell": ComponentSpec(
            type=normalized_cell_type,
            params={
                "input_size": int(input_size),
                "hidden_size": int(hidden_size),
            },
            input_ports=["input", "hidden", "cell"]
            if normalized_cell_type == "LSTM"
            else ["input", "hidden"],
            output_ports=["output", "hidden", "cell"]
            if normalized_cell_type == "LSTM"
            else ["output", "hidden"],
        ),
        "readout": ComponentSpec(
            type="Linear",
            params={
                "input_size": int(hidden_size),
                "output_size": int(out_size),
                "use_bias": True,
                "activation": out_nonlinearity,
            },
            input_ports=["input"],
            output_ports=["output"],
        ),
    }
    wires = [
        WireSpec(
            source_node="input_mux",
            source_port="output",
            target_node="cell",
            target_port="input",
        ),
    ]
    input_ports = ["input", "feedback"]
    input_bindings = {"input": ("input_mux", "in_0"), "feedback": ("input_mux", "in_1")}
    wires.extend(
        [
            WireSpec(
                source_node="cell",
                source_port="hidden",
                target_node="readout",
                target_port="input",
            ),
            *network_recurrent_wires(
                normalized_cell_type,
                int(hidden_size),
            ),
        ]
    )
    return GraphSpec(
        nodes=nodes,
        wires=wires,
        input_ports=input_ports,
        output_ports=["output", "hidden"],
        input_bindings=input_bindings,
        output_bindings={
            "output": ("readout", "output"),
            "hidden": ("cell", "hidden"),
        },
        parameter_constraints=parameter_constraints,
        metadata=GraphMetadata(
            name=name,
            description=description,
            created_at=now,
            updated_at=now,
            version="1.0.0",
        ),
    )


def network_template_graph(params: dict[str, Any] | None = None) -> GraphSpec:
    """Build the legacy recurrent controller preset as an explicit graph.

    ``Network`` is no longer a durable wrapper component. This compatibility
    entry point returns the ordinary template graph directly.
    """
    params = dict(params or {})
    input_size = int(params.get("input_size", 6))
    hidden_size = int(params.get("hidden_size", 100))
    out_size = int(params.get("out_size", params.get("output_size", 2)))
    hidden_type = str(params.get("hidden_type", "GRUCell"))
    cell_type = "LSTM" if hidden_type in {"LSTM", "LSTMCell"} else "GRU"
    out_nonlinearity = str(params.get("out_nonlinearity", "identity"))
    raw_population_structure = params.get("population_structure")
    population_structure = None
    if isinstance(raw_population_structure, PopulationStructure):
        population_structure = raw_population_structure
    elif isinstance(raw_population_structure, Mapping):
        population_structure = population_structure_from_spec(hidden_size, raw_population_structure)
    return recurrent_controller_template_graph(
        input_size=input_size,
        hidden_size=hidden_size,
        out_size=out_size,
        cell_type=cell_type,
        out_nonlinearity=out_nonlinearity,
        population_structure=population_structure,
    )


def simple_feedback_template_graph(params: dict[str, Any] | None = None) -> GraphSpec:
    """Build the canonical SimpleFeedback graph template.

    The template is executable when the referenced leaf node builders support
    the chosen mechanics and feedback-channel parameters. Legacy Python
    constructors may still be used as compatibility factories, but this graph
    shape is the durable portable representation.
    """

    params = dict(params or {})
    network_params = dict(params.get("network", {}))
    network_params.setdefault("input_size", 8)
    mechanics_type = str(params.get("mechanics_type", "PointMass"))
    mechanics_params = dict(params.get("mechanics", {"dt": 0.01}))
    motor_delay = int(params.get("motor_delay", 0))
    feedback_delay = int(params.get("feedback_delay", 0))
    feedback_noise_std = float(params.get("feedback_noise_std", 0.0) or 0.0)
    motor_noise_std = float(params.get("motor_noise_std", 0.0) or 0.0)
    tau_rise = float(params.get("tau_rise", 0.0) or 0.0)
    tau_decay = float(params.get("tau_decay", 0.0) or 0.0)

    network_template = network_template_graph(network_params)
    nodes: dict[str, ComponentSpec] = {
        **network_template.nodes,
        "feedback": ComponentSpec(
            type="Channel",
            params={
                "delay": feedback_delay,
                "noise_std": feedback_noise_std,
                "add_noise": feedback_noise_std != 0.0,
            },
            input_ports=["input"],
            output_ports=["output"],
        ),
        "feedback_ravel": ComponentSpec(
            type="Ravel",
            params={},
            input_ports=["input"],
            output_ports=["output"],
        ),
        "efferent": ComponentSpec(
            type="Channel",
            params={
                "delay": motor_delay,
                "noise_std": motor_noise_std,
                "add_noise": motor_noise_std != 0.0,
            },
            input_ports=["input"],
            output_ports=["output"],
        ),
        "mechanics": ComponentSpec(
            type=mechanics_type,
            params=mechanics_params,
            input_ports=["force"],
            output_ports=["effector", "state"],
        ),
    }
    wires = [
        *network_template.wires,
        WireSpec(
            source_node="feedback",
            source_port="output",
            target_node=network_template.input_bindings["feedback"][0],
            target_port=network_template.input_bindings["feedback"][1],
        ),
        WireSpec(
            source_node=network_template.output_bindings["output"][0],
            source_port=network_template.output_bindings["output"][1],
            target_node="efferent",
            target_port="input",
        ),
    ]
    if tau_rise == 0.0 and tau_decay == 0.0:
        wires.append(
            WireSpec(
                source_node="efferent",
                source_port="output",
                target_node="mechanics",
                target_port="force",
            )
        )
    else:
        nodes["force_filter"] = ComponentSpec(
            type="FirstOrderFilter",
            params={
                "tau_rise": tau_rise,
                "tau_decay": tau_decay,
                "dt": float(mechanics_params.get("dt", 0.01)),
                "init_value": 0.0,
            },
            input_ports=["input"],
            output_ports=["output"],
        )
        wires.append(
            WireSpec(
                source_node="efferent",
                source_port="output",
                target_node="force_filter",
                target_port="input",
            )
        )
        wires.append(
            WireSpec(
                source_node="force_filter",
                source_port="output",
                target_node="mechanics",
                target_port="force",
            )
        )
    wires.append(
        WireSpec(
            source_node="mechanics",
            source_port="effector",
            target_node="feedback_ravel",
            target_port="input",
            temporality="recurrent",
            recurrent_initializer={
                "kind": "state_output",
                "scope": "trial",
                "source": "state_initializer",
                "state_slot": "effector",
            },
        )
    )
    wires.append(
        WireSpec(
            source_node="feedback_ravel",
            source_port="output",
            target_node="feedback",
            target_port="input",
        )
    )

    return GraphSpec(
        nodes=nodes,
        wires=wires,
        input_ports=["input"],
        output_ports=["effector"],
        input_bindings={"input": network_template.input_bindings["input"]},
        output_bindings={"effector": ("mechanics", "effector")},
        parameter_constraints=network_template.parameter_constraints,
    )


BUILTIN_GRAPH_TEMPLATES: tuple[GraphTemplateMetadata, ...] = (
    GraphTemplateMetadata(
        id="feedbax.templates.recurrent_controller",
        name="Recurrent Controller",
        category="Neural Networks",
        kind="executable",
        description="Explicit Mux, recurrent cell, and readout controller graph.",
        component_type="Subgraph",
    ),
    GraphTemplateMetadata(
        id="feedbax.templates.simple_feedback",
        name="Simple Feedback Loop",
        category="Sensorimotor",
        kind="executable",
        description="Feedback channel, controller, efferent path, and mechanics loop.",
        component_type="Subgraph",
    ),
)


__all__ = [
    "BUILTIN_GRAPH_TEMPLATES",
    "GraphTemplateMetadata",
    "network_recurrent_wires",
    "network_template_graph",
    "recurrent_controller_template_graph",
    "recurrent_zero_initializer",
    "simple_feedback_template_graph",
]
