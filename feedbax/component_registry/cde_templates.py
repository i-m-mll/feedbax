from __future__ import annotations

from typing import Protocol

from feedbax.compiler.cde_templates import cde_step_spec
from feedbax.contracts.component import PortType, PortTypeSpec
from feedbax.contracts.graph import GraphSpec, GraphUIState, NodeUIState

from .declarations import DeclaredComponent, declare_component


class _Registry(Protocol):
    def register(self, meta: DeclaredComponent) -> None: ...


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
