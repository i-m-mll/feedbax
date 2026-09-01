from __future__ import annotations

from typing import Protocol

from feedbax.contracts.component import PortType, PortTypeSpec
from feedbax.contracts.domain import MECHANICS_DOMAIN_ID
from feedbax.contracts.graphs.templates import (
    BUILTIN_GRAPH_TEMPLATES,
    recurrent_controller_template_graph,
    simple_feedback_template_graph,
)
from feedbax.contracts.graphs.mechanics_templates import (
    mass_spring_damper_template_graph,
    point_mass_with_muscles_template_graph,
    two_link_arm_6muscle_template_graph,
)

from .declarations import DeclaredComponent, _template_ui_state, declare_component


class _Registry(Protocol):
    def register(self, meta: DeclaredComponent) -> None: ...


def register_builtin_graph_templates(registry: _Registry) -> None:
    metadata = {template.id: template for template in BUILTIN_GRAPH_TEMPLATES}

    network_meta = metadata["feedbax.templates.recurrent_controller"]
    network_graph = recurrent_controller_template_graph(
        input_size=6,
        hidden_size=100,
        out_size=2,
        cell_type="GRU",
        out_nonlinearity="tanh",
    )
    registry.register(
        declare_component(
            name=network_meta.name,
            category=network_meta.category,
            description=network_meta.description,
            param_schema=[],
            input_ports=list(network_graph.input_ports),
            output_ports=list(network_graph.output_ports),
            icon="CircuitBoard",
            is_composite=True,
            port_types=PortTypeSpec(
                inputs={
                    "input": PortType(dtype="vector"),
                    "feedback": PortType(dtype="vector"),
                },
                outputs={
                    "output": PortType(dtype="vector"),
                    "hidden": PortType(dtype="vector"),
                },
            ),
            template_graph=network_graph,
            template_ui_state=_template_ui_state(
                {
                    "input_mux": (80, 160),
                    "cell": (280, 160),
                    "readout": (500, 160),
                }
            ),
            template_id=network_meta.id,
            template_kind=network_meta.kind,
            trainable_by_default=True,
        )
    )

    msd_graph = mass_spring_damper_template_graph()
    registry.register(
        declare_component(
            name="MassSpringDamper",
            category="Templates",
            description="Executable mechanics mass-spring-damper template.",
            param_schema=[],
            input_ports=["force"],
            output_ports=["position", "velocity"],
            icon="Waves",
            domain=MECHANICS_DOMAIN_ID,
            interior_domain=MECHANICS_DOMAIN_ID,
            is_composite=True,
            port_types=PortTypeSpec(
                inputs={"force": PortType(dtype="scalar")},
                outputs={
                    "position": PortType(dtype="scalar"),
                    "velocity": PortType(dtype="scalar"),
                },
            ),
            template_graph=msd_graph,
            template_id="feedbax.templates.mechanics.mass_spring_damper",
            template_kind="executable",
        )
    )

    muscle_graph = point_mass_with_muscles_template_graph()
    registry.register(
        declare_component(
            name="PointMassWithMuscles",
            category="Templates",
            description="Executable mechanics point mass with one antagonistic muscle pair.",
            param_schema=[],
            input_ports=["flexor", "extensor"],
            output_ports=["position", "velocity"],
            icon="Dumbbell",
            domain=MECHANICS_DOMAIN_ID,
            interior_domain=MECHANICS_DOMAIN_ID,
            is_composite=True,
            port_types=PortTypeSpec(
                inputs={
                    "flexor": PortType(dtype="scalar"),
                    "extensor": PortType(dtype="scalar"),
                },
                outputs={
                    "position": PortType(dtype="scalar"),
                    "velocity": PortType(dtype="scalar"),
                },
            ),
            template_graph=muscle_graph,
            template_id="feedbax.templates.mechanics.point_mass_with_muscles",
            template_kind="executable",
        )
    )

    arm_graph = two_link_arm_6muscle_template_graph()
    registry.register(
        declare_component(
            name="TwoLinkArm6Muscle",
            category="Templates",
            description=(
                "Editable planar multibody two-link arm with six rigid-tendon muscle paths."
            ),
            param_schema=[],
            input_ports=["excitation"],
            output_ports=["effector", "state"],
            icon="Activity",
            domain=MECHANICS_DOMAIN_ID,
            interior_domain=MECHANICS_DOMAIN_ID,
            is_composite=True,
            port_types=PortTypeSpec(
                inputs={"excitation": PortType(dtype="vector", shape=[6])},
                outputs={
                    "effector": PortType(dtype="state"),
                    "state": PortType(dtype="state"),
                },
            ),
            template_graph=arm_graph,
            template_ui_state=_template_ui_state(
                {
                    "world": (40, 140),
                    "anchor": (180, 140),
                    "shoulder": (320, 80),
                    "upper": (460, 80),
                    "elbow": (600, 80),
                    "forearm": (740, 80),
                    "excitation": (320, 300),
                    "effector": (900, 80),
                    "state": (900, 190),
                    "effector_marker": (900, 300),
                    **{
                        f"muscle_{index}": (460 + (index % 3) * 140, 250 + (index // 3) * 90)
                        for index in range(6)
                    },
                }
            ),
            template_id="feedbax.templates.mechanics.two_link_arm_6muscle",
            template_kind="executable",
        )
    )

    feedback_meta = metadata["feedbax.templates.simple_feedback"]
    feedback_graph = simple_feedback_template_graph()
    registry.register(
        declare_component(
            name=feedback_meta.name,
            category=feedback_meta.category,
            description=feedback_meta.description,
            param_schema=[],
            input_ports=list(feedback_graph.input_ports),
            output_ports=list(feedback_graph.output_ports),
            icon="Radar",
            is_composite=True,
            port_types=PortTypeSpec(
                inputs={"input": PortType(dtype="vector")},
                outputs={"effector": PortType(dtype="state")},
            ),
            template_graph=feedback_graph,
            template_ui_state=_template_ui_state(
                {
                    "feedback": (120, 260),
                    "feedback_ravel": (120, 110),
                    "input_mux": (340, 180),
                    "cell": (520, 180),
                    "readout": (700, 180),
                    "efferent": (880, 180),
                    "mechanics": (1060, 180),
                },
            ),
            template_id=feedback_meta.id,
            template_kind=feedback_meta.kind,
            trainable_by_default=True,
        )
    )
