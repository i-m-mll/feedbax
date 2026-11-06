"""Composition of the built-in graph template owners."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from feedbax.compiler.mechanics_templates import (
    mass_spring_damper_template_definition,
    point_mass_with_muscles_template_definition,
    two_link_arm_6muscle_template_definition,
)
from feedbax.compiler.templates import (
    recurrent_controller_template_definition,
    simple_feedback_template_definition,
)

if TYPE_CHECKING:
    from .discovery import ComponentPack


_GRAPH_TEMPLATE_FACTORIES = (
    recurrent_controller_template_definition,
    mass_spring_damper_template_definition,
    point_mass_with_muscles_template_definition,
    two_link_arm_6muscle_template_definition,
    simple_feedback_template_definition,
)


def builtin_graph_template_pack() -> ComponentPack:
    from .discovery import ComponentPack

    return ComponentPack(components=_GRAPH_TEMPLATE_FACTORIES)


def builtin_muscle_template_pack() -> ComponentPack:
    from feedbax.compiler.muscle_templates import (
        arm_6muscle_rigid_tendon_template_definition,
        point_mass_8muscle_relu_template_definition,
    )
    from feedbax.mechanics import muscle_representation

    from . import representation
    from .discovery import ComponentPack

    return ComponentPack(
        components=(
            arm_6muscle_rigid_tendon_template_definition,
            point_mass_8muscle_relu_template_definition,
        ),
        sources=(Path(muscle_representation.__file__), Path(representation.__file__)),
    )
