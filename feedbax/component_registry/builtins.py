"""Bounded discovery of built-in component owners."""

from __future__ import annotations

from feedbax.acausal.system import acausal_system_definition
from feedbax.components.penzai import penzai_adapter_definition
from feedbax.mechanics.analytical_component import analytical_musculoskeletal_plant_definition
from feedbax.mechanics.direct_force_components import (
    point_mass_definition,
    two_link_arm_definition,
)
from feedbax.runtime.graph import subgraph_definition
from feedbax.tasks.reach_components import (
    delayed_reaches_definition,
    simple_reaches_definition,
    stabilization_definition,
)

from .acausal_adapters import acausal_component_pack
from .discovery import ComponentPack, discover_modules
from .templates import builtin_graph_template_pack, builtin_muscle_template_pack


_OWNER_MODULES = (
    "feedbax.runtime.components",
    "feedbax.runtime.channel",
    "feedbax.runtime.filters",
    "feedbax.runtime.feedback",
    "feedbax.runtime.state_feedback",
    "feedbax.runtime.signal_transforms",
    "feedbax.runtime.affine_composer",
    "feedbax.control.continuous",
    "feedbax.control.discrete",
    "feedbax.control.pid",
    "feedbax.control.affine",
    "feedbax.models.networks",
    "feedbax.mechanics.force_fields",
    "feedbax.mechanics.linear_state_space",
    "feedbax.mechanics.muscles.relu_muscle",
    "feedbax.mechanics.muscles.thelen_muscle",
    "feedbax.mechanics.muscle_graph",
    "feedbax.components.equinox",
)


def builtin_component_packs() -> tuple[ComponentPack, ...]:
    """Build the fixed set of packages shipped with Feedbax."""
    return (
        discover_modules(*_OWNER_MODULES),
        ComponentPack(
            components=(
                subgraph_definition,
                acausal_system_definition,
                penzai_adapter_definition,
                point_mass_definition,
                two_link_arm_definition,
                analytical_musculoskeletal_plant_definition,
                simple_reaches_definition,
                delayed_reaches_definition,
                stabilization_definition,
            )
        ),
        acausal_component_pack(),
        builtin_graph_template_pack(),
        builtin_muscle_template_pack(),
    )
