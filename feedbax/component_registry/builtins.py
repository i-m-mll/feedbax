"""Bounded discovery of built-in component owners."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .discovery import ComponentPack

_PORTABLE_MODULES = (
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
    "feedbax.mechanics.direct_force_components",
    "feedbax.mechanics.analytical_component",
    "feedbax.tasks.reach_components",
    "feedbax.components.penzai",
)


def builtin_component_packs() -> tuple[ComponentPack, ...]:
    """Build the fixed set of packages shipped with Feedbax."""

    from feedbax.acausal.system import AcausalSystem
    from feedbax.component_migrations import control, mechanics, networks, penzai, runtime, tasks
    from feedbax.component_migrations.composites import HISTORIES as COMPOSITE_HISTORIES
    from feedbax.component_migrations.equinox import HISTORIES as EQUINOX_HISTORIES
    from feedbax.runtime.graph import Subgraph
    from feedbax.studio import (
        acausal_representation,
        builtin_component_presentation,
        component_presentation,
        direct_force_presentation,
        muscle_representation,
        reach_presentation,
        representation,
    )

    from .acausal_adapters import acausal_component_pack
    from .templates import builtin_graph_template_pack, builtin_muscle_template_pack

    histories = {
        owner: history
        for group in (runtime, control, mechanics, networks, tasks, penzai)
        for owner, history in group.HISTORIES.items()
    }
    presentation = builtin_component_presentation.builtin_component_presentation()
    presentation_sources = tuple(
        Path(module.__file__).resolve()
        for module in (
            acausal_representation,
            builtin_component_presentation,
            component_presentation,
            direct_force_presentation,
            muscle_representation,
            reach_presentation,
            representation,
        )
    )
    packs = (
        ComponentPack(
            namespace="feedbax",
            modules=_PORTABLE_MODULES,
            histories=histories,
        ),
        ComponentPack(
            namespace="feedbax.equinox",
            modules=("feedbax.components.equinox",),
            histories=EQUINOX_HISTORIES,
            sources=(Path(__file__).resolve().parents[2] / "scripts/generate_eqx_components.py",),
        ),
        ComponentPack(
            namespace="feedbax",
            components=(Subgraph, AcausalSystem),
            histories=COMPOSITE_HISTORIES,
        ),
        acausal_component_pack(),
        builtin_graph_template_pack(),
        builtin_muscle_template_pack(),
    )
    return tuple(
        replace(pack, presentation=presentation, sources=(*pack.sources, *presentation_sources))
        for pack in packs
    )
