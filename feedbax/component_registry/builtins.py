"""Bounded discovery of built-in component owners."""

from __future__ import annotations

from importlib import import_module
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
    from feedbax.acausal.system_history import HISTORIES as SYSTEM_HISTORIES
    from feedbax.components.equinox_history import HISTORIES as EQUINOX_HISTORIES
    from feedbax.runtime.graph import Subgraph
    from feedbax.runtime.graph_history import HISTORIES as GRAPH_HISTORIES

    from .acausal_adapters import acausal_component_pack
    from .templates import builtin_graph_template_pack, builtin_muscle_template_pack

    histories = {
        owner: history
        for module in _PORTABLE_MODULES
        for owner, history in import_module(f"{module}_history").HISTORIES.items()
    }
    return (
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
            histories={**GRAPH_HISTORIES, **SYSTEM_HISTORIES},
        ),
        acausal_component_pack(),
        builtin_graph_template_pack(),
        builtin_muscle_template_pack(),
    )
