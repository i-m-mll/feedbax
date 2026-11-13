"""Composition root for the retained envelope and worker entry points."""

from collections.abc import Iterable, Sequence
from typing import Any

from feedbax.component_registry.retained import retained_component_registry
from feedbax.training.retained_studio import studio_graph_training_program

from .application import ApplicationRegistryBundle, _new_registry_bundle
from .bootstrap import BootstrapState
from .composition import _compose


def new_retained_registry_bundle() -> ApplicationRegistryBundle:
    components = retained_component_registry()
    return _new_registry_bundle(components, studio_graph_training_program(components))


async def compose_retained_application(
    *,
    modules: Sequence[str] = (),
    entry_points: Iterable[Any] | None = None,
) -> BootstrapState:
    return await _compose(
        new_retained_registry_bundle, modules=modules, entry_points=entry_points
    )
