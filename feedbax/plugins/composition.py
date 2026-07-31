"""Application-process composition around the typed plugin bootstrap."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

from .application import new_registration_context
from .bootstrap import BootstrapState, bootstrap_application

T = TypeVar("T")


async def compose_application(
    *,
    modules: Sequence[str] = (),
    local_component_source: Path | None = Path.home() / ".feedbax" / "components",
) -> BootstrapState:
    """Discover and bootstrap one isolated application registry state."""
    return await bootstrap_application(
        new_registration_context(local_component_source=local_component_source),
        modules=modules,
    )


async def bootstrap_and_dispatch(
    dispatch: Callable[[BootstrapState], T],
    *,
    modules: Sequence[str] = (),
    local_component_source: Path | None = Path.home() / ".feedbax" / "components",
) -> T:
    """Bootstrap once, then invoke a synchronous process dispatcher."""
    state = await compose_application(
        modules=modules,
        local_component_source=local_component_source,
    )
    return dispatch(state)
