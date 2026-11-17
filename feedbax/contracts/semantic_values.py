"""Versioned keys for model and task values."""

from typing import Literal

from feedbax.contracts.project_sources import pointer


def semantic_key(
    root: Literal["model", "task"],
    facet: Literal["value", "state", "input", "output", "edge", "carry"],
    owner: tuple[str, ...],
    symbol: str,
    *,
    wire: str | None = None,
) -> str:
    """Project a declared address using escaped tokens and facet-specific suffixes."""
    if facet == "carry":
        if wire is None:
            raise ValueError("A carry address requires a declared wire ID")
        return pointer(("v1", root, facet, *owner, wire, symbol))
    if wire is not None:
        raise ValueError("Only a carry address takes both a wire ID and a symbol")
    return pointer(("v1", root, facet, *owner, symbol))
