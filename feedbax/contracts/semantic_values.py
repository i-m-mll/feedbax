"""Versioned keys for model and task values."""

from typing import Literal

from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.studio_place import path_parts


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


def model_state_address(key: str) -> ModelAddress:
    """Resolve a versioned model state key to its authored field address."""
    version, root, facet, *scope, component, symbol = path_parts(key)
    if (version, root, facet) != ("v1", "model", "state"):
        raise ValueError(f"Expected a model state key: {key}")
    return ModelAddress.at_state(tuple(scope), component, symbol)
