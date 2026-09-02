"""Internal parent-chain resolution for content-pinned delta documents."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from feedbax.contracts.matrix_core import ContentPinnedJsonBase
from feedbax.contracts.run_matrix import MatrixCompositionDelta, apply_composition_deltas


NodeT = TypeVar("NodeT")
LayerT = TypeVar("LayerT")


@dataclass(frozen=True)
class _FlattenedParentDeltas(Generic[LayerT]):
    payload: dict[str, Any]
    attribution: dict[str, str]
    layers: list[LayerT]


def _flatten_content_pinned_parent_deltas(
    node: NodeT,
    *,
    repo_root: Path | str | None,
    envelope_hash: Callable[[NodeT], str],
    parent_of: Callable[[NodeT], ContentPinnedJsonBase],
    load_parent: Callable[..., dict[str, Any]],
    deltas_of: Callable[[NodeT], Sequence[MatrixCompositionDelta]],
    parse_delta_parent: Callable[[dict[str, Any]], NodeT],
    terminal_payload: Callable[[dict[str, Any]], dict[str, Any]],
    layer_from_node: Callable[[str, NodeT], LayerT],
    delta_schema_id: str,
    terminal_schema_id: str,
    cycle_error: str,
    invalid_parent_error: Callable[[Any], str],
) -> _FlattenedParentDeltas[LayerT]:
    """Walk one pinned parent chain and apply its deltas from root to child."""
    chain: list[tuple[str, NodeT]] = []
    seen: set[str] = set()
    current = node
    while True:
        digest = envelope_hash(current)
        if digest in seen:
            raise ValueError(cycle_error)
        seen.add(digest)
        chain.append((digest, current))
        parent_payload = load_parent(parent_of(current), repo_root=repo_root)
        parent_schema_id = parent_payload.get("schema_id")
        if parent_schema_id == delta_schema_id:
            current = parse_delta_parent(parent_payload)
            continue
        if parent_schema_id != terminal_schema_id:
            raise ValueError(invalid_parent_error(parent_schema_id))
        payload = terminal_payload(parent_payload)
        break

    attribution: dict[str, str] = {}
    written: set[str] = set()
    layers: list[LayerT] = []
    for digest, current in reversed(chain):
        payload, local_attribution, written = apply_composition_deltas(
            payload,
            list(deltas_of(current)),
            ancestor_written_paths=written,
        )
        attribution.update(local_attribution)
        layers.append(layer_from_node(digest, current))
    return _FlattenedParentDeltas(
        payload=payload,
        attribution=attribution,
        layers=layers,
    )
