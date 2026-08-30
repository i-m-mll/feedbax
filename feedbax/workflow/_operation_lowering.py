"""Shared construction primitive for layer-owned operation lowerers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .plan import Operation


def operation(
    *,
    type_id: str,
    compiled_schema_id: str,
    semantic_hash: str,
    input_types: Mapping[str, str],
    determinism: str,
    cache_policy: str,
    effect: str,
    capabilities: Sequence[str] = (),
    parameters: Mapping[str, Any] | None = None,
) -> Operation:
    return Operation(
        type_id=type_id,
        parameters={
            "compiled_schema_id": compiled_schema_id,
            "semantic_hash": semantic_hash,
            **dict(parameters or {}),
        },
        input_types=dict(input_types),
        output_types={"primary": compiled_schema_id},
        determinism=determinism,
        cache_policy=cache_policy,
        effect=effect,
        capabilities=tuple(capabilities),
    )
