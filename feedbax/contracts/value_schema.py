"""Shared value-schema records for provider-owned contract surfaces."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SchemaOrigin = Literal[
    "declared",
    "inferred_static",
    "runtime_sample",
    "curated_fallback",
    "unknown",
]


class ValueSchema(BaseModel):
    """Provider-owned value type metadata for selectable Studio surfaces."""

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    kind: str
    dtype: str | None = None
    shape: list[Any] | None = None
    rank: int | None = None
    units: str | None = None
    frame: str | None = None
    origin: SchemaOrigin = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)
