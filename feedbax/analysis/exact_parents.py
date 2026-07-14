"""Versioned exact-parent input contract for staged analysis execution."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from feedbax.contracts.manifest import ParentRef, StrictModel


STAGED_EXACT_PARENTS_SCHEMA_ID = "feedbax.spec.staged_exact_parents"
STAGED_EXACT_PARENTS_SCHEMA_VERSION = "feedbax.spec.staged_exact_parents.v1"


class StagedExactParentEntry(StrictModel):
    """One durable parent ref bound to its resolved execution location."""

    parent: ParentRef
    execution_uri: str


class StagedExactParents(StrictModel):
    """Ordered, authoritative parent membership for one staged execution."""

    schema_id: Literal["feedbax.spec.staged_exact_parents"]
    schema_version: Literal["feedbax.spec.staged_exact_parents.v1"]
    parents: list[StagedExactParentEntry] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
