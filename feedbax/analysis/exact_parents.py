"""Versioned exact-parent input contract for staged analysis execution."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from feedbax.contracts.base import (
    ParentRef,
    StrictModel,
)
from feedbax.contracts.material_dependencies import MaterialDependencySet


STAGED_EXACT_PARENTS_SCHEMA_ID = "feedbax.spec.staged_exact_parents"
STAGED_EXACT_PARENTS_SCHEMA_VERSION_V1 = "feedbax.spec.staged_exact_parents.v1"
STAGED_EXACT_PARENTS_SCHEMA_VERSION = "feedbax.spec.staged_exact_parents.v2"
STAGED_EXACT_PARENTS_MIGRATION_TABLE = {
    STAGED_EXACT_PARENTS_SCHEMA_VERSION_V1: STAGED_EXACT_PARENTS_SCHEMA_VERSION,
}


class StagedExactParentEntry(StrictModel):
    """One durable parent ref bound to its resolved execution location."""

    parent: ParentRef
    execution_uri: str
    material_dependencies: MaterialDependencySet | None = None


class StagedExactParents(StrictModel):
    """Ordered, authoritative parent membership for one staged execution."""

    schema_id: Literal["feedbax.spec.staged_exact_parents"]
    schema_version: Literal["feedbax.spec.staged_exact_parents.v2"]
    parents: list[StagedExactParentEntry] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


def migrate_staged_exact_parents(
    payload: StagedExactParents | dict[str, Any],
) -> StagedExactParents:
    """Load v2 or deterministically migrate completed-only v1 declarations."""
    if isinstance(payload, StagedExactParents):
        return StagedExactParents.model_validate(payload.model_dump(mode="json"))
    data = dict(payload)
    if data.get("schema_id") != STAGED_EXACT_PARENTS_SCHEMA_ID:
        raise ValueError(
            "unsupported StagedExactParents schema_id: "
            f"{data.get('schema_id')!r}; expected {STAGED_EXACT_PARENTS_SCHEMA_ID!r}"
        )
    version = data.get("schema_version")
    if version == STAGED_EXACT_PARENTS_SCHEMA_VERSION_V1:
        migrated = dict(data)
        migrated["schema_version"] = STAGED_EXACT_PARENTS_SCHEMA_VERSION
        migrated["parents"] = [
            {**dict(entry), "material_dependencies": None}
            for entry in data.get("parents", ())
        ]
        data = migrated
    elif version != STAGED_EXACT_PARENTS_SCHEMA_VERSION:
        raise ValueError(
            "unsupported StagedExactParents schema_version: "
            f"{version!r}; expected {STAGED_EXACT_PARENTS_SCHEMA_VERSION!r}; "
            f"migration table={STAGED_EXACT_PARENTS_MIGRATION_TABLE!r}"
        )
    return StagedExactParents.model_validate(data)
