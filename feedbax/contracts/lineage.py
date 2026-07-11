"""Append-only, content-pinned training execution lineage DAG."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from feedbax.contracts.manifest import ArtifactRef, StrictModel
from feedbax.contracts.spec_storage import (
    store_canonical_json_artifact,
    training_spec_sha256,
    validate_sha256,
)

LINEAGE_EVENT_SCHEMA_ID = "feedbax.manifest.training_lineage_event"
LINEAGE_EVENT_SCHEMA_VERSION = f"{LINEAGE_EVENT_SCHEMA_ID}.v1"


class LineageParentRef(StrictModel):
    execution_hash: str
    event_hash: str | None = None
    symbolic_name: str | None = None

    @model_validator(mode="after")
    def _hashes(self) -> "LineageParentRef":
        validate_sha256(self.execution_hash, field_name="/parents/execution_hash")
        if self.event_hash is not None:
            validate_sha256(self.event_hash, field_name="/parents/event_hash")
        return self


class LineageEvent(StrictModel):
    """One immutable assertion; corrections append rather than mutate old evidence."""

    schema_id: str = LINEAGE_EVENT_SCHEMA_ID
    schema_version: str = LINEAGE_EVENT_SCHEMA_VERSION
    event_kind: Literal["execution", "graft_correction"]
    execution_hash: str
    parents: list[LineageParentRef] = Field(default_factory=list)
    original_event_hash: str | None = None
    correction_mode: Literal["supersedes_for_interpretation", "new_execution"] | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _validate_event(self) -> "LineageEvent":
        validate_sha256(self.execution_hash, field_name="/execution_hash")
        if self.schema_id != LINEAGE_EVENT_SCHEMA_ID:
            raise ValueError(f"/schema_id expected {LINEAGE_EVENT_SCHEMA_ID!r}")
        if self.schema_version != LINEAGE_EVENT_SCHEMA_VERSION:
            raise ValueError(
                f"/schema_version expected {LINEAGE_EVENT_SCHEMA_VERSION!r}; "
                "migration_intentionally_absent=yes"
            )
        correction_fields = self.original_event_hash is not None and self.correction_mode is not None
        if self.event_kind == "graft_correction" and not correction_fields:
            raise ValueError("/original_event_hash and /correction_mode required for graft")
        if self.event_kind == "execution" and correction_fields:
            raise ValueError("/original_event_hash is only valid for graft corrections")
        if self.original_event_hash is not None:
            validate_sha256(self.original_event_hash, field_name="/original_event_hash")
        return self

    @property
    def content_hash(self) -> str:
        return training_spec_sha256(self.model_dump(mode="json", exclude_none=True))


def store_lineage_event(
    event: LineageEvent,
    root: Path,
    logical_name: str,
) -> ArtifactRef:
    """Persist an immutable event through the shared content-addressed custody helper."""
    return store_canonical_json_artifact(
        event.model_dump(mode="json", exclude_none=True),
        root=root,
        role="training_lineage_event",
        logical_name=logical_name,
    )


class LineageDag:
    """In-memory replay view over append-only lineage events."""

    def __init__(self, events: list[LineageEvent] | None = None) -> None:
        self._events: dict[str, LineageEvent] = {}
        for event in events or []:
            self.append(event)

    @property
    def events(self) -> tuple[LineageEvent, ...]:
        return tuple(self._events.values())

    def append(self, event: LineageEvent) -> str:
        digest = event.content_hash
        if digest in self._events:
            return digest
        if event.original_event_hash is not None and event.original_event_hash not in self._events:
            raise ValueError("/original_event_hash must name an earlier append-only event")
        if any(parent.execution_hash == event.execution_hash for parent in event.parents):
            raise ValueError("/parents lineage self-cycle rejected")
        self._events[digest] = event
        if self._has_cycle():
            del self._events[digest]
            raise ValueError("/parents lineage cycle rejected")
        return digest

    def interpreted_parents(self, execution_hash: str) -> tuple[LineageParentRef, ...]:
        candidates = [event for event in self._events.values() if event.execution_hash == execution_hash]
        if not candidates:
            raise KeyError(execution_hash)
        superseding = [
            event
            for event in candidates
            if event.event_kind == "graft_correction"
            and event.correction_mode == "supersedes_for_interpretation"
        ]
        return tuple((superseding or candidates)[-1].parents)

    def _has_cycle(self) -> bool:
        edges = {
            event.execution_hash: {parent.execution_hash for parent in event.parents}
            for event in self._events.values()
        }
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            if any(visit(parent) for parent in edges.get(node, ())):
                return True
            visiting.remove(node)
            visited.add(node)
            return False

        return any(visit(node) for node in edges)
