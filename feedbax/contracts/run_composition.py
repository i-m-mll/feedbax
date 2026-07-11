"""Hierarchical authored-intent composition and execution dependency contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field, model_validator

from feedbax.contracts.manifest import StrictModel
from feedbax.contracts.run_matrix import (
    ContinuationReconciliation,
    DurableSlotTransform,
    ExecutionDependency,
    ForkFromSelectedCheckpoint,
    LineageGraftDependency,
    MatrixCompositionDelta,
    StoppedRowStatus,
    TaskIdentityGate,
    apply_composition_deltas,
)
from feedbax.contracts.spec_storage import training_spec_sha256, validate_sha256

COMPOSITION_SCHEMA_ID = "feedbax.spec.training_run_composition"
COMPOSITION_SCHEMA_VERSION = f"{COMPOSITION_SCHEMA_ID}.v1"
EXECUTION_DEPENDENCY_SCHEMA_ID = "feedbax.spec.training_execution_dependencies"
EXECUTION_DEPENDENCY_SCHEMA_VERSION = f"{EXECUTION_DEPENDENCY_SCHEMA_ID}.v1"


class InlineIntentParent(StrictModel):
    kind: Literal["inline"] = "inline"
    payload: dict[str, Any]
    schema_id: str
    schema_version: str


class AuthoredIntentParent(StrictModel):
    kind: Literal["authored_intent"] = "authored_intent"
    ref: str
    content_hash: str
    symbolic_name: str | None = None

    @model_validator(mode="after")
    def _hash(self) -> "AuthoredIntentParent":
        _digest(self.content_hash, "/parent/content_hash")
        return self


class ResolvedOutputParent(StrictModel):
    kind: Literal["resolved_output"] = "resolved_output"
    ref: str
    resolved_root_hash: str
    row_id: str | None = None
    checkpoint_transaction_id: str | None = None
    symbolic_name: str | None = None

    @model_validator(mode="after")
    def _hash(self) -> "ResolvedOutputParent":
        _digest(self.resolved_root_hash, "/parent/resolved_root_hash")
        return self


CompositionParent: TypeAlias = Annotated[
    InlineIntentParent | AuthoredIntentParent | ResolvedOutputParent,
    Field(discriminator="kind"),
]


CompositionDelta = MatrixCompositionDelta


class CompositionNode(StrictModel):
    """Single-parent composition; only the flattened terminal is target-validated."""

    schema_id: str = COMPOSITION_SCHEMA_ID
    schema_version: str = COMPOSITION_SCHEMA_VERSION
    name: str
    parent: CompositionParent
    deltas: list[CompositionDelta] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    selectors: list[dict[str, Any]] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def _identity(self) -> "CompositionNode":
        if self.schema_id != COMPOSITION_SCHEMA_ID:
            raise ValueError(f"/schema_id expected {COMPOSITION_SCHEMA_ID!r}")
        if self.schema_version != COMPOSITION_SCHEMA_VERSION:
            raise ValueError(
                f"/schema_version expected {COMPOSITION_SCHEMA_VERSION!r}; "
                "migration_intentionally_absent=yes"
            )
        if not self.name.strip():
            raise ValueError("/name must not be empty")
        return self


class NamedCompositionLane(StrictModel):
    name: str
    node: CompositionNode


class ExecutionDependencyLayer(StrictModel):
    schema_id: str = EXECUTION_DEPENDENCY_SCHEMA_ID
    schema_version: str = EXECUTION_DEPENDENCY_SCHEMA_VERSION
    dependencies: list[ExecutionDependency] = Field(default_factory=list)

    @model_validator(mode="after")
    def _identity(self) -> "ExecutionDependencyLayer":
        if self.schema_id != EXECUTION_DEPENDENCY_SCHEMA_ID:
            raise ValueError(f"/schema_id expected {EXECUTION_DEPENDENCY_SCHEMA_ID!r}")
        if self.schema_version != EXECUTION_DEPENDENCY_SCHEMA_VERSION:
            raise ValueError(
                f"/schema_version expected {EXECUTION_DEPENDENCY_SCHEMA_VERSION!r}; "
                "migration_intentionally_absent=yes"
            )
        return self


class FlattenedIntent(StrictModel):
    payload: dict[str, Any]
    attribution: dict[str, str]
    layer_hashes: list[str]


ParentResolver = Callable[
    [AuthoredIntentParent | ResolvedOutputParent], CompositionNode | dict[str, Any]
]


def composition_envelope(node: CompositionNode) -> dict[str, Any]:
    """Identity allowlist; readability refs and lane names are not identity-bearing."""
    parent = node.parent.model_dump(mode="json", exclude_none=True)
    parent.pop("ref", None)
    parent.pop("symbolic_name", None)
    return {
        "schema_id": node.schema_id,
        "schema_version": node.schema_version,
        "parent": parent,
        "deltas": [item.model_dump(mode="json", exclude_none=True) for item in node.deltas],
        "sources": node.sources,
        "selectors": node.selectors,
        "seeds": node.seeds,
    }


def authored_envelope_hash(node: CompositionNode) -> str:
    return training_spec_sha256(composition_envelope(node))


def flatten_composition(node: CompositionNode, resolve_parent: ParentResolver) -> FlattenedIntent:
    """Flatten root-to-child, checking pins and cross-layer override acknowledgements."""
    layers: list[CompositionNode] = []
    seen: set[str] = set()
    current = node
    while True:
        digest = authored_envelope_hash(current)
        if digest in seen:
            raise ValueError("/parent authored composition cycle detected")
        seen.add(digest)
        layers.append(current)
        parent = current.parent
        if isinstance(parent, InlineIntentParent):
            payload = dict(parent.payload)
            break
        resolved = resolve_parent(parent)
        if isinstance(parent, AuthoredIntentParent):
            if not isinstance(resolved, CompositionNode):
                raise ValueError("/parent authored_intent must resolve to CompositionNode")
            actual = authored_envelope_hash(resolved)
            if actual != parent.content_hash:
                raise ValueError(
                    "materializer drift: /parent/content_hash mismatch; "
                    f"archived={parent.content_hash!r}, computed={actual!r}"
                )
            current = resolved
            continue
        if not isinstance(resolved, dict):
            raise ValueError("/parent resolved_output must resolve to a mapping")
        actual = training_spec_sha256(resolved)
        if actual != parent.resolved_root_hash:
            raise ValueError("materializer drift: /parent/resolved_root_hash mismatch")
        payload = dict(resolved)
        break

    attribution: dict[str, str] = {}
    written: set[str] = set()
    hashes: list[str] = []
    for layer in reversed(layers):
        hashes.append(authored_envelope_hash(layer))
        payload, local_attribution, written = apply_composition_deltas(
            payload, layer.deltas, ancestor_written_paths=written
        )
        attribution.update(local_attribution)
    return FlattenedIntent(payload=payload, attribution=attribution, layer_hashes=hashes)


def composed_intent_hash(value: FlattenedIntent | Mapping[str, Any]) -> str:
    payload = value.payload if isinstance(value, FlattenedIntent) else value
    return training_spec_sha256(payload)


def _digest(value: str, path: str) -> None:
    validate_sha256(value, field_name=path)
