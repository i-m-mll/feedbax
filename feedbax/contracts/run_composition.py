"""Hierarchical authored-intent composition and execution dependency contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field, model_validator

from feedbax.contracts.authored_canonical import CANONICAL_PIN_ALGORITHM
from feedbax.contracts.base import StrictModel
from feedbax.contracts.run_matrix import (
    ContinuationReconciliation as ContinuationReconciliation,
    DurableSlotTransform as DurableSlotTransform,
    ExecutionDependency,
    ForkFromSelectedCheckpoint as ForkFromSelectedCheckpoint,
    LineageGraftDependency as LineageGraftDependency,
    MatrixCompositionDelta,
    StoppedRowStatus as StoppedRowStatus,
    TaskIdentityGate as TaskIdentityGate,
    apply_composition_deltas,
)
from feedbax.contracts.spec_storage import training_spec_sha256, validate_sha256
from feedbax.contracts.strict_json import strict_json_loads

COMPOSITION_SCHEMA_ID = "feedbax.spec.training_run_composition"
COMPOSITION_SCHEMA_VERSION_V1 = f"{COMPOSITION_SCHEMA_ID}.v1"
COMPOSITION_SCHEMA_VERSION = f"{COMPOSITION_SCHEMA_ID}.v2"
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


class CanonicalJsonDocumentPin(StrictModel):
    """Repository document pinned in the canonical-json-v1 hash domain."""

    ref: str
    sha256: str

    @model_validator(mode="after")
    def _pin(self) -> "CanonicalJsonDocumentPin":
        if not self.ref.strip():
            raise ValueError("document pin ref must not be empty")
        _digest(self.sha256, "/sha256")
        return self


class CompiledTrainingRowParent(StrictModel):
    """One exact governed row from a pinned matrix-v6 compile."""

    kind: Literal["compiled_training_row"] = "compiled_training_row"
    matrix: CanonicalJsonDocumentPin
    compile_lock: CanonicalJsonDocumentPin
    row_id: str
    symbolic_name: str | None = None

    @model_validator(mode="after")
    def _row(self) -> "CompiledTrainingRowParent":
        if not self.row_id.strip():
            raise ValueError("/parent/row_id must not be empty")
        return self


CompositionParentV2: TypeAlias = Annotated[
    InlineIntentParent
    | AuthoredIntentParent
    | ResolvedOutputParent
    | CompiledTrainingRowParent,
    Field(discriminator="kind"),
]


CompositionDelta = MatrixCompositionDelta


class CompositionNode(StrictModel):
    """Single-parent composition; only the flattened terminal is target-validated."""

    schema_id: str = COMPOSITION_SCHEMA_ID
    schema_version: str = COMPOSITION_SCHEMA_VERSION_V1
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
        if self.schema_version != COMPOSITION_SCHEMA_VERSION_V1:
            raise ValueError(
                f"/schema_version expected {COMPOSITION_SCHEMA_VERSION_V1!r}; "
                "migration_intentionally_absent=yes"
            )
        if not self.name.strip():
            raise ValueError("/name must not be empty")
        return self


class CompositionNodeV2(StrictModel):
    """Composition v2, adding only the compiled-matrix-row parent."""

    schema_id: str = COMPOSITION_SCHEMA_ID
    schema_version: str = COMPOSITION_SCHEMA_VERSION
    name: str
    parent: CompositionParentV2
    deltas: list[CompositionDelta] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    selectors: list[dict[str, Any]] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def _identity(self) -> "CompositionNodeV2":
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


CompositionDocument: TypeAlias = CompositionNode | CompositionNodeV2


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
    [AuthoredIntentParent | ResolvedOutputParent | CompiledTrainingRowParent],
    CompositionDocument | dict[str, Any],
]


def composition_identity_projection(node: CompositionDocument) -> dict[str, Any]:
    """Identity allowlist; readability refs and lane names are not identity-bearing."""
    parent = node.parent.model_dump(mode="json", exclude_none=True)
    if isinstance(node.parent, CompiledTrainingRowParent):
        parent = {
            "kind": node.parent.kind,
            "matrix": {
                "sha256": node.parent.matrix.sha256,
                "pin_algorithm": CANONICAL_PIN_ALGORITHM,
            },
            "compile_lock": {
                "sha256": node.parent.compile_lock.sha256,
                "pin_algorithm": CANONICAL_PIN_ALGORITHM,
            },
            "row_id": node.parent.row_id,
        }
    else:
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


def authored_envelope_hash(node: CompositionDocument) -> str:
    return training_spec_sha256(composition_identity_projection(node))


def parse_composition_node(document: Mapping[str, Any]) -> CompositionDocument:
    """Parse one explicit composition version without inferring or migrating it."""
    version = document.get("schema_version")
    if version == COMPOSITION_SCHEMA_VERSION_V1:
        return CompositionNode.model_validate(document)
    if version == COMPOSITION_SCHEMA_VERSION:
        return CompositionNodeV2.model_validate(document)
    raise ValueError(
        f"unsupported {COMPOSITION_SCHEMA_ID} schema_version {version!r}; "
        f"supported={[COMPOSITION_SCHEMA_VERSION_V1, COMPOSITION_SCHEMA_VERSION]!r}; "
        "migration_intentionally_absent=yes"
    )


def flatten_composition(node: CompositionDocument, resolve_parent: ParentResolver) -> FlattenedIntent:
    """Flatten root-to-child, checking pins and cross-layer override acknowledgements."""
    layers: list[CompositionDocument] = []
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
            if not isinstance(resolved, (CompositionNode, CompositionNodeV2)):
                raise ValueError("/parent authored_intent must resolve to a composition node")
            actual = authored_envelope_hash(resolved)
            if actual != parent.content_hash:
                raise ValueError(
                    "materializer drift: /parent/content_hash mismatch; "
                    f"archived={parent.content_hash!r}, computed={actual!r}"
                )
            current = resolved
            continue
        if isinstance(parent, CompiledTrainingRowParent):
            if not isinstance(resolved, dict):
                raise ValueError("/parent compiled_training_row must resolve to a mapping")
            payload = dict(resolved)
            break
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


def flatten_repo_composition(
    node: CompositionNode,
    *,
    repo_root: Path,
    source_ref: str | None = None,
) -> FlattenedIntent:
    """Flatten a composition whose parent refs name JSON beneath ``repo_root``.

    ``source_ref`` identifies the already-loaded child document, when known, so
    a child that points back to itself fails as a reference cycle before any
    misleading hash-drift diagnosis.
    """
    root = repo_root.resolve()
    resolved_refs: set[Path] = set()
    if source_ref is not None:
        resolved_refs.add(_repo_json_path(root, source_ref))

    def resolve_parent(
        parent: AuthoredIntentParent | ResolvedOutputParent,
    ) -> CompositionNode | dict[str, Any]:
        path = _repo_json_path(root, parent.ref)
        if path in resolved_refs:
            raise ValueError(f"/parent/ref authored composition cycle: {parent.ref}")
        resolved_refs.add(path)
        try:
            document = strict_json_loads(path.read_text(encoding="utf-8"), ref=parent.ref)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"/parent/ref cannot load JSON document: {parent.ref}") from exc
        if not isinstance(document, dict):
            raise ValueError(f"/parent/ref must resolve to a JSON object: {parent.ref}")
        if isinstance(parent, AuthoredIntentParent):
            return CompositionNode.model_validate(document)
        return document

    return flatten_composition(node, resolve_parent)


def composed_intent_hash(value: FlattenedIntent | Mapping[str, Any]) -> str:
    payload = value.payload if isinstance(value, FlattenedIntent) else value
    return training_spec_sha256(payload)


def _digest(value: str, path: str) -> None:
    validate_sha256(value, field_name=path)


def _repo_json_path(repo_root: Path, ref: str) -> Path:
    path = Path(ref)
    if path.is_absolute():
        raise ValueError(f"repository document ref must be relative: {ref}")
    resolved = (repo_root / path).resolve()
    if not resolved.is_relative_to(repo_root):
        raise ValueError(f"repository document ref escapes repo root: {ref}")
    return resolved
