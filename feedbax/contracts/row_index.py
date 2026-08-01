"""Authenticated row index and closed row-set selectors.

The row index is the substrate a downstream authoring layer selects rows from.
It deliberately straddles the compile/run custody boundary:

* :class:`AuthenticatedRowIndex` carries only compile-time facts — stable row
  ids, deterministic order, display labels, and bounded normalized tags. It is
  emitted at compile time from an already-locked row set, so a first-time
  figure compiles without any post-hoc production record.
* :class:`RowIndexCustodyBindings` carries the post-run half — the
  artifact/manifest custody bound to each row and role. It attaches through the
  run receipt after the rows have actually been produced.

Row-set selection is a closed tagged union with exactly two members,
``{"mode": "all"}`` and ``{"mode": "tag", "tag": ...}``. There is no
conjunction, negation, regular expression, ordering, comparison, or nesting,
and none may be added without a new schema version. A selector is expanded
exactly once, at compile time, into an explicit ordered row-id list pinned to
the digest of the index it was expanded against
(:class:`ResolvedRowSet`); consumers receive resolved rows and never a live
selector.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field, field_validator, model_validator

from feedbax.contracts.manifest import (
    ParentRef,
    StrictModel,
    authenticated_manifest_ref_profile,
    canonical_json_bytes,
    sha256_bytes,
)

ROW_INDEX_SCHEMA_ID = "feedbax.spec.authenticated_row_index"
ROW_INDEX_SCHEMA_VERSION = "feedbax.spec.authenticated_row_index.v1"
ROW_INDEX_CUSTODY_SCHEMA_ID = "feedbax.spec.row_index_custody_bindings"
ROW_INDEX_CUSTODY_SCHEMA_VERSION = "feedbax.spec.row_index_custody_bindings.v1"
RESOLVED_ROW_SET_SCHEMA_ID = "feedbax.spec.resolved_row_set"
RESOLVED_ROW_SET_SCHEMA_VERSION = "feedbax.spec.resolved_row_set.v1"

MAX_ROW_TAGS = 8
MAX_ROW_TAG_LENGTH = 40
MAX_ROW_INDEX_ROWS = 512

_ROW_ID_PATTERN = re.compile(r"^[a-z0-9]([a-z0-9._-]*[a-z0-9])?$")
_ROW_TAG_PATTERN = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
_BINDING_KEY_PATTERN = re.compile(r"^[a-z0-9]([a-z0-9_-]*[a-z0-9])?$")


class RowSelectionErrorCode(StrEnum):
    """Stable failure vocabulary for row-index selection and binding."""

    EMPTY_SELECTION = "empty-selection"
    DUPLICATE_ROW_ID = "duplicate-row-id"
    AMBIGUOUS_ROW_BINDING = "ambiguous-row-binding"
    UNRESOLVED_ROW_KEY = "unresolved-row-key"
    INDEX_MISMATCH = "row-index-mismatch"


class RowSelectionError(ValueError):
    """One stable error code plus the selection context that produced it."""

    def __init__(
        self,
        code: RowSelectionErrorCode,
        message: str,
        *,
        index_id: str | None = None,
        row_id: str | None = None,
        binding_key: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.index_id = index_id
        self.row_id = row_id
        self.binding_key = binding_key


def normalize_row_tags(tags: Sequence[str]) -> tuple[str, ...]:
    """Return one bounded, normalized, sorted tag set or fail closed.

    Tag order is not semantic (a tag set is a set), so tags canonicalize to
    sorted order. Duplicates are rejected rather than silently collapsed
    because a repeated tag carries no information.
    """
    if len(tags) > MAX_ROW_TAGS:
        raise ValueError(f"row declares {len(tags)} tags, more than the {MAX_ROW_TAGS} allowed")
    seen: set[str] = set()
    for tag in tags:
        if not isinstance(tag, str):
            raise ValueError("row tags must be strings")
        if len(tag) > MAX_ROW_TAG_LENGTH:
            raise ValueError(f"row tag {tag!r} exceeds {MAX_ROW_TAG_LENGTH} characters")
        if not _ROW_TAG_PATTERN.match(tag):
            raise ValueError(
                f"row tag {tag!r} is not a normalized lowercase hyphenated token"
            )
        if tag in seen:
            raise ValueError(f"row declares duplicate tag {tag!r}")
        seen.add(tag)
    return tuple(sorted(seen))


class RowIndexEntry(StrictModel):
    """One stable row identity, its display label, and its bounded tag set."""

    row_id: str
    label: str
    tags: list[str] = Field(default_factory=list)

    @field_validator("row_id")
    @classmethod
    def _validate_row_id(cls, value: str) -> str:
        if not _ROW_ID_PATTERN.match(value):
            raise ValueError(f"row id {value!r} is not a normalized lowercase token")
        return value

    @field_validator("label")
    @classmethod
    def _validate_label(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("row label must be nonempty")
        return value

    @field_validator("tags")
    @classmethod
    def _validate_tags(cls, value: list[str]) -> list[str]:
        return list(normalize_row_tags(value))


def derive_row_label(row_id: str) -> str:
    """Return the deterministic display label for *row_id*.

    The rule is fixed rather than authored: a row label is its id with word
    separators rendered as spaces, so a label can never drift away from the
    identity it names.
    """
    return row_id.replace("-", " ")


class AuthenticatedRowIndex(StrictModel):
    """Compile-time row identity, order, and tags for one selectable row set.

    This half of the index is emitted at compile time from the locked row set.
    It carries no custody: see :class:`RowIndexCustodyBindings` for the
    post-run artifact/manifest half.
    """

    schema_id: str = ROW_INDEX_SCHEMA_ID
    schema_version: str = ROW_INDEX_SCHEMA_VERSION
    index_id: str
    rows: list[RowIndexEntry] = Field(min_length=1, max_length=MAX_ROW_INDEX_ROWS)

    @model_validator(mode="after")
    def _validate_index(self) -> "AuthenticatedRowIndex":
        if self.schema_id != ROW_INDEX_SCHEMA_ID:
            raise ValueError(f"unsupported AuthenticatedRowIndex schema_id: {self.schema_id!r}")
        if self.schema_version != ROW_INDEX_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported AuthenticatedRowIndex schema_version: {self.schema_version!r}"
            )
        if not self.index_id.strip():
            raise ValueError("AuthenticatedRowIndex index_id must be nonempty")
        row_ids = [entry.row_id for entry in self.rows]
        duplicates = sorted({value for value in row_ids if row_ids.count(value) > 1})
        if duplicates:
            raise RowSelectionError(
                RowSelectionErrorCode.DUPLICATE_ROW_ID,
                f"AuthenticatedRowIndex declares duplicate row ids: {duplicates}",
                index_id=self.index_id,
                row_id=duplicates[0],
            )
        return self

    @property
    def row_ids(self) -> tuple[str, ...]:
        """Return the deterministic row order declared by this index."""
        return tuple(entry.row_id for entry in self.rows)

    def entry(self, row_id: str) -> RowIndexEntry:
        """Return one declared row entry or fail closed."""
        for entry in self.rows:
            if entry.row_id == row_id:
                return entry
        raise RowSelectionError(
            RowSelectionErrorCode.UNRESOLVED_ROW_KEY,
            f"row id {row_id!r} is not declared by row index {self.index_id!r}",
            index_id=self.index_id,
            row_id=row_id,
        )

    def canonical_sha256(self) -> str:
        """Return the pinned digest of this index's canonical bytes."""
        return sha256_bytes(canonical_json_bytes(self))


class AllRowsSelector(StrictModel):
    """Select every row the index declares, in index order.

    ``index`` is a locator, not a predicate: it names the row index the
    selector is expanded against and is resolved by the caller, which owns how
    its own indices are addressed. It adds no selection power and is pinned
    into the expansion result for audit.
    """

    mode: Literal["all"] = "all"
    index: str | None = None


class TagRowsSelector(StrictModel):
    """Select every row carrying exactly this tag, in index order.

    Matching is set membership on one normalized tag. There is deliberately no
    conjunction, negation, regular expression, comparison, ordering, or
    nesting; adding any of them requires a new schema version.
    """

    mode: Literal["tag"]
    tag: str
    index: str | None = None

    @field_validator("tag")
    @classmethod
    def _validate_tag(cls, value: str) -> str:
        normalize_row_tags([value])
        return value


RowSetSelector: TypeAlias = Annotated[
    AllRowsSelector | TagRowsSelector, Field(discriminator="mode")
]


class ResolvedRowSet(StrictModel):
    """One selector expanded once into an explicit ordered row-id list."""

    schema_id: str = RESOLVED_ROW_SET_SCHEMA_ID
    schema_version: str = RESOLVED_ROW_SET_SCHEMA_VERSION
    index_id: str
    index_ref: str | None = None
    index_sha256: str
    selector: RowSetSelector
    row_ids: list[str] = Field(min_length=1)
    labels: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_resolution(self) -> "ResolvedRowSet":
        if self.schema_id != RESOLVED_ROW_SET_SCHEMA_ID:
            raise ValueError(f"unsupported ResolvedRowSet schema_id: {self.schema_id!r}")
        if self.schema_version != RESOLVED_ROW_SET_SCHEMA_VERSION:
            raise ValueError(f"unsupported ResolvedRowSet schema_version: {self.schema_version!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", self.index_sha256):
            raise ValueError("ResolvedRowSet index_sha256 must be lowercase sha256")
        if len(self.row_ids) != len(set(self.row_ids)):
            raise RowSelectionError(
                RowSelectionErrorCode.DUPLICATE_ROW_ID,
                "ResolvedRowSet expanded to duplicate row ids",
                index_id=self.index_id,
            )
        if len(self.labels) != len(self.row_ids):
            raise ValueError("ResolvedRowSet labels must be one per resolved row")
        return self


def expand_row_selector(
    selector: RowSetSelector | dict[str, Any],
    index: AuthenticatedRowIndex,
) -> ResolvedRowSet:
    """Expand one closed selector against *index* into explicit ordered rows.

    Expansion happens exactly once, at compile time. The result pins the digest
    of the index it was expanded against, so a later index revision cannot
    silently change what the consumer already resolved.

    Raises:
        RowSelectionError: If the selection is empty. Duplicate row ids are
            rejected when the index is constructed.
    """
    if isinstance(selector, dict):
        selector = _validate_selector(selector)
    if isinstance(selector, AllRowsSelector):
        matched = list(index.rows)
    else:
        matched = [entry for entry in index.rows if selector.tag in entry.tags]
    if not matched:
        detail = (
            "index declares no rows"
            if isinstance(selector, AllRowsSelector)
            else f"no row carries tag {selector.tag!r}"
        )
        raise RowSelectionError(
            RowSelectionErrorCode.EMPTY_SELECTION,
            f"row-set selector resolved to no rows in index {index.index_id!r}: {detail}",
            index_id=index.index_id,
        )
    return ResolvedRowSet(
        index_id=index.index_id,
        index_ref=selector.index,
        index_sha256=index.canonical_sha256(),
        selector=selector,
        row_ids=[entry.row_id for entry in matched],
        labels=[entry.label for entry in matched],
    )


def _validate_selector(payload: dict[str, Any]) -> AllRowsSelector | TagRowsSelector:
    mode = payload.get("mode")
    if mode == "all":
        return AllRowsSelector.model_validate(payload)
    if mode == "tag":
        return TagRowsSelector.model_validate(payload)
    raise ValueError(
        f"row-set selector mode {mode!r} is not one of the closed modes 'all' and 'tag'"
    )


class RowCustodyBinding(StrictModel):
    """One post-run authenticated artifact bound to one row and binding key."""

    row_id: str
    binding_key: str
    parent: ParentRef

    @field_validator("binding_key")
    @classmethod
    def _validate_binding_key(cls, value: str) -> str:
        if not _BINDING_KEY_PATTERN.match(value):
            raise ValueError(f"row binding key {value!r} is not a normalized lowercase token")
        return value

    @model_validator(mode="after")
    def _validate_authentication(self) -> "RowCustodyBinding":
        if authenticated_manifest_ref_profile(self.parent) is None:
            raise ValueError(
                f"row custody binding {self.row_id}/{self.binding_key} must reference an "
                "authenticated manifest"
            )
        return self

    def authenticated_profile(self) -> tuple[str, int]:
        """Return the bound manifest's ``(sha256, size_bytes)`` custody profile."""
        profile = authenticated_manifest_ref_profile(self.parent)
        assert profile is not None  # guaranteed by validation
        return profile


class RowIndexCustodyBindings(StrictModel):
    """Post-run custody attaching produced artifacts to compile-time rows.

    This document is produced by the run receipt layer, never hand-authored,
    and never required for a first-time compile.
    """

    schema_id: str = ROW_INDEX_CUSTODY_SCHEMA_ID
    schema_version: str = ROW_INDEX_CUSTODY_SCHEMA_VERSION
    index_id: str
    bindings: list[RowCustodyBinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_bindings(self) -> "RowIndexCustodyBindings":
        if self.schema_id != ROW_INDEX_CUSTODY_SCHEMA_ID:
            raise ValueError(f"unsupported RowIndexCustodyBindings schema_id: {self.schema_id!r}")
        if self.schema_version != ROW_INDEX_CUSTODY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported RowIndexCustodyBindings schema_version: {self.schema_version!r}"
            )
        keys = [(binding.row_id, binding.binding_key) for binding in self.bindings]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            row_id, binding_key = duplicates[0]
            raise RowSelectionError(
                RowSelectionErrorCode.AMBIGUOUS_ROW_BINDING,
                "row custody bindings declare more than one manifest for "
                f"{row_id!r}/{binding_key!r}",
                index_id=self.index_id,
                row_id=row_id,
                binding_key=binding_key,
            )
        return self

    def resolve(self, row_id: str, binding_key: str) -> RowCustodyBinding:
        """Return the single custody binding for one row and key or fail closed."""
        matches = [
            binding
            for binding in self.bindings
            if binding.row_id == row_id and binding.binding_key == binding_key
        ]
        if not matches:
            raise RowSelectionError(
                RowSelectionErrorCode.UNRESOLVED_ROW_KEY,
                f"row custody bindings carry no {binding_key!r} artifact for row {row_id!r}",
                index_id=self.index_id,
                row_id=row_id,
                binding_key=binding_key,
            )
        return matches[0]

    def require_index(self, index: AuthenticatedRowIndex) -> None:
        """Fail closed when these bindings do not belong to *index*."""
        if self.index_id != index.index_id:
            raise RowSelectionError(
                RowSelectionErrorCode.INDEX_MISMATCH,
                f"row custody bindings for {self.index_id!r} do not belong to row index "
                f"{index.index_id!r}",
                index_id=index.index_id,
            )
        declared = set(index.row_ids)
        unknown = sorted({binding.row_id for binding in self.bindings} - declared)
        if unknown:
            raise RowSelectionError(
                RowSelectionErrorCode.UNRESOLVED_ROW_KEY,
                f"row custody bindings name rows absent from index {index.index_id!r}: {unknown}",
                index_id=index.index_id,
                row_id=unknown[0],
            )


#: Declarative custody request supplied by a downstream authoring layer: one
#: mapping from row id to binding key to the authenticated manifest ref that
#: produced that row's artifact. Feedbax owns everything else — index matching,
#: authentication checks, ordering, serialization, and the atomic write.
RowCustodyBindingMap: TypeAlias = "Mapping[str, Mapping[str, ParentRef | Mapping[str, Any]]]"


def build_row_index_custody_bindings(
    index: AuthenticatedRowIndex,
    bindings: RowCustodyBindingMap,
    *,
    required_binding_keys: Sequence[str] = (),
) -> RowIndexCustodyBindings:
    """Build post-run custody bindings for *index* from a declarative row map.

    The caller supplies only ``{row_id: {binding_key: parent}}``. This function
    owns index matching (every declared row must be declared by *index*),
    authenticated ``ParentRef`` validation (delegated to
    :class:`RowCustodyBinding`), and deterministic ordering: rows follow index
    order and binding keys sort within a row, so the same declaration always
    produces the same document bytes.

    Args:
        index: Compile-time row index the custody attaches to.
        bindings: Declarative row/binding-role mapping.
        required_binding_keys: Binding keys every index row must carry. Empty
            means partial custody is admissible; a named key that is missing for
            any index row fails closed rather than writing a partial document.

    Raises:
        RowSelectionError: If a declared row is not in *index*, or a required
            binding key is missing for an index row.
        ValueError: If a parent is not an authenticated manifest ref or a
            binding key is not a normalized lowercase token.
    """
    declared = set(index.row_ids)
    unknown = sorted(set(bindings) - declared)
    if unknown:
        raise RowSelectionError(
            RowSelectionErrorCode.UNRESOLVED_ROW_KEY,
            f"row custody declaration names rows absent from index {index.index_id!r}: {unknown}",
            index_id=index.index_id,
            row_id=unknown[0],
        )
    required = list(dict.fromkeys(required_binding_keys))
    ordered: list[RowCustodyBinding] = []
    for row_id in index.row_ids:
        row_bindings = bindings.get(row_id, {})
        missing = [key for key in required if key not in row_bindings]
        if missing:
            raise RowSelectionError(
                RowSelectionErrorCode.UNRESOLVED_ROW_KEY,
                f"row {row_id!r} is missing required custody bindings {missing} "
                f"for index {index.index_id!r}",
                index_id=index.index_id,
                row_id=row_id,
                binding_key=missing[0],
            )
        for binding_key in sorted(row_bindings):
            parent = row_bindings[binding_key]
            ordered.append(
                RowCustodyBinding(
                    row_id=row_id,
                    binding_key=binding_key,
                    parent=(
                        parent if isinstance(parent, ParentRef) else ParentRef.model_validate(parent)
                    ),
                )
            )
    document = RowIndexCustodyBindings(index_id=index.index_id, bindings=ordered)
    document.require_index(index)
    return document


def serialize_row_index_custody_bindings(document: RowIndexCustodyBindings) -> str:
    """Return the canonical sidecar text for one custody-bindings document.

    Keys are sorted at every level and the text ends with a newline, so a
    regenerated sidecar is byte-comparable against its tracked predecessor.
    """
    payload = document.model_dump(mode="json", exclude_none=True)
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_row_index_custody_bindings(
    document: RowIndexCustodyBindings,
    path: Path | str,
    *,
    index: AuthenticatedRowIndex | None = None,
) -> Path:
    """Atomically write one custody-bindings sidecar and return its path.

    The document is fully serialized to a sibling temporary file and renamed
    into place, so a reader never observes a partially written sidecar. When
    *index* is supplied the document is matched against it first.
    """
    if index is not None:
        document.require_index(index)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = serialize_row_index_custody_bindings(document)
    handle, temporary_name = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def load_row_index_custody_bindings(path: Path | str) -> RowIndexCustodyBindings:
    """Load one custody-bindings sidecar, failing closed on foreign identity."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("row custody bindings document must be a JSON object")
    return RowIndexCustodyBindings.model_validate(payload)


__all__ = [
    "MAX_ROW_INDEX_ROWS",
    "MAX_ROW_TAGS",
    "MAX_ROW_TAG_LENGTH",
    "RESOLVED_ROW_SET_SCHEMA_ID",
    "RESOLVED_ROW_SET_SCHEMA_VERSION",
    "ROW_INDEX_CUSTODY_SCHEMA_ID",
    "ROW_INDEX_CUSTODY_SCHEMA_VERSION",
    "ROW_INDEX_SCHEMA_ID",
    "ROW_INDEX_SCHEMA_VERSION",
    "AllRowsSelector",
    "AuthenticatedRowIndex",
    "ResolvedRowSet",
    "RowCustodyBinding",
    "RowCustodyBindingMap",
    "RowIndexCustodyBindings",
    "RowIndexEntry",
    "RowSelectionError",
    "RowSelectionErrorCode",
    "RowSetSelector",
    "TagRowsSelector",
    "build_row_index_custody_bindings",
    "derive_row_label",
    "expand_row_selector",
    "load_row_index_custody_bindings",
    "normalize_row_tags",
    "serialize_row_index_custody_bindings",
    "write_row_index_custody_bindings",
]
