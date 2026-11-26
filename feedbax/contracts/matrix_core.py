"""Domain-independent contracts and expansion for base/row/delta matrices."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from copy import copy, deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from string import Formatter
from typing import Any, Generic, Literal, TypeVar

from pydantic import Field, TypeAdapter, model_validator

from feedbax.contracts._authoring_inputs import AuthoringTextReader, read_authoring_text
from feedbax.contracts.expression_storage import StoredTime
from feedbax.contracts.expression_text import parse_stored_expression, print_stored_expression
from feedbax.contracts.expression_values import Expression, UnresolvedValue
from feedbax.contracts.expressions import ContextItem, ExpressionContext, ValueExpr, evaluate_query
from feedbax.contracts.extraction import SourceBinding, load_expression_context, set_dotted_path
from feedbax.contracts.base import (
    StrictModel,
    canonical_json_bytes,
    sha256_bytes,
)
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.strict_json import strict_json_loads


PayloadT = TypeVar("PayloadT", bound=StrictModel)
_PATH_SAFE_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ARRAY_INDEX_RE = re.compile(r"^(0|[1-9][0-9]*)$")

MATRIX_AXIS_VALUE_GENERATOR_SCHEMA_ID = "feedbax.spec.matrix_axis_value_generator"
MATRIX_AXIS_VALUE_GENERATOR_SCHEMA_VERSION = "feedbax.spec.matrix_axis_value_generator.v1"
_GENERATOR_ID_FORMAT_FIELDS = ("index", "value")
_FORMATTER = Formatter()

SOURCE_DOCUMENT_INHERITANCE_SCHEMA_ID = "feedbax.spec.source_document_inheritance"
SOURCE_DOCUMENT_INHERITANCE_SCHEMA_VERSION = "feedbax.spec.source_document_inheritance.v1"
# Reserved top-level key under which a source document declares content-pinned
# sub-document inheritance. A document that does not carry this key is passed
# through byte-for-byte, so existing source documents are unaffected.
SOURCE_DOCUMENT_INHERITANCE_KEY = "__inherit__"


class ContentPinnedJsonBase(StrictModel):
    """Relative JSON document whose canonical content hash is authoritative.

    The ``sha256`` pin always covers the whole referenced file. An optional
    ``payload_path`` is a JSON-pointer-lite selector — an ordered sequence of
    segments where each segment is an object key or a decimal array index —
    applied to the verified whole-file document to yield the effective inherited
    sub-document. Selection happens strictly after hash verification, so the pin
    remains a whole-file content pin regardless of ``payload_path``.
    """

    ref: str
    sha256: str
    payload_path: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _validate_base(self) -> "ContentPinnedJsonBase":
        if not self.ref.strip() or Path(self.ref).is_absolute():
            raise ValueError("content-pinned JSON base ref must be a non-empty relative path")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("content-pinned JSON base sha256 must be lowercase hexadecimal")
        if self.payload_path is not None:
            if not self.payload_path:
                raise ValueError(
                    "content-pinned JSON base payload_path must be omitted or a non-empty sequence"
                )
            for segment in self.payload_path:
                if not segment:
                    raise ValueError(
                        "content-pinned JSON base payload_path segments must be non-empty strings"
                    )
        return self


class MatrixAxisValue(StrictModel):
    """One named value on an authored matrix axis."""

    id: str
    label: str | None = None
    deltas: list[OverridePatch] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_value(self) -> "MatrixAxisValue":
        _validate_path_safe_id(self.id, "axis value id")
        paths = [delta.path for delta in self.deltas]
        if len(paths) != len(set(paths)):
            raise ValueError(f"axis value {self.id!r} has duplicate delta paths")
        for delta in self.deltas:
            if delta.op != "remove":
                _require_json_value(delta.value, f"axis value {self.id!r} delta {delta.path!r}")
        return self


class MatrixAxisValueGenerator(StrictModel):
    """Versioned declarative generator that expands to ordered axis values.

    The ``integer_range`` kind binds one delta path to ``range(start, stop, step)``
    and names each generated value by formatting ``id_format`` over the fields
    ``{value}`` and ``{index}``. Expansion is deterministic: one declaration always
    yields the same ordered values, ids, and deltas as the equivalent
    hand-enumerated axis. ``kind`` leaves room for further declarative forms, such
    as an authored value list mapped over one path.
    """

    schema_id: str = MATRIX_AXIS_VALUE_GENERATOR_SCHEMA_ID
    schema_version: str = MATRIX_AXIS_VALUE_GENERATOR_SCHEMA_VERSION
    kind: Literal["integer_range"] = "integer_range"
    path: str
    start: int
    stop: int
    step: int = 1
    id_format: str
    op: Literal["add", "replace"] = "replace"

    @model_validator(mode="after")
    def _validate_generator(self) -> "MatrixAxisValueGenerator":
        if self.schema_id != MATRIX_AXIS_VALUE_GENERATOR_SCHEMA_ID:
            raise ValueError(f"unsupported axis value generator schema_id: {self.schema_id!r}")
        if self.schema_version != MATRIX_AXIS_VALUE_GENERATOR_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported axis value generator schema_version: {self.schema_version!r}"
            )
        _validate_dotted_path(self.path, "axis value generator path")
        _validate_generator_id_format(self.id_format)
        if self.step == 0:
            raise ValueError("axis value generator step must be non-zero")
        if not range(self.start, self.stop, self.step):
            raise ValueError(
                "axis value generator produces no values: "
                f"start={self.start}, stop={self.stop}, step={self.step}"
            )
        expand_axis_value_generator(self)
        return self


class MatrixAxis(StrictModel):
    """One ordered authored axis with ordered named values.

    An axis declares either enumerated ``values`` or one ``generator``; read the
    ordered values through :meth:`resolved_values` rather than the ``values``
    field, which is empty for a generator-form axis.
    """

    id: str
    label: str | None = None
    values: list[MatrixAxisValue] = Field(default_factory=list)
    generator: MatrixAxisValueGenerator | None = None

    def resolved_values(self) -> list[MatrixAxisValue]:
        """Return this axis's ordered canonical values, expanding any generator."""
        if self.generator is None:
            return list(self.values)
        return expand_axis_value_generator(self.generator)

    @model_validator(mode="after")
    def _validate_axis(self) -> "MatrixAxis":
        _validate_path_safe_id(self.id, "axis id")
        if self.generator is not None and self.values:
            raise ValueError(f"axis {self.id!r} cannot declare both values and a generator")
        if self.generator is None and not self.values:
            raise ValueError(f"axis {self.id!r} requires enumerated values or a generator")
        value_ids = [value.id for value in self.resolved_values()]
        if len(value_ids) != len(set(value_ids)):
            raise ValueError(f"axis {self.id!r} value ids must be unique")
        return self


class MatrixAxisCoordinate(StrictModel):
    """One deterministic coordinate selected from an ordered axis product."""

    row_id: str
    value_indices: dict[str, int]
    value_ids: dict[str, str]
    deltas: list[OverridePatch] = Field(default_factory=list)


class RowDerivation(StrictModel):
    """A value derived after a row's ordered deltas have been applied."""

    output_path: str
    query: ValueExpr

    @model_validator(mode="after")
    def _validate_output_path(self) -> "RowDerivation":
        _validate_dotted_path(self.output_path, "derivations.output_path")
        return self


class MatrixRow(StrictModel):
    """One named condition row with ordered typed deltas."""

    row_id: str
    label: str | None = None
    deltas: list[OverridePatch] = Field(default_factory=list)
    derivations: list[RowDerivation] = Field(default_factory=list)
    output_path: str | None = None
    spec_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_row(self) -> "MatrixRow":
        if not _PATH_SAFE_RE.fullmatch(self.row_id):
            raise ValueError(f"row.row_id is not path-safe: {self.row_id!r}")
        paths = [derivation.output_path for derivation in self.derivations]
        if len(paths) != len(set(paths)):
            raise ValueError("row.derivations output_path values must be unique")
        for field_name in ("output_path", "spec_path"):
            value = getattr(self, field_name)
            if value is not None and Path(value).is_absolute():
                raise ValueError(f"row.{field_name} must be relative")
        return self


class RowMatrixSpec(StrictModel, Generic[PayloadT]):
    """Generic base plus condition rows and external derivation sources."""

    base: PayloadT
    rows: list[MatrixRow] = Field(min_length=1)
    sources: list[SourceBinding] = Field(default_factory=list)
    derivations: list[RowDerivation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_matrix(self) -> "RowMatrixSpec[PayloadT]":
        row_ids = [row.row_id for row in self.rows]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError("rows row_id values must be unique")
        aliases = [source.alias for source in self.sources]
        if len(aliases) != len(set(aliases)):
            raise ValueError("sources aliases must be unique")
        paths = [derivation.output_path for derivation in self.derivations]
        if len(paths) != len(set(paths)):
            raise ValueError("derivations output_path values must be unique")
        return self


class MaterializedMatrixRow(StrictModel, Generic[PayloadT]):
    """One fully resolved row and its deterministic relative paths."""

    row_id: str
    payload: PayloadT
    output_path: str
    spec_path: str


def load_content_pinned_json_base(
    base: ContentPinnedJsonBase,
    *,
    repo_root: Path | str | None,
    read_text: AuthoringTextReader = read_authoring_text,
) -> dict[str, Any]:
    """Load and verify one canonical-JSON content-pinned object."""
    _document, selected = load_content_pinned_json_document(
        base, repo_root=repo_root, read_text=read_text
    )
    return selected


def load_content_pinned_json_document(
    base: ContentPinnedJsonBase,
    *,
    repo_root: Path | str | None,
    read_text: AuthoringTextReader = read_authoring_text,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one verified whole document and its selected object.

    The whole document is returned for durable custody; the selected object is
    exactly the value returned by :func:`load_content_pinned_json_base`.
    """
    if repo_root is None:
        raise ValueError("content-pinned JSON base requires repo_root")
    root = Path(repo_root).resolve()
    target = (root / base.ref).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"content-pinned JSON base escapes repo_root: {base.ref!r}") from exc
    try:
        # The content pin is computed over the *parsed* document, so a base that
        # stated one member twice could satisfy its pin with the last value while
        # a reader of the bytes sees the first. The strict loader refuses it.
        text = read_text(Path(repo_root) / base.ref, resolved_path=target)
        payload = strict_json_loads(text, ref=base.ref)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load content-pinned JSON base {base.ref!r}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("content-pinned JSON base must contain a JSON object")
    _require_json_value(payload, "content-pinned JSON base")
    actual_sha256 = sha256_bytes(canonical_json_bytes(payload))
    if actual_sha256 != base.sha256:
        raise ValueError(
            f"content-pinned JSON base hash mismatch for {base.ref!r}: "
            f"expected {base.sha256}, got {actual_sha256}"
        )
    selected = (
        payload
        if base.payload_path is None
        else _select_payload_sub_document(payload, base.payload_path, base.ref)
    )
    if not isinstance(selected, dict):
        raise ValueError(
            f"content-pinned JSON base {base.ref!r} payload_path "
            f"{list(base.payload_path)!r} must select a JSON object, got "
            f"{type(selected).__name__}"
        )
    return payload, selected


def _select_payload_sub_document(
    payload: dict[str, Any],
    payload_path: Sequence[str],
    ref: str,
) -> Any:
    """Resolve a JSON-pointer-lite selector against a verified whole-file document.

    Fails closed on a missing object key, an out-of-range array index, a malformed
    array-index segment, or traversal into a scalar where a container is required.
    """
    node: Any = payload
    for depth, segment in enumerate(payload_path):
        location = f"{ref!r} payload_path {list(payload_path)!r} segment {depth} ({segment!r})"
        if isinstance(node, dict):
            if segment not in node:
                raise ValueError(f"{location}: missing object key")
            node = node[segment]
        elif isinstance(node, list):
            if not _ARRAY_INDEX_RE.fullmatch(segment):
                raise ValueError(
                    f"{location}: array index must be a canonical non-negative decimal integer"
                )
            index = int(segment)
            if index >= len(node):
                raise ValueError(
                    f"{location}: array index out of range for length {len(node)}"
                )
            node = node[index]
        else:
            raise ValueError(
                f"{location}: cannot traverse into {type(node).__name__} value"
            )
    return node


class InheritedSubDocument(StrictModel):
    """One content-pinned sub-document grafted into an absent local target path.

    ``parent`` reuses :class:`ContentPinnedJsonBase`, so inheritance retains
    whole-file digest verification followed by ``payload_path`` sub-document
    selection. ``target`` is the dotted path in the consuming document where the
    resolved sub-document is grafted; its leaf must be absent locally so neither
    the local document nor its inherited content silently shadows the other.
    """

    target: str
    parent: ContentPinnedJsonBase

    @model_validator(mode="after")
    def _validate_entry(self) -> "InheritedSubDocument":
        _validate_dotted_path(self.target, "inherited sub-document target")
        return self


class SourceDocumentInheritance(StrictModel):
    """Reserved inheritance envelope declared inside a source document.

    This is the value stored under :data:`SOURCE_DOCUMENT_INHERITANCE_KEY`. It is
    a durable authored structure, so it carries an explicit schema identity and
    rejects unknown versions rather than silently accepting them.
    """

    schema_id: str = SOURCE_DOCUMENT_INHERITANCE_SCHEMA_ID
    schema_version: str = SOURCE_DOCUMENT_INHERITANCE_SCHEMA_VERSION
    inherit: list[InheritedSubDocument] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_inheritance(self) -> "SourceDocumentInheritance":
        if self.schema_id != SOURCE_DOCUMENT_INHERITANCE_SCHEMA_ID:
            raise ValueError(
                f"unsupported source document inheritance schema_id {self.schema_id!r}"
            )
        if self.schema_version != SOURCE_DOCUMENT_INHERITANCE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported source document inheritance schema_version "
                f"{self.schema_version!r}"
            )
        targets = [entry.target for entry in self.inherit]
        if len(set(targets)) != len(targets):
            raise ValueError("source document inheritance targets must be unique")
        return self


def materialize_inherited_document(
    document: Any,
    *,
    repo_root: Path | str | None,
    read_text: AuthoringTextReader = read_authoring_text,
) -> Any:
    """Resolve declared content-pinned inheritance into the effective document.

    A document that is not a mapping, or a mapping without the reserved
    :data:`SOURCE_DOCUMENT_INHERITANCE_KEY`, is returned unchanged so behavior is
    byte-for-byte identical to loading it directly. Otherwise each declared parent
    is verified and sub-document-selected through
    :func:`load_content_pinned_json_base` (whole-file digest, then ``payload_path``)
    and grafted into its absent target path. The validated declaration is retained
    under the reserved key as in-band inheritance provenance (pinned parent ``ref``,
    ``sha256``, and ``payload_path`` per grafted target).

    Fails closed on digest mismatch, missing or invalid ``payload_path``, and any
    collision where an inherited subtree would land on a locally-present key.
    """
    effective, _custody = materialize_inherited_document_with_custody(
        document, repo_root=repo_root, read_text=read_text
    )
    return effective


def materialize_inherited_document_with_custody(
    document: Any,
    *,
    repo_root: Path | str | None,
    read_text: AuthoringTextReader = read_authoring_text,
) -> tuple[Any, list[tuple[ContentPinnedJsonBase, dict[str, Any], dict[str, Any]]]]:
    """Materialize inheritance and return every verified whole/selected parent."""
    if not isinstance(document, dict) or SOURCE_DOCUMENT_INHERITANCE_KEY not in document:
        return document, []
    declaration = SourceDocumentInheritance.model_validate(
        document[SOURCE_DOCUMENT_INHERITANCE_KEY]
    )
    effective = deepcopy(document)
    custody: list[tuple[ContentPinnedJsonBase, dict[str, Any], dict[str, Any]]] = []
    for entry in declaration.inherit:
        whole_document, sub_document = load_content_pinned_json_document(
            entry.parent, repo_root=repo_root, read_text=read_text
        )
        _graft_absent_target(effective, entry.target, sub_document)
        custody.append((entry.parent, whole_document, sub_document))
    return effective, custody


def _graft_absent_target(root: dict[str, Any], target: str, value: Any) -> None:
    """Graft ``value`` at a dotted ``target`` whose leaf must be absent locally."""
    parts = target.split(".")
    node: Any = root
    for depth, part in enumerate(parts[:-1]):
        traversed = ".".join(parts[: depth + 1])
        if isinstance(node, dict):
            if part not in node:
                node[part] = {}
            node = node[part]
            continue
        if isinstance(node, list):
            if not _ARRAY_INDEX_RE.fullmatch(part):
                raise ValueError(
                    f"inherited target {target!r} segment {traversed!r} must be a "
                    "canonical non-negative decimal array index"
                )
            index = int(part)
            if index >= len(node):
                raise ValueError(
                    f"inherited target {target!r} segment {traversed!r} array index "
                    f"out of range for length {len(node)}"
                )
            node = node[index]
            continue
        raise ValueError(
            f"inherited target {target!r} traverses scalar segment {traversed!r}"
        )
    if not isinstance(node, dict):
        traversed = ".".join(parts[:-1])
        raise ValueError(
            f"inherited target {target!r} leaf parent {traversed!r} is not an object"
        )
    leaf = parts[-1]
    if leaf in node:
        raise ValueError(
            f"inherited target {target!r} collides with a locally-present key"
        )
    node[leaf] = deepcopy(value)


def ordered_index_product(
    axis_lengths: Sequence[tuple[str, int]],
) -> list[dict[str, int]]:
    """Return a stable Cartesian product in authored axis order."""
    validate_index_product_axes(axis_lengths)
    coordinates: list[dict[str, int]] = [{}]
    for axis_id, length in axis_lengths:
        coordinates = [
            {**coordinate, axis_id: index}
            for coordinate in coordinates
            for index in range(length)
        ]
    return coordinates


def validate_index_product_axes(axis_lengths: Sequence[tuple[str, int]]) -> None:
    """Validate the ordered axes of a Cartesian product before enumeration."""
    if not axis_lengths:
        raise ValueError("ordered index product requires at least one axis")
    axis_ids = [axis_id for axis_id, _ in axis_lengths]
    if len(axis_ids) != len(set(axis_ids)):
        raise ValueError("ordered index product axis ids must be unique")
    for axis_id, length in axis_lengths:
        _validate_path_safe_id(axis_id, "axis id")
        if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
            raise ValueError(f"ordered index product axis {axis_id!r} must be non-empty")

def expand_axis_value_generator(
    generator: MatrixAxisValueGenerator,
) -> list[MatrixAxisValue]:
    """Expand one declarative generator into its ordered canonical axis values."""
    values: list[MatrixAxisValue] = []
    for index, number in enumerate(range(generator.start, generator.stop, generator.step)):
        try:
            value_id = generator.id_format.format(value=number, index=index)
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(
                f"axis value generator id_format {generator.id_format!r} failed for "
                f"value {number}: {exc}"
            ) from exc
        values.append(
            MatrixAxisValue(
                id=value_id,
                deltas=[OverridePatch(path=generator.path, value=number, op=generator.op)],
            )
        )
    return values


def expand_matrix_axes(axes: Sequence[MatrixAxis]) -> list[MatrixAxisCoordinate]:
    """Expand ordered authored axes into canonical row coordinates and deltas."""
    axis_ids = [axis.id for axis in axes]
    if len(axis_ids) != len(set(axis_ids)):
        raise ValueError("matrix axis ids must be unique")
    axis_values = [axis.resolved_values() for axis in axes]
    indexed = ordered_index_product(
        [(axis.id, len(values)) for axis, values in zip(axes, axis_values)]
    )
    coordinates: list[MatrixAxisCoordinate] = []
    row_ids: set[str] = set()
    expected = set(axis_ids)
    for indices in indexed:
        if set(indices) != expected:
            raise ValueError("ordered axis product produced an incomplete coordinate")
        selected = [
            values[indices[axis.id]] for axis, values in zip(axes, axis_values)
        ]
        value_ids = {axis.id: value.id for axis, value in zip(axes, selected)}
        row_id = "--".join(
            f"{axis.id}-{value.id}" for axis, value in zip(axes, selected)
        )
        _validate_path_safe_id(row_id, "generated row id")
        if row_id in row_ids:
            raise ValueError(f"generated matrix row_id collision: {row_id!r}")
        row_ids.add(row_id)
        deltas = [delta for value in selected for delta in value.deltas]
        paths = [delta.path for delta in deltas]
        duplicates = sorted({path for path in paths if paths.count(path) > 1})
        if duplicates:
            raise ValueError(
                f"matrix coordinate {row_id!r} selects duplicate delta paths {duplicates!r}"
            )
        coordinates.append(
            MatrixAxisCoordinate(
                row_id=row_id,
                value_indices=dict(indices),
                value_ids=value_ids,
                deltas=deltas,
            )
        )
    return coordinates


def derive_row_path(
    row_id: str,
    *,
    explicit_path: str | None = None,
    suffix: str = ".json",
) -> str:
    """Derive a relative path from ``row_id``, unless an explicit path is supplied."""
    if not _PATH_SAFE_RE.fullmatch(row_id):
        raise ValueError(f"row_id is not path-safe: {row_id!r}")
    path = explicit_path if explicit_path is not None else f"{row_id}{suffix}"
    if Path(path).is_absolute():
        raise ValueError("row path must be relative")
    return path


def materialize_matrix_rows(
    spec: RowMatrixSpec[PayloadT],
    *,
    repo_root: Path | str | None = None,
    read_text: AuthoringTextReader = read_authoring_text,
) -> list[MaterializedMatrixRow[PayloadT]]:
    """Apply each row's deltas, then evaluate its derivations against that row."""
    if spec.sources and repo_root is None:
        raise ValueError("matrix sources require repo_root")
    source_context = (
        load_expression_context(spec.sources, repo_root, read_text=read_text)
        if spec.sources
        else ExpressionContext()
    )
    materialized: list[MaterializedMatrixRow[PayloadT]] = []
    for row in spec.rows:
        payload = apply_override_patches(
            spec.base.model_dump(mode="python"),
            row.deltas,
            error_context="delta",
        )
        apply_row_derivations(
            payload,
            [*spec.derivations, *row.derivations],
            source_context=source_context,
        )
        materialized.append(
            MaterializedMatrixRow[spec.base.__class__](
                row_id=row.row_id,
                payload=spec.base.__class__.model_validate(payload),
                output_path=derive_row_path(
                    row.row_id, explicit_path=row.output_path, suffix="/output.json"
                ),
                spec_path=derive_row_path(
                    row.row_id, explicit_path=row.spec_path, suffix="/spec.json"
                ),
            )
        )
    return materialized


def apply_row_derivations(
    payload: dict[str, Any],
    derivations: Sequence[RowDerivation],
    *,
    source_context: ExpressionContext,
    before_write: Callable[[dict[str, Any], str], None] | None = None,
) -> None:
    """Evaluate ordered derivations against the evolving delta-applied row payload.

    ``before_write`` is an optional domain guard; without it, derivations retain the
    existing matrix-core overwrite behavior.
    """
    for derivation in derivations:
        if before_write is not None:
            before_write(payload, derivation.output_path)
        context = ExpressionContext(
            items={
                **source_context.items,
                "row": ContextItem(kind="matrix_row", payload=deepcopy(payload)),
            }
        )
        value = evaluate_query(derivation.query, context)
        set_dotted_path(payload, derivation.output_path, value)


def apply_override_patches(
    payload: dict[str, Any],
    patches: list[OverridePatch | dict[str, Any]],
    *,
    error_context: Literal["override", "delta"] = "override",
) -> dict[str, Any]:
    """Apply ordered ``OverridePatch`` records to a deep copy of ``payload``."""
    result = deepcopy(payload)
    for raw_patch in patches:
        patch = (
            raw_patch
            if isinstance(raw_patch, OverridePatch)
            else OverridePatch.model_validate(raw_patch)
        )
        apply_override_patch_in_place(result, patch, error_context=error_context)
    return result


def apply_override_patch_in_place(
    root: dict[str, Any],
    patch: OverridePatch,
    *,
    error_context: Literal["override", "delta"] = "override",
) -> None:
    """Apply one validated patch to an owned payload, copying any inserted value."""
    apply_value_patch_in_place(
        root, patch.path.split("."), patch.op, patch.value, path=patch.path, error_context=error_context, by_id=True,
    )


def apply_override_patch_copy(root: dict[str, Any], patch: OverridePatch) -> dict[str, Any]:
    """Apply a patch to new containers along its path, sharing untouched subtrees."""
    return _patch_copy(root, patch, by_id=True)


type ExpressionSiteKind = Literal["expression", "time"]
type ExpressionSite = Callable[[tuple[str, ...]], ExpressionSiteKind | None]


@dataclass(frozen=True)
class ExpressionNodePatch:
    """A whole-field stored write and the matching node edit for provenance."""

    field: tuple[str, ...]
    before: dict[str, Any]
    node: OverridePatch


_EXPRESSION = TypeAdapter(Expression)
_STORED_TIME = TypeAdapter(StoredTime)


def prepare_stored_patch(
    root: dict[str, Any], patch: OverridePatch, *, expression_site: ExpressionSite | None = None,
) -> tuple[OverridePatch, ExpressionNodePatch | None]:
    """Translate a typed expression-node path to one validated authored field write."""
    site = _stored_expression_site(root, patch.path, expression_site)
    if site is None:
        return patch, None
    field, kind, raw = site
    parts = tuple(patch.path.split("."))[len(field):]
    leaf_value = _validate_expression_node_path(parts, patch.path)
    before = _parse_expression_for_patch(raw, patch.path).model_dump(mode="json")
    if kind == "time" and isinstance(_STORED_TIME.validate_python(raw), UnresolvedValue):
        raise ValueError(f"Expected a literal time at {patch.path!r}.")
    value = patch.value
    if not leaf_value and patch.op != "remove":
        value = _parse_expression_for_patch(value, patch.path).model_dump(mode="json")
    node = patch.model_copy(update={"path": ".".join(parts), "value": value})
    changed = replay_stored_patch_copy(before, node)
    text = print_stored_expression(_EXPRESSION.validate_python(changed))
    _parse_expression_for_patch(text, patch.path)
    if kind == "time" and isinstance(_STORED_TIME.validate_python(text), UnresolvedValue):
        raise ValueError(f"Expected a literal time at {patch.path!r}.")
    scalar = patch.model_copy(update={"path": ".".join(field), "op": "replace", "value": text})
    return scalar, ExpressionNodePatch(field, before, node)


def _parse_expression_for_patch(raw: Any, path: str) -> Expression:
    if type(raw) not in (str, float, int, bool):
        raise ValueError(f"Expected an authored expression scalar or text at {path!r}.")
    parsed = parse_stored_expression(raw)
    if isinstance(parsed, tuple):
        raise ValueError(f"Cannot patch expression at {path!r}: {parsed[0].reason}")
    return parsed.value


def _validate_expression_node_path(parts: tuple[str, ...], path: str) -> bool:
    position = 0
    while position + 1 < len(parts):
        parent, child = parts[position:position + 2]
        if parent == "args" and _ARRAY_INDEX_RE.fullmatch(child) is None:
            raise ValueError(f"An expression argument needs a non-negative decimal index: {path!r}")
        if parent not in {"args", "kwargs"}:
            raise ValueError(f"Expected an expression argument or keyword: {path!r}")
        position += 2
    if position == len(parts):
        return False
    if parts[position] == "value":
        return True
    raise ValueError(f"Expected an expression node or its literal value: {path!r}")


def apply_stored_patch_in_place(
    root: dict[str, Any],
    patch: OverridePatch,
    *,
    error_context: Literal["override", "delta"] = "override",
    expression_site: ExpressionSite | None = None,
) -> ExpressionNodePatch | None:
    """Apply an authored sweep patch, with absent object keys already removed."""
    physical, expression = prepare_stored_patch(root, patch, expression_site=expression_site)
    apply_value_patch_in_place(
        root, physical.path.split("."), physical.op, physical.value, path=patch.path, error_context=error_context,
        allow_absent_remove=True,
    )
    return expression


def apply_stored_patch_copy(
    root: dict[str, Any], patch: OverridePatch, *, expression_site: ExpressionSite | None = None,
) -> dict[str, Any]:
    """Apply an authored sweep patch, copying the containers along its path."""
    physical, _ = prepare_stored_patch(root, patch, expression_site=expression_site)
    return replay_stored_patch_copy(root, physical)


def replay_stored_patch_copy(root: dict[str, Any], patch: OverridePatch) -> dict[str, Any]:
    """Replay a checked path in an origin tree, or apply an internally generated path."""
    return _patch_copy(root, patch, by_id=False, allow_absent_remove=True)


def _patch_copy(
    root: dict[str, Any], patch: OverridePatch, *, by_id: bool, allow_absent_remove: bool = False,
) -> dict[str, Any]:
    parts = patch.path.split(".")
    result = copy(root)
    parent = _resolve_patch_parent(
        result, parts, path=patch.path, error_context="override", copy_path=True, by_id=by_id,
    )
    _apply_patch_target(
        parent, parts[-1], patch.op, patch.value, patch.path, "override",
        by_id=by_id, allow_absent_remove=allow_absent_remove,
    )
    return result


def apply_value_patch_in_place(
    root: dict[str, Any],
    parts: Sequence[str],
    op: Literal["add", "replace", "remove"],
    value: Any,
    *,
    path: str,
    error_context: Literal["override", "delta"] = "override",
    by_id: bool = False,
    allow_absent_remove: bool = False,
) -> None:
    """Apply the common patch semantics to an already decoded path.

    Lowered variation paths use ``by_id`` to name identified list members. Authored sweep paths
    can remove an absent object key, while revision-specific edits require the key to exist.
    """
    parent = _resolve_patch_parent(root, parts, path=path, error_context=error_context, by_id=by_id)
    _apply_patch_target(
        parent, parts[-1], op, value, path, error_context, by_id=by_id, allow_absent_remove=allow_absent_remove,
    )


def ensure_value_collection(
    root: dict[str, Any], parts: Sequence[str], empty: dict | list, *, path: str,
) -> None:
    parent = _resolve_patch_parent(root, parts, path=path, error_context="override")
    key = parts[-1]
    if not isinstance(parent, dict):
        raise ValueError(f"collection path must name an object member: {path!r}")
    if key not in parent:
        parent[key] = empty
    elif not isinstance(parent[key], dict if isinstance(empty, dict) else list):
        raise ValueError(f"collection at {path!r} has the wrong kind")


class MirrorList(list[Any]):
    """A list that mirrors another one member for member, and names its members by that list's ids."""

    def __init__(self, children: Iterable[Any], mirrored: Sequence[Any]):
        super().__init__(children)
        self.ids = _carried_ids(mirrored)


def _carried_ids(items: Sequence[Any]) -> tuple[str, ...] | None:
    if isinstance(items, MirrorList):
        return items.ids
    if not items or not all(isinstance(item, dict) and isinstance(item.get("id"), str) for item in items):
        return None
    return tuple(item["id"] for item in items)


def member_ids(items: Sequence[Any]) -> tuple[str, ...] | None:
    """The ids of a list whose members each carry one, or ``None`` for a list addressed by index."""
    ids = _carried_ids(items)
    if ids is None:
        return None
    if len(set(ids)) != len(ids):
        repeated = sorted({member for member in ids if ids.count(member) > 1})
        raise ValueError(f"list members share an id: {', '.join(repeated)}")
    return ids


def resolve_segment(container: Any, segment: str, *, path: str, by_id: bool = True) -> str | int | None:
    """The key or index ``segment`` names in ``container``, or ``None`` when it names no member.

    In a lowered path (``by_id``), a member of a list whose members carry ids is named by its id; an index or an
    unknown id under such a list is refused, naming the ids it has. Any other list is addressed by index.
    """
    if isinstance(container, dict):
        return segment if segment in container else None
    if not isinstance(container, list):
        return None
    ids = member_ids(container) if by_id else None
    if ids is not None:
        if segment in ids:
            return ids.index(segment)
        reason = "is addressed by its id" if segment.isdigit() else "is not an id of this list"
        raise ValueError(f"{segment!r} {reason}; the ids are {', '.join(ids)}: {path!r}")
    if segment.isdigit() and int(segment) < len(container):
        return int(segment)
    return None


def validate_stored_patch_path(root: Any, path: str) -> None:
    """Validate the visible path before a document reference is followed."""
    _stored_expression_site(root, path, None)


def _stored_expression_site(
    root: Any, path: str, expression_site: ExpressionSite | None,
) -> tuple[tuple[str, ...], ExpressionSiteKind, Any] | None:
    parts = tuple(path.split("."))
    value = root
    for index, part in enumerate(parts):
        if isinstance(value, list):
            raise ValueError(f"A stored path cannot traverse an array: {path!r}")
        key = resolve_segment(value, part, path=path, by_id=False)
        if key is None:
            return None
        child = value[key]
        if index < len(parts) - 1:
            if part in {"links", "task_links"} and isinstance(child, dict) and all(
                isinstance(link, str) for link in child.values()
            ):
                raise ValueError(f"Link members require a JSON pointer: {path!r}")
            if expression_site is not None and not isinstance(child, (dict, list)):
                prefix = parts[:index + 1]
                if (kind := expression_site(prefix)) is not None:
                    return prefix, kind, child
        value = child
    return None


def stored_parts(root: Any, parts: Sequence[str]) -> tuple[str, ...]:
    """The segments of a JSON pointer's ``parts`` into ``root`` as a stored path names them: a member of a list of
    identified items by its id rather than its index."""
    named, current = [], root
    for part in parts:
        if isinstance(current, list) and (ids := member_ids(current)) is not None and part.isdigit() and int(part) < len(ids):
            named.append(ids[int(part)])
            current = current[int(part)]
            continue
        named.append(part)
        key = resolve_segment(current, part, path="/".join(parts), by_id=False)
        current = current[key] if key is not None else None
    return tuple(named)


def _apply_patch_target(
    parent: Any,
    leaf: str,
    op: Literal["add", "replace", "remove"],
    value: Any,
    path: str,
    error_context: Literal["override", "delta"],
    *,
    by_id: bool,
    allow_absent_remove: bool = False,
) -> None:
    if op == "add" and isinstance(parent, list) and (
        leaf == "-" or (not by_id or member_ids(parent) is None) and leaf.isdigit() and int(leaf) == len(parent)
    ):
        parent.append(deepcopy(value))
        return
    key = resolve_segment(parent, leaf, path=path, by_id=by_id)
    if op == "add":
        if isinstance(parent, list) and key is not None:
            parent.insert(key, deepcopy(value))
            return
        if key is not None:
            if error_context == "delta":
                raise ValueError(f"add delta path already exists: {path!r}")
            raise ValueError(f"override add patch path already exists: {path!r}")
        if not isinstance(parent, dict):
            raise ValueError(f"patch path cannot set segment {leaf!r}: {path!r}")
        parent[leaf] = deepcopy(value)
        return
    if key is None:
        if allow_absent_remove and op == "remove" and isinstance(parent, dict):
            return
        if error_context == "delta":
            raise ValueError(f"{op} delta path is missing: {path!r}")
        if op == "replace":
            raise ValueError(f"override replace patch targets a missing key/index: {path!r}")
        raise ValueError(f"override remove patch targets a missing key/index: {path!r}")
    if op == "replace":
        parent[key] = deepcopy(value)
    else:
        del parent[key]


def _resolve_patch_parent(
    root: Any,
    parts: Sequence[str],
    *,
    path: str,
    error_context: Literal["override", "delta"],
    copy_path: bool = False,
    by_id: bool = False,
) -> Any:
    if not parts:
        raise ValueError(f"patch path must be non-empty: {path!r}")
    current = root
    for part in parts[:-1]:
        key = resolve_segment(current, part, path=path, by_id=by_id)
        if key is None:
            prefix = "delta path" if error_context == "delta" else "patch path"
            raise ValueError(f"{prefix} cannot traverse missing segment {part!r}: {path!r}")
        child = current[key]
        if copy_path and isinstance(child, (dict, list)):
            child = copy(child)
            current[key] = child
        current = child
    return current


def _validate_dotted_path(path: str, field: str) -> None:
    if not path.strip() or any(not part for part in path.split(".")):
        raise ValueError(f"{field} is not dotted-path-like: {path!r}")


def _validate_generator_id_format(id_format: str) -> None:
    try:
        parsed = list(_FORMATTER.parse(id_format))
    except ValueError as exc:
        raise ValueError(
            f"axis value generator id_format is not a valid format string: {id_format!r}: {exc}"
        ) from exc
    referenced: list[str] = []
    for _literal, field_name, format_spec, _conversion in parsed:
        if field_name is None:
            continue
        if format_spec is not None and "{" in format_spec:
            raise ValueError(
                "axis value generator id_format must not nest replacement fields: "
                f"{id_format!r}"
            )
        if field_name not in _GENERATOR_ID_FORMAT_FIELDS:
            raise ValueError(
                f"axis value generator id_format field {field_name!r} is not one of "
                f"{list(_GENERATOR_ID_FORMAT_FIELDS)}"
            )
        referenced.append(field_name)
    if not referenced:
        raise ValueError(
            "axis value generator id_format must reference at least one of "
            f"{list(_GENERATOR_ID_FORMAT_FIELDS)}: {id_format!r}"
        )


def _validate_path_safe_id(value: str, field: str) -> None:
    if not _PATH_SAFE_RE.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"{field} is not path-safe: {value!r}")


def _require_json_value(value: Any, field: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} must contain finite JSON numbers")
        return
    if isinstance(value, list):
        for item in value:
            _require_json_value(item, field)
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{field} must contain only string JSON object keys")
        for item in value.values():
            _require_json_value(item, field)
        return
    raise ValueError(f"{field} contains a non-JSON value of type {type(value).__name__}")


__all__ = [
    "ContentPinnedJsonBase",
    "InheritedSubDocument",
    "MATRIX_AXIS_VALUE_GENERATOR_SCHEMA_ID",
    "MATRIX_AXIS_VALUE_GENERATOR_SCHEMA_VERSION",
    "SOURCE_DOCUMENT_INHERITANCE_KEY",
    "SOURCE_DOCUMENT_INHERITANCE_SCHEMA_ID",
    "SOURCE_DOCUMENT_INHERITANCE_SCHEMA_VERSION",
    "SourceDocumentInheritance",
    "apply_row_derivations",
    "MaterializedMatrixRow",
    "MatrixAxis",
    "MatrixAxisCoordinate",
    "MatrixAxisValue",
    "MatrixAxisValueGenerator",
    "MatrixRow",
    "RowDerivation",
    "RowMatrixSpec",
    "derive_row_path",
    "expand_axis_value_generator",
    "expand_matrix_axes",
    "load_content_pinned_json_base",
    "load_content_pinned_json_document",
    "materialize_inherited_document",
    "materialize_inherited_document_with_custody",
    "materialize_matrix_rows",
    "ordered_index_product",
]
