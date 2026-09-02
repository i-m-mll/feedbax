"""Domain-independent contracts and expansion for base/row/delta matrices."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
import json
import math
from pathlib import Path
import re
from string import Formatter
from typing import Any, Generic, Literal, TypeVar

from pydantic import Field, model_validator

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
) -> dict[str, Any]:
    """Load and verify one canonical-JSON content-pinned object."""
    _document, selected = load_content_pinned_json_document(base, repo_root=repo_root)
    return selected


def load_content_pinned_json_document(
    base: ContentPinnedJsonBase,
    *,
    repo_root: Path | str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one verified whole document and its selected object.

    The whole document is returned for durable custody; the selected object is
    exactly the value returned by :func:`load_content_pinned_json_base`.
    """
    if repo_root is None:
        raise ValueError("content-pinned JSON base requires repo_root")
    root = Path(repo_root).resolve()
    path = (root / base.ref).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"content-pinned JSON base escapes repo_root: {base.ref!r}") from exc
    try:
        # The content pin is computed over the *parsed* document, so a base that
        # stated one member twice could satisfy its pin with the last value while
        # a reader of the bytes sees the first. The strict loader refuses it.
        payload = strict_json_loads(path.read_text(encoding="utf-8"), ref=base.ref)
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
        document, repo_root=repo_root
    )
    return effective


def materialize_inherited_document_with_custody(
    document: Any,
    *,
    repo_root: Path | str | None,
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
            entry.parent, repo_root=repo_root
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
    if not axis_lengths:
        raise ValueError("ordered index product requires at least one axis")
    axis_ids = [axis_id for axis_id, _ in axis_lengths]
    if len(axis_ids) != len(set(axis_ids)):
        raise ValueError("ordered index product axis ids must be unique")
    for axis_id, length in axis_lengths:
        _validate_path_safe_id(axis_id, "axis id")
        if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
            raise ValueError(f"ordered index product axis {axis_id!r} must be non-empty")

    coordinates: list[dict[str, int]] = [{}]
    for axis_id, length in axis_lengths:
        coordinates = [
            {**coordinate, axis_id: index}
            for coordinate in coordinates
            for index in range(length)
        ]
    return coordinates


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
) -> list[MaterializedMatrixRow[PayloadT]]:
    """Apply each row's deltas, then evaluate its derivations against that row."""
    if spec.sources and repo_root is None:
        raise ValueError("matrix sources require repo_root")
    source_context = (
        load_expression_context(spec.sources, repo_root) if spec.sources else ExpressionContext()
    )
    materialized: list[MaterializedMatrixRow[PayloadT]] = []
    for row in spec.rows:
        payload = _apply_deltas(spec.base.model_dump(mode="python"), row.deltas)
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


def _apply_deltas(payload: dict[str, Any], deltas: list[OverridePatch]) -> dict[str, Any]:
    # Keep this core independent from the training matrix module while reusing
    # the shared OverridePatch contract.
    result = deepcopy(payload)
    for delta in deltas:
        parts = delta.path.split(".")
        parent: Any = result
        for part in parts[:-1]:
            if isinstance(parent, dict) and part in parent:
                parent = parent[part]
            elif isinstance(parent, list) and part.isdigit() and int(part) < len(parent):
                parent = parent[int(part)]
            else:
                raise ValueError(
                    f"delta path cannot traverse missing segment {part!r}: {delta.path!r}"
                )
        leaf = parts[-1]
        exists = (
            leaf in parent
            if isinstance(parent, dict)
            else leaf.isdigit() and int(leaf) < len(parent)
        )
        if delta.op == "add" and exists:
            raise ValueError(f"add delta path already exists: {delta.path!r}")
        if delta.op in {"replace", "remove"} and not exists:
            raise ValueError(f"{delta.op} delta path is missing: {delta.path!r}")
        if delta.op == "remove":
            del parent[leaf if isinstance(parent, dict) else int(leaf)]
        else:
            parent[leaf if isinstance(parent, dict) else int(leaf)] = deepcopy(delta.value)
    return result


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
