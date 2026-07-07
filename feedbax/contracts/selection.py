"""SelectionSpec v2 contracts and deterministic manifest-index query helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from pydantic import Field, TypeAdapter, model_validator

from feedbax.contracts.expressions import (
    ContextItem,
    Expr,
    ExpressionContext,
    ExpressionEvaluationError,
    evaluate_expr,
)
from feedbax.contracts.manifest import ParentRef, StrictModel, utc_now


SELECTION_SPEC_SCHEMA_ID = "feedbax.spec.selection"
SELECTION_SPEC_SCHEMA_VERSION_V1 = "feedbax.spec.selection.v1"
SELECTION_SPEC_SCHEMA_VERSION = "feedbax.spec.selection.v2"

SelectionMode = Literal["explicit", "query", "frozen"]
SelectionReprocessMode = Literal["missing", "missing_failed", "all", "stale"]


class TopKByMetricPerGroup(StrictModel):
    """Select the top K rows in each group after other predicates match."""

    metric_path: str
    group_by_path: str
    k: int = Field(default=1, ge=1)
    order: Literal["asc", "desc"] = "asc"


class ManifestPredicate(StrictModel):
    """Deterministic predicate over typed manifest-index rows."""

    manifest_kind: str = "TrainingRunManifest"
    run_ids: list[str] = Field(default_factory=list)
    source_set_ids: list[str] = Field(default_factory=list)
    statuses: list[str] = Field(default_factory=list)
    has_checkpoint: bool | None = None
    tags: list[str] = Field(default_factory=list)
    metadata_equals: dict[str, Any] = Field(default_factory=dict)
    params_equals: dict[str, Any] = Field(default_factory=dict)
    path_equals: dict[str, Any] = Field(default_factory=dict)
    expression: dict[str, Any] | None = None
    top_k_by_metric_per_group: TopKByMetricPerGroup | None = None


class SelectionSpec(StrictModel):
    """Versioned selection contract for explicit, live-query, and frozen inputs."""

    schema_id: Literal["feedbax.spec.selection"] = SELECTION_SPEC_SCHEMA_ID
    schema_version: Literal["feedbax.spec.selection.v2"] = SELECTION_SPEC_SCHEMA_VERSION
    mode: SelectionMode = "explicit"
    manifest_kind: str = "TrainingRunManifest"
    ids: list[str] = Field(default_factory=list)
    query: ManifestPredicate | None = None
    frozen_refs: list[ParentRef] = Field(default_factory=list)
    frozen_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_selection_spec(self) -> "SelectionSpec":
        if self.schema_id != SELECTION_SPEC_SCHEMA_ID:
            raise ValueError(
                f"unsupported SelectionSpec schema_id {self.schema_id!r}; "
                f"expected {SELECTION_SPEC_SCHEMA_ID!r}"
            )
        if self.schema_version != SELECTION_SPEC_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported SelectionSpec schema_version {self.schema_version!r}; "
                f"expected {SELECTION_SPEC_SCHEMA_VERSION!r}"
            )
        if self.mode == "explicit" and self.query is not None:
            raise ValueError("explicit SelectionSpec must not include query")
        if self.mode == "query" and self.query is None:
            raise ValueError("query SelectionSpec requires query")
        if self.mode == "frozen":
            if self.query is None:
                raise ValueError("frozen SelectionSpec requires query")
            if not self.frozen_refs:
                raise ValueError("frozen SelectionSpec requires frozen_refs")
        return self


class SelectionPreview(StrictModel):
    """Stable preview of a selection query over manifest-index rows."""

    selection_spec: SelectionSpec
    match_count: int
    parent_refs: list[ParentRef]
    truncated: bool = False


class SelectionRefreshDiff(StrictModel):
    """Manual refresh result for a frozen selection."""

    frozen_refs: list[ParentRef]
    current_refs: list[ParentRef]
    new_refs: list[ParentRef]
    gone_refs: list[ParentRef]
    unchanged_refs: list[ParentRef]
    reprocess_counts: dict[SelectionReprocessMode, int]


class ManifestIndexRow(StrictModel):
    """Typed, testable view of one manifest-index record."""

    id: str
    kind: str
    schema_version: str
    created_at: str
    status: str | None = None
    path: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class _ScoredRow:
    row: ManifestIndexRow
    metric: float
    group: Any


_EXPR_ADAPTER: TypeAdapter[Expr] = TypeAdapter(Expr)


def migrate_selection_spec_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate legacy id-list selection payloads to ``SelectionSpec`` v2."""

    data = dict(payload)
    version = data.get("schema_version")
    if version == SELECTION_SPEC_SCHEMA_VERSION:
        return data
    if version not in (None, SELECTION_SPEC_SCHEMA_VERSION_V1):
        raise ValueError(
            f"unsupported SelectionSpec schema_version {version!r}; "
            f"expected {SELECTION_SPEC_SCHEMA_VERSION!r}"
        )

    ids = _legacy_selection_ids(data)
    if ids is None and version == SELECTION_SPEC_SCHEMA_VERSION_V1:
        ids = []
    if ids is None:
        raise ValueError("legacy SelectionSpec payload requires a recognized id list")

    manifest_kind = _legacy_manifest_kind(data)
    migrated = {
        "schema_id": SELECTION_SPEC_SCHEMA_ID,
        "schema_version": SELECTION_SPEC_SCHEMA_VERSION,
        "mode": "explicit",
        "manifest_kind": manifest_kind,
        "ids": ids,
        "metadata": {
            key: value
            for key, value in data.items()
            if key
            not in {
                "schema_id",
                "schema_version",
                "mode",
                "manifest_kind",
                "ids",
                "training_run_ids",
                "eval_run_ids",
                "evaluation_run_ids",
                "analysis_run_ids",
                "report_ids",
            }
        },
    }
    SelectionSpec.model_validate(migrated)
    return migrated


def freeze_selection_spec(
    spec: SelectionSpec,
    rows: Sequence[ManifestIndexRow | Mapping[str, Any]],
) -> SelectionSpec:
    """Return a frozen selection with a staged snapshot of matched ``ParentRef`` rows."""

    if spec.mode == "explicit":
        manifest_kind = spec.manifest_kind
        refs = [
            ParentRef(
                kind=manifest_kind,
                id=id_,
                role=_role_for_kind(manifest_kind),
            )
            for id_ in spec.ids
        ]
        query = ManifestPredicate(manifest_kind=manifest_kind, run_ids=list(spec.ids))
    elif spec.query is not None:
        refs = select_parent_refs(spec.query, rows)
        query = spec.query
        manifest_kind = query.manifest_kind
    else:
        raise ValueError("cannot freeze SelectionSpec without explicit ids or query")
    return SelectionSpec(
        mode="frozen",
        manifest_kind=manifest_kind,
        query=query,
        frozen_refs=refs,
        frozen_at=utc_now(),
        metadata=dict(spec.metadata),
    )


def preview_selection_spec(
    spec: SelectionSpec,
    rows: Sequence[ManifestIndexRow | Mapping[str, Any]],
    *,
    limit: int | None = None,
) -> SelectionPreview:
    """Preview a SelectionSpec over manifest-index rows without mutating it."""

    if spec.mode == "explicit":
        refs = [
            ParentRef(kind=spec.manifest_kind, id=id_, role=_role_for_kind(spec.manifest_kind))
            for id_ in spec.ids
        ]
    elif spec.mode == "frozen":
        refs = list(spec.frozen_refs)
    elif spec.query is not None:
        refs = select_parent_refs(spec.query, rows)
    else:
        raise ValueError("query SelectionSpec requires query")

    all_count = len(refs)
    if limit is not None and limit >= 0:
        refs = refs[:limit]
    return SelectionPreview(
        selection_spec=spec,
        match_count=all_count,
        parent_refs=refs,
        truncated=len(refs) < all_count,
    )


def refresh_selection_spec(
    spec: SelectionSpec,
    rows: Sequence[ManifestIndexRow | Mapping[str, Any]],
    *,
    failed_parent_ids: Iterable[str] = (),
    stale_parent_ids: Iterable[str] = (),
) -> SelectionRefreshDiff:
    """Compare a frozen selection snapshot with current query matches."""

    if spec.mode != "frozen" or spec.query is None:
        raise ValueError("refresh requires a frozen SelectionSpec with query")
    current_refs = select_parent_refs(spec.query, rows)
    frozen_refs = list(spec.frozen_refs)
    frozen_by_key = {_parent_ref_key(ref): ref for ref in frozen_refs}
    current_by_key = {_parent_ref_key(ref): ref for ref in current_refs}
    new_refs = [ref for key, ref in current_by_key.items() if key not in frozen_by_key]
    gone_refs = [ref for key, ref in frozen_by_key.items() if key not in current_by_key]
    unchanged_refs = [ref for key, ref in current_by_key.items() if key in frozen_by_key]
    failed_ids = set(failed_parent_ids)
    stale_ids = set(stale_parent_ids)
    failed_refs = [ref for ref in current_refs if ref.id in failed_ids]
    stale_refs = [ref for ref in current_refs if ref.id in stale_ids]
    return SelectionRefreshDiff(
        frozen_refs=frozen_refs,
        current_refs=current_refs,
        new_refs=new_refs,
        gone_refs=gone_refs,
        unchanged_refs=unchanged_refs,
        reprocess_counts={
            "missing": len(new_refs),
            "missing_failed": len(_unique_parent_refs([*new_refs, *failed_refs])),
            "all": len(current_refs),
            "stale": len(stale_refs),
        },
    )


def manifest_index_rows_from_records(
    records: Iterable[Mapping[str, Any]],
) -> list[ManifestIndexRow]:
    """Coerce raw ``feedbax.persistence.manifest_index`` records into typed rows."""

    rows: list[ManifestIndexRow] = []
    for record in records:
        payload = record.get("payload")
        if not isinstance(payload, dict):
            raw_payload = record.get("payload_json")
            payload = json.loads(raw_payload) if isinstance(raw_payload, str) else {}
        rows.append(
            ManifestIndexRow(
                id=str(record.get("id") or payload.get("id")),
                kind=str(record.get("kind") or payload.get("kind")),
                schema_version=str(record.get("schema_version") or payload.get("schema_version")),
                created_at=str(record.get("created_at") or payload.get("created_at") or ""),
                status=(
                    str(record["status"])
                    if record.get("status") is not None
                    else str(payload["status"])
                    if payload.get("status") is not None
                    else None
                ),
                path=str(record["path"]) if record.get("path") is not None else None,
                payload=payload,
            )
        )
    return rows


def select_parent_refs(
    predicate: ManifestPredicate,
    rows: Sequence[ManifestIndexRow | Mapping[str, Any]],
) -> list[ParentRef]:
    """Return deterministic ``ParentRef`` matches for a manifest predicate."""

    matched = [row for row in _coerce_rows(rows) if _matches_without_top_k(predicate, row)]
    if predicate.top_k_by_metric_per_group is not None:
        matched = _apply_top_k(predicate.top_k_by_metric_per_group, matched)
    return [_parent_ref_for_row(row) for row in matched]


def predicate_matches_row(
    predicate: ManifestPredicate,
    row: ManifestIndexRow | Mapping[str, Any],
) -> bool:
    """Return whether one row satisfies all predicate terms, except group top-k."""

    return _matches_without_top_k(predicate, _coerce_row(row))


def _coerce_rows(rows: Sequence[ManifestIndexRow | Mapping[str, Any]]) -> list[ManifestIndexRow]:
    return [_coerce_row(row) for row in rows]


def _coerce_row(row: ManifestIndexRow | Mapping[str, Any]) -> ManifestIndexRow:
    if isinstance(row, ManifestIndexRow):
        return row
    return manifest_index_rows_from_records([row])[0]


def _matches_without_top_k(predicate: ManifestPredicate, row: ManifestIndexRow) -> bool:
    if row.kind != predicate.manifest_kind:
        return False
    if predicate.run_ids and row.id not in set(predicate.run_ids):
        return False
    status = row.status or row.payload.get("status")
    if predicate.statuses and status not in set(predicate.statuses):
        return False
    if predicate.source_set_ids and _source_set_id(row) not in set(predicate.source_set_ids):
        return False
    if predicate.has_checkpoint is not None and _has_checkpoint(row) != predicate.has_checkpoint:
        return False
    if predicate.tags and not set(predicate.tags).issubset(_tags(row)):
        return False
    if not _equals_all(row.payload.get("metadata", {}), predicate.metadata_equals):
        return False
    if not _equals_all(_params_payload(row), predicate.params_equals):
        return False
    if not _equals_all(row.payload, predicate.path_equals):
        return False
    if predicate.expression is not None and not _expression_matches(predicate.expression, row):
        return False
    return True


def _apply_top_k(top_k: TopKByMetricPerGroup, rows: Sequence[ManifestIndexRow]) -> list[ManifestIndexRow]:
    groups: dict[Any, list[_ScoredRow]] = {}
    for row in rows:
        metric = _number_at_path(row.payload, top_k.metric_path)
        if metric is None:
            continue
        group = _value_at_path(row.payload, top_k.group_by_path)
        groups.setdefault(_hashable_group(group), []).append(
            _ScoredRow(row=row, metric=metric, group=group)
        )

    selected: list[ManifestIndexRow] = []
    reverse = top_k.order == "desc"
    for key in sorted(groups, key=lambda value: repr(value)):
        scored = sorted(
            groups[key],
            key=lambda item: (
                item.metric if not reverse else -item.metric,
                item.row.created_at,
                item.row.id,
            ),
        )
        selected.extend(item.row for item in scored[: top_k.k])
    return sorted(selected, key=lambda row: (row.created_at, row.id))


def _expression_matches(expression: Mapping[str, Any], row: ManifestIndexRow) -> bool:
    expr = _EXPR_ADAPTER.validate_python(expression)
    ctx = ExpressionContext(
        items={
            "manifest": ContextItem(kind=row.kind, payload=row.payload),
            "row": ContextItem(kind="ManifestIndexRow", payload=row.model_dump(mode="json")),
            "metadata": ContextItem(kind="dict", payload=row.payload.get("metadata", {})),
            "params": ContextItem(kind="dict", payload=_params_payload(row)),
        }
    )
    try:
        return bool(evaluate_expr(expr, ctx))
    except ExpressionEvaluationError:
        return False


def _params_payload(row: ManifestIndexRow) -> dict[str, Any]:
    payload = row.payload
    for key in ("training_spec", "evaluation_spec", "analysis_spec", "report_spec"):
        value = payload.get(key)
        if isinstance(value, dict) and isinstance(value.get("inline"), dict):
            inline = value["inline"]
            params = inline.get("params")
            return dict(params) if isinstance(params, dict) else dict(inline)
    return {}


def _equals_all(actual: Any, expected: Mapping[str, Any]) -> bool:
    for key, expected_value in expected.items():
        try:
            actual_value = _value_at_path(actual, key)
        except (KeyError, TypeError):
            return False
        if actual_value != expected_value:
            return False
    return True


def _value_at_path(payload: Any, path: str) -> Any:
    current = payload
    if not path:
        return current
    for part in path.split("."):
        if isinstance(current, Mapping):
            current = current[part]
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            current = current[int(part)]
        else:
            raise TypeError(f"cannot traverse {part!r} in {type(current).__name__}")
    return current


def _number_at_path(payload: Mapping[str, Any], path: str) -> float | None:
    try:
        value = _value_at_path(payload, path)
    except (KeyError, TypeError, ValueError):
        return None
    return float(value) if isinstance(value, (int, float)) else None


def _source_set_id(row: ManifestIndexRow) -> str | None:
    for path in (
        "run_set_id",
        "metadata.source_set_id",
        "metadata.run_set_id",
        "metadata.studio.source_set_id",
        "metadata.studio.run_set_id",
    ):
        try:
            value = _value_at_path(row.payload, path)
        except (KeyError, TypeError):
            continue
        if isinstance(value, str):
            return value
    return None


def _has_checkpoint(row: ManifestIndexRow) -> bool:
    custody = row.payload.get("checkpoint_custody")
    if isinstance(custody, list) and len(custody) > 0:
        return True
    artifacts = row.payload.get("artifacts")
    if not isinstance(artifacts, list):
        return False
    return any(
        isinstance(artifact, dict)
        and artifact.get("role") in {"training_checkpoint", "checkpoint", "model_checkpoint"}
        for artifact in artifacts
    )


def _tags(row: ManifestIndexRow) -> set[str]:
    tags: set[str] = set()
    for value in (row.payload.get("tags"), row.payload.get("metadata", {}).get("tags")):
        if isinstance(value, list):
            tags.update(item for item in value if isinstance(item, str))
    return tags


def _parent_ref_for_row(row: ManifestIndexRow) -> ParentRef:
    return ParentRef(
        kind=row.kind,
        id=row.id,
        role=_role_for_kind(row.kind),
        uri=row.path,
        metadata={"status": row.status} if row.status is not None else {},
    )


def _role_for_kind(kind: str) -> str:
    return {
        "TrainingRunManifest": "training_run",
        "EvaluationRunManifest": "evaluation_run",
        "AnalysisRunManifest": "analysis_run",
        "ReportManifest": "report",
    }.get(kind, "manifest")


def _parent_ref_key(ref: ParentRef) -> tuple[str, str, str | None]:
    return (ref.kind, ref.id, ref.role)


def _unique_parent_refs(refs: Sequence[ParentRef]) -> list[ParentRef]:
    by_key: dict[tuple[str, str, str | None], ParentRef] = {}
    for ref in refs:
        by_key.setdefault(_parent_ref_key(ref), ref)
    return list(by_key.values())


def _hashable_group(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _legacy_selection_ids(payload: Mapping[str, Any]) -> list[str] | None:
    for key in (
        "ids",
        "training_run_ids",
        "eval_run_ids",
        "evaluation_run_ids",
        "analysis_run_ids",
        "report_ids",
    ):
        value = payload.get(key)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return list(value)
    return None


def _legacy_manifest_kind(payload: Mapping[str, Any]) -> str:
    explicit = payload.get("manifest_kind")
    if isinstance(explicit, str):
        return explicit
    if "eval_run_ids" in payload or "evaluation_run_ids" in payload:
        return "EvaluationRunManifest"
    if "analysis_run_ids" in payload:
        return "AnalysisRunManifest"
    if "report_ids" in payload:
        return "ReportManifest"
    return "TrainingRunManifest"
