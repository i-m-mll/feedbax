from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from feedbax.contracts.manifest import ParentRef
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.contracts.selection import (
    SELECTION_SPEC_SCHEMA_VERSION,
    SELECTION_SPEC_SCHEMA_VERSION_V1,
    ManifestPredicate,
    SelectionSpec,
    TopKByMetricPerGroup,
    freeze_selection_spec,
    manifest_index_rows_from_records,
    migrate_selection_spec_payload,
    refresh_selection_spec,
    select_parent_refs,
)


def _record(
    id_: str,
    *,
    kind: str = "TrainingRunManifest",
    source_set: str = "sweep-a",
    status: str = "completed",
    loss: float = 1.0,
    group: str = "baseline",
    include_group: bool = True,
    checkpoint: bool = True,
    tags: list[str] | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {"tags": tags or []}
    if include_group:
        metadata["studio"] = {"axis_coordinates": {"shape": group}}
    payload = {
        "id": id_,
        "kind": kind,
        "schema_version": "feedbax.manifest.v1",
        "status": status,
        "run_set_id": source_set,
        "checkpoint_custody": [{"kind": "Checkpoint", "id": f"{id_}:ckpt"}]
        if checkpoint
        else [],
        "summary_metrics": {"loss": loss},
        "metadata": metadata,
    }
    return {
        "id": id_,
        "kind": kind,
        "schema_version": "feedbax.manifest.v1",
        "created_at": f"2026-07-0{len(id_)}T00:00:00+00:00",
        "status": status,
        "path": f"/tmp/{id_}.json",
        "payload_json": json.dumps(payload),
    }


def test_legacy_id_list_migrates_to_explicit_selection_spec_v2() -> None:
    migrated = migrate_selection_spec_payload(
        {"schema_version": SELECTION_SPEC_SCHEMA_VERSION_V1, "training_run_ids": ["a", "b"]}
    )

    assert migrated == {
        "schema_id": "feedbax.spec.selection",
        "schema_version": SELECTION_SPEC_SCHEMA_VERSION,
        "mode": "explicit",
        "manifest_kind": "TrainingRunManifest",
        "ids": ["a", "b"],
        "metadata": {},
    }
    assert SelectionSpec.model_validate(migrated).ids == ["a", "b"]


def test_selection_spec_registry_migrates_v1_and_rejects_unknown_versions() -> None:
    result = default_spec_registry.migrate(
        "SelectionSpec",
        {"schema_version": SELECTION_SPEC_SCHEMA_VERSION_V1, "eval_run_ids": ["eval-a"]},
    )

    assert result.migrated
    assert result.payload["schema_version"] == SELECTION_SPEC_SCHEMA_VERSION
    assert result.payload["manifest_kind"] == "EvaluationRunManifest"
    assert result.payload["ids"] == ["eval-a"]

    with pytest.raises(UnsupportedSpecVersion):
        default_spec_registry.migrate(
            "SelectionSpec",
            {"schema_version": "feedbax.spec.selection.v0", "training_run_ids": ["a"]},
        )


def test_manifest_predicate_filters_rows_and_applies_top_k_per_group() -> None:
    rows = manifest_index_rows_from_records(
        [
            _record("run-a", loss=0.4, group="short", tags=["keep"]),
            _record("run-b", loss=0.2, group="short", tags=["keep"]),
            _record("run-c", loss=0.3, group="long", tags=["keep"]),
            _record("run-d", loss=0.1, group="long", checkpoint=False, tags=["keep"]),
            _record("run-e", source_set="other", loss=0.05, group="short", tags=["keep"]),
        ]
    )
    predicate = ManifestPredicate(
        source_set_ids=["sweep-a"],
        statuses=["completed"],
        has_checkpoint=True,
        tags=["keep"],
        top_k_by_metric_per_group=TopKByMetricPerGroup(
            metric_path="summary_metrics.loss",
            group_by_path="metadata.studio.axis_coordinates.shape",
            k=1,
            order="asc",
        ),
    )

    refs = select_parent_refs(predicate, rows)

    assert [ref.id for ref in refs] == ["run-b", "run-c"]


def test_top_k_skips_rows_missing_group_by_path() -> None:
    rows = manifest_index_rows_from_records(
        [
            _record("run-a", loss=0.1, include_group=False, tags=["keep"]),
            _record("run-b", loss=0.2, group="short", tags=["keep"]),
            _record("run-c", loss=0.3, group="long", tags=["keep"]),
        ]
    )
    predicate = ManifestPredicate(
        tags=["keep"],
        top_k_by_metric_per_group=TopKByMetricPerGroup(
            metric_path="summary_metrics.loss",
            group_by_path="metadata.studio.axis_coordinates.shape",
            k=1,
            order="asc",
        ),
    )

    refs = select_parent_refs(predicate, rows)

    assert [ref.id for ref in refs] == ["run-b", "run-c"]


def test_freeze_and_refresh_report_new_gone_and_reprocess_counts() -> None:
    initial_rows = manifest_index_rows_from_records([_record("run-a"), _record("run-b")])
    query = ManifestPredicate(source_set_ids=["sweep-a"], statuses=["completed"])
    frozen = freeze_selection_spec(SelectionSpec(mode="query", query=query), initial_rows)
    current_rows = manifest_index_rows_from_records([_record("run-b"), _record("run-c")])

    diff = refresh_selection_spec(
        frozen,
        current_rows,
        failed_parent_ids=["run-b"],
        stale_parent_ids=["run-c"],
    )

    assert [ref.id for ref in frozen.frozen_refs] == ["run-a", "run-b"]
    assert [ref.id for ref in diff.current_refs] == ["run-b", "run-c"]
    assert [ref.id for ref in diff.new_refs] == ["run-c"]
    assert [ref.id for ref in diff.gone_refs] == ["run-a"]
    assert diff.reprocess_counts == {
        "missing": 1,
        "missing_failed": 2,
        "all": 2,
        "stale": 1,
    }


def test_freeze_query_preserves_query_manifest_kind() -> None:
    rows = manifest_index_rows_from_records(
        [_record("eval-a", kind="EvaluationRunManifest", tags=["keep"])]
    )
    query = ManifestPredicate(manifest_kind="EvaluationRunManifest", tags=["keep"])

    frozen = freeze_selection_spec(SelectionSpec(mode="query", query=query), rows)

    assert frozen.manifest_kind == "EvaluationRunManifest"
    assert frozen.frozen_refs[0].kind == "EvaluationRunManifest"


def test_selection_spec_rejects_non_empty_cross_mode_fields() -> None:
    query = {"manifest_kind": "TrainingRunManifest", "tags": ["keep"]}
    frozen_ref = {"kind": "TrainingRunManifest", "id": "run-a", "role": "training_run"}

    invalid_payloads = [
        (
            {"mode": "explicit", "ids": ["run-a"], "query": query},
            "explicit SelectionSpec must not include query",
        ),
        (
            {"mode": "explicit", "ids": ["run-a"], "frozen_refs": [frozen_ref]},
            "explicit SelectionSpec must not include frozen_refs",
        ),
        (
            {"mode": "query", "ids": ["run-a"], "query": query},
            "query SelectionSpec must not include ids",
        ),
        (
            {"mode": "query", "query": query, "frozen_refs": [frozen_ref]},
            "query SelectionSpec must not include frozen_refs",
        ),
        (
            {"mode": "frozen", "ids": ["run-a"], "query": query, "frozen_refs": [frozen_ref]},
            "frozen SelectionSpec must not include ids",
        ),
    ]

    for payload, message in invalid_payloads:
        with pytest.raises(ValidationError, match=message):
            SelectionSpec.model_validate(payload)


def test_selection_spec_normalizes_empty_cross_mode_fields() -> None:
    query = ManifestPredicate(tags=["keep"])
    frozen_ref = ParentRef(kind="TrainingRunManifest", id="run-a", role="training_run")

    query_spec = SelectionSpec(mode="query", ids=[], query=query, frozen_refs=[])
    frozen_spec = SelectionSpec(mode="frozen", ids=[], query=query, frozen_refs=[frozen_ref])
    explicit_spec = SelectionSpec(mode="explicit", ids=["run-a"], query=None, frozen_refs=[])

    assert query_spec.ids == []
    assert query_spec.frozen_refs == []
    assert frozen_spec.frozen_refs == [frozen_ref]
    assert explicit_spec.ids == ["run-a"]
