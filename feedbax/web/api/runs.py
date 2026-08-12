"""API router for training and evaluation run discovery."""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import String

from feedbax.contracts.manifest import (
    BaseManifest,
    EntrypointRef,
    EvaluationRunManifest,
    EvaluationRunSpec,
    ParentRef,
    Provenance,
    TrainingRunManifest,
    default_manifest_root,
    evaluation_run_manifest_id,
    load_manifest,
    spec_payload,
    utc_now,
    write_manifest,
)
from feedbax.contracts.manifest_packet import (
    ManifestPacketConflictError,
    ManifestPacketValidationError,
    import_manifest_packet,
)
from feedbax.contracts.selection import (
    SelectionPreview,
    SelectionRefreshDiff,
    SelectionSpec,
    manifest_index_rows_from_records,
    preview_selection_spec,
    refresh_selection_spec,
)
from feedbax.persistence.database import (
    EvaluationRecord,
    ModelRecord,
    db_session,
)
from feedbax.persistence.manifest_index import (
    get_indexed_manifest_record,
    iter_manifest_files,
    iter_indexed_manifest_records_by_kind,
    remove_manifest_from_index,
    rebuild_manifest_index,
)
from feedbax.training.checkpoint_custody import (
    LEGACY_CHECKPOINT_ADOPTION_DOCS,
    LEGACY_CHECKPOINT_ADOPTION_ENTRYPOINT,
    detect_known_legacy_checkpoint_layout,
)
from feedbax.web.services.training_service import training_service

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class LegacyCheckpointInfo(BaseModel):
    """Detected pre-custody checkpoint root that needs adoption."""

    layout_id: str
    layout_name: str
    message: str
    docs: str = LEGACY_CHECKPOINT_ADOPTION_DOCS
    adoption_entrypoint: str = LEGACY_CHECKPOINT_ADOPTION_ENTRYPOINT


class TrainingRunInfo(BaseModel):
    """Summary metadata for a training run.

    Wire format uses snake_case; the frontend converts to camelCase.
    """

    id: str
    name: str
    created_at: str  # ISO 8601
    status: str
    hyperparams: dict[str, Any]
    metrics: dict[str, Any] = Field(default_factory=dict)
    uri: Optional[str] = None
    stage_id: Optional[str] = None
    scenario_id: Optional[str] = None
    planned: bool = False
    checkpoint_available: bool = False
    source_issue: Optional[str] = None
    provenance_id: Optional[str] = None
    superseded_by: Optional[str] = None
    legacy_checkpoint: Optional[LegacyCheckpointInfo] = None


class EvalRunInfo(BaseModel):
    """Summary metadata for an evaluation run."""

    id: str
    training_run_id: str
    name: str
    created_at: str  # ISO 8601
    status: str
    description: Optional[str] = None
    training_run_ids: list[str] = Field(default_factory=list)
    uri: Optional[str] = None


class CreateEvalRunRequest(BaseModel):
    """Body for ``POST /evaluation``."""

    training_run_id: str
    name: str
    eval_params: dict[str, Any] = {}


class SupersedeTrainingRunRequest(BaseModel):
    """Body for marking a completed training run as superseded."""

    superseded_by: Optional[str] = None
    reason: Optional[str] = None


class SelectionPreviewRequest(BaseModel):
    """Body for previewing a SelectionSpec over indexed manifests."""

    selection_spec: SelectionSpec
    limit: int = Field(default=50, ge=0)


class SelectionRefreshRequest(BaseModel):
    """Body for comparing a frozen SelectionSpec with current index matches."""

    selection_spec: SelectionSpec
    failed_parent_ids: list[str] = Field(default_factory=list)
    stale_parent_ids: list[str] = Field(default_factory=list)


class TrainingRunCompareRequest(BaseModel):
    """Bounded field request for comparing selected training runs."""

    run_ids: list[str] = Field(min_length=2)
    param_fields: list[str] = Field(default_factory=list, max_length=64)
    metric_fields: list[str] = Field(default_factory=list, max_length=64)


class TrainingRunCompareRow(BaseModel):
    """Selected parameter and metric values for one compared training run."""

    id: str
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)


class TrainingRunCompareResponse(BaseModel):
    """Comparison response containing only requested fields."""

    rows: list[TrainingRunCompareRow]


class ManifestImportRequest(BaseModel):
    """Import manifests from a packet or a local runs directory."""

    path: str
    root: Optional[str] = None


class ManifestImportResponse(BaseModel):
    """Summary returned after importing manifests into the active root."""

    root: str
    source_path: str
    imported_manifest_ids: list[str] = Field(default_factory=list)
    skipped_manifest_ids: list[str] = Field(default_factory=list)
    manifest_count: int = 0
    artifact_count: int = 0
    included_artifact_count: int = 0
    external_artifact_count: int = 0
    index_path: Optional[str] = None
    training_runs: list[TrainingRunInfo] = Field(default_factory=list)
    eval_runs: list[EvalRunInfo] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_hyperparams(record: ModelRecord) -> dict[str, Any]:
    """Extract key hyperparameters from a ModelRecord for display.

    Pulls explicitly-defined parameter columns that are useful for
    at-a-glance differentiation of training runs.
    """
    params: dict[str, Any] = {}

    for attr in ("model__n_replicates", "n_batches", "pert__type", "pert__std"):
        try:
            value = getattr(record, attr, None)
            if value is not None:
                # Use a shorter display key
                display_key = attr.replace("model__", "").replace("pert__", "pert_")
                params[display_key] = value
        except Exception:
            continue

    return params


def _summarize_perturbation_config(config: Optional[dict[str, Any]]) -> Optional[str]:
    """Build a short human-readable description from a perturbation config."""
    if not config:
        return None

    parts: list[str] = []
    pert_type = config.get("type") or config.get("pert_type")
    if pert_type:
        parts.append(str(pert_type))
    pert_std = config.get("std") or config.get("pert_std")
    if pert_std is not None:
        parts.append(f"std={pert_std}")

    return ", ".join(parts) if parts else None


def _payload_inline(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if isinstance(value, dict) and isinstance(value.get("inline"), dict):
        return value["inline"]
    return {}


def _path_value(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _selected_values(source: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for field in fields:
        value = source.get(field)
        if value is None and "." in field:
            value = _path_value(source, field)
        selected[field] = value
    return selected


def _studio_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    studio = metadata.get("studio")
    return studio if isinstance(studio, dict) else {}


def _provenance(payload: dict[str, Any]) -> dict[str, Any]:
    provenance = payload.get("provenance")
    return provenance if isinstance(provenance, dict) else {}


def _training_hyperparams(payload: dict[str, Any]) -> dict[str, Any]:
    training = _payload_inline(payload, "training_spec")
    params: dict[str, Any] = {}
    for key in ("n_batches", "batch_size", "n_warmup_batches", "seed"):
        if key in training:
            params[key] = training[key]
    optimizer = training.get("optimizer")
    if isinstance(optimizer, dict):
        optimizer_params = optimizer.get("params")
        if isinstance(optimizer_params, dict) and "learning_rate" in optimizer_params:
            params["learning_rate"] = optimizer_params["learning_rate"]
    axis_coordinates = _studio_metadata(payload).get("axis_coordinates")
    if isinstance(axis_coordinates, dict):
        params.update(
            {
                f"axis_{key}": value
                for key, value in axis_coordinates.items()
                if isinstance(value, (str, int, float, bool))
            }
        )
    return params


def _source_issue(payload: dict[str, Any]) -> str | None:
    issues = _provenance(payload).get("issues")
    if isinstance(issues, list):
        return next((item for item in issues if isinstance(item, str)), None)
    return None


def _training_name(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    studio = _studio_metadata(payload)
    for value in (
        metadata.get("name") if isinstance(metadata, dict) else None,
        metadata.get("label") if isinstance(metadata, dict) else None,
        studio.get("label"),
        studio.get("planned_training_run_id"),
        payload.get("job_id"),
        payload.get("id"),
    ):
        if isinstance(value, str) and value:
            return value
    return str(payload.get("id", "training run"))


def _checkpoint_available(payload: dict[str, Any]) -> bool:
    checkpoint_custody = payload.get("checkpoint_custody")
    if isinstance(checkpoint_custody, list) and checkpoint_custody:
        return True
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        return False
    return any(
        isinstance(artifact, dict) and artifact.get("role") == "training_checkpoint"
        for artifact in artifacts
    )


def _local_path_from_uri(uri: Any) -> Path | None:
    if not isinstance(uri, str) or not uri:
        return None
    if uri.startswith("file://"):
        return Path(uri[7:])
    if "://" in uri:
        return None
    return Path(uri)


def _checkpoint_candidate_paths(payload: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    for key in ("checkpoint_custody", "artifacts"):
        records = payload.get(key)
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            if key == "artifacts" and record.get("role") != "training_checkpoint":
                continue
            path = _local_path_from_uri(record.get("uri") or record.get("path"))
            if path is not None:
                candidates.append(path)
    return candidates


def _legacy_checkpoint_info(payload: dict[str, Any]) -> LegacyCheckpointInfo | None:
    for path in _checkpoint_candidate_paths(payload):
        layout = detect_known_legacy_checkpoint_layout(path)
        if layout is None:
            continue
        return LegacyCheckpointInfo(
            layout_id=layout.layout_id,
            layout_name=layout.name,
            message=(
                "Checkpoint predates checkpoint custody; adoption required before "
                f"Studio can load it. See {LEGACY_CHECKPOINT_ADOPTION_DOCS}."
            ),
        )
    return None


def _training_summary_from_index_row(row: dict[str, Any]) -> TrainingRunInfo:
    payload = json.loads(row["payload_json"])
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    studio = _studio_metadata(payload)
    metrics = payload.get("summary_metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    return TrainingRunInfo(
        id=str(payload["id"]),
        name=_training_name(payload),
        created_at=str(payload.get("created_at") or row["created_at"]),
        status=str(payload.get("status") or row["status"] or "unknown"),
        hyperparams=_training_hyperparams(payload),
        metrics=metrics,
        uri=str(row["path"]),
        stage_id=studio.get("stage_id") if isinstance(studio.get("stage_id"), str) else None,
        scenario_id=studio.get("scenario_id")
        if isinstance(studio.get("scenario_id"), str)
        else None,
        planned=bool(metadata.get("planned")) if isinstance(metadata, dict) else False,
        checkpoint_available=_checkpoint_available(payload),
        source_issue=_source_issue(payload),
        provenance_id=str(payload["id"]),
        superseded_by=metadata.get("superseded_by")
        if isinstance(metadata, dict) and isinstance(metadata.get("superseded_by"), str)
        else None,
        legacy_checkpoint=_legacy_checkpoint_info(payload),
    )


def _compare_row_from_index_row(
    row: dict[str, Any],
    *,
    param_fields: list[str],
    metric_fields: list[str],
) -> TrainingRunCompareRow:
    payload = json.loads(row["payload_json"])
    studio = _studio_metadata(payload)
    hyperparams = _training_hyperparams(payload)
    axis_coordinates = studio.get("axis_coordinates")
    axis_coordinates = axis_coordinates if isinstance(axis_coordinates, dict) else {}
    params = {
        **hyperparams,
        **{
            str(key): value
            for key, value in axis_coordinates.items()
        },
    }
    metrics = payload.get("summary_metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    return TrainingRunCompareRow(
        id=str(payload["id"]),
        params=_selected_values(params, param_fields),
        metrics=_selected_values(metrics, metric_fields),
    )


def _eval_summary_from_index_row(row: dict[str, Any]) -> EvalRunInfo:
    payload = json.loads(row["payload_json"])
    spec = _payload_inline(payload, "evaluation_spec")
    input_runs = [
        ref.get("id")
        for ref in payload.get("input_training_runs", [])
        if isinstance(ref, dict) and isinstance(ref.get("id"), str)
    ]
    training_run_id = input_runs[0] if input_runs else ""
    params = spec.get("params") if isinstance(spec.get("params"), dict) else {}
    description = _summarize_perturbation_config(params) or str(spec.get("evaluation_type", ""))
    return EvalRunInfo(
        id=str(payload["id"]),
        training_run_id=training_run_id,
        training_run_ids=input_runs,
        name=str(params.get("label") or payload.get("job_id") or payload["id"]),
        created_at=str(payload.get("created_at") or row["created_at"]),
        status=str(payload.get("status") or row["status"] or "unknown"),
        description=description or None,
        uri=str(row["path"]),
    )


def _load_training_manifest_from_index(training_run_id: str) -> tuple[TrainingRunManifest, Path]:
    row = get_indexed_manifest_record(training_run_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Training run {training_run_id!r} not found")
    path = Path(row["path"])
    manifest = load_manifest(path)
    if not isinstance(manifest, TrainingRunManifest):
        raise HTTPException(
            status_code=409,
            detail=f"Manifest {training_run_id!r} is {type(manifest).__name__}, not TrainingRunManifest",
        )
    return manifest, path


def _selection_index_rows(spec: SelectionSpec):
    manifest_kind = spec.query.manifest_kind if spec.query is not None else spec.manifest_kind
    return manifest_index_rows_from_records(
        iter_indexed_manifest_records_by_kind(manifest_kind)
    )


def _load_evaluation_manifest_from_index(eval_run_id: str) -> tuple[EvaluationRunManifest, Path]:
    row = get_indexed_manifest_record(eval_run_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Evaluation run {eval_run_id!r} not found")
    path = Path(row["path"])
    manifest = load_manifest(path)
    if not isinstance(manifest, EvaluationRunManifest):
        raise HTTPException(
            status_code=409,
            detail=f"Manifest {eval_run_id!r} is {type(manifest).__name__}, not EvaluationRunManifest",
        )
    return manifest, path


def _mark_dependent_evaluations_stale(
    training_run_id: str,
    *,
    superseded_by: str | None,
    reason: str | None,
    superseded_at: str,
) -> int:
    """Mark indexed evaluations derived from a superseded training run stale."""

    updated_count = 0
    for row in iter_indexed_manifest_records_by_kind("EvaluationRunManifest"):
        manifest = load_manifest(row["path"])
        if not isinstance(manifest, EvaluationRunManifest):
            continue
        if not any(parent.id == training_run_id for parent in manifest.input_training_runs):
            continue

        metadata = dict(manifest.metadata)
        raw_stale_parent_ids = metadata.get("stale_parent_ids")
        stale_parent_ids = {
            parent_id
            for parent_id in (
                raw_stale_parent_ids if isinstance(raw_stale_parent_ids, list) else []
            )
            if isinstance(parent_id, str)
        }
        raw_supersessions = metadata.get("upstream_supersessions")
        upstream_supersessions = (
            dict(raw_supersessions) if isinstance(raw_supersessions, dict) else {}
        )
        supersession = {
            "superseded_at": superseded_at,
            "superseded_by": superseded_by,
            "reason": reason,
        }
        if (
            manifest.status == "stale"
            and training_run_id in stale_parent_ids
            and upstream_supersessions.get(training_run_id) == supersession
        ):
            continue

        stale_parent_ids.add(training_run_id)
        upstream_supersessions[training_run_id] = supersession
        metadata.update(
            {
                "staleness_reason": "upstream superseded",
                "stale_parent_ids": sorted(stale_parent_ids),
                "upstream_supersessions": upstream_supersessions,
            }
        )
        metadata.setdefault("stale_at", superseded_at)
        metadata.setdefault("stale_from_status", manifest.status)
        write_manifest(
            manifest.model_copy(update={"status": "stale", "metadata": metadata}),
            root=default_manifest_root(),
        )
        updated_count += 1
    return updated_count


def _summaries_for_manifest_ids(
    manifest_ids: set[str],
    *,
    root: Path | str | None = None,
) -> tuple[list[TrainingRunInfo], list[EvalRunInfo]]:
    training_runs = [
        _training_summary_from_index_row(row)
        for row in iter_indexed_manifest_records_by_kind("TrainingRunManifest", root=root)
        if row["id"] in manifest_ids
    ]
    eval_runs = [
        _eval_summary_from_index_row(row)
        for row in iter_indexed_manifest_records_by_kind("EvaluationRunManifest", root=root)
        if row["id"] in manifest_ids
    ]
    return training_runs, eval_runs


def _resolved_import_paths(request: ManifestImportRequest) -> tuple[Path, Path]:
    source = Path(request.path).expanduser()
    root = Path(request.root).expanduser() if request.root else default_manifest_root()
    return source, root


def _manifest_with_import_metadata(
    manifest: BaseManifest,
    *,
    source_root: Path,
    source_path: Path,
) -> BaseManifest:
    return manifest.model_copy(
        update={
            "metadata": {
                **manifest.metadata,
                "imported_from": {
                    "source": "runs_dir",
                    "source_root": str(source_root),
                    "source_path": str(source_path),
                    "imported_at": utc_now().isoformat(),
                },
            }
        }
    )


def _import_runs_dir(source_root: Path, target_root: Path) -> ManifestImportResponse:
    if not source_root.exists() or not source_root.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Runs directory does not exist: {source_root}",
        )
    if source_root.resolve() == target_root.resolve():
        index_path = rebuild_manifest_index(target_root)
        all_ids = {
            row["id"]
            for kind in ("TrainingRunManifest", "EvaluationRunManifest")
            for row in iter_indexed_manifest_records_by_kind(kind, root=target_root)
        }
        training_runs, eval_runs = _summaries_for_manifest_ids(all_ids, root=target_root)
        return ManifestImportResponse(
            root=str(target_root),
            source_path=str(source_root),
            imported_manifest_ids=[],
            skipped_manifest_ids=[],
            manifest_count=len(iter_manifest_files(target_root)),
            index_path=str(index_path),
            training_runs=training_runs,
            eval_runs=eval_runs,
        )

    imported_ids: list[str] = []
    skipped_ids: list[str] = []
    manifest_paths = iter_manifest_files(source_root)
    if iter_manifest_files(target_root):
        rebuild_manifest_index(target_root)
    for manifest_path in manifest_paths:
        manifest = load_manifest(manifest_path)
        if get_indexed_manifest_record(manifest.id, root=target_root) is not None:
            skipped_ids.append(manifest.id)
            continue
        imported = _manifest_with_import_metadata(
            manifest,
            source_root=source_root,
            source_path=manifest_path,
        )
        write_manifest(imported, root=target_root)
        imported_ids.append(manifest.id)

    index_path = rebuild_manifest_index(target_root)
    training_runs, eval_runs = _summaries_for_manifest_ids(
        set(imported_ids + skipped_ids),
        root=target_root,
    )
    return ManifestImportResponse(
        root=str(target_root),
        source_path=str(source_root),
        imported_manifest_ids=imported_ids,
        skipped_manifest_ids=skipped_ids,
        manifest_count=len(manifest_paths),
        index_path=str(index_path),
        training_runs=training_runs,
        eval_runs=eval_runs,
    )


def _legacy_training_runs_from_model_db() -> list[TrainingRunInfo]:
    """Return legacy completed rows for model DB records without manifests."""
    from sqlalchemy import func

    with db_session(autocommit=False) as session:
        row_num = (
            func.row_number()
            .over(
                partition_by=(ModelRecord.expt_name, ModelRecord.hash),
                order_by=ModelRecord.created_at.asc(),
            )
            .label("rn")
        )
        earliest = (
            func.min(ModelRecord.created_at)
            .over(partition_by=(ModelRecord.expt_name, ModelRecord.hash))
            .label("earliest")
        )
        subq = (
            session.query(ModelRecord, row_num, earliest)
            .filter(ModelRecord.is_path_defunct == False)  # noqa: E712
            .subquery()
        )
        from sqlalchemy.orm import aliased

        RecordAlias = aliased(ModelRecord, subq)
        rows = (
            session.query(RecordAlias, subq.c.earliest)
            .filter(subq.c.rn == 1)
            .order_by(subq.c.earliest.desc())
            .all()
        )

    return [
        TrainingRunInfo(
            id=record.hash,
            name=record.expt_name or record.hash[:12],
            created_at=earliest_ts.isoformat() if earliest_ts else "",
            status="completed",
            hyperparams=_extract_hyperparams(record),
            metrics={},
            provenance_id=record.hash,
        )
        for record, earliest_ts in rows
    ]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/training")
async def list_training_runs() -> list[TrainingRunInfo]:
    """List training runs from the durable manifest index."""
    indexed = [
        _training_summary_from_index_row(row)
        for row in iter_indexed_manifest_records_by_kind("TrainingRunManifest")
    ]
    by_id = {row.id: row for row in indexed}
    for live in training_service.list_live_training_runs():
        by_id.setdefault(live["id"], TrainingRunInfo.model_validate(live))
    for legacy in _legacy_training_runs_from_model_db():
        by_id.setdefault(legacy.id, legacy)
    return list(by_id.values())


@router.post("/selection/preview", response_model=SelectionPreview)
async def preview_selection(request: SelectionPreviewRequest) -> SelectionPreview:
    """Preview the current manifest-index matches for a selection spec."""

    rows = _selection_index_rows(request.selection_spec)
    return preview_selection_spec(request.selection_spec, rows, limit=request.limit)


@router.post("/selection/refresh", response_model=SelectionRefreshDiff)
async def refresh_selection(request: SelectionRefreshRequest) -> SelectionRefreshDiff:
    """Compare a frozen selection snapshot with current manifest-index matches."""

    rows = _selection_index_rows(request.selection_spec)
    return refresh_selection_spec(
        request.selection_spec,
        rows,
        failed_parent_ids=request.failed_parent_ids,
        stale_parent_ids=request.stale_parent_ids,
    )


@router.post("/training/compare", response_model=TrainingRunCompareResponse)
async def compare_training_runs(
    request: TrainingRunCompareRequest,
) -> TrainingRunCompareResponse:
    """Return only requested fields for selected training-run comparison."""

    rows: list[TrainingRunCompareRow] = []
    missing: list[str] = []
    for run_id in request.run_ids:
        row = get_indexed_manifest_record(run_id)
        if row is None:
            missing.append(run_id)
            continue
        if row["kind"] != "TrainingRunManifest":
            raise HTTPException(
                status_code=409,
                detail=f"Manifest {run_id!r} is {row['kind']}, not TrainingRunManifest",
            )
        rows.append(
            _compare_row_from_index_row(
                row,
                param_fields=request.param_fields,
                metric_fields=request.metric_fields,
            )
        )
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Training manifests not found: {missing}",
        )
    return TrainingRunCompareResponse(rows=rows)


@router.post("/import/packet", response_model=ManifestImportResponse)
async def import_manifest_packet_route(
    request: ManifestImportRequest,
) -> ManifestImportResponse:
    """Import a manifest packet directory into the active manifest root."""

    source, root = _resolved_import_paths(request)
    try:
        result = import_manifest_packet(source, root=root)
    except (ManifestPacketConflictError, ManifestPacketValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    imported_ids = set(result.imported_manifest_ids + result.skipped_manifest_ids)
    training_runs, eval_runs = _summaries_for_manifest_ids(imported_ids, root=root)
    return ManifestImportResponse(
        root=result.root,
        source_path=str(source),
        imported_manifest_ids=result.imported_manifest_ids,
        skipped_manifest_ids=result.skipped_manifest_ids,
        manifest_count=len(result.imported_manifest_ids) + len(result.skipped_manifest_ids),
        artifact_count=result.artifact_count,
        included_artifact_count=result.included_artifact_count,
        external_artifact_count=result.external_artifact_count,
        index_path=result.index_path,
        training_runs=training_runs,
        eval_runs=eval_runs,
    )


@router.post("/import/runs-dir", response_model=ManifestImportResponse)
async def import_runs_dir_route(
    request: ManifestImportRequest,
) -> ManifestImportResponse:
    """Import or re-index a FEEDBAX_RUNS_DIR-style manifest root."""

    source, root = _resolved_import_paths(request)
    try:
        return _import_runs_dir(source, root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/training/{training_run_id}/evals")
async def list_eval_runs(training_run_id: str) -> list[EvalRunInfo]:
    """List evaluation runs associated with a training run."""
    manifest_rows = [
        _eval_summary_from_index_row(row)
        for row in iter_indexed_manifest_records_by_kind("EvaluationRunManifest")
    ]
    manifest_matches = [
        row for row in manifest_rows if training_run_id in row.training_run_ids
    ]
    if manifest_matches:
        return manifest_matches

    indexed_training_row = get_indexed_manifest_record(training_run_id)
    if indexed_training_row is not None:
        _load_training_manifest_from_index(training_run_id)
        return []

    with db_session(autocommit=False) as session:
        # Verify the training run exists
        model = (
            session.query(ModelRecord)
            .filter(ModelRecord.hash == training_run_id)
            .first()
        )
        if model is None:
            raise HTTPException(
                status_code=404,
                detail=f"Training run '{training_run_id}' not found",
            )

        # EvaluationRecord.model_hashes is a JSON column containing a list
        # of model hash strings.  SQLite stores JSON as text, so we search
        # for the JSON-quoted hash to avoid substring false positives (e.g.
        # hash "abc" matching "abcdef123").
        quoted_hash = f'"{training_run_id}"'
        evals = (
            session.query(EvaluationRecord)
            .filter(EvaluationRecord.archived == False)  # noqa: E712
            .filter(
                EvaluationRecord.model_hashes.cast(String).contains(quoted_hash)
            )
            .order_by(EvaluationRecord.created_at.desc())
            .all()
        )

    results: list[EvalRunInfo] = []
    for ev in evals:
        # Build a descriptive name from available metadata
        name = ev.expt_name or ev.hash[:12]

        # Summarize what this evaluation tested
        description = _summarize_perturbation_config(ev.perturbation_config)
        if not description and ev.task_variants:
            description = f"{len(ev.task_variants)} task variant(s)"

        results.append(
            EvalRunInfo(
                id=ev.hash,
                training_run_id=training_run_id,
                name=name,
                created_at=ev.created_at.isoformat() if ev.created_at else "",
                status="completed",
                description=description,
            )
        )

    return results


@router.get("/training/{training_run_id}/manifest")
async def get_training_run_manifest(training_run_id: str) -> dict[str, Any]:
    """Return the durable training manifest payload for a Studio run row."""

    manifest, _path = _load_training_manifest_from_index(training_run_id)
    return manifest.model_dump(mode="json", exclude_none=True)


@router.get("/evaluation/{eval_run_id}/manifest")
async def get_evaluation_run_manifest(eval_run_id: str) -> dict[str, Any]:
    """Return the durable evaluation manifest payload for a Studio run row."""

    manifest, _path = _load_evaluation_manifest_from_index(eval_run_id)
    return manifest.model_dump(mode="json", exclude_none=True)


@router.post("/training/{training_run_id}/cancel", response_model=TrainingRunInfo)
async def cancel_training_run(training_run_id: str) -> TrainingRunInfo:
    """Mark a pending or running training manifest as cancelled."""
    manifest, _path = _load_training_manifest_from_index(training_run_id)
    if manifest.status not in {"pending", "running"}:
        raise HTTPException(
            status_code=409,
            detail=f"Training run {training_run_id!r} cannot be cancelled from {manifest.status!r}",
        )
    updated = manifest.model_copy(
        update={
            "status": "cancelled",
            "completed_at": utc_now(),
            "metadata": {
                **manifest.metadata,
                "cancelled_at": utc_now().isoformat(),
            },
        }
    )
    write_manifest(updated, root=default_manifest_root())
    row = get_indexed_manifest_record(training_run_id)
    if row is None:
        raise HTTPException(status_code=500, detail="Cancelled manifest was not re-indexed")
    return _training_summary_from_index_row(row)


@router.delete("/training/{training_run_id}", response_model=TrainingRunInfo)
async def delete_training_run(training_run_id: str) -> TrainingRunInfo:
    """Delete a pending training manifest; completed runs must be superseded."""
    manifest, path = _load_training_manifest_from_index(training_run_id)
    summary_row = get_indexed_manifest_record(training_run_id)
    if summary_row is None:
        raise HTTPException(status_code=404, detail=f"Training run {training_run_id!r} not found")
    summary = _training_summary_from_index_row(summary_row)
    if manifest.status != "pending":
        raise HTTPException(
            status_code=409,
            detail="Only pending training manifests may be deleted; supersede completed runs instead.",
        )
    path.unlink(missing_ok=True)
    remove_manifest_from_index(training_run_id)
    return summary


@router.post("/training/{training_run_id}/supersede", response_model=TrainingRunInfo)
async def supersede_training_run(
    training_run_id: str,
    payload: SupersedeTrainingRunRequest,
) -> TrainingRunInfo:
    """Mark a completed training manifest as superseded without deleting it."""
    manifest, _path = _load_training_manifest_from_index(training_run_id)
    if manifest.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Only completed training runs can be superseded; got {manifest.status!r}.",
        )
    if payload.superseded_by == training_run_id:
        raise HTTPException(status_code=409, detail="A training run cannot supersede itself.")

    already_superseded = "superseded_at" in manifest.metadata
    if already_superseded:
        previous_target = manifest.metadata.get("superseded_by")
        previous_reason = manifest.metadata.get("superseded_reason")
        if previous_target != payload.superseded_by or previous_reason != payload.reason:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Training run {training_run_id!r} is already superseded by "
                    f"{previous_target!r}."
                ),
            )
        superseded_at = str(manifest.metadata["superseded_at"])
    else:
        superseded_at = utc_now().isoformat()
        updated = manifest.model_copy(
            update={
                "metadata": {
                    **manifest.metadata,
                    "superseded_at": superseded_at,
                    "superseded_by": payload.superseded_by,
                    "superseded_reason": payload.reason,
                }
            }
        )
        write_manifest(updated, root=default_manifest_root())

    _mark_dependent_evaluations_stale(
        training_run_id,
        superseded_by=payload.superseded_by,
        reason=payload.reason,
        superseded_at=superseded_at,
    )
    row = get_indexed_manifest_record(training_run_id)
    if row is None:
        raise HTTPException(status_code=500, detail="Superseded manifest was not re-indexed")
    return _training_summary_from_index_row(row)


@router.post("/evaluation", response_model=EvalRunInfo)
async def create_eval_run(payload: CreateEvalRunRequest) -> EvalRunInfo:
    """Create a pending evaluation run on the canonical manifest path.

    This endpoint registers the intent to run an evaluation with the
    given parameters.  The actual evaluation computation is triggered
    separately (via ``POST /api/analyses/jobs``).
    """
    _load_training_manifest_from_index(payload.training_run_id)
    training_ref = ParentRef(
        kind="TrainingRunManifest",
        id=payload.training_run_id,
        role="training_run",
    )
    run_spec = EvaluationRunSpec(
        evaluation_type="feedbax.studio.default_eval",
        training_run_ids=[payload.training_run_id],
        inputs=[training_ref],
        params={**payload.eval_params, "label": payload.name},
    )
    run_id = evaluation_run_manifest_id(run_spec)
    existing_row = get_indexed_manifest_record(run_id)
    if existing_row is not None:
        if existing_row["kind"] != "EvaluationRunManifest":
            raise HTTPException(
                status_code=409,
                detail=f"Manifest {run_id!r} is not an EvaluationRunManifest",
            )
        return _eval_summary_from_index_row(existing_row)
    manifest = EvaluationRunManifest(
        id=run_id,
        status="pending",
        evaluation_spec=spec_payload(
            "EvaluationRunSpec",
            run_spec.model_dump(mode="json", exclude_none=True),
        ),
        input_training_runs=[training_ref],
        provenance=Provenance(
            entrypoint=EntrypointRef(
                kind="feedbax-studio-api",
                name="create_eval_run",
            ),
            parents=[training_ref],
        ),
        metadata={"name": payload.name, "planned": True},
    )

    try:
        write_manifest(manifest, root=default_manifest_root())
        row = get_indexed_manifest_record(run_id)
        if row is None:
            raise RuntimeError("evaluation manifest was not indexed")
        logger.info(
            "Created evaluation manifest %s for training run %s",
            run_id,
            payload.training_run_id,
        )
    except Exception as exc:
        logger.exception("Could not persist evaluation manifest %s", run_id)
        raise HTTPException(
            status_code=500,
            detail=f"Could not persist evaluation manifest {run_id!r}",
        ) from exc

    return _eval_summary_from_index_row(row)
