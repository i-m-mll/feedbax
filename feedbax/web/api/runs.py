"""API router for training and evaluation run discovery."""

from __future__ import annotations

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from feedbax.contracts.manifest import (
    EvaluationRunManifest,
    TrainingRunManifest,
    default_manifest_root,
    load_manifest,
    utc_now,
    write_manifest,
)
from feedbax.persistence.database import (
    EvaluationRecord,
    ModelRecord,
    db_session,
)
from feedbax.persistence.manifest_index import (
    get_indexed_manifest_record,
    iter_indexed_manifest_records_by_kind,
    remove_manifest_from_index,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


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
    for legacy in _legacy_training_runs_from_model_db():
        by_id.setdefault(legacy.id, legacy)
    return list(by_id.values())


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
                EvaluationRecord.model_hashes.cast(str).contains(quoted_hash)
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
    updated = manifest.model_copy(
        update={
            "metadata": {
                **manifest.metadata,
                "superseded_at": utc_now().isoformat(),
                "superseded_by": payload.superseded_by,
                "superseded_reason": payload.reason,
            }
        }
    )
    write_manifest(updated, root=default_manifest_root())
    row = get_indexed_manifest_record(training_run_id)
    if row is None:
        raise HTTPException(status_code=500, detail="Superseded manifest was not re-indexed")
    return _training_summary_from_index_row(row)


@router.post("/evaluation", response_model=EvalRunInfo)
async def create_eval_run(payload: CreateEvalRunRequest) -> EvalRunInfo:
    """Create a new evaluation run entry.

    This endpoint registers the intent to run an evaluation with the
    given parameters.  The actual evaluation computation is triggered
    separately (via ``POST /api/analyses/jobs``).

    A new ``EvaluationRecord`` is created before the endpoint reports success.
    """
    import hashlib
    import json

    run_id = hashlib.sha256(
        json.dumps(
            {
                "training_run_id": payload.training_run_id,
                "name": payload.name,
                "eval_params": payload.eval_params,
                "ts": datetime.utcnow().isoformat(),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]

    now = datetime.utcnow()
    description = _summarize_perturbation_config(payload.eval_params)

    try:
        with db_session() as session:
            record = EvaluationRecord(
                hash=run_id,
                expt_name=payload.name,
                model_hashes=[payload.training_run_id],
                perturbation_config=payload.eval_params,
                created_at=now,
            )
            session.add(record)
        logger.info(
            "Created evaluation run %s for training run %s",
            run_id,
            payload.training_run_id,
        )
    except Exception as exc:
        logger.exception("Could not persist eval run %s to DB", run_id)
        raise HTTPException(
            status_code=500,
            detail=f"Could not persist evaluation run {run_id!r}",
        ) from exc

    return EvalRunInfo(
        id=run_id,
        training_run_id=payload.training_run_id,
        name=payload.name,
        created_at=now.isoformat(),
        status="running",
        description=description,
    )
