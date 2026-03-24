"""API router for training and evaluation run discovery."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from feedbax.database import (
    EvaluationRecord,
    ModelRecord,
    db_session,
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


class EvalRunInfo(BaseModel):
    """Summary metadata for an evaluation run."""

    id: str
    training_run_id: str
    name: str
    created_at: str  # ISO 8601
    status: str
    description: Optional[str] = None


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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/training")
async def list_training_runs() -> list[TrainingRunInfo]:
    """List all training runs.

    Each distinct ``expt_name`` in the model database represents a training
    experiment.  Within each experiment, models are grouped by their hash
    and the earliest creation timestamp is used as the run date.

    All records returned from the database are post-training, so the
    status is always ``completed``.

    Uses a window function to select one representative record per
    (expt_name, hash) group in a single query, avoiding N+1 per-row
    lookups.
    """
    from sqlalchemy import func, over

    with db_session(autocommit=False) as session:
        # Use ROW_NUMBER to pick one representative record per group while
        # also computing the earliest created_at.  This collapses the
        # previous two-query pattern (grouped aggregation + per-row fetch)
        # into a single pass.
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

        results: list[TrainingRunInfo] = []
        for record, earliest_ts in rows:
            results.append(
                TrainingRunInfo(
                    id=record.hash,
                    name=record.expt_name or record.hash[:12],
                    created_at=earliest_ts.isoformat() if earliest_ts else "",
                    status="completed",
                    hyperparams=_extract_hyperparams(record),
                )
            )

    return results


@router.get("/training/{training_run_id}/evals")
async def list_eval_runs(training_run_id: str) -> list[EvalRunInfo]:
    """List evaluation runs associated with a training run.

    The ``training_run_id`` is a model hash.  Evaluations whose
    ``model_hashes`` JSON array contains this hash are returned.
    """
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
