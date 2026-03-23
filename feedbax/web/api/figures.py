"""API router for figure browsing and retrieval."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from feedbax.database import (
    EvaluationRecord,
    FigureRecord,
    db_session,
)

router = APIRouter()

# Maps file extension to MIME type for figure file serving.
_CONTENT_TYPES: dict[str, str] = {
    "json": "application/json",
    "html": "text/html",
    "png": "image/png",
    "svg": "image/svg+xml",
    "webp": "image/webp",
    "pdf": "application/pdf",
}


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class FigureInfo(BaseModel):
    """Summary metadata for a single figure."""

    hash: str
    evaluation_hash: str
    identifier: str
    figure_type: str
    saved_formats: list[str]
    created_at: datetime
    modified_at: datetime
    expt_name: Optional[str] = None
    pert__type: Optional[str] = None
    pert__std: Optional[float] = None
    model_hashes: Optional[list[str]] = None

    model_config = {"from_attributes": True}


class FigureListResponse(BaseModel):
    """Paginated list of figures."""

    items: list[FigureInfo]
    total: int
    limit: int
    offset: int


class FigureDetail(BaseModel):
    """Full metadata for a single figure, including available files on disk."""

    hash: str
    evaluation_hash: str
    identifier: str
    figure_type: str
    saved_formats: list[str]
    created_at: datetime
    modified_at: datetime
    expt_name: Optional[str] = None
    pert__type: Optional[str] = None
    pert__std: Optional[float] = None
    model_hashes: Optional[list[str]] = None
    available_files: list[str]

    model_config = {"from_attributes": True}


class EvaluationFigureSummary(BaseModel):
    """Summary of figures grouped by evaluation."""

    evaluation_hash: str
    expt_name: Optional[str] = None
    figure_count: int
    latest_figure_date: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _figure_to_info(
    record: FigureRecord,
    expt_name: Optional[str] = None,
) -> FigureInfo:
    """Convert a FigureRecord (+ optional eval context) to a FigureInfo."""
    return FigureInfo(
        hash=record.hash,
        evaluation_hash=record.evaluation_hash,
        identifier=record.identifier,
        figure_type=record.figure_type,
        saved_formats=list(record.saved_formats) if record.saved_formats else [],
        created_at=record.created_at,
        modified_at=record.modified_at,
        expt_name=expt_name,
        pert__type=record.pert__type,
        pert__std=record.pert__std,
        model_hashes=list(record.model_hashes) if record.model_hashes else None,
    )


def _available_files(record: FigureRecord) -> list[str]:
    """Return list of format extensions whose files exist on disk."""
    available: list[str] = []
    formats = list(record.saved_formats) if record.saved_formats else []
    for fmt in formats:
        if record.get_path(fmt).exists():
            available.append(fmt)
    return available


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


# NOTE: /evaluations must be registered before /{figure_hash} so that
# "evaluations" is not captured as a path parameter.
@router.get("/evaluations")
async def list_evaluations_with_figures() -> list[EvaluationFigureSummary]:
    """List distinct evaluations that have at least one (non-archived) figure.

    Returns evaluation_hash, expt_name, figure count, and the date of the
    most recently created figure.  Results are ordered by latest figure date
    descending.
    """
    from sqlalchemy import func

    with db_session(autocommit=False) as session:
        rows = (
            session.query(
                FigureRecord.evaluation_hash,
                EvaluationRecord.expt_name,
                func.count(FigureRecord.id).label("figure_count"),
                func.max(FigureRecord.created_at).label("latest_figure_date"),
            )
            .join(
                EvaluationRecord,
                FigureRecord.evaluation_hash == EvaluationRecord.hash,
            )
            .filter(FigureRecord.archived == False)  # noqa: E712
            .group_by(FigureRecord.evaluation_hash, EvaluationRecord.expt_name)
            .order_by(func.max(FigureRecord.created_at).desc())
            .all()
        )

    return [
        EvaluationFigureSummary(
            evaluation_hash=row.evaluation_hash,
            expt_name=row.expt_name,
            figure_count=row.figure_count,
            latest_figure_date=row.latest_figure_date,
        )
        for row in rows
    ]


@router.get("/", response_model=FigureListResponse)
async def list_figures(
    evaluation_hash: Optional[str] = Query(default=None),
    expt_name: Optional[str] = Query(default=None),
    figure_type: Optional[str] = Query(default=None),
    identifier: Optional[str] = Query(default=None, description="Partial match"),
    pert_type: Optional[str] = Query(default=None),
    pert_std: Optional[float] = Query(default=None),
    date_from: Optional[datetime] = Query(default=None),
    date_to: Optional[datetime] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> FigureListResponse:
    """List and search figures with optional filters and pagination."""
    with db_session(autocommit=False) as session:
        query = (
            session.query(FigureRecord, EvaluationRecord.expt_name)
            .join(
                EvaluationRecord,
                FigureRecord.evaluation_hash == EvaluationRecord.hash,
            )
            .filter(FigureRecord.archived == False)  # noqa: E712
        )

        if evaluation_hash is not None:
            query = query.filter(FigureRecord.evaluation_hash == evaluation_hash)
        if expt_name is not None:
            query = query.filter(EvaluationRecord.expt_name == expt_name)
        if figure_type is not None:
            query = query.filter(FigureRecord.figure_type == figure_type)
        if identifier is not None:
            query = query.filter(FigureRecord.identifier.ilike(f"%{identifier}%"))
        if pert_type is not None:
            query = query.filter(FigureRecord.pert__type == pert_type)
        if pert_std is not None:
            query = query.filter(FigureRecord.pert__std == pert_std)
        if date_from is not None:
            query = query.filter(FigureRecord.created_at >= date_from)
        if date_to is not None:
            query = query.filter(FigureRecord.created_at <= date_to)

        total = query.count()

        rows = (
            query
            .order_by(FigureRecord.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    items = [_figure_to_info(fig_rec, eval_expt) for fig_rec, eval_expt in rows]

    return FigureListResponse(items=items, total=total, limit=limit, offset=offset)


@router.get("/{figure_hash}", response_model=FigureDetail)
async def get_figure(figure_hash: str) -> FigureDetail:
    """Get full metadata for a single figure by its hash."""
    with db_session(autocommit=False) as session:
        row = (
            session.query(FigureRecord, EvaluationRecord.expt_name)
            .join(
                EvaluationRecord,
                FigureRecord.evaluation_hash == EvaluationRecord.hash,
            )
            .filter(FigureRecord.hash == figure_hash)
            .first()
        )

    if row is None:
        raise HTTPException(status_code=404, detail=f"Figure '{figure_hash}' not found")

    fig_rec, eval_expt = row
    return FigureDetail(
        hash=fig_rec.hash,
        evaluation_hash=fig_rec.evaluation_hash,
        identifier=fig_rec.identifier,
        figure_type=fig_rec.figure_type,
        saved_formats=list(fig_rec.saved_formats) if fig_rec.saved_formats else [],
        created_at=fig_rec.created_at,
        modified_at=fig_rec.modified_at,
        expt_name=eval_expt,
        pert__type=fig_rec.pert__type,
        pert__std=fig_rec.pert__std,
        model_hashes=list(fig_rec.model_hashes) if fig_rec.model_hashes else None,
        available_files=_available_files(fig_rec),
    )


@router.get("/{figure_hash}/file")
async def get_figure_file(
    figure_hash: str,
    format: str = Query(default="json", description="File format: json, html, png, svg, webp, pdf"),
):
    """Serve the actual figure file in the requested format.

    For Plotly figures, ``json`` is the default and returns the full Plotly
    JSON specification (suitable for interactive rendering in the browser).
    """
    with db_session(autocommit=False) as session:
        fig_rec = (
            session.query(FigureRecord)
            .filter(FigureRecord.hash == figure_hash)
            .first()
        )

        if fig_rec is None:
            raise HTTPException(
                status_code=404, detail=f"Figure '{figure_hash}' not found",
            )

        # Normalise the requested format.
        fmt = format.strip(".").lower()

        saved = set(fig_rec.saved_formats) if fig_rec.saved_formats else set()
        if fmt not in saved:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Format '{fmt}' not available for figure '{figure_hash}'. "
                    f"Saved formats: {sorted(saved)}"
                ),
            )

        path = fig_rec.get_path(fmt)

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File for figure '{figure_hash}' in format '{fmt}' not found on disk",
        )

    content_type = _CONTENT_TYPES.get(fmt, "application/octet-stream")

    # For text-based formats (json, html, svg) we can use Response directly;
    # for binary formats FileResponse handles streaming efficiently.
    if fmt in {"json", "html", "svg"}:
        data = path.read_text(encoding="utf-8")
        return Response(content=data, media_type=content_type)

    return FileResponse(path=str(path), media_type=content_type)
