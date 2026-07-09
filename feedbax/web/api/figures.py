"""Manifest-backed API router for figure browsing and retrieval."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from feedbax.analysis.figures import FIGURE_RENDER_ROLE
from feedbax.contracts.manifest import FigureManifest, default_manifest_root, load_manifest
from feedbax.persistence.manifest_index import iter_manifest_files
from feedbax.plot.constructors import (
    constructor_catalog,
    registered_figure_pieces,
    registered_figure_templates,
)

router = APIRouter()

_CONTENT_TYPES: dict[str, str] = {
    "json": "application/json",
    "html": "text/html",
    "png": "image/png",
    "svg": "image/svg+xml",
    "webp": "image/webp",
    "pdf": "application/pdf",
}


class FigureInfo(BaseModel):
    """Summary metadata for one manifest-backed figure."""

    hash: str
    manifest_id: str
    name: str
    identifier: str
    figure_type: str
    saved_formats: list[str]
    created_at: datetime
    modified_at: datetime
    status: str | None = None
    template: str | None = None
    constructors: list[str] = []
    input_manifest_ids: list[str] = []
    expt_name: Optional[str] = None
    pert__type: Optional[str] = None
    pert__std: Optional[float] = None
    model_hashes: Optional[list[str]] = None


class FigureListResponse(BaseModel):
    """Paginated list of figures."""

    items: list[FigureInfo]
    total: int
    limit: int
    offset: int


class FigureDetail(FigureInfo):
    """Full metadata for a single figure, including available files."""

    available_files: list[str]
    artifacts: list[dict[str, Any]]
    binding_records: list[dict[str, Any]]


class EvaluationFigureSummary(BaseModel):
    """Compatibility summary of figures grouped by input manifest."""

    evaluation_hash: str
    expt_name: Optional[str] = None
    figure_count: int
    latest_figure_date: Optional[datetime] = None


class RegistryItem(BaseModel):
    """Studio-enumerable figure registry item."""

    name: str
    description: str
    metadata: dict[str, Any] = {}


def _figure_manifests(root: Path | None = None) -> list[FigureManifest]:
    root_path = root or default_manifest_root()
    manifests: list[FigureManifest] = []
    for path in iter_manifest_files(root_path):
        try:
            manifest = load_manifest(path)
        except Exception:
            continue
        if isinstance(manifest, FigureManifest):
            manifests.append(manifest)
    return sorted(manifests, key=lambda item: item.created_at, reverse=True)


def _figure_spec_inline(manifest: FigureManifest) -> dict[str, Any]:
    return dict(manifest.figure_spec.inline)


def _available_formats(manifest: FigureManifest) -> list[str]:
    formats: set[str] = set()
    for artifact in manifest.artifacts:
        if artifact.role != FIGURE_RENDER_ROLE:
            continue
        if artifact.media_type == "application/json":
            formats.add("json")
        elif artifact.media_type == "text/html":
            formats.add("html")
        elif artifact.uri:
            suffix = Path(artifact.uri).suffix.strip(".")
            if suffix:
                formats.add(suffix)
    return sorted(formats)


def _figure_to_info(manifest: FigureManifest) -> FigureInfo:
    spec = _figure_spec_inline(manifest)
    constructors = sorted(manifest.constructor_versions)
    figure_type = manifest.template_name or (constructors[0] if constructors else "custom")
    return FigureInfo(
        hash=manifest.id,
        manifest_id=manifest.id,
        name=str(spec.get("name", manifest.id)),
        identifier=str(spec.get("name", manifest.id)),
        figure_type=figure_type,
        saved_formats=_available_formats(manifest),
        created_at=manifest.created_at,
        modified_at=manifest.created_at,
        status=manifest.status,
        template=manifest.template_name,
        constructors=constructors,
        input_manifest_ids=[ref.id for ref in manifest.inputs],
        expt_name=spec.get("figure_routing", {}).get("experiment"),
    )


def _render_artifact(manifest: FigureManifest, fmt: str):
    for artifact in manifest.artifacts:
        if artifact.role != FIGURE_RENDER_ROLE or artifact.uri is None:
            continue
        if fmt == "json" and artifact.media_type == "application/json":
            return artifact
        if fmt == "html" and artifact.media_type == "text/html":
            return artifact
        if Path(artifact.uri).suffix.strip(".").lower() == fmt:
            return artifact
    return None


@router.get("/constructors")
async def list_figure_constructors() -> list[dict[str, Any]]:
    """List registered figure constructors with descriptions and params schemas."""
    return constructor_catalog()


@router.get("/templates", response_model=list[RegistryItem])
async def list_figure_templates() -> list[RegistryItem]:
    """List registered figure templates."""
    return [
        RegistryItem(
            name=template.name,
            description=template.description,
            metadata=template.model_dump(mode="json", exclude_none=True),
        )
        for template in registered_figure_templates()
    ]


@router.get("/pieces", response_model=list[RegistryItem])
async def list_figure_pieces() -> list[RegistryItem]:
    """List registered figure pieces."""
    return [
        RegistryItem(
            name=piece.name,
            description=piece.description,
            metadata=piece.model_dump(mode="json", exclude_none=True),
        )
        for piece in registered_figure_pieces()
    ]


@router.get("/evaluations")
async def list_evaluations_with_figures() -> list[EvaluationFigureSummary]:
    """List input manifests that have at least one generated figure."""
    grouped: dict[str, EvaluationFigureSummary] = {}
    for manifest in _figure_manifests():
        for input_ref in manifest.inputs:
            if input_ref.kind != "EvaluationRunManifest":
                continue
            current = grouped.get(input_ref.id)
            if current is None:
                grouped[input_ref.id] = EvaluationFigureSummary(
                    evaluation_hash=input_ref.id,
                    figure_count=1,
                    latest_figure_date=manifest.created_at,
                )
            else:
                current.figure_count += 1
                if (
                    current.latest_figure_date is None
                    or manifest.created_at > current.latest_figure_date
                ):
                    current.latest_figure_date = manifest.created_at
    return sorted(
        grouped.values(),
        key=lambda item: item.latest_figure_date or datetime.min,
        reverse=True,
    )


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
    """List and search manifest-backed figures with optional filters."""
    del pert_type, pert_std
    items = [_figure_to_info(manifest) for manifest in _figure_manifests()]
    if evaluation_hash is not None:
        items = [item for item in items if evaluation_hash in item.input_manifest_ids]
    if expt_name is not None:
        items = [item for item in items if item.expt_name == expt_name]
    if figure_type is not None:
        items = [item for item in items if item.figure_type == figure_type]
    if identifier is not None:
        needle = identifier.lower()
        items = [item for item in items if needle in item.identifier.lower()]
    if date_from is not None:
        items = [item for item in items if item.created_at >= date_from]
    if date_to is not None:
        items = [item for item in items if item.created_at <= date_to]
    total = len(items)
    return FigureListResponse(
        items=items[offset : offset + limit],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{figure_hash}", response_model=FigureDetail)
async def get_figure(figure_hash: str) -> FigureDetail:
    """Get full metadata for a single figure manifest."""
    manifest = next(
        (item for item in _figure_manifests() if item.id == figure_hash),
        None,
    )
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"Figure '{figure_hash}' not found")
    info = _figure_to_info(manifest)
    return FigureDetail(
        **info.model_dump(),
        available_files=_available_formats(manifest),
        artifacts=[
            artifact.model_dump(mode="json", exclude_none=True)
            for artifact in manifest.artifacts
        ],
        binding_records=[
            record.model_dump(mode="json", exclude_none=True)
            for record in manifest.binding_records
        ],
    )


@router.get("/{figure_hash}/file")
async def get_figure_file(
    figure_hash: str,
    format: str = Query(default="json", description="File format: json, html, png, svg, webp, pdf"),
):
    """Serve the rendered figure artifact in the requested format."""
    fmt = format.strip(".").lower()
    if fmt not in _CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{fmt}'. Allowed formats: {sorted(_CONTENT_TYPES)}",
        )
    manifest = next(
        (item for item in _figure_manifests() if item.id == figure_hash),
        None,
    )
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"Figure '{figure_hash}' not found")
    artifact = _render_artifact(manifest, fmt)
    if artifact is None or artifact.uri is None:
        raise HTTPException(
            status_code=404,
            detail=f"File for figure '{figure_hash}' in format '{fmt}' not found",
        )
    path = Path(artifact.uri)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File for figure '{figure_hash}' in format '{fmt}' not found on disk",
        )
    content_type = _CONTENT_TYPES.get(fmt, "application/octet-stream")
    if fmt in {"json", "html", "svg"}:
        return Response(content=path.read_text(encoding="utf-8"), media_type=content_type)
    return FileResponse(path=str(path), media_type=content_type)
