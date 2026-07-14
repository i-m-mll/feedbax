"""Registered execution for manifest-canonical report specs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from feedbax.analysis.specs import find_manifest_by_id
from feedbax.analysis.manifest_inputs import (
    is_authenticated_manifest_ref,
    resolve_manifest_input,
)
from feedbax.analysis.rendering import render_markdown_note
from feedbax.analysis.validation import (
    ReportRecipeProtocol,
    validate_namespaced_type_key,
    validate_report_recipe,
)
from feedbax.contracts.manifest import (
    AnalysisDataProduct,
    AnalysisRunManifest,
    AnyManifest,
    ArtifactRef,
    EntrypointRef,
    EvaluationRunManifest,
    ManifestStatus,
    ParentRef,
    Provenance,
    ReportManifest,
    ReportSpec,
    SpecPayload,
    collect_git_provenance,
    default_manifest_root,
    report_manifest_id,
    spec_payload,
    store_bytes_artifact,
    store_json_artifact,
    write_manifest,
)

REPORT_RENDER_ROLE = "report_render"
REPORT_RENDER_MEDIA_TYPES = frozenset({"text/markdown", "text/html", "application/json"})
BUNDLE_SUMMARY_REPORT_TYPE = "feedbax.bundle_summary"
STUDIO_REPORT_TYPE = "feedbax.studio_report"


@dataclass(frozen=True)
class ResolvedReportInput:
    """A manifest parent resolved from a ``ReportSpec`` input ref."""

    ref: ParentRef
    manifest: AnyManifest | None
    path: Path | None
    produced_data: list[AnalysisDataProduct] = field(default_factory=list)


@dataclass(frozen=True)
class ReportRecipeResult:
    """Artifacts and metadata returned by a registered report recipe."""

    artifacts: list[ArtifactRef] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    regeneration_specs: list[SpecPayload | ParentRef | ArtifactRef] = field(default_factory=list)


ReportRecipe = ReportRecipeProtocol

_REPORT_RECIPES: dict[str, ReportRecipe] = {}


class ReportRecipeExecutionError(RuntimeError):
    """Raised after a registered report recipe fails and a failed manifest is written."""

    def __init__(self, manifest: ReportManifest, path: Path, cause: BaseException):
        super().__init__(
            f"Report recipe for {manifest.id!r} failed; failed manifest written to {path}"
        )
        self.manifest = manifest
        self.path = path
        self.__cause__ = cause


def register_report_recipe(
    report_type: str,
    recipe: ReportRecipe,
    *,
    replace: bool = False,
) -> None:
    """Register an executable report recipe by stable type key."""
    report_type = validate_namespaced_type_key(report_type, field="report_type")
    if report_type in _REPORT_RECIPES and not replace:
        raise ValueError(f"Report recipe {report_type!r} is already registered")
    _REPORT_RECIPES[report_type] = validate_report_recipe(report_type, recipe)


def unregister_report_recipe(report_type: str) -> None:
    """Remove a previously registered report recipe."""
    _REPORT_RECIPES.pop(report_type, None)


def registered_report_types() -> tuple[str, ...]:
    """Return registered executable report type keys."""
    return tuple(sorted(_REPORT_RECIPES))


def get_report_recipe(report_type: str) -> ReportRecipe:
    """Return a registered report recipe or raise a clear execution error."""
    try:
        return _REPORT_RECIPES[report_type]
    except KeyError as exc:
        available = ", ".join(registered_report_types()) or "none"
        raise ValueError(
            f"Report recipe {report_type!r} is not registered. "
            f"Registered report recipes: {available}."
        ) from exc


def coerce_report_spec(value: ReportSpec | Mapping[str, Any] | Path | str) -> ReportSpec:
    """Load a ``ReportSpec`` from an object, mapping, or JSON file path."""
    if isinstance(value, ReportSpec):
        return value
    if isinstance(value, Mapping):
        return ReportSpec.model_validate(value)
    path = Path(value)
    return ReportSpec.model_validate_json(path.read_text(encoding="utf-8"))


def resolve_report_inputs(
    spec: ReportSpec,
    *,
    root: Path | str | None = None,
) -> list[ResolvedReportInput]:
    """Resolve ``ReportSpec.inputs`` to manifests and analysis products."""
    root_path = Path(root) if root is not None else default_manifest_root()
    resolved: list[ResolvedReportInput] = []
    for ref in spec.inputs:
        manifest: AnyManifest | None = None
        manifest_path: Path | None = None
        produced_data: list[AnalysisDataProduct] = []
        if is_authenticated_manifest_ref(ref):
            authenticated = resolve_manifest_input(ref, root_path)
            manifest, manifest_path = authenticated.manifest, authenticated.path
        elif ref.kind.endswith("Manifest"):
            manifest, manifest_path = find_manifest_by_id(ref.id, root=root_path)
        if isinstance(manifest, AnalysisRunManifest):
            produced_data = list(manifest.produced_data)
        elif isinstance(manifest, EvaluationRunManifest):
            produced_data = []
        resolved.append(
            ResolvedReportInput(
                ref=ref,
                manifest=manifest,
                path=manifest_path,
                produced_data=produced_data,
            )
        )
    return resolved


def execute_report_spec(
    spec: ReportSpec | Mapping[str, Any] | Path | str,
    *,
    root: Path | str | None = None,
    provenance: Provenance | None = None,
    issues: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    regeneration_specs: Sequence[SpecPayload | ParentRef | ArtifactRef] = (),
) -> tuple[ReportManifest, Path]:
    """Execute a serialized report spec and write a truthful manifest."""
    report_spec = coerce_report_spec(spec)
    recipe = get_report_recipe(report_spec.report_type)
    root_path = Path(root) if root is not None else default_manifest_root()
    manifest_id = report_manifest_id(report_spec)

    prov = provenance.model_copy(deep=True) if provenance is not None else collect_git_provenance()
    prov.parents = list(report_spec.inputs)
    if issues:
        prov.issues.extend(issue for issue in issues if issue not in prov.issues)
    if prov.entrypoint is None:
        prov.entrypoint = EntrypointRef(
            kind="feedbax-report-recipe",
            name=report_spec.report_type,
        )

    manifest_metadata = dict(metadata or {})
    resolved_inputs = resolve_report_inputs(report_spec, root=root_path)
    try:
        result = recipe(report_spec, root_path, resolved_inputs)
        _validate_report_result(report_spec.report_type, result)
        manifest = _build_report_manifest(
            manifest_id=manifest_id,
            report_spec=report_spec,
            status="completed",
            provenance=prov,
            artifacts=result.artifacts,
            regeneration_specs=[*regeneration_specs, *result.regeneration_specs],
            metadata={
                **manifest_metadata,
                **result.metadata,
                "summary": result.summary,
            },
        )
        return manifest, write_manifest(manifest, root=root_path)
    except Exception as exc:
        manifest = _build_report_manifest(
            manifest_id=manifest_id,
            report_spec=report_spec,
            status="failed",
            provenance=prov,
            artifacts=[],
            regeneration_specs=list(regeneration_specs),
            metadata={
                **manifest_metadata,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            },
        )
        path = write_manifest(manifest, root=root_path)
        raise ReportRecipeExecutionError(manifest, path, exc) from exc


def _validate_report_result(report_type: str, result: ReportRecipeResult) -> None:
    renders = [artifact for artifact in result.artifacts if artifact.role == REPORT_RENDER_ROLE]
    if not renders:
        raise ValueError(
            f"Report recipe {report_type!r} must return at least one "
            f"{REPORT_RENDER_ROLE!r} artifact"
        )
    supported = [
        artifact
        for artifact in renders
        if artifact.media_type in REPORT_RENDER_MEDIA_TYPES and artifact.sha256
    ]
    if not supported:
        media_types = ", ".join(sorted(REPORT_RENDER_MEDIA_TYPES))
        raise ValueError(
            f"Report recipe {report_type!r} must return a {REPORT_RENDER_ROLE!r} "
            f"artifact with sha256 and media_type in {{{media_types}}}"
        )


def _build_report_manifest(
    *,
    manifest_id: str,
    report_spec: ReportSpec,
    status: ManifestStatus,
    provenance: Provenance,
    artifacts: list[ArtifactRef],
    regeneration_specs: list[SpecPayload | ParentRef | ArtifactRef],
    metadata: dict[str, Any],
) -> ReportManifest:
    return ReportManifest(
        id=manifest_id,
        status=status,
        report_spec=spec_payload(
            "ReportSpec",
            report_spec.model_dump(mode="json", exclude_none=True),
        ),
        inputs=list(report_spec.inputs),
        provenance=provenance,
        artifacts=artifacts,
        regeneration_specs=regeneration_specs,
        metadata=metadata,
    )


def _bundle_summary_recipe(
    report_spec: ReportSpec,
    root: Path,
    inputs: Sequence[ResolvedReportInput],
) -> ReportRecipeResult:
    bundle_params = report_spec.params.get("bundle", {})
    bundle_name = str(bundle_params.get("name", "analysis-bundle"))
    stage_name = str(bundle_params.get("stage", "report"))
    report_index = bundle_params.get("index", 0)
    input_manifest_ids = [resolved.ref.id for resolved in inputs]
    report_body = {
        "kind": "AnalysisBundleReport",
        "bundle": bundle_name,
        "stage": stage_name,
        "input_manifest_ids": input_manifest_ids,
        "params": dict(report_spec.params.get("stage_params", {})),
    }
    json_artifact = store_json_artifact(
        report_body,
        root=root,
        role="report",
        logical_name=f"{bundle_name}-{stage_name}-{report_index}.json",
        metadata={"bundle": bundle_name, "stage": stage_name},
    )
    markdown = _markdown_report(
        title=f"{bundle_name} / {stage_name}",
        narrative=report_spec.narrative,
        rows=[
            ("Report type", report_spec.report_type),
            ("Input manifests", ", ".join(input_manifest_ids) or "none"),
        ],
    )
    render_artifact = store_bytes_artifact(
        markdown.encode("utf-8"),
        root=root,
        role=REPORT_RENDER_ROLE,
        logical_name=f"{bundle_name}-{stage_name}-{report_index}.md",
        media_type="text/markdown",
        suffix=".md",
        metadata={"bundle": bundle_name, "stage": stage_name},
    )
    return ReportRecipeResult(
        artifacts=[json_artifact, render_artifact],
        summary={"input_manifests": len(input_manifest_ids)},
        metadata={"bundle": bundle_params},
    )


def _studio_report_recipe(
    report_spec: ReportSpec,
    root: Path,
    inputs: Sequence[ResolvedReportInput],
) -> ReportRecipeResult:
    studio_params = report_spec.params.get("studio", {})
    job_id = str(studio_params.get("job_id", "studio-report"))
    stage_id = str(studio_params.get("stage_id", "report"))
    title = str(studio_params.get("title", "Studio report"))
    input_manifest_ids = [resolved.ref.id for resolved in inputs]
    report_body = {
        "kind": "StudioReportProduct",
        "job_id": job_id,
        "stage_id": stage_id,
        "input_analysis_products": input_manifest_ids,
        "title": title,
        "status": "completed",
    }
    json_artifact = store_json_artifact(
        report_body,
        root=root,
        role="report",
        logical_name=f"{job_id}-report.json",
        metadata={"stage_id": stage_id, "job_id": job_id},
    )
    markdown = _markdown_report(
        title=title,
        narrative=report_spec.narrative,
        rows=[
            ("Job", job_id),
            ("Stage", stage_id),
            ("Input manifests", ", ".join(input_manifest_ids) or "none"),
        ],
    )
    render_artifact = store_bytes_artifact(
        markdown.encode("utf-8"),
        root=root,
        role=REPORT_RENDER_ROLE,
        logical_name=f"{job_id}-report.md",
        media_type="text/markdown",
        suffix=".md",
        metadata={"stage_id": stage_id, "job_id": job_id},
    )
    return ReportRecipeResult(
        artifacts=[json_artifact, render_artifact],
        summary={"input_manifests": len(input_manifest_ids)},
        metadata={"studio": studio_params},
    )


def _markdown_report(
    *,
    title: str,
    narrative: str | None,
    rows: Sequence[tuple[str, str]],
) -> str:
    return render_markdown_note(title=title, narrative=narrative, rows=rows)


register_report_recipe(BUNDLE_SUMMARY_REPORT_TYPE, _bundle_summary_recipe, replace=True)
register_report_recipe(STUDIO_REPORT_TYPE, _studio_report_recipe, replace=True)
