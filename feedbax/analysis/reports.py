"""Registered execution for manifest-canonical report specs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import html
import json
import math
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import Field, model_validator

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
    FigureManifest,
    ManifestStatus,
    ParentRef,
    Provenance,
    ReportManifest,
    ReportSpec,
    SpecPayload,
    StrictModel,
    collect_git_provenance,
    default_manifest_root,
    report_manifest_id,
    sha256_bytes,
    spec_payload,
    store_bytes_artifact,
    store_json_artifact,
    write_manifest,
)

REPORT_RENDER_ROLE = "report_render"
REPORT_RENDER_MEDIA_TYPES = frozenset({"text/markdown", "text/html", "application/json"})
BUNDLE_SUMMARY_REPORT_TYPE = "feedbax.bundle_summary"
STUDIO_REPORT_TYPE = "feedbax.studio_report"
ORDERED_FIGURE_REPORT_TYPE = "feedbax.ordered_figure_report"
ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_ID = "feedbax.spec.report.ordered_figure"
ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_VERSION = "feedbax.spec.report.ordered_figure.v1"

OrderedFigureReportApplicability = Literal["included", "not_applicable"]
OrderedFigureReportScalar = str | int | float | bool | None


def _require_authored_text(value: str, *, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value


class OrderedFigureReportFigure(StrictModel):
    """One authored figure placement in an ordered report section."""

    input_role: str | None = None
    caption: str
    applicability: OrderedFigureReportApplicability = "included"
    not_applicable_reason: str | None = None

    @model_validator(mode="after")
    def _validate_applicability(self) -> "OrderedFigureReportFigure":
        _require_authored_text(self.caption, field_name="figure caption")
        if self.applicability == "included":
            if self.input_role is None:
                raise ValueError("included figure requires input_role")
            _require_authored_text(self.input_role, field_name="figure input_role")
            if self.not_applicable_reason is not None:
                raise ValueError("included figure cannot declare not_applicable_reason")
        else:
            if self.input_role is not None:
                raise ValueError("not-applicable figure cannot declare input_role")
            if self.not_applicable_reason is None:
                raise ValueError("not-applicable figure requires not_applicable_reason")
            _require_authored_text(
                self.not_applicable_reason,
                field_name="figure not_applicable_reason",
            )
        return self


class OrderedFigureReportScalarTable(StrictModel):
    """A small authored table whose cells are JSON-native scalar values."""

    title: str | None = None
    columns: list[str]
    rows: list[list[OrderedFigureReportScalar]]

    @model_validator(mode="after")
    def _validate_table(self) -> "OrderedFigureReportScalarTable":
        if self.title is not None:
            _require_authored_text(self.title, field_name="scalar table title")
        if not self.columns:
            raise ValueError("scalar table requires at least one column")
        for column in self.columns:
            _require_authored_text(column, field_name="scalar table column")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("scalar table columns must be unique")
        for row_index, row in enumerate(self.rows):
            if len(row) != len(self.columns):
                raise ValueError(
                    f"scalar table row {row_index} has {len(row)} cells; "
                    f"expected {len(self.columns)}"
                )
            for cell in row:
                if isinstance(cell, float) and not math.isfinite(cell):
                    raise ValueError("scalar table cells must be finite JSON values")
        return self


class OrderedFigureReportSection(StrictModel):
    """One authored section in an ordered figure report."""

    title: str
    framing: str | None = None
    applicability: OrderedFigureReportApplicability = "included"
    not_applicable_reason: str | None = None
    figures: list[OrderedFigureReportFigure] = Field(default_factory=list)
    tables: list[OrderedFigureReportScalarTable] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_applicability(self) -> "OrderedFigureReportSection":
        _require_authored_text(self.title, field_name="section title")
        if self.applicability == "included":
            if self.not_applicable_reason is not None:
                raise ValueError("included section cannot declare not_applicable_reason")
        else:
            if self.not_applicable_reason is None:
                raise ValueError("not-applicable section requires not_applicable_reason")
            _require_authored_text(
                self.not_applicable_reason,
                field_name="section not_applicable_reason",
            )
            if self.figures or self.tables:
                raise ValueError("not-applicable section cannot declare figure or table content")
        return self


class OrderedFigureReportParams(StrictModel):
    """Versioned authored content for the generic ordered-figure recipe."""

    schema_id: Literal["feedbax.spec.report.ordered_figure"]
    schema_version: Literal["feedbax.spec.report.ordered_figure.v1"]
    title: str
    introduction: str | None = None
    sections: list[OrderedFigureReportSection]
    output_name: str = "ordered-figure-report.md"

    @model_validator(mode="after")
    def _validate_report(self) -> "OrderedFigureReportParams":
        _require_authored_text(self.title, field_name="report title")
        if not self.sections:
            raise ValueError("ordered figure report requires at least one section")
        output_suffix = Path(self.output_name).suffix
        if (
            not self.output_name.strip()
            or Path(self.output_name).name != self.output_name
            or output_suffix not in {".html", ".md"}
        ):
            raise ValueError(
                "output_name must be a Markdown or HTML filename without path components"
            )
        included_roles = [
            figure.input_role
            for section in self.sections
            if section.applicability == "included"
            for figure in section.figures
            if figure.applicability == "included"
        ]
        if len(set(included_roles)) != len(included_roles):
            raise ValueError("included figure input_role values must be unique")
        return self


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


def _ordered_figure_report_recipe(
    report_spec: ReportSpec,
    root: Path,
    inputs: Sequence[ResolvedReportInput],
) -> ReportRecipeResult:
    if report_spec.narrative is not None:
        raise ValueError("ordered figure report narrative must be authored as params.introduction")
    params = OrderedFigureReportParams.model_validate(report_spec.params)
    inputs_by_role = _ordered_figure_inputs_by_role(params, inputs)
    output_suffix = Path(params.output_name).suffix
    if output_suffix == ".html":
        rendered = render_ordered_figure_report_html(params, inputs_by_role)
        media_type = "text/html"
    else:
        rendered = render_ordered_figure_report_markdown(params, inputs_by_role)
        media_type = "text/markdown"
    render_artifact = store_bytes_artifact(
        rendered.encode("utf-8"),
        root=root,
        role=REPORT_RENDER_ROLE,
        logical_name=params.output_name,
        media_type=media_type,
        suffix=output_suffix,
        metadata={
            "report_type": ORDERED_FIGURE_REPORT_TYPE,
            "params_schema_id": params.schema_id,
            "params_schema_version": params.schema_version,
        },
    )
    included_figures = sum(
        figure.applicability == "included"
        for section in params.sections
        if section.applicability == "included"
        for figure in section.figures
    )
    not_applicable_items = sum(
        section.applicability == "not_applicable" for section in params.sections
    ) + sum(
        figure.applicability == "not_applicable"
        for section in params.sections
        if section.applicability == "included"
        for figure in section.figures
    )
    return ReportRecipeResult(
        artifacts=[render_artifact],
        summary={
            "sections": len(params.sections),
            "included_figures": included_figures,
            "not_applicable_items": not_applicable_items,
            "scalar_tables": sum(
                len(section.tables)
                for section in params.sections
                if section.applicability == "included"
            ),
        },
        metadata={
            "ordered_figure_report": {
                "schema_id": params.schema_id,
                "schema_version": params.schema_version,
            }
        },
    )


def _ordered_figure_inputs_by_role(
    params: OrderedFigureReportParams,
    inputs: Sequence[ResolvedReportInput],
) -> dict[str, FigureManifest]:
    expected_roles = {
        str(figure.input_role)
        for section in params.sections
        if section.applicability == "included"
        for figure in section.figures
        if figure.applicability == "included"
    }
    resolved_by_role: dict[str, FigureManifest] = {}
    for resolved in inputs:
        role = resolved.ref.role
        if role is None or not role.strip():
            raise ValueError(
                f"ordered figure report input {resolved.ref.id!r} requires a non-blank role"
            )
        if role in resolved_by_role:
            raise ValueError(f"ordered figure report input role {role!r} is duplicated")
        if role not in expected_roles:
            raise ValueError(f"ordered figure report input role {role!r} is not referenced")
        if resolved.ref.kind != "FigureManifest":
            raise ValueError(
                f"ordered figure report input role {role!r} must reference FigureManifest"
            )
        if not isinstance(resolved.manifest, FigureManifest):
            raise ValueError(
                f"ordered figure report input role {role!r} did not resolve to FigureManifest"
            )
        if resolved.manifest.status != "completed":
            raise ValueError(
                f"ordered figure report input role {role!r} has non-completed status "
                f"{resolved.manifest.status!r}"
            )
        renders = [
            artifact
            for artifact in resolved.manifest.artifacts
            if artifact.role == "figure_render" and artifact.sha256 and artifact.uri
        ]
        if not renders:
            raise ValueError(
                f"ordered figure report input role {role!r} has no materialized "
                "figure_render artifact"
            )
        resolved_by_role[role] = resolved.manifest
    missing_roles = sorted(expected_roles - resolved_by_role.keys())
    if missing_roles:
        raise ValueError(
            "ordered figure report is missing required input roles: "
            + ", ".join(repr(role) for role in missing_roles)
        )
    return resolved_by_role


def render_ordered_figure_report_markdown(
    params: OrderedFigureReportParams,
    inputs_by_role: Mapping[str, FigureManifest],
) -> str:
    """Render an ordered-figure report as Markdown."""
    lines = [f"# {params.title}", ""]
    if params.introduction:
        lines.extend([params.introduction, ""])
    for section in params.sections:
        lines.extend([f"## {section.title}", ""])
        if section.framing:
            lines.extend([section.framing, ""])
        if section.applicability == "not_applicable":
            lines.extend([f"Not applicable: {section.not_applicable_reason}", ""])
            continue
        for figure in section.figures:
            if figure.applicability == "not_applicable":
                lines.extend(
                    [
                        f"**{figure.caption}**",
                        "",
                        f"Not applicable: {figure.not_applicable_reason}",
                        "",
                    ]
                )
                continue
            manifest = inputs_by_role[str(figure.input_role)]
            renders = [
                artifact
                for artifact in manifest.artifacts
                if artifact.role == "figure_render" and artifact.sha256 and artifact.uri
            ]
            for artifact in renders:
                label = _escape_markdown_text(artifact.logical_name)
                uri = str(artifact.uri).replace(">", "%3E")
                if artifact.media_type.startswith("image/"):
                    lines.extend([f"![{_escape_markdown_text(figure.caption)}](<{uri}>)", ""])
                else:
                    lines.extend([f"[{label}](<{uri}>)", ""])
            lines.extend([f"*{figure.caption}*", ""])
        for table in section.tables:
            if table.title:
                lines.extend([f"### {table.title}", ""])
            header = " | ".join(_escape_markdown_cell(column) for column in table.columns)
            lines.append(f"| {header} |")
            lines.append("| " + " | ".join("---" for _ in table.columns) + " |")
            for row in table.rows:
                lines.append(
                    "| "
                    + " | ".join(_escape_markdown_cell(_scalar_text(cell)) for cell in row)
                    + " |"
                )
            lines.append("")
    return "\n".join(lines)


def render_ordered_figure_report_html(
    params: OrderedFigureReportParams,
    inputs_by_role: Mapping[str, FigureManifest],
) -> str:
    """Render a self-contained ordered-figure report with interactive Plotly figures."""
    lines = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1">',
        f"  <title>{html.escape(params.title)}</title>",
        "  <style>",
        "    body { color: #202124; font-family: system-ui, sans-serif; "
        "line-height: 1.5; margin: 0 auto; max-width: 1100px; padding: 2rem; }",
        "    figure { margin: 1.5rem 0 2rem; }",
        "    figcaption { color: #4b5563; font-style: italic; margin-top: 0.5rem; }",
        "    table { border-collapse: collapse; margin: 1rem 0 2rem; width: 100%; }",
        "    th, td { border: 1px solid #d1d5db; padding: 0.4rem 0.6rem; text-align: left; }",
        "    th { background: #f3f4f6; }",
        "    .not-applicable { color: #4b5563; }",
        "  </style>",
        "</head>",
        "<body>",
        f"  <h1>{html.escape(params.title)}</h1>",
    ]
    if params.introduction:
        lines.extend(_html_text_block(params.introduction, indent="  "))

    rendered_figure_count = 0
    for section in params.sections:
        lines.append(f"  <section><h2>{html.escape(section.title)}</h2>")
        if section.framing:
            lines.extend(_html_text_block(section.framing, indent="    "))
        if section.applicability == "not_applicable":
            lines.append(
                '    <p class="not-applicable">Not applicable: '
                f"{html.escape(str(section.not_applicable_reason))}</p>"
            )
            lines.append("  </section>")
            continue
        for figure in section.figures:
            if figure.applicability == "not_applicable":
                lines.extend(
                    [
                        f"    <p><strong>{html.escape(figure.caption)}</strong></p>",
                        '    <p class="not-applicable">Not applicable: '
                        f"{html.escape(str(figure.not_applicable_reason))}</p>",
                    ]
                )
                continue
            manifest = inputs_by_role[str(figure.input_role)]
            plotly_artifacts = _plotly_json_artifacts(
                manifest,
                input_role=str(figure.input_role),
            )
            lines.append("    <figure>")
            for artifact in plotly_artifacts:
                rendered_figure_count += 1
                lines.append(
                    _plotly_html_fragment(
                        artifact,
                        div_id=f"ordered-figure-{rendered_figure_count}",
                        include_plotlyjs=rendered_figure_count == 1,
                    )
                )
            lines.append(f"      <figcaption>{html.escape(figure.caption)}</figcaption>")
            lines.append("    </figure>")
        for table in section.tables:
            if table.title:
                lines.append(f"    <h3>{html.escape(table.title)}</h3>")
            lines.extend(
                [
                    "    <table>",
                    "      <thead><tr>",
                    *[
                        f"        <th>{html.escape(column)}</th>"
                        for column in table.columns
                    ],
                    "      </tr></thead>",
                    "      <tbody>",
                ]
            )
            for row in table.rows:
                lines.extend(
                    [
                        "        <tr>",
                        *[
                            f"          <td>{html.escape(_scalar_text(cell))}</td>"
                            for cell in row
                        ],
                        "        </tr>",
                    ]
                )
            lines.extend(["      </tbody>", "    </table>"])
        lines.append("  </section>")
    lines.extend(["</body>", "</html>", ""])
    return "\n".join(lines)


def _html_text_block(value: str, *, indent: str) -> list[str]:
    escaped = html.escape(value).replace("\n", "<br>\n")
    return [f"{indent}<p>{escaped}</p>"]


def _plotly_json_artifacts(
    manifest: FigureManifest,
    *,
    input_role: str,
) -> list[ArtifactRef]:
    artifacts = [
        artifact
        for artifact in manifest.artifacts
        if artifact.role == "figure_render"
        and artifact.media_type == "application/json"
        and artifact.metadata.get("format") == "plotly-json"
        and artifact.sha256
        and artifact.uri
    ]
    if not artifacts:
        raise ValueError(
            f"ordered figure report HTML input role {input_role!r} has no materialized "
            "plotly-json figure_render artifact"
        )
    return artifacts


def _plotly_html_fragment(
    artifact: ArtifactRef,
    *,
    div_id: str,
    include_plotlyjs: bool,
) -> str:
    path = Path(str(artifact.uri))
    data = path.read_bytes()
    digest = sha256_bytes(data)
    if digest != artifact.sha256:
        raise ValueError(
            "ordered figure report Plotly JSON SHA-256 mismatch: "
            f"logical_name={artifact.logical_name!r}, expected={artifact.sha256!r}, "
            f"computed={digest!r}"
        )
    try:
        import plotly.io as pio
    except ImportError as exc:
        raise RuntimeError(
            "HTML ordered figure reports require Plotly; install the Feedbax analysis extra"
        ) from exc

    figure = pio.from_json(data.decode("utf-8"))
    fragment = figure.to_html(
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        div_id=_safe_html_id(div_id),
    )
    return "\n".join(f"      {line}" for line in fragment.splitlines())


def _safe_html_id(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")
    return normalized or "ordered-figure"


def _scalar_text(value: OrderedFigureReportScalar) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _escape_markdown_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def _escape_markdown_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _markdown_report(
    *,
    title: str,
    narrative: str | None,
    rows: Sequence[tuple[str, str]],
) -> str:
    return render_markdown_note(title=title, narrative=narrative, rows=rows)


register_report_recipe(BUNDLE_SUMMARY_REPORT_TYPE, _bundle_summary_recipe, replace=True)
register_report_recipe(STUDIO_REPORT_TYPE, _studio_report_recipe, replace=True)
register_report_recipe(
    ORDERED_FIGURE_REPORT_TYPE,
    _ordered_figure_report_recipe,
    replace=True,
)
