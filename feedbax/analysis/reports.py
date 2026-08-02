"""Registered execution for manifest-canonical report specs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import html
import json
import math
from pathlib import Path
import re
from string import Formatter
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from feedbax.analysis.execution_context import (
    EMPTY_STAGED_EXECUTION_CONTEXT,
    StagedArtifactProviderRootBinding,
    StagedCheckpointCustodyRootBinding,
    StagedExecutionContext,
    StagedManifestRootBinding,
    StagedParentExecutionLocation,
    resolve_staged_execution_context,
    with_staged_manifest_provider_inputs,
    with_staged_parent_execution_locations,
)
from feedbax.analysis.exact_parents import StagedExactParents, migrate_staged_exact_parents
from feedbax.analysis.manifest_inputs import (
    is_authenticated_manifest_ref,
    resolve_manifest_input,
)
from feedbax.analysis.rendering import render_markdown_note
from feedbax.analysis.specs import find_manifest_by_id
from feedbax.analysis.validation import (
    ReportRecipeProtocol,
    validate_namespaced_type_key,
    validate_report_recipe,
)
from feedbax.contracts.staged_execution import StagedExecutionDescriptor
from feedbax.contracts.strict_json import DuplicateJsonKeyError, strict_json_loads
from feedbax.contracts.figures import (
    FIGURE_SPEC_SCHEMA_ID,
    FIGURE_SPEC_SCHEMA_VERSION,
    FigureSpec,
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
    canonical_json_bytes,
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
ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_VERSION = "feedbax.spec.report.ordered_figure.v3"
ORDERED_FIGURE_REPORT_SCALAR_PROJECTION_SCHEMA_ID = "feedbax.spec.report.scalar_table_projection"
ORDERED_FIGURE_REPORT_SCALAR_PROJECTION_SCHEMA_VERSION = (
    "feedbax.spec.report.scalar_table_projection.v1"
)
ORDERED_FIGURE_REPORT_COMPOSITE_SCALAR_SCHEMA_ID = "feedbax.spec.report.composite_scalar_table_cell"
ORDERED_FIGURE_REPORT_COMPOSITE_SCALAR_SCHEMA_VERSION = (
    "feedbax.spec.report.composite_scalar_table_cell.v1"
)

OrderedFigureReportApplicability = Literal["included", "not_applicable"]
OrderedFigureReportScalar = str | int | float | bool | None


def _require_authored_text(value: str, *, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value


class OrderedFigureReportFigure(StrictModel):
    """One authored figure placement in an ordered report section."""

    input_role: str | None = None
    figure_spec_sha256: str | None = None
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
            if self.figure_spec_sha256 is None:
                raise ValueError("included figure requires figure_spec_sha256")
            if self.not_applicable_reason is not None:
                raise ValueError("included figure cannot declare not_applicable_reason")
        else:
            if self.input_role is not None:
                raise ValueError("not-applicable figure cannot declare input_role")
            if self.figure_spec_sha256 is not None:
                raise ValueError("not-applicable figure cannot declare figure_spec_sha256")
            if self.not_applicable_reason is None:
                raise ValueError("not-applicable figure requires not_applicable_reason")
            _require_authored_text(
                self.not_applicable_reason,
                field_name="figure not_applicable_reason",
            )
        return self

    @field_validator("figure_spec_sha256")
    @classmethod
    def _validate_figure_spec_sha256(cls, value: str | None) -> str | None:
        if value is not None and re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("figure_spec_sha256 must be a lowercase SHA-256 digest")
        return value


class OrderedFigureReportScalarProjection(StrictModel):
    """One exact scalar selected from a custody-backed analysis data product."""

    schema_id: Literal["feedbax.spec.report.scalar_table_projection"] = (
        ORDERED_FIGURE_REPORT_SCALAR_PROJECTION_SCHEMA_ID
    )
    schema_version: Literal["feedbax.spec.report.scalar_table_projection.v1"] = (
        ORDERED_FIGURE_REPORT_SCALAR_PROJECTION_SCHEMA_VERSION
    )
    kind: Literal["custody_projection"] = "custody_projection"
    input_role: str
    product_role: str
    product_schema_id: str
    product_schema_version: str
    artifact_role: str
    artifact_provider: str
    path: list[str | int]

    @model_validator(mode="after")
    def _validate_projection(self) -> "OrderedFigureReportScalarProjection":
        for field_name in (
            "input_role",
            "product_role",
            "product_schema_id",
            "product_schema_version",
            "artifact_role",
            "artifact_provider",
        ):
            _require_authored_text(getattr(self, field_name), field_name=field_name)
        if not self.path:
            raise ValueError("scalar table projection path must not be empty")
        for component in self.path:
            if isinstance(component, str):
                _require_authored_text(component, field_name="scalar table projection path")
            elif component < 0:
                raise ValueError("scalar table projection list indices must be non-negative")
        return self


class OrderedFigureReportCompositeScalarCell(StrictModel):
    """One formatted cell composed from separately authenticated scalar projections."""

    schema_id: Literal["feedbax.spec.report.composite_scalar_table_cell"] = (
        ORDERED_FIGURE_REPORT_COMPOSITE_SCALAR_SCHEMA_ID
    )
    schema_version: Literal["feedbax.spec.report.composite_scalar_table_cell.v1"] = (
        ORDERED_FIGURE_REPORT_COMPOSITE_SCALAR_SCHEMA_VERSION
    )
    kind: Literal["composite_scalar"] = "composite_scalar"
    format: str
    projections: dict[str, OrderedFigureReportScalarProjection]

    @model_validator(mode="after")
    def _validate_composite(self) -> "OrderedFigureReportCompositeScalarCell":
        _require_authored_text(self.format, field_name="composite scalar cell format")
        if not self.projections:
            raise ValueError("composite scalar cell requires at least one projection")
        for name in self.projections:
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
                raise ValueError("composite scalar projection names must be identifier-like")
        try:
            fields = [
                (field_name, format_spec, conversion)
                for _, field_name, format_spec, conversion in Formatter().parse(self.format)
                if field_name is not None
            ]
        except ValueError as exc:
            raise ValueError("composite scalar cell format is malformed") from exc
        field_names = [field_name for field_name, _, _ in fields]
        if len(field_names) != len(set(field_names)):
            raise ValueError("composite scalar cell format must use each projection once")
        if set(field_names) != set(self.projections):
            raise ValueError(
                "composite scalar cell format fields must exactly match projection names"
            )
        for field_name, format_spec, conversion in fields:
            if conversion is not None:
                raise ValueError("composite scalar cell format conversions are not supported")
            significant_digits = re.fullmatch(r"\.([1-9][0-9]*)g", format_spec)
            if significant_digits is None or int(significant_digits.group(1)) > 17:
                raise ValueError(
                    f"composite scalar projection {field_name!r} requires an authored "
                    "significant-digit format from '.1g' through '.17g'"
                )
        return self


OrderedFigureReportTableCell = (
    OrderedFigureReportCompositeScalarCell
    | OrderedFigureReportScalarProjection
    | OrderedFigureReportScalar
)


class OrderedFigureReportScalarTable(StrictModel):
    """A small table of authored or custody-projected JSON-native scalar values."""

    title: str | None = None
    columns: list[str]
    rows: list[list[OrderedFigureReportTableCell]]

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

    schema_id: str
    schema_version: str
    title: str
    introduction: str | None = None
    sections: list[OrderedFigureReportSection]
    output_name: str = "ordered-figure-report.md"

    @model_validator(mode="after")
    def _validate_report(self) -> "OrderedFigureReportParams":
        if self.schema_id != ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_ID:
            raise ValueError(
                "unsupported ordered figure report schema_id: "
                f"{self.schema_id!r}; expected {ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_ID!r}"
            )
        if self.schema_version != ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_VERSION:
            raise ValueError(
                "unsupported ordered figure report schema_version: "
                f"{self.schema_version!r}; expected "
                f"{ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_VERSION!r}; "
                "older report specs must be re-authored"
            )
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
        projection_roles = {
            projection.input_role
            for section in self.sections
            for table in section.tables
            for row in table.rows
            for cell in row
            for projection in _table_cell_projections(cell)
        }
        overlapping_roles = sorted(set(included_roles) & projection_roles)
        if overlapping_roles:
            raise ValueError(
                "figure and scalar projection input roles must be disjoint: "
                + ", ".join(repr(role) for role in overlapping_roles)
            )
        return self


@dataclass(frozen=True)
class ResolvedReportInput:
    """A manifest parent resolved from a ``ReportSpec`` input ref."""

    ref: ParentRef
    manifest: AnyManifest | None
    path: Path | None
    produced_data: list[AnalysisDataProduct] = field(default_factory=list)
    execution_context: StagedExecutionContext = field(
        default=EMPTY_STAGED_EXECUTION_CONTEXT,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class ResolvedReportScalarProjection:
    """Authenticated scalar value and the exact product bytes that supplied it."""

    value: OrderedFigureReportScalar
    input_ref: ParentRef
    product: AnalysisDataProduct
    artifact: ArtifactRef


@dataclass(frozen=True)
class ReportRecipeResult:
    """Artifacts and metadata returned by a registered report recipe."""

    artifacts: list[ArtifactRef] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    regeneration_specs: list[SpecPayload | ParentRef | ArtifactRef] = field(default_factory=list)


ReportRecipe = ReportRecipeProtocol


class ReportRecipeRegistry:
    """Isolated report recipe registry."""

    def __init__(self) -> None:
        self._sealed = False
        self._recipes: dict[str, ReportRecipe] = {}

    def register(self, report_type: str, recipe: ReportRecipe) -> None:
        if self._sealed:
            raise RuntimeError("report recipe registry is sealed")
        report_type = validate_namespaced_type_key(report_type, field="report_type")
        if report_type in self._recipes:
            raise ValueError(f"Report recipe {report_type!r} is already registered")
        self._recipes[report_type] = validate_report_recipe(report_type, recipe)

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._recipes))

    def get(self, report_type: str) -> ReportRecipe:
        try:
            return self._recipes[report_type]
        except KeyError as exc:
            raise ValueError(
                f"Report recipe {report_type!r} is not registered; available={list(self.keys())!r}"
            ) from exc

    def seal(self) -> None:
        self._sealed = True


class ReportRecipeExecutionError(RuntimeError):
    """Raised after a registered report recipe fails and a failed manifest is written."""

    def __init__(self, manifest: ReportManifest, path: Path, cause: BaseException):
        super().__init__(
            f"Report recipe for {manifest.id!r} failed; failed manifest written to {path}"
        )
        self.manifest = manifest
        self.path = path
        self.__cause__ = cause


def coerce_report_spec(value: ReportSpec | Mapping[str, Any] | Path | str) -> ReportSpec:
    """Load a ``ReportSpec`` from an object, mapping, or JSON file path.

    Serialized documents are admitted through the registered schema family, so
    an unversioned historical document is accepted as the named v1 baseline
    while an unknown or explicitly rejected version fails closed.
    """
    from feedbax.contracts.migrations import migrate_report_spec_payload

    if isinstance(value, ReportSpec):
        return value
    if isinstance(value, Mapping):
        payload: Mapping[str, Any] = value
    else:
        payload = strict_json_loads(Path(value).read_text(encoding="utf-8"), ref=str(value))
        if not isinstance(payload, Mapping):
            raise ValueError("ReportSpec document must be a JSON object")
    return ReportSpec.model_validate(migrate_report_spec_payload(payload).payload)


def resolve_report_inputs(
    spec: ReportSpec,
    *,
    root: Path | str | None = None,
    execution_context: StagedExecutionContext = EMPTY_STAGED_EXECUTION_CONTEXT,
) -> list[ResolvedReportInput]:
    """Resolve ``ReportSpec.inputs`` to manifests and analysis products."""
    root_path = Path(root) if root is not None else default_manifest_root()
    resolved: list[ResolvedReportInput] = []
    for ref in spec.inputs:
        manifest: AnyManifest | None = None
        manifest_path: Path | None = None
        produced_data: list[AnalysisDataProduct] = []
        if is_authenticated_manifest_ref(ref):
            locations = [
                location
                for location in execution_context.parent_execution_locations
                if location.parent == ref
            ]
            if len(locations) > 1:
                raise ValueError(
                    f"report input {ref.id!r} has ambiguous retained runtime authorities"
                )
            if locations:
                authenticated = execution_context.resolve_manifest_input(ref)
            else:
                try:
                    authenticated = resolve_manifest_input(ref, root_path)
                except FileNotFoundError:
                    if not (
                        execution_context.manifest_roots
                        or execution_context.opened_artifact_providers
                    ):
                        raise
                    bound = with_staged_manifest_provider_inputs(execution_context, [ref])
                    authenticated = bound.resolve_manifest_input(ref)
            manifest, manifest_path = authenticated.manifest, authenticated.path
        elif ref.kind.endswith("Manifest"):
            # This branch addresses a manifest by identifier alone, so the only
            # thing the ref states that the lookup can be held to is the kind it
            # declared. A generic recipe that asks for an analysis parent and is
            # handed a same-id figure would otherwise read the wrong record and
            # produce a report describing something else. Artifact fulfillment
            # never reaches here: every parent it lowers carries an
            # authenticated profile, so this guards standalone execution.
            #
            # The kind is stated to the lookup rather than checked after it,
            # because the addressing tiers are what a wrong-kind record has to
            # be refused at: checking afterwards accepts whichever tier
            # answered first and only then notices.
            manifest, manifest_path = find_manifest_by_id(
                ref.id, root=root_path, expected_kind=ref.kind
            )
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
                execution_context=execution_context,
            )
        )
    return resolved


def resolve_report_scalar_projection(
    projection: OrderedFigureReportScalarProjection,
    inputs: Sequence[ResolvedReportInput],
) -> ResolvedReportScalarProjection:
    """Resolve one exact scalar from authenticated immutable JSON product bytes."""
    input_matches = [resolved for resolved in inputs if resolved.ref.role == projection.input_role]
    if len(input_matches) != 1:
        raise ValueError(
            f"scalar projection input role {projection.input_role!r} resolved "
            f"{len(input_matches)} inputs; expected exactly one"
        )
    resolved_input = input_matches[0]
    if not isinstance(resolved_input.manifest, AnalysisRunManifest):
        raise ValueError(
            f"scalar projection input role {projection.input_role!r} must resolve "
            "to AnalysisRunManifest"
        )
    if resolved_input.manifest.status != "completed":
        raise ValueError(
            f"scalar projection input role {projection.input_role!r} has non-completed "
            f"status {resolved_input.manifest.status!r}"
        )
    products = [
        product
        for product in resolved_input.produced_data
        if product.role == projection.product_role
    ]
    if len(products) != 1:
        raise ValueError(
            f"scalar projection product role {projection.product_role!r} resolved "
            f"{len(products)} products; expected exactly one"
        )
    product = products[0]
    expected_schema = (
        projection.product_schema_id,
        projection.product_schema_version,
    )
    observed_schema = (product.product_schema_id, product.product_schema_version)
    if observed_schema != expected_schema:
        raise ValueError(
            f"scalar projection product role {projection.product_role!r} schema mismatch: "
            f"expected {expected_schema!r}, observed {observed_schema!r}"
        )
    artifacts = [
        artifact
        for artifact in product.artifacts
        if artifact.role == projection.artifact_role and artifact.media_type == "application/json"
    ]
    if len(artifacts) != 1:
        raise ValueError(
            f"scalar projection artifact role {projection.artifact_role!r} resolved "
            f"{len(artifacts)} JSON artifacts; expected exactly one"
        )
    artifact = artifacts[0]
    try:
        provider = resolved_input.execution_context.artifact_provider(projection.artifact_provider)
        raw = provider.get_bytes(artifact)
    except Exception as exc:
        raise ValueError(
            f"scalar projection artifact provider {projection.artifact_provider!r} "
            f"rejected product role {projection.product_role!r}"
        ) from exc
    try:
        value: Any = strict_json_loads(
            raw, ref=f"scalar projection product role {projection.product_role!r} artifact"
        )
    except DuplicateJsonKeyError as exc:
        raise ValueError(
            f"scalar projection product role {projection.product_role!r} artifact states a "
            f"member twice: {exc}"
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"scalar projection product role {projection.product_role!r} artifact is not valid JSON"
        ) from exc
    traversed: list[str | int] = []
    for component in projection.path:
        traversed.append(component)
        if isinstance(component, str):
            if not isinstance(value, Mapping) or component not in value:
                raise ValueError(f"scalar projection path is missing mapping key at {traversed!r}")
            value = value[component]
        else:
            if not isinstance(value, list) or component < 0 or component >= len(value):
                raise ValueError(f"scalar projection path has invalid list index at {traversed!r}")
            value = value[component]
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError(
            f"scalar projection path {projection.path!r} resolved non-scalar {type(value).__name__}"
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"scalar projection path {projection.path!r} resolved a non-finite float")
    return ResolvedReportScalarProjection(
        value=value,
        input_ref=resolved_input.ref,
        product=product,
        artifact=artifact,
    )


def execute_report_spec(
    spec: ReportSpec | Mapping[str, Any] | Path | str,
    *,
    registry: ReportRecipeRegistry,
    root: Path | str | None = None,
    provenance: Provenance | None = None,
    issues: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    regeneration_specs: Sequence[SpecPayload | ParentRef | ArtifactRef] = (),
    execution_context: StagedExecutionContext = EMPTY_STAGED_EXECUTION_CONTEXT,
    execution_descriptor: StagedExecutionDescriptor | Mapping[str, Any] | None = None,
    artifact_provider_bindings: Sequence[StagedArtifactProviderRootBinding] = (),
    manifest_root_bindings: Sequence[StagedManifestRootBinding] = (),
    checkpoint_custody_bindings: Sequence[StagedCheckpointCustodyRootBinding] = (),
) -> tuple[ReportManifest, Path]:
    """Execute a serialized report spec and write a truthful manifest."""
    report_spec = coerce_report_spec(spec)
    recipe = registry.get(report_spec.report_type)
    root_path = Path(root) if root is not None else default_manifest_root()
    manifest_id = report_manifest_id(report_spec)
    explicit_runtime = (
        execution_descriptor is not None
        or bool(artifact_provider_bindings)
        or bool(manifest_root_bindings)
        or bool(checkpoint_custody_bindings)
    )
    if explicit_runtime:
        if execution_context is not EMPTY_STAGED_EXECUTION_CONTEXT:
            raise ValueError("execution_context cannot be combined with direct runtime bindings")
        execution_context = resolve_staged_execution_context(
            execution_descriptor,
            artifact_provider_bindings=artifact_provider_bindings,
            manifest_root_bindings=manifest_root_bindings,
            checkpoint_custody_bindings=checkpoint_custody_bindings,
        )
        authenticated_inputs = [
            ref for ref in report_spec.inputs if is_authenticated_manifest_ref(ref)
        ]
        if authenticated_inputs and (
            execution_context.manifest_roots or execution_context.opened_artifact_providers
        ):
            execution_context = with_staged_manifest_provider_inputs(
                execution_context,
                authenticated_inputs,
            )

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
    resolved_inputs = resolve_report_inputs(
        report_spec,
        root=root_path,
        execution_context=execution_context,
    )
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


def execute_authored_report_spec(
    spec: ReportSpec | Mapping[str, Any] | Path | str,
    *,
    registry: ReportRecipeRegistry,
    exact_parents: StagedExactParents | Mapping[str, Any],
    root: Path | str,
    execution_descriptor: StagedExecutionDescriptor | Mapping[str, Any] | None = None,
    artifact_provider_bindings: Sequence[StagedArtifactProviderRootBinding] = (),
    checkpoint_custody_bindings: Sequence[StagedCheckpointCustodyRootBinding] = (),
) -> tuple[ReportManifest, Path]:
    """Execute one authored report against authoritative exact staged parents.

    ``exact_parents`` is the complete runtime input membership. Every input
    authored in ``spec`` must occur byte-for-byte in that membership. Additional
    terminal parents are allowed only when they do not replace an authored input
    by role or ID.
    """
    report_spec = coerce_report_spec(spec)
    if isinstance(exact_parents, StagedExactParents):
        exact_payload = exact_parents.model_dump(mode="json")
    elif isinstance(exact_parents, Mapping):
        exact_payload = dict(exact_parents)
    else:
        raise TypeError("exact_parents must be StagedExactParents or a mapping")
    exact = migrate_staged_exact_parents(exact_payload)
    root_path = Path(root)

    material_entries = [
        entry.parent.id for entry in exact.parents if entry.material_dependencies is not None
    ]
    if material_entries:
        raise ValueError(
            "authored report execution cannot ignore StagedExactParents "
            "material_dependencies; route these parents through the shared staged "
            f"analysis bundle preflight first: parents={sorted(material_entries)!r}"
        )

    exact_refs = tuple(entry.parent for entry in exact.parents)
    _validate_authored_report_exact_parent_membership(report_spec.inputs, exact_refs)

    execution_context = resolve_staged_execution_context(
        execution_descriptor,
        artifact_provider_bindings=artifact_provider_bindings,
        checkpoint_custody_bindings=checkpoint_custody_bindings,
    )
    execution_context = with_staged_parent_execution_locations(
        execution_context,
        [
            StagedParentExecutionLocation(
                parent=entry.parent,
                root=root_path,
                execution_uri=entry.execution_uri,
            )
            for entry in exact.parents
        ],
    )
    for parent in exact_refs:
        if not is_authenticated_manifest_ref(parent):
            raise ValueError(
                f"exact report parent {parent.id!r} must be an authenticated manifest ref"
            )
        execution_context.resolve_manifest_input(parent)

    execution_spec = report_spec.model_copy(
        update={"inputs": list(exact_refs)},
        deep=True,
    )
    return execute_report_spec(
        execution_spec,
        registry=registry,
        root=root_path,
        execution_context=execution_context,
    )


def _validate_authored_report_exact_parent_membership(
    authored_inputs: Sequence[ParentRef],
    exact_parents: Sequence[ParentRef],
) -> None:
    """Require exact authored refs while allowing non-substituting terminal extensions."""
    exact_by_value = {
        parent.model_dump_json(exclude_none=False): parent for parent in exact_parents
    }
    if len(exact_by_value) != len(exact_parents):
        raise ValueError("StagedExactParents contains a duplicate complete ParentRef")
    exact_parent_ids = [parent.id for parent in exact_parents]
    if len(set(exact_parent_ids)) != len(exact_parent_ids):
        raise ValueError("StagedExactParents contains a duplicate ParentRef id")

    authored_values = {parent.model_dump_json(exclude_none=False) for parent in authored_inputs}
    for authored in authored_inputs:
        serialized = authored.model_dump_json(exclude_none=False)
        if serialized in exact_by_value:
            continue
        substitutions = [
            candidate
            for candidate in exact_parents
            if candidate.id == authored.id
            or (authored.role is not None and candidate.role == authored.role)
        ]
        if substitutions:
            raise ValueError(
                f"authored report input {authored.id!r} role {authored.role!r} must occur "
                "byte-identically in StagedExactParents; role/ID substitution is forbidden"
            )
        raise ValueError(
            f"authored report input {authored.id!r} role {authored.role!r} is absent "
            "from authoritative StagedExactParents"
        )

    authored = tuple(authored_inputs)
    for extension in exact_parents:
        if extension.model_dump_json(exclude_none=False) in authored_values:
            continue
        if any(
            extension.id == expected.id
            or (
                extension.role is not None
                and expected.role is not None
                and extension.role == expected.role
            )
            for expected in authored
        ):
            raise ValueError(
                f"terminal exact parent {extension.id!r} role {extension.role!r} "
                "conflicts with an authored report input by role or ID"
            )


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
    resolved_projections: dict[str, OrderedFigureReportScalar] = {}
    projection_artifacts: dict[str, ArtifactRef] = {}
    for section in params.sections:
        for table in section.tables:
            for row in table.rows:
                for cell in row:
                    for projection in _table_cell_projections(cell):
                        key = projection.model_dump_json(exclude_none=True)
                        if key in resolved_projections:
                            continue
                        resolved = resolve_report_scalar_projection(projection, inputs)
                        resolved_projections[key] = resolved.value
                        projection_artifacts[resolved.artifact.artifact_id] = resolved.artifact
    output_suffix = Path(params.output_name).suffix
    if output_suffix == ".html":
        rendered = render_ordered_figure_report_html(
            params,
            inputs_by_role,
            resolved_projections,
        )
        media_type = "text/html"
    else:
        rendered = render_ordered_figure_report_markdown(
            params,
            inputs_by_role,
            resolved_projections,
        )
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
                "scalar_projection_artifact_ids": sorted(projection_artifacts),
            }
        },
        regeneration_specs=list(projection_artifacts.values()),
    )


def _ordered_figure_inputs_by_role(
    params: OrderedFigureReportParams,
    inputs: Sequence[ResolvedReportInput],
) -> dict[str, FigureManifest]:
    expected_figure_roles = {
        str(figure.input_role)
        for section in params.sections
        if section.applicability == "included"
        for figure in section.figures
        if figure.applicability == "included"
    }
    expected_figure_spec_sha256 = {
        str(figure.input_role): str(figure.figure_spec_sha256)
        for section in params.sections
        if section.applicability == "included"
        for figure in section.figures
        if figure.applicability == "included"
    }
    expected_projection_roles = {
        projection.input_role
        for section in params.sections
        if section.applicability == "included"
        for table in section.tables
        for row in table.rows
        for cell in row
        for projection in _table_cell_projections(cell)
    }
    expected_roles = expected_figure_roles | expected_projection_roles
    resolved_by_role: dict[str, FigureManifest] = {}
    seen_roles: set[str] = set()
    for resolved in inputs:
        role = resolved.ref.role
        if role is None or not role.strip():
            raise ValueError(
                f"ordered figure report input {resolved.ref.id!r} requires a non-blank role"
            )
        if role in seen_roles:
            raise ValueError(f"ordered figure report input role {role!r} is duplicated")
        seen_roles.add(role)
        if role not in expected_roles:
            raise ValueError(f"ordered figure report input role {role!r} is not referenced")
        if role in expected_projection_roles:
            if resolved.ref.kind != "AnalysisRunManifest":
                raise ValueError(
                    f"ordered figure report projection input role {role!r} must "
                    "reference AnalysisRunManifest"
                )
            continue
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
        figure_spec_sha256 = _validated_figure_spec_sha256(
            resolved.manifest,
            input_role=role,
        )
        expected_sha256 = expected_figure_spec_sha256[role]
        if figure_spec_sha256 != expected_sha256:
            raise ValueError(
                f"ordered figure report input role {role!r} FigureSpec SHA-256 mismatch: "
                f"expected={expected_sha256!r}, computed={figure_spec_sha256!r}"
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
    missing_roles = sorted(expected_roles - seen_roles)
    if missing_roles:
        raise ValueError(
            "ordered figure report is missing required input roles: "
            + ", ".join(repr(role) for role in missing_roles)
        )
    return resolved_by_role


def _validated_figure_spec_sha256(
    manifest: FigureManifest,
    *,
    input_role: str,
) -> str:
    payload = manifest.figure_spec
    if payload.kind != "FigureSpec":
        raise ValueError(
            f"ordered figure report input role {input_role!r} has embedded spec kind "
            f"{payload.kind!r}; expected 'FigureSpec'"
        )
    if (
        payload.schema_id != FIGURE_SPEC_SCHEMA_ID
        or payload.schema_version != FIGURE_SPEC_SCHEMA_VERSION
    ):
        raise ValueError(
            f"ordered figure report input role {input_role!r} has unsupported embedded "
            f"FigureSpec schema {payload.schema_id!r}/{payload.schema_version!r}; expected "
            f"{FIGURE_SPEC_SCHEMA_ID!r}/{FIGURE_SPEC_SCHEMA_VERSION!r}"
        )
    try:
        FigureSpec.model_validate(payload.inline)
    except ValueError as exc:
        raise ValueError(
            f"ordered figure report input role {input_role!r} has malformed embedded FigureSpec"
        ) from exc
    return sha256_bytes(canonical_json_bytes(payload.inline))


def render_ordered_figure_report_markdown(
    params: OrderedFigureReportParams,
    inputs_by_role: Mapping[str, FigureManifest],
    resolved_projections: Mapping[str, OrderedFigureReportScalar],
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
                    + " | ".join(
                        _escape_markdown_cell(
                            _scalar_text(_resolved_table_cell(cell, resolved_projections))
                        )
                        for cell in row
                    )
                    + " |"
                )
            lines.append("")
    return "\n".join(lines)


def render_ordered_figure_report_html(
    params: OrderedFigureReportParams,
    inputs_by_role: Mapping[str, FigureManifest],
    resolved_projections: Mapping[str, OrderedFigureReportScalar],
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
                    *[f"        <th>{html.escape(column)}</th>" for column in table.columns],
                    "      </tr></thead>",
                    "      <tbody>",
                ]
            )
            for row in table.rows:
                lines.extend(
                    [
                        "        <tr>",
                        *[
                            "          <td>"
                            + html.escape(
                                _scalar_text(_resolved_table_cell(cell, resolved_projections))
                            )
                            + "</td>"
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


def _resolved_table_cell(
    cell: OrderedFigureReportTableCell,
    resolved_projections: Mapping[str, OrderedFigureReportScalar],
) -> OrderedFigureReportScalar:
    if isinstance(cell, OrderedFigureReportCompositeScalarCell):
        return _composite_scalar_text(cell, resolved_projections)
    if isinstance(cell, OrderedFigureReportScalarProjection):
        return resolved_projections[cell.model_dump_json(exclude_none=True)]
    return cell


def _table_cell_projections(
    cell: OrderedFigureReportTableCell,
) -> tuple[OrderedFigureReportScalarProjection, ...]:
    if isinstance(cell, OrderedFigureReportCompositeScalarCell):
        return tuple(cell.projections[name] for name in sorted(cell.projections))
    if isinstance(cell, OrderedFigureReportScalarProjection):
        return (cell,)
    return ()


def _composite_scalar_text(
    cell: OrderedFigureReportCompositeScalarCell,
    resolved_projections: Mapping[str, OrderedFigureReportScalar],
) -> str:
    values: dict[str, int | float] = {}
    for name, projection in cell.projections.items():
        value = resolved_projections[projection.model_dump_json(exclude_none=True)]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"composite scalar projection {name!r} resolved non-numeric {type(value).__name__}"
            )
        values[name] = value
    return cell.format.format_map(values)


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


def register_builtin_report_recipes(registry: ReportRecipeRegistry) -> None:
    """Seed one fresh registry with Feedbax report recipes."""
    registry.register(BUNDLE_SUMMARY_REPORT_TYPE, _bundle_summary_recipe)
    registry.register(STUDIO_REPORT_TYPE, _studio_report_recipe)
    registry.register(ORDERED_FIGURE_REPORT_TYPE, _ordered_figure_report_recipe)
