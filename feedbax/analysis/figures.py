"""Execution for declarative figure specs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
import json
from pathlib import Path
import re
from typing import Any

import plotly.graph_objs as go

from feedbax.analysis.execution_context import StagedExecutionContext
from feedbax.analysis.manifest_inputs import (
    is_authenticated_manifest_ref,
    resolve_manifest_input,
    resolve_contained_manifest_input,
)
from feedbax.contracts.expressions import (
    ContextItem,
    Coalesce,
    ExpressionContext,
    ExpressionEvaluationError,
    ExpressionPathMissing,
    ValueQuery,
    evaluate_expr,
    evaluate_query,
    expression_hash,
)
from feedbax.contracts.figures import FigureInputAuthority, FigureSpec, TraceBinding
from feedbax.contracts.manifest import (
    AnyManifest,
    EntrypointRef,
    FigureBindingRecord,
    FigureManifest,
    FigurePieceResolution,
    ManifestStatus,
    ParentRef,
    Provenance,
    ArtifactRef,
    canonical_json_bytes,
    collect_git_provenance,
    default_manifest_root,
    figure_manifest_id,
    media_type_for_extension,
    sha256_bytes,
    spec_payload,
    store_artifact,
    store_json_artifact,
    write_manifest,
)
from feedbax.plot.constructors import (
    FigureConstructorRegistration,
    PanelContent,
    get_figure_constructor,
    get_figure_piece,
    get_figure_template,
)
from feedbax.plot.io import save_figure_with_spec
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider

FIGURE_RENDER_ROLE = "figure_render"
FIGURE_SPEC_ROLE = "figure_spec"
FIGURE_MANIFEST_ROLE = "figure_manifest"
FIGURE_RENDER_MEDIA_TYPES = frozenset(
    media_type_for_extension(extension)
    for extension in ("json", "html", "png", "svg", "pdf")
)


@dataclass(frozen=True)
class ResolvedFigureInput:
    """Manifest parent resolved from a FigureSpec input ref."""

    ref: ParentRef
    manifest: AnyManifest | None
    path: Path | None
    artifact_payloads: Mapping[str, Any] = field(default_factory=dict)
    artifact_refs: tuple[ArtifactRef, ...] = ()


class FigureInputAuthorityError(ValueError):
    """Stable fail-closed error for figure input authority violations."""


@dataclass
class FigureExecutionTrace:
    """Mutable per-execution provenance accumulated before manifest write."""

    binding_records: list[FigureBindingRecord] = field(default_factory=list)
    constructor_versions: dict[str, str] = field(default_factory=dict)
    resolved_pieces: list[FigurePieceResolution] = field(default_factory=list)


@dataclass(frozen=True)
class TraceBindingPlan:
    """A trace binding paired with its template slot multiplicity."""

    binding: TraceBinding
    multiplicity: str


@dataclass(frozen=True)
class FacetCombination:
    """One deterministic Cartesian product entry for template facets."""

    values: dict[str, Any]
    payloads: dict[str, Any]
    label: str


@dataclass(frozen=True)
class RenderedFigure:
    """One rendered output within a FigureManifest execution."""

    name: str
    figure: go.Figure


class FigureSpecExecutionError(RuntimeError):
    """Raised after figure execution fails and a failed manifest is written."""

    def __init__(self, manifest: FigureManifest, path: Path, cause: BaseException):
        super().__init__(
            f"Figure spec for {manifest.id!r} failed; failed manifest written to {path}"
        )
        self.manifest = manifest
        self.path = path
        self.__cause__ = cause


def coerce_figure_spec(value: FigureSpec | Mapping[str, Any] | Path | str) -> FigureSpec:
    """Load a FigureSpec from an object, mapping, or JSON file path."""
    if isinstance(value, FigureSpec):
        return value
    if isinstance(value, Mapping):
        return FigureSpec.model_validate(value)
    path = Path(value)
    return FigureSpec.model_validate_json(path.read_text(encoding="utf-8"))


def resolve_figure_inputs(
    spec: FigureSpec,
    *,
    root: Path | str | None = None,
    execution_context: StagedExecutionContext | None = None,
) -> list[ResolvedFigureInput]:
    """Resolve and materialize every external figure input before effects."""
    root_path = Path(root) if root is not None else None
    resolved: list[ResolvedFigureInput] = []
    for ref in spec.inputs:
        manifest: AnyManifest | None = None
        manifest_path: Path | None = None
        if is_authenticated_manifest_ref(ref):
            if execution_context is not None:
                try:
                    authenticated = execution_context.resolve_manifest_input(ref)
                except Exception as exc:
                    raise FigureInputAuthorityError(
                        f"figure input authority rejected exact parent {ref.id!r}"
                    ) from exc
            elif root_path is not None:
                authenticated = resolve_manifest_input(ref, root_path)
            else:
                raise FigureInputAuthorityError(
                    f"figure input authority is missing for exact parent {ref.id!r}"
                )
            manifest, manifest_path = authenticated.manifest, authenticated.path
        elif ref.kind.endswith("Manifest"):
            if root_path is None or ref.uri is None:
                raise FigureInputAuthorityError(
                    f"figure input parent {ref.id!r} requires an authenticated reference "
                    "profile or explicit contained locator"
                )
            try:
                contained = resolve_contained_manifest_input(ref, root_path, ref.uri)
            except Exception as exc:
                raise FigureInputAuthorityError(
                    f"figure input authority rejected contained parent {ref.id!r}"
                ) from exc
            manifest, manifest_path = contained.manifest, contained.path
        authority = _authority_for_parent(spec, ref)
        payloads, artifact_refs = _resolve_authority_payloads(
            authority, manifest, execution_context
        )
        resolved.append(
            ResolvedFigureInput(
                ref=ref,
                manifest=manifest,
                path=manifest_path,
                artifact_payloads=payloads,
                artifact_refs=artifact_refs,
            )
        )
    return resolved


def _authority_for_parent(spec: FigureSpec, ref: ParentRef) -> FigureInputAuthority | None:
    matches = [authority for authority in spec.input_authorities if authority.parent == ref]
    if len(matches) > 1:
        raise FigureInputAuthorityError(
            f"figure input authority is ambiguous for exact parent {ref.id!r}"
        )
    return matches[0] if matches else None


def _resolve_authority_payloads(
    authority: FigureInputAuthority | None,
    manifest: AnyManifest | None,
    execution_context: StagedExecutionContext | None,
) -> tuple[dict[str, Any], tuple[ArtifactRef, ...]]:
    if authority is None or not authority.artifact_payloads:
        return {}, ()
    if manifest is None:
        raise FigureInputAuthorityError("figure artifact payload parent manifest is unavailable")
    if manifest.status != "completed":
        raise FigureInputAuthorityError("figure artifact payload parent status mismatch")
    resolved: dict[str, Any] = {}
    resolved_refs: list[ArtifactRef] = []
    for selector in authority.artifact_payloads:
        if authority.parent.role != selector.manifest_role:
            raise FigureInputAuthorityError(
                f"figure artifact payload manifest role mismatch for {selector.name!r}"
            )
        artifacts = list(manifest.artifacts)
        for data_product in getattr(manifest, "produced_data", []):
            artifacts.extend(data_product.artifacts)
        matches = [artifact for artifact in artifacts if artifact.role == selector.artifact_role]
        if not matches:
            raise FigureInputAuthorityError(
                f"figure artifact payload is missing for selector {selector.name!r}"
            )
        if len(matches) != 1:
            raise FigureInputAuthorityError(
                f"figure artifact payload is duplicated for selector {selector.name!r}"
            )
        artifact = matches[0]
        if artifact.media_type != selector.media_type:
            raise FigureInputAuthorityError(
                f"figure artifact payload media type mismatch for {selector.name!r}"
            )
        if execution_context is None:
            raise FigureInputAuthorityError(
                f"figure artifact provider binding is missing for {selector.name!r}"
            )
        try:
            provider = execution_context.artifact_provider(selector.artifact_provider)
            raw = provider.get_bytes(artifact)
        except Exception as exc:
            raise FigureInputAuthorityError(
                f"figure artifact provider rejected payload {selector.name!r}"
            ) from exc
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FigureInputAuthorityError(
                f"figure artifact payload is not valid JSON for {selector.name!r}"
            ) from exc
        if selector.payload_schema_id is not None and (
            not isinstance(payload, Mapping)
            or payload.get("schema_id") != selector.payload_schema_id
        ):
            raise FigureInputAuthorityError(
                f"figure artifact payload schema_id mismatch for {selector.name!r}"
            )
        if selector.payload_schema_version is not None and (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != selector.payload_schema_version
        ):
            raise FigureInputAuthorityError(
                f"figure artifact payload schema_version mismatch for {selector.name!r}"
            )
        resolved[selector.name] = payload
        resolved_refs.append(artifact)
    return resolved, tuple(resolved_refs)


def execute_figure_spec(
    spec: FigureSpec | Mapping[str, Any] | Path | str,
    *,
    root: Path | str | None = None,
    provenance: Provenance | None = None,
    issues: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    execution_context: StagedExecutionContext | None = None,
) -> tuple[FigureManifest, Path]:
    """Execute a declarative figure spec and write a FigureManifest."""
    figure_spec = coerce_figure_spec(spec)
    root_path = Path(root) if root is not None else default_manifest_root()
    manifest_id = figure_manifest_id(figure_spec)
    prov = provenance.model_copy(deep=True) if provenance is not None else collect_git_provenance()
    prov.parents = list(figure_spec.inputs)
    if issues:
        prov.issues.extend(issue for issue in issues if issue not in prov.issues)
    if prov.entrypoint is None:
        prov.entrypoint = EntrypointRef(kind="feedbax-figure-spec", name=figure_spec.name)

    exec_trace = FigureExecutionTrace()
    resolved_inputs = resolve_figure_inputs(
        figure_spec, root=root, execution_context=execution_context
    )
    try:
        context = _figure_expression_context(figure_spec, resolved_inputs)
        if figure_spec.run_condition is not None and not evaluate_expr(
            figure_spec.run_condition,
            context,
        ):
            manifest = _build_figure_manifest(
                manifest_id=manifest_id,
                figure_spec=figure_spec,
                status="completed",
                provenance=prov,
                artifacts=[],
                resolved_inputs=resolved_inputs,
                execution_trace=exec_trace,
                metadata={**dict(metadata or {}), "skipped": "run_condition false"},
            )
            return manifest, write_manifest(manifest, root=root_path)

        figures = _build_figures(figure_spec, context, exec_trace, root_path)
        artifacts = _write_figure_custody(
            figures,
            figure_spec,
            root=root_path,
            manifest_id=manifest_id,
        )
        manifest = _build_figure_manifest(
            manifest_id=manifest_id,
            figure_spec=figure_spec,
            status="completed",
            provenance=prov,
            artifacts=artifacts,
            resolved_inputs=resolved_inputs,
            execution_trace=exec_trace,
            metadata=dict(metadata or {}),
        )
        return manifest, write_manifest(manifest, root=root_path)
    except Exception as exc:
        manifest = _build_figure_manifest(
            manifest_id=manifest_id,
            figure_spec=figure_spec,
            status="failed",
            provenance=prov,
            artifacts=[],
            resolved_inputs=[],
            execution_trace=exec_trace,
            metadata=dict(metadata or {}),
            failure={"type": _failure_type(exc), "message": str(exc)},
        )
        path = write_manifest(manifest, root=root_path)
        raise FigureSpecExecutionError(manifest, path, exc) from exc


def _failure_type(exc: Exception) -> str:
    if isinstance(exc, ExpressionPathMissing):
        return "ExpressionPathMissing"
    return type(exc).__name__


def _figure_expression_context(
    spec: FigureSpec,
    resolved_inputs: Sequence[ResolvedFigureInput],
) -> ExpressionContext:
    payloads = []
    items: dict[str, ContextItem] = {
        "params": ContextItem(kind="params", payload=dict(spec.metadata)),
    }
    for resolved in resolved_inputs:
        manifest_payload = _manifest_payload(resolved.manifest, resolved.ref)
        if resolved.artifact_payloads:
            manifest_payload["artifact_payloads"] = dict(resolved.artifact_payloads)
        payloads.append(manifest_payload)
        role = resolved.ref.role
        if role:
            existing = items.get(role)
            if existing is None:
                items[role] = ContextItem(kind="parent_role", payload=manifest_payload)
            else:
                if not isinstance(existing.payload, list):
                    existing.payload = [existing.payload]
                existing.payload.append(manifest_payload)
        items[f"manifest:{resolved.ref.id}"] = ContextItem(
            kind="manifest",
            payload=manifest_payload,
        )
    items["manifests"] = ContextItem(kind="manifests", payload=payloads)
    if len(payloads) == 1:
        items["manifest"] = ContextItem(kind="manifest", payload=payloads[0])
    return ExpressionContext(items=items)


def _manifest_payload(manifest: AnyManifest | None, ref: ParentRef) -> dict[str, Any]:
    if manifest is None:
        return {"kind": ref.kind, "id": ref.id, "metadata": _portable_value(ref.metadata)}
    payload: dict[str, Any] = {
        "kind": manifest.kind,
        "id": manifest.id,
        "metadata": _portable_value(manifest.metadata),
        "artifacts": [
            artifact.model_dump(mode="json", exclude_none=True, exclude={"uri"})
            for artifact in manifest.artifacts
        ],
    }
    produced_data = getattr(manifest, "produced_data", None)
    if produced_data is not None:
        payload["produced_data"] = [
            _portable_value(product.model_dump(mode="json", exclude_none=True))
            for product in produced_data
        ]
    for attr in ("summary_metrics", "inputs"):
        if hasattr(manifest, attr):
            value = getattr(manifest, attr)
            if isinstance(value, list):
                payload[attr] = _portable_value([
                    item.model_dump(mode="json", exclude_none=True)
                    if hasattr(item, "model_dump")
                    else item
                    for item in value
                ])
            else:
                payload[attr] = _portable_value(value)
    return payload


_RUNTIME_CONTEXT_KEYS = frozenset(
    {"uri", "path", "root", "callback", "execution_context"}
)


def _portable_value(value: Any) -> Any:
    """Remove runtime locator/capability fields from constructor-visible payloads."""
    if isinstance(value, Mapping):
        return {
            key: _portable_value(item)
            for key, item in value.items()
            if key not in _RUNTIME_CONTEXT_KEYS
        }
    if isinstance(value, list):
        return [_portable_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_portable_value(item) for item in value)
    return value


def _route_assembler_params(
    assembler_params: Mapping[str, Any],
    *,
    panel_registration: FigureConstructorRegistration,
    figure_registration: FigureConstructorRegistration,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Route authored assembler params to their owning strict constructor models."""
    authored_fields = set(assembler_params)
    panel_fields = set(panel_registration.params_model.model_fields)
    figure_fields = set(figure_registration.params_model.model_fields)

    ambiguous_fields = sorted(authored_fields & panel_fields & figure_fields)
    if ambiguous_fields:
        raise ValueError(
            "FigureSpec assembler_params fields are owned by both the panel and figure "
            f"parameter models: {ambiguous_fields}"
        )

    unknown_fields = sorted(authored_fields - panel_fields - figure_fields)
    if unknown_fields:
        raise ValueError(
            "FigureSpec assembler_params fields are not owned by either the panel or figure "
            f"parameter model: {unknown_fields}"
        )

    return (
        {key: value for key, value in assembler_params.items() if key in panel_fields},
        {key: value for key, value in assembler_params.items() if key in figure_fields},
    )


def _build_figures(
    spec: FigureSpec,
    context: ExpressionContext,
    exec_trace: FigureExecutionTrace,
    root: Path,
) -> list[RenderedFigure]:
    template = get_figure_template(spec.template) if spec.template else None
    assembler_key = spec.assembler or template.assembler
    assembler_params = {
        **(getattr(template, "assembler_params", {}) if template is not None else {}),
        **spec.assembler_params,
    }

    facet_combinations = _facet_combinations(spec, template, context)
    custom_registration = get_figure_constructor(assembler_key)
    if custom_registration.tier == "custom_figure":
        exec_trace.constructor_versions[assembler_key] = custom_registration.version
        params = custom_registration.params(assembler_params)
        if facet_combinations and template.facet_target == "panels":
            raise ValueError(
                "facet_target='panels' requires registered trace/panel/figure constructors; "
                "custom_figure assemblers cannot expose panel composition"
            )
        combinations = facet_combinations or [
            FacetCombination(values={}, payloads={}, label="figure")
        ]
        return [
            RenderedFigure(
                name=combination.label,
                figure=custom_registration.callable(
                    _facet_context(context, combination).items,
                    params,
                ),  # type: ignore[misc]
            )
            for combination in combinations
        ]

    trace_bindings = _trace_bindings_for_spec(spec, template)
    panel_constructor_key = assembler_params.get("panel_constructor", "feedbax.comparison_grid")
    panel_registration = get_figure_constructor(panel_constructor_key, tier="panel")
    exec_trace.constructor_versions[panel_constructor_key] = panel_registration.version
    figure_registration = get_figure_constructor(assembler_key, tier="figure")
    exec_trace.constructor_versions[assembler_key] = figure_registration.version
    panel_values, figure_values = _route_assembler_params(
        assembler_params,
        panel_registration=panel_registration,
        figure_registration=figure_registration,
    )
    panel_params = panel_registration.params(panel_values)
    figure_params = figure_registration.params(figure_values)

    if facet_combinations and template.facet_target == "panels":
        panel_contents = _facet_panel_contents(
            spec,
            context,
            facet_combinations,
            trace_bindings,
            exec_trace,
            root,
        )
        figure = panel_registration.callable(panel_contents, panel_params)  # type: ignore[misc]
        return [
            RenderedFigure(
                name="figure",
                figure=figure_registration.callable(
                    figure,
                    panel_contents,
                    figure_params,
                ),  # type: ignore[misc]
            )
        ]

    combinations = facet_combinations or [FacetCombination(values={}, payloads={}, label="figure")]
    rendered: list[RenderedFigure] = []
    for combination in combinations:
        facet_context = _facet_context(context, combination)
        panel_contents = _panel_contents(
            spec,
            facet_context,
            trace_bindings,
            exec_trace,
            root,
            nonfacet_context=context,
        )
        figure = panel_registration.callable(panel_contents, panel_params)  # type: ignore[misc]
        rendered.append(
            RenderedFigure(
                name=combination.label,
                figure=figure_registration.callable(
                    figure,
                    panel_contents,
                    figure_params,
                ),  # type: ignore[misc]
            )
        )
    return rendered


def _facet_combinations(
    spec: FigureSpec,
    template: Any | None,
    context: ExpressionContext,
) -> list[FacetCombination]:
    dimensions = list(getattr(template, "facet_by", [])) if template is not None else []
    if not dimensions:
        if spec.facet_bindings:
            raise ValueError("FigureSpec facet_bindings require a template with facet_by")
        return []
    missing = [dimension for dimension in dimensions if dimension not in spec.facet_bindings]
    extra = sorted(set(spec.facet_bindings) - set(dimensions))
    if missing or extra:
        raise ValueError(
            "FigureSpec facet_bindings must exactly match template facet_by; "
            f"missing={missing}, extra={extra}"
        )

    enumerated: list[list[tuple[Any, Any]]] = []
    for dimension in dimensions:
        values = _evaluate_value(spec.facet_bindings[dimension], context)
        if isinstance(values, Mapping):
            entries = list(values.items())
        elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            entries = [(value, value) for value in values]
        else:
            raise ValueError(
                f"facet binding {dimension!r} must evaluate to a mapping or sequence, "
                f"got {type(values).__name__}"
            )
        if not entries:
            raise ValueError(f"facet binding {dimension!r} evaluated to no values")
        enumerated.append(entries)

    combinations = []
    for entries in product(*enumerated):
        mapping = {
            dimension: entry[0] for dimension, entry in zip(dimensions, entries, strict=True)
        }
        payloads = {
            dimension: entry[1] for dimension, entry in zip(dimensions, entries, strict=True)
        }
        label = ", ".join(f"{name}={value}" for name, value in mapping.items())
        combinations.append(FacetCombination(values=mapping, payloads=payloads, label=label))
    return combinations


def _facet_context(
    context: ExpressionContext,
    combination: FacetCombination,
) -> ExpressionContext:
    if not combination.values:
        return context
    facet_items = {
        name: ContextItem(kind="figure_facet", payload=value)
        for name, value in combination.values.items()
    }
    return ExpressionContext(
        items={
            **context.items,
            "facet": ContextItem(kind="figure_facet", payload=dict(combination.values)),
            "facet_values": ContextItem(
                kind="figure_facet_values",
                payload=dict(combination.payloads),
            ),
            **facet_items,
        }
    )


def _trace_bindings_for_spec(spec: FigureSpec, template: Any | None) -> list[TraceBindingPlan]:
    bindings: list[TraceBindingPlan] = []
    if template is not None:
        for slot in getattr(template, "slots", []):
            raw = spec.slot_bindings.get(slot.name)
            if raw is None:
                if slot.required:
                    raise ValueError(f"FigureSpec missing required template slot {slot.name!r}")
                continue
            slot_bindings = raw if isinstance(raw, list) else [raw]
            if slot.multiplicity in {"one", "per_facet"} and len(slot_bindings) != 1:
                raise ValueError(
                    f"Template slot {slot.name!r} with multiplicity {slot.multiplicity!r} "
                    "requires exactly one TraceBinding"
                )
            for binding in slot_bindings:
                merged_params = {**slot.params_defaults, **binding.params}
                bindings.append(
                    TraceBindingPlan(
                        binding=binding.model_copy(
                            update={
                                "constructor": binding.constructor or slot.constructor,
                                "params": merged_params,
                            }
                        ),
                        multiplicity=slot.multiplicity,
                    )
                )
    bindings.extend(
        TraceBindingPlan(binding=binding, multiplicity="many") for binding in spec.traces
    )
    piece_names = [
        *(getattr(template, "default_pieces", []) if template is not None else []),
        *spec.pieces,
    ]
    excluded = set(spec.exclude_pieces)
    for piece_name in piece_names:
        if piece_name in excluded:
            continue
        piece = get_figure_piece(piece_name)
        bindings.append(
            TraceBindingPlan(
                binding=TraceBinding(
                    name=piece.name,
                    constructor=piece.constructor,
                    piece=piece.name,
                    data={},
                    params={"label": piece.label, **dict(piece.style)},
                    required=False,
                ),
                multiplicity="many",
            )
        )
    return bindings


def _panel_contents(
    spec: FigureSpec,
    context: ExpressionContext,
    trace_bindings: Sequence[TraceBindingPlan],
    exec_trace: FigureExecutionTrace,
    root: Path,
    *,
    nonfacet_context: ExpressionContext | None = None,
) -> list[PanelContent]:
    panel_map = {panel.name: panel for panel in spec.panels}
    traces_by_panel: dict[str, list[Any]] = {name: [] for name in panel_map}
    if not traces_by_panel:
        traces_by_panel["main"] = []
    for plan in trace_bindings:
        binding = plan.binding
        panel_name = binding.panel or next(iter(traces_by_panel))
        traces = _execute_trace_binding(
            binding,
            context if plan.multiplicity == "per_facet" else (nonfacet_context or context),
            exec_trace,
            root,
            panel_name=panel_name,
        )
        traces_by_panel.setdefault(panel_name, []).extend(traces)
    contents: list[PanelContent] = []
    for index, (panel_name, traces) in enumerate(traces_by_panel.items()):
        panel = panel_map.get(panel_name)
        title = _evaluate_panel_title(panel.title, context) if panel is not None else panel_name
        axes_labels = panel.axes_labels.model_dump() if panel and panel.axes_labels else None
        x_axis = panel.x_axis.model_dump(exclude_none=True) if panel and panel.x_axis else None
        y_axis = panel.y_axis.model_dump(exclude_none=True) if panel and panel.y_axis else None
        contents.append(
            PanelContent(
                name=panel_name,
                traces=tuple(traces),
                title=title,
                row=panel.row if panel else index + 1,
                col=panel.col if panel else 1,
                axes_labels=axes_labels,
                x_axis=x_axis,
                y_axis=y_axis,
            )
        )
    return contents


def _facet_panel_contents(
    spec: FigureSpec,
    context: ExpressionContext,
    combinations: Sequence[FacetCombination],
    trace_bindings: Sequence[TraceBindingPlan],
    exec_trace: FigureExecutionTrace,
    root: Path,
) -> list[PanelContent]:
    base_panel = spec.panels[0] if spec.panels else None
    contents: list[PanelContent] = []
    for index, combination in enumerate(combinations):
        facet_context = _facet_context(context, combination)
        traces: list[Any] = []
        for plan in trace_bindings:
            traces.extend(
                _execute_trace_binding(
                    plan.binding,
                    facet_context if plan.multiplicity == "per_facet" else context,
                    exec_trace,
                    root,
                    panel_name=combination.label,
                )
            )
        title = (
            _evaluate_panel_title(base_panel.title, facet_context)
            if base_panel is not None and base_panel.title is not None
            else combination.label
        )
        contents.append(
            PanelContent(
                name=combination.label,
                traces=tuple(traces),
                title=title,
                row=index + 1,
                col=1,
                axes_labels=(
                    base_panel.axes_labels.model_dump()
                    if base_panel is not None and base_panel.axes_labels is not None
                    else None
                ),
                x_axis=(
                    base_panel.x_axis.model_dump(exclude_none=True)
                    if base_panel is not None and base_panel.x_axis is not None
                    else None
                ),
                y_axis=(
                    base_panel.y_axis.model_dump(exclude_none=True)
                    if base_panel is not None and base_panel.y_axis is not None
                    else None
                ),
            )
        )
    return contents


def _execute_trace_binding(
    binding: TraceBinding,
    context: ExpressionContext,
    exec_trace: FigureExecutionTrace,
    root: Path,
    *,
    panel_name: str | None = None,
) -> list[Any]:
    expression_hashes = _binding_expression_hashes(binding)
    if binding.include_when is not None and not evaluate_expr(binding.include_when, context):
        exec_trace.binding_records.append(
            FigureBindingRecord(
                name=binding.name,
                status="omitted",
                reason="include_when evaluated false",
                panel=panel_name or binding.panel,
                constructor=binding.constructor,
                expression_hashes=expression_hashes,
            )
        )
        return []
    try:
        data = _binding_data(binding, context, exec_trace, root)
    except ExpressionPathMissing as exc:
        if binding.required:
            exec_trace.binding_records.append(
                FigureBindingRecord(
                    name=binding.name,
                    status="failed",
                    reason=str(exc),
                    panel=panel_name or binding.panel,
                    constructor=binding.constructor,
                    expression_hashes=expression_hashes,
                )
            )
            raise
        exec_trace.binding_records.append(
            FigureBindingRecord(
                name=binding.name,
                status="omitted",
                reason=f"optional data binding absent: {exc}",
                panel=panel_name or binding.panel,
                constructor=binding.constructor,
                expression_hashes=expression_hashes,
            )
        )
        return []
    except ExpressionEvaluationError as exc:
        exec_trace.binding_records.append(
            FigureBindingRecord(
                name=binding.name,
                status="failed",
                reason=str(exc),
                panel=panel_name or binding.panel,
                constructor=binding.constructor,
                expression_hashes=expression_hashes,
            )
        )
        raise

    registration = get_figure_constructor(binding.constructor, tier="trace")
    exec_trace.constructor_versions[binding.constructor] = registration.version
    params = registration.params(binding.params)
    traces = list(registration.callable(data, params))  # type: ignore[misc]
    exec_trace.binding_records.append(
        FigureBindingRecord(
            name=binding.name,
            status="included",
            panel=panel_name or binding.panel,
            constructor=binding.constructor,
            expression_hashes=expression_hashes,
        )
    )
    return traces


def _binding_data(
    binding: TraceBinding,
    context: ExpressionContext,
    exec_trace: FigureExecutionTrace,
    root: Path,
) -> dict[str, Any]:
    data = {name: _evaluate_value(value, context) for name, value in binding.data.items()}
    if binding.piece is not None:
        piece = get_figure_piece(binding.piece)
        data = {**_piece_data(piece, root), **data}
        data.setdefault("label", piece.label)
        data.update({key: value for key, value in piece.style.items() if key not in data})
        exec_trace.resolved_pieces.append(_piece_resolution(piece))
    return data


def _piece_data(piece: Any, root: Path) -> dict[str, Any]:
    if piece.artifact_ref is None or piece.artifact_ref.uri is None:
        return {}
    if piece.artifact_ref.media_type != "application/json":
        return {}
    if piece.artifact_ref.uri.startswith("artifact://sha256/"):
        raw = ImmutableArtifactBlobProvider(root.absolute()).get_bytes(piece.artifact_ref)
        payload = json.loads(raw)
    else:
        path = Path(piece.artifact_ref.uri)
        if not path.is_absolute():
            path = root / path
        payload = json.loads(path.read_text(encoding="utf-8"))
    if piece.data_path:
        payload = _get_payload_path(payload, piece.data_path)
    if not isinstance(payload, dict):
        raise ValueError(
            f"FigurePiece {piece.name!r} data_path must resolve to a mapping, "
            f"got {type(payload).__name__}"
        )
    return dict(payload)


def _get_payload_path(payload: Any, path: str) -> Any:
    current = payload
    for part in path.split("."):
        if isinstance(current, list):
            current = current[int(part)]
        elif isinstance(current, dict):
            current = current[part]
        else:
            current = getattr(current, part)
    return current


def _evaluate_value(value: Any, context: ExpressionContext) -> Any:
    if isinstance(value, (ValueQuery, Coalesce)):
        return evaluate_query(value, context)
    if isinstance(value, Mapping):
        if "queries" in value:
            return evaluate_query(Coalesce.model_validate(value), context)
        if "item" in value:
            return evaluate_query(ValueQuery.model_validate(value), context)
        return {key: _evaluate_value(child, context) for key, child in value.items()}
    if isinstance(value, list):
        return [_evaluate_value(child, context) for child in value]
    return value


def _binding_expression_hashes(binding: TraceBinding) -> list[str]:
    hashes: list[str] = []
    if binding.include_when is not None:
        hashes.append(expression_hash(binding.include_when))
    for value in binding.data.values():
        if isinstance(value, (ValueQuery, Coalesce)):
            hashes.append(expression_hash(value))
        elif isinstance(value, Mapping):
            try:
                if "queries" in value:
                    hashes.append(expression_hash(Coalesce.model_validate(value)))
                elif "item" in value:
                    hashes.append(expression_hash(ValueQuery.model_validate(value)))
            except Exception:
                continue
    return hashes


def _evaluate_panel_title(value: Any, context: ExpressionContext) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(_evaluate_value(value, context))


def _piece_resolution(piece: Any) -> FigurePieceResolution:
    if piece.artifact_ref is not None:
        return FigurePieceResolution(
            name=piece.name,
            source_kind="artifact_ref",
            artifact_refs=[piece.artifact_ref],
        )
    if piece.manifest_predicate is not None:
        return FigurePieceResolution(
            name=piece.name,
            source_kind="manifest_predicate",
            metadata={"predicate": piece.manifest_predicate.model_dump(mode="json")},
        )
    return FigurePieceResolution(
        name=piece.name,
        source_kind="generator_spec",
        metadata={"generator_spec": piece.generator_spec.model_dump(mode="json")},
    )


def _write_figure_custody(
    figures: Sequence[RenderedFigure],
    spec: FigureSpec,
    *,
    root: Path,
    manifest_id: str,
) -> list[ArtifactRef]:
    output_dir = root / "figures" / manifest_id.replace(":", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    render_format = str(spec.figure_routing.get("render_format", "json"))
    artifacts: list[ArtifactRef] = []
    for index, rendered in enumerate(figures):
        suffix = "" if len(figures) == 1 else f"-{_safe_figure_name(rendered.name, index)}"
        output_name = f"figure{suffix}"
        spec_path, render_path = save_figure_with_spec(
            rendered.figure,
            {
                "kind": "FigureSpec",
                "figure_spec": spec.model_dump(mode="json", exclude_none=True),
                "facet": rendered.name if len(figures) > 1 else None,
            },
            output_dir,
            name=output_name,
            render_format=render_format,
        )
        artifacts.append(
            store_artifact(
                spec_path,
                root=root,
                role=FIGURE_SPEC_ROLE,
                logical_name=f"{spec.name}{suffix}-spec.json",
                media_type="application/json",
                metadata={
                    "figure": spec.name,
                    "facet": rendered.name if len(figures) > 1 else None,
                },
            )
        )
        if render_path is not None:
            media_type = media_type_for_extension(render_path.suffix)
            artifacts.append(
                store_artifact(
                    render_path,
                    root=root,
                    role=FIGURE_RENDER_ROLE,
                    logical_name=render_path.name,
                    media_type=media_type,
                    metadata={
                        "figure": spec.name,
                        "facet": rendered.name if len(figures) > 1 else None,
                        "format": render_format,
                    },
                )
            )
        artifacts.append(
            store_json_artifact(
                rendered.figure.to_plotly_json(),
                root=root,
                role=FIGURE_RENDER_ROLE,
                logical_name=f"{spec.name}{suffix}.plotly.json",
                metadata={
                    "figure": spec.name,
                    "facet": rendered.name if len(figures) > 1 else None,
                    "format": "plotly-json",
                },
            )
        )
    return artifacts


def _safe_figure_name(value: str, index: int) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-").lower()
    return normalized or f"facet-{index + 1}"


def _build_figure_manifest(
    *,
    manifest_id: str,
    figure_spec: FigureSpec,
    status: ManifestStatus,
    provenance: Provenance,
    artifacts: list[ArtifactRef],
    resolved_inputs: Sequence[ResolvedFigureInput],
    execution_trace: FigureExecutionTrace,
    metadata: dict[str, Any],
    failure: dict[str, Any] | None = None,
) -> FigureManifest:
    binding_payload = [
        record.model_dump(mode="json", exclude_none=True)
        for record in execution_trace.binding_records
    ]
    return FigureManifest(
        id=manifest_id,
        status=status,
        figure_spec=spec_payload(
            "FigureSpec",
            figure_spec.model_dump(mode="json", exclude_none=True),
        ),
        inputs=list(figure_spec.inputs),
        resolved_inputs=[item.ref for item in resolved_inputs if item.manifest is not None],
        resolved_pieces=list(execution_trace.resolved_pieces),
        constructor_versions=dict(execution_trace.constructor_versions),
        template_name=figure_spec.template,
        binding_records=list(execution_trace.binding_records),
        expression_results_digest=sha256_bytes(canonical_json_bytes(binding_payload)),
        provenance=provenance,
        artifacts=artifacts,
        regeneration_specs=[
            artifact
            for resolved in resolved_inputs
            for artifact in resolved.artifact_refs
        ],
        failure=failure,
        metadata=metadata,
    )


def figure_manifest_plotly_json(manifest: FigureManifest) -> dict[str, Any] | None:
    """Load the Plotly JSON artifact for a FigureManifest when available."""
    for artifact in manifest.artifacts:
        if artifact.role != FIGURE_RENDER_ROLE:
            continue
        if artifact.media_type != "application/json" or artifact.uri is None:
            continue
        import json

        return json.loads(Path(artifact.uri).read_text(encoding="utf-8"))
    return None
