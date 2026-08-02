"""Execution for declarative figure specs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
from itertools import product
import json
from pathlib import Path
import re
from typing import Any

import plotly.graph_objs as go
from pydantic import ValidationError

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
from feedbax.contracts.figures import (
    FIGURE_COMPOSITION_SPEC_SCHEMA_ID,
    FIGURE_PANEL_SCHEMA_ID,
    FIGURE_PANEL_SCHEMA_VERSION,
    FIGURE_SPEC_SCHEMA_ID,
    FIGURE_SPEC_SCHEMA_VERSION,
    FigureColorbar,
    FigureCompositionDocument,
    FigureCompositionLayer,
    FigureCompositionProvenance,
    FigureCompositionSpec,
    FigureDataProductArtifactPayload,
    FigureInputAuthority,
    FigureInputAuthoritySpec,
    FigureRuntimeArtifactProviderBinding,
    FigureRuntimeBindingSpec,
    FigureSpec,
    FigureTemplate,
    PanelSpec,
    ResolvedFigureSpec,
    TraceBinding,
    TraceFamily,
    figure_composition_payload_identity_sha256,
    substitute_index,
)
from feedbax.contracts.matrix_core import (
    SOURCE_DOCUMENT_INHERITANCE_KEY,
    SourceDocumentInheritance,
    load_content_pinned_json_document,
    materialize_inherited_document_with_custody,
)
from feedbax.contracts.migrations import default_spec_registry
from feedbax.contracts.run_matrix import apply_composition_deltas
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
    SpecPayload,
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
from feedbax.plot.colors import sample_colorscale_at, sample_colorscale_unique
from feedbax.plot.constructors import (
    FigureConstructorRegistration,
    FigureRegistry,
    PanelContent,
    get_figure_constructor,
    get_figure_piece,
    get_figure_template,
)
from feedbax.plot.io import save_figure_with_spec
from feedbax.persistence.artifact_custody import (
    ArtifactBlobIntegrityError,
    ImmutableArtifactBlobProvider,
)

FIGURE_RENDER_ROLE = "figure_render"
FIGURE_SPEC_ROLE = "figure_spec"
FIGURE_MANIFEST_ROLE = "figure_manifest"
FIGURE_RENDER_MEDIA_TYPES = frozenset(
    media_type_for_extension(extension) for extension in ("json", "html", "png", "svg", "pdf")
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
class TraceFamilyMember:
    """One expanded trace-family member and the color its index was given.

    ``value`` is the physical quantity the member stands for when its family
    declares values, and ``None`` otherwise. It is what positioned the member's
    color, and what a value-keyed colorbar reads instead of the index.
    """

    index: int | str
    color: str | None
    binding: TraceBinding
    value: float | None = None


@dataclass(frozen=True)
class TraceFamilyExpansion:
    """The deterministic expansion of one declared trace family.

    ``members`` is the family's color key in index order. A colorbar
    declaration composes with a family by reading these index/color pairs
    instead of re-sampling the colorscale, so the key and the traces cannot
    disagree.
    """

    family: TraceFamily
    members: tuple[TraceFamilyMember, ...]


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


FigureSpecInput = (
    FigureSpec | FigureCompositionSpec | ResolvedFigureSpec | Mapping[str, Any] | Path | str
)
MAX_FIGURE_COMPOSITION_DEPTH = 64


def _figure_authored_mapping(value: FigureSpecInput) -> dict[str, Any]:
    """Load one direct or composed figure document without changing its bytes."""
    if isinstance(value, FigureSpec):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, FigureCompositionSpec):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, ResolvedFigureSpec):
        return deepcopy(value.authored)
    if isinstance(value, Mapping):
        payload = deepcopy(dict(value))
        _require_durable_figure_identity(payload, label="figure mapping")
        return payload
    payload = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("figure document must contain a JSON object")
    _require_durable_figure_identity(payload, label=f"figure document {str(value)!r}")
    return payload


def _require_durable_figure_identity(payload: Mapping[str, Any], *, label: str) -> None:
    missing = [field for field in ("schema_id", "schema_version") if field not in payload]
    if missing:
        raise ValueError(f"{label} must explicitly declare {', '.join(missing)}")


def _figure_authored_ref(value: FigureSpecInput) -> str:
    if isinstance(value, (str, Path)):
        return str(value)
    return "<inline>"


def figure_composition_envelope(spec: FigureCompositionSpec) -> dict[str, Any]:
    """Return the canonical authored identity envelope, excluding the parent locator."""
    return _raw_figure_composition_envelope(
        spec.model_dump(mode="json", exclude_none=True)
    )


def figure_composition_envelope_hash(spec: FigureCompositionSpec) -> str:
    """Return deterministic authored identity for one composition envelope."""
    return spec.identity_sha256()


def _raw_figure_composition_envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return identity bytes from one exact authored v1 or v2 envelope."""
    parent = deepcopy(dict(payload["parent"]))
    parent.pop("ref", None)
    return {
        "schema_id": payload["schema_id"],
        "schema_version": payload["schema_version"],
        "parent": parent,
        "deltas": deepcopy(payload["deltas"]),
    }


def _raw_figure_composition_envelope_hash(payload: Mapping[str, Any]) -> str:
    return figure_composition_payload_identity_sha256(payload)


def _current_figure_spec(payload: Mapping[str, Any]) -> FigureSpec:
    if payload.get("schema_id", FIGURE_SPEC_SCHEMA_ID) != FIGURE_SPEC_SCHEMA_ID:
        raise ValueError(
            "figure composition parent must declare schema_id "
            f"{FIGURE_SPEC_SCHEMA_ID!r}, got {payload.get('schema_id')!r}"
        )
    migrated = default_spec_registry.migrate(
        "FigureSpec",
        payload,
        assume_current="schema_version" not in payload,
    ).payload
    return FigureSpec.model_validate(migrated)


def _current_figure_composition_spec(
    payload: Mapping[str, Any],
) -> FigureCompositionSpec:
    migrated = default_spec_registry.migrate("FigureCompositionSpec", payload).payload
    return FigureCompositionSpec.model_validate(migrated)


def resolve_figure_spec(
    value: FigureSpecInput,
    *,
    repo_root: Path | str | None = None,
    registry: FigureRegistry | None = None,
) -> ResolvedFigureSpec:
    """Resolve direct or composed authoring to one ordinary current FigureSpec.

    The returned authored identity covers the direct document or composition
    envelope. The resolved identity covers only the ordinary FigureSpec bytes
    consumed by display and execution. Composition locators and lineage never
    enter the resolved semantic hash.
    """
    if isinstance(value, ResolvedFigureSpec):
        resolved_payload = value.figure_spec.model_dump(mode="json", exclude_none=True)
        if sha256_bytes(canonical_json_bytes(resolved_payload)) != value.resolved_identity_sha256:
            raise ValueError("ResolvedFigureSpec resolved identity disagrees with FigureSpec")
        _validate_resolved_template(value.figure_spec, registry)
        return value
    authored = _figure_authored_mapping(value)
    authored_schema_id = str(authored.get("schema_id", FIGURE_SPEC_SCHEMA_ID))
    if authored_schema_id != FIGURE_COMPOSITION_SPEC_SCHEMA_ID:
        figure_spec = _current_figure_spec(authored)
        authored_identity = sha256_bytes(canonical_json_bytes(authored))
        resolved_identity = sha256_bytes(
            canonical_json_bytes(figure_spec.model_dump(mode="json", exclude_none=True))
        )
        _validate_resolved_template(figure_spec, registry)
        return ResolvedFigureSpec(
            authored=authored,
            authored_schema_id=authored_schema_id,
            authored_identity_sha256=authored_identity,
            resolved_identity_sha256=resolved_identity,
            figure_spec=figure_spec,
        )

    spec = _current_figure_composition_spec(authored)
    chain: list[tuple[str, FigureCompositionSpec, dict[str, Any]]] = []
    loaded_documents: list[tuple[dict[str, Any], dict[str, Any], Any]] = []
    first_seen: dict[str, int] = {}
    parent_first_seen: dict[tuple[str, tuple[str, ...] | None], int] = {}
    current = spec
    current_authored = authored
    while True:
        if len(chain) >= MAX_FIGURE_COMPOSITION_DEPTH:
            path = " -> ".join(node.parent.ref for _, node, _ in chain)
            raise ValueError(
                f"figure composition exceeds maximum depth {MAX_FIGURE_COMPOSITION_DEPTH}: {path}"
            )
        digest = _raw_figure_composition_envelope_hash(current_authored)
        if digest in first_seen:
            cycle_start = first_seen[digest]
            cycle = [node.parent.ref for _, node, _ in chain[cycle_start:]]
            raise ValueError(f"figure composition cycle detected: {' -> '.join(cycle)}")
        first_seen[digest] = len(chain)
        chain.append((digest, current, current_authored))
        parent_key = (current.parent.ref, current.parent.payload_path)
        if parent_key in parent_first_seen:
            cycle_start = parent_first_seen[parent_key]
            cycle = [node.parent.ref for _, node, _ in chain[cycle_start:]]
            raise ValueError(f"figure composition cycle detected: {' -> '.join(cycle)}")
        parent_first_seen[parent_key] = len(chain) - 1
        parent_document, parent_payload = load_content_pinned_json_document(
            current.parent, repo_root=repo_root
        )
        _require_durable_figure_identity(
            parent_payload,
            label=f"figure composition parent {current.parent.ref!r}",
        )
        loaded_documents.append((parent_document, parent_payload, current.parent))
        parent_schema_id = parent_payload.get("schema_id", FIGURE_SPEC_SCHEMA_ID)
        if parent_schema_id == FIGURE_COMPOSITION_SPEC_SCHEMA_ID:
            current_authored = deepcopy(parent_payload)
            current = _current_figure_composition_spec(parent_payload)
            continue
        if parent_schema_id != FIGURE_SPEC_SCHEMA_ID:
            raise ValueError(
                "figure composition parent must declare schema_id "
                f"{FIGURE_SPEC_SCHEMA_ID!r} or {FIGURE_COMPOSITION_SPEC_SCHEMA_ID!r}, "
                f"got {parent_schema_id!r} at {current.parent.ref!r}"
            )
        source_inheritance = None
        inherited_source_documents: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []
        if SOURCE_DOCUMENT_INHERITANCE_KEY in parent_payload:
            source_inheritance = SourceDocumentInheritance.model_validate(
                parent_payload[SOURCE_DOCUMENT_INHERITANCE_KEY]
            )
            parent_payload, inherited_source_documents = (
                materialize_inherited_document_with_custody(
                parent_payload, repo_root=repo_root
                )
            )
            parent_payload.pop(SOURCE_DOCUMENT_INHERITANCE_KEY, None)
        payload = _current_figure_spec(parent_payload).model_dump(mode="json", exclude_none=True)
        root_parent = current.parent
        break

    attribution: dict[str, str] = {}
    written: set[str] = set()
    layers: list[FigureCompositionLayer] = []
    for digest, node, _raw_node in reversed(chain):
        local_attribution: dict[str, str] = {}
        for delta in node.deltas:
            before = deepcopy(payload)
            declared = {
                addition.path: (addition.schema_id, addition.schema_version)
                for addition in (delta.same_schema_structural_additions or [])
            }
            payload, delta_attribution, written = apply_composition_deltas(
                payload,
                [delta.matrix_delta()],
                ancestor_written_paths=written,
                allowed_same_schema_structural_additions_by_layer={
                    delta.layer_id: declared
                },
            )
            added = _figure_added_structural_identities(before, payload)
            if declared != added:
                missing = {
                    path: identity
                    for path, identity in added.items()
                    if declared.get(path) != identity
                }
                invalid = {
                    path: identity
                    for path, identity in declared.items()
                    if added.get(path) != identity
                }
                raise ValueError(
                    f"/deltas/{delta.layer_id} same-schema structural additions must "
                    "exactly qualify the complete added typed structure; "
                    f"unqualified_or_mismatched={missing!r}, "
                    f"declared_but_absent_or_invalid={invalid!r}"
                )
            local_attribution.update(delta_attribution)
        qualified = {
            path: f"{digest}:{layer_id}" for path, layer_id in local_attribution.items()
        }
        attribution.update(qualified)
        layer_ids = [delta.layer_id for delta in node.deltas]
        layers.append(
            FigureCompositionLayer(
                envelope_sha256=digest,
                parent_ref=node.parent.ref,
                parent_sha256=node.parent.sha256,
                layer_ids=layer_ids,
                qualified_layer_ids=[f"{digest}:{layer_id}" for layer_id in layer_ids],
                parent_payload_path=node.parent.payload_path,
            )
        )
    figure_spec = FigureSpec.model_validate(payload)
    resolved_payload = figure_spec.model_dump(mode="json", exclude_none=True)
    resolved_identity = sha256_bytes(canonical_json_bytes(resolved_payload))
    custody_documents: list[FigureCompositionDocument] = []
    for order, (document, selected, parent) in enumerate(reversed(loaded_documents)):
        custody_documents.append(
            FigureCompositionDocument(
                order=order,
                role=(
                    "root_figure"
                    if selected.get("schema_id") == FIGURE_SPEC_SCHEMA_ID
                    else "composition_envelope"
                ),
                ref=parent.ref,
                sha256=sha256_bytes(canonical_json_bytes(document)),
                payload_path=parent.payload_path,
                selected_sha256=sha256_bytes(canonical_json_bytes(selected)),
                inline=document,
                selected_inline=selected,
            )
        )
    custody_documents.append(
        FigureCompositionDocument(
            order=len(custody_documents),
            role="authored_leaf",
            ref=_figure_authored_ref(value),
            sha256=sha256_bytes(canonical_json_bytes(authored)),
            selected_sha256=sha256_bytes(canonical_json_bytes(authored)),
            inline=authored,
            selected_inline=authored,
        )
    )
    inherited_custody: list[FigureCompositionDocument] = []
    inherited_seen: set[tuple[str, str, tuple[str, ...] | None]] = set()
    for parent, document, selected in inherited_source_documents:
        key = (parent.ref, parent.sha256, parent.payload_path)
        if key in inherited_seen:
            continue
        inherited_seen.add(key)
        inherited_custody.append(
            FigureCompositionDocument(
                order=len(inherited_custody),
                role="inherited_source",
                ref=parent.ref,
                sha256=sha256_bytes(canonical_json_bytes(document)),
                payload_path=parent.payload_path,
                selected_sha256=sha256_bytes(canonical_json_bytes(selected)),
                inline=document,
                selected_inline=selected,
            )
        )
    provenance = FigureCompositionProvenance(
        authored_envelope_sha256=_raw_figure_composition_envelope_hash(authored),
        root_figure=root_parent,
        layers=layers,
        documents=custody_documents,
        inherited_documents=inherited_custody,
        attribution=attribution,
        resolved_figure_spec_sha256=resolved_identity,
        source_inheritance=source_inheritance,
    )
    _validate_resolved_template(figure_spec, registry)
    return ResolvedFigureSpec(
        authored=authored,
        authored_schema_id=authored_schema_id,
        authored_identity_sha256=provenance.authored_envelope_sha256,
        resolved_identity_sha256=resolved_identity,
        figure_spec=figure_spec,
        composition=provenance,
    )


def _figure_added_structural_identities(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, tuple[str, str]]:
    """Return every typed path introduced by one figure delta."""
    before_identities = _figure_structural_identities(before)
    after_identities = _figure_structural_identities(after)
    return {
        path: identity
        for path, identity in after_identities.items()
        if path not in before_identities
    }


def _figure_structural_identities(
    payload: dict[str, Any],
) -> dict[str, tuple[str, str]]:
    identities: dict[str, tuple[str, str]] = {}

    def visit(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            if isinstance(value.get("schema_id"), str) and isinstance(
                value.get("schema_version"), str
            ):
                identities[path or "/"] = (
                    value["schema_id"],
                    value["schema_version"],
                )
            if (
                value.get("schema_id") == FIGURE_SPEC_SCHEMA_ID
                and value.get("schema_version") == FIGURE_SPEC_SCHEMA_VERSION
            ):
                panels = value.get("panels", [])
                if not isinstance(panels, list):
                    raise ValueError("FigureSpec panels must remain a list during composition")
                panel_prefix = f"{path}.panels" if path else "panels"
                for index, panel in enumerate(panels):
                    try:
                        PanelSpec.model_validate(panel)
                    except ValidationError as error:
                        raise ValueError(
                            f"same-schema structural addition "
                            f"'{panel_prefix}.{index}' is not a valid PanelSpec: {error}"
                        ) from error
                    identities[f"{panel_prefix}.{index}"] = (
                        FIGURE_PANEL_SCHEMA_ID,
                        FIGURE_PANEL_SCHEMA_VERSION,
                    )
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}.{index}" if path else str(index))

    visit(payload)
    return identities


def _validate_resolved_template(
    spec: FigureSpec,
    registry: FigureRegistry | None,
) -> None:
    if spec.template is None or registry is None:
        return
    try:
        get_figure_template(spec.template, registry=registry)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"resolved FigureSpec names unknown template {spec.template!r}") from exc


def coerce_figure_spec(
    value: FigureSpecInput,
    *,
    repo_root: Path | str | None = None,
    registry: FigureRegistry | None = None,
) -> FigureSpec:
    """Resolve any supported authoring root to one ordinary current FigureSpec."""
    return resolve_figure_spec(value, repo_root=repo_root, registry=registry).figure_spec


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
            authority, ref, manifest, execution_context
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


def _authority_for_parent(spec: FigureSpec, ref: ParentRef) -> FigureInputAuthoritySpec | None:
    matches = [
        authority
        for authority in spec.input_authorities
        if (
            authority.parent
            if isinstance(authority, FigureInputAuthority)
            else authority.resolve_parent(spec.inputs)
        )
        == ref
    ]
    if len(matches) > 1:
        raise FigureInputAuthorityError(
            f"figure input authority is ambiguous for exact parent {ref.id!r}"
        )
    return matches[0] if matches else None


def _resolve_authority_payloads(
    authority: FigureInputAuthoritySpec | None,
    parent: ParentRef,
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
        if parent.role != selector.manifest_role:
            raise FigureInputAuthorityError(
                f"figure artifact payload manifest role mismatch for {selector.name!r}"
            )
        if isinstance(selector, FigureDataProductArtifactPayload):
            products = [
                product
                for product in getattr(manifest, "produced_data", [])
                if product.role == selector.product_role
            ]
            if not products:
                raise FigureInputAuthorityError(
                    f"figure data product is missing for selector {selector.name!r}"
                )
            if len(products) != 1:
                raise FigureInputAuthorityError(
                    f"figure data product is duplicated for selector {selector.name!r}"
                )
            product = products[0]
            if (
                product.product_schema_id != selector.product_schema_id
                or product.product_schema_version != selector.product_schema_version
            ):
                raise FigureInputAuthorityError(
                    f"figure data product schema mismatch for selector {selector.name!r}"
                )
            artifacts = list(product.artifacts)
        else:
            # The v1 selector deliberately covers the manifest's historical
            # top-level-plus-produced artifact namespace. Typed product
            # selection uses FigureDataProductArtifactPayload instead.
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
            provider = execution_context.artifact_provider_for_parent(
                parent,
                selector.artifact_provider,
            )
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


@dataclass(frozen=True)
class FigureExecutionPlan:
    """Everything a figure execution derives before any effect happens.

    `FigureManifest` identity is minted from the resolved authored spec, while
    runtime inputs, authorities, and artifact-provider bindings are overlaid
    afterwards and recorded separately. Cache admission therefore needs the
    same derivation the executor uses, so the derivation lives here rather than
    inside :func:`execute_figure_spec`.
    """

    resolution: ResolvedFigureSpec
    resolved_spec: FigureSpec
    execution_spec: FigureSpec
    manifest_id: str
    authored_payload: SpecPayload
    composition_payloads: tuple[SpecPayload, ...]
    runtime_binding_payload: SpecPayload | None
    runtime_overlay: bool


def plan_figure_execution(
    spec: FigureSpecInput,
    *,
    runtime_inputs: Sequence[ParentRef] | None = None,
    runtime_input_authorities: Sequence[FigureInputAuthoritySpec] | None = None,
    runtime_metadata: Mapping[str, Any] | None = None,
    repo_root: Path | str | None = None,
    execution_context: StagedExecutionContext | None = None,
    registry: FigureRegistry,
) -> FigureExecutionPlan:
    """Derive one figure's identity, embedded spec, and runtime binding record.

    This is an internal Feedbax helper shared by figure execution and figure
    receipt admission. It performs no filesystem effects.
    """
    resolution = resolve_figure_spec(spec, repo_root=repo_root, registry=registry)
    resolved_spec = resolution.figure_spec
    runtime_update: dict[str, Any] = {}
    if runtime_inputs is not None:
        runtime_update["inputs"] = list(runtime_inputs)
    if runtime_input_authorities is not None:
        runtime_update["input_authorities"] = list(runtime_input_authorities)
    if runtime_metadata is not None:
        runtime_update["metadata"] = {
            **resolved_spec.metadata,
            **dict(runtime_metadata),
        }
    figure_spec = (
        resolved_spec.model_copy(update=runtime_update, deep=True)
        if runtime_update
        else resolved_spec
    )
    # Revalidate runtime overlays as one complete FigureSpec before any effects.
    if runtime_update:
        try:
            figure_spec = FigureSpec.model_validate(
                figure_spec.model_dump(mode="python", exclude_none=False)
            )
        except ValidationError as exc:
            raise FigureInputAuthorityError(
                "figure runtime input/authority binding is invalid"
            ) from exc
    semantic_payload = (
        resolved_spec.model_dump(mode="json", exclude_none=True)
        if resolution.composition is not None
        else resolution.authored
    )
    resolved_payload = spec_payload("FigureSpec", semantic_payload)
    composition_payloads: tuple[SpecPayload, ...] = ()
    if resolution.composition is not None:
        composition_payloads = (
            spec_payload("FigureCompositionSpec", resolution.authored),
            spec_payload(
                "FigureCompositionProvenance",
                resolution.composition.model_dump(mode="json", exclude_none=True),
            ),
        )
    runtime_binding_payload = _figure_runtime_binding_payload(
        resolution.authored_identity_sha256,
        resolution.resolved_identity_sha256,
        figure_spec,
        execution_context,
        runtime_overlay=bool(runtime_update),
        runtime_metadata=dict(runtime_metadata or {}),
    )
    return FigureExecutionPlan(
        resolution=resolution,
        resolved_spec=resolved_spec,
        execution_spec=figure_spec,
        manifest_id=figure_manifest_id(resolved_spec),
        authored_payload=resolved_payload,
        composition_payloads=composition_payloads,
        runtime_binding_payload=runtime_binding_payload,
        runtime_overlay=bool(runtime_update),
    )


def execute_figure_spec(
    spec: FigureSpecInput,
    *,
    runtime_inputs: Sequence[ParentRef] | None = None,
    runtime_input_authorities: Sequence[FigureInputAuthoritySpec] | None = None,
    runtime_metadata: Mapping[str, Any] | None = None,
    repo_root: Path | str | None = None,
    root: Path | str | None = None,
    provenance: Provenance | None = None,
    issues: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    execution_context: StagedExecutionContext | None = None,
    registry: FigureRegistry,
) -> tuple[FigureManifest, Path]:
    """Resolve and execute exactly the ordinary FigureSpec exposed by display."""
    plan = plan_figure_execution(
        spec,
        runtime_inputs=runtime_inputs,
        runtime_input_authorities=runtime_input_authorities,
        runtime_metadata=runtime_metadata,
        repo_root=repo_root,
        execution_context=execution_context,
        registry=registry,
    )
    resolved_spec = plan.resolved_spec
    figure_spec = plan.execution_spec
    resolved_payload = plan.authored_payload
    composition_payloads = list(plan.composition_payloads)
    runtime_binding_payload = plan.runtime_binding_payload
    root_path = Path(root) if root is not None else default_manifest_root()
    manifest_id = plan.manifest_id
    prov = provenance.model_copy(deep=True) if provenance is not None else collect_git_provenance()
    prov.parents = list(figure_spec.inputs)
    if issues:
        prov.issues.extend(issue for issue in issues if issue not in prov.issues)
    if prov.entrypoint is None:
        prov.entrypoint = EntrypointRef(kind="feedbax-figure-spec", name=resolved_spec.name)

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
                resolved_spec=resolved_spec,
                execution_spec=figure_spec,
                authored_payload=resolved_payload,
                composition_payloads=composition_payloads,
                runtime_binding_payload=runtime_binding_payload,
                status="completed",
                provenance=prov,
                artifacts=[],
                resolved_inputs=resolved_inputs,
                execution_trace=exec_trace,
                metadata={**dict(metadata or {}), "skipped": "run_condition false"},
            )
            return manifest, write_manifest(manifest, root=root_path)

        figures = _build_figures(figure_spec, context, exec_trace, root_path, registry)
        artifacts = _write_figure_custody(
            figures,
            resolved_spec,
            root=root_path,
            manifest_id=manifest_id,
        )
        manifest = _build_figure_manifest(
            manifest_id=manifest_id,
            resolved_spec=resolved_spec,
            execution_spec=figure_spec,
            authored_payload=resolved_payload,
            composition_payloads=composition_payloads,
            runtime_binding_payload=runtime_binding_payload,
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
            resolved_spec=resolved_spec,
            execution_spec=figure_spec,
            authored_payload=resolved_payload,
            composition_payloads=composition_payloads,
            runtime_binding_payload=runtime_binding_payload,
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


def _figure_runtime_binding_payload(
    authored_source_sha256: str,
    resolved_spec_sha256: str,
    execution_spec: FigureSpec,
    execution_context: StagedExecutionContext | None,
    *,
    runtime_overlay: bool,
    runtime_metadata: dict[str, Any],
) -> SpecPayload | None:
    bindings = (
        execution_context.parent_artifact_provider_bindings if execution_context is not None else ()
    )
    if not runtime_overlay and not bindings:
        return None
    runtime_spec = FigureRuntimeBindingSpec(
        authored_figure_source_sha256=authored_source_sha256,
        resolved_figure_spec_sha256=resolved_spec_sha256,
        inputs=list(execution_spec.inputs),
        input_authorities=list(execution_spec.input_authorities),
        runtime_metadata=runtime_metadata,
        artifact_provider_bindings=[
            FigureRuntimeArtifactProviderBinding(
                parent=binding.parent,
                authored_provider=binding.authored_provider,
                runtime_provider=binding.runtime_provider,
            )
            for binding in bindings
        ],
    )
    return spec_payload(
        "FigureRuntimeBindingSpec",
        runtime_spec.model_dump(mode="json", exclude_none=True),
    )


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
        # The id-addressed entry names one manifest. Two inputs sharing an id
        # but not a payload cannot both be that one manifest, and a silent
        # last-win here would hand every expression the second one without
        # anything recording that the first was displaced.
        manifest_key = f"manifest:{resolved.ref.id}"
        existing_manifest = items.get(manifest_key)
        if existing_manifest is not None and existing_manifest.payload != manifest_payload:
            raise FigureInputAuthorityError(
                f"figure inputs bind two different manifests to id {resolved.ref.id!r}; "
                "one identifier addresses one manifest inside one figure"
            )
        items[manifest_key] = ContextItem(
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
                payload[attr] = _portable_value(
                    [
                        item.model_dump(mode="json", exclude_none=True)
                        if hasattr(item, "model_dump")
                        else item
                        for item in value
                    ]
                )
            else:
                payload[attr] = _portable_value(value)
    return payload


_RUNTIME_CONTEXT_KEYS = frozenset({"uri", "path", "root", "callback", "execution_context"})


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


def _panel_grid_declarations(spec: FigureSpec) -> list[str]:
    """Return the grid-shape panel fields a spec declares, in stable order.

    These are the declarations only a panel assembler that composes an explicit
    subplot grid can honor: a non-Cartesian panel type and multi-cell spans. A
    spec that declares one against an assembler that ignores it would render a
    figure that quietly disagrees with what was authored.
    """
    declarations: list[str] = []
    for field_name, declared in (
        ("panel_type", lambda panel: panel.panel_type is not None),
        ("row_span", lambda panel: panel.row_span is not None),
        ("col_span", lambda panel: panel.col_span is not None),
    ):
        if any(declared(panel) for panel in spec.panels):
            declarations.append(field_name)
    return declarations


def _build_figures(
    spec: FigureSpec,
    context: ExpressionContext,
    exec_trace: FigureExecutionTrace,
    root: Path,
    registry: FigureRegistry,
) -> list[RenderedFigure]:
    template = get_figure_template(spec.template, registry=registry) if spec.template else None
    assembler_key = spec.assembler or template.assembler
    assembler_params = {
        **(getattr(template, "assembler_params", {}) if template is not None else {}),
        **spec.assembler_params,
    }
    if "colorbar" in assembler_params:
        raise ValueError(
            "a colorbar is declared by FigureSpec.colorbar, not by assembler_params, so that "
            "its binding to a trace family is validated with the rest of the spec"
        )

    facet_combinations = _facet_combinations(spec, template, context)
    custom_registration = get_figure_constructor(assembler_key, registry=registry)
    if custom_registration.tier == "custom_figure":
        exec_trace.constructor_versions[assembler_key] = custom_registration.version
        if any(panel.equal_data_aspect is not None for panel in spec.panels):
            raise ValueError(
                "FigureSpec declares equal_data_aspect, which requires a registered "
                f"panel assembler; {assembler_key!r} is a custom_figure assembler"
            )
        declared = _panel_grid_declarations(spec)
        if declared:
            raise ValueError(
                f"FigureSpec declares {declared}, which requires a registered panel "
                f"assembler; {assembler_key!r} is a custom_figure assembler"
            )
        if spec.colorbar is not None:
            raise ValueError(
                "FigureSpec declares a colorbar, which requires a registered figure "
                f"assembler that renders one; {assembler_key!r} is a custom_figure assembler"
            )
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

    trace_bindings = resolve_figure_trace_bindings(spec, template, registry=registry)
    panel_constructor_key = assembler_params.get("panel_constructor", "feedbax.comparison_grid")
    panel_registration = get_figure_constructor(
        panel_constructor_key, registry=registry, tier="panel"
    )
    exec_trace.constructor_versions[panel_constructor_key] = panel_registration.version
    if (
        any(panel.equal_data_aspect is not None for panel in spec.panels)
        and panel_registration.key != "feedbax.comparison_grid"
    ):
        raise ValueError(
            "FigureSpec declares equal_data_aspect, but panel assembler "
            f"{panel_registration.key!r} does not support it"
        )
    declared = _panel_grid_declarations(spec)
    if declared and panel_registration.key != "feedbax.comparison_grid":
        raise ValueError(
            f"FigureSpec declares {declared}, but panel assembler "
            f"{panel_registration.key!r} does not support it"
        )
    figure_registration = get_figure_constructor(assembler_key, registry=registry, tier="figure")
    exec_trace.constructor_versions[assembler_key] = figure_registration.version
    panel_values, figure_values = _route_assembler_params(
        assembler_params,
        panel_registration=panel_registration,
        figure_registration=figure_registration,
    )
    colorbar = resolve_figure_colorbar(spec)
    if colorbar is not None:
        if "colorbar" not in figure_registration.params_model.model_fields:
            raise ValueError(
                "FigureSpec declares a colorbar, which requires a figure assembler that "
                f"renders one; {assembler_key!r} declares no colorbar parameter"
            )
        if colorbar.placement is not None and (
            figure_registration.key != "feedbax.grid_figure"
            or panel_registration.key != "feedbax.comparison_grid"
        ):
            raise ValueError(
                "FigureSpec colorbar panel placement requires the "
                "feedbax.comparison_grid/feedbax.grid_figure assembler pair; got "
                f"{panel_registration.key!r}/{figure_registration.key!r}"
            )
        figure_values = {**figure_values, "colorbar": colorbar}
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
            registry,
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
            registry,
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


def expand_trace_families(spec: FigureSpec) -> tuple[TraceFamilyExpansion, ...]:
    """Expand every declared trace family into its enumerated trace bindings.

    This is the single source of family expansion: figure execution compiles
    the returned bindings, and colorbar-style keys read the same expansions.
    """
    return tuple(_expand_trace_family(family) for family in (spec.trace_families or []))


def _expand_trace_family(family: TraceFamily) -> TraceFamilyExpansion:
    indices = family.index.resolve()
    values = family.member_values()
    colors = _family_colors(family, len(indices))
    legend_index = None if family.legend_index is None else str(family.legend_index)
    members: list[TraceFamilyMember] = []
    for index, value, color in zip(indices, values, colors, strict=True):
        binding = substitute_index(family.trace, index, value)
        params: dict[str, Any] = {}
        if color is not None:
            params["color"] = color
        if legend_index is not None:
            params["showlegend"] = str(index) == legend_index
        if params:
            binding = binding.model_copy(update={"params": {**binding.params, **params}})
        members.append(TraceFamilyMember(index=index, color=color, binding=binding, value=value))
    return TraceFamilyExpansion(family=family, members=tuple(members))


def _family_colors(family: TraceFamily, count: int) -> tuple[str | None, ...]:
    """Return one color per member, positioned by value when the family has values.

    Without values, members are sampled at uniform positions along the
    colorscale, which is index spacing. With values, each member is sampled at
    its value's normalized position over the declared value domain, so members
    that are close in the quantity they sweep are close in color.
    """
    if family.colorscale is None:
        return (None,) * count
    if family.values is None:
        sampled = tuple(sample_colorscale_unique(family.colorscale, count, colortype="rgb"))
    else:
        low = min(family.values)
        span = max(family.values) - low
        positions = [(value - low) / span for value in family.values]
        sampled = tuple(sample_colorscale_at(family.colorscale, positions, colortype="rgb"))
    if len(sampled) != count:
        raise ValueError(
            f"trace family {family.name!r} colorscale sampled {len(sampled)} colors "
            f"for {count} indices"
        )
    return sampled


def ordered_family_members(
    expansions: Sequence[TraceFamilyExpansion],
) -> tuple[TraceFamilyMember, ...]:
    """Return every expanded family member in draw order.

    Families expand as blocks, in declaration order. Families that declare a
    shared ``interleave_group`` expand together instead, one member per family
    at each index position, at the place the group's first-declared family
    occupies — the per-position draw order a hand-enumerated spec would write
    out. The contract has already established that the group's families
    enumerate equally many indices.
    """
    groups: dict[str, list[TraceFamilyExpansion]] = {}
    for expansion in expansions:
        group = expansion.family.interleave_group
        if group is not None:
            groups.setdefault(group, []).append(expansion)
    ordered: list[TraceFamilyMember] = []
    emitted: set[str] = set()
    for expansion in expansions:
        group = expansion.family.interleave_group
        if group is None:
            ordered.extend(expansion.members)
        elif group not in emitted:
            emitted.add(group)
            for position in zip(*(member.members for member in groups[group]), strict=True):
                ordered.extend(position)
    return tuple(ordered)


def resolve_figure_colorbar(spec: FigureSpec) -> FigureColorbar | None:
    """Resolve a declared colorbar into the concrete key an assembler renders.

    A standalone declaration already names its colorscale and range and is
    returned unchanged. A family-bound declaration is resolved by reading the
    family's expanded members and placing each assigned color at its position
    in the family's own domain, so the rendered key and the traces cannot
    disagree about which color belongs to which member. An unevenly spaced
    domain therefore yields an unevenly stopped scale, which is the honest
    picture of the assignment rather than a re-sampling of the scale.

    That domain is the family's declared values when it has them, and its index
    set otherwise. A value domain is authoritative: the resolved range is
    exactly ``[min(values), max(values)]``, which the contract already requires
    the declaration to leave unset. An index domain may be relabeled onto the
    declared range instead, since the indices are only names for positions.
    """
    colorbar = spec.colorbar
    if colorbar is None or colorbar.family is None:
        return colorbar
    expansions = {expansion.family.name: expansion for expansion in expand_trace_families(spec)}
    # FigureSpec validation guarantees the named family exists, declares a
    # colorscale, and enumerates at least two members over a numeric domain.
    expansion = expansions[colorbar.family]
    if expansion.family.values is None:
        positions = [float(int(member.index)) for member in expansion.members]
    else:
        positions = [float(member.value) for member in expansion.members]  # type: ignore[arg-type]
    pairs = sorted(zip(positions, (str(member.color) for member in expansion.members)))
    low = pairs[0][0]
    span = pairs[-1][0] - low
    stops = [((position - low) / span, color) for position, color in pairs]
    return colorbar.model_copy(
        update={
            "family": None,
            "colorscale": stops,
            "range": colorbar.range if colorbar.range is not None else (low, low + span),
        }
    )


def resolve_figure_trace_bindings(
    spec: FigureSpec,
    template: FigureTemplate | None,
    *,
    registry: FigureRegistry,
) -> tuple[TraceBindingPlan, ...]:
    """Resolve every declared trace through the one public binding path.

    Slot families first expand to ordinary bindings; those bindings then pass
    through exactly the same slot defaults, constructor inheritance, and
    multiplicity checks as hand-enumerated ``slot_bindings``.
    """
    bindings: list[TraceBindingPlan] = []
    slot_families = {family.slot: family for family in (spec.slot_families or [])}
    if template is not None:
        known_slots = {slot.name for slot in template.slots}
        unknown_slots = sorted((set(spec.slot_bindings) | set(slot_families)) - known_slots)
        if unknown_slots:
            raise ValueError(f"FigureSpec binds unknown template slots: {unknown_slots}")
        for slot in template.slots:
            raw = spec.slot_bindings.get(slot.name)
            family = slot_families.get(slot.name)
            if raw is None and family is None:
                if slot.required:
                    raise ValueError(f"FigureSpec missing required template slot {slot.name!r}")
                continue
            if family is not None:
                slot_bindings = list(family.expand())
            else:
                slot_bindings = raw if isinstance(raw, list) else [raw]
            if not slot_bindings and slot.required:
                raise ValueError(f"FigureSpec missing required template slot {slot.name!r}")
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
    elif spec.slot_bindings or slot_families:
        unknown_slots = sorted(set(spec.slot_bindings) | set(slot_families))
        raise ValueError(
            f"FigureSpec without a template cannot bind template slots: {unknown_slots}"
        )
    bindings.extend(
        TraceBindingPlan(binding=binding, multiplicity="many") for binding in spec.traces
    )
    # Families expand in declaration order, after the individually declared
    # traces, so a family spec and its hand-enumerated equivalent agree on order.
    bindings.extend(
        TraceBindingPlan(binding=member.binding, multiplicity="many")
        for member in ordered_family_members(expand_trace_families(spec))
    )
    piece_names = [
        *(getattr(template, "default_pieces", []) if template is not None else []),
        *spec.pieces,
    ]
    excluded = set(spec.exclude_pieces)
    for piece_name in piece_names:
        if piece_name in excluded:
            continue
        piece = get_figure_piece(piece_name, registry=registry)
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
    collisions = sorted(
        name for name, count in Counter(plan.binding.name for plan in bindings).items() if count > 1
    )
    if collisions:
        raise ValueError(f"FigureSpec resolved trace names collide: {collisions}")
    return tuple(bindings)


def _panel_contents(
    spec: FigureSpec,
    context: ExpressionContext,
    trace_bindings: Sequence[TraceBindingPlan],
    exec_trace: FigureExecutionTrace,
    root: Path,
    registry: FigureRegistry,
    *,
    nonfacet_context: ExpressionContext | None = None,
) -> list[PanelContent]:
    # A panel name is how a trace binding addresses its panel, so two panels
    # sharing one name make the address ambiguous and the map silently keeps
    # the last of them. The refusal is here, at the map, rather than in
    # ``FigureSpec`` validation: the spec is a durable schema surface, and
    # rejecting duplicate names there would retroactively refuse to load saved
    # specs that this version can still describe.
    duplicate_panels = sorted(
        name for name, count in Counter(panel.name for panel in spec.panels).items() if count > 1
    )
    if duplicate_panels:
        raise ValueError(f"FigureSpec declares duplicate panel names: {duplicate_panels}")
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
            registry,
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
        z_axis = panel.z_axis.model_dump(exclude_none=True) if panel and panel.z_axis else None
        equal_data_aspect = (
            panel.equal_data_aspect.model_dump(exclude_none=True)
            if panel and panel.equal_data_aspect
            else None
        )
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
                equal_data_aspect=equal_data_aspect,
                z_axis=z_axis,
                panel_type=panel.panel_type if panel else None,
                row_span=panel.row_span if panel else None,
                col_span=panel.col_span if panel else None,
                camera=panel.camera if panel else None,
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
    registry: FigureRegistry,
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
                    registry,
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
                equal_data_aspect=(
                    base_panel.equal_data_aspect.model_dump(exclude_none=True)
                    if (base_panel is not None and base_panel.equal_data_aspect is not None)
                    else None
                ),
                z_axis=(
                    base_panel.z_axis.model_dump(exclude_none=True)
                    if base_panel is not None and base_panel.z_axis is not None
                    else None
                ),
                panel_type=base_panel.panel_type if base_panel is not None else None,
                row_span=base_panel.row_span if base_panel is not None else None,
                col_span=base_panel.col_span if base_panel is not None else None,
                camera=base_panel.camera if base_panel is not None else None,
            )
        )
    return contents


def _execute_trace_binding(
    binding: TraceBinding,
    context: ExpressionContext,
    exec_trace: FigureExecutionTrace,
    root: Path,
    registry: FigureRegistry,
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
        data = _binding_data(binding, context, exec_trace, root, registry)
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

    registration = get_figure_constructor(binding.constructor, registry=registry, tier="trace")
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
    registry: FigureRegistry,
) -> dict[str, Any]:
    data = {name: _evaluate_value(value, context) for name, value in binding.data.items()}
    if binding.piece is not None:
        piece = get_figure_piece(binding.piece, registry=registry)
        data = {**_piece_data(piece, root), **data}
        data.setdefault("label", piece.label)
        data.update({key: value for key, value in piece.style.items() if key not in data})
        exec_trace.resolved_pieces.append(_piece_resolution(piece))
    return data


def _verify_declared_piece_bytes(piece: Any, ref: ArtifactRef, raw: bytes) -> None:
    """Refuse piece bytes that disagree with the reference's declared profile.

    A path- or URI-addressed piece is *addressed* by its location, so the
    location alone proves nothing about what is stored there now. Whatever
    profile the reference does declare is therefore the only authentication
    available, and it is checked before the bytes are parsed or used. The
    canonical-CAS branch performs the equivalent check inside its provider.

    A reference that declares no profile is left as it is: this verifies a
    claim, it does not invent one.
    """
    if ref.size_bytes is not None and len(raw) != ref.size_bytes:
        raise ArtifactBlobIntegrityError(
            f"FigurePiece {piece.name!r} artifact size mismatch at {ref.uri}: "
            f"expected {ref.size_bytes}, found {len(raw)}"
        )
    if ref.sha256 is not None:
        digest = hashlib.sha256(raw).hexdigest()
        if digest != ref.sha256:
            raise ArtifactBlobIntegrityError(
                f"FigurePiece {piece.name!r} artifact sha256 mismatch at {ref.uri}: "
                f"expected {ref.sha256}, found {digest}"
            )


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
        raw = path.read_bytes()
        _verify_declared_piece_bytes(piece, piece.artifact_ref, raw)
        payload = json.loads(raw.decode("utf-8"))
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
    resolved_spec: FigureSpec,
    execution_spec: FigureSpec,
    authored_payload: SpecPayload,
    composition_payloads: Sequence[SpecPayload],
    runtime_binding_payload: SpecPayload | None,
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
        figure_spec=authored_payload,
        inputs=list(execution_spec.inputs),
        resolved_inputs=[item.ref for item in resolved_inputs if item.manifest is not None],
        resolved_pieces=list(execution_trace.resolved_pieces),
        constructor_versions=dict(execution_trace.constructor_versions),
        template_name=resolved_spec.template,
        binding_records=list(execution_trace.binding_records),
        expression_results_digest=sha256_bytes(canonical_json_bytes(binding_payload)),
        provenance=provenance,
        artifacts=artifacts,
        regeneration_specs=[
            *composition_payloads,
            *([runtime_binding_payload] if runtime_binding_payload is not None else []),
            *(artifact for resolved in resolved_inputs for artifact in resolved.artifact_refs),
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
