"""Stage the local input files declared by one fulfillment candidate."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

from feedbax._atomic import write_bytes_atomic
from feedbax._secure_fs import canonicalize_trusted_system_aliases
from feedbax.analysis.evaluation_inputs import resolve_evaluation_inputs
from feedbax.analysis.execution_context import (
    EMPTY_STAGED_EXECUTION_CONTEXT,
    StagedExecutionContext,
    StagedParentExecutionLocation,
    with_staged_manifest_provider_inputs,
    with_staged_parent_artifact_provider_bindings,
    with_staged_parent_execution_locations,
)
from feedbax.analysis.figures import figure_piece_inputs
from feedbax.analysis.fulfillment import evaluation_expected_parents, resolve_artifact_bytes
from feedbax.analysis.fulfillment_adapters import (
    AnalysisNodeRequest,
    EvaluationNodeRequest,
    FigureNodeRequest,
    FulfillmentEnvironment,
    NodeRequest,
    ReportNodeRequest,
    figure_execution_plan,
    node_execution_context,
    report_execution_spec,
)
from feedbax.analysis.evaluation import evaluation_states_cache_matches
from feedbax.analysis.manifest_inputs import (
    ResolvedManifestInput,
    is_authenticated_manifest_ref,
    resolve_contained_manifest_input,
    resolve_manifest_input,
    restated_parent_differences,
)
from feedbax.analysis.specs import find_manifest_by_id
from feedbax.contracts.base import ArtifactRef, ParentRef
from feedbax.contracts.evaluation_states import EVALUATION_STATES_ARTIFACT_ROLE
from feedbax.contracts.manifest import (
    EvaluationRunManifest,
    EvaluationRunSpec,
    evaluation_states_cache_path,
)
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider


def stage_node_inputs(
    request: NodeRequest, *, environment: FulfillmentEnvironment, destination: Path
) -> NodeRequest:
    """Copy this node's local inputs and rebind its local staged manifest locations."""
    root = Path(environment.root).resolve()
    context = node_execution_context(request, environment)
    exact_locators: dict[str, str] = {}
    if isinstance(request, EvaluationNodeRequest):
        parents = evaluation_expected_parents(request.spec)
    elif isinstance(request, AnalysisNodeRequest):
        parents = list(request.spec.inputs)
    elif isinstance(request, FigureNodeRequest):
        spec = figure_execution_plan(request, environment=environment).execution_spec
        parents = list(spec.inputs)
        for artifact in figure_piece_inputs(spec, registry=environment.registries.figures):
            _stage_figure_piece(artifact, root=root, destination=destination)
    elif isinstance(request, ReportNodeRequest):
        parents = list(report_execution_spec(request).inputs)
        if request.exact_parents is not None and request.execution_context is None:
            # The authored report adapter binds these locations itself when no
            # per-node context was supplied.
            context = EMPTY_STAGED_EXECUTION_CONTEXT
            exact_locators = {
                entry.parent.id: entry.execution_uri for entry in request.exact_parents.parents
            }
    else:
        raise TypeError(f"{request.node_key!r} has no single candidate input closure")

    locations = list(context.parent_execution_locations)
    seen: set[str] = set()

    def stage_parent(ref: ParentRef, *, needs_states: bool = False) -> None:
        key = ref.model_dump_json()
        if key in seen:
            return
        seen.add(key)
        resolved, location = _resolve_parent(
            ref, request=request, root=root, context=context, exact_locators=exact_locators
        )
        if resolved is None:
            return
        if location is None or (
            location.artifact_provider is None
            and canonicalize_trusted_system_aliases(location.root) == root
        ):
            source_root = location.root if location is not None else root
            relative = resolved.path.relative_to(source_root)
            _stage_bytes(destination / relative, resolved.raw_bytes)
            if location is not None and location in context.parent_execution_locations:
                locations[context.parent_execution_locations.index(location)] = replace(
                    location, root=destination.resolve()
                )
        manifest = resolved.manifest
        products = getattr(manifest, "produced_data", ())
        for artifact in (*manifest.artifacts, *(a for p in products for a in p.artifacts)):
            _stage_root_artifact(artifact, root=root, destination=destination)
        if needs_states and isinstance(manifest, EvaluationRunManifest):
            cache = evaluation_states_cache_path(manifest.id, root=root)
            try:
                _stage_file(cache, root=root, destination=destination)
            except FileNotFoundError:
                pass
            if not any(a.role == EVALUATION_STATES_ARTIFACT_ROLE for a in manifest.artifacts) and (
                not evaluation_states_cache_matches(destination / cache.relative_to(root), manifest)
            ):
                # The executor needs the upstream inputs to regenerate an unusable cache.
                run_spec = EvaluationRunSpec.model_validate(manifest.evaluation_spec.inline)
                for upstream in evaluation_expected_parents(run_spec):
                    stage_parent(upstream)

    for ref in parents:
        stage_parent(
            ref,
            needs_states=isinstance(request, AnalysisNodeRequest)
            and request.spec.evaluation_states_policy != "require_durable",
        )

    if tuple(locations) != context.parent_execution_locations:
        staged_context = with_staged_parent_execution_locations(context, locations)
        staged_context = with_staged_parent_artifact_provider_bindings(
            staged_context, context.parent_artifact_provider_bindings
        )
        return replace(request, execution_context=staged_context)
    return request


def _resolve_parent(
    ref: ParentRef,
    *,
    request: NodeRequest,
    root: Path,
    context: StagedExecutionContext,
    exact_locators: dict[str, str],
) -> tuple[ResolvedManifestInput | None, StagedParentExecutionLocation | None]:
    if is_authenticated_manifest_ref(ref):
        matches = [
            location
            for location in context.parent_execution_locations
            if location.parent == ref
            or (
                isinstance(request, EvaluationNodeRequest)
                and ref.kind == "TrainingRunManifest"
                and not restated_parent_differences(ref, location.parent)
            )
        ]
        if matches:
            if len(matches) != 1:
                raise ValueError(f"ambiguous staged parent locations for {ref.id!r}")
            location = matches[0]
            return context.resolve_manifest_input(location.parent), location
        try:
            return resolve_manifest_input(ref, root, exact_locators.get(ref.id)), None
        except FileNotFoundError:
            if not isinstance(request, ReportNodeRequest) or exact_locators or not (
                context.manifest_roots or context.opened_artifact_providers
            ):
                raise
            bound = with_staged_manifest_provider_inputs(context, [ref])
            return bound.resolve_manifest_input(ref), bound.parent_execution_location(ref)

    if not ref.kind.endswith("Manifest"):
        return None, None
    if isinstance(request, FigureNodeRequest):
        if ref.uri is None:
            raise ValueError(f"figure input {ref.id!r} has no contained locator")
        locator = ref.uri
    elif (
        isinstance(request, EvaluationNodeRequest)
        and ref.kind == "TrainingRunManifest"
        and ref in request.spec.inputs
        and len(request.spec.inputs) == 1
        and ref.role == "training_run"
    ):
        parent, = resolve_evaluation_inputs(
            request.spec, manifest_root=root, execution_context=context
        )
        resolved = resolve_contained_manifest_input(ref, root, parent.reference)
        if (hashlib.sha256(resolved.raw_bytes).hexdigest(), len(resolved.raw_bytes)) != (
            parent.sha256, parent.size_bytes
        ):
            raise ValueError(f"evaluation parent {ref.id!r} changed during staging")
        return resolved, None
    else:
        _, path = find_manifest_by_id(ref.id, root=root, expected_kind=ref.kind)
        locator = path.relative_to(root).as_posix()
    return resolve_contained_manifest_input(ref, root, locator), None


def _stage_root_artifact(artifact: ArtifactRef, *, root: Path, destination: Path) -> None:
    resolution = resolve_artifact_bytes(artifact, root=root)
    source = resolution.path
    # A provider-backed or URI-addressed artifact may have no copy in the recipe root.
    if source is None or not source.is_file():
        return
    if artifact.sha256 is None or artifact.size_bytes is None:
        raise ValueError(f"local fulfillment input {artifact.logical_name!r} has no byte profile")
    _stage_file(source, root=root, destination=destination, artifact=artifact)


def _stage_figure_piece(artifact: ArtifactRef, *, root: Path, destination: Path) -> None:
    uri = artifact.uri
    assert uri is not None
    if uri.startswith("artifact://sha256/"):
        provider = ImmutableArtifactBlobProvider(root)
        raw = provider.get_bytes(artifact)
        ImmutableArtifactBlobProvider(destination.resolve()).store_object(raw)
    elif not Path(uri).is_absolute():
        _stage_file(root / uri, root=root, destination=destination, artifact=artifact)


def _stage_file(
    source: Path, *, root: Path, destination: Path, artifact: ArtifactRef | None = None
) -> None:
    relative = source.relative_to(root)
    if ".." in relative.parts or not source.resolve(strict=True).is_relative_to(root):
        raise ValueError(f"fulfillment input {source} escapes {root}")
    raw = source.read_bytes()
    if artifact is not None and (
        (artifact.size_bytes is not None and len(raw) != artifact.size_bytes)
        or (artifact.sha256 is not None and hashlib.sha256(raw).hexdigest() != artifact.sha256)
    ):
        raise ValueError(f"fulfillment input {source} disagrees with its declared byte profile")
    _stage_bytes(destination / relative, raw)


def _stage_bytes(target: Path, raw: bytes) -> None:
    if target.is_file() and target.read_bytes() == raw:
        return
    write_bytes_atomic(target, raw)
