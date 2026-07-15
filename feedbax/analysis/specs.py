"""Executable registry for manifest-canonical analysis run specs."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from feedbax.analysis.analysis import AbstractAnalysis
from feedbax.analysis.context import AnalysisRunContext
from feedbax.analysis.evaluation import (
    EvaluationStatesArtifactNotFound,
    execute_evaluation_run_spec,
    load_evaluation_states,
    load_evaluation_states_cache,
    write_evaluation_states_cache,
)
from feedbax.contracts.evaluation_states import (
    EVALUATION_STATES_ARTIFACT_ROLE,
    EVALUATION_STATES_CONTAINER_SCHEMA_ID,
    EVALUATION_STATES_CONTAINER_SCHEMA_VERSION,
    EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V1,
    EVALUATION_STATES_MEDIA_TYPE,
    EVALUATION_STATES_STORAGE_BACKEND,
    EVALUATION_STATES_STORAGE_BACKEND_V1,
    EvaluationStatesContainerError,
    EvaluationStatesHashMismatch,
)
from feedbax.analysis.execution import run_analyses_with_context
from feedbax.analysis.execution_context import (
    EMPTY_STAGED_EXECUTION_CONTEXT,
    StagedExecutionContext,
)
from feedbax.analysis.manifest_inputs import (
    ResolvedManifestInput,
    is_authenticated_manifest_ref,
    resolve_manifest_input,
)
from feedbax.analysis.validation import (
    AnalysisRecipeProtocol,
    validate_analysis_recipe,
    validate_namespaced_type_key,
)
from feedbax.contracts.manifest import (
    AnalysisEvaluationStateResolutionCode,
    AnalysisEvaluationStateResolutionDiagnostic,
    AnalysisEvaluationStateSource,
    AnalysisRunManifest,
    AnalysisRunSpec,
    AnyManifest,
    ArtifactRef,
    EvaluationRunManifest,
    EvaluationRunSpec,
    ParentRef,
    Provenance,
    default_manifest_root,
    analysis_run_manifest_id,
    evaluation_states_cache_path,
    load_manifest,
)
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.persistence.manifest_index import find_manifest_paths_by_id, iter_manifest_files
from feedbax.analysis.types import AnalysisInputData


@dataclass(frozen=True)
class ResolvedAnalysisInput:
    """A manifest parent resolved from an ``AnalysisRunSpec`` input ref."""

    ref: ParentRef
    manifest: AnyManifest | None
    path: Path | None
    states: Any = None
    evaluation_state_source: AnalysisEvaluationStateSource | None = None


@dataclass(frozen=True)
class AnalysisRecipeResult:
    """Executable payload returned by a registered analysis recipe."""

    analyses: dict[str, AbstractAnalysis]
    data: AnalysisInputData
    common_inputs: dict[str, Any] = field(default_factory=dict)
    custom_dependencies: dict[str, AbstractAnalysis] = field(default_factory=dict)


AnalysisRecipe = AnalysisRecipeProtocol
AnalysisRecipeResultValidator = Callable[[str, AnalysisRecipeResult], None]

_ANALYSIS_RECIPES: dict[str, AnalysisRecipe] = {}


class AnalysisRecipeExecutionError(RuntimeError):
    """Raised after a registered analysis recipe fails and a failed manifest is written."""

    def __init__(self, manifest: AnalysisRunManifest, path: Path, cause: BaseException):
        super().__init__(
            f"Analysis recipe for {manifest.id!r} failed; failed manifest written to {path}"
        )
        self.manifest = manifest
        self.path = path
        self.__cause__ = cause


class AnalysisEvaluationStatesResolutionError(RuntimeError):
    """Raised when authored evaluation-state consumption policy cannot be satisfied."""

    def __init__(
        self,
        diagnostic: AnalysisEvaluationStateResolutionDiagnostic,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic
        if cause is not None:
            self.__cause__ = cause


def register_analysis_recipe(
    analysis_type: str,
    recipe: AnalysisRecipe,
    *,
    replace: bool = False,
) -> None:
    """Register an executable analysis recipe by stable type key."""
    analysis_type = validate_namespaced_type_key(
        analysis_type,
        field="analysis_type",
    )
    if analysis_type in _ANALYSIS_RECIPES and not replace:
        raise ValueError(f"Analysis recipe {analysis_type!r} is already registered")
    _ANALYSIS_RECIPES[analysis_type] = validate_analysis_recipe(analysis_type, recipe)


def unregister_analysis_recipe(analysis_type: str) -> None:
    """Remove a previously registered analysis recipe."""
    _ANALYSIS_RECIPES.pop(analysis_type, None)


def registered_analysis_types() -> tuple[str, ...]:
    """Return registered executable analysis type keys."""
    return tuple(sorted(_ANALYSIS_RECIPES))


def get_analysis_recipe(analysis_type: str) -> AnalysisRecipe:
    """Return a registered analysis recipe or raise a clear execution error."""
    try:
        return _ANALYSIS_RECIPES[analysis_type]
    except KeyError as exc:
        available = ", ".join(registered_analysis_types()) or "none"
        raise ValueError(
            f"Analysis recipe {analysis_type!r} is not registered. "
            f"Registered analysis recipes: {available}."
        ) from exc


def coerce_analysis_run_spec(
    value: AnalysisRunSpec | Mapping[str, Any] | Path | str,
) -> AnalysisRunSpec:
    """Load an ``AnalysisRunSpec`` from an object, mapping, or JSON file path."""
    if isinstance(value, AnalysisRunSpec):
        return value
    raw: Mapping[str, Any]
    if isinstance(value, Mapping):
        raw = value
    else:
        raw = json.loads(Path(value).read_text(encoding="utf-8"))
    source_version = raw.get("schema_version")
    result = default_spec_registry.migrate(
        "AnalysisRunSpec",
        raw,
        source_version=source_version if isinstance(source_version, str) else None,
        assume_current=source_version is None,
    )
    return AnalysisRunSpec.model_validate(result.payload)


def find_manifest_by_id(
    manifest_id: str, *, root: Path | str | None = None
) -> tuple[AnyManifest, Path]:
    """Find one manifest by ID under a manifest root."""
    root_path = Path(root) if root is not None else default_manifest_root()
    matches: list[tuple[AnyManifest, Path]] = []
    indexed_paths = find_manifest_paths_by_id(manifest_id, root=root_path)
    for manifest_path in indexed_paths:
        manifest = load_manifest(manifest_path)
        if manifest.id == manifest_id:
            matches.append((manifest, manifest_path))
    if matches:
        return _single_manifest_match(manifest_id, root_path, matches)

    for manifest_path in iter_manifest_files(root_path):
        manifest = load_manifest(manifest_path)
        if manifest.id == manifest_id:
            matches.append((manifest, manifest_path))
    return _single_manifest_match(manifest_id, root_path, matches)


def _single_manifest_match(
    manifest_id: str,
    root_path: Path,
    matches: list[tuple[AnyManifest, Path]],
) -> tuple[AnyManifest, Path]:
    if not matches:
        raise FileNotFoundError(f"Manifest {manifest_id!r} not found under {root_path}")
    if len(matches) > 1:
        paths = ", ".join(str(path) for _manifest, path in matches)
        raise ValueError(f"Manifest {manifest_id!r} is duplicated under {root_path}: {paths}")
    return matches[0]


def resolve_analysis_inputs(
    spec: AnalysisRunSpec,
    *,
    root: Path | str | None = None,
    authenticated_inputs: Mapping[int, ResolvedManifestInput] | None = None,
    execution_context: StagedExecutionContext = EMPTY_STAGED_EXECUTION_CONTEXT,
) -> list[ResolvedAnalysisInput]:
    """Resolve ``AnalysisRunSpec.inputs`` to manifests and cached evaluation states."""
    root_path = Path(root) if root is not None else default_manifest_root()
    resolved: list[ResolvedAnalysisInput] = []
    for index, ref in enumerate(spec.inputs):
        manifest: AnyManifest | None = None
        manifest_path: Path | None = None
        states: Any = None
        state_source: AnalysisEvaluationStateSource | None = None
        if is_authenticated_manifest_ref(ref):
            authenticated = (
                authenticated_inputs.get(index) if authenticated_inputs is not None else None
            )
            if authenticated is None:
                authenticated = resolve_manifest_input(ref, root_path)
            manifest, manifest_path = authenticated.manifest, authenticated.path
        elif ref.kind.endswith("Manifest"):
            manifest, manifest_path = find_manifest_by_id(ref.id, root=root_path)
        if ref.kind == "EvaluationRunManifest":
            if spec.evaluation_states_policy == "require_durable":
                states, state_source = _load_required_durable_evaluation_states(
                    ref,
                    manifest,
                    root=root_path,
                )
                resolved.append(
                    ResolvedAnalysisInput(
                        ref=ref,
                        manifest=manifest,
                        path=manifest_path,
                        states=states,
                        evaluation_state_source=state_source,
                    )
                )
                continue
            states_path = evaluation_states_cache_path(ref.id, root=root_path)
            if states_path.exists():
                states = load_evaluation_states_cache(states_path, manifest_id=ref.id)
                state_source = AnalysisEvaluationStateSource(
                    source_kind="evaluation_cache",
                    requested_evaluation_manifest_id=ref.id,
                    evaluation_manifest_authority=_portable_manifest_authority(ref),
                    supplying_evaluation_manifest_id=(
                        manifest.id if isinstance(manifest, EvaluationRunManifest) else None
                    ),
                )
            else:
                if isinstance(manifest, EvaluationRunManifest):
                    try:
                        states = load_evaluation_states(manifest, root=root_path)
                    except EvaluationStatesArtifactNotFound:
                        rederived = _rederive_evaluation_states(
                            ref.id,
                            manifest,
                            root=root_path,
                            execution_context=execution_context,
                        )
                        state_source = AnalysisEvaluationStateSource(
                            source_kind="analysis_time_recompute",
                            requested_evaluation_manifest_id=ref.id,
                            evaluation_manifest_authority=_portable_manifest_authority(ref),
                            supplying_evaluation_manifest_id=manifest.id,
                            resulting_evaluation_manifest_id=rederived.id,
                        )
                    else:
                        states_path.parent.mkdir(parents=True, exist_ok=True)
                        write_evaluation_states_cache(
                            states_path,
                            manifest_id=ref.id,
                            states=states,
                        )
                        state_source = _durable_state_source(ref, manifest)
                else:
                    rederived = _rederive_evaluation_states(
                        ref.id,
                        manifest,
                        root=root_path,
                        execution_context=execution_context,
                    )
                    state_source = AnalysisEvaluationStateSource(
                        source_kind="analysis_time_recompute",
                        requested_evaluation_manifest_id=ref.id,
                        evaluation_manifest_authority=_portable_manifest_authority(ref),
                        resulting_evaluation_manifest_id=rederived.id,
                    )
            if states is None:
                states = load_evaluation_states_cache(states_path, manifest_id=ref.id)
        resolved.append(
            ResolvedAnalysisInput(
                ref=ref,
                manifest=manifest,
                path=manifest_path,
                states=states,
                evaluation_state_source=state_source,
            )
        )
    return resolved


def _portable_manifest_authority(ref: ParentRef) -> ParentRef:
    """Preserve authored authority fields without persisting a machine-local locator."""
    return ref.model_copy(update={"uri": None})


def _resolution_error(
    *,
    code: AnalysisEvaluationStateResolutionCode,
    manifest_id: str,
    message: str,
    artifact_id: str | None = None,
    details: dict[str, Any] | None = None,
    cause: BaseException | None = None,
) -> AnalysisEvaluationStatesResolutionError:
    return AnalysisEvaluationStatesResolutionError(
        AnalysisEvaluationStateResolutionDiagnostic(
            code=code,
            evaluation_manifest_id=manifest_id,
            message=message,
            artifact_id=artifact_id,
            details=details or {},
        ),
        cause,
    )


def _evaluation_states_artifact_for_durable_policy(
    ref: ParentRef,
    manifest: AnyManifest | None,
) -> tuple[EvaluationRunManifest, ArtifactRef]:
    if not isinstance(manifest, EvaluationRunManifest):
        raise _resolution_error(
            code="provenance_mismatch",
            manifest_id=ref.id,
            message=(
                "require_durable expected an EvaluationRunManifest matching the authored "
                f"input {ref.id!r}; found {type(manifest).__name__}."
            ),
            details={"resolved_manifest_kind": type(manifest).__name__},
        )
    if manifest.id != ref.id:
        raise _resolution_error(
            code="provenance_mismatch",
            manifest_id=ref.id,
            message=(
                "require_durable resolved an evaluation manifest with different identity: "
                f"requested={ref.id!r}, resolved={manifest.id!r}."
            ),
            details={"resolved_evaluation_manifest_id": manifest.id},
        )
    if manifest.status != "completed":
        raise _resolution_error(
            code="provenance_mismatch",
            manifest_id=ref.id,
            message=(
                "require_durable only accepts completed evaluation authority; "
                f"manifest {ref.id!r} has status {manifest.status!r}."
            ),
            details={"evaluation_manifest_status": manifest.status},
        )
    matches = [
        artifact for artifact in manifest.artifacts if artifact.role == EVALUATION_STATES_ARTIFACT_ROLE
    ]
    if not matches:
        raise _resolution_error(
            code="missing_durable_states",
            manifest_id=ref.id,
            message=(
                f"Evaluation manifest {ref.id!r} has no durable evaluation_states artifact; "
                "rerun evaluation with params.states_custody='durable' or select "
                "evaluation_states_policy='recompute'."
            ),
        )
    if len(matches) != 1:
        raise _resolution_error(
            code="provenance_mismatch",
            manifest_id=ref.id,
            message=(
                f"Evaluation manifest {ref.id!r} has {len(matches)} evaluation_states "
                "artifacts; durable authority must be unique."
            ),
            details={"artifact_count": len(matches)},
        )
    return manifest, matches[0]


def _validate_durable_artifact_evidence(
    ref: ParentRef,
    manifest: EvaluationRunManifest,
    artifact: ArtifactRef,
) -> None:
    if not artifact.artifact_id or not artifact.sha256:
        raise _resolution_error(
            code="custody_unavailable",
            manifest_id=ref.id,
            artifact_id=artifact.artifact_id,
            message=(
                f"Durable evaluation states for {ref.id!r} lack content-addressed custody "
                "identity (artifact_id and sha256 are required)."
            ),
        )
    if artifact.size_bytes is None or artifact.size_bytes < 0:
        raise _resolution_error(
            code="custody_unavailable",
            manifest_id=ref.id,
            artifact_id=artifact.artifact_id,
            message=(
                f"Durable evaluation states for {ref.id!r} lack a nonnegative declared "
                "artifact size."
            ),
            details={"artifact_size_bytes": artifact.size_bytes},
        )
    expected_artifact_id = f"artifact://sha256/{artifact.sha256}"
    if artifact.artifact_id != expected_artifact_id:
        raise _resolution_error(
            code="custody_unavailable",
            manifest_id=ref.id,
            artifact_id=artifact.artifact_id,
            message=(
                "Durable evaluation-state artifact identity does not match its sha256: "
                f"artifact_id={artifact.artifact_id!r}, expected={expected_artifact_id!r}."
            ),
        )
    metadata = artifact.metadata
    metadata_manifest_id = metadata.get("manifest_id")
    if metadata_manifest_id != manifest.id:
        raise _resolution_error(
            code="provenance_mismatch",
            manifest_id=ref.id,
            artifact_id=artifact.artifact_id,
            message=(
                "Durable evaluation-state artifact provenance does not name its supplying "
                f"manifest: metadata.manifest_id={metadata_manifest_id!r}, "
                f"expected={manifest.id!r}."
            ),
            details={"artifact_manifest_id": metadata_manifest_id},
        )
    schema_id = metadata.get("schema_id")
    schema_version = metadata.get("schema_version")
    storage_backend = metadata.get("storage_backend")
    expected_storage = {
        EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V1: EVALUATION_STATES_STORAGE_BACKEND_V1,
        EVALUATION_STATES_CONTAINER_SCHEMA_VERSION: EVALUATION_STATES_STORAGE_BACKEND,
    }
    if (
        schema_id != EVALUATION_STATES_CONTAINER_SCHEMA_ID
        or schema_version not in expected_storage
        or storage_backend != expected_storage.get(schema_version)
    ):
        raise _resolution_error(
            code="schema_mismatch",
            manifest_id=ref.id,
            artifact_id=artifact.artifact_id,
            message=(
                "Durable evaluation-state artifact schema/storage metadata is unsupported: "
                f"schema_id={schema_id!r}, schema_version={schema_version!r}, "
                f"storage_backend={storage_backend!r}."
            ),
            details={
                "schema_id": schema_id,
                "schema_version": schema_version,
                "storage_backend": storage_backend,
            },
        )
    if artifact.media_type != EVALUATION_STATES_MEDIA_TYPE:
        raise _resolution_error(
            code="schema_mismatch",
            manifest_id=ref.id,
            artifact_id=artifact.artifact_id,
            message=(
                "Durable evaluation-state artifact has unsupported media type: "
                f"{artifact.media_type!r}; expected {EVALUATION_STATES_MEDIA_TYPE!r}."
            ),
            details={"media_type": artifact.media_type},
        )


def _durable_state_source(
    ref: ParentRef,
    manifest: EvaluationRunManifest,
) -> AnalysisEvaluationStateSource:
    artifacts = [
        artifact for artifact in manifest.artifacts if artifact.role == EVALUATION_STATES_ARTIFACT_ROLE
    ]
    artifact = artifacts[0]
    return AnalysisEvaluationStateSource(
        source_kind="durable",
        requested_evaluation_manifest_id=ref.id,
        evaluation_manifest_authority=_portable_manifest_authority(ref),
        supplying_evaluation_manifest_id=manifest.id,
        artifact_id=artifact.artifact_id,
        artifact_sha256=artifact.sha256,
        artifact_size_bytes=artifact.size_bytes,
        artifact_storage_backend=artifact.storage_backend,
        container_schema_id=artifact.metadata.get("schema_id"),
        container_schema_version=artifact.metadata.get("schema_version"),
        container_storage_backend=artifact.metadata.get("storage_backend"),
    )


def _load_required_durable_evaluation_states(
    ref: ParentRef,
    manifest: AnyManifest | None,
    *,
    root: Path,
) -> tuple[Any, AnalysisEvaluationStateSource]:
    manifest, artifact = _evaluation_states_artifact_for_durable_policy(ref, manifest)
    _validate_durable_artifact_evidence(ref, manifest, artifact)
    try:
        states = load_evaluation_states(manifest, root=root)
    except UnsupportedSpecVersion as exc:
        raise _resolution_error(
            code="schema_mismatch",
            manifest_id=ref.id,
            artifact_id=artifact.artifact_id,
            message=f"Durable evaluation-state container schema is unsupported: {exc}",
            cause=exc,
        ) from exc
    except ValidationError as exc:
        schema_fields = {"schema_id", "schema_version", "storage_backend"}
        if any(error["loc"] and error["loc"][0] in schema_fields for error in exc.errors()):
            raise _resolution_error(
                code="schema_mismatch",
                manifest_id=ref.id,
                artifact_id=artifact.artifact_id,
                message=f"Durable evaluation-state container header is inconsistent: {exc}",
                cause=exc,
            ) from exc
        raise _resolution_error(
            code="custody_unavailable",
            manifest_id=ref.id,
            artifact_id=artifact.artifact_id,
            message=(
                f"Durable evaluation-state container for {ref.id!r} is malformed: {exc}"
            ),
            cause=exc,
        ) from exc
    except (
        FileNotFoundError,
        OSError,
        zipfile.BadZipFile,
        json.JSONDecodeError,
        UnicodeDecodeError,
        KeyError,
        EvaluationStatesHashMismatch,
        EvaluationStatesContainerError,
    ) as exc:
        raise _resolution_error(
            code="custody_unavailable",
            manifest_id=ref.id,
            artifact_id=artifact.artifact_id,
            message=(
                f"Durable evaluation-state custody for {ref.id!r} could not be verified: {exc}"
            ),
            cause=exc,
        ) from exc
    return states, _durable_state_source(ref, manifest)


def _rederive_evaluation_states(
    manifest_id: str,
    manifest: AnyManifest | None,
    *,
    root: Path,
    execution_context: StagedExecutionContext,
) -> EvaluationRunManifest:
    if not isinstance(manifest, EvaluationRunManifest):
        raise TypeError(
            f"Expected EvaluationRunManifest {manifest_id!r}, got {type(manifest).__name__}"
        )
    if manifest.status != "completed":
        raise ValueError(
            f"Cannot re-derive states for evaluation manifest {manifest_id!r} "
            f"with status {manifest.status!r}"
        )
    run_spec = EvaluationRunSpec.model_validate(manifest.evaluation_spec.inline)
    metadata = {key: value for key, value in manifest.metadata.items() if key != "cache"}
    rederived, _path = execute_evaluation_run_spec(
        run_spec,
        root=root,
        provenance=manifest.provenance,
        metadata=metadata,
        execution_context=execution_context,
    )
    if rederived.id != manifest_id:
        raise ValueError(
            f"Evaluation spec for {manifest_id!r} re-derived manifest {rederived.id!r}"
        )
    return rederived


def requested_outputs_from_spec(spec: AnalysisRunSpec) -> set[str] | None:
    """Return the requested output set encoded in analysis spec params."""
    raw = spec.params.get("requested_outputs", spec.params.get("outputs"))
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("AnalysisRunSpec params requested_outputs/outputs must be a list")
    outputs = {str(item) for item in raw}
    return outputs or None


def execute_analysis_run_spec(
    spec: AnalysisRunSpec | Mapping[str, Any] | Path | str,
    *,
    root: Path | str | None = None,
    provenance: Provenance | None = None,
    issues: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    execution_context: StagedExecutionContext = EMPTY_STAGED_EXECUTION_CONTEXT,
    validate_result: AnalysisRecipeResultValidator | None = None,
    fig_dump_path: Path | str | None = None,
    fig_dump_formats: Sequence[str] = ("html",),
    use_cache: bool = True,
    force: bool = False,
) -> tuple[AnalysisRunManifest, Path]:
    """Execute a serialized analysis spec and write an ``AnalysisRunManifest``."""
    run_spec = coerce_analysis_run_spec(spec)
    root_path = Path(root) if root is not None else default_manifest_root()
    authenticated_inputs: dict[int, ResolvedManifestInput] = {}
    for index, input_ref in enumerate(run_spec.inputs):
        if is_authenticated_manifest_ref(input_ref):
            authenticated_inputs[index] = resolve_manifest_input(input_ref, root_path)
    recipe = get_analysis_recipe(run_spec.analysis_type)
    manifest_id = analysis_run_manifest_id(run_spec)
    if use_cache and not force:
        try:
            existing_manifest, existing_path = find_manifest_by_id(manifest_id, root=root_path)
        except FileNotFoundError:
            pass
        else:
            if (
                isinstance(existing_manifest, AnalysisRunManifest)
                and existing_manifest.status == "completed"
            ):
                return existing_manifest, existing_path

    context = AnalysisRunContext(
        spec=run_spec,
        root=root_path,
        fig_dump_path=fig_dump_path,
        fig_dump_formats=tuple(fig_dump_formats),
        provenance=provenance,
        issues=issues,
        metadata=metadata,
    )

    try:
        resolved_inputs = resolve_analysis_inputs(
            run_spec,
            root=root_path,
            authenticated_inputs=authenticated_inputs,
            execution_context=execution_context,
        )
        context.record_evaluation_state_sources(
            source
            for resolved_input in resolved_inputs
            if (source := resolved_input.evaluation_state_source) is not None
        )
        result = recipe(run_spec, root_path, resolved_inputs, execution_context)
        if not result.analyses:
            raise ValueError(f"Analysis recipe {run_spec.analysis_type!r} returned no analyses")
        if validate_result is not None:
            validate_result(run_spec.analysis_type, result)
        run_analyses_with_context(
            result.analyses,
            result.data,
            context,
            fig_dump_path=Path(fig_dump_path) if fig_dump_path is not None else None,
            fig_dump_formats=list(fig_dump_formats),
            custom_dependencies=result.custom_dependencies,
            requested_outputs=requested_outputs_from_spec(run_spec),
            **result.common_inputs,
        )
    except Exception as exc:
        if isinstance(exc, AnalysisEvaluationStatesResolutionError):
            context.record_evaluation_state_resolution_diagnostic(exc.diagnostic)
        if context.manifest_path is None:
            manifest, path = context.finalize(
                status="failed",
                metadata={
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                },
            )
            raise AnalysisRecipeExecutionError(manifest, path, exc) from exc
        raise

    if context.manifest_path is None:
        raise RuntimeError("Analysis execution completed without writing a manifest")
    manifest = load_manifest(context.manifest_path)
    if not isinstance(manifest, AnalysisRunManifest):
        raise TypeError(f"Expected AnalysisRunManifest, got {type(manifest).__name__}")
    return manifest, context.manifest_path
