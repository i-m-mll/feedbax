"""Executable registry for manifest-canonical analysis run specs."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import jax.tree_util as jtu
from pydantic import ValidationError

from feedbax.analysis.analysis import AbstractAnalysis
from feedbax.analysis.context import AnalysisRunContext
from feedbax.analysis.evaluation import (
    EvaluationStatesArtifactNotFound,
    EVALUATION_STATES_CACHE_SCHEMA_VERSION,
    execute_evaluation_run_spec,
    load_evaluation_states,
    load_evaluation_states_cache,
    write_evaluation_states_cache,
)
from feedbax.contracts.evaluation_states import (
    EVALUATION_STATES_ARTIFACT_ROLE,
    EvaluationStatesCustodyUnavailable,
    EvaluationStatesHashMismatch,
    EvaluationStatesProvenanceMismatch,
    EvaluationStatesSchemaMismatch,
    load_authenticated_evaluation_states_artifact,
)
from feedbax.analysis.execution_context import (
    EMPTY_STAGED_EXECUTION_CONTEXT,
    StagedArtifactProviderRootBinding,
    StagedCheckpointCustodyRootBinding,
    StagedExecutionContext,
    StagedExecutionContextError,
    StagedManifestRootBinding,
    resolve_staged_execution_context,
    with_staged_manifest_provider_inputs,
    with_staged_repo_root,
)
from feedbax.analysis.manifest_inputs import (
    ResolvedManifestInput,
    authenticated_manifest_ref,
    is_authenticated_manifest_ref,
    resolve_manifest_input,
)
from feedbax.analysis.validation import (
    AnalysisRecipeProtocol,
    EvaluationStatesStructureProviderProtocol,
    validate_analysis_recipe,
    validate_evaluation_states_structure_provider,
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
from feedbax.contracts.analysis_composition import (
    AnalysisRunDeltaSpec,
    FlattenedAnalysisRun,
    analysis_composition_provenance,
    flatten_analysis_run_delta,
    is_analysis_run_delta_payload,
)
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.contracts.run_aliases import RunAliasCatalog, resolve_run_aliases
from feedbax.contracts.staged_execution import StagedExecutionDescriptor
from feedbax.persistence.manifest_index import find_manifest_paths_by_id, iter_manifest_files
from feedbax.analysis.types import AnalysisInputData


@dataclass(frozen=True)
class ResolvedAnalysisInput:
    """A manifest parent resolved from an ``AnalysisRunSpec`` input ref.

    ``manifest_input`` retains the exact already-authenticated manifest bytes,
    reference, parsed value, and runtime path when the authored ref carries an
    authenticated authority profile.
    """

    ref: ParentRef
    manifest: AnyManifest | None
    path: Path | None
    states: Any = None
    evaluation_state_source: AnalysisEvaluationStateSource | None = None
    evaluation_state_receipt: EvaluationStateMaterializationReceipt | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    manifest_input: ResolvedManifestInput | None = field(
        default=None,
        repr=False,
        compare=False,
    )


_EVALUATION_STATE_RECEIPT_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class EvaluationStateMaterializationReceipt:
    """Resolver-issued runtime proof binding one state object to its source path.

    ``proof_kind`` is deliberately truthful about the source mechanism:
    durable artifacts are content-authenticated, evaluation caches are
    manifest-keyed/versioned, and recomputed states carry an authenticated
    resulting manifest. The receipt is runtime-only and is not a durable schema.
    """

    source: AnalysisEvaluationStateSource
    proof_kind: Literal[
        "authenticated_artifact",
        "manifest_keyed_cache",
        "authenticated_recompute",
    ]
    _states: Any = field(repr=False, compare=False)

    def __init__(
        self,
        source: AnalysisEvaluationStateSource,
        proof_kind: Literal[
            "authenticated_artifact",
            "manifest_keyed_cache",
            "authenticated_recompute",
        ],
        states: Any,
        *,
        _token: object,
    ) -> None:
        if _token is not _EVALUATION_STATE_RECEIPT_TOKEN:
            raise TypeError(
                "EvaluationStateMaterializationReceipt is issued by resolve_analysis_inputs"
            )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "proof_kind", proof_kind)
        object.__setattr__(self, "_states", states)

    def matches(self, states: Any, source: AnalysisEvaluationStateSource) -> bool:
        """Return whether this receipt binds the exact resolved object and source."""
        return self._states is states and self.source == source


def _evaluation_state_receipt(
    states: Any,
    source: AnalysisEvaluationStateSource,
) -> EvaluationStateMaterializationReceipt:
    proof_kind = {
        "durable": "authenticated_artifact",
        "evaluation_cache": "manifest_keyed_cache",
        "analysis_time_recompute": "authenticated_recompute",
    }[source.source_kind]
    return EvaluationStateMaterializationReceipt(
        source,
        proof_kind,
        states,
        _token=_EVALUATION_STATE_RECEIPT_TOKEN,
    )


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
_EVALUATION_STATES_STRUCTURE_PROVIDERS: dict[str, EvaluationStatesStructureProviderProtocol] = {}


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
    evaluation_states_structure: EvaluationStatesStructureProviderProtocol | None = None,
) -> None:
    """Register an executable analysis recipe by stable type key."""
    analysis_type = validate_namespaced_type_key(
        analysis_type,
        field="analysis_type",
    )
    if analysis_type in _ANALYSIS_RECIPES and not replace:
        raise ValueError(f"Analysis recipe {analysis_type!r} is already registered")
    validated_recipe = validate_analysis_recipe(analysis_type, recipe)
    validated_structure = (
        validate_evaluation_states_structure_provider(
            analysis_type,
            evaluation_states_structure,
        )
        if evaluation_states_structure is not None
        else None
    )
    _ANALYSIS_RECIPES[analysis_type] = validated_recipe
    if validated_structure is None:
        _EVALUATION_STATES_STRUCTURE_PROVIDERS.pop(analysis_type, None)
    else:
        _EVALUATION_STATES_STRUCTURE_PROVIDERS[analysis_type] = validated_structure


def unregister_analysis_recipe(analysis_type: str) -> None:
    """Remove a previously registered analysis recipe."""
    _ANALYSIS_RECIPES.pop(analysis_type, None)
    _EVALUATION_STATES_STRUCTURE_PROVIDERS.pop(analysis_type, None)


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


def _get_evaluation_states_structure_provider(
    analysis_type: str,
) -> EvaluationStatesStructureProviderProtocol | None:
    return _EVALUATION_STATES_STRUCTURE_PROVIDERS.get(analysis_type)


def resolve_analysis_run_authoring(
    value: AnalysisRunSpec | AnalysisRunDeltaSpec | Mapping[str, Any] | Path | str,
    *,
    repo_root: Path | str | None = None,
    run_alias_catalogs: Sequence[RunAliasCatalog | Mapping[str, Any]] = (),
) -> tuple[AnalysisRunSpec, FlattenedAnalysisRun | None]:
    """Resolve either analysis authoring kind to one strict ``AnalysisRunSpec``.

    Delta-authored documents are flattened against their content-pinned parents
    and then validated as the unchanged ``AnalysisRunSpec`` contract, so every
    downstream loader, executor, and manifest path stays identical. Direct
    ``AnalysisRunSpec`` documents are handled byte-identically and return a
    ``None`` flattening record.
    """
    if isinstance(value, AnalysisRunSpec):
        return value, None
    if isinstance(value, AnalysisRunDeltaSpec) or is_analysis_run_delta_payload(value):
        delta = (
            value
            if isinstance(value, AnalysisRunDeltaSpec)
            else AnalysisRunDeltaSpec.model_validate(value)
        )
        flattened = flatten_analysis_run_delta(delta, repo_root=repo_root)
        resolved_payload = resolve_run_aliases(flattened.payload, run_alias_catalogs)
        flattened = flattened.model_copy(update={"payload": resolved_payload})
        return AnalysisRunSpec.model_validate(resolved_payload), flattened
    raw: Mapping[str, Any]
    if isinstance(value, Mapping):
        raw = value
    else:
        raw = json.loads(Path(value).read_text(encoding="utf-8"))
    if is_analysis_run_delta_payload(raw):
        delta = AnalysisRunDeltaSpec.model_validate(raw)
        flattened = flatten_analysis_run_delta(delta, repo_root=repo_root)
        resolved_payload = resolve_run_aliases(flattened.payload, run_alias_catalogs)
        flattened = flattened.model_copy(update={"payload": resolved_payload})
        return AnalysisRunSpec.model_validate(resolved_payload), flattened
    source_version = raw.get("schema_version")
    result = default_spec_registry.migrate(
        "AnalysisRunSpec",
        raw,
        source_version=source_version if isinstance(source_version, str) else None,
        assume_current=source_version is None,
    )
    return AnalysisRunSpec.model_validate(
        resolve_run_aliases(result.payload, run_alias_catalogs)
    ), None


def coerce_analysis_run_spec(
    value: AnalysisRunSpec | AnalysisRunDeltaSpec | Mapping[str, Any] | Path | str,
    *,
    repo_root: Path | str | None = None,
    run_alias_catalogs: Sequence[RunAliasCatalog | Mapping[str, Any]] = (),
) -> AnalysisRunSpec:
    """Load an ``AnalysisRunSpec`` from an object, mapping, JSON file path, or delta spec."""
    return resolve_analysis_run_authoring(
        value,
        repo_root=repo_root,
        run_alias_catalogs=run_alias_catalogs,
    )[0]


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


def _analysis_input_authentication(
    spec: AnalysisRunSpec,
    ref: ParentRef,
) -> tuple[bool, bool]:
    """Return whether one input requires and declares manifest authentication."""

    requires_authenticated_evaluation = (
        ref.kind == "EvaluationRunManifest" and spec.evaluation_states_policy == "require_durable"
    )
    try:
        has_authenticated_claim = is_authenticated_manifest_ref(ref)
    except ValueError as exc:
        if requires_authenticated_evaluation:
            raise _authenticated_manifest_resolution_error(ref, exc) from exc
        raise
    if requires_authenticated_evaluation and not has_authenticated_claim:
        raise _resolution_error(
            code="custody_unavailable",
            manifest_id=ref.id,
            message=(
                "require_durable requires a content-authenticated EvaluationRunManifest "
                "authority with schema, version, SHA-256, and byte-size evidence."
            ),
            details={"required_ref_schema": "feedbax.ref.authenticated_manifest.v1"},
        )
    return requires_authenticated_evaluation, has_authenticated_claim


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
        manifest_input: ResolvedManifestInput | None = None
        states: Any = None
        state_source: AnalysisEvaluationStateSource | None = None
        requires_authenticated_evaluation, has_authenticated_claim = _analysis_input_authentication(
            spec, ref
        )
        if has_authenticated_claim:
            authenticated = (
                authenticated_inputs.get(index) if authenticated_inputs is not None else None
            )
            if authenticated is None:
                try:
                    authenticated = resolve_manifest_input(ref, root_path)
                except Exception as exc:
                    if requires_authenticated_evaluation:
                        raise _authenticated_manifest_resolution_error(ref, exc) from exc
                    raise
            manifest, manifest_path = authenticated.manifest, authenticated.path
            manifest_input = authenticated
        elif ref.kind.endswith("Manifest"):
            manifest, manifest_path = find_manifest_by_id(ref.id, root=root_path)
        if ref.kind == "EvaluationRunManifest":
            if spec.evaluation_states_policy == "require_durable":
                states, state_source = _load_required_durable_evaluation_states(
                    ref,
                    manifest,
                    root=root_path,
                    structure_provider=_get_evaluation_states_structure_provider(
                        spec.analysis_type
                    ),
                )
                resolved.append(
                    ResolvedAnalysisInput(
                        ref=ref,
                        manifest=manifest,
                        path=manifest_path,
                        states=states,
                        evaluation_state_source=state_source,
                        evaluation_state_receipt=_evaluation_state_receipt(states, state_source),
                        manifest_input=manifest_input,
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
                    cache_schema_version=EVALUATION_STATES_CACHE_SCHEMA_VERSION,
                    cache_key=ref.id,
                )
            else:
                if isinstance(manifest, EvaluationRunManifest):
                    try:
                        states = load_evaluation_states(manifest, root=root_path)
                    except EvaluationStatesArtifactNotFound:
                        rederived, rederived_path = _rederive_evaluation_states(
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
                            resulting_evaluation_manifest_authority=authenticated_manifest_ref(
                                rederived,
                                rederived_path,
                                "evaluation_run",
                            ),
                        )
                    else:
                        states_path.parent.mkdir(parents=True, exist_ok=True)
                        write_evaluation_states_cache(
                            states_path,
                            manifest_id=ref.id,
                            states=states,
                        )
                        state_source = _durable_state_source(
                            ref,
                            manifest,
                            manifest_path=manifest_path,
                        )
                else:
                    rederived, rederived_path = _rederive_evaluation_states(
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
                        resulting_evaluation_manifest_authority=authenticated_manifest_ref(
                            rederived,
                            rederived_path,
                            "evaluation_run",
                        ),
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
                evaluation_state_receipt=(
                    _evaluation_state_receipt(states, state_source)
                    if states is not None and state_source is not None
                    else None
                ),
                manifest_input=manifest_input,
            )
        )
    return resolved


def _portable_manifest_authority(ref: ParentRef) -> ParentRef:
    """Preserve authored authority fields without persisting a machine-local locator."""
    return ref.model_copy(update={"uri": None})


def _authenticated_manifest_resolution_error(
    ref: ParentRef,
    cause: BaseException,
) -> AnalysisEvaluationStatesResolutionError:
    message = str(cause)
    if isinstance(cause, (UnsupportedSpecVersion, ValidationError)):
        code: AnalysisEvaluationStateResolutionCode = "schema_mismatch"
    elif " kind mismatch" in message or " id mismatch" in message:
        code = "provenance_mismatch"
    else:
        code = "custody_unavailable"
    return _resolution_error(
        code=code,
        manifest_id=ref.id,
        message=(
            "require_durable could not authenticate the authored EvaluationRunManifest "
            f"authority: {cause}"
        ),
        details={"authority_failure_type": type(cause).__name__},
        cause=cause,
    )


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
        artifact
        for artifact in manifest.artifacts
        if artifact.role == EVALUATION_STATES_ARTIFACT_ROLE
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


def _durable_state_source(
    ref: ParentRef,
    manifest: EvaluationRunManifest,
    *,
    manifest_path: Path | None = None,
) -> AnalysisEvaluationStateSource:
    artifacts = [
        artifact
        for artifact in manifest.artifacts
        if artifact.role == EVALUATION_STATES_ARTIFACT_ROLE
    ]
    artifact = artifacts[0]
    authority = _portable_manifest_authority(ref)
    if not is_authenticated_manifest_ref(authority):
        if manifest_path is None:
            raise ValueError(
                "durable evaluation-state provenance requires the supplying manifest path"
            )
        authority = authenticated_manifest_ref(
            manifest,
            manifest_path,
            "evaluation_run",
        )
    return AnalysisEvaluationStateSource(
        source_kind="durable",
        requested_evaluation_manifest_id=ref.id,
        evaluation_manifest_authority=authority,
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
    structure_provider: EvaluationStatesStructureProviderProtocol | None,
) -> tuple[Any, AnalysisEvaluationStateSource]:
    manifest, artifact = _evaluation_states_artifact_for_durable_policy(ref, manifest)
    try:
        structure = structure_provider(manifest) if structure_provider is not None else None
    except Exception as exc:
        raise _resolution_error(
            code="custody_unavailable",
            manifest_id=ref.id,
            artifact_id=artifact.artifact_id,
            message=f"Evaluation states structure provider failed: {exc}",
            details={"provider_failure_type": type(exc).__name__, "provider_failure": str(exc)},
            cause=exc,
        ) from exc
    if structure is not None and not isinstance(structure, jtu.PyTreeDef):
        raise _resolution_error(
            code="custody_unavailable",
            manifest_id=ref.id,
            artifact_id=artifact.artifact_id,
            message=(
                "Evaluation states structure provider returned "
                f"{type(structure).__name__}, expected PyTreeDef."
            ),
        )
    try:
        states = load_authenticated_evaluation_states_artifact(
            artifact,
            root=root,
            manifest_id=manifest.id,
            structure=structure,
        )
    except EvaluationStatesSchemaMismatch as exc:
        raise _resolution_error(
            code="schema_mismatch",
            manifest_id=ref.id,
            artifact_id=artifact.artifact_id,
            message=f"Durable evaluation-state schema for {ref.id!r} is invalid: {exc}",
            cause=exc,
        ) from exc
    except EvaluationStatesProvenanceMismatch as exc:
        raise _resolution_error(
            code="provenance_mismatch",
            manifest_id=ref.id,
            artifact_id=artifact.artifact_id,
            message=f"Durable evaluation-state provenance for {ref.id!r} is invalid: {exc}",
            cause=exc,
        ) from exc
    except (EvaluationStatesCustodyUnavailable, EvaluationStatesHashMismatch) as exc:
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
) -> tuple[EvaluationRunManifest, Path]:
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
    rederived, path = execute_evaluation_run_spec(
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
    return rederived, path


def requested_outputs_from_spec(spec: AnalysisRunSpec) -> set[str] | None:
    """Return the requested output set encoded in analysis spec params."""
    raw = spec.params.get("requested_outputs", spec.params.get("outputs"))
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("AnalysisRunSpec params requested_outputs/outputs must be a list")
    outputs = {str(item) for item in raw}
    return outputs or None


def _resolve_authenticated_input_authorities(
    spec: AnalysisRunSpec,
    *,
    root: Path,
    execution_context: StagedExecutionContext,
) -> dict[int, ResolvedManifestInput]:
    """Authenticate authored manifest authorities before cache or recipe effects."""

    authenticated_inputs: dict[int, ResolvedManifestInput] = {}
    for index, ref in enumerate(spec.inputs):
        requires_authenticated_evaluation, has_authenticated_claim = _analysis_input_authentication(
            spec, ref
        )
        if not has_authenticated_claim:
            continue
        try:
            if execution_context.parent_execution_locations:
                authenticated_inputs[index] = execution_context.resolve_manifest_input(ref)
            else:
                authenticated_inputs[index] = resolve_manifest_input(ref, root)
        except Exception as exc:
            if requires_authenticated_evaluation:
                raise _authenticated_manifest_resolution_error(ref, exc) from exc
            raise
    return authenticated_inputs


def _preflight_checkpoint_binding_names(
    value: Any,
    execution_context: StagedExecutionContext,
) -> None:
    """Validate authored checkpoint binding names before recipe execution."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "checkpoint_custody_binding":
                if not isinstance(item, str) or not item:
                    raise StagedExecutionContextError(
                        "checkpoint_custody_binding must be a nonempty string"
                    )
                execution_context.checkpoint_custody_root(item)
            else:
                _preflight_checkpoint_binding_names(item, execution_context)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _preflight_checkpoint_binding_names(item, execution_context)


def execute_analysis_run_spec(
    spec: AnalysisRunSpec | AnalysisRunDeltaSpec | Mapping[str, Any] | Path | str,
    *,
    root: Path | str | None = None,
    repo_root: Path | str | None = None,
    run_alias_catalogs: Sequence[RunAliasCatalog | Mapping[str, Any]] = (),
    provenance: Provenance | None = None,
    issues: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    execution_context: StagedExecutionContext = EMPTY_STAGED_EXECUTION_CONTEXT,
    execution_descriptor: StagedExecutionDescriptor | Mapping[str, Any] | None = None,
    artifact_provider_bindings: Sequence[StagedArtifactProviderRootBinding] = (),
    manifest_root_bindings: Sequence[StagedManifestRootBinding] = (),
    checkpoint_custody_bindings: Sequence[StagedCheckpointCustodyRootBinding] = (),
    validate_result: AnalysisRecipeResultValidator | None = None,
    fig_dump_path: Path | str | None = None,
    fig_dump_formats: Sequence[str] = ("html",),
    use_cache: bool = True,
    force: bool = False,
) -> tuple[AnalysisRunManifest, Path]:
    """Execute a serialized analysis spec and write an ``AnalysisRunManifest``.

    Delta-authored specs are flattened against their content-pinned parents
    (which requires ``repo_root``) before execution.
    """
    run_spec, flattened = resolve_analysis_run_authoring(
        spec,
        repo_root=repo_root,
        run_alias_catalogs=run_alias_catalogs,
    )
    root_path = Path(root) if root is not None else default_manifest_root()
    explicit_runtime = (
        execution_descriptor is not None
        or bool(artifact_provider_bindings)
        or bool(manifest_root_bindings)
        or bool(checkpoint_custody_bindings)
    )
    if explicit_runtime:
        if execution_context is not EMPTY_STAGED_EXECUTION_CONTEXT:
            raise StagedExecutionContextError(
                "execution_context cannot be combined with direct runtime bindings"
            )
        execution_context = resolve_staged_execution_context(
            execution_descriptor,
            artifact_provider_bindings=artifact_provider_bindings,
            manifest_root_bindings=manifest_root_bindings,
            checkpoint_custody_bindings=checkpoint_custody_bindings,
        )
        authenticated_parents = [
            ref for ref in run_spec.inputs if is_authenticated_manifest_ref(ref)
        ]
        if authenticated_parents and (
            execution_context.manifest_roots or execution_context.opened_artifact_providers
        ):
            execution_context = with_staged_manifest_provider_inputs(
                execution_context,
                authenticated_parents,
            )
        _preflight_checkpoint_binding_names(run_spec.params, execution_context)
    execution_context = with_staged_repo_root(execution_context, repo_root)
    if flattened is not None:
        # Mirror the evaluation-matrix convention: delta-authored runs embed the
        # single canonical composition-provenance record into durable manifest
        # metadata. Direct specs (``flattened is None``) stay byte-identical.
        metadata = {
            **(metadata or {}),
            "analysis_composition": analysis_composition_provenance(flattened),
        }
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
        authenticated_inputs = _resolve_authenticated_input_authorities(
            run_spec,
            root=root_path,
            execution_context=execution_context,
        )
        recipe = get_analysis_recipe(run_spec.analysis_type)
        manifest_id = analysis_run_manifest_id(run_spec)
        if use_cache and not force:
            try:
                existing_manifest, existing_path = find_manifest_by_id(
                    manifest_id,
                    root=root_path,
                )
            except FileNotFoundError:
                pass
            else:
                if (
                    isinstance(existing_manifest, AnalysisRunManifest)
                    and existing_manifest.status == "completed"
                ):
                    return existing_manifest, existing_path

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
        from feedbax.analysis.execution import run_analyses_with_context

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
