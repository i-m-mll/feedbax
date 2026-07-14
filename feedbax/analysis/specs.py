"""Executable registry for manifest-canonical analysis run specs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from feedbax.analysis.analysis import AbstractAnalysis
from feedbax.analysis.context import AnalysisRunContext
from feedbax.analysis.evaluation import (
    EvaluationStatesArtifactNotFound,
    execute_evaluation_run_spec,
    load_evaluation_states,
    load_evaluation_states_cache,
    write_evaluation_states_cache,
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
    AnalysisRunManifest,
    AnalysisRunSpec,
    AnyManifest,
    EvaluationRunManifest,
    EvaluationRunSpec,
    ParentRef,
    Provenance,
    default_manifest_root,
    analysis_run_manifest_id,
    evaluation_states_cache_path,
    load_manifest,
)
from feedbax.persistence.manifest_index import find_manifest_paths_by_id, iter_manifest_files
from feedbax.analysis.types import AnalysisInputData


@dataclass(frozen=True)
class ResolvedAnalysisInput:
    """A manifest parent resolved from an ``AnalysisRunSpec`` input ref."""

    ref: ParentRef
    manifest: AnyManifest | None
    path: Path | None
    states: Any = None


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
    if isinstance(value, Mapping):
        return AnalysisRunSpec.model_validate(value)
    path = Path(value)
    return AnalysisRunSpec.model_validate_json(path.read_text(encoding="utf-8"))


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
            states_path = evaluation_states_cache_path(ref.id, root=root_path)
            if not states_path.exists():
                if isinstance(manifest, EvaluationRunManifest):
                    try:
                        states = load_evaluation_states(manifest, root=root_path)
                    except EvaluationStatesArtifactNotFound:
                        _rederive_evaluation_states(
                            ref.id,
                            manifest,
                            root=root_path,
                            execution_context=execution_context,
                        )
                    else:
                        states_path.parent.mkdir(parents=True, exist_ok=True)
                        write_evaluation_states_cache(
                            states_path,
                            manifest_id=ref.id,
                            states=states,
                        )
                else:
                    _rederive_evaluation_states(
                        ref.id,
                        manifest,
                        root=root_path,
                        execution_context=execution_context,
                    )
            if states is None:
                states = load_evaluation_states_cache(states_path, manifest_id=ref.id)
        resolved.append(
            ResolvedAnalysisInput(
                ref=ref,
                manifest=manifest,
                path=manifest_path,
                states=states,
            )
        )
    return resolved


def _rederive_evaluation_states(
    manifest_id: str,
    manifest: AnyManifest | None,
    *,
    root: Path,
    execution_context: StagedExecutionContext,
) -> None:
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
