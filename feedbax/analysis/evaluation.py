"""Registered execution for manifest-canonical evaluation runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dill as pickle

from feedbax.contracts.evaluation_states import (
    EVALUATION_STATES_ARTIFACT_ROLE,
    load_evaluation_states_artifact,
    store_evaluation_states_artifact,
)
from feedbax.contracts.manifest import (
    ArtifactRef,
    EntrypointRef,
    EvaluationRunManifest,
    EvaluationRunSpec,
    ManifestStatus,
    Provenance,
    collect_git_provenance,
    default_manifest_root,
    evaluation_run_manifest_id,
    evaluation_states_cache_path,
    load_manifest,
    safe_manifest_key,
    spec_payload,
    write_manifest,
)
from feedbax.analysis.validation import EvaluationRecipeProtocol, validate_evaluation_recipe


@dataclass(frozen=True)
class EvaluationRecipeResult:
    """Result returned by an evaluation recipe."""

    states: Any = None
    summary_metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[ArtifactRef] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


EvaluationRecipe = EvaluationRecipeProtocol

_EVALUATION_RECIPES: dict[str, EvaluationRecipe] = {}


class EvaluationRecipeExecutionError(RuntimeError):
    """Raised after a registered recipe fails and a failed manifest is written."""

    def __init__(self, manifest: EvaluationRunManifest, path: Path, cause: BaseException):
        super().__init__(
            f"Evaluation recipe for {manifest.id!r} failed; failed manifest written to {path}"
        )
        self.manifest = manifest
        self.path = path
        self.__cause__ = cause


class EvaluationStatesArtifactNotFound(LookupError):
    """Raised when a manifest has no durable evaluation-states artifact."""


def register_evaluation_recipe(
    evaluation_type: str,
    recipe: EvaluationRecipe,
    *,
    replace: bool = False,
) -> None:
    """Register an executable evaluation recipe by stable type key."""
    if not evaluation_type.strip():
        raise ValueError("evaluation_type must not be empty")
    if evaluation_type in _EVALUATION_RECIPES and not replace:
        raise ValueError(f"Evaluation recipe {evaluation_type!r} is already registered")
    _EVALUATION_RECIPES[evaluation_type] = validate_evaluation_recipe(evaluation_type, recipe)


def unregister_evaluation_recipe(evaluation_type: str) -> None:
    """Remove a previously registered evaluation recipe."""
    _EVALUATION_RECIPES.pop(evaluation_type, None)


def get_evaluation_recipe(evaluation_type: str) -> EvaluationRecipe:
    """Return a registered recipe or raise a clear unsupported-execution error."""
    try:
        return _EVALUATION_RECIPES[evaluation_type]
    except KeyError as exc:
        available = ", ".join(sorted(_EVALUATION_RECIPES)) or "none"
        raise ValueError(
            f"Evaluation recipe {evaluation_type!r} is not registered. "
            f"Registered evaluation recipes: {available}."
        ) from exc


def coerce_evaluation_run_spec(value: EvaluationRunSpec | Mapping[str, Any] | Path | str) -> EvaluationRunSpec:
    """Load an ``EvaluationRunSpec`` from an object, mapping, or JSON file path."""
    if isinstance(value, EvaluationRunSpec):
        return value
    if isinstance(value, Mapping):
        return EvaluationRunSpec.model_validate(value)
    path = Path(value)
    return EvaluationRunSpec.model_validate_json(path.read_text(encoding="utf-8"))


def execute_evaluation_run_spec(
    spec: EvaluationRunSpec | Mapping[str, Any] | Path | str,
    *,
    root: Path | str | None = None,
    provenance: Provenance | None = None,
    issues: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    use_cache: bool = True,
    force: bool = False,
) -> tuple[EvaluationRunManifest, Path]:
    """Execute a serialized evaluation spec and write a truthful manifest.

    The evaluated states cache is ephemeral and lives under
    ``{FEEDBAX_RUNS_DIR}/cache/states`` keyed by the deterministic evaluation
    manifest identifier, not by a Studio database hash.
    """
    run_spec = coerce_evaluation_run_spec(spec)
    recipe = get_evaluation_recipe(run_spec.evaluation_type)
    root_path = Path(root) if root is not None else default_manifest_root()
    manifest_id = evaluation_run_manifest_id(run_spec)
    states_path = evaluation_states_cache_path(manifest_id, root=root_path)
    manifest_path = (
        root_path / "manifests" / "evaluation_runs" / f"{safe_manifest_key(manifest_id)}.json"
    )
    states_path.parent.mkdir(parents=True, exist_ok=True)

    prov = (
        provenance.model_copy(deep=True)
        if provenance is not None
        else collect_git_provenance()
    )
    prov.parents = list(run_spec.inputs)
    if issues:
        prov.issues.extend(issue for issue in issues if issue not in prov.issues)
    if prov.entrypoint is None:
        prov.entrypoint = EntrypointRef(
            kind="feedbax-evaluation-recipe",
            name=run_spec.evaluation_type,
        )

    manifest_metadata = dict(metadata or {})
    states_custody = _states_custody_for_spec(run_spec)
    cache_metadata: dict[str, Any] = {
        "states_path": str(states_path),
        "states_cache_key": manifest_id,
        "states_cache_hit": False,
    }
    try:
        cache_hit = False
        if use_cache and not force and states_path.exists():
            with states_path.open("rb") as stream:
                states = pickle.load(stream)
            previous_manifest = _load_completed_evaluation_manifest(manifest_path, manifest_id)
            summary_metrics = dict(previous_manifest.summary_metrics) if previous_manifest else {}
            artifacts = list(previous_manifest.artifacts) if previous_manifest else []
            result_metadata = (
                dict(previous_manifest.metadata) if previous_manifest else {}
            )
            result = EvaluationRecipeResult(
                states=states,
                summary_metrics=summary_metrics,
                artifacts=artifacts,
                metadata=result_metadata,
            )
            cache_metadata["states_cache_hit"] = True
            cache_hit = True
        else:
            result = recipe(run_spec, root_path, states_path)
            if use_cache and result.states is not None:
                with states_path.open("wb") as stream:
                    pickle.dump(result.states, stream)
                cache_metadata["states_cache_saved"] = True
        if states_custody == "durable":
            result = _with_durable_states_artifact(
                result,
                root=root_path,
                manifest_id=manifest_id,
                allow_existing=cache_hit,
            )

        summary_metrics = {
            "input_training_runs": len(run_spec.inputs),
            **result.summary_metrics,
        }
        manifest = _build_evaluation_manifest(
            manifest_id=manifest_id,
            run_spec=run_spec,
            status="completed",
            provenance=prov,
            summary_metrics=summary_metrics,
            artifacts=result.artifacts,
            metadata={
                **manifest_metadata,
                **result.metadata,
                "cache": cache_metadata,
            },
        )
        return manifest, write_manifest(manifest, root=root_path)
    except Exception as exc:
        manifest = _build_evaluation_manifest(
            manifest_id=manifest_id,
            run_spec=run_spec,
            status="failed",
            provenance=prov,
            summary_metrics={"input_training_runs": len(run_spec.inputs)},
            artifacts=[],
            metadata={
                **manifest_metadata,
                "cache": cache_metadata,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            },
        )
        path = write_manifest(manifest, root=root_path)
        raise EvaluationRecipeExecutionError(manifest, path, exc) from exc


def _load_completed_evaluation_manifest(
    path: Path,
    manifest_id: str,
) -> EvaluationRunManifest | None:
    if not path.exists():
        return None
    manifest = load_manifest(path)
    if not isinstance(manifest, EvaluationRunManifest):
        return None
    if manifest.id != manifest_id or manifest.status != "completed":
        return None
    return manifest


def load_evaluation_states(
    manifest: EvaluationRunManifest,
    *,
    root: Path | str | None = None,
) -> Any:
    """Load durable evaluation states from a completed evaluation manifest."""
    artifact = _evaluation_states_artifact(manifest)
    root_path = Path(root) if root is not None else default_manifest_root()
    return load_evaluation_states_artifact(artifact, root=root_path)


def _states_custody_for_spec(run_spec: EvaluationRunSpec) -> str:
    raw = run_spec.params.get("states_custody", "cache")
    if raw not in {"cache", "durable"}:
        raise ValueError(
            "EvaluationRunSpec params states_custody must be 'cache' or 'durable'; "
            f"got {raw!r}"
        )
    return str(raw)


def _evaluation_states_artifact(manifest: EvaluationRunManifest) -> ArtifactRef:
    matches = [
        artifact
        for artifact in manifest.artifacts
        if artifact.role == EVALUATION_STATES_ARTIFACT_ROLE
    ]
    if not matches:
        raise EvaluationStatesArtifactNotFound(
            f"Evaluation manifest {manifest.id!r} has no evaluation_states artifact."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Evaluation manifest {manifest.id!r} has multiple evaluation_states artifacts."
        )
    return matches[0]


def _with_durable_states_artifact(
    result: EvaluationRecipeResult,
    *,
    root: Path,
    manifest_id: str,
    allow_existing: bool = False,
) -> EvaluationRecipeResult:
    existing = [
        artifact
        for artifact in result.artifacts
        if artifact.role == EVALUATION_STATES_ARTIFACT_ROLE
    ]
    if existing and allow_existing:
        if len(existing) > 1:
            raise ValueError(
                f"Evaluation manifest {manifest_id!r} has multiple evaluation_states artifacts."
            )
        return result
    if existing:
        raise ValueError(
            "Evaluation recipes must not emit evaluation_states artifacts directly; "
            "durable state custody is handled by execute_evaluation_run_spec."
        )
    artifact = store_evaluation_states_artifact(
        result.states,
        root=root,
        manifest_id=manifest_id,
    )
    return EvaluationRecipeResult(
        states=result.states,
        summary_metrics=result.summary_metrics,
        artifacts=[*result.artifacts, artifact],
        metadata=result.metadata,
    )


def _build_evaluation_manifest(
    *,
    manifest_id: str,
    run_spec: EvaluationRunSpec,
    status: ManifestStatus,
    provenance: Provenance,
    summary_metrics: dict[str, Any],
    artifacts: list[ArtifactRef],
    metadata: dict[str, Any],
) -> EvaluationRunManifest:
    return EvaluationRunManifest(
        id=manifest_id,
        status=status,
        evaluation_spec=spec_payload(
            "EvaluationRunSpec",
            run_spec.model_dump(mode="json", exclude_none=True),
        ),
        input_training_runs=list(run_spec.inputs),
        summary_metrics=summary_metrics,
        provenance=provenance,
        artifacts=artifacts,
        metadata=metadata,
    )
