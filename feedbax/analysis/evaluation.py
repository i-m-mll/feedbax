"""Registered execution for manifest-canonical evaluation runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dill as pickle

from feedbax.manifest import (
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
    cache_metadata: dict[str, Any] = {
        "states_path": str(states_path),
        "states_cache_key": manifest_id,
        "states_cache_hit": False,
    }
    try:
        if use_cache and not force and states_path.exists():
            with states_path.open("rb") as stream:
                states = pickle.load(stream)
            result = EvaluationRecipeResult(
                states=states,
                summary_metrics={"states_cache_hit": True},
                metadata={"states_cache_hit": True},
            )
            cache_metadata["states_cache_hit"] = True
        else:
            result = recipe(run_spec, root_path, states_path)
            if use_cache and result.states is not None:
                with states_path.open("wb") as stream:
                    pickle.dump(result.states, stream)
                cache_metadata["states_cache_saved"] = True

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
