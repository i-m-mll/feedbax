"""Pytest-able conformance helpers for downstream evaluation producers.

The v1 producer contract is intentionally small and runnable in downstream CI:
register a namespaced evaluation type, publish or explicitly waive a params
schema family, return states with ``metadata["states_schema"]``, and let
Feedbax own manifest/cache custody through ``execute_evaluation_run_spec``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from feedbax.analysis.evaluation import (
    EvaluationRecipe,
    EvaluationRecipeExecutionError,
    coerce_evaluation_run_spec,
    execute_evaluation_run_spec,
    get_evaluation_recipe,
    register_evaluation_recipe,
    STATES_SCHEMA_METADATA_KEY,
)
from feedbax.analysis.validation import (
    validate_evaluation_recipe,
    validate_namespaced_type_key,
)
from feedbax.contracts.manifest import EvaluationRunSpec
from feedbax.contracts.migrations import SpecSchemaRegistry, default_spec_registry


SpecFactory = Callable[[], EvaluationRunSpec | Mapping[str, Any] | Path | str]


@dataclass(frozen=True)
class EvaluationRecipeContractReport:
    """Summary returned by :func:`check_evaluation_recipe` after passing checks."""

    evaluation_type: str
    manifest_id: str
    params_schema_family: str
    params_schema_waived: bool
    states_schema: str
    cache_round_trip: bool
    failure_manifest_id: str


def evaluation_params_schema_family_id(evaluation_type: str) -> str:
    """Return the expected params schema family id for an evaluation type."""
    validate_namespaced_type_key(evaluation_type, field="evaluation_type")
    package, name = evaluation_type.split(".", 1)
    return f"{package}.spec.evaluation.{name}"


def check_evaluation_recipe(
    evaluation_type: str,
    spec_factory: SpecFactory,
    *,
    root: Path | str | None = None,
    registry: SpecSchemaRegistry = default_spec_registry,
    params_schema_waiver: str | None = None,
) -> EvaluationRecipeContractReport:
    """Assert one registered evaluation recipe satisfies the v1 producer contract.

    Args:
        evaluation_type: Registered recipe key. Must be ``<package>.<name>``.
        spec_factory: Callable returning an ``EvaluationRunSpec``, mapping, or
            JSON path for a representative successful run.
        root: Optional manifest/cache root. A temporary root is used by default.
        registry: Schema registry used to check the params schema family.
        params_schema_waiver: Explicit reason allowing a producer to ship before
            its params family is registered. Downstream CI should omit this once
            a family exists or when strict params governance is enabled.

    Returns:
        A compact report for downstream tests to inspect or snapshot.

    Raises:
        AssertionError: If the recipe violates any conformance check.
        RecipeValidationError: If the registered type key or callable shape is
            invalid.
    """
    validate_namespaced_type_key(evaluation_type, field="evaluation_type")
    recipe = get_evaluation_recipe(evaluation_type)
    validate_evaluation_recipe(evaluation_type, recipe)
    spec = coerce_evaluation_run_spec(spec_factory())
    assert spec.evaluation_type == evaluation_type, (
        "spec_factory returned EvaluationRunSpec for "
        f"{spec.evaluation_type!r}, expected {evaluation_type!r}"
    )

    params_schema_family = evaluation_params_schema_family_id(evaluation_type)
    params_schema_waived = _params_schema_waived_or_registered(
        params_schema_family,
        registry=registry,
        waiver=params_schema_waiver,
    )

    if root is None:
        with TemporaryDirectory(prefix="feedbax-eval-contract-") as tmp:
            return _check_evaluation_recipe_in_root(
                evaluation_type,
                recipe,
                spec,
                root=Path(tmp),
                params_schema_family=params_schema_family,
                params_schema_waived=params_schema_waived,
            )
    return _check_evaluation_recipe_in_root(
        evaluation_type,
        recipe,
        spec,
        root=Path(root),
        params_schema_family=params_schema_family,
        params_schema_waived=params_schema_waived,
    )


def _params_schema_waived_or_registered(
    params_schema_family: str,
    *,
    registry: SpecSchemaRegistry,
    waiver: str | None,
) -> bool:
    registered = any(
        family.kind == params_schema_family or family.identity == params_schema_family
        for family in registry.families()
    )
    if registered:
        return False
    if waiver and waiver.strip():
        return True
    raise AssertionError(
        "Evaluation params schema family is not registered. Expected a "
        f"SpecSchemaFamily with kind or schema_id {params_schema_family!r}, "
        "or pass params_schema_waiver with an explicit temporary reason."
    )


def _check_evaluation_recipe_in_root(
    evaluation_type: str,
    recipe: EvaluationRecipe,
    spec: EvaluationRunSpec,
    *,
    root: Path,
    params_schema_family: str,
    params_schema_waived: bool,
) -> EvaluationRecipeContractReport:
    manifest, path = execute_evaluation_run_spec(
        spec,
        root=root,
        use_cache=True,
        force=True,
    )
    assert path.exists(), f"completed manifest path does not exist: {path}"
    assert manifest.status == "completed", f"expected completed manifest, got {manifest.status!r}"

    states_schema = manifest.metadata.get(STATES_SCHEMA_METADATA_KEY)
    assert isinstance(states_schema, str) and states_schema.strip(), (
        "Evaluation recipes that return states must declare a non-empty "
        f"metadata[{STATES_SCHEMA_METADATA_KEY!r}] string."
    )
    cache = manifest.metadata.get("cache")
    assert isinstance(cache, dict), "completed manifest metadata must include cache details"
    states_path = cache.get("states_path")
    assert isinstance(states_path, str) and Path(states_path).exists(), (
        "evaluation states cache was not written"
    )

    cached_manifest, cached_path = execute_evaluation_run_spec(
        spec,
        root=root,
        use_cache=True,
        force=False,
    )
    assert cached_path == path, "cache round-trip rewrote a different manifest path"
    assert cached_manifest.id == manifest.id, "cache round-trip changed manifest id"
    assert cached_manifest.metadata["cache"]["states_cache_hit"] is True, (
        "second execution did not load states from the executor cache"
    )

    failure_manifest_id = _check_failure_manifest(
        evaluation_type,
        recipe,
        spec,
        root=root,
    )

    return EvaluationRecipeContractReport(
        evaluation_type=evaluation_type,
        manifest_id=manifest.id,
        params_schema_family=params_schema_family,
        params_schema_waived=params_schema_waived,
        states_schema=states_schema,
        cache_round_trip=True,
        failure_manifest_id=failure_manifest_id,
    )


def _check_failure_manifest(
    evaluation_type: str,
    recipe: EvaluationRecipe,
    spec: EvaluationRunSpec,
    *,
    root: Path,
) -> str:
    failing_spec = spec.model_copy(
        deep=True,
        update={
            "params": {
                **spec.params,
                "__feedbax_contract_probe__": "failure-manifest",
            }
        },
    )

    def failing_recipe(
        _run_spec: EvaluationRunSpec,
        _root: Path,
        _states_path: Path,
    ) -> None:
        raise RuntimeError("feedbax evaluation contract failure probe")

    register_evaluation_recipe(evaluation_type, failing_recipe, replace=True)
    try:
        try:
            execute_evaluation_run_spec(
                failing_spec,
                root=root,
                use_cache=False,
                force=True,
            )
        except EvaluationRecipeExecutionError as exc:
            assert exc.path.exists(), f"failed manifest path does not exist: {exc.path}"
            assert exc.manifest.status == "failed", (
                f"expected failed manifest, got {exc.manifest.status!r}"
            )
            assert exc.manifest.metadata["error"]["type"] == "RuntimeError"
            return exc.manifest.id
        raise AssertionError("failing evaluation recipe did not raise an execution error")
    finally:
        register_evaluation_recipe(evaluation_type, recipe, replace=True)
