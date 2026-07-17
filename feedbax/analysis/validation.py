"""Public validation helpers for registered analysis execution recipes."""

from __future__ import annotations

import inspect
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import jax.tree_util as jtu

from feedbax.contracts.manifest import (
    AnalysisRunSpec,
    EvaluationRunManifest,
    EvaluationRunSpec,
    ReportSpec,
)

if TYPE_CHECKING:
    from feedbax.analysis.execution_context import StagedExecutionContext
    from feedbax.analysis.evaluation import EvaluationRecipeResult
    from feedbax.analysis.reports import ReportRecipeResult, ResolvedReportInput
    from feedbax.analysis.specs import AnalysisRecipeResult, ResolvedAnalysisInput
else:
    EvaluationRecipeResult = Any
    ReportRecipeResult = Any
    ResolvedReportInput = Any
    AnalysisRecipeResult = Any
    ResolvedAnalysisInput = Any
    StagedExecutionContext = Any


class RecipeValidationError(ValueError):
    """Raised when a registered execution recipe violates its call contract."""


_TYPE_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")


def validate_namespaced_type_key(type_key: str, *, field: str = "type key") -> str:
    """Validate a producer-owned recipe type key.

    Evaluation, analysis, and report producers use lowercase dotted keys of the
    form ``<package>.<name>``. The first segment is the registering package's
    import name; later segments are package-owned names. Bare keys are rejected
    so downstream packages can register recipes without colliding with Feedbax
    or each other.
    """
    if not isinstance(type_key, str):
        raise RecipeValidationError(
            f"{field} must be a string; got {type(type_key).__name__}"
        )
    if not type_key.strip():
        raise RecipeValidationError(f"{field} must not be empty")
    if type_key != type_key.strip():
        raise RecipeValidationError(
            f"{field} must not contain leading or trailing whitespace: {type_key!r}"
        )
    if not _TYPE_KEY_PATTERN.fullmatch(type_key):
        raise RecipeValidationError(
            f"{field} must be a lowercase dotted key '<package>.<name>'; "
            f"got {type_key!r}. Each segment must start with a lowercase letter "
            "and contain only lowercase letters, digits, or underscores."
        )
    return type_key


class EvaluationRecipeProtocol(Protocol):
    """Callable protocol for registered ``EvaluationRunSpec`` recipes."""

    def __call__(
        self,
        run_spec: EvaluationRunSpec,
        root: Path,
        states_path: Path,
        execution_context: StagedExecutionContext,
        /,
    ) -> EvaluationRecipeResult:
        """Execute one evaluation run spec."""


class AnalysisRecipeProtocol(Protocol):
    """Callable protocol for registered executable ``AnalysisRunSpec`` recipes."""

    def __call__(
        self,
        run_spec: AnalysisRunSpec,
        root: Path,
        inputs: Sequence[ResolvedAnalysisInput],
        execution_context: StagedExecutionContext,
        /,
    ) -> AnalysisRecipeResult:
        """Build executable analyses for one analysis run spec."""


class EvaluationStatesStructureProviderProtocol(Protocol):
    """Callable protocol for supplying a typed evaluation-states structure."""

    def __call__(self, manifest: EvaluationRunManifest, /) -> jtu.PyTreeDef:
        """Return the trusted pytree structure for one evaluation manifest."""


class ReportRecipeProtocol(Protocol):
    """Callable protocol for registered executable ``ReportSpec`` recipes."""

    def __call__(
        self,
        report_spec: ReportSpec,
        root: Path,
        inputs: Sequence[ResolvedReportInput],
        /,
    ) -> ReportRecipeResult:
        """Build rendered report artifacts for one report spec."""


def validate_evaluation_recipe(
    evaluation_type: str,
    recipe: Any,
) -> EvaluationRecipeProtocol:
    """Validate and return a registered evaluation recipe callable."""
    validate_namespaced_type_key(evaluation_type, field="evaluation_type")
    _validate_callable_shape(
        kind="Evaluation recipe",
        type_key=evaluation_type,
        recipe=recipe,
        example_args=(object(), object(), object(), object()),
        expected="(run_spec, root, states_path, execution_context)",
    )
    return recipe


def validate_evaluation_batch_recipe(evaluation_type: str, recipe: Any) -> Any:
    """Validate and return a registered batched evaluation recipe callable."""
    validate_namespaced_type_key(evaluation_type, field="evaluation_type")
    _validate_callable_shape(
        kind="Evaluation batch recipe",
        type_key=evaluation_type,
        recipe=recipe,
        example_args=(object(), object()),
        expected="(items, execution_context)",
    )
    return recipe


def validate_analysis_recipe(
    analysis_type: str,
    recipe: Any,
) -> AnalysisRecipeProtocol:
    """Validate and return a registered analysis recipe callable."""
    validate_namespaced_type_key(analysis_type, field="analysis_type")
    _validate_callable_shape(
        kind="Analysis recipe",
        type_key=analysis_type,
        recipe=recipe,
        example_args=(object(), object(), object(), object()),
        expected="(run_spec, root, inputs, execution_context)",
    )
    return recipe


def validate_evaluation_states_structure_provider(
    analysis_type: str,
    provider: Any,
) -> EvaluationStatesStructureProviderProtocol:
    """Validate and return an evaluation-states structure provider callable."""
    _validate_callable_shape(
        kind="Evaluation states structure provider",
        type_key=analysis_type,
        recipe=provider,
        example_args=(object(),),
        expected="(evaluation_manifest)",
    )
    return provider


def validate_report_recipe(
    report_type: str,
    recipe: Any,
) -> ReportRecipeProtocol:
    """Validate and return a registered report recipe callable."""
    validate_namespaced_type_key(report_type, field="report_type")
    _validate_callable_shape(
        kind="Report recipe",
        type_key=report_type,
        recipe=recipe,
        example_args=(object(), object(), object()),
        expected="(report_spec, root, inputs)",
    )
    return recipe


def _validate_callable_shape(
    *,
    kind: str,
    type_key: str,
    recipe: Any,
    example_args: tuple[object, ...],
    expected: str,
) -> None:
    if not callable(recipe):
        raise RecipeValidationError(
            f"{kind} {type_key!r} must be callable with signature {expected}; "
            f"got non-callable {type(recipe).__name__}."
        )

    try:
        signature = inspect.signature(recipe)
    except (TypeError, ValueError) as exc:
        raise RecipeValidationError(
            f"{kind} {type_key!r} must expose an inspectable signature {expected}; "
            f"{exc}"
        ) from exc

    try:
        signature.bind(*example_args)
    except TypeError as exc:
        argument_count = {2: "two", 3: "three", 4: "four"}.get(
            len(example_args),
            str(len(example_args)),
        )
        raise RecipeValidationError(
            f"{kind} {type_key!r} must accept {argument_count} positional arguments "
            f"{expected}; signature {signature} is invalid: {exc}"
        ) from exc
