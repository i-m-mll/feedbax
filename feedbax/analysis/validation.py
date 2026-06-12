"""Public validation helpers for registered analysis execution recipes."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from feedbax.manifest import AnalysisRunSpec, EvaluationRunSpec

if TYPE_CHECKING:
    from feedbax.analysis.evaluation import EvaluationRecipeResult
    from feedbax.analysis.specs import AnalysisRecipeResult, ResolvedAnalysisInput
else:
    EvaluationRecipeResult = Any
    AnalysisRecipeResult = Any
    ResolvedAnalysisInput = Any


class RecipeValidationError(ValueError):
    """Raised when a registered execution recipe violates its call contract."""


class EvaluationRecipeProtocol(Protocol):
    """Callable protocol for registered ``EvaluationRunSpec`` recipes."""

    def __call__(
        self,
        run_spec: EvaluationRunSpec,
        root: Path,
        states_path: Path,
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
        /,
    ) -> AnalysisRecipeResult:
        """Build executable analyses for one analysis run spec."""


def validate_evaluation_recipe(
    evaluation_type: str,
    recipe: Any,
) -> EvaluationRecipeProtocol:
    """Validate and return a registered evaluation recipe callable."""
    _validate_callable_shape(
        kind="Evaluation recipe",
        type_key=evaluation_type,
        recipe=recipe,
        example_args=(object(), object(), object()),
        expected="(run_spec, root, states_path)",
    )
    return recipe


def validate_analysis_recipe(
    analysis_type: str,
    recipe: Any,
) -> AnalysisRecipeProtocol:
    """Validate and return a registered analysis recipe callable."""
    _validate_callable_shape(
        kind="Analysis recipe",
        type_key=analysis_type,
        recipe=recipe,
        example_args=(object(), object(), object()),
        expected="(run_spec, root, inputs)",
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
        raise RecipeValidationError(
            f"{kind} {type_key!r} must accept three positional arguments "
            f"{expected}; signature {signature} is invalid: {exc}"
        ) from exc
