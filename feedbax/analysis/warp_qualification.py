"""Evaluation registration for the authored MJX-Warp physics fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from feedbax.analysis.evaluation import (
    EvaluationAuthoringSchema,
    EvaluationRecipeRegistry,
    EvaluationRecipeResult,
)
from feedbax.analysis.execution_context import StagedExecutionContext
from feedbax.contracts.base import StrictModel
from feedbax.contracts.manifest import EvaluationRunSpec


EVALUATION_TYPE = "feedbax.mjx_warp_qualification"
_PARAMS_SCHEMA = "feedbax.spec.mjx_warp_qualification"


class WarpQualificationParams(StrictModel):
    device: Literal["cpu", "cuda"]


def _evaluate(
    spec: EvaluationRunSpec,
    _root: Path,
    _states_path: Path,
    _context: StagedExecutionContext,
    /,
) -> EvaluationRecipeResult:
    from feedbax.mechanics.warp_qualification import run_qualification

    if spec.training_run_ids or spec.inputs:
        raise ValueError("MJX-Warp qualification uses only its authored synthetic fixtures")
    params = WarpQualificationParams.model_validate(spec.params)
    receipt = run_qualification(backend="warp", device=params.device, fixture="all")
    return EvaluationRecipeResult(
        summary_metrics={"qualified_fixtures": len(receipt["fixtures"])},
        metadata={"warp_qualification": receipt},
    )


def register_warp_qualification(registry: EvaluationRecipeRegistry) -> None:
    registry.register(EVALUATION_TYPE, _evaluate)
    registry.register_authoring_schema(
        EVALUATION_TYPE,
        EvaluationAuthoringSchema(
            schema_id=_PARAMS_SCHEMA,
            schema_version=f"{_PARAMS_SCHEMA}.v1",
            params_model=WarpQualificationParams,
            axis_profiles=({"device": ("cpu",)}, {"device": ("cuda",)}),
        ),
    )
