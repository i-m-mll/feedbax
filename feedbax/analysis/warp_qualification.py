"""Evaluation registration for the authored MJX-Warp physics fixtures."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from feedbax.analysis.evaluation import (
    EvaluationAuthoringSchema,
    EvaluationBatchItem,
    EvaluationBatchRowError,
    EvaluationRecipeRegistry,
    EvaluationRecipeResult,
)
from feedbax.analysis.evaluation_compaction import (
    EvaluationBatchConsumerInput,
    EvaluationBatchConsumerRegistry,
    EvaluationBatchFinalizeInput,
    EvaluationBatchFragment,
    EvaluationBatchMergeInput,
    EvaluationBatchMergeState,
)
from feedbax.analysis.execution_context import StagedExecutionContext
from feedbax.contracts.base import StrictModel
from feedbax.contracts.manifest import EvaluationRunSpec


EVALUATION_TYPE = "feedbax.mjx_warp_qualification"
EXTERNAL_MUSCLE_EVALUATION_TYPE = "feedbax.mjx_warp_external_muscles"
RECEIPT_CONSUMER_ID = "feedbax.mjx_warp_receipt"
RECEIPT_CONSUMER_VERSION = "v1"
RECEIPT_SCHEMA_ID = "feedbax.manifest.mjx_warp_receipt"
RECEIPT_SCHEMA_VERSION = f"{RECEIPT_SCHEMA_ID}.v1"
RECEIPT_ROLE = "warp_qualification_receipt"


class WarpQualificationParams(StrictModel):
    device: Literal["cpu", "cuda"]


def _qualify(spec: EvaluationRunSpec) -> EvaluationRecipeResult:
    if spec.evaluation_type == EXTERNAL_MUSCLE_EVALUATION_TYPE:
        from feedbax.mechanics.warp_muscle_qualification import run_qualification
    else:
        from feedbax.mechanics.warp_qualification import run_qualification

    if spec.training_run_ids or spec.inputs:
        raise ValueError("MJX-Warp qualification uses only its authored synthetic fixtures")
    params = WarpQualificationParams.model_validate(spec.params)
    receipt = run_qualification(backend="warp", device=params.device, fixture="all")
    return EvaluationRecipeResult(
        states=receipt,
        summary_metrics={"qualified_fixtures": len(receipt["fixtures"])},
        metadata={"states_schema": RECEIPT_SCHEMA_VERSION},
    )


def _evaluate(
    spec: EvaluationRunSpec,
    _root: Path,
    _states_path: Path,
    _context: StagedExecutionContext,
    /,
) -> EvaluationRecipeResult:
    return _qualify(spec)


def _evaluate_batch(
    items: Sequence[EvaluationBatchItem], _context: StagedExecutionContext
) -> Sequence[EvaluationRecipeResult]:
    (item,) = items
    try:
        return (_qualify(item.spec),)
    except Exception as error:
        raise EvaluationBatchRowError(item.row_id, error) from error


def _compact(value: EvaluationBatchConsumerInput) -> EvaluationBatchFragment:
    (receipt,) = value.states
    return EvaluationBatchFragment(receipt, RECEIPT_SCHEMA_ID, RECEIPT_SCHEMA_VERSION, RECEIPT_ROLE)


def _merge(value: EvaluationBatchMergeInput) -> EvaluationBatchMergeState:
    if value.prior_merge_state is not None:
        raise ValueError("MJX-Warp qualification produces exactly one receipt per matrix")
    return EvaluationBatchMergeState(value.fragment, RECEIPT_SCHEMA_ID, RECEIPT_SCHEMA_VERSION)


def _finalize(value: EvaluationBatchFinalizeInput) -> EvaluationBatchFragment:
    return EvaluationBatchFragment(
        value.terminal_merge_state, RECEIPT_SCHEMA_ID, RECEIPT_SCHEMA_VERSION, RECEIPT_ROLE
    )


def register_warp_qualification(
    recipes: EvaluationRecipeRegistry, consumers: EvaluationBatchConsumerRegistry
) -> None:
    for evaluation_type in (EVALUATION_TYPE, EXTERNAL_MUSCLE_EVALUATION_TYPE):
        schema_id = evaluation_type.replace("feedbax.", "feedbax.spec.", 1)
        recipes.register(evaluation_type, _evaluate, batch_recipe=_evaluate_batch)
        recipes.register_authoring_schema(
            evaluation_type,
            EvaluationAuthoringSchema(
                schema_id=schema_id,
                schema_version=f"{schema_id}.v1",
                params_model=WarpQualificationParams,
                axis_profiles=({"device": ("cpu",)}, {"device": ("cuda",)}),
            ),
        )
    consumers.register(
        RECEIPT_CONSUMER_ID,
        RECEIPT_CONSUMER_VERSION,
        compact=_compact,
        merge=_merge,
        finalize=_finalize,
        accepts_fresh_states=True,
    )
