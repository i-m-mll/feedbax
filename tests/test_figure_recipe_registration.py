from __future__ import annotations

from pathlib import Path

from feedbax.analysis.evaluation import (
    EvaluationRecipeRegistry,
    EvaluationRecipeResult,
)
from feedbax.contracts.figures import FigurePiece
from feedbax.contracts.base import ArtifactRef
from feedbax.contracts.manifest import EvaluationRunSpec
from feedbax.plot.constructors import FigureRegistry


def test_downstream_figure_pieces_are_isolated_and_queryable() -> None:
    registry = FigureRegistry()
    piece = FigurePiece(
        name="downstream.metric_trace",
        description="Render one computed row metric.",
        artifact_ref=ArtifactRef(
            role="figure_piece",
            logical_name="metric.json",
            media_type="application/json",
            uri="/tmp/downstream-metric.json",
        ),
        label="Metric",
        constructor="feedbax.profile_band",
    )
    registry.register_piece(piece)

    assert piece in registry.pieces()
    assert registry.piece(piece.name) == piece

    isolated = FigureRegistry()
    assert isolated.piece(piece.name) is None


def test_downstream_computation_recipes_are_listed() -> None:
    registry = EvaluationRecipeRegistry()

    def recipe(
        run_spec: EvaluationRunSpec,
        root: Path,
        states_path: Path,
        execution_context,
    ) -> EvaluationRecipeResult:
        del run_spec, root, states_path, execution_context
        return EvaluationRecipeResult(summary_metrics={"value": 1.0})

    registry.register("downstream.checkpoint_metric", recipe)
    assert "downstream.checkpoint_metric" in registry.keys()
