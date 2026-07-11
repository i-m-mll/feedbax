from __future__ import annotations

from pathlib import Path

import pytest

from feedbax.analysis.evaluation import (
    EvaluationRecipeResult,
    register_evaluation_recipe,
    registered_evaluation_recipes,
    unregister_evaluation_recipe,
)
from feedbax.contracts.figures import FigurePiece, FigureTemplate
from feedbax.contracts.manifest import ArtifactRef, EvaluationRunSpec
from feedbax.plot.constructors import (
    get_figure_piece,
    get_figure_template,
    register_figure_piece,
    register_figure_template,
    registered_figure_pieces,
    registered_figure_templates,
    unregister_figure_piece,
    unregister_figure_template,
)


def test_downstream_figure_recipes_are_listed_and_can_be_unregistered() -> None:
    template = FigureTemplate(
        name="downstream.condition_comparison",
        description="Compare resolved condition rows.",
        assembler="feedbax.grid_figure",
        facet_by=["condition"],
    )
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
    try:
        register_figure_template(template)
        register_figure_piece(piece)

        assert template in registered_figure_templates()
        assert piece in registered_figure_pieces()
        assert get_figure_template(template.name) is template
        assert get_figure_piece(piece.name) is piece
    finally:
        unregister_figure_template(template.name)
        unregister_figure_piece(piece.name)

    with pytest.raises(ValueError, match="Registered templates:"):
        get_figure_template(template.name)
    with pytest.raises(ValueError, match="Registered pieces:"):
        get_figure_piece(piece.name)


def test_downstream_computation_recipes_are_listed() -> None:
    def recipe(
        run_spec: EvaluationRunSpec,
        root: Path,
        states_path: Path,
    ) -> EvaluationRecipeResult:
        del run_spec, root, states_path
        return EvaluationRecipeResult(summary_metrics={"value": 1.0})

    try:
        register_evaluation_recipe("downstream.checkpoint_metric", recipe)
        assert "downstream.checkpoint_metric" in registered_evaluation_recipes()
    finally:
        unregister_evaluation_recipe("downstream.checkpoint_metric")
