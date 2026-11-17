"""Feedbax's reusable dataset figures."""

from typing import Annotated

from pydantic import Field, StrictInt

from feedbax.contracts.base import StrictModel
from feedbax.contracts.figure_document import FIGURE_DOCUMENT_SCHEMA, FigureDocument
from feedbax.plot.templates import figure_template


class VelocitySummarySettings(StrictModel):
    checkpoint_index: Annotated[StrictInt, Field(ge=0)] = 0


@figure_template(settings=VelocitySummarySettings, label="Velocity summary")
def VelocitySummary(settings: VelocitySummarySettings) -> FigureDocument:
    """Reach-aligned velocity profiles with a variation band for each condition."""
    return FigureDocument.model_validate({
        "schema_version": FIGURE_DOCUMENT_SCHEMA,
        "data": ["summary"],
        "theme": "plotly_white",
        "view": {"hovermode": "x unified"},
        "layout": [{
            "name": "velocity",
            "row": 1,
            "col": 1,
            "title": "Reach-aligned velocity by condition",
            "axes_labels": {"x": "Time (s)", "y": "Velocity (m/s)"},
        }],
        "layers": [{
            "name": "trial-summary",
            "mark": "band",
            "panel": "velocity",
            "role": "summary",
            "select": {"checkpoint": {"index": settings.checkpoint_index}},
            "group": ["condition"],
            "encode": {"x": "state_time", "y": "mean", "lower": "lower", "upper": "upper"},
            "style": {"label": "{condition}"},
        }],
    })
