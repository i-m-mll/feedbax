"""Authored figure presets available in managed Analyze."""

from importlib.resources import files

from feedbax.contracts.figure_document import FigureDocument


def velocity_summary_figure() -> FigureDocument:
    return FigureDocument.model_validate_json(
        files("feedbax.analysis").joinpath("velocity-summary.figure.json").read_bytes()
    )
