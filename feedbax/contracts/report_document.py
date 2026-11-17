"""Ordered authored blocks for a report composed from saved products."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, StrictStr, field_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.project_inputs import NonemptyText, ProjectName


class ReportText(StrictModel):
    markdown: StrictStr

    @field_validator("markdown")
    @classmethod
    def _nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Report Markdown must contain text.")
        return value


class ReportFigure(StrictModel):
    figure: ProjectName
    caption: NonemptyText | None = None


class ReportDocument(StrictModel):
    """A report's order and captions, independent of product custody."""

    schema_version: Literal["feedbax.report_document.v1"] = "feedbax.report_document.v1"
    title: NonemptyText
    blocks: tuple[ReportText | ReportFigure, ...] = Field(min_length=1)
    completeness: Literal["complete", "allow_omissions"] = "complete"

    @property
    def figure_roles(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(block.figure for block in self.blocks if isinstance(block, ReportFigure))
        )
