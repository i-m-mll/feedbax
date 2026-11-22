"""Revision-bound freezing of a saved measured subject."""

from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from feedbax.contracts.base import StrictModel
from feedbax.contracts.project_inputs import ProjectName, RowKey
from feedbax.contracts.request_refs import Sha256
from feedbax.contracts.studio_project import StudioProjectFilePatch


class StudioSubsetToRowsRequest(StrictModel):
    revision: Sha256
    variant: RowKey
    step: ProjectName
    input: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_-]*(?:\.(?:0|[1-9][0-9]*))?$")]
    """The selection to freeze: its role, or its role and its index in a list."""
    attempts: list[str] = Field(min_length=1)


class StudioSubsetToRowsEdit(StrictModel):
    revision: Sha256
    files: list[StudioProjectFilePatch]


class StudioRunRuleRequest(StrictModel):
    """A run rule as it is being written, against the runs of the step it reads."""

    revision: Sha256
    variant: RowKey
    step: ProjectName
    """The producer: the step whose runs the rule chooses."""
    rule: str
    ranked: bool = True
    """Whether the rule may rank by a result (``lowest``, ``highest``); a sweep exception may not."""


class StudioRunRuleChosen(StrictModel):
    kind: Literal["chosen"] = "chosen"
    runs: list[RowKey]
    """The producer's rows the rule's axis terms choose; a lowest or highest term ranks among them."""
    count: int
    """The runs the selection gives: the rows chosen, or the ranks a lowest or highest term asks for."""


class StudioRunRuleRefused(StrictModel):
    kind: Literal["refused"] = "refused"
    reason: str


class StudioRunRuleCheck(StrictModel):
    """What a run rule chooses among the producer's ``total`` rows, or why it is refused."""

    total: int
    result: Annotated[StudioRunRuleChosen | StudioRunRuleRefused, Field(discriminator="kind")]
