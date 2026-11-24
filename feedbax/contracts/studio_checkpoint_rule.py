"""Revision-bound Studio edits to an Evaluate subject's checkpoint rule."""

from typing import Annotated, Literal

from pydantic import Field

from feedbax.contracts.base import StrictModel
from feedbax.contracts.jobs import StudioSavedTrainKey
from feedbax.contracts.model_edit import AxisValueRuns, ListedRuns, StepRuns
from feedbax.contracts.project_inputs import BatchCheckpoint
from feedbax.contracts.request_refs import Sha256, UpdateRule
from feedbax.contracts.studio_project import StudioProjectFilePatch
from feedbax.contracts.studio_scope import StudioStoreChoice


class StudioCheckpointRuleSet(StrictModel):
    kind: Literal["set"]
    rule: Literal["last"] | BatchCheckpoint | UpdateRule
    scope: Annotated[StepRuns | AxisValueRuns | ListedRuns, Field(discriminator="kind")]


class StudioCheckpointRuleToSweep(StrictModel):
    kind: Literal["to_sweep"]


class StudioCheckpointRuleRequest(StrictModel):
    revision: Sha256
    run: StudioSavedTrainKey
    edit: Annotated[StudioCheckpointRuleSet | StudioCheckpointRuleToSweep, Field(discriminator="kind")]
    store_choice: Sha256 | None = None


class StudioCheckpointRuleEdit(StrictModel):
    revision: Sha256
    files: list[StudioProjectFilePatch]
    store_choices: list[StudioStoreChoice] | None = None
