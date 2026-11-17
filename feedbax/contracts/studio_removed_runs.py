"""Revision-bound review of the runs an authored edit takes out of the saved Plan."""

from feedbax.contracts.base import StrictModel
from feedbax.contracts.jobs import StudioSavedTrainKey
from feedbax.contracts.request_refs import Sha256
from feedbax.contracts.studio_project import StudioProjectFilePatch


class StudioRemovedRunsRequest(StrictModel):
    revision: Sha256
    files: list[StudioProjectFilePatch]


class StudioRemovedRuns(StrictModel):
    """The planned runs whose request the edit removes or replaces, so their results leave the project."""

    revision: Sha256
    runs: list[StudioSavedTrainKey]
