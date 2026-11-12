"""Studio reads and edits the shared authored project on disk."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, FiniteFloat, JsonValue

from feedbax.contracts.base import StrictModel
from feedbax.contracts.project import Document, Pointer, Project, Sha256, Step
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.project_io import ProjectConfiguration


class PresentationPosition(StrictModel):
    x: FiniteFloat
    y: FiniteFloat


class PresentationCamera(PresentationPosition):
    zoom: Annotated[FiniteFloat, Field(gt=0)]


class DocumentPresentation(StrictModel):
    positions: dict[str, PresentationPosition] = Field(default_factory=dict)
    camera: PresentationCamera | None = None


class StepPresentation(StrictModel):
    sweep_order: list[ProjectName] = Field(default_factory=list)
    folded_sweep_id: ProjectName | None = None


class PresentationDocument(StrictModel):
    documents: dict[ProjectName, DocumentPresentation] = Field(default_factory=dict)
    steps: dict[ProjectName, StepPresentation] = Field(default_factory=dict)
    visible_groups: list[ProjectName] = Field(default_factory=list)


class StudioProjectRevision(StrictModel):
    project_id: ProjectName
    revision: Sha256
    presentation_revision: Sha256


class AuthoredFileAddress(StrictModel):
    kind: Literal["document", "step", "variant"]
    id: ProjectName


class StudioAuthoredFileRevision(StrictModel):
    address: AuthoredFileAddress
    revision: Sha256


class StudioProjectSnapshot(StudioProjectRevision):
    project: Project
    semantic_root: Sha256
    presentation: PresentationDocument | None
    presentation_error: Literal["unreadable_presentation"] | None = None
    file_revisions: list[StudioAuthoredFileRevision]


type EditPath = Annotated[Pointer, Field(min_length=1)]


class SetProjectValue(StrictModel):
    op: Literal["add", "replace"]
    path: EditPath
    value: JsonValue


class RemoveProjectValue(StrictModel):
    op: Literal["remove"]
    path: EditPath


type ProjectValueEdit = Annotated[SetProjectValue | RemoveProjectValue, Field(discriminator="op")]


class StudioProjectFilePatch(StrictModel):
    address: AuthoredFileAddress
    patches: list[ProjectValueEdit] = Field(min_length=1)


class StudioProjectPatch(StudioProjectFilePatch):
    revision: Sha256


class StudioProjectPatchBatch(StrictModel):
    revision: Sha256
    files: list[StudioProjectFilePatch] = Field(min_length=1)


class StudioPresentationWrite(StrictModel):
    revision: Sha256
    presentation: PresentationDocument


class StudioProjectDocument(StrictModel):
    configuration: ProjectConfiguration
    snapshot: StudioProjectSnapshot


class StudioProjectRename(StrictModel):
    revision: Sha256
    name: ProjectName


class StudioTrainStep(Step):
    kind: Literal["train"]


class StudioAnalyzeStep(Step):
    kind: Literal["analyze"]


class StudioReportStep(Step):
    kind: Literal["report"]


class CreateNamedDocument(StrictModel):
    kind: Literal["document"]
    id: ProjectName
    document: Document


class CreateNamedStep(StrictModel):
    kind: Literal["step"]
    id: ProjectName
    step: Annotated[
        StudioTrainStep | StudioAnalyzeStep | StudioReportStep, Field(discriminator="kind")
    ]


type NamedFileCreation = Annotated[
    CreateNamedDocument | CreateNamedStep, Field(discriminator="kind")
]


class StudioProjectCreateFile(StrictModel):
    revision: Sha256
    file: NamedFileCreation
