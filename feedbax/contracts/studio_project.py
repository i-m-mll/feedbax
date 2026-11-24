"""Studio reads and edits the shared authored project on disk."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, FiniteFloat, JsonValue, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.jobs import StudioProjectWindowMapProblem
from feedbax.contracts.project import Document, Project, Sha256, Step
from feedbax.contracts.project import ProjectValueEdit as ProjectValueEdit
from feedbax.contracts.project import RemoveProjectValue as RemoveProjectValue
from feedbax.contracts.project import SetProjectValue as SetProjectValue
from feedbax.contracts.project_inputs import NonemptyText, ProjectName
from feedbax.contracts.project_io import ProjectConfiguration


class PresentationPosition(StrictModel):
    x: FiniteFloat
    y: FiniteFloat


class PresentationCamera(PresentationPosition):
    zoom: Annotated[FiniteFloat, Field(gt=0)]


class DocumentPresentation(StrictModel):
    positions: dict[str, PresentationPosition] = Field(default_factory=dict)
    camera: PresentationCamera | None = None
    flipped: dict[str, bool] = Field(default_factory=dict)
    collapsed: dict[str, bool] = Field(default_factory=dict)
    waypoints: dict[str, list[PresentationPosition]] = Field(default_factory=dict)


class StepPresentation(StrictModel):
    sweep_order: list[ProjectName] = Field(default_factory=list)
    model_positions: dict[ProjectName, dict[str, PresentationPosition]] = Field(default_factory=dict)


class PresentationDocument(StrictModel):
    documents: dict[ProjectName, DocumentPresentation] = Field(default_factory=dict)
    steps: dict[ProjectName, StepPresentation] = Field(default_factory=dict)
    visible_groups: list[ProjectName] = Field(default_factory=list)


class StudioProjectRevision(StrictModel):
    project_id: ProjectName
    revision: Sha256
    presentation_revision: Sha256


class AuthoredFileAddress(StrictModel):
    kind: Literal["document", "step", "variant", "presentation", "declaration"]
    id: ProjectName


PRESENTATION_ADDRESS = AuthoredFileAddress(kind="presentation", id="presentation")
DECLARATION_ADDRESS = AuthoredFileAddress(kind="declaration", id="declaration")


class StudioAuthoredFileRevision(StrictModel):
    address: AuthoredFileAddress
    revision: Sha256
    data: str | None = None


class StudioAuthoredDeclaration(StrictModel):
    revision: Sha256
    authored_revision: Sha256
    data: str


class StudioProjectSnapshot(StudioProjectRevision):
    project: Project
    semantic_root: Sha256
    presentation: PresentationDocument | None
    presentation_error: Literal["unreadable_presentation"] | None = None
    file_revisions: list[StudioAuthoredFileRevision]
    declaration: StudioAuthoredDeclaration | None = None
    settings: dict[str, JsonValue] = Field(default_factory=dict)
    default_keep_superseded_days: int | None = Field(default=None, ge=0)
    composed: dict[str, JsonValue] = Field(default_factory=dict)
    """The composed payload of each based document whose base chain composes."""
    problems: list[StudioProjectWindowMapProblem] = Field(default_factory=list)


class StudioProjectFilePatch(StrictModel):
    address: AuthoredFileAddress
    patches: list[ProjectValueEdit] = Field(min_length=1)


class StudioProjectPatch(StudioProjectFilePatch):
    revision: Sha256
    presentation_revision: Sha256 | None = None


class StudioProjectPatchBatch(StrictModel):
    revision: Sha256
    presentation_revision: Sha256 | None = None
    files: list[StudioProjectFilePatch] = Field(min_length=1)
    created: list[CreateNamedDocument] = Field(default_factory=list)
    """Documents created in the same transaction as the file patches."""


class ReplayFileAddress(StrictModel):
    kind: Literal["document", "step", "variant"]
    id: ProjectName


class ReplayDeclarationAddress(StrictModel):
    kind: Literal["declaration"]
    id: Literal["declaration"]


class StudioReplayFile(StrictModel):
    address: ReplayFileAddress | ReplayDeclarationAddress
    before: Sha256 | None
    after: str | None


class StudioReplayPresentation(StrictModel):
    before: Sha256
    after: PresentationDocument


class StudioProjectReplay(StrictModel):
    revision: Sha256
    files: list[StudioReplayFile] = Field(min_length=1)
    presentation: StudioReplayPresentation | None = None


class StudioPresentationWrite(StrictModel):
    revision: Sha256
    presentation: PresentationDocument


class StudioProjectDocument(StrictModel):
    configuration: ProjectConfiguration
    snapshot: StudioProjectSnapshot


class StudioProjectRename(StrictModel):
    revision: Sha256
    name: ProjectName


type MemberFamily = Literal["documents", "steps"]
type KeyedFamily = Literal["documents", "steps", "elements"]


class _KeyedMember(StrictModel):
    """A member; an operation or figure is named with its Analysis document, and only it is."""

    family: KeyedFamily
    member: ProjectName
    document: ProjectName | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def _element_document(self) -> _KeyedMember:
        if (self.family == "elements") != (self.document is not None):
            raise ValueError("an operation or figure is named with its document, and only it")
        return self


class RenameMember(_KeyedMember):
    """Set or clear the label of a document, step, operation or figure; no key, reference, run or request changes."""

    kind: Literal["rename"]
    label: NonemptyText | None


class ChangeKey(_KeyedMember):
    """Change the key of a document, step, operation or figure, and rewrite every authored reference to it.

    An operation's or figure's key changes across its document's family.
    """

    kind: Literal["change_key"]
    key: ProjectName


class DuplicateMember(StrictModel):
    """Copy a document or step and its presentation under a new name, generated when omitted.

    A document's copy is a root document holding the source's composed content; an Analysis document's copy gives
    its operations and figures keys no other document holds.
    """

    kind: Literal["duplicate"]
    family: MemberFamily
    member: ProjectName
    name: ProjectName | None = None


class DetachMember(StrictModel):
    """Replace a based document by a root document holding its composed content."""

    kind: Literal["detach"]
    member: ProjectName


class SplitCard(StrictModel):
    """Move some of an Analyze card's operations and figures to a new card reading a new Analysis document.

    A wire the cut crosses becomes a card-to-card edge, and no request changes.
    """

    kind: Literal["split"]
    member: ProjectName
    elements: Annotated[list[ProjectName], Field(min_length=1)]


class MergeCards(StrictModel):
    """Join Analyze cards into ``member``, the inverse of Split; card-to-card edges between them become wires."""

    kind: Literal["merge"]
    member: ProjectName
    others: Annotated[list[ProjectName], Field(min_length=1)]


class StudioMemberCommand(StrictModel):
    revision: Sha256
    presentation_revision: Sha256
    command: Annotated[
        RenameMember | ChangeKey | DuplicateMember | DetachMember | SplitCard | MergeCards, Field(discriminator="kind"),
    ]


class StudioMemberCommandResult(StrictModel):
    member: ProjectName
    snapshot: StudioProjectSnapshot
    problems: list[StudioProjectWindowMapProblem] = Field(default_factory=list)


class StudioTrainStep(Step):
    kind: Literal["train"]


class StudioEvaluateStep(Step):
    kind: Literal["evaluate"]


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
        StudioTrainStep | StudioEvaluateStep | StudioAnalyzeStep | StudioReportStep, Field(discriminator="kind")
    ]


type NamedFileCreation = Annotated[
    CreateNamedDocument | CreateNamedStep, Field(discriminator="kind")
]


class StudioProjectCreateFile(StrictModel):
    revision: Sha256
    file: NamedFileCreation
