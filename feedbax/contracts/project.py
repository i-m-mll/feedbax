"""The semantic project shared by text authoring and Studio."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue, StringConstraints, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.project_inputs import InputBinding, NonemptyText, ProjectName, RowKey
from feedbax.contracts.project_sweep import PatchPath, Sweep
from feedbax.contracts.run_matrix import MatrixCompositionDelta

type Payload = dict[str, JsonValue]
type Pointer = Annotated[str, StringConstraints(pattern=r"^(?:/(?:[^~/]|~[01])*)*$")]
type Sha256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
type StepKind = Literal["train", "evaluate", "analyze", "report"]


def json_pointer_parts(pointer: str | None) -> tuple[str, ...]:
    """Decode an already validated JSON Pointer."""
    if not pointer:
        return ()
    return tuple(token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/"))


class Document(StrictModel):
    schema_id: NonemptyText = Field(alias="schema")
    payload: JsonValue


class DocumentReference(StrictModel):
    document: ProjectName
    pointer: Pointer | None = None


class InlineBase(StrictModel):
    inline: Payload


class StepBase(StrictModel):
    step: ProjectName
    row: RowKey | None = None


class RunBase(StrictModel):
    run: NonemptyText


class ExternalBase(StrictModel):
    path: NonemptyText
    sha256: Sha256
    pointer: Pointer | None = None


type Base = InlineBase | DocumentReference | StepBase | RunBase | ExternalBase


class StepNote(StrictModel):
    issue: str | None = None
    reason: str | None = None
    replaces: str | None = None


class Step(StrictModel):
    kind: StepKind
    base: Base
    deltas: list[MatrixCompositionDelta] = Field(default_factory=list)
    sweep: Sweep | None = None
    inputs: dict[ProjectName, InputBinding | None] = Field(default_factory=dict)
    requires: list[PatchPath] = Field(default_factory=list)
    note: StepNote | None = None
    group: ProjectName | None = None


class Project(StrictModel):
    documents: dict[ProjectName, Document] = Field(default_factory=dict)
    steps: dict[ProjectName, Step] = Field(default_factory=dict)
    sweep: Sweep | None = None

    @model_validator(mode="after")
    def _check_project_variants(self) -> Project:
        if self.sweep is not None and any(row is None for row in self.sweep.rows.values()):
            raise ValueError("project variants have no inherited rows to remove")
        return self
