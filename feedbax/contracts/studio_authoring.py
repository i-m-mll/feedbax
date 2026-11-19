"""Wire contracts for authoring discovery and revision-bound field editing."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue, StrictFloat, StrictInt

from feedbax.contracts.base import StrictModel
from feedbax.contracts.component import ComponentDefinition
from feedbax.expression_registry import FunctionInfo
from feedbax.contracts.operation import OperationDefinition
from feedbax.contracts.project import Sha256
from feedbax.contracts.project_inputs import NonemptyText, ProjectName
from feedbax.contracts.studio_project import ProjectValueEdit


class StudioAuthoringDeclaration(StrictModel):
    key: str
    label: str
    renamed_from: tuple[str, ...]
    settings_schema: dict[str, JsonValue]


class StudioAuthoringSurface(StrictModel):
    project_grammar: dict[str, JsonValue]
    step_payloads: dict[str, dict[str, JsonValue]]
    document_schemas: dict[str, dict[str, JsonValue]]
    training_methods: list[StudioAuthoringDeclaration]


class StudioAuthoredDocumentFamily(StrictModel):
    schema_id: str | None
    payload_schema: dict[str, JsonValue]


class StudioAuthoringOrigin(StrictModel):
    package: str
    distribution: str
    version: str


class StudioAuthoringRegistry(StrictModel):
    entries: list[
        ComponentDefinition | OperationDefinition | StudioAuthoringDeclaration | FunctionInfo
    ]
    origins: dict[str, StudioAuthoringOrigin]


class StudioAuthoringCatalog(StrictModel):
    """Authored schemas and the registry declarations those schemas can name."""

    project_grammar: dict[str, JsonValue]
    step_grammar: dict[str, JsonValue]
    step_payloads: dict[str, dict[str, JsonValue]]
    document_families: dict[str, StudioAuthoredDocumentFamily]
    registries: dict[str, StudioAuthoringRegistry]


class StudioAuthoringOption(StrictModel):
    value: str
    label: str


class StudioAuthoringChoice(StrictModel):
    kind: Literal["choice"]
    options: list[StudioAuthoringOption]


class StudioAuthoringNumber(StrictModel):
    kind: Literal["number"]
    min: float | None = None
    max: float | None = None
    step: int | None = None


class StudioAuthoringSource(StrictModel):
    kind: Literal["document", "step", "variant"]
    name: str
    path: str


class StudioAuthoringField(StrictModel):
    id: str
    label: str
    value: str | int | float | None
    source: StudioAuthoringSource | None
    control: (
        Annotated[StudioAuthoringChoice | StudioAuthoringNumber, Field(discriminator="kind")] | None
    )


class StudioAuthoringChange(StrictModel):
    """Sets a declared field, or any value under ``/settings`` that the method's settings model accepts.

    At ``/method`` or a class-valued field the value is the chosen method's or class's key; the server keeps each
    current setting the chosen one declares and accepts, and takes its defaults for the rest.
    """

    field: str
    value: JsonValue


class StudioToSweep(StrictModel):
    kind: Literal["toSweep"]


class StudioSweepEdit(StrictModel):
    kind: Literal["addValue", "toList"]
    sweep_id: ProjectName


class StudioToField(StrictModel):
    kind: Literal["toField"]
    sweep_id: ProjectName
    value_id: ProjectName


class StudioSetSweepValue(StrictModel):
    kind: Literal["setSweepValue"]
    sweep_id: ProjectName
    value_id: ProjectName
    value: StrictInt | StrictFloat


class StudioNameSweepValue(StrictModel):
    kind: Literal["nameSweepValue"]
    sweep_id: ProjectName
    value_id: ProjectName
    label: NonemptyText | None


class StudioAuthoringSweepChange(StrictModel):
    field: str
    edit: Annotated[
        StudioToSweep | StudioSweepEdit | StudioToField | StudioSetSweepValue | StudioNameSweepValue,
        Field(discriminator="kind"),
    ]


class StudioTrainAuthoringRequest(StrictModel):
    revision: Sha256
    change: StudioAuthoringChange | None = None


class StudioSweptParameterMissing(StrictModel):
    """A class that lacks a parameter the step sweeps, so the step cannot take it; both named by their titles."""

    class_label: NonemptyText
    parameter_label: NonemptyText


class StudioTrainAuthoring(StrictModel):
    project_id: ProjectName
    step_id: ProjectName
    revision: Sha256
    fields: list[StudioAuthoringField]
    settings: dict[str, JsonValue] | None
    patches: list[ProjectValueEdit]
    documents: list[ProjectName]
    refusal: StudioSweptParameterMissing | None = None
