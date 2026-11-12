"""Wire contracts for authoring discovery and revision-bound field editing."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue, StrictFloat, StrictInt

from feedbax.contracts.base import StrictModel
from feedbax.contracts.project import Sha256
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.studio_project import SetProjectValue


class StudioAuthoringMethod(StrictModel):
    key: str
    schema_id: str
    schema_version: str
    owner: str
    package: str | None
    plugin: str | None
    payload_schema: dict[str, JsonValue]


class StudioAuthoringSurface(StrictModel):
    project_grammar: dict[str, JsonValue]
    step_payloads: dict[str, dict[str, JsonValue]]
    document_schemas: dict[str, dict[str, JsonValue]]
    training_methods: list[StudioAuthoringMethod]


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
    kind: Literal["document", "step"]
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
    field: str
    value: str | StrictInt | StrictFloat


class StudioTrainAuthoringRequest(StrictModel):
    revision: Sha256
    change: StudioAuthoringChange | None = None


class StudioTrainAuthoring(StrictModel):
    project_id: ProjectName
    step_id: ProjectName
    revision: Sha256
    fields: list[StudioAuthoringField]
    patches: list[SetProjectValue]
    documents: list[ProjectName]
