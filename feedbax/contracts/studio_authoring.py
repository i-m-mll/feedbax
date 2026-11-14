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
from feedbax.contracts.studio_field import StudioSweepList, StudioSweepRange, StudioSweepSampler
from feedbax.contracts.studio_project import ProjectValueEdit


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


class StudioAuthoredDocumentFamily(StrictModel):
    schema_id: str | None
    payload_schema: dict[str, JsonValue]


class StudioAuthoringOrigin(StrictModel):
    plugin: str
    distribution: str | None


class StudioAuthoringRegistry(StrictModel):
    entries: list[ComponentDefinition | OperationDefinition | StudioAuthoringMethod | FunctionInfo]
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


    sweep: Annotated[StudioSweepList | StudioSweepRange | StudioSweepSampler, Field(discriminator="kind")] | None = None
    can_sweep: bool = False


class StudioAuthoringChange(StrictModel):
    field: str
    value: str | StrictInt | StrictFloat


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
    change: StudioAuthoringChange | StudioAuthoringSweepChange | None = None


class StudioTrainAuthoring(StrictModel):
    project_id: ProjectName
    step_id: ProjectName
    revision: Sha256
    fields: list[StudioAuthoringField]
    patches: list[ProjectValueEdit]
    documents: list[ProjectName]
