"""Wire contracts for authoring discovery and revision-bound field editing."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue, StrictBool, StrictFloat, StrictInt, StrictStr

from feedbax.contracts.base import StrictModel
from feedbax.contracts.component import ComponentDefinition
from feedbax.contracts.graph import WireSpec
from feedbax.contracts.jobs import StudioSavedTrainKey
from feedbax.contracts.operation import OperationDefinition
from feedbax.contracts.project import DocumentOverridePath, Sha256
from feedbax.contracts.project_inputs import NonemptyText, ProjectName
from feedbax.contracts.project_sweep import RangeValues, SampleValues
from feedbax.contracts.studio_project import PresentationPosition, ProjectValueEdit, StudioProjectFilePatch


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
    entries: list[ComponentDefinition | OperationDefinition | StudioAuthoringMethod]
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


class StudioDocumentAuthoringChange(StrictModel):
    scope: Literal["variant"] | None = None
    lit_steps: list[ProjectName] = Field(min_length=1)
    path: DocumentOverridePath
    value: StrictStr | StrictInt | StrictFloat | StrictBool


class StudioModelAddComponent(StrictModel):
    kind: Literal["addComponent"]
    component: Literal["feedbax.GRU", "feedbax.Linear", "feedbax.PointMass"]
    revision: Annotated[StrictInt, Field(gt=0)]
    position: PresentationPosition | None = None


class StudioModelConnectPorts(StrictModel):
    kind: Literal["connectPorts"]
    wire: WireSpec


class StudioModelExposeOutput(StrictModel):
    kind: Literal["exposeOutput"]
    source_node: NonemptyText
    source_port: NonemptyText


class StudioModelAuthoringChange(StrictModel):
    lit_steps: list[ProjectName] = Field(min_length=1)
    operation: Annotated[
        StudioModelAddComponent | StudioModelConnectPorts | StudioModelExposeOutput,
        Field(discriminator="kind"),
    ]


class StudioDocumentAuthoringRequest(StrictModel):
    revision: Sha256
    run: StudioSavedTrainKey
    field: Literal["graph", "task"]
    presentation_revision: Sha256 | None = None
    change: StudioDocumentAuthoringChange | StudioModelAuthoringChange | None = None


class StudioDocumentAuthoring(StrictModel):
    revision: Sha256
    document: ProjectName
    payload: JsonValue
    sources: dict[str, StudioAuthoringSource]
    variant_model_paths: list[DocumentOverridePath] | None = None
    variant_model_reason: NonemptyText | None = None
    files: list[StudioProjectFilePatch]
    created_component: str | None = None
    presentation_revision: Sha256 | None = None


class StudioSweepValue(StrictModel):
    id: ProjectName
    label: str | None
    value: int | float


class StudioSweepList(StrictModel):
    kind: Literal["list"]
    id: ProjectName
    values: list[StudioSweepValue]
    editable: bool


class StudioSweepRange(StrictModel):
    kind: Literal["range"]
    id: ProjectName
    rule: RangeValues
    can_list: bool


class StudioSweepSampler(StrictModel):
    kind: Literal["sampler"]
    id: ProjectName
    rule: SampleValues
    can_list: bool


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


ReachCoordinate = Annotated[StrictFloat, Field(ge=-3.4028234663852886e38, le=3.4028234663852886e38)]


class StudioReachPosition(StrictModel):
    x: ReachCoordinate
    y: ReachCoordinate


class StudioReachTargetEdit(StrictModel):
    kind: Literal["bindReachTarget"]
    input: NonemptyText
    position: StudioReachPosition


class StudioReachTargetAuthoring(StrictModel):
    inputs: list[StudioAuthoringOption]
    input: str | None
    position: StudioReachPosition | None


class StudioTrainAuthoringRequest(StrictModel):
    revision: Sha256
    change: StudioAuthoringChange | StudioAuthoringSweepChange | StudioReachTargetEdit | None = None


class StudioTrainAuthoring(StrictModel):
    project_id: ProjectName
    step_id: ProjectName
    revision: Sha256
    fields: list[StudioAuthoringField]
    patches: list[ProjectValueEdit]
    documents: list[ProjectName]
    reach_target: StudioReachTargetAuthoring | None = None
