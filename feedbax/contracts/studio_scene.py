"""Read-only Scene geometry projected from a named authored Model."""

from typing import Annotated, Literal

from pydantic import Field, FiniteFloat, JsonValue

from feedbax.contracts.base import StrictModel
from feedbax.contracts.domain import DomainDiagnostic
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.representation import RepresentationMusclePathPointSpec
from feedbax.contracts.request_refs import Sha256


class ScenePlanarLink(StrictModel):
    body_id: str
    joint_id: str
    parent_frame: str
    child_frame: str
    qpos_index: int = Field(ge=0)
    parent_pivot: tuple[FiniteFloat, FiniteFloat]
    body_pivot: tuple[FiniteFloat, FiniteFloat]
    reference_angle: FiniteFloat
    length: FiniteFloat = Field(gt=0)


class ScenePlanarMuscle(StrictModel):
    node_id: str
    points: Annotated[list[RepresentationMusclePathPointSpec], Field(min_length=2)] | None


class ScenePlanarModel(StrictModel):
    model_identity: Sha256
    world_frame: str
    links: list[ScenePlanarLink] = Field(min_length=1)
    reference_qpos: list[FiniteFloat] = Field(min_length=1)
    muscles: list[ScenePlanarMuscle]


class StudioPlanarSceneRequest(StrictModel):
    revision: Sha256
    document_id: ProjectName
    node_path: list[str] = Field(min_length=1)


class StudioPlanarSceneResponse(StudioPlanarSceneRequest):
    project_id: ProjectName
    model: ScenePlanarModel | None
    diagnostics: list[DomainDiagnostic]


type ScenePoint = tuple[FiniteFloat, FiniteFloat]


class SceneLinkGeometry(StrictModel):
    body: ModelAddress
    joint: ModelAddress
    start: ScenePoint
    end: ScenePoint
    pivot: ScenePoint


class SceneMuscleGeometry(StrictModel):
    address: ModelAddress
    points: Annotated[list[ScenePoint], Field(min_length=2)]


class ScenePointGeometry(StrictModel):
    kind: Literal["point"]
    address: ModelAddress
    point: ModelAddress
    position: ScenePoint


class ScenePlanarGeometry(StrictModel):
    kind: Literal["planar"]
    address: ModelAddress
    links: Annotated[list[SceneLinkGeometry], Field(min_length=1)]
    muscles: list[SceneMuscleGeometry]


type SceneMechanismGeometry = Annotated[
    ScenePointGeometry | ScenePlanarGeometry, Field(discriminator="kind")
]


class StudioSceneModelSource(StrictModel):
    revision: Sha256
    document_id: ProjectName


class StudioSceneGeometryRequest(StudioSceneModelSource):
    state_values: dict[str, JsonValue] = Field(default_factory=dict)
    """Trial initial values or one recorded step, including configuration, by state Model address."""


class StudioSceneGeometryResponse(StudioSceneModelSource):
    project_id: ProjectName
    mechanisms: list[SceneMechanismGeometry]
    diagnostics: list[DomainDiagnostic]
