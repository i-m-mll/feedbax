"""Read-only Scene geometry projected from a named authored Model."""

from typing import Annotated, Literal

from pydantic import Field, FiniteFloat, JsonValue

from feedbax.contracts.base import StrictModel
from feedbax.contracts.domain import DomainDiagnostic
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.request_refs import Sha256


type ScenePoint = tuple[FiniteFloat, FiniteFloat]


class SceneAnchorGeometry(StrictModel):
    address: ModelAddress
    signal: ModelAddress | None
    position: ScenePoint


class SceneLinkGeometry(StrictModel):
    body: ModelAddress
    body_name: str
    joint: ModelAddress
    joint_name: str
    start: ScenePoint
    end: ScenePoint
    pivot: ScenePoint


class SceneMuscleGeometry(StrictModel):
    address: ModelAddress
    name: str
    points: Annotated[list[ScenePoint], Field(min_length=2)]


class ScenePointGeometry(StrictModel):
    kind: Literal["point"]
    address: ModelAddress
    name: str
    point: ModelAddress
    position: ScenePoint


class ScenePlanarGeometry(StrictModel):
    kind: Literal["planar"]
    address: ModelAddress
    name: str
    links: Annotated[list[SceneLinkGeometry], Field(min_length=1)]
    muscles: list[SceneMuscleGeometry]


type SceneMechanismGeometry = Annotated[
    ScenePointGeometry | ScenePlanarGeometry, Field(discriminator="kind")
]


class StudioSceneModelSource(StrictModel):
    revision: Sha256
    document_id: ProjectName


class StudioSceneGeometryRequest(StudioSceneModelSource):
    poses: Annotated[list[dict[str, JsonValue]], Field(min_length=1)]
    """State values by Model address for each frame. An empty map requests the rest pose.

    A native planar pose uses its mechanism's `skeleton/qpos` state in compiled joint order.
    """


class StudioSceneGeometryFrame(StrictModel):
    mechanisms: list[SceneMechanismGeometry]
    diagnostics: list[DomainDiagnostic]
    anchors: list[SceneAnchorGeometry] = Field(default_factory=list)


class StudioSceneGeometryResponse(StudioSceneModelSource):
    project_id: ProjectName
    frames: Annotated[list[StudioSceneGeometryFrame], Field(min_length=1)]
    """Geometry in the input pose order."""
