"""Read-only planar geometry projected from a named authored model."""

from typing import Annotated

from pydantic import Field, FiniteFloat

from feedbax.contracts.base import StrictModel
from feedbax.contracts.domain import DomainDiagnostic
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
