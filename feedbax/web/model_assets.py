"""Studio model assets stored in its durable content store."""

from __future__ import annotations

from feedbax.compiler.model_assets import model_asset_refs
from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.graph import GraphSpec
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.web import config


def _provider() -> ImmutableArtifactBlobProvider:
    return ImmutableArtifactBlobProvider((config.BASE_DIR / "content").absolute())


def import_mujoco_asset(data: bytes) -> ContentObjectRef:
    from feedbax.mechanics.authored_model import validate_spatial_mjcf

    validate_spatial_mjcf(data)
    return _provider().store_rooted_object(data, namespace="studio:mujoco")


def model_artifact_provider(graph: GraphSpec) -> ImmutableArtifactBlobProvider | None:
    """Resolve the same Studio content store for project and training compilers."""
    return _provider() if next(model_asset_refs(graph), None) is not None else None
