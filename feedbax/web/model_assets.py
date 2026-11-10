"""Studio model assets stored in its durable content store."""

from __future__ import annotations

from feedbax.contracts.acausal import AcausalGraphSpec
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
    pending = list((graph.subgraphs or {}).values())
    while pending:
        interior = pending.pop()
        if isinstance(interior, AcausalGraphSpec) and interior.mujoco_asset is not None:
            return _provider()
        pending.extend((interior.subgraphs or {}).values())
    return None
