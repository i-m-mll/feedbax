"""Content-addressed assets referenced by an authored model."""

from collections.abc import Iterator

from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.graph import GraphSpec


def model_asset_refs(
    graph: GraphSpec | AcausalGraphSpec,
    *,
    path: str = "/graph",
) -> Iterator[tuple[str, ContentObjectRef]]:
    if isinstance(graph, AcausalGraphSpec) and graph.mujoco_asset is not None:
        yield f"{path}/mujoco_asset/asset", graph.mujoco_asset.asset
    for name, interior in (graph.subgraphs or {}).items():
        token = name.replace("~", "~0").replace("/", "~1")
        yield from model_asset_refs(interior, path=f"{path}/subgraphs/{token}")
