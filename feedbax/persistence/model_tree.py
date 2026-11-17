"""Validate the member layout of a semantic model tree against its graph."""

from collections.abc import Mapping

from feedbax.compiler.model_assets import model_asset_refs
from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.model_checkpoint import MODEL_TREE_ARRAY_PATH, MODEL_TREE_GRAPH_PATH


class ModelTreeLayoutError(ValueError):
    def __init__(self, path: str, detail: str):
        self.path = path
        self.detail = detail
        super().__init__(f"{path}: {detail}")


def model_tree_asset_path(ref: ContentObjectRef) -> str:
    return f"assets/{ref.sha256}"


def model_tree_assets(
    graph: GraphSpec, members: Mapping[str, ContentObjectRef]
) -> tuple[tuple[str, ContentObjectRef], ...]:
    assets = tuple(model_asset_refs(graph))
    expected = {
        MODEL_TREE_GRAPH_PATH,
        MODEL_TREE_ARRAY_PATH,
        *(model_tree_asset_path(asset) for _, asset in assets),
    }
    if members.keys() != expected:
        raise ModelTreeLayoutError(
            "/v1/model", "model tree members disagree with the effective graph"
        )
    for path, asset in assets:
        if members[model_tree_asset_path(asset)] != asset:
            raise ModelTreeLayoutError(path, "model tree pins a different asset")
    return assets
