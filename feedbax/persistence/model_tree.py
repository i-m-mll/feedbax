"""Validate the member layout of a semantic model tree against its graph."""

from dataclasses import dataclass

from feedbax.compiler.model_assets import model_asset_refs
from feedbax.contracts.artifact_custody import ArtifactBlobProvider, ContentObjectRef, ContentTreeRef
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.model_checkpoint import (
    MODEL_TREE_GRAPH_PATH,
    MODEL_TREE_MANIFEST_PATH,
    ModelReplica,
    ModelReplicaManifest,
    SemanticModelEncoding,
    model_replica_array_path,
)
from feedbax.contracts.strict_json import strict_model_validate_json


class ModelTreeLayoutError(ValueError):
    def __init__(self, path: str, detail: str):
        self.path = path
        self.detail = detail
        super().__init__(f"{path}: {detail}")


def model_tree_asset_path(ref: ContentObjectRef) -> str:
    return f"assets/{ref.sha256}"


@dataclass(frozen=True)
class ModelTreeReplica:
    descriptor: ModelReplica
    arrays: ContentObjectRef


@dataclass(frozen=True)
class ModelTree:
    graph: GraphSpec
    replicas: tuple[ModelTreeReplica, ...]
    assets: tuple[tuple[str, ContentObjectRef], ...]
    members: dict[str, ContentObjectRef]


def read_model_tree(
    store: ArtifactBlobProvider, tree: ContentTreeRef, encoding: SemanticModelEncoding,
) -> ModelTree:
    members = {file.relative_path: file.object for file in store.get_tree(tree).files}
    graph_ref = members.get(MODEL_TREE_GRAPH_PATH)
    manifest_ref = members.get(MODEL_TREE_MANIFEST_PATH)
    if graph_ref is None:
        raise ModelTreeLayoutError("/graph", "semantic model tree is missing its graph")
    if manifest_ref is None or manifest_ref.sha256 != encoding.descriptor_sha256:
        raise ModelTreeLayoutError("/model", "semantic model manifest differs from its encoding")
    graph = strict_model_validate_json(GraphSpec, store.get_object(graph_ref))
    raw = store.get_object(manifest_ref)
    manifest = strict_model_validate_json(ModelReplicaManifest, raw)
    if raw != canonical_json_v2_bytes(manifest.model_dump(mode="json")):
        raise ModelTreeLayoutError("/model", "semantic model manifest is not canonical")

    assets = tuple(model_asset_refs(graph))
    arrays = tuple(model_replica_array_path(index) for index in range(len(manifest.replicas)))
    expected = {
        MODEL_TREE_GRAPH_PATH,
        MODEL_TREE_MANIFEST_PATH,
        *arrays,
        *(model_tree_asset_path(asset) for _, asset in assets),
    }
    if members.keys() != expected:
        raise ModelTreeLayoutError(
            "/model", "model tree members disagree with the effective graph and replica manifest"
        )
    for path, asset in assets:
        if members[model_tree_asset_path(asset)] != asset:
            raise ModelTreeLayoutError(path, "model tree pins a different asset")
    replicas = tuple(
        ModelTreeReplica(replica, members[path])
        for replica, path in zip(manifest.replicas, arrays, strict=True)
    )
    return ModelTree(graph, replicas, assets, members)
