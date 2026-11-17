"""Wire contract for a semantic model checkpoint tree."""

from pydantic import Field, model_validator

from feedbax.contracts.request_refs import Name, ReferenceValue, Sha256


MODEL_TREE_MEDIA_TYPE = "application/vnd.feedbax.semantic-model-tree+json"
MODEL_TREE_GRAPH_PATH = "graph.json"
MODEL_TREE_MANIFEST_PATH = "model.json"


class SemanticModelEncoding(ReferenceValue):
    """Identity of the semantic model manifest inside the immutable tree."""

    descriptor_sha256: Sha256


class ModelReplica(ReferenceValue):
    id: Name
    array_descriptor_sha256: Sha256


class ModelReplicaManifest(ReferenceValue):
    replicas: tuple[ModelReplica, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_ids(self) -> "ModelReplicaManifest":
        if len({replica.id for replica in self.replicas}) != len(self.replicas):
            raise ValueError("Semantic model replica IDs must be unique")
        return self


def model_replica_array_path(index: int) -> str:
    return f"replicas/{index}.safetensors"
