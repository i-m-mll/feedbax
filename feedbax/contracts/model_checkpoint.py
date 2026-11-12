"""Wire contract for a semantic model checkpoint tree."""

from feedbax.contracts.base import StrictModel
from feedbax.contracts.request_refs import Sha256


MODEL_TREE_MEDIA_TYPE = "application/vnd.feedbax.semantic-model-tree+json"
MODEL_TREE_GRAPH_PATH = "graph.json"
MODEL_TREE_ARRAY_PATH = "model.safetensors"


class SemanticModelEncoding(StrictModel):
    """Independent digest of the array container descriptor in the model tree."""

    array_descriptor_sha256: Sha256
