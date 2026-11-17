"""Read semantic model sources from exact retained Train receipts."""

from feedbax.contracts.artifact_custody import ContentTreeRef
from feedbax.contracts.effective_success import BoundMember, read_effective_request, read_success_object, select_success_product, success_content_ref
from feedbax.contracts.model_checkpoint import MODEL_TREE_MEDIA_TYPE, SemanticModelEncoding
from feedbax.contracts.model_input import CheckpointModelSource, InitialModelSource
from feedbax.contracts.request_refs import LastMember, ProductRef
from feedbax.contracts.request_specs import DocumentTrainingRequestSpec
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider


def checkpoint_model_source(
    bound: BoundMember, *, artifacts: ImmutableArtifactBlobProvider,
) -> CheckpointModelSource:
    """Read the authored Model and exact semantic product selected by a frozen binding."""
    if bound.selection is None or bound.content.media_type != MODEL_TREE_MEDIA_TYPE:
        raise ValueError("Model input requires a bound semantic model product")
    receipt = read_success_object(bound.selection.receipt, artifacts.get_object)
    request = read_effective_request(receipt, artifacts.get_object)
    if not isinstance(spec := request.spec, DocumentTrainingRequestSpec):
        raise ValueError("Semantic model receipt requires an authored Train definition")
    model = select_success_product(
        ProductRef(request_id=request.id, role="checkpoint", select=LastMember()), receipt,
    )
    if model.name != bound.selection.member or model.content != bound.content:
        raise ValueError("Bound model differs from its exact producer product")
    tree = success_content_ref(model.content)
    if not isinstance(tree, ContentTreeRef):
        raise ValueError("Semantic model requires a content tree")
    documents = spec.trial_documents
    return CheckpointModelSource(
        tree=tree, encoding=SemanticModelEncoding.model_validate(model.descriptor),
        initial=InitialModelSource(
            model_name=documents.model_name, document=documents.model, seed=spec.seed or 0,
        ),
    )
