"""Read semantic model sources from exact retained model-product receipts."""

from feedbax.contracts.artifact_custody import ContentObjectRef, ContentTreeRef
from feedbax.contracts.effective_success import (
    BoundMember, Product, Success, read_effective_request, read_success_object, select_success_product, success_content_ref,
)
from feedbax.contracts.model_checkpoint import MODEL_TREE_MEDIA_TYPE, SemanticModelEncoding
from feedbax.contracts.model_input import CheckpointModelSource, InitialModelSource, ParameterModelSource, SynthesizedModelSource
from feedbax.contracts.request_refs import LastMember, ProductRef, is_synthesized_model_product
from feedbax.contracts.request_specs import DocumentTrainingRequestSpec, SynthesisRequestSpec
from feedbax.contracts.training import TrialDocuments
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.checkpoint_publication import model_product, read_retained_checkpoint


def model_product_source(
    bound: BoundMember, *, artifacts: ImmutableArtifactBlobProvider,
) -> tuple[ParameterModelSource, TrialDocuments]:
    """Read the semantic model and resolved trial documents of its exact producer."""
    if (source := bound.checkpoint_model) is not None:
        checkpoint, spec = read_retained_checkpoint(artifacts, source.publication, source.producer)
        model = model_product(checkpoint)
        if (bound.content, bound.descriptor) != (model.content, model.descriptor):
            raise ValueError("Bound model differs from its checkpoint publication")
        return _model_source(checkpoint.model.tree, checkpoint.model.encoding, spec), spec.trial_documents
    if bound.selection is None or bound.content.media_type != MODEL_TREE_MEDIA_TYPE:
        raise ValueError("Model input requires a bound semantic model product")
    receipt = read_success_object(bound.selection.receipt, artifacts.get_object)
    request = read_effective_request(receipt, artifacts.get_object)
    spec = request.spec
    if isinstance(spec, DocumentTrainingRequestSpec):
        reference = ProductRef(request_id=request.id, role="checkpoint", select=LastMember())
    elif isinstance(spec, SynthesisRequestSpec):
        reference = ProductRef(request_id=request.id, role="model")
    else:
        raise ValueError("Semantic model receipt requires a model-producing definition")
    model = select_success_product(reference, receipt)
    if (model.name, model.content, model.descriptor) != (bound.selection.member, bound.content, bound.descriptor):
        raise ValueError("Bound model differs from its exact producer product")
    tree, encoding = _model_layout(model)
    if isinstance(spec, SynthesisRequestSpec):
        source = SynthesizedModelSource(tree=tree, encoding=encoding, initial=InitialModelSource(document=spec.trial_documents.model))
    else:
        source = _model_source(tree, encoding, spec)
    return source, spec.trial_documents


def _model_layout(model: Product) -> tuple[ContentTreeRef, SemanticModelEncoding]:
    tree = success_content_ref(model.content)
    if not isinstance(tree, ContentTreeRef) or model.content.media_type != MODEL_TREE_MEDIA_TYPE:
        raise ValueError("Semantic model requires a content tree")
    return tree, SemanticModelEncoding.model_validate(model.descriptor)


def read_synthesized_model(
    store: ImmutableArtifactBlobProvider, receipt: ContentObjectRef, reference: ProductRef,
) -> tuple[SynthesizedModelSource, SynthesisRequestSpec, Success]:
    if not is_synthesized_model_product(reference):
        raise ValueError("A Synthesize run supplies its one model product.")
    success = read_success_object(receipt, store.get_object)
    request = read_effective_request(success, store.get_object)
    if request.id != reference.request_id or not isinstance(spec := request.spec, SynthesisRequestSpec):
        raise ValueError("The frozen model product differs from its Synthesize request.")
    tree, encoding = _model_layout(select_success_product(reference, success))
    source = SynthesizedModelSource(tree=tree, encoding=encoding, initial=InitialModelSource(document=spec.trial_documents.model))
    return source, spec, success


def _model_source(
    tree: ContentTreeRef, encoding: SemanticModelEncoding, spec: DocumentTrainingRequestSpec,
) -> tuple[CheckpointModelSource, TrialDocuments]:
    documents = spec.trial_documents
    source = CheckpointModelSource(
        tree=tree, encoding=encoding,
        initial=InitialModelSource(
            document=documents.model, seed=spec.seed or 0,
        ),
    )
    return source, documents
