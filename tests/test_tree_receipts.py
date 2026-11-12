import errno

import numpy as np
import pytest

from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.effective_success import INDEX_REF, Product, Selection, success_content_ref
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.model_checkpoint import (
    MODEL_TREE_ARRAY_PATH,
    MODEL_TREE_GRAPH_PATH,
    MODEL_TREE_MEDIA_TYPE,
    SemanticModelEncoding,
)
from feedbax.contracts.request_identity import encode_request
from feedbax.contracts.request_refs import ContentRef, InputMember, NamedMember, ProductRef
from feedbax.contracts.request_specs import ReportRequestSpec, TrainingRequestSpec
from feedbax.contracts.store_closure import ReceiptRoot, StoreClosureRef
from feedbax.persistence.array_container import encode_array_container
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView, store_request
from feedbax.persistence.store_sync import sync_store


def test_selected_model_tree_publishes_binds_and_imports(tmp_path):
    source = ImmutableArtifactBlobProvider(tmp_path / "source")
    packet = ImmutableArtifactBlobProvider(tmp_path / "packet")
    destination = ImmutableArtifactBlobProvider(tmp_path / "destination")
    graph = GraphSpec()
    spec = TrainingRequestSpec.model_validate(
        {
            "graph": graph,
            "task": {"type": "proof"},
            "objective": {"kind": "external", "payload": {}},
            "method_ref": {"package": "feedbax", "name": "proof", "version": "1"},
            "method_payload": {"schema_id": "proof.method", "schema_version": "1"},
            "training_config": {},
        }
    )
    request = store_request(source, encode_request(spec, {}))
    encoded = encode_array_container(
        {"weight": np.array([1.25, -2], dtype=np.float32)}, metadata={}
    )
    members = {
        MODEL_TREE_GRAPH_PATH: source.store_object(
            canonical_json_v2_bytes(graph.model_dump(mode="json"))
        ),
        MODEL_TREE_ARRAY_PATH: source.store_object(encoded.data),
    }
    tree, _ = source.publish_rooted_tree(members, root_id="fixture:model", root_kind="logical-ref")
    encoding = SemanticModelEncoding(array_descriptor_sha256=encoded.descriptor_sha256)
    model = Product(
        name="model",
        content=ContentRef(
            tree=True,
            sha256=tree.descriptor.sha256,
            size=tree.descriptor.size_bytes,
            media_type=MODEL_TREE_MEDIA_TYPE,
        ),
        descriptor=encoding.model_dump(mode="json"),
    )
    optimizer_ref = source.store_object(b"optimizer payload")
    optimizer = Product(
        name="optimizer",
        content=ContentRef(sha256=optimizer_ref.sha256, size=optimizer_ref.size_bytes),
    )
    extra_tree, _ = source.publish_rooted_tree(
        {**members, "optimizer": optimizer_ref},
        root_id="fixture:extra",
        root_kind="logical-ref",
    )
    extra_content = model.content.model_copy(
        update={"sha256": extra_tree.descriptor.sha256, "size": extra_tree.descriptor.size_bytes}
    )
    with pytest.raises(ValueError, match="model tree members disagree"):
        extra_model = model.model_copy(update={"content": extra_content})
        SuccessView(source).publish(
            request, attempt="extra-member", products={"checkpoint": (extra_model,)}
        )
    with pytest.raises(ValueError, match="array descriptor digest mismatch"):
        mismatched = model.model_copy(update={"descriptor": {"array_descriptor_sha256": "0" * 64}})
        SuccessView(source).publish(
            request, attempt="wrong-descriptor", products={"checkpoint": (mismatched,)}
        )
    assert source.read_ref(INDEX_REF) is None

    published = SuccessView(source).publish(
        request, attempt="train-proof", products={"checkpoint": (model, optimizer)}
    )
    selection = ProductRef(
        request_id=request.id, role="checkpoint", select=NamedMember(member="model")
    )
    consumer = store_request(
        source,
        encode_request(
            ReportRequestSpec(report_type="proof.model-consumer"),
            {"model": (InputMember(key="subject", reference=selection),)},
        ),
    )
    _, inputs = SuccessView(source).bind(consumer)
    [bound] = inputs["model"]
    assert bound.selection == Selection(receipt=published.ref, member="model")
    assert bound.content == model.content and bound.descriptor == model.descriptor
    assert success_content_ref(bound.content) == tree

    closure = StoreClosureRef(roots=(ReceiptRoot(receipt=published.ref, product=selection),))
    try:
        sync_store(source, packet, closure, destination_ref="packets/model")
    except OSError as error:
        if error.errno != errno.ENOTSUP:
            raise
        pytest.skip("test volume does not support copy-on-write cloning")
    imported = sync_store(packet, destination, "packets/model", destination_ref="packets/model")
    assert imported.closure == closure and imported.reused == 0
    assert destination.get_object(published.ref) == source.get_object(published.ref)
    assert destination.get_tree(tree) == source.get_tree(tree)
    for ref in members.values():
        assert destination.get_object(ref) == source.get_object(ref)
    for store in (packet, destination):
        assert store.read_ref(INDEX_REF) is None
        with pytest.raises(FileNotFoundError):
            store.verify_object(optimizer_ref)
