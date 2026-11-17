import errno
import json

import numpy as np
import pytest

from feedbax.cli import main
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.effective_success import INDEX_REF, Product, Selection, success_content_ref
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.model_checkpoint import (
    MODEL_TREE_ARRAY_PATH,
    MODEL_TREE_GRAPH_PATH,
    MODEL_TREE_MEDIA_TYPE,
    SemanticModelEncoding,
)
from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.contracts.request_identity import encode_request
from feedbax.contracts.request_refs import ContentRef, InputMember, NamedMember, ProductRef
from feedbax.contracts.request_specs import ReportRequestSpec
from feedbax.contracts.store_closure import ReceiptRoot, StoreClosureRef
from feedbax.persistence.array_container import encode_array_container
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView, run_ref_name, store_request
from tests.test_request_identity import _managed_train


def test_selected_model_tree_publishes_binds_and_imports(tmp_path, monkeypatch, capsys):
    source_dir, destination_dir = tmp_path / "source", tmp_path / "destination"
    for root in (source_dir, destination_dir):
        for name, raw in dump_text_project(
            Project(), ProjectConfiguration(project=root.name, store="custom-store")
        ).items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    source = ImmutableArtifactBlobProvider(source_dir / "custom-store")
    packet_dir = tmp_path / "packet"
    destination = ImmutableArtifactBlobProvider(destination_dir / "custom-store")
    graph = GraphSpec()
    spec = _managed_train()
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
    work = source_dir / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    assert main(["store", "export", "unknown", "--dest", "../../packet"]) == 2
    assert "No committed receipt" in capsys.readouterr().err
    assert not packet_dir.exists()

    code = main(
        [
            "store",
            "export",
            "train-proof",
            "--dest",
            "../../packet",
            "--product",
            "checkpoint",
            "--member",
            "model",
        ]
    )
    export = capsys.readouterr()
    if code == 2 and f"[Errno {errno.ENOTSUP}]" in export.err:
        pytest.skip("test volume does not support copy-on-write cloning")
    assert code == 0, export.err
    exported = json.loads(export.out)
    assert exported["closure"] == closure.model_dump(mode="json")
    packet = ImmutableArtifactBlobProvider(packet_dir)
    assert main(["store", "export", "train-proof", "--dest", "../../packet"]) == 2
    assert "Destination closure ref changed" in capsys.readouterr().err
    assert packet.read_ref(exported["ref"]) == exported["closure"]

    argv = [
        "store",
        "import",
        str(packet_dir),
        "--project-dir",
        str(destination_dir),
        "--name",
        "model",
    ]
    assert main(argv) == 0
    imported = json.loads(capsys.readouterr().out)
    assert imported["store"] == str(destination.root)
    assert imported["ref"] == "packets/model"
    assert imported["closure"] == exported["closure"] and imported["reused"] == 0
    ref_path = destination.root / "refs" / imported["ref"]
    before = ref_path.stat()
    assert main(argv) == 0
    repeat = json.loads(capsys.readouterr().out)
    assert repeat == imported | {"cloned": 0, "reused": imported["cloned"]}
    after = ref_path.stat()
    assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)
    assert destination.read_ref(imported["ref"]) == imported["closure"]
    assert destination.get_object(published.ref) == source.get_object(published.ref)
    assert destination.get_tree(tree) == source.get_tree(tree)
    for ref in members.values():
        assert destination.get_object(ref) == source.get_object(ref)
    for store in (packet, destination):
        assert store.read_ref(INDEX_REF) is None
        with pytest.raises(FileNotFoundError):
            store.verify_object(optimizer_ref)

    run_ref = run_ref_name(request.id)
    assert destination.read_ref(run_ref) == imported["closure"]
    assert packet.read_ref(run_ref) is None

    next_result = SuccessView(source).publish(
        request, attempt="new-train", products={"checkpoint": (model, optimizer)}
    )
    successor_packet = tmp_path / "next-packet"
    assert main([
        "store", "export", "new-train", "--dest", str(successor_packet),
        "--product", "checkpoint", "--member", "model",
    ]) == 0
    capsys.readouterr()
    assert main([
        "store", "import", str(successor_packet), "--project-dir", str(destination_dir),
        "--name", "different",
    ]) == 2
    refusal = capsys.readouterr().err
    assert published.ref.sha256 in refusal and next_result.ref.sha256 in refusal
    assert destination.read_ref(run_ref) == imported["closure"]
    assert destination.read_ref("packets/different") is None
