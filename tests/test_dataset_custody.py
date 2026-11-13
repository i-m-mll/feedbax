"""Verify portable dataset bindings at the immutable content store boundary."""

from hashlib import sha256

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save

from feedbax.contracts.artifact_custody import ArtifactBlobIntegrityError
from feedbax.contracts.dataset import DatasetProductRef
from feedbax.contracts.request_refs import ContentRef
from feedbax.dataset import Dataset, Variable
from feedbax.persistence.array_container import ArrayContainerError, encode_array_container
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import (
    encode_dataset,
    import_dataset,
    open_stored_dataset,
    publish_dataset,
)
from feedbax.persistence.store_check import check_store


@pytest.fixture
def retained_dataset():
    return Dataset.from_arrays(
        {
            "mechanics.effector.pos": Variable(
                np.array([[0.0, 0.2, 0.7], [0.0, 0.4, 0.8]]),
                ("trial", "state_sample"),
                units="m",
            ),
            "controller.command": Variable(
                np.array([[3.0, 2.0], [6.0, 1.0]]), ("trial", "transition_sample")
            ),
            "timing_applicability": Variable(np.array(["bounded", "inapplicable"]), ("trial",)),
            "gain": Variable(np.array(3.0), ()),
        },
        coords={
            "time_s": Variable(np.array([0.0, 0.1, 0.2]), ("state_sample",), units="s"),
            "trial": Variable(np.array(["calibration", "verification"]), ("trial",)),
        },
    )


def _binding(data, descriptor_sha256):
    return DatasetProductRef(
        content=ContentRef(sha256=sha256(data).hexdigest(), size=len(data)),
        descriptor_sha256=descriptor_sha256,
    )


def _change_commands(path):
    with safe_open(path, framework="numpy") as reader:
        metadata = reader.metadata()
        arrays = {name: reader.get_tensor(name) for name in reader.keys()}
    arrays["controller.command"][1, 0] = -9.0
    return save(arrays, metadata=metadata)


def test_import_checks_expected_blob_identity_before_publication(tmp_path, retained_dataset):
    encoded = encode_dataset(retained_dataset)
    store = ImmutableArtifactBlobProvider(tmp_path / "cas")
    actual = _binding(encoded.data, encoded.descriptor_sha256)
    wrong = DatasetProductRef(
        content=ContentRef(sha256="0" * 64, size=len(encoded.data)),
        descriptor_sha256=encoded.descriptor_sha256,
    )

    with pytest.raises(ArtifactBlobIntegrityError):
        import_dataset(store, encoded.data, wrong)
    assert not store.object_path(actual.object_ref).exists()
    assert not store.object_path(wrong.object_ref).exists()

    assert import_dataset(store, encoded.data, actual) == actual
    assert store.get_object(actual.object_ref) == encoded.data


@pytest.mark.parametrize("malformation", ["tensor", "dimensions", "text_envelope"])
def test_exact_but_malformed_imports_never_enter_custody(tmp_path, retained_dataset, malformation):
    store = ImmutableArtifactBlobProvider(tmp_path / "cas")
    if malformation == "tensor":
        encoded = encode_dataset(retained_dataset)
        source = tmp_path / "untrusted.safetensors"
        source.write_bytes(encoded.data)
        payload = _change_commands(source)
        failure = "digest"
    elif malformation == "dimensions":
        descriptor = retained_dataset.descriptor
        metadata = descriptor.json_value()
        metadata["dimensions"]["state_sample"] = 4
        encoded = encode_array_container(
            {name: retained_dataset.data(name) for name in descriptor.tensor_names},
            metadata=metadata,
        )
        payload = encoded.data
        failure = "shape"
    else:
        text_only = Dataset.from_arrays({"phase": Variable(np.array("ready"), ())})
        encoded = encode_dataset(text_only)
        payload = encoded.data + b"trailing bytes"
        failure = "safetensors"

    expected = _binding(payload, encoded.descriptor_sha256)
    with pytest.raises(ValueError, match=failure):
        import_dataset(store, payload, expected)
    assert not store.object_path(expected.object_ref).exists()


def test_stored_reads_use_custody_for_slices_and_scrub_is_explicit(tmp_path, retained_dataset):
    store = ImmutableArtifactBlobProvider(tmp_path / "cas")
    binding = publish_dataset(store, retained_dataset)
    ref = DatasetProductRef.model_validate_json(binding.model_dump_json())
    with open_stored_dataset(store, ref) as opened:
        np.testing.assert_allclose(
            opened.data("mechanics.effector.pos", (1, slice(1, None))), [0.4, 0.8]
        )
        for index in (
            (-1, -1),
            (1, slice(None, None, -1)),
            (slice(None, None, -1), slice(None, None, -2)),
            (slice(None), slice(None, None, 2)),
            (slice(9, 2),),
        ):
            name = "mechanics.effector.pos"
            np.testing.assert_array_equal(opened.data(name, index), retained_dataset.data(name, index))
        np.testing.assert_allclose(opened.data("time_s"), [0.0, 0.1, 0.2])
        assert opened.data("trial").tolist() == ["calibration", "verification"]
        assert opened.data("timing_applicability", (1,)).item() == "inapplicable"
        assert opened.data("gain").item() == 3.0
    store.verify_object(ref.object_ref)

    wrong_descriptor = DatasetProductRef(content=ref.content, descriptor_sha256="0" * 64)
    with pytest.raises(ArrayContainerError, match="descriptor digest"):
        with open_stored_dataset(store, wrong_descriptor):
            pytest.fail("accepted a descriptor without the owning binding")

    path = store.object_path(ref.object_ref)
    corrupted = _change_commands(path)
    assert len(corrupted) == ref.content.size
    mode = path.stat().st_mode & 0o777
    path.chmod(mode | 0o200)
    try:
        path.write_bytes(corrupted)
    finally:
        path.chmod(mode)

    with open_stored_dataset(store, ref) as opened:
        np.testing.assert_allclose(
            opened.data("mechanics.effector.pos", (1, slice(1, None))), [0.4, 0.8]
        )
    with pytest.raises(ArtifactBlobIntegrityError):
        store.verify_object(ref.object_ref)


def test_check_uses_the_anchored_tensor_digest_even_when_object_bytes_match(tmp_path, retained_dataset):
    store = ImmutableArtifactBlobProvider(tmp_path / "cas")
    encoded = encode_dataset(retained_dataset)
    source = tmp_path / "source.safetensors"
    source.write_bytes(encoded.data)
    malformed = _change_commands(source)
    ref = _binding(malformed, encoded.descriptor_sha256)
    assert store.store_object(malformed) == ref.object_ref
    store.verify_object(ref.object_ref)
    assert store.compare_and_swap_ref("fixtures/dataset", expected=None, replacement=ref.model_dump(mode="json"))

    result = check_store(store)

    (finding,) = result.findings
    assert finding.code == "damaged_object" and finding.address == ref.object_ref.object_id
    assert "digest mismatch" in finding.detail
    assert store.get_object(ref.object_ref) == malformed
