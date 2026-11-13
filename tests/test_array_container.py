"""Contracts shared by checkpoint slots and labelled datasets."""

from hashlib import sha256
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save

from feedbax.persistence.array_container import (
    ArrayContainerError,
    decode_array_container,
    encode_array_container,
    open_array_container,
)


def test_equal_values_encode_identically_across_processes(tmp_path: Path) -> None:
    expected = {
        "mechanics.effector.pos": np.arange(12, dtype=np.float32).reshape(3, 4),
        "clock": np.array(2, dtype=np.uint32),
        "empty": np.empty((2, 0), dtype=np.bool_),
    }
    metadata = {"z": ["position", 0.5], "α": {"weak_type": True}}
    first = encode_array_container(expected, metadata=metadata)
    output = tmp_path / "other-process.safetensors"
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from pathlib import Path
import numpy as np
from feedbax.persistence.array_container import encode_array_container

tensors = {
    "empty": np.empty((2, 0), dtype=np.bool_),
    "clock": np.array(2, dtype=">u4"),
    "mechanics.effector.pos": np.arange(12, dtype=">f4").reshape(3, 4).T.copy().T,
}
value = encode_array_container(tensors, metadata={"α": {"weak_type": True}, "z": ["position", 0.5]})
Path(sys.argv[1]).write_bytes(value.data)
""",
            str(output),
        ],
        check=True,
        timeout=10,
    )
    assert output.read_bytes() == first.data

    decoded = decode_array_container(first.data, expected_descriptor_sha256=first.descriptor_sha256)
    assert decoded.descriptor.metadata == metadata
    for name, values in expected.items():
        np.testing.assert_array_equal(decoded.tensors[name], values, strict=True)
    assert decoded.descriptor.tensors["clock"].shape == ()
    assert decoded.descriptor.tensors["clock"].dtype == "uint32"


def test_read_only_the_selected_tensor_and_distinguish_unverified_slices(tmp_path: Path) -> None:
    samples = np.arange(12, dtype=np.float32).reshape(3, 4)
    clock = np.array(1, dtype=np.int32)
    encoded = encode_array_container({"samples": samples, "clock": clock}, metadata=None)
    path = tmp_path / "states.safetensors"
    path.write_bytes(encoded.data)
    with safe_open(path, framework="numpy") as native:
        native_metadata = native.metadata()
        assert native_metadata is not None and len(native_metadata) == 1

    path.write_bytes(save({"samples": samples + 1, "clock": clock}, metadata=native_metadata))
    with open_array_container(path, expected_descriptor_sha256=encoded.descriptor_sha256) as reader:
        np.testing.assert_array_equal(reader.read_tensor("clock"), clock)
        with pytest.raises(ArrayContainerError, match="samples.*digest mismatch"):
            reader.read_tensor("samples")
        np.testing.assert_array_equal(
            reader.read_unverified_slice("samples", (-1, slice(1, 3))), samples[-1, 1:3] + 1
        )

    with pytest.raises(ArrayContainerError, match="samples.*digest mismatch"):
        decode_array_container(
            path.read_bytes(), expected_descriptor_sha256=encoded.descriptor_sha256
        )
    path.write_bytes(save({"samples": samples.T, "clock": clock}, metadata=native_metadata))
    with pytest.raises(ArrayContainerError, match="header disagree.*samples"):
        with open_array_container(path, expected_descriptor_sha256=encoded.descriptor_sha256):
            pytest.fail("expected shape mismatch before opening tensor payload")


def test_metadata_only_objects_and_malformed_descriptors(tmp_path: Path) -> None:
    metadata = {"scalar": None, "fields": [2, 0, False]}
    encoded = encode_array_container({}, metadata=metadata)
    decoded = decode_array_container(encoded.data, expected_descriptor_sha256=encoded.descriptor_sha256)
    assert decoded.descriptor.metadata == metadata
    assert not decoded.tensors
    with pytest.raises(ArrayContainerError, match="descriptor digest mismatch"):
        decode_array_container(encoded.data, expected_descriptor_sha256="0" * 64)

    path = tmp_path / "metadata.safetensors"
    path.write_bytes(encoded.data)
    with safe_open(path, framework="numpy") as native:
        native_metadata = native.metadata()
        assert native_metadata
    key, descriptor = next(iter(native_metadata.items()))
    noncanonical = descriptor + " "
    with pytest.raises(ArrayContainerError, match="not canonical JSON"):
        decode_array_container(
            save({}, metadata={key: noncanonical}),
            expected_descriptor_sha256=sha256(noncanonical.encode()).hexdigest(),
        )

    header = b'{"__metadata__":{},"__metadata__":{}}'
    with pytest.raises(ArrayContainerError, match="duplicate.*__metadata__"):
        decode_array_container(
            len(header).to_bytes(8, "little") + header,
            expected_descriptor_sha256=encoded.descriptor_sha256,
        )
    with pytest.raises(ArrayContainerError, match="invalid.*name"):
        encode_array_container({"__metadata__": np.array(1)}, metadata=None)
    with pytest.raises(ArrayContainerError, match="unsupported.*unsafe"):
        encode_array_container({"unsafe": np.array([1], dtype=object)}, metadata=None)


@pytest.mark.parametrize(
    ("raw_metadata", "reason"),
    [
        ('{"trial":1,"trial":2}', "duplicate.*trial"),
        ('{"trial":9007199254740992}', "unsafe_integer"),
        (r'{"trial":"\ud800"}', "lone_surrogate"),
        (r'{"trial":"\u00e9"}', "not canonical JSON"),
    ],
)
def test_descriptor_open_rejects_invalid_metadata_even_with_matching_digest(
    tmp_path: Path, raw_metadata: str, reason: str,
) -> None:
    descriptor = (
        '{"metadata":' + raw_metadata + ',"schema":"feedbax.array_container.v1","tensors":{}}'
    )
    path = tmp_path / "metadata.safetensors"
    path.write_bytes(save({}, metadata={"feedbax": descriptor}))
    with pytest.raises(ArrayContainerError, match=reason):
        with open_array_container(
            path, expected_descriptor_sha256=sha256(descriptor.encode()).hexdigest(),
        ):
            pytest.fail("invalid metadata was exposed")
