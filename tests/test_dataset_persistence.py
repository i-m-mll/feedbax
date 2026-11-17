"""Exercise the shared storage boundary with selected labelled fields."""

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

from feedbax.dataset import Dataset, Variable
from feedbax.persistence.array_container import ArrayContainerError
from feedbax.persistence.dataset import encode_dataset, open_dataset


def test_dataset_checks_only_the_selected_payload_against_a_bound_descriptor(tmp_path):
    path = tmp_path / "evaluation.safetensors"
    position = np.array([[0, 1, 3], [1, 4, 5]], dtype=np.float32)
    commands = np.array([[1, 2], [3, 1]], dtype=np.float32)
    dataset = Dataset.from_arrays(
        {
            "plant.position": Variable(position, ("trial", "state_sample"), units="m"),
            "controller.command": Variable(commands, ("trial", "transition_sample")),
            "timing_applicability": Variable(np.array(["bounded", "none"]), ("trial",)),
        },
        coords={
            "time_s": Variable(
                np.array([[0.0, 0.2, 0.4], [0.0, 0.5, 1.0]]),
                ("trial", "state_sample"),
                units="s",
            )
        },
    )
    encoded = encode_dataset(dataset)
    path.write_bytes(encoded.data)

    with safe_open(path, framework="numpy") as independent_reader:
        np.testing.assert_array_equal(independent_reader.get_tensor("plant.position"), position)
        metadata = independent_reader.metadata()
        tensors = {name: independent_reader.get_tensor(name) for name in independent_reader.keys()}
    tensors["controller.command"][0, 1] = -5
    save_file(tensors, str(path), metadata=metadata)

    with open_dataset(path, expected_descriptor_sha256=encoded.descriptor_sha256) as stored:
        np.testing.assert_array_equal(stored.data("plant.position", (1,)), [1, 4, 5])
        np.testing.assert_array_equal(stored.data("time_s", (0, slice(1, None))), [0.2, 0.4])
        assert "timing_applicability" in stored.descriptor.variables
        assert stored.data("timing_applicability").tolist() == ["bounded", "none"]
        with pytest.raises(ArrayContainerError, match="digest"):
            stored.data("controller.command")

    with pytest.raises(ArrayContainerError, match="descriptor digest"):
        with open_dataset(path, expected_descriptor_sha256="0" * 64):
            pytest.fail("metadata was trusted without its product binding")
