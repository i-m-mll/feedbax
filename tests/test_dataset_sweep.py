"""Verify dataset naming and laziness on authored sweep members."""

import numpy as np
import pytest

from feedbax.contracts.project_sweep import Sweep, SweepCoordinate, SweepInputMember, expand_sweep
from feedbax.dataset import Dataset, Variable
from feedbax.dataset_sweep import stack_dataset_members


def _bound_members():
    sweep = Sweep.model_validate(
        {
            "axes": {
                "seed": {"path": "seed", "values": [1, 2**64 - 1]},
                "arm": {"values": {"second": {"label": "B"}, "first": {"label": "A"}}},
            }
        }
    )
    reads = []
    inputs = []
    for offset, row in enumerate(expand_sweep(sweep)):
        resident = Dataset.from_arrays(
            {
                "plant.position": Variable(
                    np.arange(6).reshape(2, 3) + 10 * offset, ("trial", "state_sample")
                )
            },
            coords={"trial": Variable(np.array(["left", "right"]), ("trial",))},
        )

        def load(name, index, source=resident, key=row.id):
            reads.append((key, name, index))
            return source.data(name, index)

        inputs.append(
            (
                SweepInputMember(key=row.id, coordinates=row.coordinates, label=row.label),
                Dataset.bind(
                    resident.descriptor, tensor_shapes={"plant.position": (2, 3)}, load_tensor=load
                ),
            )
        )
    return inputs, reads


def test_complete_sweep_keeps_axis_ids_values_and_selects_only_the_needed_member():
    inputs, reads = _bound_members()
    dataset = stack_dataset_members(inputs)

    assert reads == []
    assert dataset.descriptor.variables["plant.position"].dims == (
        "seed",
        "arm",
        "trial",
        "state_sample",
    )
    assert dataset.data("seed.value").dtype == np.uint64
    assert dataset.data("seed.value").tolist() == [1, 2**64 - 1]
    assert dataset.data("seed").tolist() == [
        inputs[0][0].coordinates["seed"].value_id,
        inputs[2][0].coordinates["seed"].value_id,
    ]
    assert dataset.data("arm").tolist() == ["second", "first"]
    assert dataset.data("arm.label").tolist() == ["B", "A"]
    assert dataset.data("member", (1, 0)).item() == inputs[2][0].key
    assert dataset.data("trial", (1, 0)).tolist() == ["left", "right"]

    np.testing.assert_array_equal(dataset.data("plant.position", (1, 0, 1)), [23, 24, 25])
    assert reads == [(inputs[2][0].key, "plant.position", (1,))]
    reads.clear()
    np.testing.assert_array_equal(
        dataset.select("plant.position", labels={"seed.value": 2**64 - 1, "arm": "second"}),
        [[20, 21, 22], [23, 24, 25]],
    )
    assert reads == [(inputs[2][0].key, "plant.position", ())]


def test_sparse_members_preserve_authored_order_and_structured_coordinate_values():
    inputs, _ = _bound_members()
    source = inputs[0][1]
    data = stack_dataset_members(
        [
            (
                SweepInputMember("row-b", {"condition": SweepCoordinate("b", "B", {"mass": 2})}),
                source,
            ),
            (
                SweepInputMember("row-a", {"condition": SweepCoordinate("a", None, {"mass": 1})}),
                source,
            ),
            (SweepInputMember("control"), source),
        ]
    )
    assert data.descriptor.variables["plant.position"].dims == ("member", "trial", "state_sample")
    assert data.data("member").tolist() == ["row-b", "row-a", "control"]
    assert data.data("condition").tolist() == ["b", "a", None]
    assert data.data("condition.label").tolist() == ["B", None, None]
    assert data.data("condition.value").tolist() == [{"mass": 2}, {"mass": 1}, None]


def test_sweep_name_collision_and_conflicting_id_are_refused():
    inputs, _ = _bound_members()
    source = inputs[0][1]
    with pytest.raises(ValueError, match="collide.*trial"):
        stack_dataset_members(
            [(SweepInputMember("row", {"trial": SweepCoordinate("authored")}), source)]
        )
    with pytest.raises(ValueError, match="conflicting values"):
        stack_dataset_members(
            [
                (SweepInputMember("one", {"s": SweepCoordinate("same", value=1)}), source),
                (SweepInputMember("two", {"s": SweepCoordinate("same", value=True)}), source),
            ]
        )
