"""Verify dataset naming and laziness on authored sweep members."""

import numpy as np
import pytest

from feedbax.contracts.project_sweep import Sweep, SweepCoordinate, SweepInputMember, expand_sweep
from feedbax.dataset import Dataset, Variable
from feedbax.dataset_sweep import join_rollout_conditions, stack_dataset_members


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


def test_joined_rollout_keeps_condition_axis_and_refuses_coordinate_drift():
    reads = []

    def rollout(condition, offset, trials=("right", "left")):
        resident = Dataset.from_arrays(
            {"v": Variable((np.arange(6) + offset).reshape((1, 1, 2, 3)), (
                "checkpoint", "condition", "trial", "state_time",
            ))},
            coords={
                "condition": Variable(np.asarray([condition]), ("condition",)),
                "trial": Variable(np.asarray(trials), ("trial",)),
                "state_time": Variable(np.asarray([0., 0.1, 0.2]), ("state_time",)),
            },
            attrs={"state_lifetime": "trial", "variation_sha256": condition},
        )

        def load(name, index):
            if name == "v":
                reads.append(condition)
            return resident.data(name, index)

        return Dataset.bind(
            resident.descriptor, tensor_shapes={"v": (1, 1, 2, 3), "state_time": (3,)},
            load_tensor=load,
        )

    heavy, control = rollout("b", 10), rollout("a", 0)
    joined = join_rollout_conditions([("heavy", heavy), ("control", control)])
    assert reads == []
    assert joined.data("member").tolist() == ["heavy", "control"]
    assert joined.data("condition").tolist() == ["b", "a"]
    assert joined.data("trial").tolist() == ["right", "left"]
    assert joined.descriptor.attrs["condition_provenance"] == {
        "heavy": {"variation_sha256": "b"}, "control": {"variation_sha256": "a"},
    }
    np.testing.assert_array_equal(joined.data("v", (0, 1, 0)), [0, 1, 2])
    assert reads == ["a"]
    np.testing.assert_array_equal(joined.data("v", (0, slice(None, None, -1), 0)), [[0, 1, 2], [10, 11, 12]])
    with pytest.raises(ValueError, match="coordinate|field 'trial'"):
        join_rollout_conditions([("heavy", heavy), ("control", rollout("a", 0, ("left", "right")))])
    with pytest.raises(ValueError, match="distinct condition IDs"):
        join_rollout_conditions([("heavy", heavy), ("alias", heavy)])
