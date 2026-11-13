import jax
import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.contracts.dataset import DatasetDescriptor, FieldDescriptor
from feedbax.dataset import Dataset, Variable


@pytest.mark.parametrize("bind_json", [False, True])
def test_roundtrip_distinguishes_tensor_fields_from_inline_labels(bind_json):
    timing = np.array(["bounded", "inapplicable"])
    coord = np.empty((2, 2), dtype=object)
    coord[0, 0] = "nominal"
    coord[0, 1] = None
    coord[1, 0] = {"side": ["left", "right"]}
    coord[1, 1] = 0.5
    states = np.array([[0.2, 0.3], [0.4, 0.1]])
    original = Dataset.from_arrays(
        {
            "mechanics.effector.pos": Variable(
                states, ("trial", "time"), units="m", quantity="position", frame="world"
            ),
            "timing_applicability": Variable(timing, ("trial",)),
        },
        coords={"time": Variable(coord, ("trial", "time"))},
        attrs={"parameters": {"enabled": True}},
        schema_id="experiment.states",
    )

    descriptor = DatasetDescriptor.from_json(original.descriptor.to_json())
    assert descriptor.schema_id == "experiment.states"
    assert descriptor.attrs["parameters"] == {"enabled": True}
    assert descriptor.tensor_names == {"mechanics.effector.pos"}
    state_field = descriptor.variables["mechanics.effector.pos"]
    assert (state_field.units, state_field.quantity, state_field.frame) == (
        "m",
        "position",
        "world",
    )
    assert descriptor.coords["time"].dims == ("trial", "time")

    reads = []

    def read(name, index):
        reads.append((name, index))
        return states[index]

    metadata = descriptor.json_value() if bind_json else descriptor
    restored = Dataset.bind(
        metadata, tensor_shapes={"mechanics.effector.pos": (2, 2)}, load_tensor=read
    )
    np.testing.assert_array_equal(restored.data("timing_applicability"), timing)
    assert restored.data("timing_applicability").dtype.kind == "U"
    assert restored.data("time").tolist() == coord.tolist()
    assert restored.data("time", (1, 0)).item() == {"side": ["left", "right"]}
    assert not reads

    restored.data("time")[1, 0]["side"].clear()
    if isinstance(metadata, DatasetDescriptor):
        metadata.coords.clear()
    else:
        metadata["coords"]["time"]["inline_values"][2]["side"].clear()
    assert restored.data("time", (1, 0)).item() == {"side": ["left", "right"]}
    with pytest.raises(ValueError, match="host-only"):
        restored.project("mechanics.effector.pos", coordinates=("time",))
    assert not reads

    selection = (1, slice(0, 1))
    np.testing.assert_array_equal(restored.data("mechanics.effector.pos", selection), [0.4])
    assert reads == [("mechanics.effector.pos", selection)]


@pytest.mark.parametrize(
    "shapes",
    [
        {"trajectory": (3, 2), "extra": (1,)},
        {"trajectory": (2, 3)},
        {},
    ],
)
def test_incompatible_tensor_headers_fail_before_loading(shapes):
    descriptor = DatasetDescriptor(
        dimensions={"trial": 3, "time": 2},
        variables={"trajectory": FieldDescriptor(dims=("trial", "time"))},
    )

    def unexpected_read(name, index):
        pytest.fail(f"binding read {name}{index}")

    with pytest.raises(ValueError):
        Dataset.bind(descriptor, tensor_shapes=shapes, load_tensor=unexpected_read)


def test_descriptor_refuses_non_json_labels_and_inconsistent_dimensions():
    objects = np.empty((1,), dtype=object)
    objects[0] = {"bad": float("nan")}
    with pytest.raises(ValueError):
        Dataset.from_arrays({}, coords={"label": Variable(objects, ("trial",))})
    with pytest.raises(ValueError):
        Dataset.from_arrays(
            {"state": Variable(np.empty(2), ("trial",))},
            coords={"trial": Variable(np.empty(3), ("trial",))},
        )
    with pytest.raises(ValueError):
        DatasetDescriptor.from_json(
            '{"schema_version":"feedbax.dataset.v1","dimensions":{"t":2},'
            '"variables":{"phase":{"dims":["t"],"inline_values":["start"]}}}'
        )


def test_projection_jit_vmap_and_grad_with_dynamic_numerical_coordinates():
    def project(gain, labels, units, note):
        return Dataset.from_arrays(
            {"force": Variable(jnp.array([[1.0, 2.0], [3.0, 4.0]]), ("trial", "t"), units=units)},
            coords={
                "gain": Variable(jnp.asarray(gain), ("trial",)),
                "phase": Variable(np.array(labels), ("t",)),
            },
            attrs={"note": note},
        ).project("force", coordinates=("gain",))

    traces = 0

    def work(trial):
        return jnp.sum(trial.data("force")) * trial.data("gain")

    @jax.jit
    def batch(projection):
        nonlocal traces
        traces += 1
        return jax.vmap(work)(projection)

    original = project([2.0, 3.0], ["pre", "post"], "N", "first run")
    updated = project([4.0, 5.0], ["before", "after"], "kN", "second run")
    np.testing.assert_allclose(batch(original), [6.0, 21.0])
    np.testing.assert_allclose(batch(updated), [12.0, 35.0])
    assert traces == 1

    gradient = jax.grad(lambda projection: batch(projection).sum())(original)
    np.testing.assert_allclose(gradient.data("force"), [[2.0, 2.0], [3.0, 3.0]])
    np.testing.assert_allclose(gradient.data("gain"), [3.0, 7.0])
