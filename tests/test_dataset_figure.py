"""Render the first walkthrough figure directly from a stored dataset."""

import numpy as np
import plotly.io as pio
import pytest

from feedbax.analysis.builtin_operations import register_builtin_operations
from feedbax.analysis.figure_presets import velocity_summary_figure
from feedbax.analysis.operations import OperationRegistry
from feedbax.contracts.figure_document import FigureDocument, validate_figure_bindings
from feedbax.dataset import Dataset, Variable
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import open_dataset, publish_dataset
from feedbax.plot.dataset_figure import render


def test_band_uses_condition_order_physical_time_and_stored_bounds(tmp_path, monkeypatch):
    document = velocity_summary_figure()
    registry = OperationRegistry()
    register_builtin_operations(registry)
    port = registry.resolve("feedbax.summarize_velocity").definition.outputs["summary"]
    validate_figure_bindings(document, {"summary": port})

    time = np.array([0.0, 0.031, 0.11], dtype=np.float64)
    mean = np.array([[[2, 3, 4], [6, 7, 8]], [[30, 40, 50], [60, 70, 80]]])
    lower = mean - [0.2, 0.7, 1.1]
    upper = mean + [1.5, 0.4, 0.9]
    dataset = Dataset.from_arrays(
        {
            name: Variable(values, port.variables[name].dims, units="m/s", frame="reach-aligned")
            for name, values in (("mean", mean), ("lower", lower), ("upper", upper))
        },
        coords={
            "checkpoint": Variable(np.array([31, 7]), ("checkpoint",)),
            "condition": Variable(np.array(["perturbed", "control"]), ("condition",)),
            "state_time": Variable(time, ("state_time",), units="s"),
        },
        schema_id=port.schema_ids[0],
    )
    store = ImmutableArtifactBlobProvider(tmp_path)
    ref = publish_dataset(store, dataset)
    monkeypatch.setattr(pio.templates, "default", "plotly_dark")
    with (
        store.open_owned_object(ref.object_ref) as path,
        open_dataset(path, expected_descriptor_sha256=ref.descriptor_sha256) as reopened,
    ):
        figure = render(document, {"summary": reopened})
    assert pio.templates.default == "plotly_dark"
    assert (
        figure.layout.template.layout.paper_bgcolor
        == pio.templates["plotly_white"].layout.paper_bgcolor
    )
    assert figure.layout.xaxis.title.text == "Time (s)"
    assert figure.layout.yaxis.title.text == "Velocity (m/s)"
    assert len(figure.data) == 6
    centers = figure.data[::3]
    assert [trace.name for trace in centers] == ["perturbed", "control"]
    assert centers[0].line.color != centers[1].line.color
    for index, start in enumerate((0, 3)):
        center, high, low = figure.data[start : start + 3]
        for trace, values in ((center, mean), (high, upper), (low, lower)):
            np.testing.assert_array_equal(trace.x, time)
            np.testing.assert_array_equal(trace.y, values[0, index])
        assert low.fill == "tonexty"
        assert high.showlegend is False and low.showlegend is False


def test_unknown_band_channel_is_refused_against_the_operation_declaration():
    document = velocity_summary_figure().model_dump(mode="json")
    document["layers"][0]["encode"]["upper"] = "missing_upper"
    registry = OperationRegistry()
    register_builtin_operations(registry)
    port = registry.resolve("feedbax.summarize_velocity").definition.outputs["summary"]
    with pytest.raises(ValueError, match="trial-summary.*upper.*missing_upper"):
        validate_figure_bindings(FigureDocument.model_validate(document), {"summary": port})
