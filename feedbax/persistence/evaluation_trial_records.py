"""Store and project trial records alongside a rollout's numerical data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from feedbax.contracts.base import StrictModel
from feedbax.contracts.expression_values import LENGTH, Level
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.semantic_values import model_state_address, semantic_key
from feedbax.contracts.studio_place import path_parts
from feedbax.contracts.trial_record import TaskTimelineCue, TrialRecord, TrialStateTrack, TrialTerm

if TYPE_CHECKING:
    from feedbax.dataset import Dataset, Variable
    from feedbax.execution.trial_execution import TrialEvaluation
    from feedbax.runtime.raw_state import RawStateLayout
    from feedbax.tasks.document import TrialSource


class RecordedTrialLayout(StrictModel):
    """Static field layout shared by all members of one evaluation."""

    epochs: tuple[tuple[str, str], ...]
    positions: tuple[str, ...]
    configurations: tuple[str, ...]


def _task_field(facet: str, name: str) -> str:
    if facet == "output":
        return semantic_key("task", "output", (), name)
    return pointer(("v1", "task", facet, name))


def _term_field(facet: str, name: str) -> str:
    return "objective" + pointer((facet, name))


def trial_dataset_fields(
    task: TrialSource, evaluated: TrialEvaluation, layout: RawStateLayout, *,
    leading: tuple[str, ...], labels: dict[str, str],
) -> tuple[dict[str, Variable], dict[str, Variable], RecordedTrialLayout]:
    import numpy as np

    from feedbax.dataset import Variable

    trial = evaluated.trials
    fields: dict[str, Variable] = {}
    coords: dict[str, Variable] = {}
    trial_axes = (*leading, "trial")
    transition_axes = (*trial_axes, "transition")
    variable_levels = {value.id: value.level for value in task.interface.variables}
    signal_types = {value.id: value.type for value in task.interface.signals}

    def put(path, value, axes, *, point=False):
        array = np.asarray(value)
        additional = array.ndim - (len(axes) - len(leading))
        if point and additional == 1 and array.shape[-1] == 2:
            coords["xy"] = Variable(np.asarray(["x", "y"]), ("xy",))
            fields[path] = Variable(
                array.reshape((1,) * len(leading) + array.shape), (*axes, "xy"),
                units="m", frame="world",
            )
        else:
            extra = tuple(f"{path}:axis{i}" for i in range(additional))
            fields[path] = Variable(array.reshape((1,) * len(leading) + array.shape), (*axes, *extra))

    for field, values in zip(
        ("epoch_start", "epoch_end", "epoch_outcome"), trial.epochs, strict=True,
    ):
        put(field, values, (*trial_axes, "epoch"))
    for key, value in trial.inits.items():
        put(_task_field("initial", key), value, trial_axes)
    for key, value in trial.inputs.items():
        signal_type = signal_types[key]
        put(
            _task_field("output", key), value, transition_axes,
            point=signal_type.shape == (2,) and signal_type.dimensions == LENGTH,
        )
        put(_task_field("presence", key), trial.present[key], transition_axes)
    for key, value in trial.variables.items():
        axes = transition_axes if variable_levels[key] == Level.STEP else trial_axes
        put(_task_field("variable", key), value, axes)
    for key, value in (trial.variable_defined or {}).items():
        put(_task_field("variable_defined", key), value, transition_axes)
    for name, trace in (evaluated.term_traces or {}).items():
        put(_term_field("steps", name), trace.steps, transition_axes)
        if trace.values is not None:
            put(_term_field("values", name), np.where(trace.steps, trace.values, 0.0), transition_axes)
        put(_term_field("trial_coverage", name), trace.covered, trial_axes)
        put(_term_field("trial_values", name), np.where(trace.covered, trace.reduced.value, 0.0), trial_axes)
    return fields, coords, RecordedTrialLayout(
        epochs=tuple((epoch, labels.get(epoch, epoch)) for epoch in task.epochs),
        positions=tuple(value.key for value in layout.values if value.position),
        configurations=tuple(value.key for value in layout.values if value.configuration),
    )


def read_trial_record(
    dataset: Dataset, trial_id: str, selection: tuple[int, ...], layout: RecordedTrialLayout,
) -> TrialRecord:
    """Read exactly the selected member, without touching unrelated retained states."""
    groups: dict[str, dict] = {}
    for path in dataset.descriptor.variables:
        parts = path_parts(path)
        if len(parts) == 4 and parts[:2] == ("v1", "task"):
            _, _, facet, key = parts
            groups.setdefault(facet, {})[key] = dataset.data(path, selection).tolist()
    timeline = [
        TaskTimelineCue(epoch=epoch, label=label, start=start, end=end, outcome=outcome)
        for (epoch, label), start, end, outcome in zip(
            layout.epochs,
            dataset.data("epoch_start", selection).tolist(),
            dataset.data("epoch_end", selection).tolist(),
            dataset.data("epoch_outcome", selection).tolist(),
            strict=True,
        )
    ]
    term_names = (
        path_parts("/" + path)[-1] for path in dataset.descriptor.variables if path.startswith("objective/steps/")
    )
    terms = {}
    for name in term_names:
        covered = dataset.data(_term_field("trial_coverage", name), selection).item()
        values_field = _term_field("values", name)
        terms[name] = TrialTerm(
            steps=dataset.data(_term_field("steps", name), selection).tolist(),
            covered=covered,
            values=(
                dataset.data(values_field, selection).tolist()
                if values_field in dataset.descriptor.variables else None
            ),
            value=dataset.data(_term_field("trial_values", name), selection).item() if covered else None,
        )
    return TrialRecord(
        id=trial_id, inits=groups.get("initial", {}), inputs=groups.get("output", {}),
        present=groups.get("presence", {}), variables=groups.get("variable", {}),
        variable_defined=groups.get("variable_defined", {}), timeline=timeline, terms=terms,
        states={
            model_state_address(name).to_text(): read_position_track(dataset, name, selection)
            for name in layout.positions
        },
        configurations={
            model_state_address(name).to_text(): dataset.data(name, selection).tolist()
            for name in layout.configurations
        },
    )


def read_position_track(dataset: Dataset, name: str, selection: tuple[int, ...]) -> TrialStateTrack:
    field = dataset.descriptor.variables[name]
    values = dataset.data(name, selection)
    if field.frame != "world" or values.shape != (dataset.descriptor.dimensions["state_time"], 2):
        raise ValueError("The retained position has an incompatible state layout.")
    return TrialStateTrack(samples=values.tolist(), units=field.units, frame=field.frame)
