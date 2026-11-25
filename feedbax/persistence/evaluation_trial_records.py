"""Store and project trial records alongside a rollout's numerical data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from feedbax.contracts.base import StrictModel
from feedbax.contracts.expression_registry import Evaluation
from feedbax.contracts.expression_values import LENGTH, DrawValue, Level, ValueBase
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.semantic_values import model_state_address, semantic_key
from feedbax.contracts.studio_place import path_parts
from feedbax.contracts.trial_clock import TrialClockChoice, TrialMeanOptions
from feedbax.contracts.trial_record import TaskTimelineCue, TrialRecord, TrialStateTrack, TrialTerm
from feedbax.execution.trial_clock import TrialClockRecord
from feedbax.execution.trial_places import BoundTermPlaces, TrialPlacePlan, finite_points, project_term_places
from feedbax.execution.trial_record import trial_weight_values

if TYPE_CHECKING:
    from feedbax.contracts.task_document import TaskDocument
    from feedbax.dataset import Dataset, Variable
    from feedbax.execution.trial_execution import TrialEvaluation
    from feedbax.runtime.raw_state import RawStateLayout
    from feedbax.tasks.document import TrialSource


class RecordedTrialLayout(StrictModel):
    """Static field layout shared by all members of one evaluation."""

    epochs: tuple[tuple[str, str], ...]
    positions: tuple[str, ...]
    configurations: tuple[str, ...]
    discrete_variables: tuple[TrialClockChoice, ...] = ()
    observations: dict[str, str] = Field(default_factory=dict)
    """Model origin addresses and their dataset variables, including shared raw state."""
    term_places: dict[str, BoundTermPlaces] = Field(default_factory=dict)

    def mean_options(self) -> TrialMeanOptions:
        return TrialMeanOptions(
            by=list(self.discrete_variables),
            align=[TrialClockChoice(id=epoch, label=label) for epoch, label in dict(self.epochs).items()],
        )


def _task_field(facet: str, name: str) -> str:
    if facet == "output":
        return semantic_key("task", "output", (), name)
    return pointer(("v1", "task", facet, name))


def _term_field(facet: str, name: str) -> str:
    return "objective" + pointer((facet, name))


def _point_field(term: str, index: int, facet: str) -> str:
    return "objective" + pointer(("point_places", term, str(index), facet))


def trial_dataset_fields(
    task: TrialSource, evaluated: TrialEvaluation, layout: RawStateLayout, *,
    leading: tuple[str, ...], labels: dict[str, str], document: TaskDocument | None = None,
    observation_origins: tuple[tuple[str, str], ...] = (),
    places: TrialPlacePlan | None = None,
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
        if trace.weight is not None:
            defined = trace.weight.defined & np.isfinite(trace.weight.value)
            put(_term_field("weights", name), np.where(defined, trace.weight.value, 0.0), transition_axes)
            put(_term_field("weight_defined", name), defined, transition_axes)
        put(_term_field("trial_coverage", name), trace.covered, trial_axes)
        put(_term_field("trial_values", name), np.where(trace.covered, trace.reduced.value, 0.0), trial_axes)
        for index, observed in enumerate(trace.points):
            values, defined = finite_points(observed.value, observed.defined)
            put(_point_field(name, index, "values"), values, transition_axes, point=True)
            put(_point_field(name, index, "defined"), defined, transition_axes)
    raw_states = {
        model_state_address(value.key).to_text(): value.key for value in layout.values
        if value.key.startswith("/v1/model/state/")
    }
    observations = {}
    for operand, origin in observation_origins:
        if origin in observations:
            continue
        if origin in raw_states:
            observations[origin] = raw_states[origin]
        else:
            path = semantic_key("model", "output", (), origin)
            samples = evaluated.retained[operand]
            if not np.isfinite(samples).all():
                raise FloatingPointError(f"Nonfinite observations at {origin}")
            put(path, samples, transition_axes)
            observations[origin] = path
    choices = {
        variable.id for variable in document.variables
        if isinstance(variable.value, DrawValue) and variable.value.family == "choice"
    } if document is not None else set()
    if document is not None:
        # All rows of a fixed cohort assign the same fields. Each one has a finite
        # authored set of values in this evaluation, even for real-valued positions.
        choices.update(document.validation_trials[0].variables)
    return fields, coords, RecordedTrialLayout(
        epochs=tuple((epoch, labels.get(epoch, epoch)) for epoch in task.epochs),
        positions=tuple(value.key for value in layout.values if value.position),
        configurations=tuple(value.key for value in layout.values if value.configuration),
        discrete_variables=tuple(
            TrialClockChoice(id=value.id, label=value.name)
            for value in task.interface.variables
            if value.level < Level.STEP and (value.type.base != ValueBase.REAL or value.id in choices)
        ),
        observations=observations,
        term_places=places.terms if places is not None else {},
    )


def read_trial_timeline(
    dataset: Dataset, selection: tuple[int, ...], layout: RecordedTrialLayout,
) -> list[TaskTimelineCue]:
    return [
        TaskTimelineCue(epoch=epoch, label=label, start=start, end=end, outcome=outcome)
        for (epoch, label), start, end, outcome in zip(
            layout.epochs,
            dataset.data("epoch_start", selection).tolist(),
            dataset.data("epoch_end", selection).tolist(),
            dataset.data("epoch_outcome", selection).tolist(),
            strict=True,
        )
    ]


def read_trial_clock(
    dataset: Dataset, trial_id: str, selection: tuple[int, ...], layout: RecordedTrialLayout,
) -> TrialClockRecord:
    """Read only the discrete variables and epoch boundaries used by the clock."""
    return TrialClockRecord(
        id=trial_id,
        values={
            choice.id: dataset.data(_task_field("variable", choice.id), selection).tolist()
            for choice in layout.discrete_variables
        },
        timeline=read_trial_timeline(dataset, selection, layout),
        state_time=dataset.data("state_time"),
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
    term_names = (
        path_parts("/" + path)[-1] for path in dataset.descriptor.variables if path.startswith("objective/steps/")
    )
    terms = {}
    for name in term_names:
        def point(index: int) -> Evaluation | None:
            values = _point_field(name, index, "values")
            if values not in dataset.descriptor.variables:
                return None
            return Evaluation(
                dataset.data(values, selection),
                dataset.data(_point_field(name, index, "defined"), selection),
            )

        covered = dataset.data(_term_field("trial_coverage", name), selection).item()
        values_field = _term_field("values", name)
        weights_field = _term_field("weights", name)
        relations, places = project_term_places(layout.term_places.get(name), point)
        terms[name] = TrialTerm(
            steps=dataset.data(_term_field("steps", name), selection).tolist(),
            covered=covered,
            values=(
                dataset.data(values_field, selection).tolist()
                if values_field in dataset.descriptor.variables else None
            ),
            value=dataset.data(_term_field("trial_values", name), selection).item() if covered else None,
            weights=trial_weight_values(
                dataset.data(weights_field, selection), dataset.data(_term_field("weight_defined", name), selection),
            ) if weights_field in dataset.descriptor.variables else None,
            relations=relations,
            places=places,
        )
    return TrialRecord(
        id=trial_id, inits=groups.get("initial", {}), inputs=groups.get("output", {}),
        present=groups.get("presence", {}), variables=groups.get("variable", {}),
        variable_defined=groups.get("variable_defined", {}), timeline=read_trial_timeline(dataset, selection, layout),
        terms=terms,
        states={
            model_state_address(name).to_text(): read_position_track(dataset, name, selection)
            for name in layout.positions
        },
        configurations={
            model_state_address(name).to_text(): dataset.data(name, selection).tolist()
            for name in layout.configurations
        },
        observations={
            origin: dataset.data(path, selection)[1 if path.startswith("/v1/model/state/") else 0:].tolist()
            for origin, path in layout.observations.items()
        },
    )


def read_position_track(dataset: Dataset, name: str, selection: tuple[int, ...]) -> TrialStateTrack:
    field = dataset.descriptor.variables[name]
    values = dataset.data(name, selection)
    if field.frame != "world" or values.shape != (dataset.descriptor.dimensions["state_time"], 2):
        raise ValueError("The retained position has an incompatible state layout.")
    return TrialStateTrack(samples=values.tolist(), units=field.units, frame=field.frame)
