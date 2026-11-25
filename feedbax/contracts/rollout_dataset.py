"""The fields a rollout publishes for its Task signals and trial clock."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from feedbax.contracts.dataset import FieldDescriptor, TrialFigureTime
from feedbax.contracts.expression_values import LENGTH, ScopedValue, ValueType
from feedbax.contracts.operation import DatasetPort
from feedbax.contracts.request_refs import BestRank, ContentRef, ProductRef, UpdateSeriesMember
from feedbax.contracts.retained_state import RetainedStateField
from feedbax.contracts.semantic_values import semantic_key
from feedbax.contracts.task_document import Signal


type UpdateAxis = Literal["checkpoint", "update"]
XY = FieldDescriptor(dims=("xy",))


def rollout_update_axis(subject: ContentRef | ProductRef | BestRank) -> UpdateAxis | None:
    if isinstance(subject, ContentRef):
        return None
    return "update" if isinstance(subject, ProductRef) and isinstance(subject.select, UpdateSeriesMember) else "checkpoint"


def rollout_leading(update_axis: UpdateAxis | None) -> tuple[str, ...]:
    return ("condition",) if update_axis is None else (update_axis, "condition")


def rollout_signal_field(
    signal_id: str, name: str, type_: ValueType, leading: tuple[str, ...],
) -> tuple[str, FieldDescriptor]:
    key = semantic_key("task", "output", (), signal_id)
    axes = (*leading, "trial", "transition")
    if type_.shape == (2,) and type_.dimensions == LENGTH:
        return key, FieldDescriptor(dims=(*axes, "xy"), units="m", frame="world", label=name)
    components = tuple(f"{key}:axis{index}" for index in range(len(type_.shape)))
    return key, FieldDescriptor(dims=(*axes, *components), label=name)


def rollout_coordinates(update_axis: UpdateAxis | None) -> dict[str, FieldDescriptor]:
    time = TrialFigureTime()
    coordinates = {} if update_axis is None else {update_axis: FieldDescriptor(dims=(update_axis,))}
    return {
        **coordinates,
        "condition": FieldDescriptor(dims=("condition",)),
        "trial": FieldDescriptor(dims=("trial",)),
        "state_time": FieldDescriptor(dims=("state_time",), units="s", trial_time=time),
        "transition_start": FieldDescriptor(dims=("transition",), units="s", trial_time=time),
        "transition_end": FieldDescriptor(dims=("transition",), units="s", trial_time=time),
    }


def rollout_dataset_port(
    signals: Iterable[ScopedValue | Signal], schema: str, update_axis: UpdateAxis | None,
    *, model_states: Sequence[RetainedStateField] = (),
) -> DatasetPort:
    leading = rollout_leading(update_axis)
    fields = dict(rollout_signal_field(signal.id, signal.name, signal.type, leading) for signal in signals)
    fields.update({state.key: state.descriptor((*leading, "trial")) for state in model_states})
    coordinates = rollout_coordinates(update_axis) | {
        axis: FieldDescriptor(dims=(axis,)) for state in model_states for axis in state.axes
    }
    if any("xy" in field.dims for field in fields.values()):
        coordinates["xy"] = XY
    return DatasetPort(schema_ids=(schema,), variables=fields, coordinates=coordinates)


def common_rollout_port(ports: Sequence[DatasetPort]) -> DatasetPort:
    """Fields a Figure can read regardless of which rollout supplies it."""
    first = ports[0]
    if len(ports) == 1:
        return first

    def common(fields: Sequence[Mapping[str, FieldDescriptor]]) -> dict[str, FieldDescriptor]:
        shared = {}
        for key, descriptor in fields[0].items():
            label = descriptor.label
            for other in fields[1:]:
                candidate = other.get(key)
                if candidate is None or candidate.model_copy(update={"label": descriptor.label}) != descriptor:
                    break
                if label != candidate.label:
                    label = None
            else:
                shared[key] = descriptor.model_copy(update={"label": label})
        return shared

    return DatasetPort(
        schema_ids=tuple(dict.fromkeys(schema for port in ports for schema in port.schema_ids)),
        variables=common([port.variables for port in ports]),
        coordinates=common([port.coordinates for port in ports]),
    )
