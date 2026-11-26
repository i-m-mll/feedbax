"""The values of a step's axes, each gathered from the runs' coordinates that are that value.

A value is named by one ID wherever Studio and the service meet: in the window map, and in a selection of runs by
their values.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import json

from feedbax.contracts.jobs import StudioProjectWindowMapAxis, StudioProjectWindowMapAxisValue
from feedbax.contracts.project_sweep import SweepCoordinate, SweepInputCoordinate, sweep_axis_name, sweep_value_labels
from feedbax.contracts.run_rule import value_text
from feedbax.planner.project_plan import PlannedRow, RunKey


type _ValueIdentity = tuple[str, object]


def _value_identity(coordinate: SweepCoordinate, *, member: bool) -> _ValueIdentity:
    """What makes two coordinates one value: a member is its key; a number is its number; otherwise its name."""
    value = coordinate.value
    if coordinate.value_id == "":
        return "kept", ""
    if member:
        return "member", coordinate.value_id
    if isinstance(value, bool):
        return "bool", value
    if isinstance(value, int | float):
        return "number", value
    if isinstance(value, str):
        return "name", value
    return "name", coordinate.label if coordinate.label is not None else coordinate.value_id


def gather_axes(
    rows: Sequence[PlannedRow],
) -> tuple[list[StudioProjectWindowMapAxis], dict[tuple[RunKey, str], str]]:
    """The axes one step's ``rows`` carry, each value gathered from the coordinates that are that value, and the ID of
    the value each row has on each axis."""
    gathered: dict[str, dict[_ValueIdentity, list[SweepCoordinate]]] = {}
    identities: dict[tuple[RunKey, str], _ValueIdentity] = {}
    for row in rows:
        for axis, coordinate in row.selection_coordinates.items():
            identity = _value_identity(coordinate, member=isinstance(coordinate, SweepInputCoordinate))
            gathered.setdefault(axis, {}).setdefault(identity, []).append(coordinate)
            identities[row.key, axis] = identity
    axes: list[StudioProjectWindowMapAxis] = []
    ids: dict[tuple[str, _ValueIdentity], str] = {}
    for axis, values in gathered.items():
        order = list(values)
        if all(kind == "number" for kind, _ in order):
            order.sort(key=lambda identity: identity[1])
        keys = Counter(key for coordinates in values.values() for key in {item.value_id for item in coordinates})
        displayed = []
        for identity in order:
            stored = {item.value_id for item in values[identity]}
            (key,) = stored if len(stored) == 1 else (None,)
            ids[axis, identity] = key if key is not None and keys[key] == 1 else _gathered_id(identity)
            coordinate = values[identity][0]
            label, value_label = sweep_value_labels(axis, coordinate)
            displayed.append(StudioProjectWindowMapAxisValue(
                id=ids[axis, identity], name=label, value_label=value_label, rule=value_text(coordinate),
            ))
        axes.append(StudioProjectWindowMapAxis(
            id=axis, name=sweep_axis_name(axis, (item for matches in values.values() for item in matches)),
            values=displayed,
        ))
    return axes, {at: ids[at[1], identity] for at, identity in identities.items()}


def _gathered_id(identity: _ValueIdentity) -> str:
    """A value's ID when no stored key names it alone: ``=`` and the value, which no stored key can be."""
    return "=" + json.dumps(identity[1])


def axis_values(rows: Iterable[PlannedRow]) -> dict[tuple[RunKey, str], str]:
    """The ID of each row's value on each of its axes, gathered over the rows of its step and variant."""
    steps: dict[tuple[str, str], list[PlannedRow]] = defaultdict(list)
    for row in rows:
        steps[row.key.variant, row.key.step].append(row)
    return {at: value for group in steps.values() for at, value in gather_axes(group)[1].items()}


def values_rule(rows: Iterable[PlannedRow], axes: Mapping[str, Iterable[str]]) -> str:
    """The run rule naming, for each axis, the values of one step's ``rows`` whose IDs are ``axes[axis]``."""
    chosen = {axis: set(values) for axis, values in axes.items()}
    gathered = {axis.id: axis for axis in gather_axes(list(rows))[0]}
    return " ".join(
        f"{axis}=" + ",".join(value.rule for value in gathered[axis].values if value.id in chosen[axis])
        for axis in sorted(chosen)
    )
