"""The layout of gathered members stacked along their labels, decided without reading any dataset."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from math import prod

from feedbax.contracts.dataset import FieldDescriptor
from feedbax.contracts.project_sweep import SweepCoordinate, SweepInputMember, sweep_value_identity_bytes
from feedbax.contracts.request_refs import InputMember

MEMBER = "member"


def _same(left: SweepCoordinate, right: SweepCoordinate) -> bool:
    return left.value_id == right.value_id and sweep_value_identity_bytes(left.value) == (
        sweep_value_identity_bytes(right.value)
    )


@dataclass(frozen=True)
class StackLayout:
    """Labels that vary are leading dimensions, one per label when the members are their complete
    product in order and otherwise one ``member`` dimension; labels equal in every member are scalars."""

    members: tuple[SweepInputMember, ...]
    dimensions: Mapping[str, int]
    values: Mapping[str, tuple[SweepCoordinate, ...]]
    """Each varying label's values in order along its own dimension, when the members are a product."""
    constant: Mapping[str, SweepCoordinate]

    @property
    def varying(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            name for member in self.members for name in member.coordinates if name not in self.constant
        ))

    def dims(self, label: str) -> tuple[str, ...]:
        if label == MEMBER:
            return tuple(self.dimensions)
        if label in self.constant:
            return ()
        return (label,) if self.values else (MEMBER,)

    def fields(self) -> dict[str, FieldDescriptor]:
        """The coordinates the stack adds, by name, without their values."""
        fields = {MEMBER: FieldDescriptor(dims=self.dims(MEMBER))}
        for label in (*self.varying, *self.constant):
            fields[label] = FieldDescriptor(dims=self.dims(label))
            if any(label in member.coordinates and member.coordinates[label].value is not None for member in self.members):
                fields[f"{label}.value"] = FieldDescriptor(dims=self.dims(label))
        return fields


def stack_layout(members: Sequence[SweepInputMember]) -> StackLayout:
    if not members:
        raise ValueError("A dataset stack needs at least one member")
    if len({member.key for member in members}) != len(members):
        raise ValueError("A dataset stack cannot bind the same member key twice")
    by_label: dict[str, dict[str, SweepCoordinate]] = {}
    for member in members:
        for name, coordinate in member.coordinates.items():
            if name == MEMBER:
                raise ValueError("Label 'member' is reserved for the member dimension")
            prior = by_label.setdefault(name, {}).setdefault(coordinate.value_id, coordinate)
            if not _same(prior, coordinate):
                raise ValueError(f"Label {name!r} gives {coordinate.value_id!r} conflicting values")
    first = members[0].coordinates
    constant = {
        name: coordinate for name, coordinate in first.items()
        if all(name in member.coordinates and _same(member.coordinates[name], coordinate) for member in members[1:])
    }
    if len(members) == 1:
        return StackLayout(tuple(members), {}, {}, constant)
    varying = {name: values for name, values in by_label.items() if name not in constant}
    rows = [tuple(member.coordinates.get(name) for name in varying) for member in members]
    if (
        varying
        and all(None not in row for row in rows)
        and len(rows) == prod(len(values) for values in varying.values())
        and all(
            tuple(coordinate.value_id for coordinate in row) == ids
            for row, ids in zip(rows, product(*varying.values()), strict=True)
        )
    ):
        return StackLayout(
            tuple(members), {name: len(values) for name, values in varying.items()},
            {name: tuple(values.values()) for name, values in varying.items()}, constant,
        )
    return StackLayout(tuple(members), {MEMBER: len(members)}, {}, constant)


def labelled_members(members: Sequence[InputMember]) -> tuple[SweepInputMember, ...]:
    return tuple(
        SweepInputMember(member.key, {
            name: SweepCoordinate(label.value_id, value=label.value) for name, label in member.labels.items()
        })
        for member in members
    )
