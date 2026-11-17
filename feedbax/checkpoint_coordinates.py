"""Finite standard Train coordinates shared by Plan and the local Host."""

from collections.abc import Sequence

MAX_CHECKPOINT_COORDINATE = 2**53 - 1
MAX_CHECKPOINT_MEMBERS = 64


def _valid_coordinate(value: int) -> bool:
    return type(value) is int and 0 < value <= MAX_CHECKPOINT_COORDINATE


def finite_coordinates(values: Sequence[int]) -> tuple[int, ...]:
    coordinates = tuple(values)
    if (
        not 0 < len(coordinates) <= MAX_CHECKPOINT_MEMBERS
        or any(not _valid_coordinate(value) for value in coordinates)
        or len(set(coordinates)) != len(coordinates)
    ):
        raise ValueError("expected one to 64 distinct positive safe batch numbers")
    return coordinates


def cadence_coordinates(every_k: int, through: int) -> tuple[int, ...]:
    if not (
        _valid_coordinate(every_k) and _valid_coordinate(through)
        and 1 <= through // every_k <= MAX_CHECKPOINT_MEMBERS
    ):
        raise ValueError("every_k through must select one to 64 batches")
    return tuple(range(every_k, through + 1, every_k))
