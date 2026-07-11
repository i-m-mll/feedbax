"""Interpretability APIs for flattened authored intent."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from feedbax.contracts.run_composition import FlattenedIntent


@dataclass(frozen=True)
class SemanticDifference:
    path: str
    left: Any
    right: Any
    left_layer: str | None = None
    right_layer: str | None = None


def layered_semantic_diff(
    left: FlattenedIntent | Mapping[str, Any],
    right: FlattenedIntent | Mapping[str, Any],
) -> list[SemanticDifference]:
    left_payload = left.payload if isinstance(left, FlattenedIntent) else left
    right_payload = right.payload if isinstance(right, FlattenedIntent) else right
    left_attr = left.attribution if isinstance(left, FlattenedIntent) else {}
    right_attr = right.attribution if isinstance(right, FlattenedIntent) else {}
    result: list[SemanticDifference] = []
    _diff(left_payload, right_payload, "", left_attr, right_attr, result)
    return result


def detect_near_duplicate_lanes(
    lanes: Mapping[str, FlattenedIntent | Mapping[str, Any]], *, max_differences: int = 1
) -> list[tuple[str, str, list[SemanticDifference]]]:
    names = sorted(lanes)
    result = []
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            differences = layered_semantic_diff(lanes[left], lanes[right])
            if len(differences) <= max_differences:
                result.append((left, right, differences))
    return result


def _diff(
    left: Any,
    right: Any,
    path: str,
    left_attr: Mapping[str, str],
    right_attr: Mapping[str, str],
    output: list[SemanticDifference],
) -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in sorted(set(left) | set(right)):
            _diff(left.get(key), right.get(key), _join(path, str(key)), left_attr, right_attr, output)
        return
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)) and isinstance(
        right, Sequence
    ) and not isinstance(right, (str, bytes)):
        for index in range(max(len(left), len(right))):
            lhs = left[index] if index < len(left) else None
            rhs = right[index] if index < len(right) else None
            _diff(lhs, rhs, _join(path, str(index)), left_attr, right_attr, output)
        return
    if left != right:
        output.append(
            SemanticDifference(path, left, right, left_attr.get(path), right_attr.get(path))
        )


def _join(parent: str, child: str) -> str:
    return child if not parent else f"{parent}.{child}"
