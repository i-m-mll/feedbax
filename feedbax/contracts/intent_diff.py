"""Authored changes between managed requests."""


from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from feedbax.contracts._request_codec import same_typed_value

if TYPE_CHECKING:
    from feedbax.contracts.request_identity import DecodedRequest

_ABSENT = object()


@dataclass(frozen=True)
class SemanticDifference:
    path: str
    left: Any
    right: Any
    parts: tuple[str, ...] = ()


def first_request_change(left: DecodedRequest, right: DecodedRequest) -> SemanticDifference:
    """Describe the first changed authored field in the two verified definitions."""
    differences: list[SemanticDifference] = []
    _diff(
        left.spec.model_dump(mode="json"),
        right.spec.model_dump(mode="json"),
        (),
        differences,
    )
    if differences:
        change = differences[0]
        return replace(change, path=f"payload.{change.path}", parts=("payload", *change.parts))
    if left.inputs != right.inputs:
        return SemanticDifference(
            "inputs",
            {role: [member.model_dump(mode="json") for member in group] for role, group in left.inputs.items()},
            {role: [member.model_dump(mode="json") for member in group] for role, group in right.inputs.items()},
        )
    return SemanticDifference("request_id", left.id, right.id)


def _diff(
    left: Any,
    right: Any,
    parts: tuple[str, ...],
    output: list[SemanticDifference],
) -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in sorted(set(left) | set(right)):
            _diff(left.get(key, _ABSENT), right.get(key, _ABSENT), (*parts, str(key)), output)
        return
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)) and isinstance(
        right, Sequence
    ) and not isinstance(right, (str, bytes)):
        for index in range(max(len(left), len(right))):
            lhs = left[index] if index < len(left) else _ABSENT
            rhs = right[index] if index < len(right) else _ABSENT
            _diff(lhs, rhs, (*parts, str(index)), output)
        return
    if left is _ABSENT or right is _ABSENT or not same_typed_value(left, right):
        path = ".".join(parts)
        output.append(SemanticDifference(
            path, None if left is _ABSENT else left, None if right is _ABSENT else right,
            parts,
        ))
