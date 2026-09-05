"""Semantic model-and-optimizer checkpoint transformation.

The canonical semantic ABI decides which leaves correspond.  This module only
executes the bounded shape change that the ABI makes unambiguous: appending a
new region on one declared parameter axis while retaining the target
constructor's values in that region.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Literal

import jax.tree as jt
import numpy as np

from feedbax.contracts.checkpoint_initialization import (
    CheckpointStructure,
    validate_checkpoint_semantic_abi,
)
from feedbax.contracts.checkpoints import CheckpointSemanticLeafIdentity


class CheckpointTransformErrorCode(StrEnum):
    """Stable runtime refusal categories for semantic transformation."""

    AMBIGUOUS_IDENTITY = "ambiguous-identity"
    UNKNOWN_IDENTITY = "unknown-identity"
    DTYPE_MISMATCH = "dtype-mismatch"
    UNSUPPORTED_SHAPE_CHANGE = "unsupported-shape-change"
    INCOMPLETE_ACCOUNTING = "incomplete-accounting"
    INVALID_POLICY = "invalid-policy"


class CheckpointTransformRefused(ValueError):
    """A fail-closed semantic transform refusal."""

    def __init__(
        self,
        code: CheckpointTransformErrorCode,
        message: str,
        *,
        identities: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.identities = tuple(identities)


@dataclass(frozen=True)
class AddedParameterAxis:
    """One semantic parameter whose target appends values on ``axis``."""

    component_slot: str
    parameter_role: str
    axis: int


@dataclass(frozen=True)
class CheckpointLeafTransfer:
    """Complete accounting for one source/target semantic leaf pair."""

    identity: str
    slot: str
    source_path: str
    target_path: str
    disposition: Literal["copied", "extended", "target_initialized", "deliberately_reset"]
    copied_shape: tuple[int, ...] | None = None
    initialized_axis: int | None = None
    initialized_start: int | None = None
    initialized_stop: int | None = None

    def machine_record(self) -> dict[str, Any]:
        """Return a JSON-shaped accounting record for checkpoint publication."""
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class SemanticCheckpointTransformResult:
    """Transformed values plus exhaustive leaf and policy accounting."""

    values: Mapping[str, Any]
    accounting: tuple[CheckpointLeafTransfer, ...]
    optimizer_policy: Literal["preserve", "reset"]
    learning_rate_schedule_policy: Literal["continue", "restart"]

    def accounting_records(self) -> tuple[dict[str, Any], ...]:
        """Return deterministic JSON-shaped records for the storage publisher."""
        return tuple(item.machine_record() for item in self.accounting)


@dataclass(frozen=True)
class _LocatedLeaf:
    semantic: CheckpointSemanticLeafIdentity
    value: Any
    index: int


def _key_path_to_text(path: Any) -> str:
    if not path:
        return "/"
    parts: list[str] = []
    for key in path:
        name = getattr(key, "name", None)
        index = getattr(key, "idx", None)
        mapping_key = getattr(key, "key", None)
        if name is not None:
            parts.append(str(name))
        elif index is not None:
            parts.append(str(index))
        elif mapping_key is not None:
            parts.append(str(mapping_key))
        else:
            parts.append(str(key))
    return "/" + "/".join(parts)


def _semantic_index(
    structure: CheckpointStructure,
    values: Mapping[str, Any],
) -> tuple[dict[str, _LocatedLeaf], dict[str, tuple[Any, list[Any]]]]:
    index: dict[str, _LocatedLeaf] = {}
    flattened: dict[str, tuple[Any, list[Any]]] = {}
    for slot in structure.slots:
        if slot.slot not in values:
            raise CheckpointTransformRefused(
                CheckpointTransformErrorCode.INCOMPLETE_ACCOUNTING,
                f"checkpoint values are missing declared slot {slot.slot!r}",
            )
        try:
            validate_checkpoint_semantic_abi(slot.semantic_abi)
        except ValueError as exc:
            raise CheckpointTransformRefused(
                CheckpointTransformErrorCode.INCOMPLETE_ACCOUNTING,
                f"checkpoint slot {slot.slot!r} has invalid semantic ABI: {exc}",
            ) from exc
        path_leaves, treedef = jt.flatten_with_path(values[slot.slot])
        leaves = [leaf for _, leaf in path_leaves]
        by_path = {
            _key_path_to_text(path): (position, leaf)
            for position, (path, leaf) in enumerate(path_leaves)
        }
        semantic_by_path = {leaf.raw_path: leaf for leaf in slot.semantic_abi.leaves}
        if set(by_path) != set(semantic_by_path):
            raise CheckpointTransformRefused(
                CheckpointTransformErrorCode.INCOMPLETE_ACCOUNTING,
                f"semantic ABI does not account for the materialized leaves in slot {slot.slot!r}",
            )
        for path, semantic in semantic_by_path.items():
            if semantic.identity in index:
                raise CheckpointTransformRefused(
                    CheckpointTransformErrorCode.AMBIGUOUS_IDENTITY,
                    f"checkpoint semantic identity {semantic.identity!r} occurs more than once",
                    identities=(semantic.identity,),
                )
            if semantic.owner_slot != slot.slot:
                raise CheckpointTransformRefused(
                    CheckpointTransformErrorCode.INCOMPLETE_ACCOUNTING,
                    f"semantic identity {semantic.identity!r} names owner slot "
                    f"{semantic.owner_slot!r} but is stored in {slot.slot!r}",
                    identities=(semantic.identity,),
                )
            position, leaf = by_path[path]
            index[semantic.identity] = _LocatedLeaf(semantic, leaf, position)
        flattened[slot.slot] = (treedef, leaves)
    unknown_slots = sorted(set(values) - {slot.slot for slot in structure.slots})
    if unknown_slots:
        raise CheckpointTransformRefused(
            CheckpointTransformErrorCode.INCOMPLETE_ACCOUNTING,
            f"checkpoint values contain undeclared slots {unknown_slots!r}",
        )
    return index, flattened


def _dtype(value: Any) -> np.dtype[Any] | None:
    dtype = getattr(value, "dtype", None)
    return None if dtype is None else np.dtype(dtype)


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    return None if shape is None else tuple(int(dimension) for dimension in shape)


def _extension_rule(
    leaf: CheckpointSemanticLeafIdentity,
    rules: Sequence[AddedParameterAxis],
) -> AddedParameterAxis | None:
    matches = [
        rule
        for rule in rules
        if rule.component_slot == leaf.component_slot and rule.parameter_role == leaf.parameter_role
    ]
    if len(matches) > 1:
        raise CheckpointTransformRefused(
            CheckpointTransformErrorCode.AMBIGUOUS_IDENTITY,
            "more than one added-axis rule names the same semantic parameter owner",
            identities=(leaf.identity,),
        )
    return matches[0] if matches else None


def _owner_projection(leaf: CheckpointSemanticLeafIdentity) -> tuple[Any, ...]:
    return (
        leaf.owner_slot,
        leaf.component_slot,
        leaf.parameter_role,
        leaf.optimizer_binding,
        leaf.optimizer_role,
        leaf.required,
    )


def _extend_from_target(
    source: Any,
    target: Any,
    *,
    axis: int,
    identity: str,
) -> tuple[Any, int, int, int]:
    source_shape = _shape(source)
    target_shape = _shape(target)
    if source_shape is None or target_shape is None or len(source_shape) != len(target_shape):
        raise CheckpointTransformRefused(
            CheckpointTransformErrorCode.UNSUPPORTED_SHAPE_CHANGE,
            f"semantic leaf {identity!r} is not an equal-rank array extension",
            identities=(identity,),
        )
    normalized_axis = axis if axis >= 0 else len(source_shape) + axis
    if normalized_axis < 0 or normalized_axis >= len(source_shape):
        raise CheckpointTransformRefused(
            CheckpointTransformErrorCode.UNSUPPORTED_SHAPE_CHANGE,
            f"semantic leaf {identity!r} declares invalid extension axis {axis}",
            identities=(identity,),
        )
    for dimension, (source_size, target_size) in enumerate(zip(source_shape, target_shape)):
        if dimension == normalized_axis:
            if target_size <= source_size:
                raise CheckpointTransformRefused(
                    CheckpointTransformErrorCode.UNSUPPORTED_SHAPE_CHANGE,
                    f"semantic leaf {identity!r} target axis must strictly extend the source",
                    identities=(identity,),
                )
        elif source_size != target_size:
            raise CheckpointTransformRefused(
                CheckpointTransformErrorCode.UNSUPPORTED_SHAPE_CHANGE,
                f"semantic leaf {identity!r} changes more than its declared axis",
                identities=(identity,),
            )
    slices = [slice(None)] * len(source_shape)
    slices[normalized_axis] = slice(0, source_shape[normalized_axis])
    try:
        transformed = target.at[tuple(slices)].set(source)
    except AttributeError:
        transformed = np.array(target, copy=True)
        transformed[tuple(slices)] = source
    return (
        transformed,
        normalized_axis,
        source_shape[normalized_axis],
        target_shape[normalized_axis],
    )


def derive_and_apply_semantic_checkpoint_transform(
    *,
    source_structure: CheckpointStructure,
    target_structure: CheckpointStructure,
    source_values: Mapping[str, Any],
    target_values: Mapping[str, Any],
    added_axes: Sequence[AddedParameterAxis],
    optimizer_policy: Literal["preserve", "reset"],
    learning_rate_schedule_policy: Literal["continue", "restart"],
) -> SemanticCheckpointTransformResult:
    """Derive and execute an exhaustive transform from canonical ownership.

    ``target_values`` are the target constructor's authoritative initialization.
    Added regions retain those values; this function never synthesizes a model.
    """
    if optimizer_policy not in {"preserve", "reset"}:
        raise CheckpointTransformRefused(
            CheckpointTransformErrorCode.INVALID_POLICY,
            f"unsupported optimizer policy {optimizer_policy!r}",
        )
    if learning_rate_schedule_policy not in {"continue", "restart"}:
        raise CheckpointTransformRefused(
            CheckpointTransformErrorCode.INVALID_POLICY,
            "learning-rate schedule policy must be explicitly 'continue' or 'restart'",
        )
    source, _ = _semantic_index(source_structure, source_values)
    target, target_flattened = _semantic_index(target_structure, target_values)
    if set(source) != set(target):
        missing_target = sorted(set(source) - set(target))
        missing_source = sorted(set(target) - set(source))
        identities = tuple((*missing_target, *missing_source))
        raise CheckpointTransformRefused(
            CheckpointTransformErrorCode.UNKNOWN_IDENTITY,
            "source and target semantic identity sets differ; "
            f"source_only={missing_target!r} target_only={missing_source!r}",
            identities=identities,
        )

    accounting: list[CheckpointLeafTransfer] = []
    used_rules: set[tuple[str, str, int]] = set()
    used_model_rules: set[tuple[str, str, int]] = set()
    for identity in sorted(target):
        source_leaf = source[identity]
        target_leaf = target[identity]
        if _owner_projection(source_leaf.semantic) != _owner_projection(target_leaf.semantic):
            raise CheckpointTransformRefused(
                CheckpointTransformErrorCode.AMBIGUOUS_IDENTITY,
                f"semantic leaf {identity!r} changes its declared owner",
                identities=(identity,),
            )
        source_dtype = _dtype(source_leaf.value)
        target_dtype = _dtype(target_leaf.value)
        if source_dtype != target_dtype:
            raise CheckpointTransformRefused(
                CheckpointTransformErrorCode.DTYPE_MISMATCH,
                f"semantic leaf {identity!r} changes dtype from {source_dtype} to {target_dtype}",
                identities=(identity,),
            )
        slot = target_leaf.semantic.owner_slot
        treedef, target_leaves = target_flattened[slot]
        if target_leaf.semantic.optimizer_binding is not None and optimizer_policy == "reset":
            accounting.append(
                CheckpointLeafTransfer(
                    identity=identity,
                    slot=slot,
                    source_path=source_leaf.semantic.raw_path,
                    target_path=target_leaf.semantic.raw_path,
                    disposition="deliberately_reset",
                    copied_shape=_shape(target_leaf.value),
                )
            )
            continue

        rule = _extension_rule(target_leaf.semantic, added_axes)
        if _shape(source_leaf.value) == _shape(target_leaf.value):
            target_leaves[target_leaf.index] = source_leaf.value
            accounting.append(
                CheckpointLeafTransfer(
                    identity=identity,
                    slot=slot,
                    source_path=source_leaf.semantic.raw_path,
                    target_path=target_leaf.semantic.raw_path,
                    disposition="copied",
                    copied_shape=_shape(source_leaf.value),
                )
            )
        elif rule is not None:
            transformed, axis, start, stop = _extend_from_target(
                source_leaf.value,
                target_leaf.value,
                axis=rule.axis,
                identity=identity,
            )
            target_leaves[target_leaf.index] = transformed
            rule_key = (rule.component_slot, rule.parameter_role, rule.axis)
            used_rules.add(rule_key)
            if target_leaf.semantic.optimizer_binding is None:
                used_model_rules.add(rule_key)
            accounting.append(
                CheckpointLeafTransfer(
                    identity=identity,
                    slot=slot,
                    source_path=source_leaf.semantic.raw_path,
                    target_path=target_leaf.semantic.raw_path,
                    disposition="extended",
                    copied_shape=_shape(source_leaf.value),
                    initialized_axis=axis,
                    initialized_start=start,
                    initialized_stop=stop,
                )
            )
        else:
            raise CheckpointTransformRefused(
                CheckpointTransformErrorCode.UNSUPPORTED_SHAPE_CHANGE,
                f"semantic leaf {identity!r} changes shape without one declared added axis",
                identities=(identity,),
            )
        target_flattened[slot] = (treedef, target_leaves)

    declared_rules = {(rule.component_slot, rule.parameter_role, rule.axis) for rule in added_axes}
    unused_rules = declared_rules - used_rules
    if unused_rules:
        raise CheckpointTransformRefused(
            CheckpointTransformErrorCode.INCOMPLETE_ACCOUNTING,
            f"added-axis rules did not transform any semantic leaf: {sorted(unused_rules)!r}",
        )
    optimizer_only_rules = declared_rules - used_model_rules
    if optimizer_only_rules:
        raise CheckpointTransformRefused(
            CheckpointTransformErrorCode.INCOMPLETE_ACCOUNTING,
            "added-axis rules must transform exactly their canonical model owner; "
            f"missing_model_rules={sorted(optimizer_only_rules)!r}",
        )
    if len(accounting) != len(source) or {item.identity for item in accounting} != set(source):
        raise CheckpointTransformRefused(
            CheckpointTransformErrorCode.INCOMPLETE_ACCOUNTING,
            "checkpoint transform did not account for every semantic identity",
        )
    values = {
        slot: jt.unflatten(treedef, leaves) for slot, (treedef, leaves) in target_flattened.items()
    }
    return SemanticCheckpointTransformResult(
        values=values,
        accounting=tuple(accounting),
        optimizer_policy=optimizer_policy,
        learning_rate_schedule_policy=learning_rate_schedule_policy,
    )


__all__ = [
    "AddedParameterAxis",
    "CheckpointLeafTransfer",
    "CheckpointTransformErrorCode",
    "CheckpointTransformRefused",
    "SemanticCheckpointTransformResult",
    "derive_and_apply_semantic_checkpoint_transform",
]
