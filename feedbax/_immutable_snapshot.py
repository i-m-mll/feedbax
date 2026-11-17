"""Immutable copies of authenticated records and their containers."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict


_READ_ONLY_MODEL_TYPES: dict[type[BaseModel], type[BaseModel]] = {}


def _reject_snapshot_mutation(*_args: Any, **_kwargs: Any) -> None:
    raise TypeError("verified authenticated snapshots are immutable")


class _FrozenDict(dict[Any, Any]):
    """A serialization-compatible immutable dictionary snapshot."""

    __setitem__ = _reject_snapshot_mutation
    __delitem__ = _reject_snapshot_mutation
    clear = _reject_snapshot_mutation
    pop = _reject_snapshot_mutation
    popitem = _reject_snapshot_mutation
    setdefault = _reject_snapshot_mutation
    update = _reject_snapshot_mutation
    __ior__ = _reject_snapshot_mutation

    def __copy__(self) -> "_FrozenDict":
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[Any, Any]:
        copied: dict[Any, Any] = {}
        memo[id(self)] = copied
        copied.update(
            (copy.deepcopy(key, memo), copy.deepcopy(value, memo)) for key, value in self.items()
        )
        return copied


class _FrozenList(list[Any]):
    """A serialization-compatible immutable list snapshot."""

    __setitem__ = _reject_snapshot_mutation
    __delitem__ = _reject_snapshot_mutation
    append = _reject_snapshot_mutation
    clear = _reject_snapshot_mutation
    extend = _reject_snapshot_mutation
    insert = _reject_snapshot_mutation
    pop = _reject_snapshot_mutation
    remove = _reject_snapshot_mutation
    reverse = _reject_snapshot_mutation
    sort = _reject_snapshot_mutation
    __iadd__ = _reject_snapshot_mutation
    __imul__ = _reject_snapshot_mutation

    def __copy__(self) -> "_FrozenList":
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> list[Any]:
        copied: list[Any] = []
        memo[id(self)] = copied
        copied.extend(copy.deepcopy(item, memo) for item in self)
        return copied


def _immutable_model_snapshot(value: BaseModel) -> Any:
    """Return a recursively immutable, typed snapshot of one verified record."""
    model_type = type(value)
    read_only_type = _READ_ONLY_MODEL_TYPES.get(model_type)
    if read_only_type is None:
        config_values = dict(model_type.model_config)
        config_values["frozen"] = True
        config = ConfigDict(**config_values)

        def snapshot_eq(self: BaseModel, other: Any) -> bool:
            return isinstance(other, model_type) and _snapshot_comparison_value(
                self
            ) == _snapshot_comparison_value(other)

        def snapshot_deepcopy(
            self: BaseModel,
            memo: dict[int, Any] | None = None,
        ) -> BaseModel:
            memo = {} if memo is None else memo
            existing = memo.get(id(self))
            if existing is not None:
                return existing
            payload = copy.deepcopy(
                self.model_dump(mode="python", round_trip=True),
                memo,
            )
            copied = model_type.model_validate(payload)
            memo[id(self)] = copied
            return copied

        read_only_type = type(
            f"_ReadOnly{model_type.__name__}",
            (model_type,),
            {
                "model_config": config,
                "__module__": __name__,
                "__eq__": snapshot_eq,
                "__deepcopy__": snapshot_deepcopy,
            },
        )
        _READ_ONLY_MODEL_TYPES[model_type] = read_only_type
    snapshot = read_only_type.model_validate(value.model_dump(mode="python", round_trip=True))
    for field_name in type(snapshot).model_fields:
        object.__setattr__(
            snapshot,
            field_name,
            _immutable_snapshot_value(getattr(snapshot, field_name)),
        )
    return snapshot


def _immutable_snapshot_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _immutable_model_snapshot(value)
    if isinstance(value, Mapping):
        return _FrozenDict(
            (
                _immutable_snapshot_value(key),
                _immutable_snapshot_value(item),
            )
            for key, item in value.items()
        )
    if isinstance(value, list):
        return _FrozenList(_immutable_snapshot_value(item) for item in value)
    if isinstance(value, tuple):
        items = tuple(_immutable_snapshot_value(item) for item in value)
        if hasattr(value, "_fields"):
            return type(value)(*items)
        return items
    if isinstance(value, (set, frozenset)):
        return frozenset(_immutable_snapshot_value(item) for item in value)
    return value


def _snapshot_comparison_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return {
            field_name: _snapshot_comparison_value(getattr(value, field_name))
            for field_name in type(value).model_fields
        }
    if isinstance(value, Mapping):
        return {
            _snapshot_comparison_value(key): _snapshot_comparison_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return tuple(_snapshot_comparison_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_snapshot_comparison_value(item) for item in value)
    return value
