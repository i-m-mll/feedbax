"""Small configuration helpers."""

from collections.abc import Mapping, Sequence, Set
from copy import deepcopy
from typing import Any


def deep_merge(
    base: Mapping[str, Any],
    over: Mapping[str, Any],
    ignore_none: bool = True,
) -> dict[str, Any]:
    """Return ``base`` recursively updated by ``over``.

    By default, ``None`` in ``over`` means "leave the existing value alone".
    Pass ``ignore_none=False`` when ``None`` is an intentional replacement.
    """
    out: dict[str, Any] = dict(base)
    for k, v in over.items():
        if v is None and ignore_none:
            continue
        bv = out.get(k)
        if isinstance(v, Mapping) and isinstance(bv, Mapping):
            out[k] = deep_merge(bv, v, ignore_none=ignore_none)
        else:
            out[k] = v
    return out


def copy_delattr(obj: Any, *attr_names: str):
    """Return a deep copy of an object, with some attributes removed."""
    obj = deepcopy(obj)
    for attr_name in attr_names:
        delattr(obj, attr_name)
    return obj


def get_unique_label(label: str, invalid_labels: Sequence[str] | Set[str]) -> str:
    """Get a unique label by appending integers until no collision remains."""
    i = 0
    label_ = label
    while label_ in invalid_labels:
        label_ = f"{label}_{i}"
        i += 1
    return label_
