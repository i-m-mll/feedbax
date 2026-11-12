"""Authored value origins carried through static project resolution."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.matrix_core import apply_override_patch_in_place

type SourceKind = Literal["document", "step", "sweep_value", "exception", "variant", "upstream"]
type Path = tuple[str, ...]


def pointer(path: Path) -> str:
    return "".join("/" + part.replace("~", "~0").replace("/", "~1") for part in path)


@dataclass(frozen=True)
class ValueSource:
    kind: SourceKind
    name: str
    path: str = ""
    step: str | None = None
    axis: str | None = None
    value: str | None = None
    variant: str | None = None
    row: str | None = None

    def __deepcopy__(self, _memo: dict[int, Any]) -> ValueSource:
        return self

    def child(self, part: str) -> ValueSource:
        return replace(self, path=self.path + pointer((part,)))


class SourceObject(dict[str, Any]):
    def __init__(self, source: ValueSource | None, children: Mapping[str, Any]):
        super().__init__(children)
        self.source = source


class SourceList(list[Any]):
    def __init__(self, source: ValueSource | None, children: Iterator[Any]):
        super().__init__(children)
        self.source = source


def source_shape(value: Any, source: ValueSource | None, override: Any = None) -> Any:
    """Mirror the value's containers so the patch engine also moves their origins."""
    if isinstance(override, ValueSource):
        source, override = override, None
    if isinstance(value, dict):
        return SourceObject(
            getattr(override, "source", None) or source,
            {
                key: source_shape(
                    item, source.child(key) if source else None, source_at(override, (key,))
                )
                for key, item in value.items()
            },
        )
    if isinstance(value, list):
        return SourceList(
            getattr(override, "source", None) or source,
            (
                source_shape(
                    item, source.child(str(i)) if source else None, source_at(override, (str(i),))
                )
                for i, item in enumerate(value)
            ),
        )
    return getattr(override, "source", None) or source


def source_at(shape: Any, path: Path) -> Any:
    for part in path:
        if isinstance(shape, dict):
            shape = shape.get(part)
        elif isinstance(shape, list) and part.isdigit() and int(part) < len(shape):
            shape = shape[int(part)]
        else:
            return None
    return shape


def first_source(shape: Any) -> ValueSource | None:
    if isinstance(shape, ValueSource):
        return shape
    if isinstance(shape, (SourceObject, SourceList)):
        if shape.source is not None:
            return shape.source
        children = shape.values() if isinstance(shape, dict) else shape
        return next((origin for child in children if (origin := first_source(child))), None)
    return None


def patch_source(shape: dict[str, Any], patch: OverridePatch, source: Any = None) -> None:
    if patch.op != "remove":
        patch = patch.model_copy(update={"value": source})
    apply_override_patch_in_place(shape, patch)


def source_leaves(shape: Any, path: Path = ()) -> Iterator[tuple[str, ValueSource]]:
    if isinstance(shape, dict) and shape:
        for key, value in shape.items():
            yield from source_leaves(value, (*path, key))
    elif isinstance(shape, list) and shape:
        for i, value in enumerate(shape):
            yield from source_leaves(value, (*path, str(i)))
    elif (source := first_source(shape)) is not None:
        yield pointer(path), source
