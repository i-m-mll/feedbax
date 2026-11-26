"""Authored value origins carried through static project resolution."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.matrix_core import MirrorList, replay_stored_patch_copy

SourceKind = Literal["document", "step", "sweep_value", "exception", "variant", "upstream"]
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


class SourceList(MirrorList):
    """The origins of a list's members; ``mirrored`` is the list, whose ids name them."""

    def __init__(self, source: ValueSource | None, children: Iterator[Any], mirrored: list[Any]):
        super().__init__(children, mirrored)
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
            value,
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


def changed_sources(before: Any, after: Any, shape: Any, source: ValueSource) -> Any:
    """Give changed leaves their new author while retaining the other origins."""
    if before == after:
        return shape
    if isinstance(before, dict) and isinstance(after, dict):
        return SourceObject(
            getattr(shape, "source", None),
            {
                key: changed_sources(before[key], value, source_at(shape, (key,)), source)
                if key in before else source_shape(value, source)
                for key, value in after.items()
            },
        )
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return SourceList(
            getattr(shape, "source", None),
            (
                changed_sources(prior, value, source_at(shape, (str(index),)), source)
                for index, (prior, value) in enumerate(zip(before, after, strict=True))
            ),
            after,
        )
    return source_shape(after, source)


def place_source(shape: Any, parts: tuple[str, ...], source: Any) -> Any:
    """Copy ``shape`` with ``source`` at ``parts``, or with nothing there when ``source`` is None."""
    head, *rest = parts
    children = dict(shape) if isinstance(shape, dict) else {}
    if rest:
        children[head] = place_source(children.get(head), tuple(rest), source)
    elif source is None:
        children.pop(head, None)
    else:
        children[head] = source
    return SourceObject(shape.source if isinstance(shape, SourceObject) else None, children)


def first_source(shape: Any) -> ValueSource | None:
    if isinstance(shape, ValueSource):
        return shape
    if isinstance(shape, (SourceObject, SourceList)):
        if shape.source is not None:
            return shape.source
        children = shape.values() if isinstance(shape, dict) else shape
        return next((origin for child in children if (origin := first_source(child))), None)
    return None


def patch_source(
    shape: dict[str, Any], patch: OverridePatch, source: Any = None
) -> dict[str, Any]:
    """Replay a payload patch on its parallel tree of origins."""
    if patch.op != "remove":
        patch = patch.model_copy(update={"value": source})
    return replay_stored_patch_copy(shape, patch)
