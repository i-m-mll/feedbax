"""Structural driver ownership shared by authored and executable causal graphs."""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol


class _Wire(Protocol):
    @property
    def source_node(self) -> str: ...
    @property
    def source_port(self) -> str: ...
    @property
    def target_node(self) -> str: ...
    @property
    def target_port(self) -> str: ...
    @property
    def temporality(self) -> str: ...


@dataclass(frozen=True)
class _Driver:
    label: str
    path: str


class InputDriverConflict(ValueError):
    def __init__(self, target: tuple[str, str], first: _Driver, second: _Driver):
        self.target = target
        self.authored_path = second.path
        self.occupied_by = first.path
        super().__init__(
            f"Input {target[0]}.{target[1]} has multiple drivers: "
            f"{first.label} and {second.label}"
        )


def input_driver_conflicts(
    wires: Iterable[_Wire],
    input_bindings: Mapping[str, tuple[str, str]],
    base_path: str = "",
) -> Iterator[InputDriverConflict]:
    """Report each competing driver against the first owner of its input."""
    owners: dict[tuple[str, str], _Driver] = {}
    for name, (node, port) in input_bindings.items():
        target = (node, port)
        token = name.replace("~", "~0").replace("/", "~1")
        driver = _Driver(f"external input {name!r}", f"{base_path}/input_bindings/{token}")
        first = owners.setdefault(target, driver)
        if first is not driver:
            yield InputDriverConflict(target, first, driver)

    for index, wire in enumerate(wires):
        target = (wire.target_node, wire.target_port)
        driver = _Driver(
            f"{wire.temporality} wire {wire.source_node}.{wire.source_port}",
            f"{base_path}/wires/{index}",
        )
        first = owners.setdefault(target, driver)
        if first is not driver:
            yield InputDriverConflict(target, first, driver)
