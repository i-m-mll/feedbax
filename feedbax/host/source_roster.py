"""Collect the first entry to each code object without loading Feedbax."""

from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
import sys
from types import CodeType


class SourceRoster:
    def __init__(self, identifier: int, label: str):
        self.identifier = identifier
        self.label = label
        self.names: dict[str, set[str]] = defaultdict(set)
        self._observe: Callable[[str, str], bool] | None = None

    def _entered(self, code: CodeType, _offset: int):
        name = code.co_qualname.replace(".<locals>.", ".")
        self.names[code.co_filename].add(name)
        if self._observe is not None and not self._observe(code.co_filename, name):
            return None
        return sys.monitoring.DISABLE

    @contextmanager
    def collect(self, observe: Callable[[str, str], bool]) -> Iterator[None]:
        previous, self._observe = self._observe, observe
        sys.monitoring.restart_events()
        try:
            yield
        finally:
            self._observe = previous
            if previous is not None:
                sys.monitoring.restart_events()

    def __enter__(self):
        sys.monitoring.use_tool_id(self.identifier, self.label)
        sys.monitoring.register_callback(self.identifier, sys.monitoring.events.PY_START, self._entered)
        return self

    def __exit__(self, *exc):
        self.disable()
        sys.monitoring.register_callback(self.identifier, sys.monitoring.events.PY_START, None)
        sys.monitoring.free_tool_id(self.identifier)

    def enable(self) -> None:
        sys.monitoring.set_events(self.identifier, sys.monitoring.events.PY_START)

    def disable(self) -> None:
        sys.monitoring.set_events(self.identifier, 0)
