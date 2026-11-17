"""Collect the first entry to each code object without loading Feedbax."""

from collections import defaultdict
import sys
from types import CodeType


class SourceRoster:
    def __init__(self, identifier: int, label: str):
        self.identifier = identifier
        self.label = label
        self.names: dict[str, set[str]] = defaultdict(set)

    def _entered(self, code: CodeType, _offset: int):
        self.names[code.co_filename].add(code.co_qualname.replace(".<locals>.", "."))
        return sys.monitoring.DISABLE

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
