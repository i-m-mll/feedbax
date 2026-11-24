"""Resolve a root input's authored signal selection."""

from collections.abc import Mapping


def selected_signal_id(
    input_id: str, input_name: str, signals_by_name: Mapping[str, str],
    bindings: Mapping[str, str | None],
) -> str | None:
    return bindings.get(input_id, signals_by_name.get(input_name))
