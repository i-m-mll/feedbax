"""Resolve a root input's authored signal selection."""

from collections.abc import Mapping


def selected_signal_id(
    input_id: str, input_name: str, signals_by_name: Mapping[str, str],
    bindings: Mapping[str, str | None],
) -> str | None:
    """Respect explicit selections, then match by ID or by a shared display name."""
    if input_id in bindings:
        return bindings[input_id]
    if input_id in signals_by_name.values():
        return input_id
    return signals_by_name.get(input_name)
