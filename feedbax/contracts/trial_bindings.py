"""Resolve a root input's authored signal selection."""

from collections.abc import Collection, Mapping


def selected_signal_id(
    input_id: str, signal_keys: Collection[str], bindings: Mapping[str, str | None],
) -> str | None:
    """Use an explicit selection or the Task signal with the Model input's key."""
    if input_id in bindings:
        return bindings[input_id]
    return input_id if input_id in signal_keys else None
