"""Small naming helpers for the scientific runtime."""

from collections.abc import Sequence, Set


def get_unique_label(label: str, invalid_labels: Sequence[str] | Set[str]) -> str:
    """Return a label made unique by appending the first available integer."""
    index = 0
    candidate = label
    while candidate in invalid_labels:
        candidate = f"{label}_{index}"
        index += 1
    return candidate


__all__ = ["get_unique_label"]
