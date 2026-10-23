"""Small public validators shared by Feedbax contract consumers."""

from __future__ import annotations

import re
from typing import TypeGuard


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def is_sha256_digest(value: object) -> TypeGuard[str]:
    """Return whether ``value`` is a lowercase 64-hex SHA-256 digest."""
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def validate_sha256(value: object, *, field_name: str) -> str:
    """Return a valid SHA-256 digest or raise a field-specific error."""
    if not is_sha256_digest(value):
        raise ValueError(f"{field_name} must be a lowercase 64-hex sha256 digest")
    return value
