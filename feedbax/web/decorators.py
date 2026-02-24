from __future__ import annotations
from typing import Any, Callable, Dict


def register_component(**meta: Any) -> Callable[[type], type]:
    """Attach component metadata to a class for discovery by the web registry."""

    def decorator(cls: type) -> type:
        existing: Dict[str, Any] = getattr(cls, '_feedbax_component_meta', {})
        merged = {**existing, **meta}
        setattr(cls, '_feedbax_component_meta', merged)
        return cls

    return decorator
