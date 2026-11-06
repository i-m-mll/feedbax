"""Errors shared between checkpoint readers and serialized model types."""


class _LegacyWrapperLayoutError(ValueError):
    """Pickled wrapper state predates layer-owned trainable parameters."""
