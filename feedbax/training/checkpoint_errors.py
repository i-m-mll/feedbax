"""Shared error boundary for checkpoint storage and slot codecs."""


class CheckpointCustodyError(ValueError):
    """Base class for checkpoint custody failures."""
