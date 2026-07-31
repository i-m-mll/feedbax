"""Shared typed failures for concrete Feedbax registries."""


class RegistryCollisionError(ValueError):
    """Raised when a registry rejects conflicting canonical authority."""
