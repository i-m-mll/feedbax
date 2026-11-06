import pytest

from feedbax.plugins.application import ApplicationRegistryBundle, new_application_registry_bundle


@pytest.fixture(name="application_registry_bundle")
def registry_bundle_fixture() -> ApplicationRegistryBundle:
    """Return fresh caller-owned registries for explicit lifecycle tests."""
    return new_application_registry_bundle(local_component_source=None)
