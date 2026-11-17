"""Register isolated compiler test owners using the component compiler."""

from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.portable import ComponentOwner
from feedbax.plugins.declarations import DistributionSource


def register_owners(registry: ComponentRegistry, package: str, *owners: type[ComponentOwner]) -> None:
    source = DistributionSource(package, package, "1")
    registry.register_declarations((owner, source) for owner in owners)
