"""Routing for immutable component parameter migration edges."""

from collections.abc import Callable, Mapping
from typing import Any

from feedbax.contracts.migrations import ComponentMigration, UnsupportedComponentMigration

Endpoint = tuple[str, str | None]
Routes = Mapping[Endpoint, ComponentMigration]


def migration_route(
    routes: Routes,
    current_version: Callable[[str], str | None],
    source: Endpoint,
) -> tuple[Endpoint, tuple[ComponentMigration, ...]]:
    chain: list[ComponentMigration] = []
    visited: set[Endpoint] = set()
    endpoint = source
    while endpoint in routes:
        if endpoint in visited:
            raise UnsupportedComponentMigration(f"Component migration cycle at {endpoint!r}")
        visited.add(endpoint)
        type_id, version = endpoint
        if version is not None and version == current_version(type_id):
            raise UnsupportedComponentMigration(
                f"Component migration starts at the current version: {endpoint!r}"
            )
        edge = routes[endpoint]
        chain.append(edge)
        endpoint = (edge.target_type, edge.target_param_schema_version)
    return endpoint, tuple(chain)


def validate_migration_routes(
    routes: Routes,
    current_versions: Mapping[str, str],
    *,
    require_current_terminal: bool = False,
) -> None:
    migration_ids: set[str] = set()
    for endpoint, edge in routes.items():
        if edge.migration_id in migration_ids:
            raise UnsupportedComponentMigration(
                f"Duplicate component migration ID: {edge.migration_id!r}"
            )
        migration_ids.add(edge.migration_id)
        (type_id, version), _ = migration_route(routes, current_versions.get, endpoint)
        if require_current_terminal and (
            version is None or current_versions.get(type_id) != version
        ):
            raise UnsupportedComponentMigration(
                f"Component migration {edge.migration_id!r} terminates at "
                f"{(type_id, version)!r} without a current declaration"
            )


def source_endpoint(
    routes: Routes,
    current_version: Callable[[str], str | None],
    type_id: str,
    version: str | None,
    params: Mapping[str, Any],
    *,
    historical_unversioned: bool,
) -> Endpoint:
    if version is not None:
        return type_id, version
    legacy = [
        endpoint
        for endpoint, edge in routes.items()
        if endpoint[0] == type_id
        and edge.matches_unversioned is not None
        and edge.matches_unversioned(params)
    ]
    unversioned_rename = routes.get((type_id, None))
    if (
        unversioned_rename is not None
        and unversioned_rename.matches_unversioned is None
        and current_version(type_id) is None
    ):
        legacy.append((type_id, None))
    if len(legacy) == 1:
        return legacy[0]
    if len(legacy) > 1 or (
        historical_unversioned
        and not legacy
        and any(source_type == type_id for source_type, _ in routes)
    ):
        raise UnsupportedComponentMigration(
            f"Ambiguous unversioned component parameters for {type_id!r}; "
            "supply the original parameter schema version"
        )
    return type_id, current_version(type_id)
