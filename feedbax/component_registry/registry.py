from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib.metadata
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, cast

from feedbax.contracts.component import (
    ComponentDefinition,
    ComponentIdentity,
    ComponentMigrationInfo,
    PortType,
    PortTypeSpec,
)
from feedbax.contracts.graph import ParamSchema
from feedbax.contracts.migrations import (
    ComponentMigration,
    ComponentMigrationPack,
    MissingComponentOwner,
    UnsupportedComponentMigration,
)

from .builtins import register_builtin_components
from .meta import ComponentBuilder, ComponentMeta, OutputPrototypeFn


logger = logging.getLogger(__name__)
_DEFAULT_REGISTRY: ComponentRegistry | None = None
_REGISTRATION_PROVENANCE: list[str] = []


@dataclass(frozen=True)
class ComponentResolution:
    type_id: str
    params: dict[str, Any]
    param_schema_version: str | None
    meta: ComponentMeta
    migrations: tuple[ComponentMigrationInfo, ...] = ()


@contextmanager
def _registration_provenance(provenance: str | None) -> Iterator[None]:
    if provenance is None:
        yield
        return
    _REGISTRATION_PROVENANCE.append(provenance)
    try:
        yield
    finally:
        _REGISTRATION_PROVENANCE.pop()


def _current_provenance() -> str | None:
    return _REGISTRATION_PROVENANCE[-1] if _REGISTRATION_PROVENANCE else None


class ComponentRegistry:
    def __init__(
        self,
        *,
        load_user_components: bool = True,
        discover_plugins: bool = True,
    ) -> None:
        self._components: Dict[str, ComponentMeta] = {}
        self._component_migrations: Dict[tuple[str, str | None], ComponentMigration] = {}
        self._migration_owners: Dict[str, set[str]] = {}
        self._register_builtins()
        if discover_plugins:
            self.discover_entry_point_components()
        if load_user_components:
            self.load_user_components(Path.home() / ".feedbax" / "components")

    def _register_builtins(self) -> None:
        with _registration_provenance("feedbax"):
            register_builtin_components(self)
        from feedbax.serialization_builders import register_builtin_component_builders

        register_builtin_component_builders(self)

    def register(self, meta: ComponentMeta) -> None:
        if meta.provenance is None:
            meta.provenance = _current_provenance()
        self._complete_identity(meta)
        meta.migrations = self._migration_infos_for_target(meta.name)
        self._components[meta.name] = meta

    def register_component_type(
        self,
        name: str,
        builder: ComponentBuilder,
        *,
        category: str = "Custom",
        description: str = "",
        param_schema: Iterable[ParamSchema | dict[str, Any]] = (),
        input_ports: Iterable[str] = (),
        output_ports: Iterable[str] = (),
        icon: str = "box",
        port_types: PortTypeSpec | dict[str, Any] | None = None,
        is_composite: bool = False,
        template_graph: Any = None,
        template_ui_state: Any = None,
        template_id: str | None = None,
        template_kind: str | None = None,
        output_prototype_fn: OutputPrototypeFn | None = None,
        provenance: str | None = None,
        owner: str | None = None,
        param_schema_version: str = "1",
        supported_param_schema_versions: Iterable[str] | None = None,
    ) -> ComponentMeta:
        if not callable(builder):
            raise TypeError(f"Builder for component type {name!r} must be callable")
        if port_types is not None and not isinstance(port_types, PortTypeSpec):
            port_types = PortTypeSpec.model_validate(port_types)
        meta = ComponentMeta(
            name=name,
            category=category,
            description=description,
            param_schema=[
                schema if isinstance(schema, ParamSchema) else ParamSchema(**schema)
                for schema in param_schema
            ],
            input_ports=list(input_ports),
            output_ports=list(output_ports),
            icon=icon,
            port_types=port_types,
            is_composite=is_composite,
            template_graph=template_graph,
            template_ui_state=template_ui_state,
            template_id=template_id,
            template_kind=template_kind,
            builder=builder,
            output_prototype_fn=output_prototype_fn,
            provenance=provenance or _current_provenance(),
            owner=owner,
            param_schema_version=param_schema_version,
            supported_param_schema_versions=(
                list(supported_param_schema_versions)
                if supported_param_schema_versions is not None
                else [param_schema_version]
            ),
        )
        self.register(meta)
        return meta

    def register_builder(
        self,
        name: str,
        builder: ComponentBuilder,
        *,
        provenance: str | None = None,
        owner: str | None = None,
    ) -> ComponentMeta:
        if not callable(builder):
            raise TypeError(f"Builder for component type {name!r} must be callable")
        meta = self._components.get(name)
        if meta is None:
            meta = ComponentMeta(
                name=name,
                category="Custom",
                description="",
                param_schema=[],
                input_ports=[],
                output_ports=[],
                builder=builder,
                provenance=provenance or _current_provenance(),
                owner=owner,
            )
        else:
            meta.builder = builder
            if meta.provenance is None:
                meta.provenance = provenance or _current_provenance()
            if meta.owner is None:
                meta.owner = owner
        self.register(meta)
        return meta

    def register_migration(self, migration: ComponentMigration) -> None:
        if not migration.owner:
            raise ValueError("Component migration owner must be non-empty")
        key = (migration.source_type, migration.source_param_schema_version)
        if key in self._component_migrations:
            existing = self._component_migrations[key]
            raise ValueError(
                "Component migration already registered: "
                f"{existing.source_type!r} {existing.source_param_schema_version!r}"
            )
        self._component_migrations[key] = migration
        self._migration_owners.setdefault(migration.owner, set()).add(migration.source_type)
        target_meta = self._components.get(migration.target_type)
        if target_meta is not None:
            target_meta.migrations = self._migration_infos_for_target(target_meta.name)

    def register_migration_pack(self, pack: ComponentMigrationPack) -> None:
        if not pack.owner:
            raise ValueError("Component migration pack owner must be non-empty")
        for migration in pack.migrations:
            if migration.owner != pack.owner:
                raise ValueError(
                    "Component migration pack owner mismatch: "
                    f"pack={pack.owner!r}, migration={migration.owner!r}"
                )
            self.register_migration(migration)

    def resolve_component_spec(
        self,
        type_id: str,
        params: Mapping[str, Any],
        *,
        param_schema_version: str | None = None,
    ) -> ComponentResolution:
        meta = self._components.get(type_id)
        if meta is not None and self._accepts_param_schema(meta, param_schema_version):
            return ComponentResolution(
                type_id=type_id,
                params=dict(params),
                param_schema_version=param_schema_version or meta.param_schema_version,
                meta=meta,
            )

        migration = self._component_migrations.get((type_id, param_schema_version))
        if migration is None:
            owner = self._owner_for_missing_type(type_id, meta)
            if owner is not None and not self._has_owner(owner):
                raise MissingComponentOwner(
                    "Component type requires an owner migration pack that is not loaded: "
                    f"type={type_id!r}, owner={owner!r}, "
                    "install or enable the owning Feedbax component package"
                )
            owner_context = f", owner={owner!r}" if owner is not None else ""
            current_version = meta.param_schema_version if meta is not None else None
            raise UnsupportedComponentMigration(
                "No component migration registered: "
                f"type={type_id!r}{owner_context}, "
                f"source_param_schema_version={param_schema_version!r}, "
                f"current_param_schema_version={current_version!r}"
            )

        migrated_type, migrated_params, migrated_version = migration.apply(
            params,
            source_param_schema_version=param_schema_version,
        )
        migrated_meta = self._components.get(migrated_type)
        if migrated_meta is None:
            raise MissingComponentOwner(
                "Component migration target is not registered: "
                f"migration={migration.migration_id!r}, source_type={type_id!r}, "
                f"target_type={migrated_type!r}, owner={migration.owner!r}"
            )
        if not self._accepts_param_schema(migrated_meta, migrated_version):
            raise UnsupportedComponentMigration(
                "Component migration produced unsupported parameter schema version: "
                f"migration={migration.migration_id!r}, target_type={migrated_type!r}, "
                f"target_param_schema_version={migrated_version!r}, "
                f"supported={migrated_meta.supported_param_schema_versions!r}"
            )
        return ComponentResolution(
            type_id=migrated_type,
            params=migrated_params,
            param_schema_version=migrated_version or migrated_meta.param_schema_version,
            meta=migrated_meta,
            migrations=(self._migration_info(migration),),
        )

    def should_resolve_component_spec(
        self,
        type_id: str,
        *,
        param_schema_version: str | None = None,
    ) -> bool:
        if type_id in self._components:
            return True
        if (type_id, param_schema_version) in self._component_migrations:
            return True
        if param_schema_version is not None:
            return True
        return "." in type_id

    def get(self, name: str) -> Optional[ComponentMeta]:
        return self._components.get(name)

    def names(self) -> List[str]:
        return sorted(self._components)

    def executable_names(self) -> List[str]:
        return sorted(name for name, meta in self._components.items() if meta.builder is not None)

    def list_all(self) -> List[ComponentDefinition]:
        return [self._to_definition(meta) for meta in self._components.values()]

    def list_by_category(self) -> Dict[str, List[ComponentDefinition]]:
        by_category: Dict[str, List[ComponentDefinition]] = {}
        for meta in self._components.values():
            by_category.setdefault(meta.category, []).append(self._to_definition(meta))
        return by_category

    def load_user_components(self, path: Path) -> None:
        if not path.exists():
            return

        for py_file in path.glob("*.py"):
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for name in dir(module):
                obj = getattr(module, name)
                meta = getattr(obj, "_feedbax_component_meta", None)
                if not isinstance(meta, dict):
                    continue
                builder = meta.get("builder")
                if builder is None and callable(obj):

                    def _build_user_component(
                        params: Mapping[str, Any],
                        component_type: Any = obj,
                    ):
                        return component_type(**dict(params))

                    builder = _build_user_component
                if builder is None:
                    continue
                component_builder = cast(ComponentBuilder, builder)
                port_types = meta.get("port_types")
                self.register_component_type(
                    name=meta.get("name", name),
                    builder=component_builder,
                    category=meta.get("category", "Custom"),
                    description=meta.get("description", ""),
                    param_schema=meta.get("param_schema", []),
                    input_ports=meta.get("input_ports", []),
                    output_ports=meta.get("output_ports", []),
                    icon=meta.get("icon", "box"),
                    is_composite=bool(meta.get("is_composite", False)),
                    port_types=(
                        PortTypeSpec(
                            inputs={
                                key: PortType(**value)
                                for key, value in port_types.get("inputs", {}).items()
                            },
                            outputs={
                                key: PortType(**value)
                                for key, value in port_types.get("outputs", {}).items()
                            },
                        )
                        if isinstance(port_types, dict)
                        else port_types
                    ),
                    output_prototype_fn=meta.get("output_prototype_fn"),
                    provenance=f"file:{py_file}",
                )

    def discover_entry_point_components(
        self,
        entry_point_group: str = "feedbax.plugins",
        entry_points: Iterable[Any] | None = None,
    ) -> None:
        if entry_points is None:
            try:
                entry_points = importlib.metadata.entry_points(group=entry_point_group)
            except TypeError:
                all_entry_points = importlib.metadata.entry_points()
                entry_points_getter = getattr(all_entry_points, "get", None)
                if callable(entry_points_getter):
                    entry_points = cast(
                        Iterable[Any],
                        entry_points_getter(entry_point_group, []),
                    )
                else:
                    entry_points = [
                        entry_point
                        for entry_point in all_entry_points
                        if entry_point.group == entry_point_group
                    ]
        if entry_points is None:
            return

        for entry_point in entry_points:
            provenance = self._entry_point_provenance(entry_point)
            try:
                plugin = entry_point.load()
                registrar = self._component_registrar(plugin)
                if registrar is None:
                    continue
                with _registration_provenance(provenance):
                    registrar(self)
            except Exception as exc:
                logger.warning("Failed to load component entry point %s: %s", provenance, exc)
                continue

    def _component_registrar(self, plugin: Any) -> Any:
        for attr in ("register_feedbax_components", "register_components"):
            registrar = getattr(plugin, attr, None)
            if callable(registrar):
                return registrar
        if not callable(plugin):
            return None
        try:
            signature = inspect.signature(plugin)
        except (TypeError, ValueError):
            return None
        parameter_names = set(signature.parameters)
        if parameter_names & {"component_registry", "components"}:
            return plugin
        return None

    def _entry_point_provenance(self, entry_point: Any) -> str:
        dist = getattr(entry_point, "dist", None)
        if dist is not None:
            metadata = getattr(dist, "metadata", {})
            package_name = metadata.get("Name") if hasattr(metadata, "get") else None
            if package_name:
                return f"package:{package_name}"
        return f"entry-point:{getattr(entry_point, 'name', '<unknown>')}"

    def _complete_identity(self, meta: ComponentMeta) -> None:
        details = _identity_from_provenance(meta.name, meta.provenance)
        if meta.owner is None:
            meta.owner = details.owner
        if meta.identity is None:
            meta.identity = details.model_copy(update={"owner": meta.owner})

    def _to_definition(self, meta: ComponentMeta) -> ComponentDefinition:
        return ComponentDefinition(
            name=meta.name,
            category=meta.category,
            description=meta.description,
            param_schema=meta.param_schema,
            input_ports=meta.input_ports,
            output_ports=meta.output_ports,
            icon=meta.icon,
            default_params=meta.default_params,
            port_types=meta.port_types,
            is_composite=meta.is_composite,
            template_graph=meta.template_graph,
            template_ui_state=meta.template_ui_state,
            template_id=meta.template_id,
            template_kind=meta.template_kind,
            provenance=meta.provenance,
            identity=meta.identity,
            owner=meta.owner,
            param_schema_version=meta.param_schema_version,
            supported_param_schema_versions=list(meta.supported_param_schema_versions),
            migrations=list(meta.migrations),
        )

    def _migration_infos_for_target(self, target_type: str) -> list[ComponentMigrationInfo]:
        return [
            self._migration_info(migration)
            for migration in sorted(
                self._component_migrations.values(),
                key=lambda item: (
                    item.owner,
                    item.source_type,
                    item.source_param_schema_version or "",
                    item.target_type,
                ),
            )
            if migration.target_type == target_type
        ]

    def _migration_info(self, migration: ComponentMigration) -> ComponentMigrationInfo:
        return ComponentMigrationInfo(
            migration_id=migration.migration_id,
            owner=migration.owner,
            source_type=migration.source_type,
            target_type=migration.target_type,
            source_param_schema_version=migration.source_param_schema_version,
            target_param_schema_version=migration.target_param_schema_version,
            description=migration.description,
        )

    def _accepts_param_schema(
        self,
        meta: ComponentMeta,
        param_schema_version: str | None,
    ) -> bool:
        version = param_schema_version or meta.param_schema_version
        return version == meta.param_schema_version

    def _owner_for_missing_type(self, type_id: str, meta: ComponentMeta | None) -> str | None:
        if meta is not None:
            return meta.owner
        if "." in type_id:
            owner, _name = type_id.split(".", 1)
            return owner or None
        return None

    def _has_owner(self, owner: str) -> bool:
        return (
            any(meta.owner == owner for meta in self._components.values())
            or owner in self._migration_owners
        )


def _identity_from_provenance(type_id: str, provenance: str | None) -> ComponentIdentity:
    if provenance == "feedbax":
        return ComponentIdentity(
            type_id=type_id,
            owner="feedbax",
            provenance=provenance,
            provenance_kind="feedbax",
            package="feedbax",
            stable=True,
        )
    if provenance and provenance.startswith("package:"):
        package = provenance.removeprefix("package:")
        return ComponentIdentity(
            type_id=type_id,
            owner=package,
            provenance=provenance,
            provenance_kind="package",
            package=package,
            stable=True,
        )
    if provenance and provenance.startswith("file:"):
        return ComponentIdentity(
            type_id=type_id,
            owner="local",
            provenance=provenance,
            provenance_kind="file",
            stable=False,
        )
    if provenance and provenance.startswith("entry-point:"):
        return ComponentIdentity(
            type_id=type_id,
            provenance=provenance,
            provenance_kind="package",
            stable=False,
        )
    if provenance:
        return ComponentIdentity(
            type_id=type_id,
            provenance=provenance,
            provenance_kind="local",
            stable=False,
        )
    return ComponentIdentity(type_id=type_id)


def get_component_registry() -> ComponentRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
        _DEFAULT_REGISTRY = registry
        registry.discover_entry_point_components()
        registry.load_user_components(Path.home() / ".feedbax" / "components")
    return _DEFAULT_REGISTRY


def register_component_type(
    name: str,
    builder: ComponentBuilder,
    *,
    category: str = "Custom",
    description: str = "",
    param_schema: Iterable[ParamSchema | dict[str, Any]] = (),
    input_ports: Iterable[str] = (),
    output_ports: Iterable[str] = (),
    icon: str = "box",
    port_types: PortTypeSpec | dict[str, Any] | None = None,
    is_composite: bool = False,
    output_prototype_fn: OutputPrototypeFn | None = None,
    provenance: str | None = None,
    owner: str | None = None,
    param_schema_version: str = "1",
    supported_param_schema_versions: Iterable[str] | None = None,
) -> ComponentMeta:
    """Register an executable component type in the process-wide registry."""

    return get_component_registry().register_component_type(
        name=name,
        builder=builder,
        category=category,
        description=description,
        param_schema=param_schema,
        input_ports=input_ports,
        output_ports=output_ports,
        icon=icon,
        port_types=port_types,
        is_composite=is_composite,
        output_prototype_fn=output_prototype_fn,
        provenance=provenance,
        owner=owner,
        param_schema_version=param_schema_version,
        supported_param_schema_versions=supported_param_schema_versions,
    )
