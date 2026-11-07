from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from feedbax.contracts.component import (
    ComponentDefinition,
    ComponentIdentity,
    ComponentMigrationInfo,
    DynamicPortLayout,
    derive_dynamic_port_layout,
)
from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.domain import CAUSAL_DOMAIN_ID
from feedbax.contracts.migrations import (
    ComponentMigration,
    MissingComponentOwner,
    UnsupportedComponentMigration,
)
from feedbax.contracts.representation import validate_representation_against_component

from .migration_routes import migration_route, source_endpoint, validate_migration_routes
from .declarations import DeclaredComponent
from .discovery import ComponentPack, compile_owner, discover_modules, source_path


logger = logging.getLogger(__name__)
_LEGACY_INTERIOR_DOMAINS = {
    "Network": CAUSAL_DOMAIN_ID,
}


@dataclass(frozen=True)
class ComponentResolution:
    type_id: str
    params: dict[str, Any]
    param_schema_version: str | None
    meta: DeclaredComponent
    migrations: tuple[ComponentMigrationInfo, ...] = ()


@dataclass(frozen=True)
class TemplateBuilderIssue:
    template_type: str
    node_path: str
    node_type: str
    reason: str
    template_id: str | None = None

    @property
    def summary(self) -> str:
        template_ref = self.template_id or self.template_type
        return f"{template_ref}: {self.node_path} uses {self.node_type!r}, {self.reason}"


class ComponentRegistry:
    def __init__(self) -> None:
        self._sealed = False
        self._components: Dict[str, DeclaredComponent] = {}
        self._component_migrations: Dict[tuple[str, str | None], ComponentMigration] = {}
        self._migration_owners: Dict[str, set[str]] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        from .builtins import builtin_component_packs

        for pack in builtin_component_packs():
            self.ingest(pack, provenance="feedbax")

    discover_modules = staticmethod(discover_modules)

    def ingest(self, pack: ComponentPack, *, provenance: str | None = None) -> None:
        """Compile and validate a complete batch before publishing its entries."""
        self._require_mutable()
        components = dict(self._components)
        migrations = dict(self._component_migrations)
        migration_owners = {owner: set(types) for owner, types in self._migration_owners.items()}
        edges = list(pack.migrations)
        discovered = discover_modules(*pack.modules, namespace=pack.namespace, histories=pack.histories) if pack.modules else None
        batch = tuple({id(item): item for item in (*pack.components, *(discovered.components if discovered else ()))}.values())
        unknown_history = pack.histories.keys() - {item for item in batch if isinstance(item, type)}
        if unknown_history:
            raise ValueError(f"History supplied for undiscovered owners: {unknown_history!r}")
        for definition in batch:
            definition_sources: tuple[Path | None, ...] = ()
            owner_class = None
            if isinstance(definition, type):
                owner_class = definition
                definitions, definition_sources, owner_edges = compile_owner(definition, namespace=pack.namespace, history=pack.histories.get(definition))
                edges.extend(owner_edges)
            elif isinstance(definition, DeclaredComponent):
                definitions = (definition,)
            else:
                definitions = (definition(),)
                definition_sources = (source_path(definition),)
            for declared in definitions:
                meta = deepcopy(declared)
                if meta.provenance is None:
                    origin = provenance
                    if origin is None and owner_class is not None:
                        origin = f"package:{owner_class.__module__.split('.')[0]}"
                    meta = replace(meta, provenance=origin)
                meta = self._complete_identity(meta)
                component_sources = (
                    *definition_sources,
                    *pack.sources,
                    *_declared_source_paths(meta),
                )
                if meta.name in components:
                    existing = components[meta.name]
                    raise ValueError(
                        f"component type already registered: {meta.name!r} "
                        f"({existing.owner!r} from {existing.source_paths!r}; "
                        f"{meta.owner!r} from {component_sources!r})"
                    )
                runtime_type = meta.params.runtime_type
                if runtime_type is not None and meta.alias_of is None:
                    conflict = next(
                        (
                            item
                            for item in components.values()
                            if item.alias_of is None
                            and item.params.runtime_type is runtime_type
                            and (meta.owner_class is not None or item.owner_class is not None)
                        ),
                        None,
                    )
                    if conflict is not None:
                        raise ValueError(
                            f"component runtime type already registered: {runtime_type!r} "
                            f"({conflict.name!r}, {meta.name!r})"
                        )
                self._validate_representation(meta)
                components[meta.name] = replace(meta, source_paths=component_sources)
        changed_targets = set(components) - self._components.keys()
        for migration in edges:
            if not migration.owner:
                raise ValueError("Component migration owner must be non-empty")
            key = (migration.source_type, migration.source_param_schema_version)
            if key in migrations:
                existing = migrations[key]
                raise ValueError(
                    "Component migration already registered: "
                    f"{key!r} ({existing.migration_id!r}, {migration.migration_id!r})"
                )
            migrations[key] = migration
            migration_owners.setdefault(migration.owner, set()).add(migration.source_type)
            changed_targets.add(migration.target_type)
        validate_migration_routes(
            migrations,
            {name: meta.param_schema_version for name, meta in components.items()},
        )
        for target in changed_targets & components.keys():
            components[target] = self._with_migrations(
                components[target], self._migration_infos_for_target(target, migrations.values())
            )
        self._components = components
        self._component_migrations = migrations
        self._migration_owners = migration_owners

    def component_source_paths(self, root: Path) -> dict[str, tuple[str, ...] | None]:
        """Project owner sources inside a checkout; None denotes incomplete coverage."""
        root = root.resolve()
        result: dict[str, tuple[str, ...] | None] = {}
        migration_sources: dict[str, list[Path | None]] = {}
        for edge in self._component_migrations.values():
            if edge.migrate_params is not None:
                origin = source_path(edge.migrate_params)
                if origin is None and getattr(edge.migrate_params, "__module__", None) == "builtins":
                    continue
                for type_id in {edge.source_type, edge.target_type}:
                    migration_sources.setdefault(type_id, []).append(origin)
        for name, meta in sorted(self._components.items()):
            sources = (*meta.source_paths, *migration_sources.get(name, ()))
            if not sources or any(
                path is None or not path.is_absolute() or not path.is_relative_to(root)
                for path in sources
            ):
                result[name] = None
            else:
                result[name] = tuple(
                    sorted(
                        {path.relative_to(root).as_posix() for path in sources if path is not None}
                    )
                )
        return result

    def seal(self) -> None:
        validate_migration_routes(
            self._component_migrations,
            {name: meta.param_schema_version for name, meta in self._components.items()},
            require_current_terminal=True,
        )
        self._sealed = True

    def _require_mutable(self) -> None:
        if self._sealed:
            raise RuntimeError("component registry is sealed")

    def resolve_component_spec(
        self,
        type_id: str,
        params: Mapping[str, Any],
        *,
        param_schema_version: str | None = None,
        complete_params: bool = True,
        historical_unversioned: bool = False,
    ) -> ComponentResolution:
        """Resolve migration edges before optionally completing the current contract."""
        def current_version(name: str) -> str | None:
            component = self._components.get(name)
            return component.param_schema_version if component is not None else None
        source = source_endpoint(
            self._component_migrations,
            current_version,
            type_id,
            param_schema_version,
            params,
            historical_unversioned=historical_unversioned,
        )
        (resolved_type, resolved_version), chain = migration_route(
            self._component_migrations, current_version, source
        )
        meta = self._components.get(resolved_type)
        if meta is None or resolved_version != meta.param_schema_version:
            owner = self._owner_for_missing_type(resolved_type, meta)
            if owner is not None and not self._has_owner(owner):
                raise MissingComponentOwner(
                    "Component type requires an owner migration pack that is not loaded: "
                    f"type={resolved_type!r}, owner={owner!r}, "
                    "install or enable the owning Feedbax component package"
                )
            owner_context = f", owner={owner!r}" if owner is not None else ""
            raise UnsupportedComponentMigration(
                "No component migration registered to a current declaration: "
                f"type={type_id!r}{owner_context}, "
                f"source_param_schema_version={param_schema_version!r}, "
                f"current_param_schema_version={current_version(type_id)!r}, "
                f"terminal={(resolved_type, resolved_version)!r}"
            )
        migrated = dict(params)
        for edge in chain:
            _, migrated, _ = edge.apply(
                migrated, source_param_schema_version=edge.source_param_schema_version
            )
        context = {
            name: value for name, value in migrated.items() if name in meta.params.build_context_fields
        }
        authored = {name: value for name, value in migrated.items() if name not in context}
        effective = (
            meta.params.require(authored).model_dump(mode="python") if complete_params else authored
        )
        return ComponentResolution(
            type_id=resolved_type,
            params={**effective, **context},
            param_schema_version=meta.param_schema_version,
            meta=deepcopy(meta),
            migrations=tuple(self._migration_info(edge) for edge in chain),
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

    def get(self, name: str) -> Optional[DeclaredComponent]:
        meta = self._components.get(name)
        return deepcopy(meta) if meta is not None else None

    def dynamic_port_layout(
        self,
        name: str,
        params: Mapping[str, Any],
    ) -> DynamicPortLayout | None:
        """Derive a component's dynamic ports from an isolated parameter snapshot."""

        meta = self._components.get(name)
        if meta is None or meta.dynamic_port_policy is None:
            return None
        effective_params = meta.params.require(params).model_dump(mode="python")
        return derive_dynamic_port_layout(meta.dynamic_port_policy, effective_params)

    def serialize_component(
        self,
        component: Any,
        *,
        type_id: str | None = None,
        param_schema_version: str | None = None,
    ) -> tuple[str, dict[str, Any], str]:
        """Serialize one runtime leaf through the declaration that owns it."""

        if type_id is not None:
            meta = self._components.get(type_id)
            if meta is None:
                raise ValueError(f"Unknown bound component type {type_id!r}")
            if param_schema_version != meta.param_schema_version:
                raise UnsupportedComponentMigration(
                    "Runtime component binding has unsupported parameter schema version: "
                    f"type={type_id!r}, bound={param_schema_version!r}, "
                    f"current={meta.param_schema_version!r}"
                )
        else:
            matches = [
                candidate
                for candidate in self._components.values()
                if candidate.alias_of is None and type(component) is candidate.params.runtime_type
            ]
            if len(matches) != 1:
                names = sorted(candidate.name for candidate in matches)
                raise ValueError(
                    "Programmatic graph component reverse resolution must match exactly one "
                    f"registered runtime type; runtime={type(component).__name__!r}, "
                    f"matches={names!r}"
                )
            meta = matches[0]
        return meta.name, meta.params.dump(component), meta.param_schema_version

    def component_classification(self) -> dict[str, str]:
        """Return the registry-wide parameter-contract classification."""

        return {
            name: ("subgraph" if meta.is_composite else meta.params.classification)
            for name, meta in sorted(self._components.items())
        }

    def names(self) -> List[str]:
        return sorted(self._components)

    def executable_names(self) -> List[str]:
        return sorted(
            name
            for name, meta in self._components.items()
            if meta.params.classification != "unsupported"
        )

    def list_all(self) -> List[ComponentDefinition]:
        return [self._to_definition(meta) for meta in self._components.values()]

    def list_by_category(self) -> Dict[str, List[ComponentDefinition]]:
        by_category: Dict[str, List[ComponentDefinition]] = {}
        for meta in self._components.values():
            by_category.setdefault(meta.category, []).append(self._to_definition(meta))
        return by_category

    def template_builder_issues(
        self,
        meta: DeclaredComponent | None = None,
    ) -> list[TemplateBuilderIssue]:
        """Return template graph nodes that cannot be materialized as executable graph nodes."""

        metas = [meta] if meta is not None else list(self._components.values())
        issues: list[TemplateBuilderIssue] = []
        for template_meta in metas:
            if template_meta.template_graph is None:
                continue
            issues.extend(
                self._template_graph_builder_issues(
                    template_meta,
                    template_meta.template_graph,
                    node_path="template_graph",
                )
            )
        return issues

    def _template_graph_builder_issues(
        self,
        template_meta: DeclaredComponent,
        graph: Any,
        *,
        node_path: str,
    ) -> list[TemplateBuilderIssue]:
        if isinstance(graph, AcausalGraphSpec):
            from feedbax.compiler.acausal_compiler import (
                compile_acausal_authoring_report,
            )

            report = compile_acausal_authoring_report(
                graph,
                node_path=[template_meta.name],
                component_registry=self,
            )
            return [
                TemplateBuilderIssue(
                    template_type=template_meta.name,
                    template_id=template_meta.template_id,
                    node_path=node_path,
                    node_type="AcausalGraphSpec",
                    reason=diagnostic.message,
                )
                for diagnostic in report.diagnostics
                if diagnostic.severity == "error"
            ]
        issues: list[TemplateBuilderIssue] = []
        subgraphs = dict(getattr(graph, "subgraphs", None) or {})
        nodes = getattr(graph, "nodes", {})
        for node_id, node in nodes.items():
            current_path = f"{node_path}/nodes/{node_id}"
            node_type = str(getattr(node, "type", ""))
            if required_interior_domain(node_type, self) is not None or node_id in subgraphs:
                subgraph = subgraphs.get(node_id)
                if subgraph is None:
                    issues.append(
                        TemplateBuilderIssue(
                            template_type=template_meta.name,
                            template_id=template_meta.template_id,
                            node_path=current_path,
                            node_type=node_type,
                            reason="requires a subgraph but none is registered",
                        )
                    )
                else:
                    issues.extend(
                        self._template_graph_builder_issues(
                            template_meta,
                            subgraph,
                            node_path=f"{current_path}/subgraph",
                        )
                    )
                continue

            node_meta = self._components.get(node_type)
            if node_meta is None:
                issues.append(
                    TemplateBuilderIssue(
                        template_type=template_meta.name,
                        template_id=template_meta.template_id,
                        node_path=current_path,
                        node_type=node_type,
                        reason="has no registered component metadata or builder",
                    )
                )
            elif node_meta.builder is None:
                issues.append(
                    TemplateBuilderIssue(
                        template_type=template_meta.name,
                        template_id=template_meta.template_id,
                        node_path=current_path,
                        node_type=node_type,
                        reason="is registered but has no executable builder",
                    )
                )
        return issues

    def _complete_identity(self, meta: DeclaredComponent) -> DeclaredComponent:
        details = _identity_from_provenance(meta.name, meta.provenance)
        declaration = meta.declaration
        if declaration.owner == "local" and details.owner is not None:
            declaration = replace(
                declaration,
                owner=details.owner,
                schema_id=f"{details.owner}.component.{meta.name}",
            )
        identity = meta.identity or details.model_copy(update={"owner": declaration.owner})
        return replace(meta, declaration=declaration, identity=identity)

    @staticmethod
    def _with_migrations(
        meta: DeclaredComponent,
        migrations: list[ComponentMigrationInfo],
    ) -> DeclaredComponent:
        serialization = replace(meta.serialization, migrations=tuple(migrations))
        return replace(meta, serialization=serialization)

    def _to_definition(self, meta: DeclaredComponent) -> ComponentDefinition:
        meta = deepcopy(meta)
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
            dynamic_port_policy=meta.dynamic_port_policy,
            domain=meta.domain,
            interior_domain=meta.interior_domain,
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
            trainable_by_default=meta.trainable_by_default,
            representation=meta.representation,
        )

    def _validate_representation(self, meta: DeclaredComponent) -> None:
        if meta.representation is None:
            return
        issues = validate_representation_against_component(
            meta.representation,
            component_name=meta.name,
            param_schema=meta.param_schema,
            input_ports=meta.input_ports,
        )
        if issues:
            formatted = "; ".join(f"{issue.path}: {issue.message}" for issue in issues)
            raise ValueError(
                f"Invalid representation contract for component {meta.name!r}: {formatted}"
            )

    def _migration_infos_for_target(
        self, target_type: str, migrations: Iterable[ComponentMigration]
    ) -> list[ComponentMigrationInfo]:
        return [
            self._migration_info(migration)
            for migration in sorted(
                migrations,
                key=lambda item: (
                    item.owner,
                    item.source_type,
                    str(item.source_param_schema_version or ""),
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

    def _owner_for_missing_type(self, type_id: str, meta: DeclaredComponent | None) -> str | None:
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


def _declared_source_paths(meta: DeclaredComponent) -> tuple[Path | None, ...]:
    params = meta.params
    rules = (
        params.runtime_type,
        params.builder,
        params.extractor,
        params.projection,
        meta.output_prototype_fn,
        *(group for group, _ in params.constructor.groups.values()),
        *(
            ()
            if meta.compiler.input_prototype_parameter is None
            else (meta.compiler.input_prototype_parameter.resolve,)
        ),
    )
    return (
        *meta.source_paths,
        *(source_path(rule) for rule in rules if rule is not None),
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


def required_interior_domain(node_type: str, registry: Any) -> str | None:
    """Return the required interior domain for a composite component type."""
    meta = None
    if isinstance(registry, Mapping):
        meta = registry.get(node_type)
    else:
        get_meta = getattr(registry, "get", None)
        if callable(get_meta):
            meta = get_meta(node_type)
        elif isinstance(registry, (list, tuple)):
            for item in registry:
                if getattr(item, "name", None) == node_type:
                    meta = item
                    break
    if meta is None:
        return _LEGACY_INTERIOR_DOMAINS.get(node_type)
    is_composite = (
        bool(meta.get("is_composite", False))
        if isinstance(meta, Mapping)
        else bool(getattr(meta, "is_composite", False))
    )
    if not is_composite:
        return None
    interior_domain = (
        meta.get("interior_domain")
        if isinstance(meta, Mapping)
        else getattr(meta, "interior_domain", None)
    )
    return interior_domain if isinstance(interior_domain, str) and interior_domain else None


def format_missing_interior_message(
    *,
    node_name: str,
    node_type: str,
    domain_id: str,
) -> str:
    """Return a human-facing missing-interior error message."""
    if not domain_id or domain_id == CAUSAL_DOMAIN_ID:
        return f"{node_type} node {node_name!r} requires a subgraph, but none was provided"
    label = domain_id.removeprefix("feedbax.domain.").replace("_", " ")
    article = "an" if label[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    return (
        f"Node {node_name!r} ({node_type}) requires {article} {label} interior "
        f"({domain_id}) but none is present. Open it in Studio and save."
    )
