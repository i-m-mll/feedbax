from __future__ import annotations

from .meta import ComponentBuilder, ComponentMeta, OutputPrototypeFn
from .registry import (
    ComponentRegistry,
    ComponentResolution,
    TemplateBuilderIssue,
    format_missing_interior_message,
    get_component_registry,
    register_component_type,
    required_interior_domain,
)
from .cde_templates import register_cde_templates
from .domains import DomainRegistry, builtin_domain_registry
from feedbax.contracts.migrations import ComponentMigration, ComponentMigrationPack

__all__ = [
    "ComponentBuilder",
    "ComponentMigration",
    "ComponentMigrationPack",
    "ComponentMeta",
    "ComponentResolution",
    "OutputPrototypeFn",
    "ComponentRegistry",
    "DomainRegistry",
    "builtin_domain_registry",
    "format_missing_interior_message",
    "get_component_registry",
    "register_component_type",
    "register_cde_templates",
    "required_interior_domain",
    "TemplateBuilderIssue",
]
