from __future__ import annotations

from .declarations import (
    ComponentAuthoringFacet,
    ComponentBuilder,
    ComponentCompilerFacet,
    ComponentConstructorPlan,
    ComponentDeclaration,
    ComponentRuntimeFacet,
    ComponentSerializationFacet,
    ComponentStudioFacet,
    ComponentTrainingFacet,
    DeclaredComponent,
    ComponentParamContract,
    ComponentParamValidationError,
    OutputPrototypeFn,
    ParameterField,
    declare_component,
)
from .registry import (
    ComponentRegistry,
    ComponentResolution,
    TemplateBuilderIssue,
    format_missing_interior_message,
    required_interior_domain,
)
from .cde_templates import register_cde_templates
from .domains import DomainRegistry, builtin_domain_registry
from feedbax.contracts.migrations import ComponentMigration, ComponentMigrationPack

__all__ = [
    "ComponentBuilder",
    "ComponentMigration",
    "ComponentMigrationPack",
    "ComponentAuthoringFacet",
    "ComponentCompilerFacet",
    "ComponentConstructorPlan",
    "ComponentDeclaration",
    "ComponentRuntimeFacet",
    "ComponentSerializationFacet",
    "ComponentStudioFacet",
    "ComponentTrainingFacet",
    "DeclaredComponent",
    "ComponentParamContract",
    "ComponentParamValidationError",
    "ParameterField",
    "declare_component",
    "ComponentResolution",
    "OutputPrototypeFn",
    "ComponentRegistry",
    "DomainRegistry",
    "builtin_domain_registry",
    "format_missing_interior_message",
    "register_cde_templates",
    "required_interior_domain",
    "TemplateBuilderIssue",
]
