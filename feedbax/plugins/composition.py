"""Compose the same installed declarations for application and worker processes."""

from __future__ import annotations

from functools import lru_cache
from types import MappingProxyType

from feedbax.analysis.operations import OperationDeclaration
from feedbax.component_registry import ComponentRegistry
from feedbax.component_registry.discovery import component_declaration_key
from feedbax.plot.templates import FigureTemplateDeclaration
from feedbax.registry_errors import RegistryCollisionError
from feedbax.training.methods import TrainingMethodDeclaration

from .application import (
    COMPONENTS,
    FIGURE_TEMPLATES,
    OPERATIONS,
    PYTHON_TASKS,
    TRAINING_METHODS,
    empty_application_registry_bundle,
)
from .bootstrap import (
    BootstrapError,
    BootstrapErrorCode,
    BootstrapState,
    PackageDeclarations,
    PluginProvenance,
    installed_declarations,
    installed_packages,
    scan_packages,
)
from .declarations import DistributionSource, declaration_key


def build_application(
    packages: tuple[PackageDeclarations, ...] | None = None,
) -> BootstrapState:
    packages = installed_declarations() if packages is None else packages
    components = ComponentRegistry(
        (owner, package.source) for package in packages for owner in package.components
    )
    bundle = empty_application_registry_bundle(components)
    provenance = []
    for package in packages:
        source = package.source
        families: dict[str, list[str]] = {}
        current = package.source.package
        try:
            for owner in package.components:
                key = component_declaration_key(owner, source)
                families.setdefault(COMPONENTS.family, []).append(key)
            for task in package.tasks:
                current = f"{task.__module__}.{task.__qualname__}"
                key = declaration_key(task.__name__, vars(task).get("key"), source)
                bundle.python_tasks.register_declaration(task, key=key, source=source)
                families.setdefault(PYTHON_TASKS.family, []).append(key)
            for function, declaration in package.functions:
                current = f"{function.__module__}.{function.__qualname__}"
                key = declaration_key(function.__name__, declaration.key, source)
                if isinstance(declaration, OperationDeclaration):
                    bundle.operations.register_declaration(function, declaration, key=key, source=source)
                    family = OPERATIONS
                elif isinstance(declaration, TrainingMethodDeclaration):
                    bundle.training_methods.register_declaration(function, declaration, key=key, source=source)
                    family = TRAINING_METHODS
                elif isinstance(declaration, FigureTemplateDeclaration):
                    bundle.figure_templates.register_declaration(function, declaration, key=key, source=source)
                    family = FIGURE_TEMPLATES
                else:
                    raise TypeError(f"Unknown Feedbax declaration {type(declaration).__qualname__}")
                families.setdefault(family.family, []).append(key)
        except (TypeError, ValueError) as error:
            code = (
                BootstrapErrorCode.NAMESPACE_COLLISION
                if isinstance(error, RegistryCollisionError)
                else BootstrapErrorCode.REGISTRATION_FAILURE
            )
            raise BootstrapError(
                code, f"Cannot register {current!r} from {source.distribution!r}: {error}",
                plugin_id=source.distribution,
            ) from error
        provenance.append(PluginProvenance(
            source,
            MappingProxyType({family: tuple(sorted(keys)) for family, keys in families.items()}),
        ))
    return BootstrapState(bundle, tuple(provenance))


@lru_cache(maxsize=1)
def _sealed_application(sources: tuple[DistributionSource, ...]) -> BootstrapState:
    state = build_application(scan_packages(sources))
    state.bundle.seal()
    return state


def load_application() -> BootstrapState:
    """Load the application registry from the distributions installed in this process."""
    return _sealed_application(installed_packages())


async def compose_application() -> BootstrapState:
    return load_application()
