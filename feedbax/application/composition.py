"""Compose the same installed declarations for application and worker processes."""

from __future__ import annotations

from functools import lru_cache

from feedbax.analysis.operations import OperationDeclaration
from feedbax.component_registry import ComponentRegistry
from feedbax.component_registry.registry import installed_components
from feedbax.plot.templates import FigureTemplateDeclaration
from feedbax.registry_errors import RegistryCollisionError
from feedbax.synthesis.methods import SynthesisMethodDeclaration
from feedbax.tasks.registry import PythonTask, PythonTaskRegistry
from feedbax.training.methods import TrainingMethodDeclaration

from feedbax.application.bundle import BootstrapState, empty_application_registry_bundle
from feedbax.plugins.bootstrap import (
    BootstrapError,
    BootstrapErrorCode,
    PackageDeclarations,
    installed_declarations,
    installed_packages,
    scan_packages,
)
from feedbax.plugins.declarations import DistributionSource, declaration_key


def build_application(
    packages: tuple[PackageDeclarations, ...] | None = None,
) -> BootstrapState:
    packages = installed_declarations() if packages is None else packages
    components = ComponentRegistry(installed_components(packages))
    bundle = empty_application_registry_bundle(components, tuple(package.source for package in packages))
    for package in packages:
        source = package.source
        current = package.source.package
        try:
            for task in package.subclasses(PythonTask):
                current = f"{task.__module__}.{task.__qualname__}"
                key = declaration_key(task.__name__, vars(task).get("key"), source)
                bundle.python_tasks.register_declaration(task, key=key, source=source)
            for function, declaration in package.functions:
                current = f"{function.__module__}.{function.__qualname__}"
                key = declaration_key(function.__name__, declaration.key, source)
                if isinstance(declaration, OperationDeclaration):
                    bundle.operations.register_declaration(function, declaration, key=key, source=source)
                elif isinstance(declaration, TrainingMethodDeclaration):
                    bundle.training_methods.register_declaration(function, declaration, key=key, source=source)
                elif isinstance(declaration, SynthesisMethodDeclaration):
                    bundle.synthesis_methods.register_declaration(function, declaration, key=key, source=source)
                elif isinstance(declaration, FigureTemplateDeclaration):
                    bundle.figure_templates.register_declaration(function, declaration, key=key, source=source)
                else:
                    raise TypeError(f"Unknown Feedbax declaration {type(declaration).__qualname__}")
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
    return BootstrapState(bundle)


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


def application_python_tasks() -> PythonTaskRegistry:
    return load_application().bundle.python_tasks
