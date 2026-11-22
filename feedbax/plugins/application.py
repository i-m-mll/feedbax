"""Feedbax application registry bundle and declared plugin family keys."""

from __future__ import annotations

from dataclasses import dataclass

from feedbax.analysis.operations import OperationRegistry
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.expression_registry import FunctionRegistry
from feedbax.plot.templates import FigureTemplateRegistry
from feedbax.tasks.registry import PythonTaskRegistry
from feedbax.training.methods import TrainingMethodRegistry

from .bootstrap import RegistryKey
from .declarations import DeclarationRegistry, DistributionSource


@dataclass(frozen=True)
class ApplicationRegistryBundle:
    """Fresh registries owned by one completed application bootstrap."""

    components: ComponentRegistry
    training_methods: TrainingMethodRegistry
    operations: OperationRegistry
    expression_functions: FunctionRegistry
    python_tasks: PythonTaskRegistry
    figure_templates: FigureTemplateRegistry
    package_sources: tuple[DistributionSource, ...]

    def declaration_registries(self) -> tuple[DeclarationRegistry, ...]:
        return (
            self.components._components, self.python_tasks._declarations,
            self.training_methods._declarations, self.operations._entries,
            self.figure_templates._declarations,
        )

    def seal(self) -> None:
        self.components.seal()
        self.training_methods.seal()
        self.operations.seal()
        self.expression_functions.seal()
        self.python_tasks.seal()
        self.figure_templates.seal()


COMPONENTS = RegistryKey("components", "components", ComponentRegistry)
TRAINING_METHODS = RegistryKey("training_methods", "training_methods", TrainingMethodRegistry)
OPERATIONS = RegistryKey("operations", "operations", OperationRegistry)
EXPRESSION_FUNCTIONS = RegistryKey("expression_functions", "expression_functions", FunctionRegistry)
PYTHON_TASKS = RegistryKey("python_tasks", "python_tasks", PythonTaskRegistry)
FIGURE_TEMPLATES = RegistryKey("figure_templates", "figure_templates", FigureTemplateRegistry)


def empty_application_registry_bundle(
    components: ComponentRegistry, package_sources: tuple[DistributionSource, ...],
) -> ApplicationRegistryBundle:
    """Create core services around a registry of discovered components."""
    from feedbax.expression_builtins import register_builtins

    expression_functions = FunctionRegistry()
    register_builtins(expression_functions)
    return ApplicationRegistryBundle(
        components=components,
        training_methods=TrainingMethodRegistry(),
        operations=OperationRegistry(),
        expression_functions=expression_functions,
        python_tasks=PythonTaskRegistry(),
        figure_templates=FigureTemplateRegistry(),
        package_sources=package_sources,
    )


def new_application_registry_bundle() -> ApplicationRegistryBundle:
    """Return a fresh, mutable copy of the discovered application registries."""
    from .composition import build_application

    return build_application().bundle
