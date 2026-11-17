"""Feedbax application registry bundle and declared plugin family keys."""

from __future__ import annotations

from dataclasses import dataclass

from feedbax.component_registry import ComponentRegistry
from feedbax.expression_registry import FunctionRegistry
from feedbax.analysis.evaluation import EvaluationRecipeRegistry
from feedbax.analysis.evaluation_compaction import EvaluationBatchConsumerRegistry
from feedbax.analysis.evaluation_product_union import (
    EvaluationCompactProductUnionFinalizerRegistry,
)
from feedbax.analysis.operations import OperationRegistry
from feedbax.analysis.reports import ReportRecipeRegistry, register_builtin_report_recipes
from feedbax.analysis.specs import AnalysisRecipeRegistry
from feedbax.analysis.warp_qualification import register_warp_qualification
from feedbax.plot.constructors import FigureRegistry, register_default_figure_constructors
from feedbax.plot.templates import FigureTemplateRegistry
from feedbax.tasks.registry import PythonTaskRegistry
from feedbax.training.methods import TrainingMethodRegistry

from .bootstrap import RegistryKey
from .registry import ExperimentRegistry


@dataclass(frozen=True)
class ApplicationRegistryBundle:
    """Fresh registries owned by one completed application bootstrap."""

    components: ComponentRegistry
    training_methods: TrainingMethodRegistry
    experiment_packages: ExperimentRegistry
    analysis_recipes: AnalysisRecipeRegistry
    operations: OperationRegistry
    expression_functions: FunctionRegistry
    python_tasks: PythonTaskRegistry
    evaluation_recipes: EvaluationRecipeRegistry
    evaluation_batch_consumers: EvaluationBatchConsumerRegistry
    evaluation_product_union_finalizers: EvaluationCompactProductUnionFinalizerRegistry
    report_recipes: ReportRecipeRegistry
    figures: FigureRegistry
    figure_templates: FigureTemplateRegistry

    def seal(self) -> None:
        self.components.seal()
        self.training_methods.seal()
        self.experiment_packages.seal()
        self.analysis_recipes.seal()
        self.operations.seal()
        self.expression_functions.seal()
        self.python_tasks.seal()
        self.evaluation_recipes.seal()
        self.evaluation_batch_consumers.seal()
        self.evaluation_product_union_finalizers.seal()
        self.report_recipes.seal()
        self.figures.seal()
        self.figure_templates.seal()


COMPONENTS = RegistryKey("components", "components", ComponentRegistry)
TRAINING_METHODS = RegistryKey("training_methods", "training_methods", TrainingMethodRegistry)
OPERATIONS = RegistryKey("operations", "operations", OperationRegistry)
EXPRESSION_FUNCTIONS = RegistryKey("expression_functions", "expression_functions", FunctionRegistry)
PYTHON_TASKS = RegistryKey("python_tasks", "python_tasks", PythonTaskRegistry)
FIGURE_TEMPLATES = RegistryKey("figure_templates", "figure_templates", FigureTemplateRegistry)


def empty_application_registry_bundle(components: ComponentRegistry) -> ApplicationRegistryBundle:
    """Create core services around a registry of discovered components."""
    from feedbax.expression_builtins import register_builtins

    expression_functions = FunctionRegistry()
    register_builtins(expression_functions)
    figures = FigureRegistry()
    register_default_figure_constructors(figures)
    reports = ReportRecipeRegistry()
    register_builtin_report_recipes(reports)
    evaluations = EvaluationRecipeRegistry()
    evaluation_consumers = EvaluationBatchConsumerRegistry()
    register_warp_qualification(evaluations, evaluation_consumers)
    return ApplicationRegistryBundle(
        components=components,
        training_methods=TrainingMethodRegistry(),
        experiment_packages=ExperimentRegistry(),
        analysis_recipes=AnalysisRecipeRegistry(),
        operations=OperationRegistry(),
        expression_functions=expression_functions,
        python_tasks=PythonTaskRegistry(),
        evaluation_recipes=evaluations,
        evaluation_batch_consumers=evaluation_consumers,
        evaluation_product_union_finalizers=EvaluationCompactProductUnionFinalizerRegistry(),
        report_recipes=reports,
        figures=figures,
        figure_templates=FigureTemplateRegistry(),
    )


def new_application_registry_bundle() -> ApplicationRegistryBundle:
    """Return a fresh, mutable copy of the discovered application registries."""
    from .composition import build_application

    return build_application().bundle
