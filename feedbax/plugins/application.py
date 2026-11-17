"""Feedbax application registry bundle and declared plugin family keys."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
from feedbax.contracts.training import (
    DeclaredTrainingProgram,
    TrainingProgramRegistry,
)
from feedbax.plot.constructors import FigureRegistry, register_default_figure_constructors
from feedbax.plot.templates import FigureTemplateRegistry
from feedbax.tasks.registry import PythonTaskRegistry
from feedbax.training.preparation import (
    ExecutionPreparationProviderRegistry,
    ExecutionPreparationRegistration,
)
from feedbax.training.studio_executor import studio_graph_training_program
from feedbax.training.methods import TrainingMethodRegistry

from .bootstrap import RegistryKey
from .declarations import DeclarationRegistry
from .registry import ExperimentRegistry


class TrainingProgramCatalog(TrainingProgramRegistry):
    """Training declarations that derive their layer registries at composition."""

    def __init__(
        self,
        preparations: ExecutionPreparationProviderRegistry,
    ) -> None:
        super().__init__()
        self._preparations = preparations

    def register_program(self, program: DeclaredTrainingProgram[Any]) -> None:
        super().register_program(program)
        if program.preparation_provider is not None:
            self._preparations.register(
                ExecutionPreparationRegistration(
                    method_ref=program.method_ref,
                    provider=program.preparation_provider,
                    owner=program.owner,
                )
            )


@dataclass(frozen=True)
class ApplicationRegistryBundle:
    """Fresh registries owned by one completed application bootstrap."""

    components: ComponentRegistry
    training_programs: TrainingProgramCatalog
    training_methods: TrainingMethodRegistry
    execution_preparations: ExecutionPreparationProviderRegistry
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

    def declaration_registries(self) -> tuple[DeclarationRegistry, ...]:
        return (
            self.components._components, self.python_tasks._declarations,
            self.training_methods._declarations, self.operations._entries,
            self.figure_templates._declarations,
        )

    def seal(self) -> None:
        self.components.seal()
        self.training_programs.seal()
        self.training_methods.seal()
        self.execution_preparations.seal()
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
TRAINING_PROGRAMS = RegistryKey("training_programs", "training_programs", TrainingProgramCatalog)
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
    preparations = ExecutionPreparationProviderRegistry()
    training_programs = TrainingProgramCatalog(preparations)
    training_programs.register_program(studio_graph_training_program())
    return ApplicationRegistryBundle(
        components=components,
        training_programs=training_programs,
        training_methods=TrainingMethodRegistry(),
        execution_preparations=preparations,
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
