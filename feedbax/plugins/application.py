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
from feedbax.tasks.reaching import builtin_python_tasks
from feedbax.tasks.registry import PythonTaskRegistry
from feedbax.training.preparation import (
    ExecutionPreparationProviderRegistry,
    ExecutionPreparationRegistration,
)
from feedbax.training.studio_executor import studio_graph_training_program

from .bootstrap import RegistrationContext, RegistryKey
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

    def seal(self) -> None:
        self.components.seal()
        self.training_programs.seal()
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


COMPONENTS = RegistryKey(
    "components", "components", ComponentRegistry, registered_keys=lambda value: value.names()
)
TRAINING_PROGRAMS = RegistryKey(
    "training_programs",
    "training_programs",
    TrainingProgramCatalog,
    registered_keys=lambda value: value.available_keys(),
)
EXPERIMENT_PACKAGES = RegistryKey(
    "experiment_packages",
    "experiment_packages",
    ExperimentRegistry,
    registered_keys=lambda value: value.get_package_names(),
)
ANALYSIS_RECIPES = RegistryKey(
    "analysis_recipes",
    "analysis_recipes",
    AnalysisRecipeRegistry,
    registered_keys=lambda value: value.keys(),
)
OPERATIONS = RegistryKey(
    "operations", "operations", OperationRegistry, registered_keys=lambda value: value.keys()
)
EXPRESSION_FUNCTIONS = RegistryKey(
    "expression_functions", "expression_functions", FunctionRegistry,
    registered_keys=lambda value: value.keys(),
)
PYTHON_TASKS = RegistryKey(
    "python_tasks", "python_tasks", PythonTaskRegistry, registered_keys=lambda value: value.keys()
)
EVALUATION_RECIPES = RegistryKey(
    "evaluation_recipes",
    "evaluation_recipes",
    EvaluationRecipeRegistry,
    registered_keys=lambda value: value.keys(),
)
EVALUATION_BATCH_CONSUMERS = RegistryKey(
    "evaluation_batch_consumers",
    "evaluation_batch_consumers",
    EvaluationBatchConsumerRegistry,
    registered_keys=lambda value: value.keys(),
)
EVALUATION_PRODUCT_UNION_FINALIZERS = RegistryKey(
    "evaluation_product_union_finalizers",
    "evaluation_product_union_finalizers",
    EvaluationCompactProductUnionFinalizerRegistry,
    registered_keys=lambda value: value.keys(),
)
REPORT_RECIPES = RegistryKey(
    "report_recipes",
    "report_recipes",
    ReportRecipeRegistry,
    registered_keys=lambda value: value.keys(),
)
FIGURES = RegistryKey(
    "figures", "figures", FigureRegistry, registered_keys=lambda value: value.keys()
)
APPLICATION_REGISTRY_KEYS = (
    COMPONENTS,
    TRAINING_PROGRAMS,
    EXPERIMENT_PACKAGES,
    ANALYSIS_RECIPES,
    OPERATIONS,
    EXPRESSION_FUNCTIONS,
    PYTHON_TASKS,
    EVALUATION_RECIPES,
    EVALUATION_BATCH_CONSUMERS,
    EVALUATION_PRODUCT_UNION_FINALIZERS,
    REPORT_RECIPES,
    FIGURES,
)


def new_application_registry_bundle() -> ApplicationRegistryBundle:
    """Construct fresh built-in registries before plugin registration."""
    from feedbax.analysis.builtin_operations import register_builtin_operations
    from feedbax.expression_builtins import register_builtins

    operations = OperationRegistry()
    register_builtin_operations(operations)
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
        components=ComponentRegistry(),
        training_programs=training_programs,
        execution_preparations=preparations,
        experiment_packages=ExperimentRegistry(),
        analysis_recipes=AnalysisRecipeRegistry(),
        operations=operations,
        expression_functions=expression_functions,
        python_tasks=builtin_python_tasks(),
        evaluation_recipes=evaluations,
        evaluation_batch_consumers=evaluation_consumers,
        evaluation_product_union_finalizers=EvaluationCompactProductUnionFinalizerRegistry(),
        report_recipes=reports,
        figures=figures,
    )


def new_registration_context() -> RegistrationContext:
    """Return an isolated context whose bundle is private until bootstrap succeeds."""
    return RegistrationContext(
        new_application_registry_bundle,
        APPLICATION_REGISTRY_KEYS,
    )
