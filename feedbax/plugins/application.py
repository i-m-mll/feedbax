"""Feedbax application registry bundle and declared plugin family keys."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

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
    standard_supervised_training_program,
)
from feedbax.orchestration.conformance import CheckRegistry, build_core_check_registry
from feedbax.orchestration.drivers.builtins import build_builtin_driver_registry
from feedbax.orchestration.drivers.capabilities import DriverRegistry
from feedbax.plot.constructors import FigureRegistry, register_default_figure_constructors
from feedbax.training.preparation import (
    ExecutionPreparationProviderRegistry,
    ExecutionPreparationRegistration,
)
from feedbax.training.row_lowering import TrainingRowLowererRegistry
from feedbax.training.authoring import training_method_row_lowerer_registration
from feedbax.training.studio_executor import studio_graph_training_program

from .bootstrap import RegistrationContext, RegistryKey
from .registry import ExperimentRegistry


class TrainingProgramCatalog(TrainingProgramRegistry):
    """Training declarations that derive their layer registries at composition."""

    def __init__(
        self,
        row_lowerers: TrainingRowLowererRegistry,
        preparations: ExecutionPreparationProviderRegistry,
    ) -> None:
        super().__init__()
        self._row_lowerers = row_lowerers
        self._preparations = preparations

    def register_program(self, program: DeclaredTrainingProgram[BaseModel]) -> None:
        super().register_program(program)
        if program.row_lowering is not None:
            for registration in program.row_lowering.registrations:
                self._row_lowerers.register(registration)
        if program.authoring_hook is not None:
            self._row_lowerers.register(training_method_row_lowerer_registration(program, self))
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
    row_lowerers: TrainingRowLowererRegistry
    execution_preparations: ExecutionPreparationProviderRegistry
    experiment_packages: ExperimentRegistry
    analysis_recipes: AnalysisRecipeRegistry
    operations: OperationRegistry
    expression_functions: FunctionRegistry
    evaluation_recipes: EvaluationRecipeRegistry
    evaluation_batch_consumers: EvaluationBatchConsumerRegistry
    evaluation_product_union_finalizers: EvaluationCompactProductUnionFinalizerRegistry
    report_recipes: ReportRecipeRegistry
    figures: FigureRegistry
    conformance_checks: CheckRegistry
    drivers: DriverRegistry

    def seal(self) -> None:
        self.components.seal()
        self.training_programs.seal()
        self.row_lowerers.seal()
        self.execution_preparations.seal()
        self.experiment_packages.seal()
        self.analysis_recipes.seal()
        self.operations.seal()
        self.expression_functions.seal()
        self.evaluation_recipes.seal()
        self.evaluation_batch_consumers.seal()
        self.evaluation_product_union_finalizers.seal()
        self.report_recipes.seal()
        self.figures.seal()
        self.conformance_checks.seal()
        self.drivers.seal()


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
CONFORMANCE_CHECKS = RegistryKey(
    "conformance_checks",
    "conformance_checks",
    CheckRegistry,
    registered_keys=lambda value: (key for key, _check in value.items()),
)
DRIVERS = RegistryKey(
    "drivers",
    "drivers",
    DriverRegistry,
    registered_keys=lambda value: value.registered_names(),
)

APPLICATION_REGISTRY_KEYS = (
    COMPONENTS,
    TRAINING_PROGRAMS,
    EXPERIMENT_PACKAGES,
    ANALYSIS_RECIPES,
    OPERATIONS,
    EXPRESSION_FUNCTIONS,
    EVALUATION_RECIPES,
    EVALUATION_BATCH_CONSUMERS,
    EVALUATION_PRODUCT_UNION_FINALIZERS,
    REPORT_RECIPES,
    FIGURES,
    CONFORMANCE_CHECKS,
    DRIVERS,
)


def new_application_registry_bundle() -> ApplicationRegistryBundle:
    """Construct fresh built-in registries before plugin registration."""
    from feedbax.analysis.builtin_operations import register_builtin_operations
    from feedbax.expression_builtins import register_builtins

    components = ComponentRegistry()
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
    row_lowerers = TrainingRowLowererRegistry()
    preparations = ExecutionPreparationProviderRegistry()
    training_programs = TrainingProgramCatalog(row_lowerers, preparations)
    training_programs.register_program(standard_supervised_training_program())
    training_programs.register_program(studio_graph_training_program(components))
    return ApplicationRegistryBundle(
        components=components,
        training_programs=training_programs,
        row_lowerers=row_lowerers,
        execution_preparations=preparations,
        experiment_packages=ExperimentRegistry(),
        analysis_recipes=AnalysisRecipeRegistry(),
        operations=operations,
        expression_functions=expression_functions,
        evaluation_recipes=evaluations,
        evaluation_batch_consumers=evaluation_consumers,
        evaluation_product_union_finalizers=EvaluationCompactProductUnionFinalizerRegistry(),
        report_recipes=reports,
        figures=figures,
        conformance_checks=build_core_check_registry(),
        drivers=build_builtin_driver_registry(),
    )


def new_registration_context() -> RegistrationContext:
    """Return an isolated context whose bundle is private until bootstrap succeeds."""
    return RegistrationContext(
        new_application_registry_bundle,
        APPLICATION_REGISTRY_KEYS,
    )
