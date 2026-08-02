"""Feedbax application registry bundle and declared plugin family keys."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.experiment_envelope import ExperimentEnvelopeCompilerRegistry
from feedbax.envelope.entrypoint import register_builtin_experiment_envelope_compiler
from feedbax.analysis.evaluation import EvaluationRecipeRegistry
from feedbax.analysis.evaluation_compaction import EvaluationBatchConsumerRegistry
from feedbax.analysis.evaluation_product_union import (
    EvaluationCompactProductUnionFinalizerRegistry,
)
from feedbax.analysis.reports import ReportRecipeRegistry, register_builtin_report_recipes
from feedbax.analysis.specs import AnalysisRecipeRegistry
from feedbax.contracts.training import TrainingMethodRegistry, default_training_method_registry
from feedbax.orchestration.conformance import CheckRegistry, build_core_check_registry
from feedbax.orchestration.drivers.builtins import build_builtin_driver_registry
from feedbax.orchestration.drivers.capabilities import DriverRegistry
from feedbax.plot.constructors import FigureRegistry, register_default_figure_constructors
from feedbax.training.preparation import ExecutionPreparationProviderRegistry
from feedbax.training.row_lowering import TrainingRowLowererRegistry

from .bootstrap import RegistrationContext, RegistryKey
from .registry import ExperimentRegistry


@dataclass(frozen=True)
class ApplicationRegistryBundle:
    """Fresh registries owned by one completed application bootstrap."""

    components: ComponentRegistry
    training_methods: TrainingMethodRegistry
    row_lowerers: TrainingRowLowererRegistry
    execution_preparations: ExecutionPreparationProviderRegistry
    experiment_packages: ExperimentRegistry
    analysis_recipes: AnalysisRecipeRegistry
    evaluation_recipes: EvaluationRecipeRegistry
    evaluation_batch_consumers: EvaluationBatchConsumerRegistry
    evaluation_product_union_finalizers: EvaluationCompactProductUnionFinalizerRegistry
    report_recipes: ReportRecipeRegistry
    figures: FigureRegistry
    conformance_checks: CheckRegistry
    drivers: DriverRegistry
    experiment_envelope_compilers: ExperimentEnvelopeCompilerRegistry

    def seal(self) -> None:
        self.components.seal()
        self.training_methods.seal()
        self.row_lowerers.seal()
        self.execution_preparations.seal()
        self.experiment_packages.seal()
        self.analysis_recipes.seal()
        self.evaluation_recipes.seal()
        self.evaluation_batch_consumers.seal()
        self.evaluation_product_union_finalizers.seal()
        self.report_recipes.seal()
        self.figures.seal()
        self.conformance_checks.seal()
        self.drivers.seal()
        self.experiment_envelope_compilers.seal()


COMPONENTS = RegistryKey(
    "components", "components", ComponentRegistry, registered_keys=lambda value: value.names()
)
TRAINING_METHODS = RegistryKey(
    "training_methods",
    "training_methods",
    TrainingMethodRegistry,
    registered_keys=lambda value: value.available_keys(),
)
ROW_LOWERERS = RegistryKey(
    "row_lowerers",
    "row_lowerers",
    TrainingRowLowererRegistry,
    registered_keys=lambda value: value.available_keys(),
)
EXECUTION_PREPARATIONS = RegistryKey(
    "execution_preparations",
    "execution_preparations",
    ExecutionPreparationProviderRegistry,
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

EXPERIMENT_ENVELOPE_COMPILERS = RegistryKey(
    "experiment_envelope_compilers",
    "experiment_envelope_compilers",
    ExperimentEnvelopeCompilerRegistry,
    registered_keys=lambda value: value.available_keys(),
)

APPLICATION_REGISTRY_KEYS = (
    COMPONENTS,
    TRAINING_METHODS,
    ROW_LOWERERS,
    EXECUTION_PREPARATIONS,
    EXPERIMENT_PACKAGES,
    ANALYSIS_RECIPES,
    EVALUATION_RECIPES,
    EVALUATION_BATCH_CONSUMERS,
    EVALUATION_PRODUCT_UNION_FINALIZERS,
    REPORT_RECIPES,
    FIGURES,
    CONFORMANCE_CHECKS,
    DRIVERS,
    EXPERIMENT_ENVELOPE_COMPILERS,
)


def new_application_registry_bundle(
    *, local_component_source: Path | None = Path.home() / ".feedbax" / "components"
) -> ApplicationRegistryBundle:
    """Construct fresh built-in registries before plugin registration."""
    components = ComponentRegistry(load_user_components=False)
    if local_component_source is not None:
        components.load_user_components(local_component_source)
    figures = FigureRegistry()
    register_default_figure_constructors(figures)
    reports = ReportRecipeRegistry()
    register_builtin_report_recipes(reports)
    envelope_compilers = ExperimentEnvelopeCompilerRegistry()
    register_builtin_experiment_envelope_compiler(envelope_compilers)
    return ApplicationRegistryBundle(
        components=components,
        training_methods=default_training_method_registry(),
        row_lowerers=TrainingRowLowererRegistry(),
        execution_preparations=ExecutionPreparationProviderRegistry(),
        experiment_packages=ExperimentRegistry(),
        analysis_recipes=AnalysisRecipeRegistry(),
        evaluation_recipes=EvaluationRecipeRegistry(),
        evaluation_batch_consumers=EvaluationBatchConsumerRegistry(),
        evaluation_product_union_finalizers=EvaluationCompactProductUnionFinalizerRegistry(),
        report_recipes=reports,
        figures=figures,
        conformance_checks=build_core_check_registry(),
        drivers=build_builtin_driver_registry(),
        experiment_envelope_compilers=envelope_compilers,
    )


def new_registration_context(
    *, local_component_source: Path | None = Path.home() / ".feedbax" / "components"
) -> RegistrationContext:
    """Return an isolated context whose bundle is private until bootstrap succeeds."""
    return RegistrationContext(
        lambda: new_application_registry_bundle(local_component_source=local_component_source),
        APPLICATION_REGISTRY_KEYS,
    )
