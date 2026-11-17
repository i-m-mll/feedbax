"""The lazy public interface for installed Feedbax declarations."""

from importlib import import_module


_PUBLIC_ATTR_MODULES = {
    "ApplicationRegistryBundle": ".application",
    "TrainingProgramCatalog": ".application",
    "TRAINING_METHODS": ".application",
    "OPERATIONS": ".application",
    "FIGURE_TEMPLATES": ".application",
    "compose_application": ".composition",
    "load_application": ".composition",
    "COMPONENTS": ".application",
    "PYTHON_TASKS": ".application",
    "TRAINING_PROGRAMS": ".application",
    "BootstrapError": ".bootstrap",
    "BootstrapErrorCode": ".bootstrap",
    "BootstrapState": ".bootstrap",
    "PluginProvenance": ".bootstrap",
    "RegistryKey": ".bootstrap",
    "ExperimentRegistry": ".registry",
    "EvaluationAuthoringSchema": "feedbax.analysis.evaluation",
    "EvaluationBatchItem": "feedbax.analysis.evaluation",
    "EvaluationBatchRecipe": "feedbax.analysis.evaluation",
    "EvaluationBatchRowError": "feedbax.analysis.evaluation",
    "EvaluationRecipe": "feedbax.analysis.evaluation",
    "EvaluationRecipeRegistry": "feedbax.analysis.evaluation",
    "EvaluationRecipeResult": "feedbax.analysis.evaluation",
    "EvaluationBatchConsumer": "feedbax.analysis.evaluation_compaction",
    "EvaluationBatchConsumerInput": "feedbax.analysis.evaluation_compaction",
    "EvaluationBatchConsumerRegistry": "feedbax.analysis.evaluation_compaction",
    "EvaluationBatchFinalizeInput": "feedbax.analysis.evaluation_compaction",
    "EvaluationBatchFragment": "feedbax.analysis.evaluation_compaction",
    "EvaluationBatchMergeInput": "feedbax.analysis.evaluation_compaction",
    "EvaluationBatchMergeState": "feedbax.analysis.evaluation_compaction",
    "EvaluationCompactProductUnionFinalizerRegistry": "feedbax.analysis.evaluation_product_union",
    "EvaluationCompactProductUnionInput": "feedbax.analysis.evaluation_product_union",
    "AnalysisRecipe": "feedbax.analysis.specs",
    "AnalysisRecipeRegistry": "feedbax.analysis.specs",
    "AnalysisRecipeResult": "feedbax.analysis.specs",
    "EMPTY_STAGED_EXECUTION_CONTEXT": "feedbax.analysis.execution_context",
    "StagedExecutionContext": "feedbax.analysis.execution_context",
    "EvaluationStatesStructureProviderProtocol": "feedbax.analysis.validation",
    "DeclaredTrainingProgram": "feedbax.contracts.training",
    "TrainingProgramDeclaration": "feedbax.contracts.training",
    "TrainingProgramRuntimeFacet": "feedbax.contracts.training",
    "TrainingProgramPreparationFacet": "feedbax.contracts.training",
    "TrainingProgramProjectionFacet": "feedbax.contracts.training",
    "declare_training_program": "feedbax.contracts.training",
    "TrainingMethodMetadataProjector": "feedbax.contracts.training",
    "TrainingMethodScheduleProjector": "feedbax.contracts.training",
    "TrainingProgramRegistry": "feedbax.contracts.training",
    "ExecutionPreparationPlan": "feedbax.training.preparation",
    "NumericalUpdateKernel": "feedbax.training.phase_executor",
    "ExecutionPreparationProvider": "feedbax.training.preparation",
    "ExecutionPreparationRequest": "feedbax.training.preparation",
    "ExecutionPreparationResult": "feedbax.training.preparation",
}

# Submodules the eager facade used to bind as attributes of this package.
_PUBLIC_SUBMODULES = ("application", "bootstrap", "registry")

__all__ = list(_PUBLIC_ATTR_MODULES)


def __getattr__(name: str):
    if name in _PUBLIC_SUBMODULES:
        value = import_module(f".{name}", __name__)
    else:
        try:
            module_name = _PUBLIC_ATTR_MODULES[name]
        except KeyError as exc:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
        value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__, *_PUBLIC_SUBMODULES})
