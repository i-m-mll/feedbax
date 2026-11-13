"""Objective specifications, loss terms, and loss services."""

from importlib import import_module


__all__ = [
    "ConstantScheduleSpec",
    "EpochMaskSpec",
    "LossService",
    "LoweredObjective",
    "MatrixPayloadSpec",
    "MetricSpec",
    "MovementEpochRampScheduleSpec",
    "NORM_FUNCTIONS",
    "OBJECTIVE_SCHEMA_VERSION",
    "ObjectiveExecutionRequirements",
    "ObjectiveLoweringError",
    "ObjectiveSpec",
    "ObjectiveSpecModel",
    "ObjectiveTermSpec",
    "PowerLawScheduleSpec",
    "ProbeInfo",
    "FiniteDifferenceLossSpec",
    "MatrixQuadraticLossSpec",
    "ReductionSpec",
    "ScheduleSpec",
    "SelectorAddressSpec",
    "SelectorObjectiveLoss",
    "TargetSpecResult",
    "TargetStateLossSpec",
    "TargetValueSpec",
    "TaskTimelineSpec",
    "TimeRange",
    "TimelineEpochSpec",
    "TimelineEventSpec",
    "canonical_objective_payload",
    "loss_service",
    "make_streaming_loss_fn",
    "migrate_objective_spec",
    "objective_schema_models",
    "validate_objective_spec",
]


_SPEC_EXPORTS = frozenset({
    "OBJECTIVE_SCHEMA_VERSION",
    "ConstantScheduleSpec",
    "EpochMaskSpec",
    "MatrixPayloadSpec",
    "MetricSpec",
    "MovementEpochRampScheduleSpec",
    "ObjectiveSpec",
    "ObjectiveExecutionRequirements",
    "ObjectiveSpecModel",
    "ObjectiveTermSpec",
    "PowerLawScheduleSpec",
    "FiniteDifferenceLossSpec",
    "MatrixQuadraticLossSpec",
    "ReductionSpec",
    "ScheduleSpec",
    "SelectorAddressSpec",
    "TargetStateLossSpec",
    "TargetValueSpec",
    "TaskTimelineSpec",
    "TimelineEpochSpec",
    "TimelineEventSpec",
    "canonical_objective_payload",
    "migrate_objective_spec",
    "objective_schema_models",
    "validate_objective_spec",
})

_LOSS_EXPORTS = frozenset({
    "AbstractLoss",
    "AbstractTermedLoss",
    "CompositeLoss",
    "CrossTimestepLoss",
    "EffectorStraightPathLoss",
    "EpochMaskedLoss",
    "FuncTermsLoss",
    "ModelLoss",
    "NthDifferenceLoss",
    "OutputJerkLoss",
    "StateDerivativeLoss",
    "StopAtGoalLoss",
    "TargetSpec",
    "TargetStateLoss",
    "WeightsSpec",
    "nan_safe_mse",
    "power_discount",
    "reduce_over_time_with_weights",
})

_SERVICE_EXPORTS = frozenset({
    "LoweredObjective",
    "NORM_FUNCTIONS",
    "LossService",
    "ObjectiveLoweringError",
    "ProbeInfo",
    "SelectorObjectiveLoss",
    "TargetSpecResult",
    "TimeRange",
    "loss_service",
})


def __getattr__(name: str):
    if name in _SPEC_EXPORTS:
        module_name = ".spec"
    elif name in _SERVICE_EXPORTS:
        module_name = ".service"
    elif name == "make_streaming_loss_fn":
        module_name = ".streaming"
    elif name in {"TermTree", "is_termtree"}:
        module_name = ".base"
    elif name in _LOSS_EXPORTS:
        module_name = ".loss"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
