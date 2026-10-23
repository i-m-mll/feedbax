"""Execution contracts, planning, backend renderers, and native training runners."""

from __future__ import annotations

import importlib


_TRAINING_EXECUTOR_EXPORTS = {
    "ManifestEmissionConflictError",
    "TrainingRunExecutionResult",
    "TrainingRunExecutorError",
    "execute_training_run_spec",
    "load_training_run_spec",
}

_RECORD_EXPORTS = {
    "INVOCATION_SCHEMA_ID",
    "INVOCATION_SCHEMA_VERSION",
    "Invocation",
    "InvocationExecutionPolicy",
    "InvocationInput",
    "InvocationOutput",
    "UnsupportedInvocationVersionError",
    "invocation_for_operation",
    "invocation_from_document",
}

__all__ = [
    *_RECORD_EXPORTS,
    "ManifestEmissionConflictError",
    "TrainingRunExecutionResult",
    "TrainingRunExecutorError",
    "execute_training_run_spec",
    "load_training_run_spec",
]


def __getattr__(name: str):
    if name in _RECORD_EXPORTS:
        value = getattr(importlib.import_module("feedbax.execution.records"), name)
        globals()[name] = value
        return value
    if name not in _TRAINING_EXECUTOR_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module("feedbax.training.executor"), name)
    globals()[name] = value
    return value
