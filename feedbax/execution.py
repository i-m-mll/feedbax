"""Provider-neutral execution contracts for Feedbax local and cloud runs.

This module is a compatibility facade; implementation lives in the narrower
execution model, planning, backend, and local runner modules.
"""

from __future__ import annotations

from feedbax.cloud_backends import render_modal_app, write_modal_app
from feedbax.execution_models import (
    ArtifactPolicy,
    ArtifactRoute,
    ExecutionBackend,
    ExecutionCell,
    ExecutionKind,
    ExecutionModel,
    ExecutionPlan,
    ExecutionSpec,
    HealthCheck,
    InstallMode,
    LocalBackendConfig,
    LocalExecutionResult,
    ModalBackendConfig,
    PlanStep,
    RepoRole,
    RepoSource,
    RunPodBackendConfig,
    SshBackendConfig,
)
from feedbax.execution_plan import (
    default_feedbax_sources,
    load_execution_spec,
    prepare_execution_plan,
    write_execution_plan,
)
from feedbax.local_execution import run_local_execution

__all__ = [
    "ArtifactPolicy",
    "ArtifactRoute",
    "ExecutionBackend",
    "ExecutionCell",
    "ExecutionKind",
    "ExecutionModel",
    "ExecutionPlan",
    "ExecutionSpec",
    "HealthCheck",
    "InstallMode",
    "LocalBackendConfig",
    "LocalExecutionResult",
    "ModalBackendConfig",
    "PlanStep",
    "RepoRole",
    "RepoSource",
    "RunPodBackendConfig",
    "SshBackendConfig",
    "default_feedbax_sources",
    "load_execution_spec",
    "prepare_execution_plan",
    "render_modal_app",
    "run_local_execution",
    "write_execution_plan",
    "write_modal_app",
]
