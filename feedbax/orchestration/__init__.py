"""Run-orchestration support surfaces."""

from __future__ import annotations

import importlib

__all__ = [
    "EXECUTION_IDENTITY_ENVELOPE_SCHEMA_ID",
    "EXECUTION_IDENTITY_ENVELOPE_SCHEMA_VERSION",
    "RUN_ASSEMBLY_REQUEST_SCHEMA_ID",
    "RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION",
    "RUN_BUNDLE_SCHEMA_ID",
    "RUN_BUNDLE_SCHEMA_VERSION",
    "RUN_BUNDLE_SCHEMA_VERSION_V1",
    "CHECK_STATUS_FAIL",
    "CHECK_STATUS_PASS",
    "CHECK_STATUS_SKIPPED",
    "RUN_CONFORMANCE_SCHEMA_ID",
    "RUN_CONFORMANCE_SCHEMA_VERSION",
    "RUN_EVENT_SCHEMA_ID",
    "RUN_EVENT_SCHEMA_VERSION",
    "RUN_SET_STATE_SCHEMA_ID",
    "RUN_SET_STATE_SCHEMA_VERSION",
    "BatchLineFormatError",
    "AssemblyCompilerRegistry",
    "AssemblyContext",
    "AssemblyInputDeclaration",
    "AuthoredIntentRef",
    "BudgetPolicy",
    "CertificateRow",
    "CheckEntry",
    "CheckRegistry",
    "ConformanceRowArtifacts",
    "CompiledExecutionRow",
    "CompiledRowStorageResult",
    "CompiledRunSet",
    "CompilerIdentity",
    "DriverRowProbe",
    "EnvironmentDeclaration",
    "ExecutionCapsuleRef",
    "ExecutionIdentityEnvelope",
    "ImmutableInputDigest",
    "ImmutableInputIdentity",
    "InputCustodyPin",
    "LaunchPolicy",
    "LocalDriverError",
    "LocalOrchestrationDriver",
    "OrchestrationDriver",
    "PreflightCheckEntry",
    "ReconciledRunStatus",
    "RepoRevision",
    "ResolvedSnapshotRef",
    "RowIdentities",
    "RowLaunchSpec",
    "RowState",
    "RunBundle",
    "RunConformanceCertificate",
    "RunEvent",
    "RunEventEmitter",
    "RunEventProtocolError",
    "RunEventReader",
    "RunRowSpec",
    "RunAssemblyRequest",
    "RunSetState",
    "RunSetStateStore",
    "STAGE_ORDER",
    "StageEngine",
    "StageState",
    "StateLockError",
    "SchemaArtifactRef",
    "assemble_run_bundle",
    "assert_certificate_allows_completed_registration",
    "build_core_check_registry",
    "build_default_check_registry",
    "build_default_assembly_registry",
    "compute_environment_fingerprint",
    "default_orchestration_root",
    "format_batch_line",
    "mint_run_set_id",
    "persist_compiled_row",
    "run_preflight_checks",
    "run_conformance_checks",
    "write_conformance_certificate",
]

_EXPORT_MODULES = (
    "feedbax.orchestration.bundle",
    "feedbax.orchestration.assembly",
    "feedbax.orchestration.conformance",
    "feedbax.orchestration.drivers.base",
    "feedbax.orchestration.drivers.local",
    "feedbax.orchestration.events",
    "feedbax.orchestration.stages",
    "feedbax.orchestration.state",
)


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    for module_name in _EXPORT_MODULES:
        module = importlib.import_module(module_name)
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
