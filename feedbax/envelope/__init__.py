"""The authored-envelope engine: one dialect, one compiler, closed layers.

Feedbax owns the mechanism by which an authored envelope becomes a compiled
document plus a compile lock, and it owns the dialect that mechanism reads. A
project owns the science the documents carry. The split is enforced by
construction: a project contributes a
:class:`~feedbax.contracts.project_experiment.ProjectExperimentDeclaration`,
which is inert data naming two directories and one budget resource, and it
contributes no callable at all.

The kernel is four mechanisms:

* **canonical form** — one hash domain for every content pin and one deterministic
  on-disk form, so a regenerated document is byte-identical to the tracked one
  (:mod:`feedbax.contracts.authored_canonical`);
* **budgets** — per-layer caps on how much an authored document may say, with the
  document schema owned here and the numbers owned by the project
  (:mod:`feedbax.contracts.authoring_budget`, :mod:`feedbax.envelope.authoring`);
* **compilation** — parent resolution over a content-pinned lineage, assertion
  verification, echo refusal, closed-layer lowering, and lock emission
  (:mod:`feedbax.contracts.experiment_envelope_dialect`,
  :mod:`feedbax.envelope.compile`, :mod:`feedbax.envelope.resolution`);
* **the choke point** — proof that tracked compiled bytes are what the envelopes
  compile to right now (:mod:`feedbax.envelope.choke`).

The compile lock's plan/receipt boundary is the invariant everything else rests
on: a lock records what was knowable *before* a run was allocated, and never a
fact only running can produce. See
:mod:`feedbax.contracts.experiment_compile_lock`.
"""

from __future__ import annotations

from feedbax.contracts.authored_canonical import (
    CANONICAL_PIN_ALGORITHM,
    canonical_bytes,
    canonical_sha256,
    content_pin,
    emit_text,
)
from feedbax.contracts.authoring_budget import (
    AUTHORING_BUDGET_SCHEMA_ID,
    AUTHORING_BUDGET_SCHEMA_VERSION,
    AuthoringBudgets,
    LayerBudget,
    load_authoring_budget_document,
)
from feedbax.contracts.experiment_compile_lock import (
    EXPERIMENT_COMPILE_LOCK_SCHEMA_ID,
    EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION,
    RUN_RECEIPT_ONLY_FACTS,
    CompileLockInputs,
    CompilerContract,
    CompilerImplementation,
    PlanReceiptBoundaryError,
    build_compile_lock,
    check_plan_receipt_boundary,
    load_compile_lock,
)
from feedbax.envelope.authoring import (
    enforce_assertion_budget,
    enforce_budget,
    enforce_row_budget,
    guard_authored_bytes,
    read_authored_document,
)
from feedbax.envelope.choke import (
    ChokeEntry,
    ChokeFinding,
    ChokeReport,
    compare_tracked_outputs,
    format_choke_findings,
)
from feedbax.contracts.experiment_envelope_dialect import (
    EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_ID,
    EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION,
    EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION_V1,
    EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION_V2,
    EXPERIMENT_LAYER_ROOT_AUTHORITY_SCHEMA_ID,
    EXPERIMENT_LAYER_ROOT_AUTHORITY_SCHEMA_VERSION,
    EXPERIMENT_ENVELOPE_FAMILY,
    EXPERIMENT_ENVELOPE_SCHEMA_VERSION,
    EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V1,
    EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V2,
    EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V3,
    EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V4,
    EXPERIMENT_ENVELOPE_SUFFIX,
    EXPERIMENT_ENVELOPE_SUPPORTED_SCHEMA_VERSIONS,
    ROOT_TRAINING_AUTHORITY_SCHEMA_ID,
    ROOT_TRAINING_AUTHORITY_SCHEMA_VERSION,
    AnalysisBundleLayerRootAuthority,
    AnalysisRunLayerRootAuthority,
    ComparisonPolicyLayerRootAuthority,
    CompositionTrainingRootAuthoring,
    ExperimentEnvelope,
    ExperimentEnvelopeLayer,
    ExperimentLayerRootAuthority,
    FigureLayerRootAuthority,
    LayerOutputContract,
    RootTrainingRowAuthoring,
    RootSelectedCheckpointAuthoring,
    RootTrainingAuthority,
    TrainingRootAuthoring,
    TrainingRunRootAuthoring,
    compiler_contract_version_for_schema,
    parse_experiment_envelope,
)
from feedbax.envelope.compile import (
    DuplicateOutputAddressError,
    EnvelopeCompileOutcome,
    EnvelopeKernel,
    EnvelopeLayout,
    LayerCompileContext,
    LoweredLayer,
    ResolvedParent,
    UncontainedEnvelopeAliasError,
    authored_layer_of,
    check_echo,
    check_no_co_created_protected_document,
    compiled_document_pin,
    reject_echo,
    scalar_equal,
    verify_assertions,
)
from feedbax.envelope.entrypoint import (
    EXPERIMENT_ENVELOPE_IMPLEMENTATION,
    compile_experiment_envelope,
    kernel_for,
    load_project_budgets,
)
from feedbax.envelope.resolution import (
    InheritedValue,
    Lineage,
    PinnedDocument,
    build_lineage,
    load_pinned,
)

__all__ = [
    "AUTHORING_BUDGET_SCHEMA_ID",
    "AUTHORING_BUDGET_SCHEMA_VERSION",
    "CANONICAL_PIN_ALGORITHM",
    "EXPERIMENT_COMPILE_LOCK_SCHEMA_ID",
    "EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION",
    "EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_ID",
    "EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION",
    "EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION_V1",
    "EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION_V2",
    "EXPERIMENT_LAYER_ROOT_AUTHORITY_SCHEMA_ID",
    "EXPERIMENT_LAYER_ROOT_AUTHORITY_SCHEMA_VERSION",
    "EXPERIMENT_ENVELOPE_FAMILY",
    "EXPERIMENT_ENVELOPE_IMPLEMENTATION",
    "EXPERIMENT_ENVELOPE_SCHEMA_VERSION",
    "EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V1",
    "EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V2",
    "EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V3",
    "EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V4",
    "EXPERIMENT_ENVELOPE_SUFFIX",
    "EXPERIMENT_ENVELOPE_SUPPORTED_SCHEMA_VERSIONS",
    "ROOT_TRAINING_AUTHORITY_SCHEMA_ID",
    "ROOT_TRAINING_AUTHORITY_SCHEMA_VERSION",
    "RUN_RECEIPT_ONLY_FACTS",
    "AuthoringBudgets",
    "AnalysisBundleLayerRootAuthority",
    "AnalysisRunLayerRootAuthority",
    "ComparisonPolicyLayerRootAuthority",
    "ChokeEntry",
    "ChokeFinding",
    "ChokeReport",
    "CompileLockInputs",
    "CompositionTrainingRootAuthoring",
    "CompilerContract",
    "CompilerImplementation",
    "DuplicateOutputAddressError",
    "EnvelopeCompileOutcome",
    "EnvelopeKernel",
    "EnvelopeLayout",
    "ExperimentEnvelope",
    "ExperimentEnvelopeLayer",
    "ExperimentLayerRootAuthority",
    "FigureLayerRootAuthority",
    "InheritedValue",
    "LayerBudget",
    "LayerCompileContext",
    "LayerOutputContract",
    "Lineage",
    "LoweredLayer",
    "PinnedDocument",
    "PlanReceiptBoundaryError",
    "ResolvedParent",
    "RootTrainingRowAuthoring",
    "RootSelectedCheckpointAuthoring",
    "RootTrainingAuthority",
    "TrainingRootAuthoring",
    "TrainingRunRootAuthoring",
    "UncontainedEnvelopeAliasError",
    "authored_layer_of",
    "build_compile_lock",
    "build_lineage",
    "canonical_bytes",
    "canonical_sha256",
    "check_echo",
    "check_no_co_created_protected_document",
    "check_plan_receipt_boundary",
    "compare_tracked_outputs",
    "compile_experiment_envelope",
    "compiled_document_pin",
    "compiler_contract_version_for_schema",
    "content_pin",
    "emit_text",
    "enforce_assertion_budget",
    "enforce_row_budget",
    "enforce_budget",
    "format_choke_findings",
    "guard_authored_bytes",
    "kernel_for",
    "load_authoring_budget_document",
    "load_compile_lock",
    "load_pinned",
    "load_project_budgets",
    "parse_experiment_envelope",
    "read_authored_document",
    "reject_echo",
    "scalar_equal",
    "verify_assertions",
]
