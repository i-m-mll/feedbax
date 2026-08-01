"""The generic authored-envelope engine kernel.

Feedbax owns the mechanism by which an authored envelope becomes a compiled
document plus a compile lock; a downstream project owns the science that
mechanism carries. The split is enforced by construction: nothing in this package
names a layer, a family, a budget number, or a repository path, and everything
project-specific arrives through the project's
:class:`~feedbax.contracts.project_extension.ProjectExtensionDeclaration` and its
:class:`~feedbax.envelope.compile.ProjectCompilerHooks`.

The kernel is four mechanisms:

* **canonical form** — one hash domain for every content pin and one deterministic
  on-disk form, so a regenerated document is byte-identical to the tracked one
  (:mod:`feedbax.contracts.authored_canonical`);
* **budgets** — per-layer caps on how much an authored document may say, with the
  document schema owned here and the numbers owned by the project
  (:mod:`feedbax.contracts.authoring_budget`, :mod:`feedbax.envelope.authoring`);
* **compilation** — parent resolution over a content-pinned lineage, assertion
  verification, echo refusal, layer dispatch, and lock emission
  (:mod:`feedbax.envelope.compile`, :mod:`feedbax.envelope.resolution`);
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
from feedbax.envelope.compile import (
    AuthoredAssertion,
    CompiledLayer,
    EnvelopeCompileOutcome,
    EnvelopeKernel,
    EnvelopeLayout,
    LayerCompileContext,
    ProjectCompilerHooks,
    ResolvedParent,
    check_echo,
    check_no_co_created_protected_document,
    compiled_document_pin,
    reject_echo,
    scalar_equal,
    verify_assertions,
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
    "RUN_RECEIPT_ONLY_FACTS",
    "AuthoredAssertion",
    "AuthoringBudgets",
    "ChokeEntry",
    "ChokeFinding",
    "ChokeReport",
    "CompileLockInputs",
    "CompiledLayer",
    "CompilerContract",
    "CompilerImplementation",
    "EnvelopeCompileOutcome",
    "EnvelopeKernel",
    "EnvelopeLayout",
    "InheritedValue",
    "LayerBudget",
    "LayerCompileContext",
    "Lineage",
    "PinnedDocument",
    "PlanReceiptBoundaryError",
    "ProjectCompilerHooks",
    "ResolvedParent",
    "build_compile_lock",
    "build_lineage",
    "canonical_bytes",
    "canonical_sha256",
    "check_echo",
    "check_no_co_created_protected_document",
    "check_plan_receipt_boundary",
    "compare_tracked_outputs",
    "compiled_document_pin",
    "content_pin",
    "emit_text",
    "enforce_assertion_budget",
    "enforce_budget",
    "format_choke_findings",
    "guard_authored_bytes",
    "load_authoring_budget_document",
    "load_compile_lock",
    "load_pinned",
    "read_authored_document",
    "reject_echo",
    "scalar_equal",
    "verify_assertions",
]
