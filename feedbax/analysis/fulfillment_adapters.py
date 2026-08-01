"""Stable per-node-kind execution adapters for deterministic artifact fulfillment.

A fulfillment plan runner reaches every node kind through exactly one surface
per kind: evaluation (single spec or matrix), analysis run, figure, and report.
An adapter takes a node request plus already-admitted parent receipts, decides
between reuse and execution through the uniform admission validators in
:mod:`feedbax.analysis.fulfillment`, and returns receipts. Forward binding —
``ParentRef`` construction through ``authenticated_manifest_ref`` and
``StagedExactParents`` assembly — is mechanical from admitted receipts and
never re-derived from the filesystem.

Contract note: ``execute_figure_spec`` is deliberately *not* re-exported here as
a new public surface. It is outside the ratified figure inventory in
``docs/design/downstream_interface_stability.md``; figure execution reaches
downstream callers only through the adapter in this module.

Execution is strictly serial. Manifest writes are not atomic, the
content-addressed helpers are check-then-write, and the SQLite manifest index is
a shared file, so v1 excludes concurrency until single-flight ownership exists.
Ordering is deterministic and tie-broken by node key, and results are returned in
execution order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Union

from feedbax.analysis.evaluation import (
    execute_evaluation_run_spec,
    materialize_evaluation_run_matrix,
)
from feedbax.analysis.exact_parents import (
    STAGED_EXACT_PARENTS_SCHEMA_ID,
    STAGED_EXACT_PARENTS_SCHEMA_VERSION,
    StagedExactParentEntry,
    StagedExactParents,
)
from feedbax.analysis.execution_context import (
    EMPTY_STAGED_EXECUTION_CONTEXT,
    StagedExecutionContext,
)
from feedbax.analysis.figures import (
    FigureExecutionPlan,
    FigureSpecInput,
    execute_figure_spec,
    plan_figure_execution,
)
from feedbax.analysis.fulfillment import (
    AdmissionOutcome,
    FulfillmentAdmissionError,
    FulfillmentNodeKind,
    FulfillmentReceipt,
    admit_analysis_receipt,
    admit_evaluation_receipt,
    admit_figure_receipt,
    admit_report_receipt,
)
from feedbax.analysis.manifest_inputs import authenticated_manifest_ref
from feedbax.analysis.reports import (
    execute_authored_report_spec,
    execute_report_spec,
)
from feedbax.analysis.specs import execute_analysis_run_spec
from feedbax.contracts.figures import FigureInputAuthoritySpec
from feedbax.contracts.manifest import (
    AnalysisRunSpec,
    EvaluationRunSpec,
    ParentRef,
    Provenance,
    ReportSpec,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from feedbax.plugins.application import ApplicationRegistryBundle


@dataclass(frozen=True)
class FulfillmentEnvironment:
    """Everything a fulfillment run needs that is not per-node.

    Attributes:
        root: The receipt root. Every admitted and executed manifest for this
            run lives at its canonical location beneath this root.
        registries: The sealed application registries the executors dispatch on.
        repo_root: Trusted repository root for spec resolution, if any.
        execution_context: The authoritative staged execution context. Adapters
            never synthesize one; staged bindings are the caller's declaration.
        issues: Issue references recorded on produced manifests.
    """

    root: Path
    registries: "ApplicationRegistryBundle"
    repo_root: Path | None = None
    execution_context: StagedExecutionContext = EMPTY_STAGED_EXECUTION_CONTEXT
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationNodeRequest:
    """One evaluation run addressed by its logical plan key."""

    node_key: str
    spec: EvaluationRunSpec
    metadata: Mapping[str, Any] = field(default_factory=dict)
    required_output_roles: tuple[str, ...] = ()
    order: int | None = None

    node_kind: ClassVar[FulfillmentNodeKind] = "evaluation"


@dataclass(frozen=True)
class EvaluationMatrixNodeRequest:
    """One evaluation matrix, expanded to its canonical ordered row nodes."""

    node_key: str
    matrix: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)
    required_output_roles: tuple[str, ...] = ()
    order: int | None = None

    node_kind: ClassVar[FulfillmentNodeKind] = "evaluation"


@dataclass(frozen=True)
class AnalysisNodeRequest:
    """One analysis run addressed by its logical plan key."""

    node_key: str
    spec: AnalysisRunSpec
    metadata: Mapping[str, Any] = field(default_factory=dict)
    required_output_roles: tuple[str, ...] = ()
    order: int | None = None

    node_kind: ClassVar[FulfillmentNodeKind] = "analysis"


@dataclass(frozen=True)
class FigureNodeRequest:
    """One figure render, including the runtime overlay outside figure identity."""

    node_key: str
    spec: FigureSpecInput
    runtime_inputs: tuple[ParentRef, ...] | None = None
    runtime_input_authorities: tuple[FigureInputAuthoritySpec, ...] | None = None
    runtime_metadata: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    required_output_roles: tuple[str, ...] = ()
    order: int | None = None

    node_kind: ClassVar[FulfillmentNodeKind] = "figure"


@dataclass(frozen=True)
class ReportNodeRequest:
    """One report, optionally executed against authoritative exact staged parents."""

    node_key: str
    spec: ReportSpec
    exact_parents: StagedExactParents | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    required_output_roles: tuple[str, ...] = ()
    order: int | None = None

    node_kind: ClassVar[FulfillmentNodeKind] = "report"


NodeRequest = Union[
    EvaluationNodeRequest,
    EvaluationMatrixNodeRequest,
    AnalysisNodeRequest,
    FigureNodeRequest,
    ReportNodeRequest,
]


@dataclass(frozen=True)
class NodeFulfillment:
    """The outcome of fulfilling one plan node."""

    node_key: str
    node_kind: FulfillmentNodeKind
    disposition: Literal["reused", "executed"]
    receipts: tuple[FulfillmentReceipt, ...]
    admissions: tuple[AdmissionOutcome, ...]

    @property
    def receipt(self) -> FulfillmentReceipt:
        """Return the single receipt, refusing when a node produced several."""
        if len(self.receipts) != 1:
            raise ValueError(
                f"node {self.node_key!r} produced {len(self.receipts)} receipts; "
                "use .receipts for multi-receipt nodes"
            )
        return self.receipts[0]


@dataclass(frozen=True)
class FulfillmentRun:
    """The deterministic result listing for one strictly serial fulfillment."""

    results: tuple[NodeFulfillment, ...]

    @property
    def execution_order(self) -> tuple[str, ...]:
        return tuple(result.node_key for result in self.results)

    @property
    def executed(self) -> tuple[str, ...]:
        return tuple(r.node_key for r in self.results if r.disposition == "executed")

    @property
    def reused(self) -> tuple[str, ...]:
        return tuple(r.node_key for r in self.results if r.disposition == "reused")


def canonical_fulfillment_order(requests: Sequence[NodeRequest]) -> tuple[NodeRequest, ...]:
    """Return requests in the one deterministic order a fulfillment run uses.

    A request's declared ``order`` decides its position; requests that declare
    none keep their given position. Ties are broken by ``node_key`` so two runs
    over the same plan always execute in the same sequence. Duplicate logical
    keys refuse, because one key must name one node.
    """
    keys = [request.node_key for request in requests]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"duplicate fulfillment node keys: {duplicates!r}")
    decorated = [
        (request.order if request.order is not None else index, request.node_key, index, request)
        for index, request in enumerate(requests)
    ]
    decorated.sort(key=lambda item: (item[0], item[1], item[2]))
    return tuple(item[3] for item in decorated)


def fulfill_nodes(
    requests: Sequence[NodeRequest],
    *,
    environment: FulfillmentEnvironment,
) -> FulfillmentRun:
    """Fulfill every node strictly serially in canonical order.

    No node runs concurrently with another. Manifest writes are not atomic, the
    content-addressed store helpers are check-then-write, and the manifest index
    is a shared SQLite file, so overlapping execution could interleave partial
    records. Results are returned in execution order.
    """
    results = [
        fulfill_node(request, environment=environment)
        for request in canonical_fulfillment_order(requests)
    ]
    return FulfillmentRun(results=tuple(results))


def fulfill_node(
    request: NodeRequest,
    *,
    environment: FulfillmentEnvironment,
) -> NodeFulfillment:
    """Admit an existing receipt for one node, or execute it exactly once.

    An absent receipt executes. A receipt that exists but fails admission
    refuses with :class:`FulfillmentAdmissionError`; it is never silently
    re-executed over, because that would destroy the evidence of what went
    wrong.
    """
    if isinstance(request, EvaluationMatrixNodeRequest):
        return _fulfill_evaluation_matrix(request, environment=environment)
    if isinstance(request, EvaluationNodeRequest):
        return _fulfill_single(
            request,
            environment=environment,
            admit=lambda: admit_evaluation_receipt(
                request.spec,
                root=environment.root,
                required_output_roles=request.required_output_roles,
            ),
            execute=lambda: _execute_evaluation(request, environment),
        )
    if isinstance(request, AnalysisNodeRequest):
        return _fulfill_single(
            request,
            environment=environment,
            admit=lambda: admit_analysis_receipt(
                request.spec,
                root=environment.root,
                required_output_roles=request.required_output_roles,
            ),
            execute=lambda: _execute_analysis(request, environment),
        )
    if isinstance(request, FigureNodeRequest):
        plan = figure_execution_plan(request, environment=environment)
        return _fulfill_single(
            request,
            environment=environment,
            admit=lambda: admit_figure_receipt(
                plan,
                root=environment.root,
                required_output_roles=request.required_output_roles,
            ),
            execute=lambda: _execute_figure(request, plan, environment),
        )
    if isinstance(request, ReportNodeRequest):
        execution_spec = report_execution_spec(request)
        return _fulfill_single(
            request,
            environment=environment,
            admit=lambda: admit_report_receipt(
                execution_spec,
                root=environment.root,
                required_output_roles=request.required_output_roles,
            ),
            execute=lambda: _execute_report(request, environment),
        )
    raise TypeError(f"unknown fulfillment node request type: {type(request).__name__}")


def _fulfill_single(
    request: NodeRequest,
    *,
    environment: FulfillmentEnvironment,
    admit,
    execute,
) -> NodeFulfillment:
    outcome = admit()
    if outcome.admitted:
        if outcome.manifest_path is None:
            raise RuntimeError("an admitted receipt must name the bytes that authenticated it")
        return NodeFulfillment(
            node_key=request.node_key,
            node_kind=request.node_kind,
            disposition="reused",
            receipts=(
                FulfillmentReceipt(
                    node_kind=request.node_kind,
                    manifest=_load(outcome.manifest_path),
                    path=Path(outcome.manifest_path),
                    root=environment.root,
                ),
            ),
            admissions=(outcome,),
        )
    if outcome.manifest_present:
        raise FulfillmentAdmissionError(outcome)
    manifest, path = execute()
    verified = _reverify(request, environment=environment, manifest_path=path)
    return NodeFulfillment(
        node_key=request.node_key,
        node_kind=request.node_kind,
        disposition="executed",
        receipts=(
            FulfillmentReceipt(
                node_kind=request.node_kind,
                manifest=manifest,
                path=path,
                root=environment.root,
            ),
        ),
        admissions=(outcome, verified),
    )


def _reverify(
    request: NodeRequest,
    *,
    environment: FulfillmentEnvironment,
    manifest_path: Path,
) -> AdmissionOutcome:
    """Admit a just-written receipt through the same uniform validator.

    A freshly executed node must satisfy exactly the criteria a cached one does.
    Re-admitting closes the gap where a first run would accept a record that a
    later run would refuse.
    """
    if isinstance(request, EvaluationNodeRequest):
        outcome = admit_evaluation_receipt(
            request.spec,
            root=environment.root,
            path=manifest_path,
            required_output_roles=request.required_output_roles,
        )
    elif isinstance(request, AnalysisNodeRequest):
        outcome = admit_analysis_receipt(
            request.spec,
            root=environment.root,
            path=manifest_path,
            required_output_roles=request.required_output_roles,
        )
    elif isinstance(request, FigureNodeRequest):
        outcome = admit_figure_receipt(
            figure_execution_plan(request, environment=environment),
            root=environment.root,
            path=manifest_path,
            required_output_roles=request.required_output_roles,
        )
    elif isinstance(request, ReportNodeRequest):
        outcome = admit_report_receipt(
            report_execution_spec(request),
            root=environment.root,
            path=manifest_path,
            required_output_roles=request.required_output_roles,
        )
    else:  # pragma: no cover - guarded by fulfill_node
        raise TypeError(f"unknown fulfillment node request type: {type(request).__name__}")
    if not outcome.admitted:
        raise FulfillmentAdmissionError(outcome)
    return outcome


def _load(path: str):
    from feedbax.contracts.manifest import load_manifest

    return load_manifest(path)


def _execute_evaluation(
    request: EvaluationNodeRequest,
    environment: FulfillmentEnvironment,
):
    return execute_evaluation_run_spec(
        request.spec,
        registry=environment.registries.evaluation_recipes,
        root=environment.root,
        repo_root=environment.repo_root,
        issues=list(environment.issues),
        metadata=dict(request.metadata),
        execution_context=environment.execution_context,
    )


def _execute_analysis(
    request: AnalysisNodeRequest,
    environment: FulfillmentEnvironment,
):
    provenance = Provenance(parents=list(request.spec.inputs))
    return execute_analysis_run_spec(
        request.spec,
        registry=environment.registries.analysis_recipes,
        evaluation_registry=environment.registries.evaluation_recipes,
        experiment_registry=environment.registries.experiment_packages,
        root=environment.root,
        repo_root=environment.repo_root,
        provenance=provenance,
        issues=list(environment.issues),
        metadata=dict(request.metadata),
        execution_context=environment.execution_context,
    )


def _figure_execution_context(
    environment: FulfillmentEnvironment,
) -> StagedExecutionContext | None:
    """Return the context figure execution binds against, or ``None`` for root resolution.

    Figure input resolution takes one of two authorities: an explicit staged
    execution context, or the receipt root itself. A fulfillment environment
    that declares no staged bindings means the second — every parent lives at
    its canonical location beneath the receipt root. Planning and execution must
    agree on this choice, because the recorded runtime binding provenance
    depends on it.
    """
    if environment.execution_context is EMPTY_STAGED_EXECUTION_CONTEXT:
        return None
    return environment.execution_context


def figure_execution_plan(
    request: FigureNodeRequest,
    *,
    environment: FulfillmentEnvironment,
) -> FigureExecutionPlan:
    """Derive one figure node's identity, embedded spec, and runtime binding."""
    return plan_figure_execution(
        request.spec,
        runtime_inputs=(
            list(request.runtime_inputs) if request.runtime_inputs is not None else None
        ),
        runtime_input_authorities=(
            list(request.runtime_input_authorities)
            if request.runtime_input_authorities is not None
            else None
        ),
        runtime_metadata=(
            dict(request.runtime_metadata) if request.runtime_metadata is not None else None
        ),
        repo_root=environment.repo_root,
        execution_context=_figure_execution_context(environment),
        registry=environment.registries.figures,
    )


def _execute_figure(
    request: FigureNodeRequest,
    plan: FigureExecutionPlan,
    environment: FulfillmentEnvironment,
):
    return execute_figure_spec(
        plan.resolution,
        runtime_inputs=(
            list(request.runtime_inputs) if request.runtime_inputs is not None else None
        ),
        runtime_input_authorities=(
            list(request.runtime_input_authorities)
            if request.runtime_input_authorities is not None
            else None
        ),
        runtime_metadata=(
            dict(request.runtime_metadata) if request.runtime_metadata is not None else None
        ),
        repo_root=environment.repo_root,
        root=environment.root,
        issues=list(environment.issues),
        metadata=dict(request.metadata),
        execution_context=_figure_execution_context(environment),
        registry=environment.registries.figures,
    )


def report_execution_spec(request: ReportNodeRequest) -> ReportSpec:
    """Return the exact ``ReportSpec`` whose identity the report receipt carries.

    ``execute_authored_report_spec`` replaces the authored inputs with the
    authoritative exact staged parents before minting identity, so a node with
    exact parents is addressed by the substituted spec, not the authored one.
    """
    if request.exact_parents is None:
        return request.spec
    exact_refs = [entry.parent for entry in request.exact_parents.parents]
    return request.spec.model_copy(update={"inputs": exact_refs}, deep=True)


def _execute_report(
    request: ReportNodeRequest,
    environment: FulfillmentEnvironment,
):
    if request.exact_parents is not None:
        return execute_authored_report_spec(
            request.spec,
            registry=environment.registries.report_recipes,
            exact_parents=request.exact_parents,
            root=environment.root,
        )
    return execute_report_spec(
        request.spec,
        registry=environment.registries.report_recipes,
        root=environment.root,
        provenance=Provenance(parents=list(request.spec.inputs)),
        issues=list(environment.issues),
        metadata=dict(request.metadata),
        execution_context=environment.execution_context,
    )


def _fulfill_evaluation_matrix(
    request: EvaluationMatrixNodeRequest,
    *,
    environment: FulfillmentEnvironment,
) -> NodeFulfillment:
    """Fulfill every canonical row of one evaluation matrix, serially and in order.

    Rows execute into the shared receipt root through the single-spec executor.
    The matrix harness deliberately is not used here: it writes each row into
    its own ``<root>/<row_id>`` sub-root, which would fragment receipts across
    roots and make canonical receipt resolution impossible.
    """
    rows = expand_evaluation_matrix_node(request, environment=environment)
    receipts: list[FulfillmentReceipt] = []
    admissions: list[AdmissionOutcome] = []
    dispositions: list[str] = []
    for row_request in rows:
        result = fulfill_node(row_request, environment=environment)
        receipts.extend(result.receipts)
        admissions.extend(result.admissions)
        dispositions.append(result.disposition)
    return NodeFulfillment(
        node_key=request.node_key,
        node_kind="evaluation",
        disposition="reused" if all(d == "reused" for d in dispositions) else "executed",
        receipts=tuple(receipts),
        admissions=tuple(admissions),
    )


def expand_evaluation_matrix_node(
    request: EvaluationMatrixNodeRequest,
    *,
    environment: FulfillmentEnvironment,
) -> tuple[EvaluationNodeRequest, ...]:
    """Expand one matrix node into its canonical ordered per-row evaluation nodes.

    Row order is the matrix's canonical materialization order, and each row's
    logical key is ``<node_key>#<row_id>``.
    """
    materialized = materialize_evaluation_run_matrix(
        request.matrix,
        registry=environment.registries.evaluation_recipes,
        repo_root=environment.repo_root,
    )
    if isinstance(request.matrix, Mapping):
        declares_staged_parents = bool(request.matrix.get("staged_parents"))
    else:
        declares_staged_parents = bool(getattr(request.matrix, "staged_parents", None))
    if declares_staged_parents and environment.execution_context is EMPTY_STAGED_EXECUTION_CONTEXT:
        raise ValueError(
            f"evaluation matrix node {request.node_key!r} declares staged parents but the "
            "fulfillment environment supplies no staged execution context; staged bindings "
            "are the caller's explicit declaration and are never synthesized here"
        )
    return tuple(
        EvaluationNodeRequest(
            node_key=f"{request.node_key}#{row.row_id}",
            spec=row.payload,
            metadata={**dict(request.metadata), "matrix_row_id": row.row_id},
            required_output_roles=request.required_output_roles,
            order=index,
        )
        for index, row in enumerate(materialized)
    )


def receipt_parent_ref(receipt: FulfillmentReceipt, *, role: str) -> ParentRef:
    """Mint an authenticated ``ParentRef`` naming exactly this receipt's bytes."""
    return authenticated_manifest_ref(receipt.manifest, receipt.path, role)


def receipt_exact_parent_entry(
    receipt: FulfillmentReceipt,
    *,
    role: str,
) -> StagedExactParentEntry:
    """Bind one admitted receipt to its root-relative execution location."""
    try:
        execution_uri = receipt.path.relative_to(receipt.root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"receipt {receipt.manifest_id!r} at {receipt.path} escapes its root {receipt.root}"
        ) from exc
    return StagedExactParentEntry(
        parent=receipt_parent_ref(receipt, role=role),
        execution_uri=execution_uri,
    )


def staged_exact_parents_from_receipts(
    bindings: Sequence[tuple[FulfillmentReceipt, str]],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> StagedExactParents:
    """Assemble authoritative ``StagedExactParents`` from admitted receipts.

    Membership order is the given binding order; nothing is inferred, sorted, or
    deduplicated behind the caller's back.
    """
    if not bindings:
        raise ValueError("StagedExactParents requires at least one admitted receipt")
    return StagedExactParents(
        schema_id=STAGED_EXACT_PARENTS_SCHEMA_ID,
        schema_version=STAGED_EXACT_PARENTS_SCHEMA_VERSION,
        parents=[
            receipt_exact_parent_entry(receipt, role=role) for receipt, role in bindings
        ],
        metadata=dict(metadata or {}),
    )
