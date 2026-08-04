"""Stable per-node-kind execution adapters for deterministic artifact fulfillment.

A fulfillment plan runner reaches every node kind through exactly one surface
per kind: evaluation (single spec or matrix), analysis run, figure, and report.
An adapter takes a node request plus already-admitted parent receipts, decides
between reuse and execution through the uniform admission validators in
:mod:`feedbax.analysis.fulfillment`, and returns receipts. Forward binding —
``ParentRef`` construction from the profile a receipt settled at admission, and
``StagedExactParents`` assembly — is mechanical from admitted receipts and never
re-derived from the filesystem.

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

import hashlib

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Union

from feedbax.analysis.bundles import (
    AnalysisBundleSpec,
    BundleRootIdentity,
    DuplicateBundleRootError,
    VerifiedBundleRoot,
    execute_analysis_bundle,
    execute_staged_analysis_bundle,
    require_unique_root_addresses,
    select_bundle_manifests,
    verify_bundle_roots,
)
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
    MANIFEST_KIND_BY_NODE_KIND,
    AdmissionOutcome,
    AdmittedManifestBytes,
    FulfillmentAdmissionError,
    FulfillmentNodeKind,
    FulfillmentReceipt,
    admit_analysis_receipt,
    admit_evaluation_receipt,
    admit_figure_receipt,
    admit_manifest_artifact_custody,
    admit_report_receipt,
)
from feedbax.analysis.manifest_inputs import resolve_manifest_input
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
    authenticated_manifest_ref_metadata,
    authenticated_manifest_ref_profile,
    canonical_manifest_path,
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
    #: The staged execution context this node executes under, resolved by the
    #: lowering from the run's base context plus exactly this node's parents.
    #: ``None`` means the run declared no staged bindings, and execution then
    #: takes the environment's own context — which is the empty one.
    execution_context: StagedExecutionContext | None = None

    node_kind: ClassVar[FulfillmentNodeKind] = "evaluation"


@dataclass(frozen=True)
class EvaluationMatrixNodeRequest:
    """One evaluation matrix, expanded to its canonical ordered row nodes."""

    node_key: str
    matrix: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)
    required_output_roles: tuple[str, ...] = ()
    order: int | None = None
    #: The staged execution context this node executes under, resolved by the
    #: lowering from the run's base context plus exactly this node's parents.
    #: ``None`` means the run declared no staged bindings, and execution then
    #: takes the environment's own context — which is the empty one.
    execution_context: StagedExecutionContext | None = None

    node_kind: ClassVar[FulfillmentNodeKind] = "evaluation"


@dataclass(frozen=True)
class AnalysisNodeRequest:
    """One analysis run addressed by its logical plan key."""

    node_key: str
    spec: AnalysisRunSpec
    metadata: Mapping[str, Any] = field(default_factory=dict)
    required_output_roles: tuple[str, ...] = ()
    order: int | None = None
    #: The staged execution context this node executes under, resolved by the
    #: lowering from the run's base context plus exactly this node's parents.
    #: ``None`` means the run declared no staged bindings, and execution then
    #: takes the environment's own context — which is the empty one.
    execution_context: StagedExecutionContext | None = None

    node_kind: ClassVar[FulfillmentNodeKind] = "analysis"


@dataclass(frozen=True)
class AnalysisBundleNodeRequest:
    """One analysis bundle, executed as the many products its own plan states.

    A bundle is to the analysis layer what a matrix is to the evaluation layer:
    one compiled document that is not itself an execution. Its stages are, and
    each stage writes its own ordinary manifest, so a bundle node carries many
    receipts rather than one.

    ``root_inputs`` are the authenticated receipts the closure bound to this
    node. A bundle addresses its roots as the *set* of manifests its predicate
    selects, not by role, so the binding is by manifest identity: the bound ids
    constrain the selection, and a selection that does not reproduce exactly
    that set refuses rather than executing over a different one.

    ``None`` and a bound set are two different statements, and the difference is
    load-bearing rather than a convenience for an empty tuple. ``None`` means the
    compile lock declares no root references at all, so the bundle's own authored
    predicate selects its roots from whatever the manifest repository holds when
    it runs. A non-``None`` value means the lock declared the root set, and the
    predicate is then a claim that must reproduce it exactly. An empty declared
    set is neither and is refused: a bundle executing over no roots is not work.
    """

    node_key: str
    bundle: AnalysisBundleSpec
    root_inputs: tuple[ParentRef, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    order: int | None = None
    #: The staged execution context this node executes under, resolved by the
    #: lowering from the run's base context plus exactly this node's parents.
    #: ``None`` means the run declared no staged bindings, and execution then
    #: takes the environment's own context — which is the empty one.
    execution_context: StagedExecutionContext | None = None

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
    #: The staged execution context this node executes under, resolved by the
    #: lowering from the run's base context plus exactly this node's parents.
    #: ``None`` means the run declared no staged bindings, and execution then
    #: takes the environment's own context — which is the empty one.
    execution_context: StagedExecutionContext | None = None

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
    #: The staged execution context this node executes under, resolved by the
    #: lowering from the run's base context plus exactly this node's parents.
    #: ``None`` means the run declared no staged bindings, and execution then
    #: takes the environment's own context — which is the empty one.
    execution_context: StagedExecutionContext | None = None

    node_kind: ClassVar[FulfillmentNodeKind] = "report"


NodeRequest = Union[
    EvaluationNodeRequest,
    EvaluationMatrixNodeRequest,
    AnalysisNodeRequest,
    AnalysisBundleNodeRequest,
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


def node_execution_context(
    request: NodeRequest, environment: FulfillmentEnvironment
) -> StagedExecutionContext:
    """Return the staged context one node request executes under.

    A request that carries its own context carries it because the lowering
    resolved *this node's* parents into it, and that resolution is the same one
    admission, rebuild, and repair are reconstructed from — so it is used
    wherever the node is reached, not only on the first walk. A request without
    one falls back to the environment's, which is what a caller that assembled
    node requests by hand supplies, and which is the empty context whenever the
    run declared no staged bindings.
    """
    if request.execution_context is not None:
        return request.execution_context
    return environment.execution_context


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
    if isinstance(request, AnalysisBundleNodeRequest):
        return _fulfill_analysis_bundle(request, environment=environment)
    if isinstance(
        request,
        (EvaluationNodeRequest, AnalysisNodeRequest, FigureNodeRequest, ReportNodeRequest),
    ):
        return _fulfill_single(request, environment=environment)
    raise TypeError(f"unknown fulfillment node request type: {type(request).__name__}")


def admit_node(
    request: NodeRequest,
    *,
    environment: FulfillmentEnvironment,
    path: Path | None = None,
    manifest: Any | None = None,
    capture: AdmittedManifestBytes | None = None,
) -> AdmissionOutcome:
    """Admit one node's receipt through the uniform per-kind validator.

    This is the single dispatch from a node request to its kind's validator.
    Cached admission, post-execution re-admission, and custody operations all
    take this path, so no caller can accidentally admit a node against weaker
    criteria than another.

    Args:
        request: The node whose receipt is being admitted.
        environment: The environment whose ``root`` holds the receipt.
        path: Explicit manifest location, when it is not the canonical one.
        manifest: An in-memory manifest to admit instead of reading bytes from
            *path*; artifacts are still verified against the environment root.
        capture: Collects the exact manifest bytes this admission read, so a
            caller that binds the admitted receipt forward binds the read that
            decided rather than opening the path a second time.
    """
    if isinstance(request, EvaluationNodeRequest):
        return admit_evaluation_receipt(
            request.spec,
            root=environment.root,
            manifest=manifest,
            path=path,
            required_output_roles=request.required_output_roles,
            capture=capture,
        )
    if isinstance(request, AnalysisNodeRequest):
        return admit_analysis_receipt(
            request.spec,
            root=environment.root,
            manifest=manifest,
            path=path,
            required_output_roles=request.required_output_roles,
            capture=capture,
        )
    if isinstance(request, FigureNodeRequest):
        return admit_figure_receipt(
            figure_execution_plan(request, environment=environment),
            root=environment.root,
            manifest=manifest,
            path=path,
            required_output_roles=request.required_output_roles,
            capture=capture,
        )
    if isinstance(request, ReportNodeRequest):
        return admit_report_receipt(
            report_execution_spec(request),
            root=environment.root,
            manifest=manifest,
            path=path,
            required_output_roles=request.required_output_roles,
            capture=capture,
        )
    if isinstance(request, EvaluationMatrixNodeRequest):
        raise TypeError(
            f"evaluation matrix node {request.node_key!r} has no single receipt; expand it "
            "with expand_evaluation_matrix_node and admit each row"
        )
    if isinstance(request, AnalysisBundleNodeRequest):
        raise TypeError(
            f"analysis bundle node {request.node_key!r} has no single receipt; a bundle's "
            "stages are its executions, and each writes its own manifest. Admit those "
            "manifests as the per-kind receipts they are."
        )
    raise TypeError(f"unknown fulfillment node request type: {type(request).__name__}")


def execute_node(
    request: NodeRequest,
    *,
    environment: FulfillmentEnvironment,
) -> tuple[Any, Path]:
    """Execute one node unconditionally and return its manifest and path.

    Execution never consults an existing receipt. Reuse is the caller's
    decision, taken through :func:`admit_node`; custody operations execute into
    a shadow root where reuse would defeat the purpose.
    """
    if isinstance(request, EvaluationNodeRequest):
        return _execute_evaluation(request, environment)
    if isinstance(request, AnalysisNodeRequest):
        return _execute_analysis(request, environment)
    if isinstance(request, FigureNodeRequest):
        return _execute_figure(
            request,
            figure_execution_plan(request, environment=environment),
            environment,
        )
    if isinstance(request, ReportNodeRequest):
        return _execute_report(request, environment)
    if isinstance(request, EvaluationMatrixNodeRequest):
        raise TypeError(
            f"evaluation matrix node {request.node_key!r} is not a single execution; expand it "
            "with expand_evaluation_matrix_node and execute each row"
        )
    if isinstance(request, AnalysisBundleNodeRequest):
        raise TypeError(
            f"analysis bundle node {request.node_key!r} is not a single execution; its stages "
            "are, and they are driven together by execute_analysis_bundle_node"
        )
    raise TypeError(f"unknown fulfillment node request type: {type(request).__name__}")


def _fulfill_single(
    request: NodeRequest,
    *,
    environment: FulfillmentEnvironment,
) -> NodeFulfillment:
    captured = AdmittedManifestBytes()
    outcome = admit_node(request, environment=environment, capture=captured)
    if outcome.admitted:
        if outcome.manifest_path is None:
            raise RuntimeError("an admitted receipt must name the bytes that authenticated it")
        return NodeFulfillment(
            node_key=request.node_key,
            node_kind=request.node_kind,
            disposition="reused",
            receipts=(
                captured.receipt(node_kind=request.node_kind, root=environment.root),
            ),
            admissions=(outcome,),
        )
    if outcome.manifest_present:
        raise FulfillmentAdmissionError(outcome)
    _manifest, path = execute_node(request, environment=environment)
    # The receipt is the bytes re-admission read, not the executor's return
    # value and not a later reopening of the path. Re-admission is the moment
    # the written record was authenticated, so it is the moment whose bytes the
    # receipt may claim.
    written = AdmittedManifestBytes()
    verified = _reverify(
        request, environment=environment, manifest_path=path, capture=written
    )
    return NodeFulfillment(
        node_key=request.node_key,
        node_kind=request.node_kind,
        disposition="executed",
        receipts=(written.receipt(node_kind=request.node_kind, root=environment.root),),
        admissions=(outcome, verified),
    )


def _reverify(
    request: NodeRequest,
    *,
    environment: FulfillmentEnvironment,
    manifest_path: Path,
    capture: AdmittedManifestBytes | None = None,
) -> AdmissionOutcome:
    """Admit a just-written receipt through the same uniform validator.

    A freshly executed node must satisfy exactly the criteria a cached one does.
    Re-admitting closes the gap where a first run would accept a record that a
    later run would refuse.
    """
    outcome = admit_node(
        request, environment=environment, path=manifest_path, capture=capture
    )
    if not outcome.admitted:
        raise FulfillmentAdmissionError(outcome)
    return outcome


def _execute_evaluation(
    request: EvaluationNodeRequest,
    environment: FulfillmentEnvironment,
):
    """Execute one evaluation node unconditionally.

    ``execute_evaluation_run_spec`` serves the evaluated states from its cache
    by default whenever ``<root>/cache/states/<id>.pkl`` exists, and then takes
    the summary metrics and artifacts of the previously completed manifest
    rather than the recipe's. A shadow root mirrors that cache along with
    everything else a parent might resolve through, so a rebuild would rebuild
    nothing: the recipe would not run and the receipt would reproduce itself.
    ``force=True`` re-executes while leaving the cache write in place, so
    downstream state consumers still find the states this run produced;
    ``use_cache=False`` would additionally stop populating that cache, which is
    a different decision than the one :func:`execute_node` needs to make.
    """
    return execute_evaluation_run_spec(
        request.spec,
        registry=environment.registries.evaluation_recipes,
        root=environment.root,
        repo_root=environment.repo_root,
        issues=list(environment.issues),
        metadata=dict(request.metadata),
        execution_context=node_execution_context(request, environment),
        force=True,
    )


def _execute_analysis(
    request: AnalysisNodeRequest,
    environment: FulfillmentEnvironment,
):
    """Execute one analysis node unconditionally.

    ``execute_analysis_run_spec`` short-circuits by default on any completed
    manifest carrying the same id anywhere beneath its root. That admission is
    weaker than :func:`admit_node`'s — status and kind only — and it would fire
    on the mirrored receipt when a rebuild executes into shadow custody, so the
    node would return its own authoritative record instead of re-executing and
    every rebuild would match vacuously. ``use_cache=False`` opts this call path
    out of that cache entirely, which is what :func:`execute_node` promises:
    execution never consults an existing receipt. Reuse remains the caller's
    decision, taken through :func:`admit_node`.
    """
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
        execution_context=node_execution_context(request, environment),
        use_cache=False,
    )


def _figure_execution_context(
    request: FigureNodeRequest,
    environment: FulfillmentEnvironment,
) -> StagedExecutionContext | None:
    """Return the context figure execution binds against, or ``None`` for root resolution.

    Figure input resolution takes one of two authorities: an explicit staged
    execution context, or the receipt root itself. A run that declares no staged
    bindings means the second — every parent lives at its canonical location
    beneath the receipt root. Planning and execution must agree on this choice,
    because the recorded runtime binding provenance depends on it, and both
    reach it through here.
    """
    context = node_execution_context(request, environment)
    if context is EMPTY_STAGED_EXECUTION_CONTEXT:
        return None
    return context


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
        execution_context=_figure_execution_context(request, environment),
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
        execution_context=_figure_execution_context(request, environment),
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
    context = node_execution_context(request, environment)
    if request.exact_parents is not None:
        return execute_authored_report_spec(
            request.spec,
            registry=environment.registries.report_recipes,
            exact_parents=request.exact_parents,
            root=environment.root,
            # The node's parents were located once, by the walk that proved this
            # closure. Handing the resolved context over is what keeps the report
            # executing against those locations instead of resolving its own,
            # which for a parent held by a retained store or an artifact provider
            # would look beneath the receipt root and find nothing.
            execution_context=request.execution_context,
        )
    return execute_report_spec(
        request.spec,
        registry=environment.registries.report_recipes,
        root=environment.root,
        provenance=Provenance(parents=list(request.spec.inputs)),
        issues=list(environment.issues),
        metadata=dict(request.metadata),
        execution_context=context,
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


#: Manifest kind back to the fulfillment node kind whose receipt it is. A
#: bundle's stages write ordinary manifests of several kinds, and each product is
#: a receipt of exactly the kind its own manifest declares.
_NODE_KIND_BY_MANIFEST_KIND: dict[str, FulfillmentNodeKind] = {
    manifest_kind: node_kind
    for node_kind, manifest_kind in MANIFEST_KIND_BY_NODE_KIND.items()
}


def analysis_bundle_root_identities(
    request: AnalysisBundleNodeRequest,
) -> tuple[BundleRootIdentity, ...] | None:
    """Return the full identities the closure bound as this bundle's roots.

    ``None`` means the compile lock declares no root reference, so the bundle
    selects its roots the way its own document says: by its authored predicate.
    That is the only case in which ambient selection stands, and it stands
    because nothing authenticated anything to compare it against — not because a
    declared set happened to come back empty.

    A declared root set constrains the selection to exactly the receipts the lock
    named. A declared set with no members is refused here rather than degraded
    into the ambient case, because the two would otherwise be indistinguishable
    at the point where the distinction decides whether authentication applies.
    """
    if request.root_inputs is None:
        return None
    if not request.root_inputs:
        raise ValueError(
            f"analysis bundle node {request.node_key!r} declares an empty root set; a bundle "
            "executes over at least one root receipt, and an empty declaration is not the "
            "same statement as declaring no roots at all"
        )
    identities = tuple(
        BundleRootIdentity.of_parent(parent) for parent in request.root_inputs
    )
    addresses = [identity.address for identity in identities]
    duplicates = sorted(
        {f"{kind}:{run_id}" for kind, run_id in addresses if addresses.count((kind, run_id)) > 1}
    )
    if duplicates:
        raise ValueError(
            f"analysis bundle node {request.node_key!r} binds the same root receipt twice: "
            f"{duplicates!r}; one required input names one root manifest"
        )
    return identities


def analysis_bundle_root_run_ids(request: AnalysisBundleNodeRequest) -> tuple[str, ...] | None:
    """Return just the manifest ids of the bound root set, for selection.

    The bundle selection API addresses candidates by id, so this is the shape it
    takes. It is deliberately *not* the shape the equality gate compares: an id
    alone cannot tell a locked ``TrainingRunManifest`` from a selected
    ``EvaluationRunManifest`` that happens to share it. Use
    :func:`analysis_bundle_root_identities` to decide whether a selection is the
    bound set; use this only to narrow the search.
    """
    identities = analysis_bundle_root_identities(request)
    if identities is None:
        return None
    return tuple(identity.id for identity in identities)


def _require_bundle_roots(
    request: AnalysisBundleNodeRequest,
    identities: Sequence[BundleRootIdentity] | None,
    selected: Sequence[BundleRootIdentity],
    *,
    stage: str,
) -> None:
    """Refuse a selection that is not exactly the set the closure bound.

    Equality is over full identities. Kind and id together say which artifact was
    selected, and the digest — where the bound ref carries one, which it does
    whenever the lock authenticated the reference — says it is still the same
    bytes. A selection that agrees on ids while differing in either is not the
    set the plan named, and executing it would produce an artifact whose recorded
    parents describe work that was not done.

    A bound side with no digest is the lock's own statement that it authenticated
    nothing, so address equality is all there is to compare. A bound side *with* a
    digest facing a selected side without one is not that case: it is a byte
    comparison that did not happen, and it refuses. There is no flag that turns
    that refusal off — both gates over a bundle's roots now compare identities
    that were settled by a verified read, so neither has anything to skip.

    Duplicates on either side are refused before either side is keyed by
    address, because keying is exactly what makes a duplicate invisible.
    """
    if identities is None:
        return
    require_unique_root_addresses(identities, what=f"the bound root set at {stage}")
    require_unique_root_addresses(selected, what=f"the selected root set at {stage}")
    bound = {identity.address: identity for identity in identities}
    found = {identity.address: identity for identity in selected}
    differences: list[str] = []
    for address in sorted(set(bound) | set(found)):
        locked = bound.get(address)
        chosen = found.get(address)
        if locked is None:
            differences.append(f"{chosen.describe()} was selected but the plan named no such root")
            continue
        if chosen is None:
            differences.append(f"{locked.describe()} was named by the plan but not selected")
            continue
        if locked.manifest_sha256 is None:
            continue
        if chosen.manifest_sha256 is None:
            differences.append(
                f"{locked.kind}:{locked.id} is bound to manifest_sha256="
                f"{locked.manifest_sha256} size_bytes={locked.size_bytes} but the selected "
                "receipt states no byte profile to compare it with; a bound profile that "
                "cannot be checked is not a check that passed"
            )
            continue
        if (locked.manifest_sha256, locked.size_bytes) != (
            chosen.manifest_sha256,
            chosen.size_bytes,
        ):
            differences.append(
                f"{locked.kind}:{locked.id} is bound to manifest_sha256="
                f"{locked.manifest_sha256} size_bytes={locked.size_bytes} but the selected "
                f"receipt is manifest_sha256={chosen.manifest_sha256} size_bytes="
                f"{chosen.size_bytes}"
            )
    if differences:
        raise ValueError(
            f"analysis bundle node {request.node_key!r} binds root receipts "
            f"{sorted(identity.describe() for identity in identities)!r} but its predicate "
            f"{stage} selects {sorted(identity.describe() for identity in selected)!r}: "
            f"{'; '.join(differences)}. A bundle executes over exactly the receipts the plan "
            "named — the same artifacts, and the same bytes of them — so a selection that "
            "narrows, widens, substitutes a kind, or re-runs one of them describes different "
            "work."
        )


def _selected_root_identity(
    manifest: Any, *, verified: Mapping[tuple[str, str], VerifiedBundleRoot]
) -> BundleRootIdentity:
    """Return one selected manifest's identity, from the verified read of it.

    The profile comes from :func:`~feedbax.analysis.bundles.verify_bundle_roots`,
    which read this root once and proved it against what the lock pinned. It is
    not measured again here: a second read is a second file, and two reads
    bracketing a comparison are exactly how a substitution passes a gate that
    compares them.

    A selected address the bound set does not contain has no profile, and needs
    none — the address difference is what refuses it, and reading bytes to
    describe a root nobody bound would only add a read.
    """
    address = (str(manifest.kind), str(manifest.id))
    found = verified.get(address)
    if found is None:
        return BundleRootIdentity(kind=address[0], id=address[1])
    return found.identity


def _bundle_stage_receipt(
    manifest: Any,
    path: Path,
    raw: bytes | None,
    *,
    root: Path,
    node_key: str,
) -> FulfillmentReceipt:
    """Return one bundle stage product as the receipt its manifest kind makes it.

    A staged product was already read and authenticated against its stage's ref
    by :func:`~feedbax.analysis.manifest_inputs.resolve_manifest_input`, so
    *raw* is that read and the receipt is built from it. A template product came
    back from the executor with no authenticated read behind it, so *raw* is
    ``None`` and the receipt performs the one read that authenticates it.

    Either way that read authenticates the *manifest* and nothing it points at.
    Reusing a cached stage product runs the full per-kind admission, artifact
    byte custody included, so a freshly written product bound on manifest bytes
    alone would carry weaker custody on its first run than the identical
    artifacts carry on the next one. The artifact half of that admission is
    therefore run here, against the manifest this run produced, before any
    receipt is bound forward.
    """
    node_kind = _NODE_KIND_BY_MANIFEST_KIND.get(str(manifest.kind))
    if node_kind is None:
        raise ValueError(
            f"analysis bundle node {node_key!r} produced a {manifest.kind!r} manifest, which "
            f"is not a fulfillment receipt kind; supported="
            f"{sorted(_NODE_KIND_BY_MANIFEST_KIND)}"
        )
    artifacts = admit_manifest_artifact_custody(
        manifest, node_kind=node_kind, root=root, manifest_path=path
    )
    if not artifacts.admitted:
        raise FulfillmentAdmissionError(artifacts)
    if raw is not None:
        return FulfillmentReceipt.of_admitted_bytes(
            raw, node_kind=node_kind, path=path, root=root, manifest=manifest
        )
    return FulfillmentReceipt.admitted(
        node_kind=node_kind, path=path, root=root, manifest=manifest
    )


def execute_analysis_bundle_node(
    request: AnalysisBundleNodeRequest,
    *,
    environment: FulfillmentEnvironment,
) -> tuple[FulfillmentReceipt, ...]:
    """Drive one bundle's own plan and return every product it materialized.

    Which entrypoint runs follows from the bundle document's own execution shape,
    which :class:`~feedbax.analysis.bundles.AnalysisBundleSpec` already refuses to
    leave ambiguous: a staged plan runs its ordered stages, and a template bundle
    expands its templates.

    A bound root set is **read once, here**, and proved against the compile
    lock's pins before any recipe runs. That verified read is what execution
    consumes: it is handed the proved roots rather than the ids of them, so it
    performs no selection of its own and opens none of those files again.
    Passing ids down and letting execution re-select was the whole defect —
    preflight proved one set of bytes and execution went and fetched another,
    freshly authenticating whatever it found, with only an address comparison
    between the two phases to notice.

    So the closure's roots are proved once and consumed once, and both gates —
    the predicate gate before execution and the honored-set gate after it —
    compare full identities that came from that single verified read.

    Receipts come back in the bundle's own product order — stage order, then the
    order each stage materialized — so two runs over one bundle list identically.
    """
    bundle = request.bundle
    root = Path(environment.root)
    identities = analysis_bundle_root_identities(request)
    if not bundle.stages and not bundle.templates:
        raise ValueError(
            f"analysis bundle node {request.node_key!r} declares neither a staged plan nor a "
            "template set, so it states no work; a bundle defines exactly one non-empty "
            "execution shape"
        )
    # A resolved context is handed to bundle execution as itself. It is
    # deliberately not turned back into a descriptor plus root bindings: those
    # would be a second, reconstructed statement of bindings the caller already
    # made, and the two could disagree about which authority a parent came from.
    context = node_execution_context(request, environment)
    staged_context = None if context is EMPTY_STAGED_EXECUTION_CONTEXT else context
    verified: tuple[VerifiedBundleRoot, ...] | None = None
    run_ids: tuple[str, ...] | None = None
    if identities is not None:
        run_ids = tuple(identity.id for identity in identities)
        # One read per bound root, proved against the lock's pin. Everything
        # after this — the predicate gate, execution, the honored-set gate —
        # is derived from these records and never reopens the files.
        verified = verify_bundle_roots(identities, root=root)
        by_address = {item.address: item for item in verified}
        _require_bundle_roots(
            request,
            identities,
            [
                _selected_root_identity(manifest, verified=by_address)
                for manifest in select_bundle_manifests(
                    bundle, root, run_ids=run_ids, repo_root=environment.repo_root
                )
            ],
            stage="preflight",
        )
    products: list[tuple[Any, Path, bytes | None]] = []
    if bundle.stages:
        execution = execute_staged_analysis_bundle(
            bundle,
            root=root,
            repo_root=environment.repo_root,
            run_ids=None if verified is not None else run_ids,
            verified_roots=verified,
            execution_context=staged_context,
            issues=list(environment.issues),
            registries=environment.registries,
        )
        matched = tuple(execution.matched_run_ids)
        for record in execution.stages:
            for ref in record.manifest_refs:
                resolved = resolve_manifest_input(ref, root)
                # resolve_manifest_input read and authenticated these bytes
                # against the stage's own ref; the receipt is built from that
                # read rather than from a second opening of the same path.
                products.append((resolved.manifest, resolved.path, resolved.raw_bytes))
    else:
        outputs = execute_analysis_bundle(
            bundle,
            root=root,
            repo_root=environment.repo_root,
            run_ids=None if verified is not None else run_ids,
            verified_roots=verified,
            execution_context=staged_context,
            issues=list(environment.issues),
            registries=environment.registries,
        )
        matched = tuple(
            dict.fromkeys(
                run_id for expansion, _, _ in outputs for run_id in expansion.matched_run_ids
            )
        )
        products.extend((manifest, path, None) for _, manifest, path in outputs)
    if identities is not None:
        assert verified is not None
        # Two separate facts, both proved. First, that execution honored the set
        # it was handed: what it reports back is ids, and they must be exactly
        # the ids of the verified roots. Second, that the set it consumed is the
        # set the plan bound — compared over full identities, because the
        # verified records carry the profile the single read settled. The old
        # address-only comparison here is what let substituted bytes through:
        # execution had re-read and re-authenticated them, so their ids still
        # matched while their bytes were somebody else's.
        honored = {str(run_id) for run_id in matched}
        expected_ids = {item.id for item in verified}
        if honored != expected_ids:
            raise ValueError(
                f"analysis bundle node {request.node_key!r} was given the verified root set "
                f"{sorted(expected_ids)!r} but execution reports it ran over "
                f"{sorted(honored)!r}; execution consumes the roots it is handed, and a set "
                "it reports having widened, narrowed, or substituted describes different work"
            )
        _require_bundle_roots(
            request,
            identities,
            [item.identity for item in verified],
            stage="selection",
        )
    return tuple(
        _bundle_stage_receipt(manifest, path, raw, root=root, node_key=request.node_key)
        for manifest, path, raw in products
    )


def _fulfill_analysis_bundle(
    request: AnalysisBundleNodeRequest,
    *,
    environment: FulfillmentEnvironment,
) -> NodeFulfillment:
    """Fulfill one bundle node by driving the bundle its document describes.

    A bundle node has no single canonical receipt to admit against, because the
    ids of its stage products are only knowable once the stages before them have
    bound. So the node is driven rather than admitted, and reuse is the bundle
    executor's own per-stage decision, not this surface's. That is why the
    disposition is always ``executed``: reporting reuse here would claim an
    admission that was never performed.
    """
    receipts = execute_analysis_bundle_node(request, environment=environment)
    return NodeFulfillment(
        node_key=request.node_key,
        node_kind=request.node_kind,
        disposition="executed",
        receipts=receipts,
        admissions=(),
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
    context = node_execution_context(request, environment)
    if declares_staged_parents and context is EMPTY_STAGED_EXECUTION_CONTEXT:
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
            # A matrix does not execute; its rows do. The context the matrix
            # node resolved is therefore the rows' context, and it is carried
            # rather than re-derived so an expanded row is admitted, executed,
            # rebuilt, and repaired against the same authorities.
            execution_context=request.execution_context,
        )
        for index, row in enumerate(materialized)
    )


def receipt_parent_ref(receipt: FulfillmentReceipt, *, role: str) -> ParentRef:
    """Mint an authenticated ``ParentRef`` naming exactly this receipt's bytes.

    The profile comes from the receipt, which settled it when the bytes were
    admitted. Re-reading the path here would mint the digest of whatever is
    there *now*, so bytes substituted after admission would be blessed with
    their own digest by the step whose whole purpose is to authenticate them.
    """
    return ParentRef(
        kind=receipt.manifest_kind,
        id=receipt.manifest_id,
        role=role,
        uri=None,
        metadata=authenticated_manifest_ref_metadata(
            receipt.manifest_sha256, receipt.size_bytes
        ),
    )


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
