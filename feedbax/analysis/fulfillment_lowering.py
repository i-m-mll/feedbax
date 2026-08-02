"""Lowering one compiled document into the node request that executes it.

A compiled experiment document already *is* a Feedbax spec. Which node request
executes it therefore follows from the document's own ``schema_id`` and from
nothing else: not from a layer label an envelope carried, not from a project
callable, and not from a table a project contributes to. Both sides of the
mapping are Feedbax-owned, which is what makes the mapping Feedbax code.

## Inputs come from the lock, exactly once

A compiled document is a *plan*: it cannot name an authenticated receipt,
because authenticating one takes a run. The receipts a node binds are therefore
minted here, from the typed references its compile lock recorded, and a compiled
spec that already declares ``inputs`` refuses rather than being bound twice.

Which field a bound receipt lands in follows from the *consumer binding* the
reference carries, which is a closed Feedbax union:

* :class:`~feedbax.contracts.experiment_compile_lock.EvaluationSubjectBinding`
  binds the subject an evaluation evaluates;
* :class:`~feedbax.contracts.experiment_compile_lock.AnalysisInputBinding` binds
  one analysis input, addressed by its role;
* :class:`~feedbax.contracts.experiment_compile_lock.FigureRuntimeInputBinding`
  binds one figure runtime input authority;
* :class:`~feedbax.contracts.experiment_compile_lock.ReportParentBinding` binds
  one exact parent of a report;
* :class:`~feedbax.contracts.experiment_compile_lock.CheckpointInitializationBinding`
  initializes a training row, and training is never executed here, so reaching
  it from an executable node is a refusal rather than a silent drop.

The ``role`` a bound :class:`~feedbax.contracts.manifest.ParentRef` carries is
the binding's own addressing string. The ref's ``kind`` and ``id`` are never
taken from the binding: those name real bytes, and only an admitted receipt
supplies them.

## A matrix binds per row, because a matrix does not execute

An evaluation run matrix is not one execution; its rows are. So its required
inputs are not bound as the matrix's own ``inputs`` but as named *staged
parents*: authenticated once here, and injected into the base parameters every
materialized row inherits. Execution resolves them through the staged execution
context the environment declares, which is the only thing that knows where the
bound bytes actually live — so a matrix carrying staged parents refuses to run
in an environment that declares no such context.

## A bundle binds by identity, and a composition is still a figure

Two layers compile into two products each, and the second member of each pair
lowers here as what it actually is rather than as a new kind of thing.

An analysis *bundle* is to its layer what a matrix is to the evaluation layer:
one document whose stages are the executions. Its roots are the set of manifests
its own predicate selects, and a bundle names no role for them, so the closure's
required edges bind by manifest identity — the bound ids constrain the selection
and a selection that is not exactly that set refuses. Nothing invents a role
structure a bundle does not have.

A figure *composition* is authored figure identity, not a second execution: it
resolves to an ordinary current figure spec, and it therefore lowers to the same
figure node request a direct figure does, carrying the exact compiled bytes,
because those bytes are the authored identity its receipt is addressed by.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from feedbax.analysis.exact_parents import (
    STAGED_EXACT_PARENTS_SCHEMA_ID,
    STAGED_EXACT_PARENTS_SCHEMA_VERSION,
    StagedExactParents,
)
from feedbax.analysis.bundles import AnalysisBundleSpec
from feedbax.analysis.fulfillment_adapters import (
    AnalysisBundleNodeRequest,
    AnalysisNodeRequest,
    EvaluationMatrixNodeRequest,
    EvaluationNodeRequest,
    FigureNodeRequest,
    NodeRequest,
    ReportNodeRequest,
)
from feedbax.analysis.fulfillment_derivation import (
    COMPILED_PRODUCT_KINDS,
    CompiledEnvelope,
    FulfillmentDerivationError,
)
from feedbax.contracts.experiment_compile_lock import (
    AnalysisInputBinding,
    CheckpointInitializationBinding,
    EvaluationSubjectBinding,
    FigureRuntimeInputBinding,
    ReportParentBinding,
)
from feedbax.contracts.analysis_bundle_composition import ANALYSIS_BUNDLE_SPEC_SCHEMA_ID
from feedbax.contracts.figures import (
    FIGURE_COMPOSITION_SPEC_SCHEMA_ID,
    FIGURE_SPEC_SCHEMA_ID,
    FigureCompositionSpec,
)
from feedbax.contracts.manifest import (
    ANALYSIS_RUN_SPEC_SCHEMA_ID,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID,
    EVALUATION_RUN_SPEC_SCHEMA_ID,
    REPORT_SPEC_SCHEMA_ID,
    AnalysisRunSpec,
    EvaluationRunSpec,
    ParentRef,
    ReportSpec,
    StagedEvaluationPrerequisite,
)
from feedbax.contracts.staged_execution import validate_staged_binding_name

if TYPE_CHECKING:  # pragma: no cover - typing only
    from feedbax.analysis.fulfillment_driver import ClosureNode, NodeBinding


class NodeLoweringError(FulfillmentDerivationError):
    """A compiled node cannot be lowered into a node request it would execute."""


class BoundaryNodeLoweringError(NodeLoweringError):
    """A boundary node reached lowering, which only an executable node may."""


def binding_role(consumer: Any, *, ref: str) -> str:
    """Return the ``ParentRef`` role one closed consumer binding addresses by.

    The binding says how the *consumer* names the input. It never says what the
    bound bytes are: the receipt supplies the kind and the id.
    """
    if isinstance(consumer, EvaluationSubjectBinding):
        return consumer.subject_id
    if isinstance(consumer, AnalysisInputBinding):
        return consumer.role
    if isinstance(consumer, FigureRuntimeInputBinding):
        return consumer.input_role
    if isinstance(consumer, ReportParentBinding):
        return consumer.parent_id
    if isinstance(consumer, CheckpointInitializationBinding):
        raise NodeLoweringError(
            f"{ref} binds a checkpoint initialization for row {consumer.row_id!r}, which "
            "initializes a training row. Training is launched through its own orchestration "
            "entrypoint and is never executed by artifact fulfillment, so this binding cannot "
            "appear on an executable node."
        )
    raise NodeLoweringError(
        f"{ref} carries the consumer binding {type(consumer).__name__}, which this build does "
        "not bind; the consumer union is closed and every member has exactly one meaning here"
    )


def _require_unbound_inputs(document: Mapping[str, Any], *, ref: str) -> None:
    """Refuse a compiled spec that already declares inputs it cannot have.

    Authenticating an input takes a run, so a compile plan never authors one. A
    document that carries ``inputs`` anyway would be bound twice — once by
    whoever wrote them and once from the lock — and the two would silently
    disagree.
    """
    declared = document.get("inputs")
    if declared:
        raise NodeLoweringError(
            f"{ref} declares {len(list(declared))} input(s) in its compiled document; a "
            "compiled document is a plan and cannot authenticate an input. Inputs are bound "
            "from the compile lock's typed references, which is the single place they are "
            "stated."
        )


def bound_parents(
    node: "ClosureNode", *, binding: "NodeBinding"
) -> tuple[tuple[ParentRef, ...], tuple[str, ...]]:
    """Return the authenticated parents one node binds, and the roles they fill.

    Order is the compile lock's own reference order, so two walks of one closure
    bind identically. Every required reference contributes exactly one parent;
    an inapplicable reference contributes none, because it binds nothing.
    """
    compiled = node.compiled
    ref = str(compiled.lock_path)
    edges = {edge.role_path: edge for edge in binding.plan.input_edges(node.key)}
    parents: list[ParentRef] = []
    roles: list[str] = []
    for reference in compiled.plan_edge_references():
        role_path = tuple(str(reference.role_path).split("."))
        edge = edges.get(role_path)
        if edge is None:
            raise NodeLoweringError(
                f"{ref} references role {reference.role_path!r} but the plan carries no edge "
                f"for {node.key.text} at that role; the plan and the lock disagree"
            )
        if edge.status != "required":
            continue
        role = binding_role(reference.consumer, ref=ref)
        parents.append(binding.parent_ref(edge, role=role))
        roles.append(role)
    return tuple(parents), tuple(roles)


def _exact_parents(
    node: "ClosureNode", *, binding: "NodeBinding"
) -> StagedExactParents | None:
    """Bind one node's required inputs to the locations they execute from."""
    compiled = node.compiled
    ref = str(compiled.lock_path)
    edges = {edge.role_path: edge for edge in binding.plan.input_edges(node.key)}
    entries = []
    for reference in compiled.plan_edge_references():
        edge = edges.get(tuple(str(reference.role_path).split(".")))
        if edge is None or edge.status != "required":
            continue
        entries.append(
            binding.exact_parent_entry(edge, role=binding_role(reference.consumer, ref=ref))
        )
    if not entries:
        return None
    return StagedExactParents(
        schema_id=STAGED_EXACT_PARENTS_SCHEMA_ID,
        schema_version=STAGED_EXACT_PARENTS_SCHEMA_VERSION,
        parents=entries,
    )


def _document(node: "ClosureNode") -> dict[str, Any]:
    return dict(node.compiled.document)


def _lower_evaluation(node: "ClosureNode", *, binding: "NodeBinding") -> NodeRequest:
    document = _document(node)
    _require_unbound_inputs(document, ref=str(node.compiled.document_path))
    parents, _roles = bound_parents(node, binding=binding)
    spec = EvaluationRunSpec.model_validate({**document, "inputs": [
        parent.model_dump(mode="json", exclude_none=True) for parent in parents
    ]})
    return EvaluationNodeRequest(node_key=node.key.text, spec=spec, order=node.order)


def _lower_evaluation_matrix(node: "ClosureNode", *, binding: "NodeBinding") -> NodeRequest:
    document = _document(node)
    if not binding.plan.required_edges(node.key):
        return EvaluationMatrixNodeRequest(
            node_key=node.key.text, matrix=document, order=node.order
        )
    return EvaluationMatrixNodeRequest(
        node_key=node.key.text,
        matrix=_matrix_with_staged_parents(node, document, binding=binding),
        order=node.order,
    )


def _matrix_with_staged_parents(
    node: "ClosureNode",
    document: Mapping[str, Any],
    *,
    binding: "NodeBinding",
) -> dict[str, Any]:
    """Bind one matrix's required inputs as staged parents every row consumes.

    A matrix does not execute; its rows do. So a required input is bound the way
    a matrix binds anything that reaches a row: as a named staged prerequisite,
    authenticated once at the matrix level and injected into the base parameters
    every materialized row inherits. Execution then resolves it through the
    staged execution context the environment declares, which is the one place
    that knows where the bound bytes actually live.

    The binding *name* is the role the consumer binding addresses — the subject
    id for an evaluation subject — so nothing here invents an addressing string.
    """
    ref = str(node.compiled.lock_path)
    declared = document.get("staged_parents")
    if declared:
        raise NodeLoweringError(
            f"{ref} binds {len(list(declared))} staged parent(s) in its compiled document "
            "while its plan also declares required inputs. A compiled document is a plan "
            "and cannot authenticate a parent; staged parents are bound from the compile "
            "lock's typed references, which is the single place they are stated."
        )
    base = document.get("base")
    if not isinstance(base, Mapping) or "evaluation_type" not in base:
        raise NodeLoweringError(
            f"{ref} declares required input(s) on an evaluation run matrix whose base is "
            "content-pinned rather than stated inline. A staged prerequisite is injected "
            "into the base parameters every row inherits, and pinned bytes cannot receive "
            "one without breaking their pin. State the matrix base inline, or bind the "
            "subject in the pinned base itself."
        )
    parents, roles = bound_parents(node, binding=binding)
    prerequisites: dict[str, dict[str, Any]] = {}
    for parent, role in zip(parents, roles, strict=True):
        try:
            name = validate_staged_binding_name(role)
        except ValueError as exc:
            raise NodeLoweringError(
                f"{ref} binds a required input at role {role!r}, which is not a staged "
                f"execution binding name: {exc}"
            ) from exc
        if name in prerequisites:
            raise NodeLoweringError(
                f"{ref} binds two required inputs to the staged parent name {name!r}; a "
                "staged binding names exactly one authenticated parent"
            )
        prerequisites[name] = StagedEvaluationPrerequisite(parent=parent).model_dump(
            mode="json", exclude_none=True
        )
    bound = deepcopy(dict(document))
    bound["staged_parents"] = prerequisites
    bound_base = deepcopy(dict(base))
    params = dict(bound_base.get("params") or {})
    staged = dict(params.get("staged_prerequisites") or {})
    collisions = sorted(set(staged) & set(prerequisites))
    if collisions:
        raise NodeLoweringError(
            f"{ref} already states staged prerequisites {collisions} in its compiled base "
            "parameters; a compile plan cannot authenticate one, and binding it twice "
            "would let the two disagree"
        )
    staged.update(prerequisites)
    params["staged_prerequisites"] = staged
    bound_base["params"] = params
    bound["base"] = bound_base
    return bound


def _lower_analysis(node: "ClosureNode", *, binding: "NodeBinding") -> NodeRequest:
    document = _document(node)
    _require_unbound_inputs(document, ref=str(node.compiled.document_path))
    parents, _roles = bound_parents(node, binding=binding)
    spec = AnalysisRunSpec.model_validate({**document, "inputs": [
        parent.model_dump(mode="json", exclude_none=True) for parent in parents
    ]})
    return AnalysisNodeRequest(node_key=node.key.text, spec=spec, order=node.order)


def _validated_document(model: Any, document: Mapping[str, Any], *, ref: str) -> Any:
    """Return one compiled document as the Feedbax model its schema identity names.

    The document reached this lowering because the layer table recognized its
    ``schema_id``. Validating it against that identity's model is what makes the
    recognition mean something: a document that declares the identity but is not
    a member of it refuses here, named, rather than failing later inside an
    executor that assumed the table had already decided.
    """
    try:
        return model.model_validate(dict(document))
    except (ValidationError, ValueError) as exc:
        raise NodeLoweringError(
            f"{ref} declares schema_id {document.get('schema_id')!r} but is not a valid "
            f"{model.__name__}: {exc}"
        ) from exc


def _lower_analysis_bundle(node: "ClosureNode", *, binding: "NodeBinding") -> NodeRequest:
    """Lower one compiled analysis bundle into the node that drives its own plan.

    A bundle addresses its root inputs as the set of manifests its predicate
    selects, not by role, so the required edges are bound by manifest identity
    and the role each consumer binding names addresses nothing further. Nothing
    is invented to fill that gap: the ids constrain the selection exactly, and
    the adapter refuses a selection that is not the bound set.
    """
    document = _document(node)
    ref = str(node.compiled.document_path)
    bundle = _validated_document(AnalysisBundleSpec, document, ref=ref)
    parents, _roles = bound_parents(node, binding=binding)
    return AnalysisBundleNodeRequest(
        node_key=node.key.text,
        bundle=bundle,
        root_inputs=parents,
        order=node.order,
    )


def _lower_figure(node: "ClosureNode", *, binding: "NodeBinding") -> NodeRequest:
    document = _document(node)
    parents, _roles = bound_parents(node, binding=binding)
    return FigureNodeRequest(
        node_key=node.key.text,
        spec=document,
        runtime_inputs=parents if parents else None,
        order=node.order,
    )


def _lower_figure_composition(node: "ClosureNode", *, binding: "NodeBinding") -> NodeRequest:
    """Lower one compiled figure composition into the figure node it renders as.

    A composition is authored figure identity, not a second kind of execution:
    resolving it produces an ordinary current figure spec, and the receipt it
    earns is an ordinary figure receipt. So it lowers to the same node request a
    direct figure does, carrying the composition document itself — the exact
    compiled bytes, because those bytes are the figure's authored identity and
    re-serializing them through a model would address something else.

    The document is validated against its own composition model first. Resolving
    the parent chain happens at execution, where the repository the pins address
    is the environment's declaration rather than a guess made here.
    """
    document = _document(node)
    _validated_document(
        FigureCompositionSpec, document, ref=str(node.compiled.document_path)
    )
    parents, _roles = bound_parents(node, binding=binding)
    return FigureNodeRequest(
        node_key=node.key.text,
        spec=document,
        runtime_inputs=parents if parents else None,
        order=node.order,
    )


def _lower_report(node: "ClosureNode", *, binding: "NodeBinding") -> NodeRequest:
    document = _document(node)
    _require_unbound_inputs(document, ref=str(node.compiled.document_path))
    parents, _roles = bound_parents(node, binding=binding)
    exact = _exact_parents(node, binding=binding)
    spec = ReportSpec.model_validate({**document, "inputs": [
        parent.model_dump(mode="json", exclude_none=True) for parent in parents
    ]})
    return ReportNodeRequest(
        node_key=node.key.text, spec=spec, exact_parents=exact, order=node.order
    )


#: Compiled ``schema_id`` to the lowering that produces its node request. The
#: table is exhaustive over the executable members of
#: :data:`~feedbax.analysis.fulfillment_derivation.COMPILED_PRODUCT_KINDS`.
_LOWERINGS = {
    EVALUATION_RUN_SPEC_SCHEMA_ID: _lower_evaluation,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID: _lower_evaluation_matrix,
    ANALYSIS_RUN_SPEC_SCHEMA_ID: _lower_analysis,
    ANALYSIS_BUNDLE_SPEC_SCHEMA_ID: _lower_analysis_bundle,
    FIGURE_SPEC_SCHEMA_ID: _lower_figure,
    FIGURE_COMPOSITION_SPEC_SCHEMA_ID: _lower_figure_composition,
    REPORT_SPEC_SCHEMA_ID: _lower_report,
}


def lower_compiled_node(node: "ClosureNode", *, binding: "NodeBinding") -> NodeRequest:
    """Return the node request one compiled document's schema identity executes.

    Dispatch is on the compiled document's own ``schema_id``. A boundary node is
    refused rather than lowered: it is produced by another entrypoint, and
    preflight has already refused any closure that still names one, so reaching
    here means a closure was assembled without that proof.
    """
    compiled: CompiledEnvelope = node.compiled
    kind = compiled.kind
    if not kind.executable:
        raise BoundaryNodeLoweringError(
            f"{node.key.text} is a {kind.boundary!r} boundary node and is never executed by "
            "artifact fulfillment; produce it through its own entrypoint and quote the receipt"
        )
    lowering = _LOWERINGS.get(compiled.schema_id)
    if lowering is None:  # pragma: no cover - the two tables are kept exhaustive
        raise NodeLoweringError(
            f"{compiled.document_path} declares schema_id {compiled.schema_id!r}, which is a "
            f"planned layer product but has no lowering; supported={sorted(_LOWERINGS)}"
        )
    return lowering(node, binding=binding)


def supported_lowerings() -> tuple[str, ...]:
    """Return every compiled ``schema_id`` this build can execute, in order."""
    return tuple(
        schema_id for schema_id in COMPILED_PRODUCT_KINDS if schema_id in _LOWERINGS
    )


__all__ = [
    "BoundaryNodeLoweringError",
    "NodeLoweringError",
    "binding_role",
    "bound_parents",
    "lower_compiled_node",
    "supported_lowerings",
]
