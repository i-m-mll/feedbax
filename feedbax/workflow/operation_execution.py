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

## A figure input is bound and authorized from the same lock reference

Binding a figure's parent says *which manifest*; it says nothing about which
artifact of that manifest the figure reads, from which provider, at which media
type, or under which decoded payload identity. That second half is the input's
artifact contract, and it is carried by the same
:class:`~feedbax.contracts.experiment_compile_lock.FigureRuntimeInputBinding`
that names the role — so the runtime
:class:`~feedbax.contracts.figures.FigureInputRoleAuthority` is built from the
lock and from nothing else. Nothing is derived from a role name, from where a
receipt was written, or from the compiled figure document.

A row-expanded figure states the same contracts in its expansion request and
binds its per-row roles from produced custody, so the two statements are read
apart rather than merged: each is complete for the figures it covers, and reading
both over one figure would authorize one role twice. A figure that is not an
expansion whose lock binds an input with no contract is refused before any render
effect — rendering it would produce a figure carrying none of the data it names.

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

A matrix binds as many staged parents as its lock has typed references, one per
required role, and the binding *name* is always the role the consumer binding
addresses. A compiled document may restate a staged parent the lock binds — an
authored matrix that already names its prerequisite compiles into one — but it
never *authenticates* one: a restated entry must name exactly the parent the
lock authenticates, and an entry the lock does not bind at all is refused.
Non-authenticating fields of a restated entry, such as the artifact provider its
bytes are staged from, are the document's to state and are carried through.

A content-pinned base receives no injection, because injecting into pinned bytes
would break the pin they are pinned by. Such a matrix binds its authenticated
parents at the matrix level only, and the pinned rows must reference them
themselves; :func:`feedbax.analysis.evaluation.materialize_evaluation_run_matrix`
refuses per row when they do not.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from feedbax.analysis.exact_parents import (
    STAGED_EXACT_PARENTS_SCHEMA_ID,
    STAGED_EXACT_PARENTS_SCHEMA_VERSION,
    StagedExactParents,
)
from feedbax.analysis.bundles import AnalysisBundleSpec
from feedbax.analysis.execution_context import (
    EMPTY_STAGED_EXECUTION_CONTEXT,
    StagedExecutionContext,
)
from feedbax.analysis.fulfillment_adapters import (
    AnalysisBundleNodeRequest,
    AnalysisNodeRequest,
    EvaluationMatrixNodeRequest,
    EvaluationNodeRequest,
    FigureNodeRequest,
    NodeRequest,
    ReportNodeRequest,
)
from feedbax.workflow.derivation import (
    COMPILED_PRODUCT_KINDS,
    CompiledEnvelope,
    WorkflowDerivationError,
)
from feedbax.analysis.fulfillment_row_custody import (
    read_row_expansion_record,
    resolve_row_custody_overlay,
)
from feedbax.analysis.manifest_inputs import restated_parent_differences
from feedbax.contracts.experiment_compile_lock import (
    EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V1,
    AnalysisInputBinding,
    AnalysisReceiptSetBinding,
    CheckpointInitializationBinding,
    EvaluationSubjectBinding,
    FigureRuntimeInputBinding,
    ReportParentBinding,
)
from feedbax.contracts.experiment_envelope_dialect import (
    EXPERIMENT_ENVELOPE_FAMILY,
    EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V5,
    EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V6,
)
from feedbax.contracts.analysis_bundle_composition import ANALYSIS_BUNDLE_SPEC_SCHEMA_ID
from feedbax.contracts.figures import (
    FIGURE_COMPOSITION_SPEC_SCHEMA_ID,
    FIGURE_SPEC_SCHEMA_ID,
    FigureCompositionSpec,
    FigureInputAuthoritySpec,
    FigureInputRoleAuthority,
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
    from feedbax.workflow.execution import PreparedOperation, NodeBinding


class NodeLoweringError(WorkflowDerivationError):
    """A compiled node cannot be lowered into a node request it would execute."""


def binding_role(consumer: Any, *, ref: str) -> str:
    """Return the ``ParentRef`` role one closed consumer binding addresses by.

    The binding says how the *consumer* names the input. It never says what the
    bound bytes are: the receipt supplies the kind and the id.
    """
    if isinstance(consumer, EvaluationSubjectBinding):
        return consumer.subject_id
    if isinstance(consumer, (AnalysisInputBinding, AnalysisReceiptSetBinding)):
        return consumer.role
    if isinstance(consumer, FigureRuntimeInputBinding):
        return consumer.input_role
    if isinstance(consumer, ReportParentBinding):
        return consumer.parent_id
    if isinstance(consumer, CheckpointInitializationBinding):
        raise NodeLoweringError(
            f"{ref} binds a checkpoint initialization for row {consumer.row_id!r}, which "
            "initializes a training row. Training is launched through its own orchestration "
            "entrypoint and is not a local artifact operation, so this binding cannot "
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
    node: "PreparedOperation", *, binding: "NodeBinding"
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
        bound = binding.parent_refs(edge, role=role)
        parents.extend(bound)
        roles.extend(role for _ in bound)
    return tuple(parents), tuple(roles)


def _exact_parents(
    node: "PreparedOperation", *, binding: "NodeBinding"
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
        entries.extend(
            binding.exact_parent_entries(
                edge, role=binding_role(reference.consumer, ref=ref)
            )
        )
    if not entries:
        return None
    return StagedExactParents(
        schema_id=STAGED_EXACT_PARENTS_SCHEMA_ID,
        schema_version=STAGED_EXACT_PARENTS_SCHEMA_VERSION,
        parents=entries,
    )


def _node_execution_context(
    node: "PreparedOperation",
    *,
    binding: "NodeBinding",
    extra_parents: Sequence[ParentRef] = (),
) -> "StagedExecutionContext | None":
    """Return the staged context this node executes under, or ``None``.

    The unit is the node, not the run. A node binds the parents its lock states,
    and those parents were located by the same resolution that bound them — an
    in-closure producer's admitted receipt beneath the receipt root, or the one
    declared authority an external reference resolved through. Handing execution
    a context carrying exactly those locations is what lets a recipe read a
    parent's bytes without re-deriving where they are, and re-deriving is the
    step that could answer differently than the walk did.

    It is built here, in the lowering, because the lowering is the one place the
    same answer is reconstructed for every operation over a closure: ordinary
    fulfillment, cached admission, rebuild-as-verification, and repair all lower
    each node again. A context assembled during the walk and discarded after it
    would leave every later operation resolving parents a different way.

    ``extra_parents`` are complete authenticated parents no plan edge bound — a
    row-expanded figure's per-row custody bindings — and they are located by the
    same exactly-one authority search rather than assumed to sit beneath the
    receipt root.

    ``None`` means the run declared no staged bindings at all, which is the
    cold-start case and is left exactly as it was.
    """
    compiled = node.compiled
    ref = str(compiled.lock_path)
    edges = {edge.role_path: edge for edge in binding.plan.input_edges(node.key)}
    locations = []
    for reference in compiled.plan_edge_references():
        edge = edges.get(tuple(str(reference.role_path).split(".")))
        if edge is None or edge.status != "required":
            continue
        locations.extend(
            binding.parent_locations(
                edge, role=binding_role(reference.consumer, ref=ref)
            )
        )
    if binding.environment.execution_context is EMPTY_STAGED_EXECUTION_CONTEXT:
        # Locating the extra parents would search authorities that were never
        # declared, so the cold-start answer is settled before that search.
        return binding.node_execution_context(locations)
    locations.extend(binding.authenticated_parent_location(parent) for parent in extra_parents)
    return binding.node_execution_context(locations)


def _document(node: "PreparedOperation") -> dict[str, Any]:
    return dict(node.compiled.document)


def _lower_evaluation(node: "PreparedOperation", *, binding: "NodeBinding") -> NodeRequest:
    document = _document(node)
    _require_unbound_inputs(document, ref=str(node.compiled.document_path))
    parents, _roles = bound_parents(node, binding=binding)
    spec = EvaluationRunSpec.model_validate(
        {
            **document,
            "inputs": [parent.model_dump(mode="json", exclude_none=True) for parent in parents],
        }
    )
    return EvaluationNodeRequest(
        node_key=node.key.text,
        spec=spec,
        order=node.order,
        execution_context=_node_execution_context(node, binding=binding),
    )


def _lower_evaluation_matrix(node: "PreparedOperation", *, binding: "NodeBinding") -> NodeRequest:
    document = _document(node)
    context = _node_execution_context(node, binding=binding)
    if not binding.plan.required_edges(node.key):
        return EvaluationMatrixNodeRequest(
            node_key=node.key.text,
            matrix=document,
            order=node.order,
            execution_context=context,
        )
    return EvaluationMatrixNodeRequest(
        node_key=node.key.text,
        matrix=_matrix_with_staged_parents(node, document, binding=binding),
        order=node.order,
        execution_context=context,
    )


def _matrix_with_staged_parents(
    node: "PreparedOperation",
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
    As many parents are bound as the lock has required references.

    A content-pinned base receives no injection: its bytes are pinned, and the
    rows it produces must reference the bound parents themselves. Materializing
    the matrix refuses per row when they do not, so the parent is never silently
    unconsumed.
    """
    ref = str(node.compiled.lock_path)
    prerequisites = _staged_prerequisites(node, document, binding=binding)
    bound = deepcopy(dict(document))
    bound["staged_parents"] = prerequisites
    base = document.get("base")
    if isinstance(base, Mapping) and "evaluation_type" in base:
        bound["base"] = _base_with_staged_prerequisites(base, prerequisites, ref=ref)
        return bound
    if isinstance(base, Mapping) and "ref" in base and "sha256" in base:
        return bound
    raise NodeLoweringError(
        f"{ref} declares required input(s) on an evaluation run matrix whose base is "
        "neither an inline evaluation run spec nor a content-pinned document. A staged "
        "parent reaches a row through one of those two bases and through nothing else."
    )


def _declared_staged_parents(
    document: Mapping[str, Any], *, ref: str
) -> dict[str, StagedEvaluationPrerequisite]:
    """Return the staged parents one compiled matrix document already states."""
    declared = document.get("staged_parents")
    if not declared:
        return {}
    if not isinstance(declared, Mapping):
        raise NodeLoweringError(
            f"{ref} states staged_parents that are not a mapping of binding name to "
            "prerequisite; a staged parent is addressed by exactly one binding name"
        )
    try:
        return {
            str(name): StagedEvaluationPrerequisite.model_validate(value)
            for name, value in declared.items()
        }
    except ValidationError as exc:
        raise NodeLoweringError(
            f"{ref} states a staged parent this build cannot read: {exc}"
        ) from exc


def _staged_prerequisites(
    node: "PreparedOperation",
    document: Mapping[str, Any],
    *,
    binding: "NodeBinding",
) -> dict[str, dict[str, Any]]:
    """Return every staged prerequisite one matrix's compile lock authenticates.

    The lock is the sole authority for *which* parent each name binds. A name the
    document also states must name that same artifact; a name only the document
    states has no authenticated source and refuses.

    "The same artifact" is :func:`restated_parent_differences`' contract, and it
    deliberately excludes the ref's ``role``. See that function for why.
    """
    ref = str(node.compiled.lock_path)
    stated = _declared_staged_parents(document, ref=ref)
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
        declared = stated.pop(name, None)
        if declared is not None:
            differences = restated_parent_differences(declared.parent, parent)
            if differences:
                raise NodeLoweringError(
                    f"{ref} restates the staged parent {name!r} in its compiled document as a "
                    f"different artifact than its compile lock authenticates: "
                    f"{'; '.join(differences)}. A compiled document is a plan and cannot "
                    "authenticate a parent, so a restated staged parent may only restate the "
                    "artifact the lock binds."
                )
        prerequisites[name] = StagedEvaluationPrerequisite(
            parent=parent,
            artifact_provider=None if declared is None else declared.artifact_provider,
        ).model_dump(mode="json", exclude_none=True)
    if stated:
        raise NodeLoweringError(
            f"{ref} states the staged parent(s) {sorted(stated)} in its compiled document, "
            "which its compile lock does not bind. A compiled document is a plan and cannot "
            "authenticate a parent; staged parents are bound from the compile lock's typed "
            "references, which is the single place they are stated."
        )
    return prerequisites


def _base_with_staged_prerequisites(
    base: Mapping[str, Any],
    prerequisites: Mapping[str, dict[str, Any]],
    *,
    ref: str,
) -> dict[str, Any]:
    """Inject the bound prerequisites into the base parameters every row inherits."""
    bound_base = deepcopy(dict(base))
    params = dict(bound_base.get("params") or {})
    staged = dict(params.get("staged_prerequisites") or {})
    for name, prerequisite in prerequisites.items():
        stated = staged.get(name)
        if stated is not None:
            differences = _stated_prerequisite_differences(stated, prerequisite, ref=ref)
            if differences:
                raise NodeLoweringError(
                    f"{ref} already states the staged prerequisite {name!r} in its compiled "
                    f"base parameters, naming a different artifact than the one its compile "
                    f"lock authenticates: {'; '.join(differences)}. A compiled document is a "
                    "plan and cannot authenticate a parent, and binding it twice would let "
                    "the two disagree."
                )
        staged[name] = prerequisite
    params["staged_prerequisites"] = staged
    bound_base["params"] = params
    return bound_base


def _stated_prerequisite_differences(
    stated: Any, bound: Mapping[str, Any], *, ref: str
) -> tuple[str, ...]:
    """Return how one stated base prerequisite disagrees with the bound one."""
    try:
        stated_model = StagedEvaluationPrerequisite.model_validate(stated)
        bound_model = StagedEvaluationPrerequisite.model_validate(bound)
    except ValidationError as exc:
        raise NodeLoweringError(
            f"{ref} states a base staged prerequisite this build cannot read: {exc}"
        ) from exc
    return restated_parent_differences(stated_model.parent, bound_model.parent)


def _lower_analysis(node: "PreparedOperation", *, binding: "NodeBinding") -> NodeRequest:
    document = _document(node)
    _require_unbound_inputs(document, ref=str(node.compiled.document_path))
    parents, _roles = bound_parents(node, binding=binding)
    spec = AnalysisRunSpec.model_validate(
        {
            **document,
            "inputs": [parent.model_dump(mode="json", exclude_none=True) for parent in parents],
        }
    )
    return AnalysisNodeRequest(
        node_key=node.key.text,
        spec=spec,
        order=node.order,
        execution_context=_node_execution_context(node, binding=binding),
    )


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


def _lower_analysis_bundle(node: "PreparedOperation", *, binding: "NodeBinding") -> NodeRequest:
    """Lower one compiled analysis bundle into the node that drives its own plan.

    A bundle addresses its root inputs as the set of manifests its predicate
    selects, not by role, so the required edges are bound by manifest identity
    and the role each consumer binding names addresses nothing further. Nothing
    is invented to fill that gap: the ids constrain the selection exactly, and
    the adapter refuses a selection that is not the bound set.

    Whether the lock declares a root set at all is read from the lock, not
    inferred from how many parents came back. A lock with no required reference
    leaves the bundle's authored predicate as the only statement about its roots,
    and that is passed on as ``None`` — the explicit "nothing was authenticated"
    case — rather than as an empty set that would look like a declared one.
    """
    document = _document(node)
    ref = str(node.compiled.document_path)
    bundle = _validated_document(AnalysisBundleSpec, document, ref=ref)
    declares_roots = bool(binding.plan.required_edges(node.key))
    parents, _roles = bound_parents(node, binding=binding)
    return AnalysisBundleNodeRequest(
        node_key=node.key.text,
        bundle=bundle,
        root_inputs=parents if declares_roots else None,
        order=node.order,
        # A bundle's context carries the run's declared authorities and *no*
        # parent locations. Its roots are bound by manifest identity rather than
        # by role, and the bundle re-roles every one of them before its stages
        # consume it, so locating them here under the lock's roles would address
        # the same bytes by a reference no stage ever uses — and, since a staged
        # context permits one reference per location, would collide with the
        # bundle's own. The bundle locates the refs it actually executes with.
        execution_context=binding.node_execution_context(()),
    )


def _contract_unstatable_grammar(compiled: CompiledEnvelope) -> str | None:
    """Return the grammar that could not state a figure input contract, or ``None``.

    Two version facts decide it, and both are read from the lock rather than
    inferred: the lock's own schema version, which decides whether a
    ``figure_runtime_input`` binding *has* a contract field, and the envelope
    schema the lock records, which decides whether the author had vocabulary to
    fill one. A missing contract means "not decided" under either of those and
    "decided against" under neither.
    """
    lock = compiled.lock
    lock_version = lock.get("schema_version")
    if lock_version == EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V1:
        return f"compile lock {EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V1!r}"
    envelope = lock.get("envelope")
    envelope_schema = envelope.get("schema") if isinstance(envelope, Mapping) else None
    if envelope_schema not in (
        EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V5,
        EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V6,
    ) and isinstance(envelope_schema, str):
        if envelope_schema.startswith(f"{EXPERIMENT_ENVELOPE_FAMILY}."):
            return f"envelope grammar {envelope_schema!r}"
    return None


def _figure_input_authorities(
    node: "PreparedOperation",
    parents: Sequence[ParentRef],
    *,
    binding: "NodeBinding",
) -> tuple[FigureInputAuthoritySpec, ...]:
    """Return the runtime authority each plan-bound figure input reads under.

    A figure input that reads an artifact needs an authority saying *which*
    artifact, from which provider, at which media type, and under which decoded
    payload identity — otherwise the figure names a parent and reads nothing out
    of it. The compile lock's ``figure_runtime_input`` binding is where that
    contract is stated, so it is the only thing consulted here: nothing is
    derived from a role name, from where a receipt happened to be written, or
    from the figure document.

    The authority addresses its parent by role rather than by value, because the
    parent's own record already carries that role and a second copy of the same
    reference is a second thing to keep in agreement. Resolution back to the
    exact parent happens once, at
    :meth:`~feedbax.contracts.figures.FigureInputRoleAuthority.resolve_parent`,
    which refuses a role no declared input carries and a role two of them do.

    A binding that states no contract is read against the grammars the lock and
    its envelope declare. Where either of those grammars *could not* state a
    contract — a ``feedbax.spec.experiment_compile_lock.v1`` lock, or an envelope
    authored before ``feedbax.experiment_envelope.v5`` — the absence is not a
    decision, it is the defect: the reference was recorded, the compiled figure
    was emitted with no authority at all, and rendering it would produce a figure
    carrying none of the data it names. That is refused, before effects, with the
    re-authoring it needs. Where both grammars could have stated one and neither
    did, the input is bound as provenance and read from no artifact, which is a
    statement an author is entitled to make.
    """
    compiled = node.compiled
    ref = str(compiled.lock_path)
    unstatable = _contract_unstatable_grammar(compiled)
    edges = {edge.role_path: edge for edge in binding.plan.input_edges(node.key)}
    bound_roles = {parent.role for parent in parents}
    authorities: list[FigureInputAuthoritySpec] = []
    for reference in compiled.plan_edge_references():
        # A not-applicable reference states a role nothing fills and carries no
        # consumer at all; it authorizes nothing and is skipped by kind.
        consumer = getattr(reference, "consumer", None)
        if not isinstance(consumer, FigureRuntimeInputBinding):
            continue
        edge = edges.get(tuple(str(reference.role_path).split(".")))
        if edge is None or edge.status != "required":
            continue
        if consumer.contract is None:
            if unstatable is None:
                continue
            raise NodeLoweringError(
                f"{ref} binds the figure input role {consumer.input_role!r} under "
                f"{unstatable}, which has no way to state the artifact contract the bound "
                "manifest must satisfy. The reference is recorded and the compiled figure "
                "carries no input authority at all, so rendering it would produce a figure "
                f"holding none of the data it names. Re-author the figure envelope at "
                f"{EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V5!r}, stating each input's artifact "
                "role, provider, media type, and explicit payload name, and recompile"
            )
        if consumer.input_role not in bound_roles:
            raise NodeLoweringError(
                f"{ref} states an artifact contract for the figure input role "
                f"{consumer.input_role!r}, which no bound parent fills; the plan and the "
                "lock disagree about which inputs this figure has"
            )
        authorities.append(
            FigureInputRoleAuthority(
                input_role=consumer.input_role,
                artifact_payloads=[consumer.contract.artifact_payload(consumer.input_role)],
            )
        )
    return tuple(authorities)


def _lower_figure(node: "PreparedOperation", *, binding: "NodeBinding") -> NodeRequest:
    document = _document(node)
    parents, _roles = bound_parents(node, binding=binding)
    # A row-expanded figure's per-row roles are filled from the row index's
    # produced custody, which the lock names and this resolves; every other
    # figure returns None here and binds exactly as before.
    overlay = resolve_row_custody_overlay(node.compiled, repo_root=binding.environment.repo_root)
    # A row expansion declares its roles in its own expansion request and binds
    # them from custody. The plan-edge contract is how a figure that is not an
    # expansion — a root figure — states the same thing, so the two are read
    # apart: reading both over one figure would authorize a role twice, and each
    # is the whole statement for the figures it applies to.
    authorities: tuple[FigureInputAuthoritySpec, ...] = (
        (overlay.authorities if overlay is not None else ())
        if read_row_expansion_record(node.compiled) is not None
        else _figure_input_authorities(node, parents, binding=binding)
    )
    if overlay is not None:
        parents = (*parents, *overlay.inputs)
    return FigureNodeRequest(
        node_key=node.key.text,
        spec=document,
        runtime_inputs=parents if parents else None,
        runtime_input_authorities=authorities if authorities else None,
        order=node.order,
        execution_context=_node_execution_context(
            node,
            binding=binding,
            extra_parents=overlay.inputs if overlay is not None else (),
        ),
    )


def _lower_figure_composition(node: "PreparedOperation", *, binding: "NodeBinding") -> NodeRequest:
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
    _validated_document(FigureCompositionSpec, document, ref=str(node.compiled.document_path))
    parents, _roles = bound_parents(node, binding=binding)
    return FigureNodeRequest(
        node_key=node.key.text,
        spec=document,
        runtime_inputs=parents if parents else None,
        order=node.order,
        execution_context=_node_execution_context(node, binding=binding),
    )


def _lower_report(node: "PreparedOperation", *, binding: "NodeBinding") -> NodeRequest:
    document = _document(node)
    _require_unbound_inputs(document, ref=str(node.compiled.document_path))
    parents, _roles = bound_parents(node, binding=binding)
    exact = _exact_parents(node, binding=binding)
    spec = ReportSpec.model_validate(
        {
            **document,
            "inputs": [parent.model_dump(mode="json", exclude_none=True) for parent in parents],
        }
    )
    return ReportNodeRequest(
        node_key=node.key.text,
        spec=spec,
        exact_parents=exact,
        order=node.order,
        execution_context=_node_execution_context(node, binding=binding),
    )


#: Compiled ``schema_id`` to the lowering that produces its node request. The
#: table is exhaustive over the executable members of
#: :data:`~feedbax.workflow.derivation.COMPILED_PRODUCT_KINDS`.
_LOWERINGS = {
    EVALUATION_RUN_SPEC_SCHEMA_ID: _lower_evaluation,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID: _lower_evaluation_matrix,
    ANALYSIS_RUN_SPEC_SCHEMA_ID: _lower_analysis,
    ANALYSIS_BUNDLE_SPEC_SCHEMA_ID: _lower_analysis_bundle,
    FIGURE_SPEC_SCHEMA_ID: _lower_figure,
    FIGURE_COMPOSITION_SPEC_SCHEMA_ID: _lower_figure_composition,
    REPORT_SPEC_SCHEMA_ID: _lower_report,
}


def lower_compiled_node(node: "PreparedOperation", *, binding: "NodeBinding") -> NodeRequest:
    """Return the node request one compiled document's schema identity executes.

    Dispatch is on the compiled document's own ``schema_id``. External
    operations are refused at workflow preparation and reach invocation later;
    this local lowering table therefore contains only locally realizable kinds.
    """
    compiled: CompiledEnvelope = node.compiled
    lowering = _LOWERINGS.get(compiled.schema_id)
    if lowering is None:  # pragma: no cover - the two tables are kept exhaustive
        raise NodeLoweringError(
            f"{compiled.document_path} declares schema_id {compiled.schema_id!r}, which is a "
            f"planned layer product but has no lowering; supported={sorted(_LOWERINGS)}"
        )
    return lowering(node, binding=binding)


def supported_lowerings() -> tuple[str, ...]:
    """Return every compiled ``schema_id`` this build can execute, in order."""
    return tuple(schema_id for schema_id in COMPILED_PRODUCT_KINDS if schema_id in _LOWERINGS)


__all__ = [
    "NodeLoweringError",
    "binding_role",
    "bound_parents",
    "restated_parent_differences",
    "lower_compiled_node",
    "supported_lowerings",
]
