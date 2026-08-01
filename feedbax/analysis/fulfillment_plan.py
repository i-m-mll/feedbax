"""The fulfillment plan: one target's dependency closure, stated before any run.

A named artifact — a report, a figure, an analysis, an evaluation — is produced
by materializing the transitive closure of what it depends on. This module
states that closure as a document. A plan says what must exist; whether it does
is a separate question, answered by whoever executes the plan
(:mod:`feedbax.analysis.fulfillment_driver`). Nothing here reads a receipt root,
queries a manifest index, or looks for a manifest.

## Logical addressing

A node is addressed **logically** — by the ``layer`` it belongs to and the
``envelope`` (the named declaration) that produces it — and never by a manifest
id, because a dependent node's id is mintable only once its parents' receipts
bind. :class:`LogicalKey` is that address, and its :attr:`~LogicalKey.text` form
quotes the envelope so no name can forge the separator.

An input the closure does not produce is not a node. Already-produced receipts —
a run id a previous execution minted, an authenticated manifest pin, a
custody-pinned checkpoint — ride on the edge that names them, as ``external``
records, so the plan never implies there is something to execute for them.

## Typed applicability

Every input edge carries a decision: ``status`` (``required`` or
``not_applicable``), a ``reason``, and the ``basis`` on which the status was
reached (``authored`` or ``compiler_rule``). Two bases, and no third:

* **authored** — the declaration itself states the input is not applicable, so
  the decision is already materialized and nothing further is owed.
* **compiler_rule** — a closed rule over the declaration proves the target
  provides no such input. The rule is *named* on the edge
  (:attr:`PlanEdge.rule`); which rules exist is the lowering project's data,
  never a constant of this kernel.

A missing receipt is never one of these. Not-yet-executed, previously-failed,
wrong-root, and corrupt are outcomes of fulfillment, not evidence about the
science, and none of them reaches this module. Only a ``compiler_rule`` decision
may be materialized into a declaration by :func:`apply_certified_omissions`.

## What this kernel does not decide

It does not know what a layer is, how a declaration compiles, which node kinds
exist, what a rule name means, or how a project's document expresses an
omission. Those reach it as data on the plan and as the callables named in this
module's protocols.

Stability note: this module is not part of the owner-ratified downstream
inventory in ``docs/design/downstream_interface_stability.md``. The
``feedbax.fulfillment.plan`` family nonetheless carries explicit schema identity
and refuses unsupported versions with no migration fallback, because an emitted
plan is a durable emitted spec that crosses a process boundary.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar
from urllib.parse import quote, unquote

from feedbax.contracts.manifest import canonical_json_bytes


FULFILLMENT_PLAN_SCHEMA_ID = "feedbax.fulfillment.plan"
FULFILLMENT_PLAN_SCHEMA_VERSION_V1 = "feedbax.fulfillment.plan.v1"
FULFILLMENT_PLAN_SCHEMA_VERSION = FULFILLMENT_PLAN_SCHEMA_VERSION_V1

#: Enumerated, never inferred. A document from any other version is refused.
FULFILLMENT_PLAN_SUPPORTED_SCHEMA_VERSIONS = (FULFILLMENT_PLAN_SCHEMA_VERSION_V1,)

#: Versions this loader accepts, mapped to the version they migrate to. Empty
#: while v1 is the only version: an unknown version fails closed rather than
#: being read through a compatibility shim.
FULFILLMENT_PLAN_MIGRATION_TABLE: dict[str, str] = {}

#: The two statuses an input edge can carry, and the two bases a status may be
#: reached on. Both are closed: a third value is a schema change, not a new
#: string somebody writes.
APPLICABILITY_STATUSES = ("required", "not_applicable")
APPLICABILITY_BASES = ("authored", "compiler_rule")

#: The one basis whose decisions :func:`apply_certified_omissions` may
#: materialize. An ``authored`` decision is already materialized by definition.
CERTIFYING_BASIS = "compiler_rule"

PayloadT = TypeVar("PayloadT")


class FulfillmentPlanError(RuntimeError):
    """Base class for the structured refusals plan construction raises."""


class DuplicateLogicalKeyError(FulfillmentPlanError):
    """Two declarations in one closure claim the same logical key."""

    def __init__(self, key: "LogicalKey", first: str, second: str) -> None:
        self.key = key
        self.first = first
        self.second = second
        super().__init__(
            f"{first} and {second} both declare the logical node {key.text}; a logical key "
            "addresses exactly one declaration, and a plan cannot resolve a receipt for a key "
            "that means two things"
        )


class UnresolvedPlanReferenceError(FulfillmentPlanError):
    """A plan names a node no declaration in the plan provides."""

    def __init__(
        self,
        missing: "LogicalKey",
        *,
        referrer: "LogicalKey | None" = None,
        role_path: Sequence[str] = (),
        relation: str = "producer",
    ) -> None:
        self.missing = missing
        self.referrer = referrer
        self.role_path = tuple(role_path)
        self.relation = relation
        if referrer is None:
            super().__init__(
                f"{missing.text} is the plan's {relation} but no declaration in the plan "
                "provides it"
            )
            return
        super().__init__(
            f"{referrer.text} declares input {list(self.role_path)} whose {relation} "
            f"{missing.text} is not a node of this plan; a node a plan names must be a node "
            "the plan carries"
        )


class PlanCycleError(FulfillmentPlanError):
    """The dependency closure is cyclic, so it has no execution order."""

    def __init__(self, target: "LogicalKey", unplaced: Sequence[str]) -> None:
        self.target = target
        self.unplaced = tuple(unplaced)
        super().__init__(
            f"the dependency closure of {target.text} is cyclic; no execution order exists for "
            f"{list(self.unplaced)}"
        )


class UnsupportedFulfillmentPlanVersionError(ValueError):
    """A plan document declares a schema identity this build does not support."""


class CertifiedOmissionsPendingError(FulfillmentPlanError):
    """A plan certifies inapplicability its declarations have not materialized.

    This is author-actionable: :attr:`omissions` names every certified decision,
    with the rule that certified it, so the lowering project can render the exact
    edit its declaration format needs. Materializing them mechanically instead is
    an explicit caller choice (:func:`apply_certified_omissions`), never a
    fallback this module takes on its own.
    """

    def __init__(self, target: "LogicalKey", omissions: Sequence["OmissionRecord"]) -> None:
        self.target = target
        self.omissions = tuple(omissions)
        listing = ", ".join(
            f"{record.consumer.text}{list(record.role_path)} ({record.rule})"
            for record in self.omissions
        )
        super().__init__(
            f"the plan for {target.text} certifies {len(self.omissions)} input(s) as not "
            f"applicable that its declarations still include: {listing}. Author the "
            "applicability the plan states, or bind a producer to each input; materializing "
            "them mechanically is an explicit choice and mints a different identity"
        )


@dataclass(frozen=True, order=True)
class LogicalKey:
    """The stable logical address of one plan node: its layer and its envelope.

    A node is one declaration, and a declaration produces exactly one document,
    so ``(layer, envelope)`` addresses it completely and identically from every
    referrer. Role paths address *inputs*, not documents, so they live on the
    edges, where a role means something.

    Both parts are project vocabulary. ``layer`` may not contain ``":"`` and
    neither part may be empty, so :attr:`text` parses back to exactly one key.
    """

    layer: str
    envelope: str

    def __post_init__(self) -> None:
        if not self.layer or not self.envelope:
            raise ValueError("a logical key names a non-empty layer and a non-empty envelope")
        if ":" in self.layer:
            raise ValueError(
                f"{self.layer!r} is not a layer name: ':' separates a key's parts, so a layer "
                "carrying one could forge another key's text"
            )

    @property
    def text(self) -> str:
        """The canonical string form, quoted so no name can forge a separator."""
        return f"{self.layer}:{quote(self.envelope, safe='')}"

    @classmethod
    def parse(cls, text: str) -> "LogicalKey":
        """Return the key one canonical :attr:`text` form addresses."""
        layer, separator, envelope = text.partition(":")
        if not separator:
            raise ValueError(f"{text!r} is not a logical key: it names no layer")
        return cls(layer, unquote(envelope))

    def record(self) -> dict[str, str]:
        return {"layer": self.layer, "envelope": self.envelope, "key": self.text}


@dataclass(frozen=True)
class PlanNode:
    """One declaration in the closure, addressed logically.

    Attributes:
        key: The node's logical address.
        source_ref: Where the declaration lives, in the lowering project's own
            reference vocabulary. Opaque here, and the ref an expander is asked
            to expand.
        kind: The project's own node kind for this declaration. Opaque here: the
            driver's lowering callback is what turns a kind into a node request.
        content_hash: The project's hash of the document this node pins, so a
            later lowering can prove the declaration has not moved underneath it.
        execution_identity: The project's execution identity for this node, when
            it has one.
        boundary: When set, this node names an *external boundary*: a producer
            this runner may consume the receipt of but never execute. The string
            is the project's own name for the boundary.
        metadata: Extra project facts carried with the node. Serialized as
            given; nothing here reads it.
    """

    key: LogicalKey
    source_ref: str
    kind: str
    content_hash: str | None = None
    execution_identity: str | None = None
    boundary: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        return {
            **self.key.record(),
            "source_ref": self.source_ref,
            "kind": self.kind,
            "content_hash": self.content_hash,
            "execution_identity": self.execution_identity,
            "boundary": self.boundary,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "PlanNode":
        return cls(
            key=LogicalKey(record["layer"], record["envelope"]),
            source_ref=record["source_ref"],
            kind=record["kind"],
            content_hash=record.get("content_hash"),
            execution_identity=record.get("execution_identity"),
            boundary=record.get("boundary"),
            metadata=dict(record.get("metadata") or {}),
        )


@dataclass(frozen=True)
class PlanEdge:
    """One input a node declares, with the typed decision the lowerer reached.

    Exactly one of ``producer`` and ``external`` is set when the status is
    ``required``; both are ``None`` when the status is ``not_applicable``,
    because an inapplicable input binds nothing. A ``compiler_rule`` basis names
    the closed rule that certified it; an ``authored`` basis names none, because
    the declaration itself is the authority.
    """

    consumer: LogicalKey
    role_path: tuple[str, ...]
    status: str
    basis: str
    reason: str | None = None
    producer: LogicalKey | None = None
    external: Mapping[str, Any] | None = None
    rule: str | None = None

    def __post_init__(self) -> None:
        if self.status not in APPLICABILITY_STATUSES:
            raise ValueError(f"{self.status!r} is not one of {list(APPLICABILITY_STATUSES)}")
        if self.basis not in APPLICABILITY_BASES:
            raise ValueError(f"{self.basis!r} is not one of {list(APPLICABILITY_BASES)}")
        if self.status == "not_applicable" and (
            self.producer is not None or self.external is not None
        ):
            raise ValueError("an inapplicable input binds nothing, so it names no producer")
        if self.producer is not None and self.external is not None:
            raise ValueError(
                "an input binds one thing: a producer in this closure, or an external receipt"
            )
        if self.basis == CERTIFYING_BASIS and not self.rule:
            raise ValueError(
                "a compiler_rule decision names the closed rule that certified it, so the "
                "refusal, the patch, and the materialized omission all quote one rule"
            )
        if self.basis != CERTIFYING_BASIS and self.rule:
            raise ValueError(
                f"{self.basis!r} is not a rule-based decision, so it names no rule"
            )

    @property
    def certified(self) -> bool:
        """Whether a closed rule certified this input as not applicable."""
        return self.status == "not_applicable" and self.basis == CERTIFYING_BASIS

    def record(self) -> dict[str, Any]:
        return {
            "consumer": self.consumer.text,
            "role_path": list(self.role_path),
            "status": self.status,
            "basis": self.basis,
            "reason": self.reason,
            "rule": self.rule,
            "producer": self.producer.text if self.producer is not None else None,
            "external": dict(self.external) if self.external is not None else None,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "PlanEdge":
        producer = record.get("producer")
        external = record.get("external")
        return cls(
            consumer=LogicalKey.parse(record["consumer"]),
            role_path=tuple(record["role_path"]),
            status=record["status"],
            basis=record["basis"],
            reason=record.get("reason"),
            producer=LogicalKey.parse(producer) if producer is not None else None,
            external=dict(external) if external is not None else None,
            rule=record.get("rule"),
        )

    @property
    def sort_key(self) -> tuple[str, tuple[str, ...], bytes]:
        return (self.consumer.text, self.role_path, canonical_json_bytes(self.record()))


@dataclass(frozen=True)
class OmissionRecord:
    """One certified decision, as the run that materialized it records it."""

    consumer: LogicalKey
    role_path: tuple[str, ...]
    reason: str | None
    basis: str
    rule: str | None

    @classmethod
    def of(cls, edge: PlanEdge) -> "OmissionRecord":
        return cls(
            consumer=edge.consumer,
            role_path=edge.role_path,
            reason=edge.reason,
            basis=edge.basis,
            rule=edge.rule,
        )

    def record(self) -> dict[str, Any]:
        return {
            "consumer": self.consumer.text,
            "role_path": list(self.role_path),
            "reason": self.reason,
            "basis": self.basis,
            "rule": self.rule,
        }


@dataclass(frozen=True)
class FulfillmentPlan:
    """One target's dependency closure, in dependency order.

    Attributes:
        target: The node the closure was built for.
        nodes: Every node of the closure, ordered so a producer always precedes
            its consumers, ties broken on canonical key text.
        edges: Every declared input, in one canonical order.
        origin: Free-form provenance the lowering project records about how the
            plan was derived, such as its compiler identity and version.
    """

    target: LogicalKey
    nodes: tuple[PlanNode, ...]
    edges: tuple[PlanEdge, ...]
    origin: Mapping[str, Any] = field(default_factory=dict)

    def node(self, key: LogicalKey) -> PlanNode:
        """Return the node addressed by one logical key."""
        for node in self.nodes:
            if node.key == key:
                return node
        raise KeyError(key.text)

    def input_edges(self, key: LogicalKey, *, status: str | None = None) -> tuple[PlanEdge, ...]:
        """Return one node's declared inputs, in the plan's canonical order."""
        return tuple(
            edge
            for edge in self.edges
            if edge.consumer == key and (status is None or edge.status == status)
        )

    def required_edges(self, key: LogicalKey) -> tuple[PlanEdge, ...]:
        """Return one node's inputs that something must bind."""
        return self.input_edges(key, status="required")

    def certified_omissions(self, *, consumer: LogicalKey | None = None) -> tuple[PlanEdge, ...]:
        """Return the edges a closed rule certified as not applicable."""
        return tuple(
            edge
            for edge in self.edges
            if edge.certified and (consumer is None or edge.consumer == consumer)
        )

    def consumers(self) -> dict[LogicalKey, tuple[PlanEdge, ...]]:
        """Return, per producing node, the edges that name it, in canonical order."""
        grouped: dict[LogicalKey, list[PlanEdge]] = {node.key: [] for node in self.nodes}
        for edge in self.edges:
            if edge.producer is not None:
                grouped[edge.producer].append(edge)
        return {key: tuple(edges) for key, edges in grouped.items()}

    def descendants(self, key: LogicalKey) -> tuple[str, ...]:
        """Return every node downstream of one node, in canonical key order."""
        consumers = self.consumers()
        seen: set[LogicalKey] = set()
        frontier = [key]
        while frontier:
            current = frontier.pop()
            for edge in consumers.get(current, ()):
                if edge.consumer not in seen:
                    seen.add(edge.consumer)
                    frontier.append(edge.consumer)
        return tuple(sorted(item.text for item in seen))

    def boundary_nodes(self) -> tuple[PlanNode, ...]:
        """Return the nodes the lowerer marked as never executable here."""
        return tuple(node for node in self.nodes if node.boundary is not None)

    def document(self) -> dict[str, Any]:
        """Return the versioned plan document, deterministic in every field."""
        return {
            "schema_id": FULFILLMENT_PLAN_SCHEMA_ID,
            "schema_version": FULFILLMENT_PLAN_SCHEMA_VERSION_V1,
            "target": self.target.text,
            "origin": dict(self.origin),
            "nodes": [node.record() for node in self.nodes],
            "edges": [edge.record() for edge in self.edges],
        }


def read_fulfillment_plan_document(
    document: Any, *, field_ref: str = "fulfillment_plan"
) -> dict[str, Any]:
    """Admit one plan document, or fail closed on an unsupported version.

    There is no migration and no inference: a family or version this build does
    not enumerate is refused with the supported set named.
    """
    if not isinstance(document, Mapping):
        raise UnsupportedFulfillmentPlanVersionError(
            f"{field_ref} is not a fulfillment plan document: expected a mapping, got "
            f"{type(document).__name__}"
        )
    schema_id = document.get("schema_id")
    if schema_id != FULFILLMENT_PLAN_SCHEMA_ID:
        raise UnsupportedFulfillmentPlanVersionError(
            f"unsupported {field_ref} schema_id: {schema_id!r}; expected "
            f"{FULFILLMENT_PLAN_SCHEMA_ID!r}"
        )
    version = document.get("schema_version")
    if version not in FULFILLMENT_PLAN_SUPPORTED_SCHEMA_VERSIONS:
        raise UnsupportedFulfillmentPlanVersionError(
            f"unsupported {field_ref} schema_version: {version!r}; supported versions are "
            f"{list(FULFILLMENT_PLAN_SUPPORTED_SCHEMA_VERSIONS)}; no migration is defined "
            f"(migration table={FULFILLMENT_PLAN_MIGRATION_TABLE!r})"
        )
    return dict(document)


def fulfillment_plan_from_document(
    document: Any, *, field_ref: str = "fulfillment_plan"
) -> FulfillmentPlan:
    """Load one emitted plan document back into a plan, or fail closed."""
    admitted = read_fulfillment_plan_document(document, field_ref=field_ref)
    nodes = tuple(PlanNode.from_record(record) for record in admitted["nodes"])
    edges = tuple(PlanEdge.from_record(record) for record in admitted["edges"])
    return build_fulfillment_plan(
        LogicalKey.parse(admitted["target"]),
        nodes,
        edges,
        origin=admitted.get("origin") or {},
    )


def build_fulfillment_plan(
    target: LogicalKey,
    nodes: Iterable[PlanNode],
    edges: Iterable[PlanEdge],
    *,
    origin: Mapping[str, Any] | None = None,
) -> FulfillmentPlan:
    """Assemble one closure from declared nodes and edges.

    The closure is what the target reaches: nodes no path from the target
    reaches, and the edges that name them, are not part of this plan. Within it,
    every reference must resolve, one logical key names one declaration, and a
    cycle is named rather than silently truncating the plan.
    """
    declared: dict[LogicalKey, PlanNode] = {}
    for node in nodes:
        existing = declared.get(node.key)
        if existing is not None:
            if existing.source_ref != node.source_ref:
                raise DuplicateLogicalKeyError(node.key, existing.source_ref, node.source_ref)
            continue
        declared[node.key] = node
    if target not in declared:
        raise UnresolvedPlanReferenceError(target, relation="target")

    declared_edges = list(edges)
    inputs: dict[LogicalKey, list[PlanEdge]] = {key: [] for key in declared}
    for edge in declared_edges:
        if edge.producer is not None and edge.producer not in declared:
            raise UnresolvedPlanReferenceError(
                edge.producer, referrer=edge.consumer, role_path=edge.role_path
            )
        if edge.consumer not in inputs:
            raise UnresolvedPlanReferenceError(
                edge.consumer,
                referrer=edge.consumer,
                role_path=edge.role_path,
                relation="consumer",
            )
        inputs[edge.consumer].append(edge)

    reached: set[LogicalKey] = set()
    frontier = deque([target])
    while frontier:
        key = frontier.popleft()
        if key in reached:
            continue
        reached.add(key)
        for edge in inputs[key]:
            if edge.producer is not None and edge.producer not in reached:
                frontier.append(edge.producer)

    closure_nodes = {key: node for key, node in declared.items() if key in reached}
    closure_edges = [edge for edge in declared_edges if edge.consumer in reached]
    return FulfillmentPlan(
        target=target,
        nodes=_dependency_order(closure_nodes, closure_edges, target),
        edges=tuple(sorted(closure_edges, key=lambda item: item.sort_key)),
        origin=dict(origin or {}),
    )


@dataclass(frozen=True)
class EdgeDeclaration:
    """One input a node declares, before its producer has been addressed.

    An expander states inputs in its own reference vocabulary: a ``producer_ref``
    is a source ref the expander itself can expand, and the kernel is what turns
    it into a :class:`LogicalKey`. Everything else is exactly a
    :class:`PlanEdge`'s typed decision.
    """

    role_path: tuple[str, ...]
    status: str = "required"
    basis: str = "authored"
    reason: str | None = None
    producer_ref: str | None = None
    external: Mapping[str, Any] | None = None
    rule: str | None = None

    def edge(self, consumer: LogicalKey, producer: LogicalKey | None) -> PlanEdge:
        return PlanEdge(
            consumer=consumer,
            role_path=tuple(self.role_path),
            status=self.status,
            basis=self.basis,
            reason=self.reason,
            producer=producer,
            external=self.external,
            rule=self.rule,
        )


@dataclass(frozen=True)
class NodeDeclaration:
    """What one source ref expands to: a node and the inputs it declares."""

    node: PlanNode
    edges: tuple[EdgeDeclaration, ...] = ()


class NodeExpander(Protocol):
    """Expands one source ref into the node and inputs it declares.

    This is the project-facing seam of plan construction: the kernel knows how
    to reach a closure, dedupe a diamond, refuse a key collision, and order the
    result; the expander knows what a declaration is and how to read it.

    The source ref is positional-only, so an implementation names its parameter
    whatever its own vocabulary calls it.
    """

    def __call__(self, source_ref: str, /) -> NodeDeclaration:  # pragma: no cover - protocol
        ...


def expand_fulfillment_plan(
    target_ref: str,
    *,
    expand: NodeExpander,
    origin: Mapping[str, Any] | None = None,
) -> FulfillmentPlan:
    """Walk one target's dependency closure through a project's expander.

    Each source ref is expanded exactly once, so one declaration reached by two
    referrers is one node and a diamond is deduplicated rather than doubled. Two
    refs that claim one logical key refuse, because a plan cannot resolve a
    receipt for a key that means two things.
    """
    expanded: dict[str, NodeDeclaration] = {}

    def declare(source_ref: str) -> NodeDeclaration:
        if source_ref not in expanded:
            expanded[source_ref] = expand(source_ref)
        return expanded[source_ref]

    nodes: dict[LogicalKey, PlanNode] = {}
    edges: list[PlanEdge] = []
    pending: deque[str] = deque([target_ref])
    target_declaration = declare(target_ref)
    while pending:
        source_ref = pending.popleft()
        declaration = declare(source_ref)
        key = declaration.node.key
        existing = nodes.get(key)
        if existing is not None:
            if existing.source_ref != declaration.node.source_ref:
                raise DuplicateLogicalKeyError(
                    key, existing.source_ref, declaration.node.source_ref
                )
            continue
        nodes[key] = declaration.node
        for declared in declaration.edges:
            producer: LogicalKey | None = None
            if declared.producer_ref is not None:
                producer = declare(declared.producer_ref).node.key
                pending.append(declared.producer_ref)
            edges.append(declared.edge(key, producer))
    return build_fulfillment_plan(
        target_declaration.node.key, tuple(nodes.values()), tuple(edges), origin=origin
    )


def _dependency_order(
    nodes: Mapping[LogicalKey, PlanNode],
    edges: Sequence[PlanEdge],
    target: LogicalKey,
) -> tuple[PlanNode, ...]:
    """Order the closure so every producer precedes its consumers.

    Ties break on the canonical key text, so two builds of one plan emit one
    order. A diamond contributes one dependency, not two, because a repeated
    ``(producer, consumer)`` pair is counted once.
    """
    consumers: dict[LogicalKey, set[LogicalKey]] = {key: set() for key in nodes}
    remaining: dict[LogicalKey, int] = dict.fromkeys(nodes, 0)
    for edge in edges:
        if edge.producer is None or edge.consumer in consumers[edge.producer]:
            continue
        consumers[edge.producer].add(edge.consumer)
        remaining[edge.consumer] += 1
    ready = sorted((key for key, count in remaining.items() if count == 0), key=lambda k: k.text)
    ordered: list[PlanNode] = []
    while ready:
        key = ready.pop(0)
        ordered.append(nodes[key])
        for consumer in sorted(consumers[key], key=lambda k: k.text):
            remaining[consumer] -= 1
            if remaining[consumer] == 0:
                ready.append(consumer)
        ready.sort(key=lambda k: k.text)
    if len(ordered) != len(nodes):
        placed = {node.key for node in ordered}
        raise PlanCycleError(
            target, sorted(key.text for key in nodes if key not in placed)
        )
    return tuple(ordered)


def require_no_certified_omissions(
    plan: FulfillmentPlan, *, consumer: LogicalKey | None = None
) -> None:
    """Refuse a plan that still carries a certified-inapplicable input.

    This is the no-flag path. The refusal carries every certified decision and
    the rule that certified it, so the author's next action is to state that
    applicability rather than to reverse-engineer what the lowerer proved.
    """
    omissions = plan.certified_omissions(consumer=consumer)
    if omissions:
        raise CertifiedOmissionsPendingError(
            plan.target, tuple(OmissionRecord.of(edge) for edge in omissions)
        )


class OmissionApplier(Protocol):
    """Materializes certified omissions into one project document.

    The kernel decides *which* decisions may be materialized — only those a
    closed rule certified — and the project decides how its own document says
    so. An applier never sees an uncertified edge. Both parameters are
    positional-only, so an implementation names them freely.
    """

    def __call__(
        self, payload: Any, edges: Sequence[PlanEdge], /
    ) -> Any:  # pragma: no cover - protocol
        ...


def apply_certified_omissions(
    payload: PayloadT,
    plan: FulfillmentPlan,
    *,
    consumer: LogicalKey,
    apply: OmissionApplier | Callable[[PayloadT, Sequence[PlanEdge]], PayloadT],
) -> tuple[PayloadT, tuple[OmissionRecord, ...]]:
    """Materialize one node's certified omissions into its document.

    Only decisions a closed rule certified are passed to *apply*: an authored
    decision is already materialized, a required input is not a decision at all,
    and a missing receipt never reaches this module. Because the document
    changes, whatever identity is minted from it changes; that is the point, not
    a side effect. Every materialized decision is returned so the run that
    executes the result can record what it left out.
    """
    omissions = plan.certified_omissions(consumer=consumer)
    if not omissions:
        return deepcopy(payload), ()
    applied = apply(deepcopy(payload), omissions)
    return applied, tuple(OmissionRecord.of(edge) for edge in omissions)
