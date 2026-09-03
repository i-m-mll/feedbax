"""The one finite, backend-neutral scientific workflow representation.

Campaign, evaluation, analysis, report, and artifact-fulfillment authoring stay
owned by their respective layers. Each lowers into :class:`WorkflowPlan`: a
finite acyclic graph of typed operations and exact bindings. This kernel owns
closure, canonical order, logical identity, and finite guards. It never chooses
a provider, performs custody lookup, polls, retries, or dispatches an effect.

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

## Bindings, omissions, and finite guards

Every edge is a typed producer binding, a typed exact external binding, a
certified omission, or a guarded binding. A guard is a closed deterministic
predicate over one enumerated predecessor's typed output. It can select a
finite branch; it cannot add a node, loop, poll, retry, or inspect backend state.

* **authored** — the declaration itself states the input is not applicable, so
  the decision is already materialized and nothing further is owed.
* **compiler_rule** — a closed versioned Feedbax structural rule proves the
  target provides no such input. The rule is *named* on the edge
  (:attr:`PlanEdge.rule`), so the plan quotes the rule that decided rather than
  asserting the decision bare.

Both decisions are made by the compile that emitted the lock, and are already
materialized in the compiled document by the time a plan exists. Nothing
downstream re-decides applicability, and no project callback ever did.

A missing receipt is never one of these. Not-yet-executed, previously-failed,
wrong-root, and corrupt are execution outcomes, not evidence about the
science, and none of them reaches this module.

## What this kernel does not decide

It does not know how a declaration compiles, what a rule name means, or which
compiled documents exist. Those reach it as data on the plan, derived from
compile locks by the layer-owned workflow lowerers.

Stability note: this module is not part of the owner-ratified downstream
inventory in ``docs/design/downstream_interface_stability.md``. The
``feedbax.workflow.plan`` family carries explicit schema identity and refuses
the predecessor fulfillment schema explicitly. An emitted plan is a durable
spec that crosses a process boundary.
"""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import quote, unquote

from feedbax.contracts.base import canonical_json_bytes


WORKFLOW_PLAN_SCHEMA_ID = "feedbax.workflow.plan"
WORKFLOW_PLAN_SCHEMA_VERSION = "feedbax.workflow.plan.v2"
LEGACY_FULFILLMENT_PLAN_SCHEMA_ID = "feedbax.fulfillment.plan"

#: Enumerated, never inferred. A document from any other version is refused.
WORKFLOW_PLAN_SUPPORTED_SCHEMA_VERSIONS = (WORKFLOW_PLAN_SCHEMA_VERSION,)

#: The two statuses an input edge can carry, and the two bases a status may be
#: reached on. Both are closed: a third value is a schema change, not a new
#: string somebody writes.
APPLICABILITY_STATUSES = ("required", "guarded", "not_applicable")
APPLICABILITY_BASES = ("authored", "compiler_rule")
OPERATION_DETERMINISM = ("deterministic", "seeded", "nondeterministic")
OPERATION_CACHE_POLICIES = ("content_addressed", "never")
OPERATION_EFFECTS = ("pure", "local", "external", "publication")
GUARD_OPERATORS = ("equals", "not_equals", "in", "not_in")
EDGE_BINDINGS = ("single_receipt", "complete_receipt_set")

# Realization belongs to the later binding/execution layers. An authoring
# lowerer must not smuggle any of it into the operation or node metadata where
# it would become semantic plan identity.
_REALIZATION_STATE_KEYS = frozenset(
    {
        "execution_uri",
        "filesystem_root",
        "http_state",
        "instance_id",
        "physical_blob_location",
        "pod_id",
        "receipt_timestamp",
        "storage_uri",
        "studio_state",
    }
)
_REALIZATION_STATE_PREFIXES = ("attempt_", "custody_", "provider_", "retry_")

#: The basis a closed versioned rule reaches a decision on, as opposed to a
#: human authoring it in the envelope.
CERTIFYING_BASIS = "compiler_rule"


class WorkflowPlanError(RuntimeError):
    """Base class for the structured refusals plan construction raises."""


class DuplicateLogicalKeyError(WorkflowPlanError):
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


class ConflictingNodeDeclarationError(WorkflowPlanError):
    """Two declarations at one logical key state different facts about it.

    Two entries for one key are only harmless when they are the same statement
    twice. When they disagree about anything the node carries — the document it
    pins, the schema it is lowered by, the execution identity it was compiled
    under, or its typed operation — one of them is discarded, and the one that
    survives is whichever the loader happened to see first. That is a plan
    silently choosing between two mutually exclusive accounts of what a node is,
    which is exactly the choice a derived record has no authority to make.
    """

    def __init__(self, key: "LogicalKey", source_ref: str, differences: Sequence[str]) -> None:
        self.key = key
        self.source_ref = source_ref
        self.differences = tuple(differences)
        listing = "; ".join(self.differences)
        super().__init__(
            f"the plan declares the logical node {key.text} ({source_ref}) more than once, and "
            f"the declarations disagree: {listing}. Repeating a node declaration states the "
            "same node twice; two declarations that differ state two nodes at one address, and "
            "keeping either one would discard a fact the plan asserts."
        )


class DuplicateInputEdgeError(WorkflowPlanError):
    """One node declares two edges at a single input role path.

    A role path addresses one input, and a compile lock refuses to state two
    references at one role, so a plan carrying two edges there is not a derived
    record of any lock. It is also the shape that defeats every downstream
    consumer of the plan: reconciliation, binding, and exact-parent resolution
    all address an edge by ``(consumer, role_path)``, so a second edge at one
    role is silently dropped by whichever of them keys last — while remaining
    fully live for reachability, which is how an injected node reaches the
    execution order without ever being reconciled against a lock.
    """

    def __init__(
        self,
        consumer: "LogicalKey",
        role_path: Sequence[str],
        first: "PlanEdge",
        second: "PlanEdge",
    ) -> None:
        self.consumer = consumer
        self.role_path = tuple(role_path)
        self.first = first
        self.second = second
        super().__init__(
            f"{consumer.text} declares input {list(self.role_path)} twice, binding "
            f"{_edge_binding_text(first)} and {_edge_binding_text(second)}; a role path "
            "addresses exactly one input, and no compile lock states two references at one "
            "role, so a plan carrying both is not a derived record of any lock"
        )


class UnresolvedPlanReferenceError(WorkflowPlanError):
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


class PlanCycleError(WorkflowPlanError):
    """The dependency closure is cyclic, so it has no execution order."""

    def __init__(self, target: "LogicalKey", unplaced: Sequence[str]) -> None:
        self.target = target
        self.unplaced = tuple(unplaced)
        super().__init__(
            f"the dependency closure of {target.text} is cyclic; no execution order exists for "
            f"{list(self.unplaced)}"
        )


class UnsupportedWorkflowPlanVersionError(ValueError):
    """A plan document declares a schema identity this build does not support."""


class WorkflowPlanIdentityError(ValueError):
    """A workflow document's declared identity does not match its semantics."""


class UnresolvedGuardOutcomeError(WorkflowPlanError):
    """A finite guard was evaluated before its typed predecessor outcome existed."""

    def __init__(
        self, consumer: "LogicalKey", role_path: Sequence[str], guard: "PlanGuard"
    ) -> None:
        self.consumer = consumer
        self.role_path = tuple(role_path)
        self.guard = guard
        super().__init__(
            f"{consumer.text} input {list(self.role_path)} is guarded by "
            f"{guard.outcome.text}.{guard.output_role}, but that typed outcome was not supplied"
        )


class WorkflowTypeMismatchError(WorkflowPlanError):
    """A workflow binding connects incompatible durable type identities."""


def _reject_realization_state(value: Any, *, field_ref: str) -> None:
    """Refuse provider, custody, attempt, and physical-location authoring state."""
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).lower().replace("-", "_")
            if key in _REALIZATION_STATE_KEYS or key.startswith(_REALIZATION_STATE_PREFIXES):
                raise ValueError(
                    f"{field_ref} contains realization field {raw_key!r}; provider, custody, "
                    "attempt, retry, and physical-location state is bound after semantic "
                    "workflow authoring"
                )
            _reject_realization_state(item, field_ref=f"{field_ref}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_realization_state(item, field_ref=f"{field_ref}[{index}]")


@dataclass(frozen=True, order=True)
class LogicalKey:
    """The stable logical address of one plan node: its layer and its envelope.

    A node is one declaration, and a declaration produces exactly one document,
    so ``(layer, envelope)`` addresses it completely and identically from every
    referrer. Role paths address *inputs*, not documents, so they live on the
    edges, where a role means something.

    ``layer`` is one of Feedbax's artifact layers and ``envelope`` is the
    compiled output's name. ``layer`` may not contain ``":"`` and neither part
    may be empty, so :attr:`text` parses back to exactly one key.
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
class Operation:
    """One bounded scientific operation, without realization state.

    ``type_id`` is the registered semantic operation identity. ``parameters``
    are its typed semantic payload. Input and output role types are durable
    identifiers used to validate bindings before execution. Effect is a static
    semantic classification, not an attempt, provider, or custody status.
    """

    type_id: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    input_types: Mapping[str, str] = field(default_factory=dict)
    output_types: Mapping[str, str] = field(default_factory=dict)
    determinism: str = "deterministic"
    cache_policy: str = "content_addressed"
    effect: str = "pure"
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.type_id:
            raise ValueError("an operation requires a non-empty registered type_id")
        if self.determinism not in OPERATION_DETERMINISM:
            raise ValueError(f"{self.determinism!r} is not one of {list(OPERATION_DETERMINISM)}")
        if self.cache_policy not in OPERATION_CACHE_POLICIES:
            raise ValueError(
                f"{self.cache_policy!r} is not one of {list(OPERATION_CACHE_POLICIES)}"
            )
        if self.effect not in OPERATION_EFFECTS:
            raise ValueError(f"{self.effect!r} is not one of {list(OPERATION_EFFECTS)}")
        if any(not role or not type_id for role, type_id in self.input_types.items()):
            raise ValueError("operation input roles and type identities must be non-empty")
        if any(not role or not type_id for role, type_id in self.output_types.items()):
            raise ValueError("operation output roles and type identities must be non-empty")
        _reject_realization_state(self.parameters, field_ref="operation.parameters")

    def record(self) -> dict[str, Any]:
        return {
            "type_id": self.type_id,
            "parameters": dict(self.parameters),
            "input_types": dict(sorted(self.input_types.items())),
            "output_types": dict(sorted(self.output_types.items())),
            "determinism": self.determinism,
            "cache_policy": self.cache_policy,
            "effect": self.effect,
            "capabilities": sorted(set(self.capabilities)),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "Operation":
        return cls(
            type_id=str(record["type_id"]),
            parameters=dict(record.get("parameters") or {}),
            input_types=dict(record.get("input_types") or {}),
            output_types=dict(record.get("output_types") or {}),
            determinism=str(record.get("determinism", "deterministic")),
            cache_policy=str(record.get("cache_policy", "content_addressed")),
            effect=str(record.get("effect", "pure")),
            capabilities=tuple(record.get("capabilities") or ()),
        )


@dataclass(frozen=True)
class GuardPredicate:
    """A closed deterministic predicate over one prior typed outcome."""

    operator: str
    operand: Any

    def __post_init__(self) -> None:
        if self.operator not in GUARD_OPERATORS:
            raise ValueError(f"{self.operator!r} is not one of {list(GUARD_OPERATORS)}")
        canonical_json_bytes(self.operand)
        if self.operator in {"in", "not_in"} and not isinstance(self.operand, (list, tuple)):
            raise ValueError(f"guard operator {self.operator!r} requires a finite JSON array")

    def evaluate(self, outcome: Any) -> bool:
        if self.operator == "equals":
            return outcome == self.operand
        if self.operator == "not_equals":
            return outcome != self.operand
        if self.operator == "in":
            return outcome in self.operand
        return outcome not in self.operand

    def record(self) -> dict[str, Any]:
        operand = list(self.operand) if isinstance(self.operand, tuple) else self.operand
        return {"operator": self.operator, "operand": operand}


@dataclass(frozen=True)
class PlanGuard:
    """One finite branch decision, sourced from an enumerated predecessor."""

    outcome: LogicalKey
    output_role: str
    output_type: str
    predicate: GuardPredicate

    def __post_init__(self) -> None:
        if not self.output_role or not self.output_type:
            raise ValueError("a guard names a non-empty output role and type")

    def record(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.text,
            "output_role": self.output_role,
            "output_type": self.output_type,
            "predicate": self.predicate.record(),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "PlanGuard":
        predicate = record["predicate"]
        return cls(
            outcome=LogicalKey.parse(str(record["outcome"])),
            output_role=str(record["output_role"]),
            output_type=str(record["output_type"]),
            predicate=GuardPredicate(
                operator=str(predicate["operator"]), operand=predicate.get("operand")
            ),
        )


@dataclass(frozen=True)
class PlanNode:
    """One declaration in the closure, addressed logically.

    Attributes:
        key: The node's logical address.
        source_ref: The repo-relative envelope this node was compiled from, and
            the ref an expander is asked to expand.
        operation: The typed bounded operation this node performs.
        content_hash: The canonical hash of the compiled document this node pins,
            so a later lowering can prove it has not moved underneath the plan.
        execution_identity: The compile lock's execution identity for this node.
        metadata: Extra facts carried with the node. Serialized as given;
            nothing here reads it.
    """

    key: LogicalKey
    source_ref: str
    operation: Operation
    content_hash: str | None = None
    execution_identity: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _reject_realization_state(self.metadata, field_ref="plan_node.metadata")

    def record(self) -> dict[str, Any]:
        return {
            **self.key.record(),
            "source_ref": self.source_ref,
            "operation": self.operation.record(),
            "content_hash": self.content_hash,
            "execution_identity": self.execution_identity,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "PlanNode":
        return cls(
            key=LogicalKey(record["layer"], record["envelope"]),
            source_ref=record["source_ref"],
            operation=Operation.from_record(record["operation"]),
            content_hash=record.get("content_hash"),
            execution_identity=record.get("execution_identity"),
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
    binding: str | None = None
    input_type: str = "feedbax.artifact"
    reason: str | None = None
    producer: LogicalKey | None = None
    producer_output: str | None = None
    external: Mapping[str, Any] | None = None
    external_type: str | None = None
    rule: str | None = None
    guard: PlanGuard | None = None

    def __post_init__(self) -> None:
        if self.status not in APPLICABILITY_STATUSES:
            raise ValueError(f"{self.status!r} is not one of {list(APPLICABILITY_STATUSES)}")
        if self.basis not in APPLICABILITY_BASES:
            raise ValueError(f"{self.basis!r} is not one of {list(APPLICABILITY_BASES)}")
        if self.status in {"required", "guarded"} and self.binding is None:
            object.__setattr__(self, "binding", "single_receipt")
        if self.status in {"required", "guarded"} and self.binding not in EDGE_BINDINGS:
            raise ValueError(f"an active input edge requires one of {list(EDGE_BINDINGS)}")
        if self.status == "not_applicable" and self.binding is not None:
            raise ValueError("an inapplicable input binds no receipt mode")
        if self.external is not None and self.binding == "complete_receipt_set":
            raise ValueError("an exact external receipt names one receipt, not a receipt set")
        if not self.input_type:
            raise ValueError("an input edge requires a non-empty type identity")
        if self.status == "not_applicable" and (
            self.producer is not None
            or self.external is not None
            or self.producer_output is not None
            or self.external_type is not None
            or self.guard is not None
        ):
            raise ValueError("an inapplicable input binds nothing, so it names no producer")
        if self.producer is not None and self.external is not None:
            raise ValueError(
                "an input binds one thing: a producer in this closure, or an external receipt"
            )
        if self.status in {"required", "guarded"} and (
            (self.producer is None) == (self.external is None)
        ):
            raise ValueError(
                "a required or guarded input binds exactly one producer or exact external"
            )
        if self.producer is not None and not self.producer_output:
            raise ValueError("a producer binding names the producer's typed output role")
        if self.producer is None and self.producer_output is not None:
            raise ValueError("producer_output is valid only with an in-plan producer")
        if self.external is not None and not self.external_type:
            raise ValueError("an external binding declares its durable type identity")
        if self.external is None and self.external_type is not None:
            raise ValueError("external_type is valid only with an exact external binding")
        if self.status == "guarded" and self.guard is None:
            raise ValueError("a guarded input names its closed deterministic guard")
        if self.status != "guarded" and self.guard is not None:
            raise ValueError("only a guarded input may carry a guard")
        if self.basis == CERTIFYING_BASIS and not self.rule:
            raise ValueError(
                "a compiler_rule decision names the closed rule that certified it, so the "
                "refusal, the patch, and the materialized omission all quote one rule"
            )
        if self.basis != CERTIFYING_BASIS and self.rule:
            raise ValueError(f"{self.basis!r} is not a rule-based decision, so it names no rule")

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
            "binding": self.binding,
            "input_type": self.input_type,
            "reason": self.reason,
            "rule": self.rule,
            "producer": self.producer.text if self.producer is not None else None,
            "producer_output": self.producer_output,
            "external": dict(self.external) if self.external is not None else None,
            "external_type": self.external_type,
            "guard": self.guard.record() if self.guard is not None else None,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "PlanEdge":
        if record.get("status") in {"required", "guarded"} and "binding" not in record:
            raise ValueError("a workflow-plan v2 active edge states its receipt binding")
        producer = record.get("producer")
        external = record.get("external")
        guard = record.get("guard")
        return cls(
            consumer=LogicalKey.parse(record["consumer"]),
            role_path=tuple(record["role_path"]),
            status=record["status"],
            basis=record["basis"],
            binding=record.get("binding"),
            input_type=record["input_type"],
            reason=record.get("reason"),
            producer=LogicalKey.parse(producer) if producer is not None else None,
            producer_output=record.get("producer_output"),
            external=dict(external) if external is not None else None,
            external_type=record.get("external_type"),
            rule=record.get("rule"),
            guard=PlanGuard.from_record(guard) if guard is not None else None,
        )

    @property
    def sort_key(self) -> tuple[str, tuple[str, ...], bytes]:
        return (self.consumer.text, self.role_path, canonical_json_bytes(self.record()))


def _edge_binding_text(edge: PlanEdge) -> str:
    """Return what one edge binds, for naming a duplicate without dumping it."""
    if edge.producer is not None:
        return f"the product of {edge.producer.text}"
    if edge.external is not None:
        kind = edge.external.get("manifest_kind")
        manifest_id = edge.external.get("manifest_id")
        return f"the external receipt {kind}:{manifest_id}"
    return f"nothing ({edge.status})"


def _semantic_edge_record(edge: PlanEdge) -> dict[str, Any]:
    """Return one binding without later physical realization coordinates.

    External receipt identity and digest are semantic inputs. Where a provider
    happens to expose those bytes is not: changing a URI must not mint a new
    workflow for the same authenticated receipt.
    """
    record = edge.record()
    external = record.get("external")
    if external is not None:
        record["external"] = {
            key: value
            for key, value in external.items()
            if str(key).lower().replace("-", "_") not in _REALIZATION_STATE_KEYS
            and not str(key).lower().replace("-", "_").startswith(_REALIZATION_STATE_PREFIXES)
        }
    return record


@dataclass(frozen=True)
class WorkflowPlan:
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
            if edge.guard is not None and edge.guard.outcome != edge.producer:
                grouped[edge.guard.outcome].append(edge)
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

    def semantic_record(self) -> dict[str, Any]:
        """Return exactly the semantic facts covered by workflow identity."""
        return {
            "target": self.target.text,
            "nodes": [node.record() for node in self.nodes],
            "edges": [_semantic_edge_record(edge) for edge in self.edges],
        }

    @property
    def identity(self) -> str:
        """The content identity of operations, logical keys, and exact bindings."""
        return hashlib.sha256(canonical_json_bytes(self.semantic_record())).hexdigest()

    def active_edges(self, outcomes: Mapping[tuple[LogicalKey, str], Any]) -> tuple[PlanEdge, ...]:
        """Resolve finite guards from already-materialized typed outcomes.

        Missing guarded outcomes are diagnosed rather than interpreted as a
        false branch. All possible nodes remain in :attr:`nodes` regardless of
        which guarded bindings are active.
        """
        active: list[PlanEdge] = []
        for edge in self.edges:
            if edge.guard is None:
                active.append(edge)
                continue
            address = (edge.guard.outcome, edge.guard.output_role)
            if address not in outcomes:
                raise UnresolvedGuardOutcomeError(edge.consumer, edge.role_path, edge.guard)
            if edge.guard.predicate.evaluate(outcomes[address]):
                active.append(edge)
        return tuple(active)

    def document(self) -> dict[str, Any]:
        """Return the versioned plan document, deterministic in every field."""
        return {
            "schema_id": WORKFLOW_PLAN_SCHEMA_ID,
            "schema_version": WORKFLOW_PLAN_SCHEMA_VERSION,
            "identity": self.identity,
            "target": self.target.text,
            "origin": dict(self.origin),
            "nodes": [node.record() for node in self.nodes],
            "edges": [edge.record() for edge in self.edges],
        }


def read_workflow_plan_document(
    document: Any, *, field_ref: str = "workflow_plan"
) -> dict[str, Any]:
    """Admit one plan document, or fail closed on an unsupported version.

    There is no migration and no inference: a family or version this build does
    not enumerate is refused with the supported set named.
    """
    if not isinstance(document, Mapping):
        raise UnsupportedWorkflowPlanVersionError(
            f"{field_ref} is not a workflow plan document: expected a mapping, got "
            f"{type(document).__name__}"
        )
    schema_id = document.get("schema_id")
    if schema_id == LEGACY_FULFILLMENT_PLAN_SCHEMA_ID:
        raise UnsupportedWorkflowPlanVersionError(
            f"{field_ref} uses retired schema_id {LEGACY_FULFILLMENT_PLAN_SCHEMA_ID!r}; "
            "fulfillment plans cannot represent typed operations or finite guards and are "
            "explicitly rejected. Recompile the authoring source into a workflow plan."
        )
    if schema_id != WORKFLOW_PLAN_SCHEMA_ID:
        raise UnsupportedWorkflowPlanVersionError(
            f"unsupported {field_ref} schema_id: {schema_id!r}; expected "
            f"{WORKFLOW_PLAN_SCHEMA_ID!r}"
        )
    version = document.get("schema_version")
    if version not in WORKFLOW_PLAN_SUPPORTED_SCHEMA_VERSIONS:
        raise UnsupportedWorkflowPlanVersionError(
            f"unsupported {field_ref} schema_version: {version!r}; supported versions are "
            f"{list(WORKFLOW_PLAN_SUPPORTED_SCHEMA_VERSIONS)}; migration_intentionally_absent=yes"
        )
    return dict(document)


def workflow_plan_from_document(document: Any, *, field_ref: str = "workflow_plan") -> WorkflowPlan:
    """Load one emitted plan document back into a plan, or fail closed."""
    admitted = read_workflow_plan_document(document, field_ref=field_ref)
    nodes = tuple(PlanNode.from_record(record) for record in admitted["nodes"])
    edges = tuple(PlanEdge.from_record(record) for record in admitted["edges"])
    plan = build_workflow_plan(
        LogicalKey.parse(admitted["target"]),
        nodes,
        edges,
        origin=admitted.get("origin") or {},
    )
    if admitted.get("identity") != plan.identity:
        raise WorkflowPlanIdentityError(
            f"{field_ref} declares identity {admitted.get('identity')!r}, but its canonical "
            f"semantic content hashes to {plan.identity}"
        )
    return plan


def _node_declaration_differences(first: PlanNode, second: PlanNode) -> list[str]:
    """Return one difference per fact two declarations of a node disagree on."""
    differences: list[str] = []
    first_record = first.record()
    second_record = second.record()
    for name in sorted(set(first_record) | set(second_record)):
        if first_record.get(name) != second_record.get(name):
            differences.append(
                f"{name}: one declaration states {first_record.get(name)!r} and the other "
                f"states {second_record.get(name)!r}"
            )
    return differences


def build_workflow_plan(
    target: LogicalKey,
    nodes: Iterable[PlanNode],
    edges: Iterable[PlanEdge],
    *,
    origin: Mapping[str, Any] | None = None,
) -> WorkflowPlan:
    """Assemble one closure from declared nodes and edges.

    The closure is what the target reaches: nodes no path from the target
    reaches, and the edges that name them, are not part of this plan. Within it,
    every reference must resolve, one logical key names one declaration, one
    ``(consumer, role_path)`` names one edge, and a cycle is named rather than
    silently truncating the plan.
    """
    declared: dict[LogicalKey, PlanNode] = {}
    for node in nodes:
        existing = declared.get(node.key)
        if existing is not None:
            if existing.source_ref != node.source_ref:
                raise DuplicateLogicalKeyError(node.key, existing.source_ref, node.source_ref)
            # Same key, same source ref: a repeat is admitted only when it is
            # the *same declaration*. Comparing the source ref alone let a
            # second entry restate the node's pinned document, its lowering
            # schema, its execution identity, or its operation and be dropped
            # without anything comparing what it said.
            differences = _node_declaration_differences(existing, node)
            if differences:
                raise ConflictingNodeDeclarationError(node.key, node.source_ref, differences)
            continue
        declared[node.key] = node
    if target not in declared:
        raise UnresolvedPlanReferenceError(target, relation="target")

    declared_edges = list(edges)
    inputs: dict[LogicalKey, list[PlanEdge]] = {key: [] for key in declared}
    at_role: dict[tuple[LogicalKey, tuple[str, ...]], PlanEdge] = {}
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
        input_role = ".".join(edge.role_path)
        declared_input_type = declared[edge.consumer].operation.input_types.get(input_role)
        if declared_input_type is None:
            raise WorkflowTypeMismatchError(
                f"{edge.consumer.text} binds undeclared input role {input_role!r}; operation "
                f"{declared[edge.consumer].operation.type_id!r} declares "
                f"{sorted(declared[edge.consumer].operation.input_types)}"
            )
        if declared_input_type != edge.input_type:
            raise WorkflowTypeMismatchError(
                f"{edge.consumer.text} input {input_role!r} declares type "
                f"{declared_input_type!r}, but its edge states {edge.input_type!r}"
            )
        if edge.producer is not None:
            producer = declared[edge.producer]
            output_type = producer.operation.output_types.get(edge.producer_output or "")
            if output_type is None:
                raise WorkflowTypeMismatchError(
                    f"{edge.consumer.text} input {input_role!r} names absent output "
                    f"{edge.producer.text}.{edge.producer_output}; available outputs are "
                    f"{sorted(producer.operation.output_types)}"
                )
            if output_type != edge.input_type:
                raise WorkflowTypeMismatchError(
                    f"{edge.consumer.text} input {input_role!r} requires {edge.input_type!r}, "
                    f"but {edge.producer.text}.{edge.producer_output} produces {output_type!r}"
                )
        if edge.external is not None and edge.external_type != edge.input_type:
            raise WorkflowTypeMismatchError(
                f"{edge.consumer.text} input {input_role!r} requires {edge.input_type!r}, "
                f"but its exact external binding declares {edge.external_type!r}"
            )
        if edge.guard is not None:
            if edge.guard.outcome not in declared:
                raise UnresolvedPlanReferenceError(
                    edge.guard.outcome,
                    referrer=edge.consumer,
                    role_path=edge.role_path,
                    relation="guard outcome",
                )
            if edge.guard.outcome == edge.consumer:
                raise PlanCycleError(target, (edge.consumer.text,))
            guard_type = declared[edge.guard.outcome].operation.output_types.get(
                edge.guard.output_role
            )
            if guard_type != edge.guard.output_type:
                raise WorkflowTypeMismatchError(
                    f"guard for {edge.consumer.text} input {input_role!r} declares outcome type "
                    f"{edge.guard.output_type!r}, but "
                    f"{edge.guard.outcome.text}.{edge.guard.output_role} produces {guard_type!r}"
                )
        # One role, one edge. This is proved over *every* declared edge rather
        # than over the closure, so a duplicate cannot survive by hiding on the
        # side of the plan a target does not reach and then be reached later.
        first = at_role.get((edge.consumer, edge.role_path))
        if first is not None:
            raise DuplicateInputEdgeError(edge.consumer, edge.role_path, first, edge)
        at_role[(edge.consumer, edge.role_path)] = edge
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
            if edge.guard is not None and edge.guard.outcome not in reached:
                frontier.append(edge.guard.outcome)

    closure_nodes = {key: node for key, node in declared.items() if key in reached}
    closure_edges = [edge for edge in declared_edges if edge.consumer in reached]
    return WorkflowPlan(
        target=target,
        nodes=_dependency_order(closure_nodes, closure_edges, target),
        edges=tuple(sorted(closure_edges, key=lambda item: item.sort_key)),
        origin=dict(origin or {}),
    )


@dataclass(frozen=True)
class EdgeDeclaration:
    """One input a node declares, before its producer has been addressed.

    An expander states inputs by source ref: a ``producer_ref`` is a ref the
    expander itself can expand, and the kernel is what turns it into a
    :class:`LogicalKey`. Everything else is exactly a :class:`PlanEdge`'s typed
    decision.
    """

    role_path: tuple[str, ...]
    status: str = "required"
    basis: str = "authored"
    binding: str | None = None
    input_type: str = "feedbax.artifact"
    reason: str | None = None
    producer_ref: str | None = None
    producer_output: str | None = None
    external: Mapping[str, Any] | None = None
    external_type: str | None = None
    rule: str | None = None
    guard: PlanGuard | None = None

    def edge(self, consumer: LogicalKey, producer: LogicalKey | None) -> PlanEdge:
        return PlanEdge(
            consumer=consumer,
            role_path=tuple(self.role_path),
            status=self.status,
            basis=self.basis,
            binding=self.binding,
            input_type=self.input_type,
            reason=self.reason,
            producer=producer,
            producer_output=self.producer_output,
            external=self.external,
            external_type=self.external_type,
            rule=self.rule,
            guard=self.guard,
        )


@dataclass(frozen=True)
class NodeDeclaration:
    """What one source ref expands to: a node and the inputs it declares."""

    node: PlanNode
    edges: tuple[EdgeDeclaration, ...] = ()


class NodeExpander(Protocol):
    """Expands one source ref into the node and inputs it declares.

    This splits plan construction in two: the kernel knows how to reach a
    closure, dedupe a diamond, refuse a key collision, and order the result;
    the expander knows what a declaration is and how to read it.
    :mod:`feedbax.workflow.derivation` is the expander the engine
    uses, and it reads compile locks.

    The source ref is positional-only, so an implementation names its parameter
    whatever its own vocabulary calls it.
    """

    def __call__(self, source_ref: str, /) -> NodeDeclaration:  # pragma: no cover - protocol
        ...


def expand_workflow_plan(
    target_ref: str,
    *,
    expand: NodeExpander,
    origin: Mapping[str, Any] | None = None,
) -> WorkflowPlan:
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
    return build_workflow_plan(
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
        dependencies = [edge.producer] if edge.producer is not None else []
        if edge.guard is not None and edge.guard.outcome not in dependencies:
            dependencies.append(edge.guard.outcome)
        for dependency in dependencies:
            if edge.consumer in consumers[dependency]:
                continue
            consumers[dependency].add(edge.consumer)
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
        raise PlanCycleError(target, sorted(key.text for key in nodes if key not in placed))
    return tuple(ordered)


__all__ = [
    "APPLICABILITY_BASES",
    "APPLICABILITY_STATUSES",
    "CERTIFYING_BASIS",
    "GUARD_OPERATORS",
    "LEGACY_FULFILLMENT_PLAN_SCHEMA_ID",
    "OPERATION_CACHE_POLICIES",
    "OPERATION_DETERMINISM",
    "OPERATION_EFFECTS",
    "WORKFLOW_PLAN_SCHEMA_ID",
    "WORKFLOW_PLAN_SCHEMA_VERSION",
    "WORKFLOW_PLAN_SUPPORTED_SCHEMA_VERSIONS",
    "ConflictingNodeDeclarationError",
    "DuplicateInputEdgeError",
    "DuplicateLogicalKeyError",
    "EdgeDeclaration",
    "WorkflowPlan",
    "WorkflowPlanError",
    "LogicalKey",
    "Operation",
    "NodeDeclaration",
    "NodeExpander",
    "PlanCycleError",
    "PlanEdge",
    "PlanGuard",
    "PlanNode",
    "GuardPredicate",
    "UnresolvedPlanReferenceError",
    "UnresolvedGuardOutcomeError",
    "UnsupportedWorkflowPlanVersionError",
    "WorkflowPlanIdentityError",
    "WorkflowTypeMismatchError",
    "build_workflow_plan",
    "expand_workflow_plan",
    "workflow_plan_from_document",
    "read_workflow_plan_document",
]
