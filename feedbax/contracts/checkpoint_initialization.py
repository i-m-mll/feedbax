"""Closed checkpoint initialize/continue lowering.

This is the simple lowering surface that eventually replaces the legacy
``feedbax.spec.training_run_composition.v1`` fork/graft vocabulary for the two
cases downstream authoring actually needs:

* ``initialize_from`` — a model-weight warm start. Optimizer, optimizer history,
  and PRNG state are fresh.
* ``continue_from`` — checkpoint continuation. Every state slot the target
  declares as required (model, optimizer, PRNG, batch counter, and adaptive
  state where the target program has one) must be preserved exactly.

The matching rule is closed and has no knobs:

* a target leaf is restored only when the source declares the *same canonical
  path* inside the *same slot*, that slot's PyTree definition matches exactly,
  and the leaf's type, shape, and dtype match exactly;
* target paths the source does not declare fresh-initialize;
* source paths the target does not declare are ignored;
* a path present on both sides that disagrees refuses with a structured error
  naming every offending path.

There is deliberately no structural search, no similarity scoring, and no
renaming map, so "which source subtree did this target subtree come from" can
never be ambiguous. Anything the closed rule cannot express is stop-and-ask, not
a heuristic.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from feedbax.contracts.checkpoints import (
    CheckpointSlotRole,
    CheckpointTransactionManifest,
    SlotLeafFingerprint,
)
from feedbax.contracts.base import (
    ParentRef,
    StrictModel,
    authenticated_manifest_ref_profile,
)

CHECKPOINT_STRUCTURE_SCHEMA_ID = "feedbax.spec.checkpoint_structure"
CHECKPOINT_STRUCTURE_SCHEMA_VERSION = "feedbax.spec.checkpoint_structure.v1"
CHECKPOINT_INITIALIZATION_SCHEMA_ID = "feedbax.spec.checkpoint_initialization"
CHECKPOINT_INITIALIZATION_SCHEMA_VERSION = "feedbax.spec.checkpoint_initialization.v1"
CHECKPOINT_INITIALIZATION_PLAN_SCHEMA_ID = "feedbax.spec.checkpoint_initialization_plan"
CHECKPOINT_INITIALIZATION_PLAN_SCHEMA_VERSION = "feedbax.spec.checkpoint_initialization_plan.v1"

CheckpointInitializationMode = Literal["initialize_from", "continue_from"]

#: Slot roles a model-weight warm start may read from the source. Everything
#: else is fresh by construction, not by policy lookup.
INITIALIZE_FROM_SOURCE_ROLES: frozenset[str] = frozenset({"model"})


class CheckpointInitializationErrorCode(StrEnum):
    """Stable refusal vocabulary for checkpoint initialize/continue lowering."""

    SLOT_TREEDEF_MISMATCH = "slot-treedef-mismatch"
    LEAF_MISMATCH = "leaf-mismatch"
    MISSING_REQUIRED_CONTINUATION_STATE = "missing-required-continuation-state"
    UNAUTHENTICATED_SOURCE = "unauthenticated-source"


class CheckpointInitializationRefused(ValueError):
    """One stable refusal code plus the exact target paths that caused it."""

    def __init__(
        self,
        code: CheckpointInitializationErrorCode,
        message: str,
        *,
        mode: str,
        paths: Sequence[str] = (),
        slots: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.mode = mode
        self.paths = tuple(paths)
        self.slots = tuple(slots)


class CheckpointSlotStructure(StrictModel):
    """Exact canonical structure of one checkpoint slot.

    ``required`` is the target program's own declaration that this slot is part
    of its continuation state. Feedbax never infers which slots a program needs.
    """

    slot: str = Field(min_length=1)
    role: CheckpointSlotRole = "auxiliary"
    required: bool = True
    treedef: str = Field(min_length=1)
    leaves: list[SlotLeafFingerprint] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_leaf_paths(self) -> "CheckpointSlotStructure":
        paths = [leaf.path for leaf in self.leaves]
        duplicates = sorted({path for path in paths if paths.count(path) > 1})
        if duplicates:
            raise ValueError(
                f"checkpoint slot {self.slot!r} declares duplicate leaf paths: {duplicates}"
            )
        return self


class CheckpointStructure(StrictModel):
    """Exact canonical slot structure for one checkpoint side of the lowering."""

    schema_id: Literal["feedbax.spec.checkpoint_structure"] = CHECKPOINT_STRUCTURE_SCHEMA_ID
    schema_version: Literal["feedbax.spec.checkpoint_structure.v1"] = (
        CHECKPOINT_STRUCTURE_SCHEMA_VERSION
    )
    slots: list[CheckpointSlotStructure] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_slots(self) -> "CheckpointStructure":
        names = [slot.slot for slot in self.slots]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"checkpoint structure declares duplicate slots: {duplicates}")
        return self

    def slot(self, name: str) -> CheckpointSlotStructure | None:
        """Return one declared slot by name, or ``None``."""
        for entry in self.slots:
            if entry.slot == name:
                return entry
        return None


class CheckpointInitializationRequest(StrictModel):
    """Authored request to warm-start or continue from one source checkpoint."""

    schema_id: Literal["feedbax.spec.checkpoint_initialization"] = (
        CHECKPOINT_INITIALIZATION_SCHEMA_ID
    )
    schema_version: Literal["feedbax.spec.checkpoint_initialization.v1"] = (
        CHECKPOINT_INITIALIZATION_SCHEMA_VERSION
    )
    mode: CheckpointInitializationMode
    source: ParentRef

    @model_validator(mode="after")
    def _validate_source_authentication(self) -> "CheckpointInitializationRequest":
        if authenticated_manifest_ref_profile(self.source) is None:
            raise CheckpointInitializationRefused(
                CheckpointInitializationErrorCode.UNAUTHENTICATED_SOURCE,
                f"checkpoint {self.mode} source {self.source.id!r} must be an authenticated "
                "manifest ref",
                mode=self.mode,
            )
        return self


class CheckpointLeafDisagreement(StrictModel):
    """One exact disagreement between a source and target leaf at one path."""

    path: str
    field: Literal["leaf_type", "shape", "dtype", "weak_type", "static_repr_sha256"]
    source: str | None = None
    target: str | None = None


class CheckpointSlotInitializationPlan(StrictModel):
    """Lowered per-slot outcome under the closed matching rule."""

    slot: str
    role: CheckpointSlotRole
    action: Literal["restore", "fresh"]
    fresh_reason: Literal["mode_scope", "absent_from_source", "not_applicable"] = "not_applicable"
    restored_paths: list[str] = Field(default_factory=list)
    fresh_paths: list[str] = Field(default_factory=list)
    ignored_source_paths: list[str] = Field(default_factory=list)


class CheckpointInitializationPlan(StrictModel):
    """Complete lowering of one initialize/continue request to explicit actions."""

    schema_id: Literal["feedbax.spec.checkpoint_initialization_plan"] = (
        CHECKPOINT_INITIALIZATION_PLAN_SCHEMA_ID
    )
    schema_version: Literal["feedbax.spec.checkpoint_initialization_plan.v1"] = (
        CHECKPOINT_INITIALIZATION_PLAN_SCHEMA_VERSION
    )
    mode: CheckpointInitializationMode
    source: ParentRef
    slots: list[CheckpointSlotInitializationPlan] = Field(default_factory=list)
    ignored_source_slots: list[str] = Field(default_factory=list)

    @property
    def restored_paths(self) -> tuple[str, ...]:
        """Return every canonical ``<slot>/<leaf>`` path restored from the source."""
        return tuple(
            f"{plan.slot}/{path}" for plan in self.slots for path in plan.restored_paths
        )

    @property
    def fresh_paths(self) -> tuple[str, ...]:
        """Return every canonical ``<slot>/<leaf>`` path initialized fresh."""
        return tuple(f"{plan.slot}/{path}" for plan in self.slots for path in plan.fresh_paths)


def checkpoint_structure_from_manifest(
    manifest: CheckpointTransactionManifest,
) -> CheckpointStructure:
    """Project one durable checkpoint transaction into its canonical structure."""
    return CheckpointStructure(
        slots=[
            CheckpointSlotStructure(
                slot=slot.slot,
                role=slot.role,
                required=slot.required,
                treedef=slot.structural_abi_fingerprint.treedef,
                leaves=list(slot.structural_abi_fingerprint.leaves),
            )
            for slot in manifest.slots
        ]
    )


def _leaf_disagreements(
    slot: str,
    source: SlotLeafFingerprint,
    target: SlotLeafFingerprint,
) -> list[CheckpointLeafDisagreement]:
    disagreements: list[CheckpointLeafDisagreement] = []
    for field_name in ("leaf_type", "shape", "dtype", "weak_type", "static_repr_sha256"):
        source_value = getattr(source, field_name)
        target_value = getattr(target, field_name)
        if source_value != target_value:
            disagreements.append(
                CheckpointLeafDisagreement(
                    path=f"{slot}/{target.path}",
                    field=field_name,
                    source=None if source_value is None else str(source_value),
                    target=None if target_value is None else str(target_value),
                )
            )
    return disagreements


def _leaves_by_path(leaves: Iterable[SlotLeafFingerprint]) -> Mapping[str, SlotLeafFingerprint]:
    return {leaf.path: leaf for leaf in leaves}


def lower_checkpoint_initialization(
    request: CheckpointInitializationRequest,
    *,
    source: CheckpointStructure,
    target: CheckpointStructure,
) -> CheckpointInitializationPlan:
    """Lower one initialize/continue request under the closed matching rule.

    Args:
        request: The authored mode plus the authenticated source checkpoint ref.
        source: Canonical structure of the source checkpoint.
        target: Canonical structure the target program will construct fresh.

    Returns:
        An explicit per-slot plan naming exactly which canonical paths are
        restored, which fresh-initialize, and which source paths are ignored.

    Raises:
        CheckpointInitializationRefused: If a slot present on both sides has a
            different PyTree definition, if a path present on both sides
            disagrees on leaf type, shape, or dtype, or if a ``continue_from``
            target declares required continuation state the source cannot supply.
    """
    in_scope_source_slots = {
        slot.slot: slot
        for slot in source.slots
        if request.mode == "continue_from" or slot.role in INITIALIZE_FROM_SOURCE_ROLES
    }

    treedef_mismatched_slots: list[str] = []
    disagreements: list[CheckpointLeafDisagreement] = []
    slot_plans: list[CheckpointSlotInitializationPlan] = []
    matched_source_slots: set[str] = set()

    for target_slot in target.slots:
        source_slot = in_scope_source_slots.get(target_slot.slot)
        if source_slot is None:
            out_of_scope = (
                request.mode == "initialize_from"
                and source.slot(target_slot.slot) is not None
            )
            slot_plans.append(
                CheckpointSlotInitializationPlan(
                    slot=target_slot.slot,
                    role=target_slot.role,
                    action="fresh",
                    fresh_reason="mode_scope" if out_of_scope else "absent_from_source",
                    fresh_paths=[leaf.path for leaf in target_slot.leaves],
                )
            )
            continue
        matched_source_slots.add(source_slot.slot)
        if source_slot.treedef != target_slot.treedef:
            treedef_mismatched_slots.append(target_slot.slot)
            continue
        source_leaves = _leaves_by_path(source_slot.leaves)
        target_leaves = _leaves_by_path(target_slot.leaves)
        restored: list[str] = []
        fresh: list[str] = []
        for leaf in target_slot.leaves:
            source_leaf = source_leaves.get(leaf.path)
            if source_leaf is None:
                fresh.append(leaf.path)
                continue
            leaf_disagreements = _leaf_disagreements(target_slot.slot, source_leaf, leaf)
            if leaf_disagreements:
                disagreements.extend(leaf_disagreements)
                continue
            restored.append(leaf.path)
        slot_plans.append(
            CheckpointSlotInitializationPlan(
                slot=target_slot.slot,
                role=target_slot.role,
                action="restore" if restored else "fresh",
                fresh_reason="absent_from_source" if not restored else "not_applicable",
                restored_paths=restored,
                fresh_paths=fresh,
                ignored_source_paths=sorted(set(source_leaves) - set(target_leaves)),
            )
        )

    if treedef_mismatched_slots:
        raise CheckpointInitializationRefused(
            CheckpointInitializationErrorCode.SLOT_TREEDEF_MISMATCH,
            f"checkpoint {request.mode} refuses: source and target PyTree definitions differ "
            f"for slots {sorted(treedef_mismatched_slots)}; the closed matching rule does not "
            "search for structurally similar subtrees",
            mode=request.mode,
            slots=sorted(treedef_mismatched_slots),
            paths=sorted(treedef_mismatched_slots),
        )
    if disagreements:
        detail = ", ".join(
            f"{item.path} {item.field} source={item.source!r} target={item.target!r}"
            for item in disagreements
        )
        raise CheckpointInitializationRefused(
            CheckpointInitializationErrorCode.LEAF_MISMATCH,
            f"checkpoint {request.mode} refuses: source and target disagree at present paths: "
            f"{detail}",
            mode=request.mode,
            paths=sorted({item.path for item in disagreements}),
            slots=sorted({item.path.split("/", 1)[0] for item in disagreements}),
        )

    if request.mode == "continue_from":
        unsatisfied = {
            plan.slot: plan.fresh_paths
            for plan in slot_plans
            if plan.fresh_paths
            and (entry := target.slot(plan.slot)) is not None
            and entry.required
        }
        if unsatisfied:
            paths = sorted(
                f"{slot}/{path}" for slot, values in unsatisfied.items() for path in values
            )
            raise CheckpointInitializationRefused(
                CheckpointInitializationErrorCode.MISSING_REQUIRED_CONTINUATION_STATE,
                f"checkpoint continue_from refuses: the source cannot supply required "
                f"continuation state at {paths}; a warm start that discards this state must "
                "be authored as initialize_from",
                mode=request.mode,
                paths=paths,
                slots=sorted(unsatisfied),
            )

    return CheckpointInitializationPlan(
        mode=request.mode,
        source=request.source,
        slots=slot_plans,
        ignored_source_slots=sorted(
            slot.slot for slot in source.slots if slot.slot not in matched_source_slots
        ),
    )


__all__ = [
    "CHECKPOINT_INITIALIZATION_PLAN_SCHEMA_ID",
    "CHECKPOINT_INITIALIZATION_PLAN_SCHEMA_VERSION",
    "CHECKPOINT_INITIALIZATION_SCHEMA_ID",
    "CHECKPOINT_INITIALIZATION_SCHEMA_VERSION",
    "CHECKPOINT_STRUCTURE_SCHEMA_ID",
    "CHECKPOINT_STRUCTURE_SCHEMA_VERSION",
    "INITIALIZE_FROM_SOURCE_ROLES",
    "CheckpointInitializationErrorCode",
    "CheckpointInitializationMode",
    "CheckpointInitializationPlan",
    "CheckpointInitializationRefused",
    "CheckpointInitializationRequest",
    "CheckpointLeafDisagreement",
    "CheckpointSlotInitializationPlan",
    "CheckpointSlotStructure",
    "CheckpointStructure",
    "checkpoint_structure_from_manifest",
    "lower_checkpoint_initialization",
]
