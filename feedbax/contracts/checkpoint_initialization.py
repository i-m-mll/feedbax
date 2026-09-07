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

* a target leaf is restored only when the source declares the same semantic
  owner identity and the leaf's type, shape, and dtype match exactly;
* required target identities must exist, while declared optional identities may
  fresh-initialize;
* unknown required source identities refuse instead of being silently dropped;
* raw PyTree paths and definitions remain integrity and diagnostic evidence, not
  copy identity.

There is deliberately no structural search, similarity scoring, or inferred
renaming map. Component slots, parameter roles, and optimizer target bindings
make ownership explicit; anything they cannot express is stop-and-ask.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from feedbax.contracts.checkpoints import (
    CheckpointSemanticAbi,
    CheckpointSemanticLeafIdentity,
    CheckpointSlotRole,
    CheckpointTransactionManifest,
    SlotLeafFingerprint,
    StructuralAbiFingerprint,
)
from feedbax.contracts.base import (
    ParentRef,
    StrictModel,
    authenticated_manifest_ref_profile,
    canonical_json_bytes,
    sha256_bytes,
)
from feedbax.contracts.worker import OptimizerTargetBinding

CHECKPOINT_STRUCTURE_SCHEMA_ID = "feedbax.spec.checkpoint_structure"
CHECKPOINT_STRUCTURE_SCHEMA_VERSION_V1 = "feedbax.spec.checkpoint_structure.v1"
CHECKPOINT_STRUCTURE_SCHEMA_VERSION = "feedbax.spec.checkpoint_structure.v2"
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

    UNKNOWN_IDENTITY = "unknown-identity"
    DUPLICATE_IDENTITY = "duplicate-identity"
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
    """Semantic copy identity plus raw integrity evidence for one slot.

    ``required`` is the target program's own declaration that this slot is part
    of its continuation state. Feedbax never infers which slots a program needs.
    """

    slot: str = Field(min_length=1)
    role: CheckpointSlotRole = "auxiliary"
    required: bool = True
    treedef: str = Field(min_length=1)
    leaves: list[SlotLeafFingerprint] = Field(default_factory=list)
    semantic_abi: CheckpointSemanticAbi

    @model_validator(mode="after")
    def _validate_leaf_paths(self) -> "CheckpointSlotStructure":
        paths = [leaf.path for leaf in self.leaves]
        duplicates = sorted({path for path in paths if paths.count(path) > 1})
        if duplicates:
            raise ValueError(
                f"checkpoint slot {self.slot!r} declares duplicate leaf paths: {duplicates}"
            )
        if {leaf.path for leaf in self.leaves} != {
            leaf.raw_path for leaf in self.semantic_abi.leaves
        }:
            raise ValueError(
                f"checkpoint slot {self.slot!r} semantic identities do not cover exactly "
                "the raw structural leaves"
            )
        return self


class CheckpointStructure(StrictModel):
    """Exact canonical slot structure for one checkpoint side of the lowering."""

    schema_id: Literal["feedbax.spec.checkpoint_structure"] = CHECKPOINT_STRUCTURE_SCHEMA_ID
    schema_version: Literal["feedbax.spec.checkpoint_structure.v2"] = (
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


class CheckpointIdentityDiagnostic(StrictModel):
    """Generated explanation of a semantic ABI classification."""

    code: Literal["treedef-differs", "optional-source-only", "optional-target-only"]
    slot: str
    identity: str | None = None
    source_path: str | None = None
    target_path: str | None = None


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
    diagnostics: list[CheckpointIdentityDiagnostic] = Field(default_factory=list)

    @property
    def restored_paths(self) -> tuple[str, ...]:
        """Return every canonical ``<slot>/<leaf>`` path restored from the source."""
        return tuple(
            _qualified_leaf_path(plan.slot, path)
            for plan in self.slots
            for path in plan.restored_paths
        )

    @property
    def fresh_paths(self) -> tuple[str, ...]:
        """Return every canonical ``<slot>/<leaf>`` path initialized fresh."""
        return tuple(
            _qualified_leaf_path(plan.slot, path)
            for plan in self.slots
            for path in plan.fresh_paths
        )


def checkpoint_semantic_abi_sha256(
    leaves: Sequence[CheckpointSemanticLeafIdentity | Mapping[str, Any]],
) -> str:
    """Hash the ordered-by-identity semantic ownership projection."""
    projection = [
        leaf.model_dump(mode="json", exclude_none=True)
        if isinstance(leaf, CheckpointSemanticLeafIdentity)
        else dict(leaf)
        for leaf in leaves
    ]
    projection.sort(key=lambda item: str(item["identity"]))
    return sha256_bytes(canonical_json_bytes(projection))


def _path_parts(path: str) -> tuple[str, ...]:
    return tuple(part for part in path.strip("/").split("/") if part)


def _qualified_leaf_path(slot: str, path: str) -> str:
    return f"{slot}/{path.lstrip('/')}"


def _model_leaf_identity(
    *,
    slot: str,
    path: str,
) -> CheckpointSemanticLeafIdentity:
    parts = _path_parts(path)
    if not parts:
        parts = ("root",)
    node_parts = [
        parts[index + 1]
        for index, part in enumerate(parts[:-1])
        if part == "nodes"
    ]
    last_node_index = max(
        (index + 1 for index, part in enumerate(parts[:-1]) if part == "nodes"),
        default=-1,
    )
    role_parts = parts[last_node_index + 1 :] if last_node_index >= 0 else parts
    component_slot = "/".join(node_parts) if node_parts else "root"

    if "initial_hidden" in role_parts and role_parts[-1] in {"bias", "weight"}:
        hidden_index = role_parts.index("initial_hidden")
        component_suffix = role_parts[: hidden_index + 1]
        if component_suffix:
            component_slot = "/".join((component_slot, *component_suffix))
        parameter_role = (
            "initial_hidden.baseline" if role_parts[-1] == "bias" else "initial_hidden.encoder"
        )
        required = role_parts[-1] == "bias"
    else:
        parameter_role = "/".join(role_parts)
        required = True
    identity = f"model:{slot}:component:{component_slot}:parameter:{parameter_role}"
    return CheckpointSemanticLeafIdentity(
        identity=identity,
        raw_path=path,
        owner_slot=slot,
        component_slot=component_slot,
        parameter_role=parameter_role,
        required=required,
    )


def _optimizer_leaf_identity(
    *,
    slot: str,
    path: str,
    binding: OptimizerTargetBinding,
    target_leaves: Sequence[CheckpointSemanticLeafIdentity],
) -> CheckpointSemanticLeafIdentity:
    matches = [
        leaf
        for leaf in target_leaves
        if path == leaf.raw_path or path.endswith("/" + leaf.raw_path.lstrip("/"))
    ]
    path_names = [part for part in _path_parts(path) if not part.isdigit()]
    if (
        not matches
        and len(target_leaves) == 1
        and target_leaves[0].raw_path == "/"
        and path_names
        and path_names[-1] not in {"count", "step"}
    ):
        matches = [target_leaves[0]]
    if len(matches) > 1:
        raise ValueError(
            f"checkpoint optimizer slot {slot!r} path {path!r} ambiguously owns model "
            f"identities {[leaf.identity for leaf in matches]!r}"
        )
    if matches:
        target = matches[0]
        prefix = (
            path.strip("/")
            if target.raw_path == "/"
            else path[: -len(target.raw_path)].strip("/")
        )
        named_prefix = [part for part in prefix.split("/") if part and not part.isdigit()]
        if not named_prefix and path == "/" and target.raw_path == "/":
            named_prefix = ["value"]
        elif not named_prefix:
            raise ValueError(
                f"checkpoint optimizer slot {slot!r} path {path!r} has no named moment role"
            )
        optimizer_role = "/".join(named_prefix)
        identity = (
            f"optimizer:{binding.name}:role:{optimizer_role}:target:{target.identity}"
        )
        return CheckpointSemanticLeafIdentity(
            identity=identity,
            raw_path=path,
            owner_slot=slot,
            component_slot=target.component_slot,
            parameter_role=target.parameter_role,
            optimizer_binding=binding.name,
            optimizer_role=optimizer_role,
            required=target.required,
        )

    named_parts = [part for part in _path_parts(path) if not part.isdigit()]
    if not named_parts:
        raise ValueError(
            f"checkpoint optimizer slot {slot!r} path {path!r} has no stable named role"
        )
    optimizer_role = "/".join(named_parts)
    return CheckpointSemanticLeafIdentity(
        identity=f"optimizer:{binding.name}:state:{optimizer_role}",
        raw_path=path,
        owner_slot=slot,
        component_slot="optimizer-state",
        parameter_role=optimizer_role,
        optimizer_binding=binding.name,
        optimizer_role=optimizer_role,
    )


def _disambiguate_optimizer_state_leaves(
    leaves: Sequence[CheckpointSemanticLeafIdentity],
) -> list[CheckpointSemanticLeafIdentity]:
    """Retain numeric member identity only where opaque optimizer state needs it."""
    identity_counts: dict[str, int] = {}
    for leaf in leaves:
        identity_counts[leaf.identity] = identity_counts.get(leaf.identity, 0) + 1

    disambiguated: list[CheckpointSemanticLeafIdentity] = []
    for leaf in leaves:
        if identity_counts[leaf.identity] == 1 or leaf.component_slot != "optimizer-state":
            disambiguated.append(leaf)
            continue
        member_path = "/".join(
            f"{index}={part}"
            for index, part in enumerate(_path_parts(leaf.raw_path))
            if part.isdigit()
        )
        if not member_path or leaf.optimizer_binding is None or leaf.optimizer_role is None:
            disambiguated.append(leaf)
            continue
        optimizer_role = f"{leaf.optimizer_role}/member/{member_path}"
        disambiguated.append(
            leaf.model_copy(
                update={
                    "identity": (f"optimizer:{leaf.optimizer_binding}:state:{optimizer_role}"),
                    "parameter_role": optimizer_role,
                    "optimizer_role": optimizer_role,
                }
            )
        )
    return disambiguated


def derive_checkpoint_semantic_abis(
    fingerprints: Mapping[str, StructuralAbiFingerprint],
    *,
    slot_roles: Mapping[str, CheckpointSlotRole | str],
    optimizer_bindings: Sequence[OptimizerTargetBinding | Mapping[str, Any]] = (),
) -> dict[str, CheckpointSemanticAbi]:
    """Derive the one canonical named-owner map used by all checkpoint consumers.

    Model leaves are owned by named graph component slots and parameter roles.
    Optimizer moments are owned by their declared optimizer binding, named moment
    role, and the target model identity they track. Raw PyTree paths remain only
    integrity coordinates.
    """
    bindings = [
        binding
        if isinstance(binding, OptimizerTargetBinding)
        else OptimizerTargetBinding.model_validate(binding)
        for binding in optimizer_bindings
    ]
    bindings_by_slot: dict[str, OptimizerTargetBinding] = {}
    for binding in bindings:
        if binding.optimizer_slot in bindings_by_slot:
            raise ValueError(
                f"checkpoint optimizer slot {binding.optimizer_slot!r} has ambiguous bindings"
            )
        bindings_by_slot[binding.optimizer_slot] = binding

    identities: dict[str, list[CheckpointSemanticLeafIdentity]] = {}
    for slot, fingerprint in fingerprints.items():
        if slot_roles.get(slot) != "model":
            continue
        identities[slot] = [
            _model_leaf_identity(slot=slot, path=leaf.path) for leaf in fingerprint.leaves
        ]

    for slot, fingerprint in fingerprints.items():
        role = slot_roles.get(slot, "auxiliary")
        if role == "model":
            continue
        if role == "optimizer":
            binding = bindings_by_slot.get(slot)
            if binding is None:
                raise ValueError(
                    f"checkpoint optimizer slot {slot!r} has no declared optimizer binding"
                )
            target = identities.get(binding.target_slot)
            if target is None:
                raise ValueError(
                    f"checkpoint optimizer binding {binding.name!r} targets unknown model slot "
                    f"{binding.target_slot!r}"
                )
            identities[slot] = _disambiguate_optimizer_state_leaves(
                [
                    _optimizer_leaf_identity(
                        slot=slot,
                        path=leaf.path,
                        binding=binding,
                        target_leaves=target,
                    )
                    for leaf in fingerprint.leaves
                ]
            )
            continue
        identities[slot] = [
            CheckpointSemanticLeafIdentity(
                identity=f"slot:{slot}:role:{'/'.join(_path_parts(leaf.path)) or 'root'}",
                raw_path=leaf.path,
                owner_slot=slot,
                component_slot="slot",
                parameter_role="/".join(_path_parts(leaf.path)) or "root",
            )
            for leaf in fingerprint.leaves
        ]

    return {
        slot: CheckpointSemanticAbi(
            leaves=leaves,
            fingerprint_sha256=checkpoint_semantic_abi_sha256(leaves),
        )
        for slot, leaves in identities.items()
    }


def derive_legacy_checkpoint_semantic_abi(
    *,
    slot: str,
    role: str,
    fingerprint: StructuralAbiFingerprint,
) -> CheckpointSemanticAbi:
    """Migrate a path-only v1 structure without inventing cross-slot ownership.

    Model identities, including the established initial-hidden baseline/encoder
    mapping, are deterministic. Historical optimizer structures lack their phase
    binding, so they receive an explicit legacy identity and will reject against a
    current bound optimizer ABI instead of being guessed into compatibility.
    """
    if role == "model":
        leaves = [_model_leaf_identity(slot=slot, path=leaf.path) for leaf in fingerprint.leaves]
    else:
        leaves = [
            CheckpointSemanticLeafIdentity(
                identity=f"legacy:{role}:{slot}:{'/'.join(_path_parts(leaf.path)) or 'root'}",
                raw_path=leaf.path,
                owner_slot=slot,
                component_slot="legacy-path",
                parameter_role="/".join(_path_parts(leaf.path)) or "root",
            )
            for leaf in fingerprint.leaves
        ]
    return CheckpointSemanticAbi(
        algorithm_version=(
            "feedbax.training_checkpoint.semantic_abi.legacy_path.v1"
            if role != "model"
            else "feedbax.training_checkpoint.semantic_abi.v1"
        ),
        leaves=leaves,
        fingerprint_sha256=checkpoint_semantic_abi_sha256(leaves),
    )


def validate_checkpoint_semantic_abi(abi: CheckpointSemanticAbi) -> None:
    """Fail closed when a persisted semantic ownership digest is stale."""
    expected = checkpoint_semantic_abi_sha256(abi.leaves)
    if abi.fingerprint_sha256 != expected:
        raise ValueError(
            "checkpoint semantic ABI fingerprint is stale; "
            f"recorded={abi.fingerprint_sha256!r} expected={expected!r}"
        )


def checkpoint_structure_from_manifest(
    manifest: CheckpointTransactionManifest,
) -> CheckpointStructure:
    """Project one durable checkpoint transaction into its canonical structure."""
    missing = [slot.slot for slot in manifest.slots if slot.semantic_abi is None]
    if missing:
        raise ValueError(
            "checkpoint transaction has no semantic ABI for slots "
            f"{missing!r}; migrate it through the registered transaction schema first"
        )
    return CheckpointStructure(
        slots=[
            CheckpointSlotStructure(
                slot=slot.slot,
                role=slot.role,
                required=slot.required,
                treedef=slot.structural_abi_fingerprint.treedef,
                leaves=list(slot.structural_abi_fingerprint.leaves),
                semantic_abi=slot.semantic_abi,
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
                    path=_qualified_leaf_path(slot, target.path),
                    field=field_name,
                    source=None if source_value is None else str(source_value),
                    target=None if target_value is None else str(target_value),
                )
            )
    return disagreements


def _leaves_by_path(leaves: Iterable[SlotLeafFingerprint]) -> Mapping[str, SlotLeafFingerprint]:
    return {leaf.path: leaf for leaf in leaves}


def _semantic_leaves_by_identity(
    leaves: Iterable[CheckpointSemanticLeafIdentity],
) -> Mapping[str, CheckpointSemanticLeafIdentity]:
    return {leaf.identity: leaf for leaf in leaves}


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
        CheckpointInitializationRefused: If semantic identities are unknown or
            ambiguous, corresponding leaves disagree on type, shape, or dtype,
            or a ``continue_from`` target declares required state the source
            cannot supply.
    """
    in_scope_source_slots = {
        slot.slot: slot
        for slot in source.slots
        if request.mode == "continue_from" or slot.role in INITIALIZE_FROM_SOURCE_ROLES
    }

    disagreements: list[CheckpointLeafDisagreement] = []
    slot_plans: list[CheckpointSlotInitializationPlan] = []
    matched_source_slots: set[str] = set()
    diagnostics: list[CheckpointIdentityDiagnostic] = []
    required_missing_paths: dict[str, list[str]] = {}

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
            if target_slot.required:
                required_missing_paths[target_slot.slot] = [
                    leaf.raw_path for leaf in target_slot.semantic_abi.leaves if leaf.required
                ]
            continue
        matched_source_slots.add(source_slot.slot)
        if source_slot.treedef != target_slot.treedef:
            diagnostics.append(
                CheckpointIdentityDiagnostic(code="treedef-differs", slot=target_slot.slot)
            )
        source_raw_leaves = _leaves_by_path(source_slot.leaves)
        target_raw_leaves = _leaves_by_path(target_slot.leaves)
        source_leaves = _semantic_leaves_by_identity(source_slot.semantic_abi.leaves)
        target_leaves = _semantic_leaves_by_identity(target_slot.semantic_abi.leaves)
        restored: list[str] = []
        fresh: list[str] = []
        for identity, target_identity in target_leaves.items():
            source_identity = source_leaves.get(identity)
            if source_identity is None:
                fresh.append(target_identity.raw_path)
                if target_identity.required:
                    required_missing_paths.setdefault(target_slot.slot, []).append(
                        target_identity.raw_path
                    )
                else:
                    diagnostics.append(
                        CheckpointIdentityDiagnostic(
                            code="optional-target-only",
                            slot=target_slot.slot,
                            identity=identity,
                            target_path=target_identity.raw_path,
                        )
                    )
                continue
            source_leaf = source_raw_leaves[source_identity.raw_path]
            target_leaf = target_raw_leaves[target_identity.raw_path]
            leaf_disagreements = _leaf_disagreements(target_slot.slot, source_leaf, target_leaf)
            if leaf_disagreements:
                disagreements.extend(leaf_disagreements)
                continue
            restored.append(target_identity.raw_path)
        source_only = sorted(set(source_leaves) - set(target_leaves))
        unknown_source = [identity for identity in source_only if source_leaves[identity].required]
        if unknown_source:
            raise CheckpointInitializationRefused(
                CheckpointInitializationErrorCode.UNKNOWN_IDENTITY,
                f"checkpoint {request.mode} refuses unknown required source identities "
                f"{unknown_source!r} in slot {target_slot.slot!r}",
                mode=request.mode,
                paths=[source_leaves[identity].raw_path for identity in unknown_source],
                slots=[target_slot.slot],
            )
        for identity in source_only:
            diagnostics.append(
                CheckpointIdentityDiagnostic(
                    code="optional-source-only",
                    slot=target_slot.slot,
                    identity=identity,
                    source_path=source_leaves[identity].raw_path,
                )
            )
        slot_plans.append(
            CheckpointSlotInitializationPlan(
                slot=target_slot.slot,
                role=target_slot.role,
                action="restore" if restored else "fresh",
                fresh_reason="absent_from_source" if not restored else "not_applicable",
                restored_paths=restored,
                fresh_paths=fresh,
                ignored_source_paths=sorted(
                    source_leaves[identity].raw_path for identity in source_only
                ),
            )
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

    unmatched_source_slots = [
        slot
        for slot in source.slots
        if slot.slot not in matched_source_slots
        and (request.mode == "continue_from" or slot.role in INITIALIZE_FROM_SOURCE_ROLES)
    ]
    unknown_source_slots = [
        slot.slot
        for slot in unmatched_source_slots
        if any(leaf.required for leaf in slot.semantic_abi.leaves)
    ]
    if unknown_source_slots:
        raise CheckpointInitializationRefused(
            CheckpointInitializationErrorCode.UNKNOWN_IDENTITY,
            f"checkpoint {request.mode} refuses unknown required source slots "
            f"{unknown_source_slots!r}",
            mode=request.mode,
            slots=unknown_source_slots,
        )

    if request.mode == "continue_from":
        unsatisfied = required_missing_paths
        if unsatisfied:
            paths = sorted(
                _qualified_leaf_path(slot, path)
                for slot, values in unsatisfied.items()
                for path in values
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
        diagnostics=diagnostics,
    )


__all__ = [
    "CHECKPOINT_INITIALIZATION_PLAN_SCHEMA_ID",
    "CHECKPOINT_INITIALIZATION_PLAN_SCHEMA_VERSION",
    "CHECKPOINT_INITIALIZATION_SCHEMA_ID",
    "CHECKPOINT_INITIALIZATION_SCHEMA_VERSION",
    "CHECKPOINT_STRUCTURE_SCHEMA_ID",
    "CHECKPOINT_STRUCTURE_SCHEMA_VERSION",
    "CHECKPOINT_STRUCTURE_SCHEMA_VERSION_V1",
    "INITIALIZE_FROM_SOURCE_ROLES",
    "CheckpointInitializationErrorCode",
    "CheckpointInitializationMode",
    "CheckpointInitializationPlan",
    "CheckpointInitializationRefused",
    "CheckpointInitializationRequest",
    "CheckpointIdentityDiagnostic",
    "CheckpointLeafDisagreement",
    "CheckpointSlotInitializationPlan",
    "CheckpointSlotStructure",
    "CheckpointStructure",
    "checkpoint_structure_from_manifest",
    "checkpoint_semantic_abi_sha256",
    "derive_checkpoint_semantic_abis",
    "derive_legacy_checkpoint_semantic_abi",
    "lower_checkpoint_initialization",
    "validate_checkpoint_semantic_abi",
]
