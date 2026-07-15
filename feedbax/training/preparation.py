"""Runtime-only preparation hooks for plugin-owned training methods."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence, Set
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import jax.tree as jt
import jax.tree_util as jtu
import numpy as np
from pydantic import BaseModel

from feedbax.contracts.checkpoint_history import BatchHistory
from feedbax.contracts.training import TrainingRunSpec
from feedbax.contracts.spec_storage import training_spec_sha256
from feedbax.contracts.worker import (
    AxisCoordinateSpec,
    EffectivePhaseSpec,
    MaterializedMappingLevelSpec,
    MaterializedSlotAxisBinding,
    MethodContractSpec,
    validate_worker_identifier,
)
from feedbax.objectives.service import LossService
from feedbax.training.checkpoint_custody import ResumeSlotTransform
from feedbax.training.worker_validation import resolve_execution_mapping

_RESERVED_KERNEL_CONTEXT_KEYS = frozenset({"run_spec", "method_payload"})
PREPARATION_RNG_ALGORITHM_VERSION = "feedbax.preparation_rng_scope.fold_in.v1"
_PREPARATION_RNG_DOMAIN = b"feedbax.preparation_rng_scope\0"
_MATERIALIZED_PREPARATION_SEAL = object()


@jtu.register_pytree_with_keys_class
class _ImmutableDict(dict[Any, Any]):
    """Recursively frozen mapping that remains PyTree/deepcopy compatible."""

    def _reject(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("immutable Feedbax preparation mapping")

    __setitem__ = _reject
    __delitem__ = _reject
    clear = _reject
    pop = _reject
    popitem = _reject
    setdefault = _reject
    update = _reject

    def __deepcopy__(self, _memo: dict[int, Any]) -> "_ImmutableDict":
        return self

    def tree_flatten_with_keys(self):
        keys = tuple(self)
        return tuple((jtu.DictKey(key), self[key]) for key in keys), keys

    @classmethod
    def tree_unflatten(cls, keys, values):
        return cls(zip(keys, values, strict=True))


class ExecutionPreparationError(ValueError):
    """Raised when a training method cannot prepare its runtime inputs."""


@dataclass(frozen=True)
class ExecutionPreparationRequest:
    """Validated runtime request passed to an execution-preparation provider."""

    run_spec: TrainingRunSpec
    method_payload: BaseModel | None = None
    method_contract: MethodContractSpec | None = None
    effective_phase: EffectivePhaseSpec | None = None
    run_id: str | None = None
    resume: bool = False


@dataclass(frozen=True)
class ExecutionPreparationResult:
    """Narrow set of runtime-only values a provider may supply to the executor."""

    initial_slots: Mapping[str, Any]
    kernel_context: Mapping[str, Any] = field(default_factory=dict)
    loss_service: LossService | None = None
    resume_slot_transform: ResumeSlotTransform | None = None


@dataclass(frozen=True)
class PreparationRngScope:
    """Immutable named keys derived for one scalar axis coordinate."""

    algorithm_version: str
    axis_coordinates: tuple[AxisCoordinateSpec, ...]
    keys: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.algorithm_version != PREPARATION_RNG_ALGORITHM_VERSION:
            raise ExecutionPreparationError(
                f"unsupported preparation RNG algorithm {self.algorithm_version!r}"
            )
        object.__setattr__(self, "axis_coordinates", tuple(self.axis_coordinates))
        object.__setattr__(self, "keys", _freeze_rng_roots(self.keys, path="rng.keys"))


@dataclass(frozen=True)
class ScalarInstancePreparationRequest:
    """One scalar materializer request with Feedbax-owned RNG authority."""

    axis_coordinates: tuple[AxisCoordinateSpec, ...]
    rng: PreparationRngScope
    resume_template: bool

    def __post_init__(self) -> None:
        coordinates = tuple(self.axis_coordinates)
        if coordinates != self.rng.axis_coordinates:
            raise ExecutionPreparationError(
                "scalar preparation coordinates do not match their RNG scope"
            )
        object.__setattr__(self, "axis_coordinates", coordinates)


@dataclass(frozen=True)
class ScalarInstancePreparationResult:
    """Mapped state slots produced for one scalar coordinate."""

    mapped_slots: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mapped_slots",
            _freeze_named_mapping(self.mapped_slots, path="mapped_slots"),
        )


class ScalarInstanceMaterializer(Protocol):
    """Callable that prepares one scalar instance only."""

    def __call__(
        self, request: ScalarInstancePreparationRequest
    ) -> ScalarInstancePreparationResult:
        """Prepare mapped slots for exactly one Feedbax-derived coordinate."""
        ...


@dataclass(frozen=True)
class ExecutionPreparationPlan:
    """One-call provider plan consumed by Feedbax's instance materializer."""

    shared_slots: Mapping[str, Any]
    kernel_context: Mapping[str, Any]
    loss_service: LossService | None
    resume_slot_transform: ResumeSlotTransform | None
    rng_roots: Mapping[str, Any]
    materialize_instance: ScalarInstanceMaterializer

    def __post_init__(self) -> None:
        if not callable(self.materialize_instance):
            raise TypeError("materialize_instance must be callable")
        if self.loss_service is not None and not isinstance(self.loss_service, LossService):
            raise TypeError("loss_service must be a LossService when provided")
        if self.resume_slot_transform is not None and not callable(self.resume_slot_transform):
            raise TypeError("resume_slot_transform must be callable when provided")
        object.__setattr__(
            self,
            "shared_slots",
            _freeze_named_mapping(self.shared_slots, path="shared_slots"),
        )
        object.__setattr__(
            self,
            "kernel_context",
            _freeze_named_mapping(self.kernel_context, path="kernel_context"),
        )
        object.__setattr__(
            self,
            "rng_roots",
            _freeze_rng_roots(self.rng_roots, path="rng_roots"),
        )


@dataclass(frozen=True)
class MaterializedPreparationIdentity:
    """Runtime identity binding a materialized preparation to one request."""

    run_spec_sha256: str
    method_ref: str
    provider_identity: str
    mapping_levels: tuple[MaterializedMappingLevelSpec, ...]
    slot_axis_bindings: Mapping[str, tuple[MaterializedSlotAxisBinding, ...]]
    coordinate_order: tuple[tuple[AxisCoordinateSpec, ...], ...]
    rng_algorithm_version: str
    fingerprint: str


@dataclass(frozen=True, init=False)
class MaterializedExecutionPreparation:
    """Sealed runtime handoff admitted by mapped executor calls."""

    initial_slots: Mapping[str, Any]
    kernel_context: Mapping[str, Any]
    loss_service: LossService | None
    resume_slot_transform: ResumeSlotTransform | None
    mapping_levels: tuple[MaterializedMappingLevelSpec, ...]
    slot_axis_bindings: Mapping[str, tuple[MaterializedSlotAxisBinding, ...]]
    identity: MaterializedPreparationIdentity
    _seal: object = field(repr=False, compare=False)

    def __new__(cls) -> "MaterializedExecutionPreparation":
        raise TypeError(
            "MaterializedExecutionPreparation is created only by Feedbax materialization"
        )

    def __copy__(self) -> "MaterializedExecutionPreparation":
        raise TypeError("sealed materialized preparations cannot be copied")

    def __deepcopy__(self, _memo: dict[int, Any]) -> "MaterializedExecutionPreparation":
        raise TypeError("sealed materialized preparations cannot be copied")


def _freeze_runtime_value(value: Any) -> Any:
    """Defensively freeze runtime containers and copy NumPy storage."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ExecutionPreparationError("object-backed NumPy arrays are not admissible")
        copied = np.array(value, copy=True)
        copied.flags.writeable = False
        return copied
    if isinstance(value, jax.Array):
        return value
    if isinstance(value, Mapping):
        return _ImmutableDict({key: _freeze_runtime_value(item) for key, item in value.items()})
    if isinstance(value, tuple) and type(value) is not tuple:
        leaves, treedef = jt.flatten(value, is_leaf=lambda item: item is not value)
        return jt.unflatten(treedef, [_freeze_runtime_value(leaf) for leaf in leaves])
    if isinstance(value, tuple):
        return tuple(_freeze_runtime_value(item) for item in value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_runtime_value(item) for item in value)
    if isinstance(value, Set) and not isinstance(value, (str, bytes, bytearray)):
        return frozenset(_freeze_runtime_value(item) for item in value)
    if isinstance(value, (bytearray, memoryview)):
        raise ExecutionPreparationError("mutable buffer-backed values are not admissible")
    leaves, treedef = jt.flatten(value)
    if len(leaves) != 1 or not leaves or leaves[0] is not value:
        return jt.unflatten(treedef, [_freeze_runtime_value(leaf) for leaf in leaves])
    return value


def _validate_frozen_runtime_value(value: Any, *, path: str) -> None:
    """Fail closed if sealed runtime content is mutable or buffer-backed."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or value.flags.writeable:
            raise ExecutionPreparationError(f"{path} contains mutable NumPy storage")
        return
    if isinstance(value, jax.Array):
        return
    if isinstance(value, Mapping):
        if not isinstance(value, _ImmutableDict):
            raise ExecutionPreparationError(f"{path} contains a mutable mapping")
        for key, item in value.items():
            _validate_frozen_runtime_value(item, path=f"{path}[{key!r}]")
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            _validate_frozen_runtime_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, frozenset):
        for item in value:
            _validate_frozen_runtime_value(item, path=path)
        return
    if isinstance(value, (Sequence, Set, bytearray, memoryview)) and not isinstance(
        value, (str, bytes)
    ):
        raise ExecutionPreparationError(f"{path} contains a mutable container")
    leaves = jt.leaves(value)
    if len(leaves) != 1 or not leaves or leaves[0] is not value:
        for index, leaf in enumerate(leaves):
            _validate_frozen_runtime_value(leaf, path=f"{path}/leaf/{index}")


def _freeze_named_mapping(value: Mapping[str, Any], *, path: str) -> Mapping[str, Any]:
    """Freeze a mapping whose top-level keys are stable Feedbax names."""
    if not isinstance(value, Mapping):
        raise ExecutionPreparationError(f"{path} must be a mapping")
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ExecutionPreparationError(f"{path} key {key!r} must be a string identifier")
        validate_worker_identifier(key, path=f"{path}.name")
        if key in frozen:
            raise ExecutionPreparationError(f"{path} contains duplicate name {key!r}")
        frozen[key] = _freeze_runtime_value(item)
    return _ImmutableDict(frozen)


def _freeze_rng_roots(value: Mapping[str, Any], *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExecutionPreparationError(f"{path} must be a mapping")
    frozen: dict[str, Any] = {}
    for name, key in value.items():
        validate_worker_identifier(name, path=f"{path}.name")
        if name in frozen:
            raise ExecutionPreparationError(f"{path} contains duplicate root {name!r}")
        try:
            data = jax.random.key_data(key)
        except (TypeError, ValueError) as exc:
            raise ExecutionPreparationError(f"{path}[{name!r}] is not a JAX PRNG key") from exc
        if tuple(data.shape) != (2,):
            raise ExecutionPreparationError(f"{path}[{name!r}] must contain one scalar PRNG key")
        frozen[name] = key
    return _ImmutableDict(frozen)


def preparation_rng_token(label: str, value: str) -> int:
    """Return the frozen unsigned token for one preparation RNG component."""
    validate_worker_identifier(label, path="rng token label")
    digest = hashlib.sha256(
        _PREPARATION_RNG_DOMAIN + label.encode("utf-8") + b"\0" + value.encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def derive_preparation_rng_scope(
    roots: Mapping[str, Any],
    axis_coordinates: tuple[AxisCoordinateSpec, ...],
) -> PreparationRngScope:
    """Derive named scalar keys by the frozen coordinate-folding algorithm."""
    frozen_roots = _freeze_rng_roots(roots, path="rng_roots")
    coordinates = tuple(axis_coordinates)
    derived: dict[str, Any] = {}
    for root_name, root_key in frozen_roots.items():
        key = jax.random.fold_in(
            root_key,
            preparation_rng_token("algorithm", PREPARATION_RNG_ALGORITHM_VERSION),
        )
        key = jax.random.fold_in(key, preparation_rng_token("root", root_name))
        for level, coordinate in enumerate(coordinates):
            if level > 0xFFFFFFFF or coordinate.index > 0xFFFFFFFF:
                raise ExecutionPreparationError("axis coordinate exceeds unsigned 32-bit range")
            key = jax.random.fold_in(key, level)
            key = jax.random.fold_in(key, preparation_rng_token("axis", coordinate.axis))
            key = jax.random.fold_in(key, coordinate.index)
        derived[root_name] = key
    return PreparationRngScope(
        algorithm_version=PREPARATION_RNG_ALGORITHM_VERSION,
        axis_coordinates=coordinates,
        keys=derived,
    )


def _run_spec_sha256(run_spec: TrainingRunSpec) -> str:
    return training_spec_sha256(run_spec.model_dump(mode="json", exclude_none=True))


def _identity_projection(identity: MaterializedPreparationIdentity) -> dict[str, Any]:
    return {
        "run_spec_sha256": identity.run_spec_sha256,
        "method_ref": identity.method_ref,
        "provider_identity": identity.provider_identity,
        "mapping_levels": [item.model_dump(mode="json") for item in identity.mapping_levels],
        "slot_axis_bindings": {
            name: [item.model_dump(mode="json") for item in bindings]
            for name, bindings in sorted(identity.slot_axis_bindings.items())
        },
        "coordinate_order": [
            [item.model_dump(mode="json") for item in coordinates]
            for coordinates in identity.coordinate_order
        ],
        "rng_algorithm_version": identity.rng_algorithm_version,
    }


def _identity_fingerprint(identity: MaterializedPreparationIdentity) -> str:
    payload = json.dumps(
        _identity_projection(identity), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _expected_coordinate_order(
    levels: tuple[MaterializedMappingLevelSpec, ...],
) -> tuple[tuple[AxisCoordinateSpec, ...], ...]:
    if len(levels) != 1:
        raise ExecutionPreparationError("materialized preparation requires one mapping level")
    level = levels[0]
    return tuple((AxisCoordinateSpec(axis=level.axis, index=index),) for index in range(level.size))


def _build_materialized_execution_preparation(
    *,
    request: ExecutionPreparationRequest,
    provider_identity: str,
    initial_slots: Mapping[str, Any],
    kernel_context: Mapping[str, Any],
    loss_service: LossService | None,
    resume_slot_transform: ResumeSlotTransform | None,
    coordinate_order: tuple[tuple[AxisCoordinateSpec, ...], ...],
) -> MaterializedExecutionPreparation:
    """Build the sealed Part-2 handoff after scalar instances are stacked."""
    levels, bindings = resolve_execution_mapping(request.run_spec.worker_execution)
    if not levels:
        raise ExecutionPreparationError(
            "MaterializedExecutionPreparation is reserved for active mapped execution"
        )
    try:
        validate_worker_identifier(provider_identity, path="provider_identity")
    except ValueError as exc:
        raise ExecutionPreparationError(str(exc)) from exc
    expected_coordinates = _expected_coordinate_order(levels)
    if coordinate_order != expected_coordinates:
        raise ExecutionPreparationError(
            "coordinate_order must equal the complete ordered mapped-axis grid"
        )
    immutable_bindings = _ImmutableDict(
        {name: tuple(items) for name, items in sorted(bindings.items())}
    )
    provisional = MaterializedPreparationIdentity(
        run_spec_sha256=_run_spec_sha256(request.run_spec),
        method_ref=request.run_spec.method_ref.key,
        provider_identity=provider_identity,
        mapping_levels=levels,
        slot_axis_bindings=immutable_bindings,
        coordinate_order=tuple(tuple(items) for items in coordinate_order),
        rng_algorithm_version=PREPARATION_RNG_ALGORITHM_VERSION,
        fingerprint="",
    )
    identity = MaterializedPreparationIdentity(
        **{
            **provisional.__dict__,
            "fingerprint": _identity_fingerprint(provisional),
        }
    )
    instance = object.__new__(MaterializedExecutionPreparation)
    for name, value in {
        "initial_slots": _freeze_named_mapping(initial_slots, path="initial_slots"),
        "kernel_context": _freeze_named_mapping(kernel_context, path="kernel_context"),
        "loss_service": loss_service,
        "resume_slot_transform": resume_slot_transform,
        "mapping_levels": levels,
        "slot_axis_bindings": immutable_bindings,
        "identity": identity,
        "_seal": _MATERIALIZED_PREPARATION_SEAL,
    }.items():
        object.__setattr__(instance, name, value)
    return instance


def validate_materialized_execution_preparation(
    preparation: MaterializedExecutionPreparation,
    *,
    run_spec: TrainingRunSpec,
) -> None:
    """Revalidate provenance identity before mapped executor admission."""
    if getattr(preparation, "_seal", None) is not _MATERIALIZED_PREPARATION_SEAL:
        raise ExecutionPreparationError("materialized preparation lacks Feedbax provenance seal")
    levels, bindings = resolve_execution_mapping(run_spec.worker_execution)
    identity = getattr(preparation, "identity", None)
    if not isinstance(identity, MaterializedPreparationIdentity):
        raise ExecutionPreparationError("materialized preparation identity is missing or invalid")
    if identity.rng_algorithm_version != PREPARATION_RNG_ALGORITHM_VERSION:
        raise ExecutionPreparationError("materialized preparation RNG algorithm is unsupported")
    try:
        validate_worker_identifier(identity.provider_identity, path="provider_identity")
        observed_fingerprint = _identity_fingerprint(identity)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ExecutionPreparationError(
            f"materialized preparation identity is malformed: {exc}"
        ) from exc
    if identity.fingerprint != observed_fingerprint:
        raise ExecutionPreparationError("materialized preparation identity fingerprint is stale")
    if identity.run_spec_sha256 != _run_spec_sha256(run_spec):
        raise ExecutionPreparationError("materialized preparation does not match TrainingRunSpec")
    if identity.method_ref != run_spec.method_ref.key:
        raise ExecutionPreparationError("materialized preparation method identity mismatch")
    if identity.coordinate_order != _expected_coordinate_order(levels):
        raise ExecutionPreparationError("materialized preparation coordinate identity mismatch")
    if identity.mapping_levels != levels or dict(identity.slot_axis_bindings) != bindings:
        raise ExecutionPreparationError("materialized preparation mapping identity mismatch")
    preparation_levels = getattr(preparation, "mapping_levels", None)
    preparation_bindings = getattr(preparation, "slot_axis_bindings", None)
    if preparation_levels != levels or not isinstance(preparation_bindings, Mapping):
        raise ExecutionPreparationError("materialized preparation mapping structure is stale")
    if dict(preparation_bindings) != bindings:
        raise ExecutionPreparationError("materialized preparation mapping structure is stale")
    initial_slots = getattr(preparation, "initial_slots", None)
    if not isinstance(initial_slots, Mapping):
        raise ExecutionPreparationError("materialized preparation initial_slots are invalid")
    _validate_frozen_runtime_value(initial_slots, path="initial_slots")
    kernel_context = getattr(preparation, "kernel_context", None)
    if not isinstance(kernel_context, Mapping):
        raise ExecutionPreparationError("materialized preparation kernel_context is invalid")
    _validate_frozen_runtime_value(kernel_context, path="kernel_context")
    for slot, value in initial_slots.items():
        _validate_no_batch_history(value, slot=slot)
    executor_owned = executor_owned_initial_slot_names(run_spec, bindings)
    required = {
        slot.name
        for slot in run_spec.worker_execution.method_contract.state_slots
        if slot.required and slot.name in bindings
    }
    missing = sorted(required - set(initial_slots) - executor_owned)
    if missing:
        raise ExecutionPreparationError(
            f"materialized preparation is missing declared slots {missing!r}"
        )
    unknown = sorted(set(initial_slots) - set(bindings))
    if unknown:
        raise ExecutionPreparationError(
            f"materialized preparation contains undeclared slots {unknown!r}"
        )
    _validate_materialized_slot_shapes(initial_slots, bindings)


def executor_owned_initial_slot_names(
    run_spec: TrainingRunSpec,
    bindings: Mapping[str, tuple[MaterializedSlotAxisBinding, ...]],
) -> frozenset[str]:
    """Return declared slots populated by Feedbax after provider materialization."""
    return frozenset(
        slot.name
        for slot in run_spec.worker_execution.method_contract.state_slots
        if slot.role == "objective"
        and (slot.name not in bindings or bindings[slot.name][0].mode == "shared")
    )


def _is_array(value: Any) -> bool:
    return isinstance(value, (jax.Array, np.ndarray))


def _validate_no_batch_history(value: Any, *, slot: str) -> None:
    leaves = jt.leaves(value, is_leaf=lambda leaf: isinstance(leaf, BatchHistory))
    if any(isinstance(leaf, BatchHistory) for leaf in leaves):
        raise ExecutionPreparationError(
            f"mapped slot {slot!r} contains BatchHistory; mapped histories are deferred"
        )


def _same_static_value(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    try:
        equal = left == right
    except Exception:
        return left is right
    return equal if isinstance(equal, bool) else left is right


def _stack_mapped_slot(slot: str, values: Sequence[Any]) -> Any:
    flattened = [jt.flatten(value) for value in values]
    treedef = flattened[0][1]
    if any(item[1] != treedef for item in flattened[1:]):
        raise ExecutionPreparationError(
            f"mapped slot {slot!r} has divergent PyTree definitions or static fields"
        )
    leaf_columns = zip(*(item[0] for item in flattened), strict=True)
    stacked_leaves: list[Any] = []
    array_count = 0
    for leaf_index, column in enumerate(leaf_columns):
        first = column[0]
        array_flags = tuple(_is_array(leaf) for leaf in column)
        if any(array_flags):
            if not all(array_flags):
                raise ExecutionPreparationError(
                    f"mapped slot {slot!r} leaf {leaf_index} changes array/static type"
                )
            if any(type(leaf) is not type(first) for leaf in column[1:]):
                raise ExecutionPreparationError(
                    f"mapped slot {slot!r} leaf {leaf_index} changes array type"
                )
            if any(leaf.shape != first.shape or leaf.dtype != first.dtype for leaf in column[1:]):
                raise ExecutionPreparationError(
                    f"mapped slot {slot!r} leaf {leaf_index} changes shape or dtype"
                )
            stacked_leaves.append(
                np.stack(column, axis=0)
                if isinstance(first, np.ndarray)
                else jnp.stack(column, axis=0)
            )
            array_count += 1
        else:
            if any(not _same_static_value(first, leaf) for leaf in column[1:]):
                raise ExecutionPreparationError(
                    f"mapped slot {slot!r} leaf {leaf_index} has divergent static values"
                )
            stacked_leaves.append(first)
    if array_count == 0:
        raise ExecutionPreparationError(
            f"mapped slot {slot!r} must contain at least one JAX or NumPy array leaf"
        )
    return jt.unflatten(treedef, stacked_leaves)


def _validate_materialized_slot_shapes(
    slots: Mapping[str, Any],
    bindings: Mapping[str, tuple[MaterializedSlotAxisBinding, ...]],
) -> None:
    for slot, slot_bindings in bindings.items():
        if slot not in slots or slot_bindings[0].mode == "shared":
            continue
        expected_size = slot_bindings[0].size
        leaves = jt.leaves(slots[slot])
        arrays = [leaf for leaf in leaves if _is_array(leaf)]
        if not arrays:
            raise ExecutionPreparationError(
                f"mapped slot {slot!r} must contain at least one array leaf"
            )
        for leaf_index, leaf in enumerate(arrays):
            if leaf.ndim < 1 or leaf.shape[0] != expected_size:
                raise ExecutionPreparationError(
                    f"mapped slot {slot!r} array leaf {leaf_index} must have leading "
                    f"axis size {expected_size}; observed shape={leaf.shape!r}"
                )


def validate_materialized_execution_slots(
    slots: Mapping[str, Any],
    bindings: Mapping[str, tuple[MaterializedSlotAxisBinding, ...]],
    *,
    required_slots: Set[str],
) -> None:
    """Validate the complete post-merge executor state against resolved bindings."""
    missing = sorted(set(required_slots).intersection(bindings) - set(slots))
    if missing:
        raise ExecutionPreparationError(
            f"materialized execution state is missing declared slots {missing!r}"
        )
    _validate_materialized_slot_shapes(slots, bindings)


def materialize_execution_preparation(
    request: ExecutionPreparationRequest,
    plan: ExecutionPreparationPlan,
    *,
    provider_identity: str,
) -> MaterializedExecutionPreparation:
    """Materialize one complete single-level plan into a sealed executor handoff."""
    levels, bindings = resolve_execution_mapping(request.run_spec.worker_execution)
    if len(levels) != 1:
        raise ExecutionPreparationError(
            "mapped execution preparation requires exactly one resolved mapping level"
        )
    state_specs = {
        slot.name: slot for slot in request.run_spec.worker_execution.method_contract.state_slots
    }
    mapped_names = {name for name, value in bindings.items() if value[0].mode == "mapped"}
    shared_names = {name for name, value in bindings.items() if value[0].mode == "shared"}
    executor_owned = executor_owned_initial_slot_names(request.run_spec, bindings)
    unknown_shared = sorted(set(plan.shared_slots) - shared_names)
    missing_shared = sorted(
        name
        for name in shared_names
        if state_specs[name].required
        and name not in plan.shared_slots
        and name not in executor_owned
    )
    if unknown_shared or missing_shared:
        raise ExecutionPreparationError(
            "preparation plan shared-slot declarations do not match resolved bindings; "
            f"unknown={unknown_shared!r}, missing_required={missing_shared!r}"
        )

    coordinate_order = _expected_coordinate_order(levels)
    results: list[Mapping[str, Any]] = []
    optional_presence: dict[str, bool] | None = None
    required_mapped = {name for name in mapped_names if state_specs[name].required}
    for coordinate_index, coordinates in enumerate(coordinate_order):
        raw = plan.materialize_instance(
            ScalarInstancePreparationRequest(
                axis_coordinates=coordinates,
                rng=derive_preparation_rng_scope(plan.rng_roots, coordinates),
                resume_template=request.resume,
            )
        )
        if not isinstance(raw, ScalarInstancePreparationResult):
            raise ExecutionPreparationError(
                f"scalar materializer coordinate {coordinate_index} returned "
                f"{type(raw).__name__}; expected ScalarInstancePreparationResult"
            )
        observed = set(raw.mapped_slots)
        unknown = sorted(observed - mapped_names)
        missing = sorted(required_mapped - observed)
        if unknown or missing:
            raise ExecutionPreparationError(
                f"scalar materializer coordinate {coordinate_index} mapped-slot mismatch; "
                f"unknown={unknown!r}, missing_required={missing!r}"
            )
        for slot, value in raw.mapped_slots.items():
            _validate_no_batch_history(value, slot=slot)
        presence = {
            name: name in observed for name in mapped_names if not state_specs[name].required
        }
        if optional_presence is None:
            optional_presence = presence
        elif presence != optional_presence:
            raise ExecutionPreparationError(
                "optional mapped slots must be present for every coordinate or none"
            )
        if results:
            for slot in observed:
                _stack_mapped_slot(slot, (results[0][slot], raw.mapped_slots[slot]))
        results.append(raw.mapped_slots)

    present_mapped = sorted(
        required_mapped | {name for name, present in (optional_presence or {}).items() if present}
    )
    stacked = {
        name: _stack_mapped_slot(name, [result[name] for result in results])
        for name in present_mapped
    }
    initial_slots = {**plan.shared_slots, **stacked}
    _validate_materialized_slot_shapes(initial_slots, bindings)
    return _build_materialized_execution_preparation(
        request=request,
        provider_identity=provider_identity,
        initial_slots=initial_slots,
        kernel_context=plan.kernel_context,
        loss_service=plan.loss_service,
        resume_slot_transform=plan.resume_slot_transform,
        coordinate_order=coordinate_order,
    )


@runtime_checkable
class ExecutionPreparationProvider(Protocol):
    """Callable that prepares runtime-only inputs for one training method."""

    def __call__(
        self, request: ExecutionPreparationRequest
    ) -> ExecutionPreparationResult | ExecutionPreparationPlan:
        """Prepare executor inputs without mutating ``request.run_spec``."""
        ...


@dataclass(frozen=True)
class ExecutionPreparationRegistration:
    """One method-ref keyed execution-preparation provider registration."""

    method_ref: str
    provider: ExecutionPreparationProvider
    owner: str = "feedbax"
    requires_resolved_method: bool = False


class ExecutionPreparationProviderRegistry:
    """Registry containing at most one runtime provider per training method ref."""

    def __init__(self) -> None:
        self._registrations: dict[str, ExecutionPreparationRegistration] = {}

    def register(self, registration: ExecutionPreparationRegistration) -> None:
        """Register one provider, rejecting ambiguous duplicate ownership."""
        if not registration.method_ref:
            raise ValueError("execution preparation method_ref must not be empty")
        if not callable(registration.provider):
            raise TypeError(
                f"execution preparation provider for {registration.method_ref!r} must be callable"
            )
        if registration.method_ref in self._registrations:
            existing = self._registrations[registration.method_ref]
            raise ValueError(
                "execution preparation provider already registered for "
                f"{registration.method_ref!r} by {existing.owner!r}"
            )
        self._registrations[registration.method_ref] = registration

    def available_keys(self) -> tuple[str, ...]:
        """Return method refs with registered preparation providers."""
        return tuple(sorted(self._registrations))

    def get(self, method_ref: str) -> ExecutionPreparationRegistration | None:
        """Return a provider registration when one exists for ``method_ref``."""
        return self._registrations.get(method_ref)

    def prepare(
        self, request: ExecutionPreparationRequest
    ) -> ExecutionPreparationResult | ExecutionPreparationPlan:
        """Invoke the matching provider while enforcing immutability and result shape."""
        method_ref = request.run_spec.method_ref.key
        registration = self.get(method_ref)
        if registration is None:
            raise ExecutionPreparationError(
                f"/method_ref {method_ref!r} has no execution-preparation provider; "
                f"available provider keys={list(self.available_keys())!r}"
            )
        if registration.requires_resolved_method and (
            request.method_payload is None
            or request.method_contract is None
            or request.effective_phase is None
        ):
            raise ExecutionPreparationError(
                f"descriptor-backed preparation for {method_ref!r} requires the resolved "
                "method payload, contract, and effective phase"
            )
        # Mapping declarations fail before provider invocation. This keeps
        # provider side effects behind the same static gate as executor calls.
        resolve_execution_mapping(request.run_spec.worker_execution)
        if (
            request.method_contract is not None
            and request.method_contract != request.run_spec.worker_execution.method_contract
        ):
            raise ExecutionPreparationError(
                f"execution preparation method contract does not match TrainingRunSpec for "
                f"{method_ref!r}"
            )
        if (
            request.effective_phase is not None
            and request.effective_phase != request.run_spec.worker_execution.effective_phase
        ):
            raise ExecutionPreparationError(
                f"execution preparation effective phase does not match TrainingRunSpec for "
                f"{method_ref!r}"
            )

        provider_spec = deepcopy(request.run_spec)
        provider_payload = deepcopy(request.method_payload)
        provider_contract = deepcopy(request.method_contract)
        provider_effective_phase = deepcopy(request.effective_phase)
        before = provider_spec.model_dump_json()
        payload_before = (
            provider_payload.model_dump_json() if provider_payload is not None else None
        )
        contract_before = (
            provider_contract.model_dump_json() if provider_contract is not None else None
        )
        phase_before = (
            provider_effective_phase.model_dump_json()
            if provider_effective_phase is not None
            else None
        )
        provider_request = ExecutionPreparationRequest(
            run_spec=provider_spec,
            method_payload=provider_payload,
            method_contract=provider_contract,
            effective_phase=provider_effective_phase,
            run_id=request.run_id,
            resume=request.resume,
        )
        try:
            result = registration.provider(provider_request)
        except Exception as exc:
            raise ExecutionPreparationError(
                f"execution preparation failed for method_ref {method_ref!r} "
                f"(provider owner={registration.owner!r}): {exc}"
            ) from exc

        if provider_spec.model_dump_json() != before:
            raise ExecutionPreparationError(
                f"execution preparation provider for {method_ref!r} mutated TrainingRunSpec"
            )
        if provider_payload is not None and provider_payload.model_dump_json() != payload_before:
            raise ExecutionPreparationError(
                f"execution preparation provider for {method_ref!r} mutated method payload"
            )
        if provider_contract is not None and provider_contract.model_dump_json() != contract_before:
            raise ExecutionPreparationError(
                f"execution preparation provider for {method_ref!r} mutated method contract"
            )
        if (
            provider_effective_phase is not None
            and provider_effective_phase.model_dump_json() != phase_before
        ):
            raise ExecutionPreparationError(
                f"execution preparation provider for {method_ref!r} mutated effective phase"
            )
        if not isinstance(result, (ExecutionPreparationResult, ExecutionPreparationPlan)):
            raise ExecutionPreparationError(
                f"execution preparation provider for {method_ref!r} returned "
                f"{type(result).__name__}; expected ExecutionPreparationResult or "
                "ExecutionPreparationPlan"
            )
        if isinstance(result, ExecutionPreparationResult):
            slots = result.initial_slots
            slots_name = "initial_slots"
        else:
            slots = result.shared_slots
            slots_name = "shared_slots"
        if not isinstance(slots, Mapping):
            raise ExecutionPreparationError(
                f"execution preparation provider for {method_ref!r} returned non-mapping "
                f"{slots_name}"
            )
        if not isinstance(result.kernel_context, Mapping):
            raise ExecutionPreparationError(
                f"execution preparation provider for {method_ref!r} returned non-mapping "
                "kernel_context"
            )
        reserved_keys = _RESERVED_KERNEL_CONTEXT_KEYS.intersection(result.kernel_context)
        if reserved_keys:
            raise ExecutionPreparationError(
                f"execution preparation provider for {method_ref!r} returned reserved "
                f"kernel_context keys {sorted(reserved_keys)!r}"
            )
        if result.loss_service is not None and not isinstance(result.loss_service, LossService):
            raise ExecutionPreparationError(
                f"execution preparation provider for {method_ref!r} returned invalid loss_service"
            )
        if result.resume_slot_transform is not None and not callable(result.resume_slot_transform):
            raise ExecutionPreparationError(
                f"execution preparation provider for {method_ref!r} returned non-callable "
                "resume_slot_transform"
            )
        return result


def lower_zero_level_preparation_plan(
    plan: ExecutionPreparationPlan,
    *,
    resume_template: bool = False,
) -> ExecutionPreparationResult:
    """Materialize one zero-level plan through the scalar compatibility path."""
    scope = derive_preparation_rng_scope(plan.rng_roots, ())
    raw = plan.materialize_instance(
        ScalarInstancePreparationRequest(
            axis_coordinates=(),
            rng=scope,
            resume_template=resume_template,
        )
    )
    if not isinstance(raw, ScalarInstancePreparationResult):
        raise ExecutionPreparationError(
            "scalar materializer returned "
            f"{type(raw).__name__}; expected ScalarInstancePreparationResult"
        )
    overlap = sorted(set(plan.shared_slots).intersection(raw.mapped_slots))
    if overlap:
        raise ExecutionPreparationError(
            f"zero-level preparation duplicates shared and scalar slots {overlap!r}"
        )
    return ExecutionPreparationResult(
        initial_slots={**plan.shared_slots, **raw.mapped_slots},
        kernel_context=plan.kernel_context,
        loss_service=plan.loss_service,
        resume_slot_transform=plan.resume_slot_transform,
    )


DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY = ExecutionPreparationProviderRegistry()


def require_execution_preparation_provider(
    *,
    method_ref: str,
    preparation_registry: ExecutionPreparationProviderRegistry,
) -> None:
    """Fail clearly when an opted-in method lacks its required runtime provider."""
    if preparation_registry.get(method_ref) is None:
        raise ExecutionPreparationError(
            f"/method_ref {method_ref!r} requires an execution-preparation provider, but none "
            f"is registered; available provider keys={list(preparation_registry.available_keys())!r}"
        )
