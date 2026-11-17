"""Capture the exact numerical boundary and compare independently qualified evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from functools import partial
from hashlib import sha256
from math import hypot, isfinite
import struct
from types import BuiltinFunctionType, FunctionType, MethodType
from typing import TYPE_CHECKING

import jax
from jax._src.interpreters.mlir import make_ir_context
from jaxlib import lapack
from jaxlib.mlir import ir
import numpy as np
from pydantic import BaseModel, JsonValue

from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.computation_validity import (
    NORMALIZATION_VERSION,
    REFERENCE_PLATFORM,
    AdapterCall,
    ArgumentIdentity,
    ArgumentSignature,
    ArraySignature,
    ComputationComparison,
    ComputationDifferent,
    ComputationEqual,
    ComputationTrace,
    ComputationUnconfirmable,
    DriverChanged,
    EnvironmentCall,
    ForeignCall,
    LoweredProgram,
    OutputFreeCall,
    PrefixChanged,
    PrefixComparison,
    PrefixConfirmed,
    PrefixDrift,
    PrefixUnavailable,
    ProgramIdentity,
    ProgramsChanged,
    StartsChanged,
    StepCountChanged,
    TrainPrefix,
    UnidentifiedCall,
)
from feedbax.persistence.array_container import decode_array_container, encode_array_container

if TYPE_CHECKING:
    from feedbax.contracts.artifact_custody import ContentObjectRef
    from feedbax.contracts.computation_validity import ComputationChange


type Operand = jax.Array | np.ndarray
type ShapedOperand = Operand | jax.ShapeDtypeStruct
type StoreBlob = Callable[[bytes], ContentObjectRef]
type ReadBlob = Callable[[ContentObjectRef], bytes]

_TRAIN_PREFIX_STEPS = 5
_TRAIN_TOLERANCES = {"float32": (3e-5, 2e-4), "float64": (1e-12, 1e-12)}
_PARAMETERS_METADATA = "train_environment_prefix"


class UnrecordableArguments(ValueError):
    """An operand signature has no portable encoding."""


def _type_name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _array_payload(value: Operand) -> Operand:
    if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
        return jax.random.key_data(value)
    return value


def _array_bytes(value: Operand) -> memoryview:
    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.fields is not None:
        raise UnrecordableArguments(f"Unsupported operand dtype {_type_name(type(array.dtype))}")
    normalized = np.asarray(array.reshape(-1), dtype=array.dtype.newbyteorder("<"), order="C")
    return memoryview(normalized.view(np.uint8))


def _static_value(value: object) -> JsonValue:
    if value is None or type(value) in (str, bool):
        return value
    if type(value) is int:
        return ["int", str(value)]
    if type(value) is float:
        return ["float", struct.pack("<d", value).hex()]
    if type(value) is complex:
        return ["complex", struct.pack("<dd", value.real, value.imag).hex()]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, Enum):
        return ["enum", _type_name(type(value)), _static_value(value.value)]
    if isinstance(value, np.generic):
        return ["scalar", str(value.dtype), _array_bytes(np.asarray(value)).hex()]
    if isinstance(value, np.dtype):
        return ["dtype", str(value)]
    if isinstance(value, (jax.Array, np.ndarray)):
        signature = _array_signature(value).model_dump(mode="json")
        return ["array", signature, sha256(_array_bytes(_array_payload(value))).hexdigest()]
    if isinstance(value, type):
        return ["type", _type_name(value)]
    if isinstance(value, Mapping):
        return [
            "mapping", _type_name(type(value)),
            [[_static_value(key), _static_value(item)] for key, item in value.items()],
        ]
    if isinstance(value, (tuple, list)):
        return ["sequence", _type_name(type(value)), [_static_value(item) for item in value]]
    if isinstance(value, (set, frozenset)):
        items = [_static_value(item) for item in value]
        return ["set", _type_name(type(value)), sorted(items, key=canonical_json_v2_bytes)]
    if isinstance(value, slice):
        return ["slice", *map(_static_value, (value.start, value.stop, value.step))]
    if isinstance(value, BaseModel):
        return ["model", _type_name(type(value)), _static_value(value.model_dump())]
    if is_dataclass(value):
        return [
            "dataclass", _type_name(type(value)),
            [[field.name, _static_value(getattr(value, field.name))] for field in fields(value)],
        ]
    if isinstance(value, partial):
        return ["partial", *map(_static_value, (value.func, value.args, value.keywords))]
    if isinstance(value, MethodType):
        return ["method", _static_value(value.__func__), _static_value(value.__self__)]
    if isinstance(value, FunctionType):
        closure = () if value.__closure__ is None else tuple(cell.cell_contents for cell in value.__closure__)
        return [
            "function", value.__module__, value.__qualname__,
            *map(_static_value, (value.__defaults__, value.__kwdefaults__, closure)),
        ]
    if isinstance(value, BuiltinFunctionType) and getattr(value, "__module__", None) is not None:
        return ["builtin", value.__module__, value.__qualname__]
    if isinstance(value, np.ufunc):
        return ["numpy-ufunc", value.__name__]
    if type(value).__module__.startswith(("jax.", "jaxlib.")) and callable(value):
        wrapped = getattr(value, "__wrapped__", None)
        if isinstance(wrapped, FunctionType):
            return ["wrapped", _type_name(type(value)), _static_value(wrapped)]
    raise UnrecordableArguments(f"Unsupported static value {_type_name(type(value))}")


def pytree_operands(value: object) -> tuple[tuple[Operand, ...], JsonValue]:
    """Extract ordered arrays and a portable encoding of the remaining PyTree."""
    leaves, structure = jax.tree.flatten(value)
    cursor = iter(leaves)
    arrays = []

    def encode_node(tree: jax.tree_util.PyTreeDef) -> JsonValue:
        node = tree.node_data()
        if node is None:
            leaf = next(cursor)
            if isinstance(leaf, (jax.Array, np.ndarray)):
                arrays.append(leaf)
                return ["operand"]
            return ["static", _static_value(leaf)]
        cls, metadata = node
        return ["node", _type_name(cls), _static_value(metadata), [encode_node(child) for child in tree.children()]]

    try:
        static = encode_node(structure)
    except RecursionError as exc:
        raise UnrecordableArguments("Cyclic static operand metadata") from exc
    return tuple(arrays), static


def _placement(value: ShapedOperand) -> JsonValue:
    sharding = getattr(value, "sharding", None)
    committed = bool(getattr(value, "committed", sharding is not None))
    if sharding is None:
        return ["host"]
    if isinstance(sharding, jax.sharding.SingleDeviceSharding):
        device, = sharding.device_set
        return ["single", committed, sharding.memory_kind, device.process_index, device.id]
    if isinstance(sharding, jax.sharding.NamedSharding):
        mesh = sharding.mesh
        return [
            "named", committed, sharding.memory_kind, list(mesh.axis_names), list(mesh.axis_sizes),
            _static_value(getattr(mesh, "axis_types", ())), _static_value(tuple(sharding.spec)),
            _static_value(getattr(sharding.spec, "reduced", ())),
            _static_value(getattr(sharding.spec, "unreduced", ())),
            [[device.process_index, device.id] for device in mesh.devices.flat],
        ]
    raise UnrecordableArguments(f"Unsupported sharding {_type_name(type(sharding))}")


def _array_signature(value: ShapedOperand) -> ArraySignature:
    return ArraySignature(
        shape=value.shape,
        dtype=str(value.dtype),
        weak_type=getattr(value, "weak_type", False),
        placement=sha256(canonical_json_v2_bytes(_placement(value))).hexdigest(),
    )


def argument_signature(operands: Sequence[ShapedOperand], *, static: object = None) -> ArgumentSignature:
    """Describe a specialization without transferring dynamic array values."""
    try:
        encoded = canonical_json_v2_bytes(_static_value(static))
    except RecursionError as exc:
        raise UnrecordableArguments("Cyclic static operand metadata") from exc
    return ArgumentSignature(
        arrays=tuple(map(_array_signature, operands)),
        static_sha256=sha256(encoded).hexdigest(),
    )


def capture_arguments(operands: Sequence[Operand], *, static: object = None) -> ArgumentIdentity:
    """Hash argument structure and every first-call array's exact logical bytes."""
    signature = argument_signature(operands, static=static)
    digest = sha256(canonical_json_v2_bytes(signature.model_dump(mode="json")))
    for array in jax.device_get(tuple(map(_array_payload, operands))):
        data = _array_bytes(array)
        digest.update(struct.pack("<Q", data.nbytes))
        digest.update(data)
    return ArgumentIdentity(signature=signature, sha256=digest.hexdigest())


def _walk_operations(operation: ir.Operation):
    yield operation
    for region in operation.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _walk_operations(child.operation)


def _canonical_asm(module: ir.Operation) -> bytes:
    """Rename private definitions and every symbol use without rewriting any ops."""
    clone = module.clone().operation
    with clone.context:
        functions = [
            item.operation
            for item in clone.regions[0].blocks[0].operations
            if item.operation.name == "func.func"
            and "sym_visibility" in item.attributes
            and ir.StringAttr(item.attributes["sym_visibility"]).value == "private"
            and len(item.regions[0].blocks)
        ]
        names = [ir.StringAttr(item.attributes["sym_name"]).value for item in functions]
        symbols = ir.SymbolTable(clone)
        final_prefix = "__feedbax_func_"
        while any(
            f"{final_prefix}{i}" in symbols and f"{final_prefix}{i}" not in names
            for i in range(len(names))
        ):
            final_prefix += "_"
        final_names = [f"{final_prefix}{i}" for i in range(len(names))]
        if any(old != new and new in names for old, new in zip(names, final_names, strict=True)):
            temporary_prefix = "__feedbax_temporary_"
            while any(f"{temporary_prefix}{i}" in symbols for i in range(len(names))):
                temporary_prefix += "_"
            temporary_names = [f"{temporary_prefix}{i}" for i in range(len(names))]
            _rename_symbols(clone, functions, names, temporary_names)
            names = temporary_names
        _rename_symbols(clone, functions, names, final_names)
        # The standalone module's display name has no linkage to a surrounding symbol table.
        if "sym_name" in clone.attributes:
            del clone.attributes["sym_name"]
        return clone.get_asm(assume_verified=True).encode()


def _rename_symbols(
    module: ir.Operation, functions: Sequence[ir.Operation], names: Sequence[str], canonical: Sequence[str],
) -> None:
    for function, old, new in zip(functions, names, canonical, strict=True):
        if old != new:
            ir.SymbolTable.replace_all_symbol_uses(old, new, module)
            ir.SymbolTable.set_symbol_name(function, new)


def _capture_lowering(
    lowered: jax.stages.Lowered, platform: str, *,
    adapter_calls: Mapping[int, AdapterCall], output_free_calls: frozenset[int],
    retain_program: StoreBlob | None,
) -> LoweredProgram:
    module = lowered.compiler_ir(dialect="stablehlo")
    if not isinstance(module, ir.Module):
        raise TypeError("A managed numerical call requires a StableHLO module")
    calls: list[ForeignCall] = []
    environment_targets = lapack.registrations().get("cpu", ()) if platform == "cpu" else ()
    environment_names = {name for name, *_ in environment_targets}
    for op in _walk_operations(module.operation):
        if op.name != "stablehlo.custom_call":
            continue
        target = ir.StringAttr(op.attributes["call_target_name"]).value
        ordinal = len(calls)
        if ordinal in adapter_calls:
            identity = adapter_calls[ordinal]
            if identity.target != target:
                raise ValueError(f"Adapter identity has the wrong target at custom call {ordinal}")
            calls.append(identity)
        elif ordinal in output_free_calls:
            if any(str(result.type) != "!stablehlo.token" for result in op.results):
                raise ValueError(f"Custom call {ordinal} produces numerical results")
            calls.append(OutputFreeCall(target=target))
        elif target in environment_names:
            calls.append(EnvironmentCall(target=target))
        else:
            calls.append(UnidentifiedCall(target=target))
    if any(index not in range(len(calls)) for index in (*adapter_calls, *output_free_calls)):
        raise ValueError("A foreign call identity does not match the lowered program")
    return LoweredProgram(
        platform=platform,
        sha256=sha256(_canonical_asm(module.operation)).hexdigest(),
        calls=tuple(calls),
        raw_program=None if retain_program is None else retain_program(module.operation.get_asm().encode()),
    )


def capture_program(
    execution_lowered: jax.stages.Lowered, *, slot: str, specialization: ArgumentSignature,
    execution_platform: str, reference_lowered: jax.stages.Lowered | None = None,
    adapter_calls: Mapping[int, AdapterCall] | None = None,
    output_free_calls: frozenset[int] = frozenset(),
    retain_program: StoreBlob | None = None,
) -> ProgramIdentity:
    """Record the executed specialization and its fixed CPU reference, when available."""
    execution = _capture_lowering(
        execution_lowered, execution_platform,
        adapter_calls=adapter_calls or {}, output_free_calls=output_free_calls, retain_program=retain_program,
    )
    reference = execution if execution_platform == REFERENCE_PLATFORM else None
    if reference is None and reference_lowered is not None:
        reference = _capture_lowering(
            reference_lowered, REFERENCE_PLATFORM, adapter_calls={}, output_free_calls=frozenset(),
            retain_program=retain_program,
        )
    return ProgramIdentity(slot=slot, specialization=specialization, reference=reference, execution=execution)


def _different_starts(previous: ComputationTrace, present: ComputationTrace) -> tuple[str, ...]:
    former = {start.slot: start.arguments for start in previous.starts}
    current = {start.slot: start.arguments for start in present.starts}
    slots = dict.fromkeys((*former, *current))
    changed = tuple(slot for slot in slots if former.get(slot) != current.get(slot))
    return changed or (() if tuple(former) == tuple(current) else tuple(slots))


def _read_object(reference: ContentObjectRef, read_blob: ReadBlob) -> bytes:
    data = read_blob(reference)
    if len(data) != reference.size_bytes or sha256(data).hexdigest() != reference.sha256:
        raise ValueError("Computation witness disagrees with its content address")
    return data


def _program_digest(program: LoweredProgram, version: int, read_blob: ReadBlob | None) -> str | None:
    if version == NORMALIZATION_VERSION:
        return program.sha256
    if program.raw_program is None or read_blob is None:
        return None
    try:
        data = _read_object(program.raw_program, read_blob)
        with make_ir_context():
            return sha256(_canonical_asm(ir.Module.parse(data).operation)).hexdigest()
    except (OSError, ValueError, ir.MLIRError):
        return None


def compare_computation(
    previous: ComputationTrace, present: ComputationTrace, *, read_blob: ReadBlob | None = None,
) -> ComputationComparison:
    """Compare the driver's identity, starting operands and qualified exact lowerings."""
    changes: list[ComputationChange] = []
    if previous.driver != present.driver:
        changes.append(DriverChanged(previous=previous.driver, present=present.driver))
    if (slots := _different_starts(previous, present)):
        changes.append(StartsChanged(slots=slots))
    if previous.completed_steps != present.completed_steps:
        changes.append(StepCountChanged(previous=previous.completed_steps, present=present.completed_steps))
    old_programs = {(program.slot, program.specialization): program for program in previous.programs}
    new_programs = {(program.slot, program.specialization): program for program in present.programs}
    if tuple(old_programs) != tuple(new_programs):
        slots = tuple(dict.fromkeys(slot for slot, _ in (*old_programs, *new_programs)))
        changes.append(ProgramsChanged(slots=slots))
    if changes:
        return ComputationDifferent(changes=tuple(changes))

    different = []
    for key, old in old_programs.items():
        new = new_programs[key]
        if old.reference is None or new.reference is None:
            return ComputationUnconfirmable(reason="reference_unavailable", targets=(old.slot,))
        if old.execution.platform != new.execution.platform:
            return ComputationUnconfirmable(reason="execution_target", targets=(old.slot,))
        pairs = [(old.reference, new.reference)]
        if old.execution.platform != REFERENCE_PLATFORM:
            pairs.append((old.execution, new.execution))
        unknown = tuple(dict.fromkeys(
            call.target for pair in pairs for program in pair for call in program.calls
            if isinstance(call, UnidentifiedCall)
        ))
        if unknown:
            return ComputationUnconfirmable(reason="foreign_call", targets=unknown)
        for former, current in pairs:
            old_digest = _program_digest(former, previous.normalization_version, read_blob)
            new_digest = _program_digest(current, present.normalization_version, read_blob)
            if old_digest is None or new_digest is None:
                return ComputationUnconfirmable(reason="normalization", targets=(old.slot,))
            if old_digest != new_digest or former.calls != current.calls:
                different.append(old.slot)
    if different:
        return ComputationDifferent(changes=(ProgramsChanged(slots=tuple(dict.fromkeys(different))),))
    return ComputationEqual()


def capture_train_prefix(
    losses: Sequence[float] | Operand, parameters: Mapping[str, Operand], *, store_blob: StoreBlob,
) -> TrainPrefix | None:
    """Retain the actual first updates and their final trainable parameters."""
    loss_values, arrays = jax.device_get((losses, parameters))
    values = np.asarray(loss_values)
    precision = values.dtype.name
    if (
        precision not in _TRAIN_TOLERANCES or values.ndim != 1
        or not 1 <= values.size <= _TRAIN_PREFIX_STEPS or not np.all(np.isfinite(values))
        or not arrays or any(array.dtype.name not in _TRAIN_TOLERANCES for array in arrays.values())
    ):
        return None
    payload = encode_array_container(
        {name: np.asarray(array) for name, array in arrays.items()}, metadata=_PARAMETERS_METADATA,
    )
    return TrainPrefix(
        precision=precision,
        losses=tuple(values.tolist()),
        parameters=store_blob(payload.data),
        parameters_descriptor_sha256=payload.descriptor_sha256,
    )


def _parameters(prefix: TrainPrefix, read_blob: ReadBlob) -> Mapping[str, np.ndarray]:
    captured = decode_array_container(
        _read_object(prefix.parameters, read_blob),
        expected_descriptor_sha256=prefix.parameters_descriptor_sha256,
    )
    if captured.descriptor.metadata != _PARAMETERS_METADATA:
        raise ValueError("Computation witness does not hold training parameters")
    return captured.tensors


def _relative_loss(reference: Sequence[float], measured: Sequence[float]) -> float | None:
    values = []
    for old, new in zip(reference, measured, strict=True):
        if old == new:
            values.append(0.0)
            continue
        scale = max(abs(old), abs(new))
        baseline = abs(old / scale)
        if not baseline:
            return None
        value = abs(new / scale - old / scale) / baseline
        if not isfinite(value):
            return None
        values.append(value)
    return max(values)


def _relative_l2(reference: Mapping[str, np.ndarray], measured: Mapping[str, np.ndarray]) -> float | None:
    scale = max(
        float(np.max(np.abs(array), initial=0))
        for arrays in (reference, measured) for array in arrays.values()
    )
    if scale == 0:
        return 0.0
    difference = baseline = 0.0
    for name, array in reference.items():
        old = np.asarray(array, dtype=np.float64).reshape(-1) / scale
        new = np.asarray(measured[name], dtype=np.float64).reshape(-1) / scale
        difference = hypot(difference, float(np.linalg.norm(new - old)))
        baseline = hypot(baseline, float(np.linalg.norm(old)))
    if baseline == 0:
        return None
    value = difference / baseline
    return value if isfinite(value) else None


def compare_environment_prefix(
    previous: ComputationTrace, present: ComputationTrace, *, read_blob: ReadBlob,
) -> PrefixComparison:
    """Assess numerical drift for an environment change without accepting a code difference."""
    if _different_starts(previous, present):
        return PrefixChanged(reason="arguments")
    old, new = previous.prefix, present.prefix
    if old is None or new is None:
        return PrefixUnavailable(reason="output_unavailable")
    if not isinstance(old, TrainPrefix) or not isinstance(new, TrainPrefix):
        return PrefixUnavailable(reason="profile")
    if len(old.losses) != _TRAIN_PREFIX_STEPS or len(new.losses) != _TRAIN_PREFIX_STEPS:
        return PrefixUnavailable(reason="profile")
    if old.precision != new.precision:
        return PrefixChanged(reason="output_shape_or_dtype")
    try:
        former, measured = _parameters(old, read_blob), _parameters(new, read_blob)
    except (OSError, ValueError):
        return PrefixUnavailable(reason="unreadable_witness")
    if not former or former.keys() != measured.keys() or any(
        (array.shape, array.dtype) != (measured[name].shape, measured[name].dtype)
        or array.dtype.name not in _TRAIN_TOLERANCES for name, array in former.items()
    ):
        return PrefixChanged(reason="output_shape_or_dtype")
    if any(not np.all(np.isfinite(array)) for arrays in (former, measured) for array in arrays.values()):
        return PrefixChanged(reason="nonfinite_outputs")
    loss_drift = _relative_loss(old.losses, new.losses)
    parameter_drift = _relative_l2(former, measured)
    drift = PrefixDrift(max_relative_loss=loss_drift, relative_parameter_l2=parameter_drift)
    loss_tolerance, parameter_tolerance = _TRAIN_TOLERANCES[old.precision]
    if (
        loss_drift is None or parameter_drift is None
        or loss_drift > loss_tolerance or parameter_drift > parameter_tolerance
    ):
        return PrefixChanged(reason="numerical_drift", drift=drift)
    return PrefixConfirmed(drift=drift)
