"""Canonical safetensors objects with authenticated named tensors."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO, cast

import numpy as np
from pydantic import JsonValue
from safetensors import SafetensorError, safe_open
from safetensors.numpy import load, save

from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.strict_json import strict_json_loads


_DESCRIPTOR_KEY = "feedbax"
_SCHEMA = "feedbax.array_container.v1"
_MAX_HEADER_BYTES = 100_000_000  # safetensors format reader limit
_DTYPE_CODES = {
    "bool": "BOOL",
    "int8": "I8",
    "int16": "I16",
    "int32": "I32",
    "int64": "I64",
    "uint8": "U8",
    "uint16": "U16",
    "uint32": "U32",
    "uint64": "U64",
    "float16": "F16",
    "float32": "F32",
    "float64": "F64",
    "complex64": "C64",
}
SUPPORTED_DTYPES = frozenset(_DTYPE_CODES)


class ArrayContainerError(ValueError):
    """A container violates its format or its supplied integrity binding."""


class TensorDigestError(ArrayContainerError):
    """Tensor bytes disagree with their authenticated descriptor."""

    def __init__(self, name: str, expected: str, actual: str) -> None:
        self.name = name
        self.expected = expected
        self.actual = actual
        super().__init__(f"tensor {name!r} digest mismatch: expected {expected}, found {actual}")


@dataclass(frozen=True)
class TensorDescriptor:
    shape: tuple[int, ...]
    dtype: str
    sha256: str


@dataclass(frozen=True)
class ArrayContainerDescriptor:
    metadata: JsonValue
    tensors: Mapping[str, TensorDescriptor]


@dataclass(frozen=True)
class EncodedArrayContainer:
    data: bytes
    descriptor_sha256: str


@dataclass(frozen=True)
class DecodedArrayContainer:
    descriptor: ArrayContainerDescriptor
    tensors: Mapping[str, np.ndarray]


def _valid_name(name: object) -> bool:
    return isinstance(name, str) and bool(name) and name != "__metadata__"


def _tensor_sha256(array: np.ndarray) -> str:
    # Safetensors writes the tensor in little-endian C order, including scalars.
    flat = array.reshape(-1)
    data = np.asarray(flat, dtype=flat.dtype.newbyteorder("<"), order="C")
    return sha256(memoryview(data).cast("B")).hexdigest()


def encode_array_container(
    tensors: Mapping[str, np.ndarray], *, metadata: JsonValue
) -> EncodedArrayContainer:
    """Encode a tensor mapping and opaque JSON metadata. Keep inputs stable until return."""
    arrays: dict[str, np.ndarray] = {}
    descriptions: dict[str, dict[str, JsonValue]] = {}
    for name, value in tensors.items():
        if not _valid_name(name) or name in arrays:
            raise ArrayContainerError(f"invalid or duplicate tensor name {name!r}")
        if (
            not isinstance(value, np.ndarray)
            or isinstance(value, np.ma.MaskedArray)
            or value.dtype.name not in SUPPORTED_DTYPES
            or value.dtype.metadata is not None
        ):
            raise ArrayContainerError(f"unsupported dense NumPy tensor dtype for {name!r}")
        array = np.asarray(value, dtype=value.dtype.newbyteorder("<"), order="C")
        if array.dtype == np.dtype("bool"):
            array = np.not_equal(array, False)
            array = np.asarray(array)
        arrays[name] = array
        descriptions[name] = {
            "shape": list(array.shape),
            "dtype": array.dtype.name,
            "sha256": _tensor_sha256(array),
        }
    try:
        descriptor = canonical_json_v2_bytes(
            {"schema": _SCHEMA, "metadata": metadata, "tensors": descriptions}
        )
        data = save(arrays, metadata={_DESCRIPTOR_KEY: descriptor.decode("utf-8")})
    except (SafetensorError, ValueError) as exc:
        raise ArrayContainerError(f"cannot encode array container: {exc}") from exc
    return EncodedArrayContainer(data=data, descriptor_sha256=sha256(descriptor).hexdigest())


def _parse_json(raw: bytes | str, description: str) -> object:
    try:
        return strict_json_loads(raw)
    except ValueError as exc:
        raise ArrayContainerError(f"invalid {description}: {exc}") from exc


def _parse_descriptor(
    stream: BinaryIO, expected_sha256: str
) -> tuple[ArrayContainerDescriptor, str]:
    length = stream.read(8)
    size = int.from_bytes(length, "little")
    if len(length) != 8 or not 0 < size <= _MAX_HEADER_BYTES:
        raise ArrayContainerError("invalid safetensors header length")
    raw_header = stream.read(size)
    if len(raw_header) != size:
        raise ArrayContainerError("truncated safetensors header")
    header = _parse_json(raw_header, "safetensors header")
    if not isinstance(header, dict):
        raise ArrayContainerError("safetensors header must be an object")
    metadata = header.pop("__metadata__", None)
    if (
        not isinstance(metadata, dict)
        or metadata.keys() != {_DESCRIPTOR_KEY}
        or not isinstance(metadata[_DESCRIPTOR_KEY], str)
    ):
        raise ArrayContainerError("safetensors header requires exactly one Feedbax descriptor")
    descriptor_json = metadata[_DESCRIPTOR_KEY]
    try:
        raw = descriptor_json.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ArrayContainerError("descriptor must be valid UTF-8") from exc
    if sha256(raw).hexdigest() != expected_sha256:
        raise ArrayContainerError("array descriptor digest mismatch")
    payload = _parse_json(raw, "array descriptor")
    if (
        not isinstance(payload, dict)
        or payload.keys() != {"schema", "metadata", "tensors"}
        or payload["schema"] != _SCHEMA
        or not isinstance(payload["tensors"], dict)
    ):
        raise ArrayContainerError("invalid Feedbax array descriptor schema")
    try:
        canonical = canonical_json_v2_bytes(payload)
    except ValueError as exc:
        raise ArrayContainerError(f"invalid canonical array descriptor: {exc}") from exc
    if raw != canonical:
        raise ArrayContainerError("array descriptor is not canonical JSON")

    descriptions = payload["tensors"]
    if descriptions.keys() != header.keys():
        raise ArrayContainerError("descriptor and safetensors header tensor names differ")
    records = {
        name: _tensor_descriptor(name, info, header[name]) for name, info in descriptions.items()
    }
    return ArrayContainerDescriptor(
        metadata=cast(JsonValue, payload["metadata"]), tensors=MappingProxyType(records)
    ), descriptor_json


def _tensor_descriptor(name: str, info: object, header: object) -> TensorDescriptor:
    if not _valid_name(name) or not isinstance(info, dict) or info.keys() != {
        "shape", "dtype", "sha256",
    }:
        raise ArrayContainerError(f"invalid descriptor for tensor {name!r}")
    shape, dtype, digest = info["shape"], info["dtype"], info["sha256"]
    if (
        not isinstance(shape, list)
        or any(type(size) is not int or size < 0 for size in shape)
        or not isinstance(dtype, str)
        or dtype not in _DTYPE_CODES
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ArrayContainerError(f"invalid shape, dtype, or digest for tensor {name!r}")
    if (
        not isinstance(header, dict)
        or header.get("shape") != shape
        or any(type(size) is not int for size in header["shape"])
        or header.get("dtype") != _DTYPE_CODES[dtype]
    ):
        raise ArrayContainerError(f"descriptor and safetensors header disagree for tensor {name!r}")
    return TensorDescriptor(shape=tuple(shape), dtype=dtype, sha256=digest)


def _authenticate_tensor(name: str, array: np.ndarray, record: TensorDescriptor) -> np.ndarray:
    if array.shape != record.shape or array.dtype.name != record.dtype:
        raise ArrayContainerError(f"unexpected shape or dtype for tensor {name!r}")
    actual = _tensor_sha256(array)
    if actual != record.sha256:
        raise TensorDigestError(name, record.sha256, actual)
    return array


def decode_array_container(
    data: bytes, *, expected_descriptor_sha256: str
) -> DecodedArrayContainer:
    """Authenticate the descriptor and every tensor in a byte object."""
    descriptor, _ = _parse_descriptor(BytesIO(data), expected_descriptor_sha256)
    try:
        arrays = load(data)
    except (SafetensorError, ValueError) as exc:
        raise ArrayContainerError(f"cannot decode safetensors object: {exc}") from exc
    return DecodedArrayContainer(
        descriptor=descriptor,
        tensors=MappingProxyType({
            name: _authenticate_tensor(name, arrays[name], record)
            for name, record in descriptor.tensors.items()
        }),
    )


class ArrayContainerReader:
    """An authenticated descriptor with lazy tensor access within its context."""

    def __init__(
        self,
        path: Path,
        descriptor: ArrayContainerDescriptor,
        descriptor_json: str,
        scope: ExitStack,
    ) -> None:
        self._path = path
        self._descriptor = descriptor
        self._descriptor_json = descriptor_json
        self._scope: ExitStack | None = scope
        self._mapped: safe_open | None = None

    @property
    def descriptor(self) -> ArrayContainerDescriptor:
        return self._descriptor

    def _open_payload(self) -> safe_open:
        if self._scope is None:
            raise ArrayContainerError("array container reader is closed")
        if self._mapped is None:
            mapped = self._scope.enter_context(safe_open(self._path, framework="numpy"))
            if (
                mapped.metadata() != {_DESCRIPTOR_KEY: self._descriptor_json}
                or set(mapped.keys()) != self.descriptor.tensors.keys()
            ):
                self._scope.close()
                self._scope = None
                raise ArrayContainerError("array container changed after its descriptor was read")
            self._mapped = mapped
        return self._mapped

    def read_tensor(self, name: str) -> np.ndarray:
        """Authenticate and return one tensor without reading its siblings."""
        record = self.descriptor.tensors[name]
        try:
            array = self._open_payload().get_tensor(name)
        except SafetensorError as exc:
            raise ArrayContainerError(f"cannot read tensor {name!r}: {exc}") from exc
        return _authenticate_tensor(name, array, record)

    def verify_tensors(self) -> None:
        """Authenticate each tensor in turn without retaining the decoded arrays."""
        try:
            self._open_payload()
        except SafetensorError as exc:
            raise ArrayContainerError(f"cannot open safetensors payload: {exc}") from exc
        for name in self.descriptor.tensors:
            self.read_tensor(name)

    def read_unverified_slice(
        self, name: str, selection: int | slice | tuple[int | slice, ...]
    ) -> np.ndarray:
        """Read a selection whose payload trust comes from external custody."""
        record = self.descriptor.tensors[name]
        indices = selection if isinstance(selection, tuple) else (selection,)
        if len(indices) > len(record.shape):
            raise IndexError("too many indices for tensor")
        normalized: list[int | slice] = []
        reversed_axes: list[int] = []
        output_axis = 0
        for index, dimension in zip(indices, record.shape):
            if isinstance(index, slice):
                positions = range(*index.indices(dimension))
                if positions:
                    first, last = positions[0], positions[-1]
                    step = abs(positions.step) if len(positions) > 1 else 1
                    normalized.append(slice(min(first, last), max(first, last) + 1, step))
                else:
                    normalized.append(slice(0, 0))
                if positions.step < 0:
                    reversed_axes.append(output_axis)
                output_axis += 1
            elif type(index) is int and -dimension <= index < dimension:
                normalized.append(index % dimension)
            else:
                raise IndexError("expected an in-bounds integer or a slice")
        try:
            view = self._open_payload().get_slice(name)
            if tuple(view.get_shape()) != record.shape or view.get_dtype() != _DTYPE_CODES[record.dtype]:
                raise ArrayContainerError(f"unexpected shape or dtype for tensor {name!r}")
            array = view[tuple(normalized)]
            return np.flip(array, axis=tuple(reversed_axes)) if reversed_axes else array
        except SafetensorError as exc:
            raise ArrayContainerError(f"cannot slice tensor {name!r}: {exc}") from exc


@contextmanager
def open_array_container(
    path: str | Path, *, expected_descriptor_sha256: str
) -> Iterator[ArrayContainerReader]:
    """Open an authenticated descriptor. Map the file only upon payload access."""
    source = Path(path)
    with source.open("rb") as stream:
        descriptor, descriptor_json = _parse_descriptor(stream, expected_descriptor_sha256)
    with ExitStack() as scope:
        reader = ArrayContainerReader(source, descriptor, descriptor_json, scope)
        try:
            yield reader
        finally:
            reader._scope = None
