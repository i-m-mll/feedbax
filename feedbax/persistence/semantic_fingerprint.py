"""Address-free fingerprints of typed Python and JAX values."""

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import fields
from enum import Enum
import hashlib
import struct
from types import MappingProxyType
from typing import Any

import equinox as eqx
import jax
import jax.tree_util as jtu
import numpy as np


class UnsupportedSemanticValue(TypeError):
    """A value has no defined persistent semantic identity."""


def project_pytree(tree: jax.tree_util.PyTreeDef, leaf: Callable[[], object]) -> object:
    node = tree.node_data()
    if node is None:
        return leaf()
    cls, metadata = node
    return ("node", f"{cls.__module__}.{cls.__qualname__}", metadata, tuple(project_pytree(child, leaf) for child in tree.children()))


def fingerprint_semantic_values(
    *values: Any,
    header: tuple[str, ...] = (),
    project_leaf: Callable[[Any], Any] | None = None,
) -> str:
    """Hash framed values with an optional projection for opaque leaves."""
    digest = hashlib.sha256()

    def frame(value: bytes | memoryview) -> None:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)

    def text(value: str) -> None:
        frame(value.encode("utf-8", errors="surrogatepass"))

    def type_name(cls: type) -> None:
        text(cls.__module__)
        text(cls.__qualname__)

    def encode_opaque_leaf(value: Any, reason: str) -> None:
        if project_leaf is None:
            raise UnsupportedSemanticValue(reason)
        frame(b"projected-leaf")
        type_name(type(value))
        encode(project_leaf(value))

    def array_payload(value: np.ndarray, source: Any) -> None:
        dtype = value.dtype
        if dtype.hasobject or dtype.fields is not None or dtype.metadata is not None:
            encode_opaque_leaf(source, f"array dtype {dtype!r}")
            return
        text(dtype.str)
        text(dtype.name)
        encode(value.shape)
        contiguous = np.ascontiguousarray(value)
        frame(memoryview(contiguous.view(np.uint8).reshape(-1)))

    def encode(value: Any) -> None:
        cls = type(value)
        if value is None:
            frame(b"none")
        elif cls is bool:
            frame(b"bool")
            frame(bytes((value,)))
        elif cls is int:
            frame(b"int")
            text(hex(value))
        elif cls is float:
            frame(b"float")
            frame(struct.pack(">d", value))
        elif cls is complex:
            frame(b"complex")
            frame(struct.pack(">dd", value.real, value.imag))
        elif cls is str:
            frame(b"str")
            text(value)
        elif cls is bytes or cls is bytearray:
            frame(b"bytes" if cls is bytes else b"bytearray")
            frame(memoryview(value))
        elif isinstance(value, Enum):
            frame(b"enum")
            type_name(cls)
            encode(value.name)
            encode(value.value)
        elif cls is np.ndarray or (isinstance(value, np.generic) and cls is value.dtype.type):
            frame(b"numpy-array" if cls is np.ndarray else b"numpy-scalar")
            array_payload(np.asarray(value), value)
        elif isinstance(value, jax.Array) and not isinstance(value, jax.core.Tracer):
            if not value.is_fully_addressable:
                encode_opaque_leaf(value, "non-addressable JAX array")
            elif jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
                frame(b"jax-key")
                text(str(jax.random.key_impl(value)))
                encode(value.shape)
                array_payload(np.asarray(jax.random.key_data(value)), value)
            else:
                frame(b"jax-array")
                encode(value.weak_type)
                array_payload(np.asarray(value), value)
        elif cls in (dict, OrderedDict, MappingProxyType):
            frame(b"mapping")
            type_name(cls)
            encode(len(value))
            for key, item in value.items():
                encode(key)
                encode(item)
        elif cls in (tuple, list, type({}.keys())):
            frame(b"sequence")
            type_name(cls)
            encode(len(value))
            for item in value:
                encode(item)
        elif isinstance(value, eqx.Module):
            frame(b"equinox-module")
            type_name(cls)
            module_fields = fields(value)
            encode(len(module_fields))
            for field in module_fields:
                text(field.name)
                encode(getattr(value, field.name))
        else:
            encode_registered_node(value)

    def encode_registered_node(value: Any) -> None:
        at_root = True

        def stop_at_child(_: Any) -> bool:
            nonlocal at_root
            if at_root:
                at_root = False
                return False
            return True

        try:
            children, definition = jtu.tree_flatten(value, is_leaf=stop_at_child)
        except (TypeError, ValueError):
            encode_opaque_leaf(value, f"cannot flatten {type(value).__qualname__}")
            return
        node = definition.node_data()
        if node is None:
            encode_opaque_leaf(value, f"{type(value).__module__}.{type(value).__qualname__}")
            return
        _, aux = node
        frame(b"pytree")
        type_name(type(value))
        if isinstance(value, tuple) and hasattr(type(value), "_fields"):
            # JAX puts the namedtuple class itself in auxiliary data.
            aux = value._fields
        encode(aux)
        encode(len(children))
        for child in children:
            encode(child)

    try:
        for label in header:
            text(label)
        for value in values:
            encode(value)
    except RecursionError as exc:
        raise UnsupportedSemanticValue("cyclic or deeply nested value") from exc
    return digest.hexdigest()
