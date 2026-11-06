"""Versioned result-cache helpers for analysis execution."""

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import fields
from enum import Enum
import hashlib
from pathlib import Path
import pickle
import struct
from types import MappingProxyType
from typing import Any

import equinox as eqx
import jax
import jax.tree_util as jtu
import numpy as np

from feedbax._atomic import write_bytes_atomic
from feedbax.contracts.computation_revision import ANALYSIS_ENGINE_REVISION


RESULTS_CACHE_SUBDIR = "results"
ANALYSIS_RESULT_CACHE_SCHEMA_VERSION = "feedbax.analysis.result-cache.v4"

_CACHE_LOAD_ERRORS = (
    OSError,
    EOFError,
    pickle.UnpicklingError,
    AttributeError,
    ImportError,
    TypeError,
    ValueError,
)
_CACHE_SAVE_ERRORS = (OSError, pickle.PicklingError, TypeError, ValueError)


class AnalysisResultCacheCorruption(RuntimeError):
    """Raised when an analysis-result cache file cannot be trusted."""


class _UnsupportedCacheInput(TypeError):
    """A value has no defined persistent cache identity."""


def _fingerprint_cache_inputs(
    analysis_type: type,
    computation_revision: str,
    config: Mapping[str, Any],
    data: Any,
    kwargs: Mapping[str, Any],
) -> str:
    """Hash the declared computation, its configuration, and prepared values.

    Supported values are typed scalars, concrete arrays, ordered containers,
    Equinox fields (including static fields), and registered PyTrees whose
    children and auxiliary data are supported. Arrays are identified by logical
    values, dtype and shape rather than storage layout. The analysis must treat
    its inputs as read-only values. Opaque objects and Python functions cannot
    authorize reuse.
    """
    return _fingerprint_semantic_values(
        config,
        kwargs,
        data,
        header=(
            ANALYSIS_RESULT_CACHE_SCHEMA_VERSION,
            ANALYSIS_ENGINE_REVISION,
            analysis_type.__module__,
            analysis_type.__qualname__,
            computation_revision,
        ),
    )


def _fingerprint_semantic_values(
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
            raise _UnsupportedCacheInput(reason)
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
        raise _UnsupportedCacheInput("cyclic or deeply nested value") from exc
    return digest.hexdigest()


def _load_analysis_result_cache(
    cache_path: Path,
    *,
    inputs_hash: str,
) -> Any:
    import dill

    try:
        stream = cache_path.open("rb")
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise AnalysisResultCacheCorruption(str(exc)) from exc
    try:
        with stream:
            payload = dill.load(stream)
    except _CACHE_LOAD_ERRORS as exc:
        raise AnalysisResultCacheCorruption(str(exc)) from exc

    if not isinstance(payload, Mapping):
        raise AnalysisResultCacheCorruption("missing versioned payload envelope")
    schema_version = payload.get("schema_version")
    if schema_version != ANALYSIS_RESULT_CACHE_SCHEMA_VERSION:
        raise AnalysisResultCacheCorruption(
            f"schema_version={schema_version!r}, expected {ANALYSIS_RESULT_CACHE_SCHEMA_VERSION!r}"
        )
    if payload.get("inputs_hash") != inputs_hash:
        raise AnalysisResultCacheCorruption(
            f"inputs_hash={payload.get('inputs_hash')!r}, expected {inputs_hash!r}"
        )
    if "result" not in payload:
        raise AnalysisResultCacheCorruption("missing result payload")
    return payload["result"]


def _write_analysis_result_cache(
    cache_path: Path,
    *,
    inputs_hash: str,
    result: Any,
) -> None:
    import dill

    payload = {
        "schema_version": ANALYSIS_RESULT_CACHE_SCHEMA_VERSION,
        "inputs_hash": inputs_hash,
        "result": result,
    }
    write_bytes_atomic(cache_path, dill.dumps(payload))
