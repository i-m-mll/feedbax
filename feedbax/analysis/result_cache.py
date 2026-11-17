"""Versioned result-cache helpers for analysis execution."""

from collections.abc import Mapping
from pathlib import Path
import pickle
from typing import Any

from feedbax._atomic import write_bytes_atomic
from feedbax.contracts.computation_revision import ANALYSIS_ENGINE_REVISION
from feedbax.persistence.semantic_fingerprint import fingerprint_semantic_values


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
    return fingerprint_semantic_values(
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
