"""Versioned result-cache helpers for analysis execution."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import dill as pickle


RESULTS_CACHE_SUBDIR = "results"
ANALYSIS_RESULT_CACHE_SCHEMA_VERSION = "feedbax.analysis.result-cache.v1"

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


def _load_analysis_result_cache(
    cache_path: Path,
    *,
    analysis_id: str,
    inputs_hash: str,
) -> Any:
    try:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
    except _CACHE_LOAD_ERRORS as exc:
        raise AnalysisResultCacheCorruption(str(exc)) from exc

    if not isinstance(payload, Mapping):
        raise AnalysisResultCacheCorruption("missing versioned payload envelope")
    schema_version = payload.get("schema_version")
    if schema_version != ANALYSIS_RESULT_CACHE_SCHEMA_VERSION:
        raise AnalysisResultCacheCorruption(
            f"schema_version={schema_version!r}, "
            f"expected {ANALYSIS_RESULT_CACHE_SCHEMA_VERSION!r}"
        )
    if payload.get("analysis_id") != analysis_id:
        raise AnalysisResultCacheCorruption(
            f"analysis_id={payload.get('analysis_id')!r}, expected {analysis_id!r}"
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
    analysis_id: str,
    inputs_hash: str,
    result: Any,
) -> None:
    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "schema_version": ANALYSIS_RESULT_CACHE_SCHEMA_VERSION,
                "analysis_id": analysis_id,
                "inputs_hash": inputs_hash,
                "result": result,
            },
            f,
        )
