"""Metric serialization and mapped values for training checkpoints and diagnostics."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any, Literal, Self

import numpy as np
from pydantic import model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.execution_schema import (
    MAPPED_METRIC_VALUE_SCHEMA_ID,
    MAPPED_METRIC_VALUE_SCHEMA_VERSION,
    STRUCTURED_MAPPED_METRIC_VALUE_SCHEMA_ID,
    STRUCTURED_MAPPED_METRIC_VALUE_SCHEMA_VERSION,
)
from feedbax.contracts.metric_values import NumericBooleanJsonValue
from feedbax.contracts.worker import MaterializedSlotAxisBinding, ProgressCoordinate


class MetricSerializationError(ValueError):
    """Raised when a run-event stream violates the transport contract."""


class MappedMetricValue(StrictModel):
    """Lossless JSON envelope for one metric retaining declared mapped axes."""

    schema_id: Literal["feedbax.manifest.mapped_metric_value"] = MAPPED_METRIC_VALUE_SCHEMA_ID
    schema_version: Literal["feedbax.manifest.mapped_metric_value.v1"] = (
        MAPPED_METRIC_VALUE_SCHEMA_VERSION
    )
    value: Any
    shape: tuple[int, ...]
    dtype: str
    axes: tuple[MaterializedSlotAxisBinding, ...]


class StructuredMappedMetricValue(StrictModel):
    """Named numeric/boolean metric leaves retaining one replica axis."""

    schema_id: Literal["feedbax.manifest.structured_mapped_metric_value"] = (
        STRUCTURED_MAPPED_METRIC_VALUE_SCHEMA_ID
    )
    schema_version: Literal["feedbax.manifest.structured_mapped_metric_value.v1"] = (
        STRUCTURED_MAPPED_METRIC_VALUE_SCHEMA_VERSION
    )
    value: dict[str, NumericBooleanJsonValue]
    axes: tuple[MaterializedSlotAxisBinding, ...]

    @model_validator(mode="after")
    def _validate_carrier(self) -> Self:
        if len(self.axes) != 1:
            raise ValueError("structured mapped metric requires exactly one axis")
        axis = self.axes[0]
        if (axis.role, axis.level, axis.mode, axis.array_axis) != (
            "replicate",
            0,
            "mapped",
            0,
        ):
            raise ValueError("structured mapped metric requires one leading replica axis")

        def validate_node(value: dict[str, NumericBooleanJsonValue], path: str) -> None:
            if not value:
                raise ValueError(f"structured mapped metric has an empty named node at {path}")
            for name, item in value.items():
                item_path = f"{path}.{name}"
                if isinstance(item, dict):
                    validate_node(item, item_path)
                elif not isinstance(item, list):
                    raise ValueError(f"structured mapped metric leaf {item_path} must be a list")
                elif len(item) != axis.size:
                    raise ValueError(
                        f"structured mapped metric leaf {item_path} has leading size "
                        f"{len(item)}; expected {axis.size}"
                    )

        validate_node(self.value, "value")
        return self


def _normalize_structured_mapped_value(
    value: Mapping[Any, Any],
    *,
    metric_name: str,
    replica_count: int,
    path: str = "value",
) -> dict[str, Any]:
    if not value:
        raise MetricSerializationError(f"mapped metric {metric_name!r} has empty named node {path}")
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if type(key) is not str:
            raise MetricSerializationError(
                f"mapped metric {metric_name!r} requires a string name at {path}"
            )
        item_path = f"{path}.{key}"
        if isinstance(item, Mapping):
            normalized[key] = _normalize_structured_mapped_value(
                item,
                metric_name=metric_name,
                replica_count=replica_count,
                path=item_path,
            )
            continue
        try:
            array = np.asarray(item)
        except (TypeError, ValueError) as exc:
            raise MetricSerializationError(
                f"mapped metric {metric_name!r} leaf {item_path} is not an array"
            ) from exc
        if array.ndim < 1:
            raise MetricSerializationError(
                f"mapped metric {metric_name!r} leaf {item_path} lacks a replica dimension"
            )
        if array.shape[0] != replica_count:
            raise MetricSerializationError(
                f"mapped metric {metric_name!r} leaf {item_path} has leading size "
                f"{array.shape[0]}; expected {replica_count}"
            )
        if array.dtype.kind not in "biuf":
            raise MetricSerializationError(
                f"mapped metric {metric_name!r} leaf {item_path} has unsupported dtype "
                f"{array.dtype}"
            )
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise MetricSerializationError(
                f"mapped metric {metric_name!r} leaf {item_path} must be finite"
            )
        normalized[key] = array.tolist()
    return normalized


def _json_metric_value(value: Any) -> Any:
    import jax
    import jax.numpy as jnp

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, (jax.Array, np.ndarray, np.generic)):
        if value.ndim != 0:
            raise ValueError("an unmapped array must be scalar")
        dtype = value.dtype
        kind = getattr(dtype, "kind", None)
        if kind == "b":
            return bool(value.item())
        if kind in ("i", "u") or (kind == "V" and jnp.issubdtype(dtype, jnp.integer)):
            return int(value.item())
        if kind != "f" and not (kind == "V" and jnp.issubdtype(dtype, jnp.floating)):
            raise ValueError(f"unsupported scalar dtype {dtype}")
        value = float(value.item())

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("numbers must be finite")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_metric_value(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("object keys must be strings")
        return {key: _json_metric_value(item) for key, item in value.items()}
    raise ValueError(f"unsupported value type {type(value).__name__}")


def _normalize_json_metric(name: str, value: Any) -> Any:
    try:
        return _json_metric_value(value)
    except (TypeError, ValueError, RecursionError) as exc:
        raise MetricSerializationError(
            f"metric {name!r} must have a finite JSON representation: {exc}"
        ) from exc


def normalize_serialized_metrics(
    coordinate: ProgressCoordinate,
    named_metrics: Mapping[str, Any],
    slot_axis_bindings: Mapping[str, tuple[MaterializedSlotAxisBinding, ...]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Serialize shared metrics and move mapped metrics to their named carriers."""
    mapped = {
        name: bindings
        for name, bindings in slot_axis_bindings.items()
        if bindings and bindings[0].mode == "mapped" and name in named_metrics
    }
    normalized = {
        name: value if name in mapped else _normalize_json_metric(name, value)
        for name, value in named_metrics.items()
    }
    coordinate_payload = coordinate.model_dump(
        mode="json", exclude={"metrics"}, exclude_none=True
    )
    coordinate_payload["metrics"] = {
        name: (
            normalized[name]
            if name in named_metrics and value is named_metrics[name]
            else _normalize_json_metric(name, value)
        )
        for name, value in coordinate.metrics.items()
        if name not in mapped
    }
    for name, axes in mapped.items():
        if isinstance(named_metrics[name], Mapping):
            if (
                len(axes) != 1
                or axes[0].role != "replicate"
                or axes[0].level != 0
                or axes[0].array_axis != 0
            ):
                raise MetricSerializationError(
                    f"mapped metric {name!r} requires one leading replica axis"
                )
            value = _normalize_structured_mapped_value(
                named_metrics[name],
                metric_name=name,
                replica_count=axes[0].size,
            )
            normalized[name] = StructuredMappedMetricValue(
                value=value,
                axes=axes,
            ).model_dump(mode="json")
            continue
        try:
            array = np.asarray(named_metrics[name])
            value = array.tolist()
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise MetricSerializationError(
                f"mapped metric {name!r} must have a finite JSON representation"
            ) from exc
        normalized[name] = MappedMetricValue(
            value=value,
            shape=array.shape,
            dtype=str(array.dtype),
            axes=axes,
        ).model_dump(mode="json")
    return coordinate_payload, normalized
