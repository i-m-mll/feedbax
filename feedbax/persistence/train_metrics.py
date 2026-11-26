"""Immutable NDJSON deltas of a Train request's checkpointed metric history."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from hashlib import sha256
from math import fsum, isfinite
import json
import os
from pathlib import Path
from typing import Any, BinaryIO

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.effective_success import Success, read_effective_request, success_content_ref
from feedbax.contracts.request_refs import ConditionalProduct, RequestId
from feedbax.contracts.request_specs import DocumentTrainingRequestSpec
from feedbax.contracts.run_rule import metric_condition_holds
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.method_checkpoint import TrainMetricPrefix
from feedbax.contracts.strict_json import strict_json_loads
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider


@dataclass(frozen=True)
class MetricHistory:
    chunks: tuple[tuple[ContentObjectRef, TrainMetricPrefix], ...] = ()

    @property
    def references(self) -> tuple[ContentObjectRef, ...]:
        return tuple(item for ref, chunk in self.chunks for item in (ref, chunk.rows))


def read_metric_prefix(
    store: ImmutableArtifactBlobProvider, ref: ContentObjectRef | None, *, completed: int, origin: int | None = None,
) -> MetricHistory:
    reverse = []
    coordinate = completed
    while ref is not None:
        raw = store.get_object(ref)
        node = TrainMetricPrefix.model_validate_json(raw)
        if node.end_inclusive != coordinate or raw != canonical_json_v2_bytes(node.model_dump(mode="json")):
            raise ValueError("Train metric prefix differs from its exact checkpoint coordinate")
        reverse.append((ref, node))
        coordinate, ref = node.start_exclusive, node.previous
    if origin is not None and origin != coordinate:
        raise ValueError("Train metric prefix does not cover its request's initial clock")
    return MetricHistory(tuple(reversed(reverse)))


def _rows(
    store: ImmutableArtifactBlobProvider, node: TrainMetricPrefix,
) -> Iterator[tuple[bytes, dict[str, Any]]]:
    with store.open_owned_object(node.rows) as path, path.open("rb") as stream:
        expected = node.start_exclusive + 1
        for line in stream:
            value = strict_json_loads(line)
            if (
                not line.endswith(b"\n") or not isinstance(value, dict)
                or type(value.get("update")) is not int or value["update"] != expected
                or not isinstance(value.get("metrics"), dict) or expected > node.end_inclusive
            ):
                raise ValueError("Train metric chunk has a missing, repeated, or invalid row")
            yield line, value["metrics"]
            expected += 1
        if expected != node.end_inclusive + 1:
            raise ValueError("Train metric chunk does not reach its exact checkpoint")


def latest_metrics(store: ImmutableArtifactBlobProvider, history: MetricHistory) -> dict[str, Any]:
    result = {}
    if history.chunks:
        for _, result in _rows(store, history.chunks[-1][1]):
            pass
    return result


def _last_metric_row(store: ImmutableArtifactBlobProvider, success: Success) -> dict[str, Any]:
    if success.continuation is None or len(held := success.products.get("metrics", ())) != 1:
        raise ValueError("The exact Train receipt has no final metric history.")
    (product,) = held
    if product.name != "updates" or product.content.media_type != "application/x-ndjson" or product.content.tree:
        raise ValueError("The exact Train receipt has no final metric history.")
    ref = success_content_ref(product.content)
    assert isinstance(ref, ContentObjectRef)
    with store.open_owned_object(ref) as path, path.open("rb") as stream:
        end = stream.seek(0, os.SEEK_END)
        if end < 2 or stream.seek(end - 1) != end - 1 or stream.read(1) != b"\n":
            raise ValueError("The final Train metric row is incomplete.")
        offset = end - 1
        blocks = []
        while offset:
            start = max(0, offset - 8192)
            stream.seek(start)
            block = stream.read(offset - start)
            if (separator := block.rfind(b"\n")) >= 0:
                blocks.append(block[separator + 1:])
                break
            blocks.append(block)
            offset = start
    row = strict_json_loads(b"".join(reversed(blocks)))
    update = success.continuation.clock.completed_updates
    if not isinstance(row, dict) or row.get("update") != update or not isinstance(row.get("metrics"), dict):
        raise ValueError("The final Train metric row differs from the accepted checkpoint.")
    return row["metrics"]


def finite_final_metric(value: Any, replicas: int, name: str) -> float:
    """Read one scalar per replica and reduce the run's measure to their mean."""
    values = (value,) if replicas == 1 else value
    if isinstance(values, (tuple, list)) and len(values) == replicas:
        try:
            if all(type(item) in (float, int) and isfinite(item) for item in values):
                return fsum(float(item) / replicas for item in values)
        except OverflowError:
            pass
    raise ValueError(f"The Train result has no finite scalar {name!r} for each replica.")


def read_final_train_metrics(
    store: ImmutableArtifactBlobProvider, success: Success, request: RequestId,
) -> Callable[[str], float]:
    """Read finite metric scalars from the exact Train request's committed final update."""
    if success.request.id != request:
        raise ValueError("Metric condition names a different Train request.")
    metrics = _last_metric_row(store, success)

    decoded = read_effective_request(success, store.get_object)
    if not isinstance(decoded.spec, DocumentTrainingRequestSpec):
        raise ValueError("Metric condition requires a document Train request.")
    replicas = decoded.spec.training_config.replicates
    return lambda name: finite_final_metric(metrics.get(name), replicas, name)


def passes_final_metric_condition(
    store: ImmutableArtifactBlobProvider, success: Success, reference: ConditionalProduct,
) -> bool:
    return metric_condition_holds(reference.conditions, read_final_train_metrics(store, success, reference.request_id))


def materialize_metrics(store: ImmutableArtifactBlobProvider, history: MetricHistory, path: Path) -> None:
    if history.chunks:
        with path.open("xb") as stream:
            for _, node in history.chunks:
                for line, _ in _rows(store, node):
                    stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())


class MetricDelta:
    def __init__(
        self, stream: BinaryIO, path: Path, *, start: int, previous: ContentObjectRef | None,
    ) -> None:
        self.stream = stream
        self.path = path
        self.previous = previous
        self.start = self.completed = start
        self._hasher = sha256()

    def write(self, update: int, metrics: dict[str, Any]) -> None:
        if update != self.completed + 1:
            raise ValueError("Train metric history must advance one committed update at a time")
        row = (json.dumps({"update": update, "metrics": metrics}, allow_nan=False) + "\n").encode()
        self.stream.write(row)
        self._hasher.update(row)
        self.completed = update

    def checkpoint(
        self, store: ImmutableArtifactBlobProvider, completed: int,
    ) -> tuple[ContentObjectRef | None, tuple[ContentObjectRef, ...]]:
        if self.completed != completed:
            raise ValueError("Train checkpoint clock differs from its recorded metric rows")
        if self.completed == self.start:
            return self.previous, ()
        self.stream.flush()
        rows = store._store_file(self.path, sha256=self._hasher.hexdigest(), size_bytes=self.stream.tell())
        node = TrainMetricPrefix(
            start_exclusive=self.start, end_inclusive=self.completed, rows=rows, previous=self.previous,
        )
        ref = store.store_object(canonical_json_v2_bytes(node.model_dump(mode="json")))
        self.previous = ref
        self.start = completed
        self.stream.seek(0)
        self.stream.truncate()
        self._hasher = sha256()
        return ref, (ref, rows)
