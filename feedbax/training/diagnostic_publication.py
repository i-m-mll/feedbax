"""Stage native training diagnostics and publish the finished artifact atomically."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryFile
from types import TracebackType
from typing import BinaryIO

from feedbax.contracts.base import ArtifactRef, sha256_file
from feedbax.training.diagnostics import (
    METHOD_TRAINING_TRACE_ARTIFACT_ROLE,
    MethodTrainingTraceRecord,
    TrainingDiagnostics,
)


class _TraceRecords:
    """Keep Pydantic's pretty-printed record fragments in emission order."""

    def __init__(self, scratch: Path) -> None:
        self._stream = TemporaryFile(mode="w+b", dir=scratch)
        self._has_records = False

    def __enter__(self) -> _TraceRecords:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stream.close()

    @property
    def has_records(self) -> bool:
        return self._has_records

    def append(self, records: Sequence[MethodTrainingTraceRecord]) -> None:
        for record in records:
            serialized = record.model_dump_json(indent=2, exclude_none=True).encode("utf-8")
            self._stream.write(b",\n" if self._has_records else b"\n")
            self._stream.write(b"      " + serialized.replace(b"\n", b"\n      "))
            self._has_records = True

    def copy_into(self, destination: BinaryIO) -> None:
        self._stream.seek(0)
        shutil.copyfileobj(self._stream, destination)


@dataclass(frozen=True)
class _StagedTrainingDiagnostics:
    ref: ArtifactRef
    _staged: Path
    _path: Path

    def publish(self, *, conflict_error: type[ValueError]) -> Path:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(self._staged, self._path)
        except FileExistsError as exc:
            if (
                self._path.is_symlink()
                or not self._path.is_file()
                or self._path.stat().st_size != self.ref.size_bytes
                or sha256_file(self._path) != self.ref.sha256
            ):
                raise conflict_error(
                    f"training diagnostics already exist with different content: {self._path}"
                ) from exc
        self._staged.unlink()
        return self._path


def _stage_training_diagnostics(
    diagnostics: TrainingDiagnostics,
    *,
    scratch: Path,
    path: Path,
    records: _TraceRecords | None,
    learning_rates: Iterable[bytes] | None = None,
) -> _StagedTrainingDiagnostics:
    if learning_rates is not None and diagnostics.lr_trace:
        raise ValueError("training diagnostics learning-rate trace requires an empty array")
    if records is not None:
        trace = diagnostics.method_trace
        if trace is None and records.has_records:
            raise ValueError("training diagnostics trace records require a method trace header")
        if trace is not None and trace.records:
            diagnostics = diagnostics.model_copy(
                update={"method_trace": trace.model_copy(update={"records": []})}
            )

    payload = diagnostics.model_dump_json(indent=2, exclude_none=True).encode("utf-8") + b"\n"
    staged = scratch / "training-diagnostics.json"
    with staged.open("xb") as stream:
        if learning_rates is not None:
            before, field, payload = payload.partition(b'\n  "lr_trace": []')
            if not field:
                raise ValueError(
                    "training diagnostics learning-rate trace has an unexpected JSON layout"
                )
            stream.write(before + field.removesuffix(b"]"))
            separator = b"\n"
            for row in learning_rates:
                stream.write(separator + b"    " + row.replace(b"\n", b"\n    "))
                separator = b",\n"
            stream.write(b"]" if separator == b"\n" else b"\n  ]")
        if records is None or not records.has_records:
            stream.write(payload)
        else:
            # Anchor the splice to the top-level trace and its final, empty records field.
            before, trace_start, trace_and_after = payload.partition(b'\n  "method_trace": {\n')
            trace, trace_end, after = trace_and_after.partition(b"\n  }")
            empty_records = b'\n    "records": []'
            if not trace_start or not trace_end or not trace.endswith(empty_records):
                raise ValueError("training diagnostics method trace has an unexpected JSON layout")
            stream.write(
                before + trace_start + trace.removesuffix(empty_records) + b'\n    "records": ['
            )
            records.copy_into(stream)
            stream.write(b"\n    ]" + trace_end + after)
        stream.flush()
        os.fsync(stream.fileno())

    return _StagedTrainingDiagnostics(
        ref=ArtifactRef(
            role=METHOD_TRAINING_TRACE_ARTIFACT_ROLE,
            logical_name="training-diagnostics.json",
            sha256=sha256_file(staged),
            media_type="application/json",
            size_bytes=staged.stat().st_size,
            uri=str(path),
            metadata={
                "schema_id": diagnostics.schema_id,
                "schema_version": diagnostics.schema_version,
            },
        ),
        _staged=staged,
        _path=path,
    )
