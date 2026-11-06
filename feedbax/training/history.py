"""Streaming publication of native training progress."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from textwrap import indent
from typing import Any

from feedbax.contracts.artifact_store import store_artifact
from feedbax.contracts.base import ArtifactRef


class _NativeProgressRecorder:
    """Stage normalized events as the same JSON array used by training history."""

    def __init__(self, path: Path, *, retain: bool) -> None:
        self._path = path
        self._events: list[dict[str, Any]] | None = [] if retain else None
        self._has_events = False
        self._file = path.open("xb")

    def __enter__(self) -> _NativeProgressRecorder:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._file.close()

    def append(self, event: Mapping[str, Any]) -> None:
        payload = dict(event)
        rendered = indent(json.dumps(payload, indent=2, sort_keys=True), "  ")
        separator = ",\n" if self._has_events else "[\n"
        self._file.write((separator + rendered).encode())
        self._has_events = True
        if self._events is not None:
            self._events.append(payload)

    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._events) if self._events is not None else ()

    def publish(self, root: Path, run_id: str) -> ArtifactRef | None:
        if self._has_events:
            self._file.write(b"\n]\n")
        self.close()
        if not self._has_events:
            return None
        return store_artifact(
            self._path,
            root=root,
            role="training_history",
            logical_name=f"feedbax_training_history_{run_id}.json",
            media_type="application/json",
            include_original_uri=False,
        )
