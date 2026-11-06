"""Keep routine tests and affected expensive tests in one pytest invocation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


SELECTED_FILES_ENV = "FEEDBAX_AFFECTED_TEST_FILES"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    raw = os.environ.get(SELECTED_FILES_ENV)
    if raw is None:
        raise pytest.UsageError(f"{SELECTED_FILES_ENV} is required for affected collection")
    try:
        paths = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise pytest.UsageError(f"{SELECTED_FILES_ENV} is invalid JSON") from exc
    if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
        raise pytest.UsageError(f"{SELECTED_FILES_ENV} must be a JSON string array")
    root = Path(config.rootpath)
    selected = {(root / path).resolve() for path in paths}
    retained: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        expensive = item.get_closest_marker("extended") or item.get_closest_marker("slow")
        if expensive and Path(str(item.path)).resolve() not in selected:
            deselected.append(item)
        else:
            retained.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = retained
