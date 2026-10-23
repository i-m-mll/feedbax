from __future__ import annotations

from pathlib import Path


def test_python_post_merge_sync_preserves_development_environment() -> None:
    config = (Path(__file__).parents[1] / ".worktree.yaml").read_text(encoding="utf-8")

    assert 'paths: ["pyproject.toml", "uv.lock"]\n    run: uv sync --extra all' in config
    assert "\n    run: uv sync\n" not in config
