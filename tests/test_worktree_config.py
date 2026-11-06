from __future__ import annotations

from pathlib import Path
import shlex

from ruamel.yaml import YAML


def test_python_post_merge_sync_preserves_development_environment() -> None:
    path = Path(__file__).parents[1] / ".worktree.yaml"
    config = YAML(typ="safe").load(path)
    hooks = [
        shlex.split(hook["run"])
        for hook in config["post-merge"]
        if {"pyproject.toml", "uv.lock"} <= set(hook["paths"])
    ]
    assert len(hooks) == 1
    command, = hooks
    assert command[:2] == ["uv", "sync"]
    extra = command.index("--extra")
    assert command[extra + 1] == "all"
    assert not {"--no-dev", "--no-default-groups", "--only-group"} & set(command)
