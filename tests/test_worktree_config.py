from __future__ import annotations

from pathlib import Path
import shlex

from ruamel.yaml import YAML


def test_worktree_sync_preserves_the_full_development_environment() -> None:
    path = Path(__file__).parents[1] / ".worktree.yaml"
    config = YAML(typ="safe").load(path)
    hooks = [
        shlex.split(hook["run"])
        for hook in config["post-merge"]
        if {"pyproject.toml", "uv.lock"} <= set(hook["paths"])
    ]
    assert len(hooks) == 1
    setup = shlex.split(config["setup"].split("&&", 1)[0])
    for command in [setup, *hooks]:
        assert command[:2] == ["uv", "sync"]
        assert {"--locked", "--dev", "--all-extras"} <= set(command)
        assert not {"--no-dev", "--no-default-groups", "--only-group"} & set(command)
