"""Tests for the YAML config loader."""

import tomllib
from pathlib import Path

import pytest

from feedbax.config.yaml import get_yaml_loader


def test_yaml_loader_dependency_is_base_runtime_dependency():
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    dependency_names = {
        dependency.split(";", 1)[0]
        .split("[", 1)[0]
        .split("<", 1)[0]
        .split(">", 1)[0]
        .split("=", 1)[0]
        .strip()
        .lower()
        .replace("_", "-")
        for dependency in pyproject["project"]["dependencies"]
    }

    assert "ruamel-yaml" in dependency_names
    assert callable(get_yaml_loader)


def test_include_missing_file_raises_file_not_found_error(tmp_path):
    """Loading a YAML file that includes a nonexistent file raises FileNotFoundError.

    Regression: previously this raised AttributeError because the error handler
    referenced loader.stream.path, which does not exist on SafeConstructor.
    Bug: 7c22abc
    """
    # Create a YAML file that includes a missing file
    parent_yaml = tmp_path / "parent.yaml"
    parent_yaml.write_text("value: !include missing_file.yaml\n")

    yaml = get_yaml_loader(typ="safe")
    with pytest.raises(FileNotFoundError, match="missing_file"):
        with parent_yaml.open("r") as f:
            yaml.load(f)
