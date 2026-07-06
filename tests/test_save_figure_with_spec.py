"""Tests for feedbax.plot.io.save_figure_with_spec.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import hashlib
import json
import logging
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import plotly.graph_objs as go
import pytest

from feedbax.plot.io import _figure_render_filename, _sha256, save_figure, save_figure_with_spec
from feedbax.plugins.registry import ExperimentRegistry

pytestmark = pytest.mark.feedbax_contract


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_fig() -> go.Figure:
    """A minimal Plotly figure for testing."""
    fig = go.Figure()
    fig.add_scatter(x=[0, 1, 2], y=[0, 1, 0])
    return fig


@pytest.fixture()
def input_file(tmp_path: Path) -> Path:
    """A small binary file to use as a fake input artifact."""
    p = tmp_path / "artifact.bin"
    p.write_bytes(b"hello feedbax")
    return p


# ---------------------------------------------------------------------------
# _sha256 helper
# ---------------------------------------------------------------------------


def test_sha256_matches_hashlib(input_file: Path) -> None:
    """_sha256 must agree with hashlib.sha256 on the same file."""
    expected = hashlib.sha256(input_file.read_bytes()).hexdigest()
    assert _sha256(input_file) == expected


# ---------------------------------------------------------------------------
# save_figure_with_spec — spec file
# ---------------------------------------------------------------------------


def test_spec_file_exists(tmp_path: Path, simple_fig: go.Figure) -> None:
    """save_figure_with_spec must always write a spec JSON file."""
    spec_path, _ = save_figure_with_spec(simple_fig, {}, tmp_path, name="test")
    assert spec_path.exists()
    assert spec_path.suffix == ".json"


def test_spec_is_json_deserializable(tmp_path: Path, simple_fig: go.Figure) -> None:
    """The spec file must be valid JSON."""
    spec_path, _ = save_figure_with_spec(simple_fig, {}, tmp_path, name="test")
    with open(spec_path, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)


def test_spec_has_required_keys(tmp_path: Path, simple_fig: go.Figure) -> None:
    """The spec must contain 'versions' and 'timestamp'."""
    spec_path, _ = save_figure_with_spec(simple_fig, {}, tmp_path, name="test")
    with open(spec_path, encoding="utf-8") as f:
        data = json.load(f)
    assert "versions" in data
    assert "timestamp" in data


def test_versions_contain_feedbax(tmp_path: Path, simple_fig: go.Figure) -> None:
    """spec['versions'] must include a 'feedbax' entry."""
    spec_path, _ = save_figure_with_spec(simple_fig, {}, tmp_path, name="test")
    with open(spec_path, encoding="utf-8") as f:
        data = json.load(f)
    assert "feedbax" in data["versions"]
    assert data["versions"]["feedbax"] not in ("", "unknown")


def test_versions_contain_jax(tmp_path: Path, simple_fig: go.Figure) -> None:
    """spec['versions'] must include a 'jax' entry."""
    spec_path, _ = save_figure_with_spec(simple_fig, {}, tmp_path, name="test")
    with open(spec_path, encoding="utf-8") as f:
        data = json.load(f)
    assert "jax" in data["versions"]


def test_input_sha256_computed_automatically(
    tmp_path: Path, simple_fig: go.Figure, input_file: Path
) -> None:
    """When 'sha256' is absent from an inputs entry, it is computed and written."""
    spec: dict = {"inputs": [{"path": str(input_file)}]}
    spec_path, _ = save_figure_with_spec(simple_fig, spec, tmp_path, name="test")
    with open(spec_path, encoding="utf-8") as f:
        data = json.load(f)
    entry = data["inputs"][0]
    assert "sha256" in entry
    expected = hashlib.sha256(input_file.read_bytes()).hexdigest()
    assert entry["sha256"] == expected


def test_input_sha256_matches_actual_file(
    tmp_path: Path, simple_fig: go.Figure, input_file: Path
) -> None:
    """The SHA-256 stored in the spec must match the file on disk."""
    spec: dict = {"inputs": [{"path": str(input_file)}]}
    spec_path, _ = save_figure_with_spec(simple_fig, spec, tmp_path, name="test")
    with open(spec_path, encoding="utf-8") as f:
        data = json.load(f)
    stored_digest = data["inputs"][0]["sha256"]
    actual_digest = hashlib.sha256(input_file.read_bytes()).hexdigest()
    assert stored_digest == actual_digest


def test_existing_sha256_preserved(
    tmp_path: Path, simple_fig: go.Figure, input_file: Path
) -> None:
    """A pre-computed sha256 in the caller's spec is not overwritten."""
    precomputed = "deadbeef" * 8  # 64 hex chars
    spec: dict = {"inputs": [{"path": str(input_file), "sha256": precomputed}]}
    spec_path, _ = save_figure_with_spec(simple_fig, spec, tmp_path, name="test")
    with open(spec_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["inputs"][0]["sha256"] == precomputed


def test_caller_spec_not_mutated(
    tmp_path: Path, simple_fig: go.Figure, input_file: Path
) -> None:
    """save_figure_with_spec must not modify the caller's spec dict."""
    original: dict = {"inputs": [{"path": str(input_file)}], "seed": 42}
    spec_copy = {
        "inputs": [dict(e) for e in original["inputs"]],
        "seed": original["seed"],
    }
    save_figure_with_spec(simple_fig, original, tmp_path, name="test")
    # 'sha256' should NOT have been added to the caller's original dict
    assert original == spec_copy


def test_name_none_uses_timestamp(tmp_path: Path, simple_fig: go.Figure) -> None:
    """When name=None, a timestamped filename is generated."""
    spec_path, _ = save_figure_with_spec(simple_fig, {}, tmp_path)
    assert spec_path.exists()
    # Filename should match the UTC timestamp pattern YYYYMMDDTHHMMSSZ
    import re
    assert re.match(r"\d{8}T\d{6}Z\.json$", spec_path.name)


def test_dst_dir_created_if_absent(tmp_path: Path, simple_fig: go.Figure) -> None:
    """dst_dir is created when it does not exist."""
    dst = tmp_path / "new" / "nested" / "dir"
    assert not dst.exists()
    save_figure_with_spec(simple_fig, {}, dst, name="test")
    assert dst.exists()


# ---------------------------------------------------------------------------
# save_figure_with_spec — figure render
# ---------------------------------------------------------------------------


def test_plotly_json_render_written(tmp_path: Path, simple_fig: go.Figure) -> None:
    """With render_format='json', the figure is written as <name>.fig.json."""
    _, render_path = save_figure_with_spec(
        simple_fig, {}, tmp_path, name="test", save_render=True, render_format="json"
    )
    assert render_path is not None
    assert render_path.exists()
    assert render_path.name == "test.fig.json"


def test_render_filename_helper_matches_written_plotly_file(
    tmp_path: Path, simple_fig: go.Figure
) -> None:
    """Render filename derivation must round-trip through the actual writer."""
    _, render_path = save_figure_with_spec(
        simple_fig, {}, tmp_path, name="test", save_render=True, render_format="json"
    )
    assert render_path is not None
    assert render_path.name == _figure_render_filename("test", "json", "plotly")


def test_save_figure_atomic_symlink_replaces_stale_target(
    tmp_path: Path, simple_fig: go.Figure, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Routed figure saves replace an existing symlink with the actual render name."""
    package_root = tmp_path / "toy_plot_pkg"
    package_root.mkdir()
    (package_root / ".git").mkdir()
    package_file = package_root / "__init__.py"
    package_file.write_text("", encoding="utf-8")

    package = ModuleType("toy_plot_pkg")
    package.__file__ = str(package_file)

    registry = ExperimentRegistry()
    registry.register_package(
        "toy",
        package,
        parts=[],
        analysis_module_root="analysis",
        training_module_root="training",
        config_resource_root="config",
        figure_routing={
            "spec_dir_template": "results/{experiment}/figures/{topic}",
            "render_dir_template": "_artifacts/{experiment}/figures/{topic}",
            "render_format": "json",
            "create_symlink_in_spec_dir": True,
        },
    )

    import feedbax.plugins as plugins

    monkeypatch.setattr(plugins, "EXPERIMENT_REGISTRY", registry)
    spec_dir = package_root / "results" / "exp" / "figures" / "topic"
    spec_dir.mkdir(parents=True)
    stale_target = spec_dir / "stale.fig.json"
    stale_target.write_text("{}", encoding="utf-8")
    symlink_path = spec_dir / "figure.fig.json"
    symlink_path.symlink_to(stale_target.name)

    result = save_figure(simple_fig, {}, package="toy", experiment="exp", topic="topic")

    assert result["symlink_path"] == symlink_path
    assert symlink_path.is_symlink()
    assert symlink_path.resolve() == result["render_path"]
    assert not list(spec_dir.glob(".figure.fig.json.*.tmp"))


def test_save_figure_with_spec_close_closes_matplotlib_figures(tmp_path: Path) -> None:
    """Saving with close=True leaves no Matplotlib figures open."""
    plt.close("all")
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    save_figure_with_spec(fig, {}, tmp_path, name="mpl", render_format="png", close=True)

    assert plt.get_fignums() == []


def test_unknown_render_format_rejected(tmp_path: Path, simple_fig: go.Figure) -> None:
    with pytest.raises(ValueError, match="Unsupported render_format"):
        save_figure_with_spec(
            simple_fig,
            {},
            tmp_path,
            name="test",
            save_render=True,
            render_format="figg",
        )


def test_save_render_false_returns_none_render(
    tmp_path: Path, simple_fig: go.Figure
) -> None:
    """When save_render=False, render_path is None and no render file is written."""
    _, render_path = save_figure_with_spec(
        simple_fig, {}, tmp_path, name="test", save_render=False
    )
    assert render_path is None
    # Only the spec file should exist
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].name == "test.json"


def test_round_trip_spec_and_render(tmp_path: Path, simple_fig: go.Figure) -> None:
    """Spec and figure files coexist and both are valid after a save."""
    spec_path, render_path = save_figure_with_spec(
        simple_fig, {"seed": 7}, tmp_path, name="fig1"
    )
    # Spec
    with open(spec_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["seed"] == 7
    assert "versions" in data
    # Render
    assert render_path is not None
    assert render_path.exists()
    with open(render_path, encoding="utf-8") as f:
        fig_data = json.load(f)
    assert "data" in fig_data  # plotly JSON always has 'data' key


def test_extra_packages_in_versions(tmp_path: Path, simple_fig: go.Figure) -> None:
    """Extra package names passed via extra_packages appear in spec['versions']."""
    spec_path, _ = save_figure_with_spec(
        simple_fig, {}, tmp_path, name="test", extra_packages=["numpy"]
    )
    with open(spec_path, encoding="utf-8") as f:
        data = json.load(f)
    assert "numpy" in data["versions"]


def test_unknown_package_recorded_as_unknown(
    tmp_path: Path, simple_fig: go.Figure
) -> None:
    """An uninstalled package name is recorded as 'unknown' rather than raising."""
    spec_path, _ = save_figure_with_spec(
        simple_fig, {}, tmp_path, name="test",
        extra_packages=["_nonexistent_package_xyz"]
    )
    with open(spec_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["versions"]["_nonexistent_package_xyz"] == "unknown"


def test_type_error_for_unknown_fig_type(tmp_path: Path) -> None:
    """Passing an unrecognised figure type with save_render=True raises TypeError."""
    with pytest.raises(TypeError, match="Unrecognised figure type"):
        save_figure_with_spec("not-a-figure", {}, tmp_path, name="test")
