from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

from feedbax.execution.backends import render_modal_app
from feedbax.execution.container import evict_bundled_modules, rewrite_embedded_paths
from feedbax.execution.models import ExecutionCell, ExecutionSpec, RepoSource
from feedbax.execution.planning import prepare_execution_plan


def _no_embed_golden_spec() -> ExecutionSpec:
    return ExecutionSpec(
        backend="modal",
        job_id="modal-no-embed-golden",
        command="python scripts/run_cell.py",
        cells=[ExecutionCell(id="cell-a", params={"seed": 1})],
        repos=[
            RepoSource(
                name="feedbax",
                role="project",
                install_mode="github-ref",
                package="feedbax",
                git_url="https://github.com/i-m-mll/feedbax.git",
                git_ref="abc123",
            )
        ],
        modal={"use_spawn_map": False, "secrets": ["feedbax-github"]},
    )


def test_modal_no_embed_render_matches_prechange_golden() -> None:
    fixture = Path("tests/fixtures/execution/modal_no_embed_golden.txt")

    assert render_modal_app(_no_embed_golden_spec()) == fixture.read_text(encoding="utf-8")


def test_modal_local_embed_render_contains_sources_rewrites_and_uv(tmp_path: Path) -> None:
    feedbax_root = tmp_path / "feedbax"
    cookbook_root = tmp_path / "jax-cookbook"
    feedbax_root.mkdir()
    cookbook_root.mkdir()
    spec = ExecutionSpec(
        backend="modal",
        job_id="modal-embed-render",
        command="python scripts/run_cell.py --seed 1",
        cells=[ExecutionCell(id="cell-a", params={"seed": 1})],
        repos=[
            RepoSource(
                name="feedbax",
                role="project",
                install_mode="local-embed",
                package="feedbax",
                local_path=str(feedbax_root),
                extra_path_rewrites={"../../../20 Feedbax/feedbax": "/workspace/feedbax"},
            ),
            RepoSource(
                name="jax-cookbook",
                role="dependency",
                install_mode="local-embed",
                package="jax-cookbook",
                local_path=str(cookbook_root),
                target_path="/workspace/jax-cookbook",
                ignore_parts=[".git", ".venv"],
                rewrite_files=["pyproject.toml"],
            ),
        ],
        primary_repo="feedbax",
        modal={"extra_install_commands": ['uv pip install -U "jax[cuda12]"']},
    )

    rendered = render_modal_app(spec)
    plan = prepare_execution_plan(spec)

    assert rendered.count("image = image.add_local_dir(") == 2
    assert f'"{feedbax_root}"' in rendered
    assert f'"{cookbook_root}"' in rendered
    assert 'remote_path="/workspace/feedbax"' in rendered
    assert 'remote_path="/workspace/jax-cookbook"' in rendered
    assert "def _ignore_source_factory" in rendered
    assert "part.endswith(ignore_suffix_tuple)" in rendered
    assert "REWRITE_COMMAND = r'''python - <<'PY'" in rendered
    assert "def rewrite_embedded_paths(files, replacements):" in rendered
    assert "text = text.replace(old, new)" in rendered
    assert "re.sub" not in rendered
    assert '"../../../20 Feedbax/feedbax": "/workspace/feedbax"' in rendered
    assert '"/workspace/feedbax/uv.lock"' in rendered
    assert '"/workspace/jax-cookbook/uv.lock"' not in rendered
    assert ".apt_install(*APT_PACKAGES)" in rendered
    assert '.pip_install("uv")' in rendered
    assert '"uv sync"' in rendered
    assert "uv pip install -U" in rendered
    assert "jax[cuda12]" in rendered
    assert 'DEFAULT_COMMAND = "uv run --no-sync python scripts/run_cell.py --seed 1"' in rendered
    assert "source_provenance.json" in rendered
    assert plan.cloud_payload["cells"][0]["command"] == (
        "uv run --no-sync python scripts/run_cell.py --seed 1"
    )


def test_rewrite_embedded_paths_uses_literal_replacements_and_skips_missing(
    tmp_path: Path,
) -> None:
    editable_file = tmp_path / "pyproject.toml"
    editable_file.write_text("path = 'a.*b'\nother = 'axb'\n", encoding="utf-8")

    rewrite_embedded_paths(
        [editable_file, tmp_path / "missing.lock"],
        {"a.*b": "/workspace/project"},
    )

    assert editable_file.read_text(encoding="utf-8") == (
        "path = '/workspace/project'\nother = 'axb'\n"
    )


def test_evict_bundled_modules_removes_only_marked_modules() -> None:
    bundled = types.ModuleType("feedbax_test_bundled")
    bundled.__file__ = "/tmp/__modal/site/deps/feedbax_test_bundled.py"
    ordinary = types.ModuleType("feedbax_test_ordinary")
    ordinary.__file__ = "/tmp/site-packages/feedbax_test_ordinary.py"
    sys.modules[bundled.__name__] = bundled
    sys.modules[ordinary.__name__] = ordinary

    try:
        evict_bundled_modules([bundled.__name__, ordinary.__name__])

        assert bundled.__name__ not in sys.modules
        assert sys.modules[ordinary.__name__] is ordinary
    finally:
        sys.modules.pop(bundled.__name__, None)
        sys.modules.pop(ordinary.__name__, None)


def test_local_embed_requires_local_path() -> None:
    with pytest.raises(ValidationError, match="local-embed sources require local_path"):
        RepoSource(name="feedbax", install_mode="local-embed")


def test_local_embed_reproducibility_records_dirty_checkout(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
    (repo / "pyproject.toml").write_text("[project]\nname = 'fixture'\n", encoding="utf-8")
    subprocess.run(["git", "add", "pyproject.toml"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=repo, check=True)
    (repo / "pyproject.toml").write_text("[project]\nname = 'dirty'\n", encoding="utf-8")

    plan = prepare_execution_plan(
        ExecutionSpec(
            backend="modal",
            job_id="dirty-embed",
            command="python train.py",
            repos=[
                RepoSource(
                    name="fixture",
                    role="project",
                    install_mode="local-embed",
                    local_path=str(repo),
                )
            ],
        )
    )

    record = plan.reproducibility["local_embed_sources"][0]
    assert record["name"] == "fixture"
    assert record["local_path"] == str(repo)
    assert record["commit"]
    assert record["branch"] in {"main", "master"}
    assert record["dirty"] is True
    assert any("local-embed is a development mode" in warning for warning in plan.warnings)
    assert any("dirty=True" in warning for warning in plan.warnings)
