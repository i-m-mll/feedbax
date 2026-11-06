from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import pytest

from feedbax.analysis.figures import FigureSpecExecutionError, execute_figure_spec
from feedbax.analysis.reports import (
    REPORT_RENDER_ROLE,
    ReportRecipeExecutionError,
    ReportRecipeRegistry,
    ReportRecipeResult,
    execute_report_spec,
)
from feedbax.contracts.artifact_store import store_bytes_artifact
from feedbax.contracts.base import Provenance, StrictModel
from feedbax.contracts.expressions import Compare
from feedbax.contracts.figures import FigureSpec
from feedbax.contracts.manifest import FigureManifest, ReportManifest, ReportSpec, load_manifest
from feedbax.persistence import manifest_index
from feedbax.plot.constructors import FigureRegistry, register_figure_constructor

pytestmark = pytest.mark.feedbax_contract

Execute = Callable[[], tuple[FigureManifest | ReportManifest, Path]]


@pytest.fixture
def execution(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[Execute, list[Exception]]:
    failures: list[Exception] = []
    provenance = Provenance()

    if request.param == "report":
        spec = ReportSpec(report_type="testpkg.terminal_report")

        def recipe(_spec: ReportSpec, root: Path, _inputs: list[Any]) -> ReportRecipeResult:
            if failures:
                raise failures[0]
            return ReportRecipeResult(
                artifacts=[
                    store_bytes_artifact(
                        b"# Report\n",
                        root=root,
                        role=REPORT_RENDER_ROLE,
                        logical_name="report.md",
                        media_type="text/markdown",
                    )
                ]
            )

        registry = ReportRecipeRegistry()
        registry.register(spec.report_type, recipe)
        return (
            lambda: execute_report_spec(
                spec, root=tmp_path, registry=registry, provenance=provenance
            ),
            failures,
        )

    run_condition = (
        Compare(item="params", path="enabled", op="exists")
        if request.param == "skipped-figure"
        else None
    )
    figure_spec = FigureSpec(
        name="terminal-figure",
        assembler="testpkg.terminal_figure",
        run_condition=run_condition,
    )

    def constructor(_items: Mapping[str, Any], _params: StrictModel) -> go.Figure:
        if failures:
            raise failures[0]
        return go.Figure()

    figure_registry = FigureRegistry()
    register_figure_constructor(
        "testpkg.terminal_figure",
        registry=figure_registry,
        tier="custom_figure",
        constructor=constructor,
        description="Render a minimal figure for publication tests.",
    )
    return (
        lambda: execute_figure_spec(
            figure_spec, root=tmp_path, registry=figure_registry, provenance=provenance
        ),
        failures,
    )


@pytest.mark.parametrize(
    ("execution", "execution_error"),
    [("figure", FigureSpecExecutionError), ("report", ReportRecipeExecutionError)],
    indirect=["execution"],
)
def test_computation_failure_preserves_completed_receipt_and_artifacts(
    execution: tuple[Execute, list[Exception]],
    execution_error: type[FigureSpecExecutionError] | type[ReportRecipeExecutionError],
) -> None:
    execute, failures = execution
    manifest, receipt = execute()
    assert manifest.status == "completed"
    receipt_bytes = receipt.read_bytes()
    assert manifest.artifacts
    artifacts = {}
    for artifact in manifest.artifacts:
        assert artifact.uri is not None
        path = Path(artifact.uri)
        artifacts[path] = path.read_bytes()

    failure = ValueError("injected computation failure")
    failures.append(failure)
    with pytest.raises(execution_error) as exc_info:
        execute()

    assert exc_info.value.__cause__ is failure
    assert exc_info.value.path is None
    assert receipt.read_bytes() == receipt_bytes
    for path, contents in artifacts.items():
        assert path.read_bytes() == contents


@pytest.mark.parametrize(
    ("execution", "has_artifacts"),
    [("figure", True), ("skipped-figure", False), ("report", True)],
    indirect=["execution"],
)
def test_index_failure_preserves_the_completed_receipt(
    execution: tuple[Execute, list[Exception]],
    has_artifacts: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execute, _failures = execution
    indexed_receipts: list[Path] = []
    index_failure = OSError("injected index failure")

    def fail_index(path: Path, *, root: Path) -> None:
        assert path.is_file()
        indexed_receipts.append(path)
        raise index_failure

    monkeypatch.setattr(manifest_index, "index_manifest_file", fail_index)
    with pytest.raises(OSError) as exc_info:
        execute()

    assert exc_info.value is index_failure
    manifest = load_manifest(indexed_receipts[-1])
    assert manifest.status == "completed"
    assert bool(manifest.artifacts) is has_artifacts
