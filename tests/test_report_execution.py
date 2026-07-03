from __future__ import annotations

from pathlib import Path

import pytest

from feedbax.analysis.reports import (
    REPORT_RENDER_ROLE,
    REPORT_RENDER_MEDIA_TYPES,
    ReportRecipeExecutionError,
    ReportRecipeResult,
    execute_report_spec,
    get_report_recipe,
    register_report_recipe,
    unregister_report_recipe,
)
from feedbax.analysis.validation import RecipeValidationError
from feedbax.contracts.manifest import (
    AnalysisRunManifest,
    ParentRef,
    ReportSpec,
    load_manifest,
    report_manifest_id,
    spec_payload,
    store_bytes_artifact,
    write_manifest,
)


def _write_analysis_manifest(root: Path) -> ParentRef:
    manifest = AnalysisRunManifest(
        id="feedbax-analysis-run:report-input",
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec",
            {"analysis_type": "testpkg.source_analysis"},
        ),
    )
    path = write_manifest(manifest, root=root)
    return ParentRef(
        kind="AnalysisRunManifest",
        id=manifest.id,
        role="analysis_run",
        uri=str(path.relative_to(root)),
    )


def test_report_recipe_registration_rejects_bare_type_key() -> None:
    def recipe(_report_spec: ReportSpec, _root: Path, _inputs: list[object]):
        return ReportRecipeResult()

    with pytest.raises(RecipeValidationError, match="<package>\\.<name>"):
        register_report_recipe("dummy_report", recipe, replace=True)


def test_report_recipe_registry_duplicate_and_available_key_errors() -> None:
    def recipe(_report_spec: ReportSpec, _root: Path, _inputs: list[object]):
        return ReportRecipeResult()

    register_report_recipe("testpkg.registry_report", recipe, replace=True)
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_report_recipe("testpkg.registry_report", recipe)

        with pytest.raises(ValueError) as excinfo:
            get_report_recipe("testpkg.missing_report")
        assert "testpkg.registry_report" in str(excinfo.value)
    finally:
        unregister_report_recipe("testpkg.registry_report")


def test_report_spec_executes_registered_recipe_and_writes_markdown_render(
    tmp_path: Path,
) -> None:
    parent = _write_analysis_manifest(tmp_path)
    spec = ReportSpec(
        report_type="testpkg.dummy_report",
        inputs=[parent],
        params={"format": "markdown"},
        narrative="A downstream report.",
    )

    def recipe(
        report_spec: ReportSpec,
        root: Path,
        inputs: list[object],
    ) -> ReportRecipeResult:
        markdown = f"# Dummy report\n\nInputs: {len(inputs)}\n"
        artifact = store_bytes_artifact(
            markdown.encode("utf-8"),
            root=root,
            role=REPORT_RENDER_ROLE,
            logical_name="dummy-report.md",
            media_type="text/markdown",
            suffix=".md",
            metadata={"report_type": report_spec.report_type},
        )
        return ReportRecipeResult(
            artifacts=[artifact],
            summary={"inputs": len(inputs)},
            metadata={"producer": "testpkg"},
            regeneration_specs=[parent],
        )

    register_report_recipe("testpkg.dummy_report", recipe, replace=True)
    try:
        manifest, path = execute_report_spec(
            spec,
            root=tmp_path,
            issues=["132f98c"],
        )

        assert manifest.status == "completed"
        assert manifest.id == report_manifest_id(spec)
        assert path.exists()
        assert manifest.inputs == [parent]
        assert manifest.provenance.parents == [parent]
        assert manifest.provenance.issues == ["132f98c"]
        assert manifest.provenance.entrypoint is not None
        assert manifest.provenance.entrypoint.name == "testpkg.dummy_report"
        assert manifest.metadata["summary"] == {"inputs": 1}
        assert manifest.metadata["producer"] == "testpkg"
        assert manifest.regeneration_specs == [parent]

        render = manifest.artifacts[0]
        assert render.role == REPORT_RENDER_ROLE
        assert render.media_type in REPORT_RENDER_MEDIA_TYPES
        assert render.sha256 is not None
        assert Path(render.uri or "").read_text(encoding="utf-8").startswith("# Dummy report")

        loaded = load_manifest(path)
        assert loaded == manifest
    finally:
        unregister_report_recipe("testpkg.dummy_report")


def test_report_spec_writes_failed_manifest_when_recipe_omits_render(
    tmp_path: Path,
) -> None:
    parent = _write_analysis_manifest(tmp_path)
    spec = ReportSpec(report_type="testpkg.no_render_report", inputs=[parent])

    def recipe(
        _report_spec: ReportSpec,
        _root: Path,
        _inputs: list[object],
    ) -> ReportRecipeResult:
        return ReportRecipeResult()

    register_report_recipe("testpkg.no_render_report", recipe, replace=True)
    try:
        with pytest.raises(ReportRecipeExecutionError) as excinfo:
            execute_report_spec(spec, root=tmp_path)

        assert isinstance(excinfo.value.__cause__, ValueError)
        assert excinfo.value.manifest.status == "failed"
        assert excinfo.value.manifest.id == report_manifest_id(spec)
        assert excinfo.value.path.exists()

        loaded = load_manifest(excinfo.value.path)
        assert loaded.status == "failed"
        assert loaded.metadata["error"]["type"] == "ValueError"
        assert REPORT_RENDER_ROLE in loaded.metadata["error"]["message"]
    finally:
        unregister_report_recipe("testpkg.no_render_report")
