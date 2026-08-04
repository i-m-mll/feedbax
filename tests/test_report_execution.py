from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from feedbax.analysis.reports import (
    REPORT_RENDER_ROLE,
    REPORT_RENDER_MEDIA_TYPES,
    ReportRecipeExecutionError,
    ReportRecipeRegistry,
    ReportRecipeResult,
    execute_authored_report_spec,
    execute_report_spec,
)
from feedbax.analysis.exact_parents import (
    STAGED_EXACT_PARENTS_SCHEMA_ID,
    STAGED_EXACT_PARENTS_SCHEMA_VERSION,
    StagedExactParentEntry,
    StagedExactParents,
)
from feedbax.bin import analysis as analysis_cli
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
from feedbax.contracts.material_dependencies import (
    MATERIAL_DEPENDENCIES_SCHEMA_ID,
    MATERIAL_DEPENDENCIES_SCHEMA_VERSION,
    MaterialDependency,
    MaterialDependencySet,
)
from feedbax.plugins.application import new_application_registry_bundle


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
        ReportRecipeRegistry().register("dummy_report", recipe)


def test_report_recipe_registry_duplicate_and_available_key_errors() -> None:
    def recipe(_report_spec: ReportSpec, _root: Path, _inputs: list[object]):
        return ReportRecipeResult()

    registry = ReportRecipeRegistry()
    registry.register("testpkg.registry_report", recipe)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("testpkg.registry_report", recipe)

    with pytest.raises(ValueError) as excinfo:
        registry.get("testpkg.missing_report")
    assert "testpkg.registry_report" in str(excinfo.value)


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

    registry = ReportRecipeRegistry()
    registry.register("testpkg.dummy_report", recipe)
    manifest, path = execute_report_spec(
        spec,
        registry=registry,
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

    registry = ReportRecipeRegistry()
    registry.register("testpkg.no_render_report", recipe)
    with pytest.raises(ReportRecipeExecutionError) as excinfo:
        execute_report_spec(spec, root=tmp_path, registry=registry)

    assert isinstance(excinfo.value.__cause__, ValueError)
    assert excinfo.value.manifest.status == "failed"
    assert excinfo.value.manifest.id == report_manifest_id(spec)
    assert excinfo.value.path.exists()

    loaded = load_manifest(excinfo.value.path)
    assert loaded.status == "failed"
    assert loaded.metadata["error"]["type"] == "ValueError"
    assert REPORT_RENDER_ROLE in loaded.metadata["error"]["message"]


def test_authored_report_exact_parents_reject_role_or_id_substitution_before_outputs(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    manifest = AnalysisRunManifest(
        id="feedbax-analysis-run:authored",
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec",
            {"analysis_type": "testpkg.source_analysis"},
        ),
    )
    raw = manifest.model_dump_json(indent=2).encode("utf-8")
    relative = Path("parents") / "analysis.json"
    path = input_root / relative
    path.parent.mkdir()
    path.write_bytes(raw)
    authored = ParentRef(
        kind=manifest.kind,
        id=manifest.id,
        role="analysis",
        metadata={
            "ref_schema_id": "feedbax.ref.authenticated_manifest",
            "ref_schema_version": "feedbax.ref.authenticated_manifest.v1",
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        },
    )
    substituted = authored.model_copy(update={"role": "replacement"})
    exact = StagedExactParents(
        schema_id=STAGED_EXACT_PARENTS_SCHEMA_ID,
        schema_version=STAGED_EXACT_PARENTS_SCHEMA_VERSION,
        parents=[
            StagedExactParentEntry(
                parent=substituted,
                execution_uri=relative.as_posix(),
            )
        ],
    )

    with pytest.raises(ValueError, match="role/ID substitution is forbidden"):
        execute_authored_report_spec(
            ReportSpec(
                report_type="testpkg.never_runs",
                inputs=[authored],
            ),
            exact_parents=exact,
            root=input_root,
            registry=ReportRecipeRegistry(),
        )

    assert not (input_root / "manifests" / "reports").exists()


def test_authored_report_exact_parents_reject_duplicate_parent_ids_before_outputs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    parent = ParentRef(
        kind="AnalysisRunManifest",
        id="feedbax-analysis-run:duplicate",
        role="first",
    )
    exact = StagedExactParents(
        schema_id=STAGED_EXACT_PARENTS_SCHEMA_ID,
        schema_version=STAGED_EXACT_PARENTS_SCHEMA_VERSION,
        parents=[
            StagedExactParentEntry(parent=parent, execution_uri="parents/first.json"),
            StagedExactParentEntry(
                parent=parent.model_copy(update={"role": "second"}),
                execution_uri="parents/second.json",
            ),
        ],
    )

    with pytest.raises(ValueError, match="duplicate ParentRef id"):
        execute_authored_report_spec(
            ReportSpec(report_type="testpkg.never_runs"),
            exact_parents=exact,
            root=root,
            registry=ReportRecipeRegistry(),
        )

    assert not (root / "manifests" / "reports").exists()


def test_authored_report_rejects_unhandled_material_dependencies_before_outputs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    manifest = AnalysisRunManifest(
        id="feedbax-analysis-run:material-report-input",
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec",
            {"analysis_type": "testpkg.source_analysis"},
        ),
    )
    raw = manifest.model_dump_json(indent=2).encode("utf-8")
    relative = Path("parents") / "analysis.json"
    path = root / relative
    path.parent.mkdir()
    path.write_bytes(raw)
    parent = ParentRef(
        kind=manifest.kind,
        id=manifest.id,
        role="analysis",
        uri=f"artifact://sha256/{hashlib.sha256(raw).hexdigest()}",
        metadata={
            "ref_schema_id": "feedbax.ref.authenticated_manifest",
            "ref_schema_version": "feedbax.ref.authenticated_manifest.v1",
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        },
    )
    exact = StagedExactParents(
        schema_id=STAGED_EXACT_PARENTS_SCHEMA_ID,
        schema_version=STAGED_EXACT_PARENTS_SCHEMA_VERSION,
        parents=[
            StagedExactParentEntry(
                parent=parent,
                execution_uri=relative.as_posix(),
                material_dependencies=MaterialDependencySet(
                    schema_id=MATERIAL_DEPENDENCIES_SCHEMA_ID,
                    schema_version=MATERIAL_DEPENDENCIES_SCHEMA_VERSION,
                    dependencies=[MaterialDependency(name="analysis_manifest", value=parent)],
                    identity_inputs=["analysis_manifest"],
                ),
            )
        ],
    )

    with pytest.raises(ValueError, match="cannot ignore.*material_dependencies"):
        execute_authored_report_spec(
            ReportSpec(
                report_type="testpkg.never_runs",
                inputs=[parent],
            ),
            exact_parents=exact,
            root=root,
            registry=ReportRecipeRegistry(),
        )

    assert not (root / "manifests" / "reports").exists()


def test_authored_report_cli_prints_failed_manifest_payload(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    manifest = AnalysisRunManifest(
        id="feedbax-analysis-run:failed-report-input",
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec",
            {"analysis_type": "testpkg.source_analysis"},
        ),
    )
    raw = manifest.model_dump_json(indent=2).encode("utf-8")
    relative = Path("parents") / "analysis.json"
    path = root / relative
    path.parent.mkdir()
    path.write_bytes(raw)
    parent = ParentRef(
        kind=manifest.kind,
        id=manifest.id,
        role="analysis",
        metadata={
            "ref_schema_id": "feedbax.ref.authenticated_manifest",
            "ref_schema_version": "feedbax.ref.authenticated_manifest.v1",
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        },
    )
    spec_path = tmp_path / "report.json"
    spec_path.write_text(
        ReportSpec(
            report_type="testpkg.cli_failed_report",
            inputs=[parent],
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    exact_path = tmp_path / "exact.json"
    exact_path.write_text(
        StagedExactParents(
            schema_id=STAGED_EXACT_PARENTS_SCHEMA_ID,
            schema_version=STAGED_EXACT_PARENTS_SCHEMA_VERSION,
            parents=[
                StagedExactParentEntry(
                    parent=parent,
                    execution_uri=relative.as_posix(),
                )
            ],
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    def recipe(
        _report_spec: ReportSpec,
        _root: Path,
        _inputs: list[object],
    ) -> ReportRecipeResult:
        return ReportRecipeResult()

    registries = new_application_registry_bundle(local_component_source=None)
    registries.report_recipes.register("testpkg.cli_failed_report", recipe)

    async def compose_application(**_kwargs):
        return SimpleNamespace(bundle=registries)

    monkeypatch.setattr(analysis_cli, "compose_application", compose_application)
    with pytest.raises(ReportRecipeExecutionError):
        analysis_cli.main(
            [
                "report",
                str(spec_path),
                "--exact-parents",
                str(exact_path),
                "--root",
                str(root),
            ]
        )

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert payload["artifacts"] == []
    failed = load_manifest(payload["manifest_path"])
    assert failed.status == "failed"
    assert failed.metadata["error"]["type"] == "ValueError"


# --------------------------------------------------------------------------
# An authored report may be handed the context its caller already resolved
# --------------------------------------------------------------------------


def test_an_authored_report_executes_against_an_already_resolved_context(
    tmp_path: Path,
) -> None:
    """The parents live in a retained store, and the caller says so once.

    Rebuilding the locations from ``StagedExactParents`` would place every
    parent beneath the report's own root, because an exact-parent entry states a
    location *within* a root and says nothing about which root. A caller that
    already resolved its parents hands the resolution over instead.
    """
    from feedbax.analysis.execution_context import (
        StagedManifestRootBinding,
        resolve_staged_execution_context,
        with_staged_resolved_parents,
        StagedParentExecutionLocation,
    )
    from feedbax.contracts.manifest import canonical_manifest_relative_path
    from feedbax.contracts.staged_execution import (
        STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
        STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
        StagedExecutionDescriptor,
    )

    retained = tmp_path / "retained"
    retained.mkdir()
    output_root = tmp_path / "receipts"
    output_root.mkdir()

    manifest = AnalysisRunManifest(
        id="feedbax-analysis-run:resolved-context",
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec", {"analysis_type": "testpkg.source_analysis"}
        ),
    )
    raw = manifest.model_dump_json(indent=2).encode("utf-8")
    relative = canonical_manifest_relative_path(manifest.kind, manifest.id)
    path = retained / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    parent = ParentRef(
        kind=manifest.kind,
        id=manifest.id,
        role="analysis_run",
        metadata={
            "ref_schema_id": "feedbax.ref.authenticated_manifest",
            "ref_schema_version": "feedbax.ref.authenticated_manifest.v1",
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        },
    )
    exact = StagedExactParents(
        schema_id=STAGED_EXACT_PARENTS_SCHEMA_ID,
        schema_version=STAGED_EXACT_PARENTS_SCHEMA_VERSION,
        parents=[StagedExactParentEntry(parent=parent, execution_uri=relative)],
    )
    context = with_staged_resolved_parents(
        resolve_staged_execution_context(
            StagedExecutionDescriptor(
                schema_id=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
                schema_version=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
                artifact_providers={},
                checkpoint_custody={},
            ),
            manifest_root_bindings=[StagedManifestRootBinding("retained", retained)],
        ),
        [
            StagedParentExecutionLocation(
                parent=parent, root=retained, execution_uri=relative
            )
        ],
    )

    def recipe(_spec: ReportSpec, root: Path, inputs: list[object]) -> ReportRecipeResult:
        artifact = store_bytes_artifact(
            b"# resolved\n",
            root=root,
            role=REPORT_RENDER_ROLE,
            logical_name="report.md",
            media_type="text/markdown",
            suffix=".md",
        )
        return ReportRecipeResult(artifacts=[artifact], summary={"inputs": len(inputs)})

    registry = ReportRecipeRegistry()
    registry.register("testpkg.resolved_context_report", recipe)

    report, report_path = execute_authored_report_spec(
        ReportSpec(report_type="testpkg.resolved_context_report", inputs=[parent]),
        registry=registry,
        exact_parents=exact,
        root=output_root,
        execution_context=context,
    )

    assert report.status == "completed"
    assert report_path.is_file()
    assert [ref.id for ref in report.provenance.parents] == [manifest.id]
    # Nothing was copied under the report's own root to make this work.
    assert not (output_root / "manifests" / "analysis_runs").exists()


def test_an_authored_report_refuses_a_resolved_context_beside_raw_bindings(
    tmp_path: Path,
) -> None:
    from feedbax.analysis.execution_context import (
        EMPTY_STAGED_EXECUTION_CONTEXT,
        StagedArtifactProviderRootBinding,
        StagedExecutionContextError,
    )

    exact = StagedExactParents(
        schema_id=STAGED_EXACT_PARENTS_SCHEMA_ID,
        schema_version=STAGED_EXACT_PARENTS_SCHEMA_VERSION,
        parents=[
            StagedExactParentEntry(
                parent=ParentRef(
                    kind="AnalysisRunManifest",
                    id="feedbax-analysis-run:irrelevant",
                    role="analysis_run",
                ),
                execution_uri="parents/analysis.json",
            )
        ],
    )
    with pytest.raises(StagedExecutionContextError, match="cannot be combined"):
        execute_authored_report_spec(
            ReportSpec(report_type="testpkg.never_runs"),
            registry=ReportRecipeRegistry(),
            exact_parents=exact,
            root=tmp_path,
            execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
            artifact_provider_bindings=[
                StagedArtifactProviderRootBinding("results", tmp_path)
            ],
        )
    assert not (tmp_path / "manifests" / "reports").exists()
