"""Composable FigureSpec resolution, identity, migration, and CLI tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from feedbax.analysis import figures as figure_execution
from feedbax.analysis.figures import coerce_figure_spec, execute_figure_spec, resolve_figure_spec
from feedbax.contracts.figures import (
    FIGURE_SPEC_SCHEMA_ID,
    FIGURE_SPEC_SCHEMA_VERSION,
    FigureCompositionProvenance,
    FigureCompositionSpec,
    FigureSpec,
    PanelSpec,
    TraceBinding,
    TraceFamily,
    TraceFamilyIndex,
)
from feedbax.contracts.manifest import (
    OverridePatch,
    canonical_json_bytes,
    figure_manifest_id,
    sha256_bytes,
)
from feedbax.contracts.matrix_core import (
    SOURCE_DOCUMENT_INHERITANCE_KEY,
    ContentPinnedJsonBase,
)
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.contracts.run_matrix import MatrixCompositionDelta
from feedbax.plot.constructors import FigureRegistry

pytestmark = [pytest.mark.feedbax_contract]


def _family() -> TraceFamily:
    return TraceFamily(
        name="profiles",
        index=TraceFamilyIndex(values=[0, 1, 2]),
        trace=TraceBinding(
            name="profile-{index}",
            constructor="feedbax.profile_curves",
            panel="main",
            data={"y": [[1.0, 2.0], [2.0, 3.0]]},
            params={"line": {"width": 1}},
        ),
    )


def _base_spec(*, name: str = "shared") -> FigureSpec:
    return FigureSpec(
        name=name,
        assembler="feedbax.grid_figure",
        assembler_params={"rows": 1, "cols": 1},
        panels=[PanelSpec(name="main", title="Shared")],
        trace_families=[_family()],
        metadata={"family": "sisu_m2", "variant": "base"},
    )


def _write_payload(root: Path, name: str, payload: dict[str, Any]) -> ContentPinnedJsonBase:
    path = root / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ContentPinnedJsonBase(
        ref=name,
        sha256=sha256_bytes(canonical_json_bytes(payload)),
    )


def _composition(
    parent: ContentPinnedJsonBase,
    *deltas: MatrixCompositionDelta,
) -> FigureCompositionSpec:
    return FigureCompositionSpec(parent=parent, deltas=list(deltas))


def _replace(layer: str, path: str, value: Any, *, acknowledges: list[str] | None = None):
    return MatrixCompositionDelta(
        layer_id=layer,
        patches=[OverridePatch(op="replace", path=path, value=value)],
        acknowledges_ancestor_paths=acknowledges or [],
    )


def test_ordinary_figure_v2_is_unchanged_and_composition_v0_rejects() -> None:
    current = _base_spec().model_dump(mode="json", exclude_none=True)

    accepted = default_spec_registry.migrate("FigureSpec", current)

    assert accepted.source_version == FIGURE_SPEC_SCHEMA_VERSION
    assert accepted.target_version == FIGURE_SPEC_SCHEMA_VERSION
    assert accepted.migration_records == []
    assert resolve_figure_spec(current).figure_spec.model_dump(
        mode="json", exclude_none=True
    ) == current
    with pytest.raises(UnsupportedSpecVersion):
        default_spec_registry.migrate(
            "FigureSpec", {**current, "schema_version": "feedbax.spec.figure.v0"}
        )
    with pytest.raises(UnsupportedSpecVersion, match="migration_intentionally_absent=yes"):
        default_spec_registry.migrate(
            "FigureSpec", {**current, "schema_version": "feedbax.spec.figure.v1"}
        )
    for kind, schema_id in (
        ("FigureCompositionSpec", "feedbax.spec.figure_composition"),
        ("FigureCompositionProvenance", "feedbax.spec.figure_composition_provenance"),
    ):
        with pytest.raises(UnsupportedSpecVersion, match="migration_intentionally_absent=yes"):
            default_spec_registry.migrate(
                kind,
                {"schema_id": schema_id, "schema_version": f"{schema_id}.v0"},
            )


def test_composition_reaches_panels_and_trace_families_with_ordered_precedence(
    tmp_path: Path,
) -> None:
    root_ref = _write_payload(
        tmp_path,
        "base.figure.json",
        _base_spec().model_dump(mode="json", exclude_none=True),
    )
    parent = _composition(
        root_ref,
        _replace("shared-title", "panels.0.title", "Parent"),
        _replace("shared-width", "trace_families.0.trace.params.line.width", 2),
    )
    parent_ref = _write_payload(
        tmp_path,
        "parent.figure-composition.json",
        parent.model_dump(mode="json", exclude_none=True),
    )
    child = _composition(
        parent_ref,
        _replace(
            "publication-title",
            "panels.0.title",
            "Child",
            acknowledges=["panels.0.title"],
        ),
        _replace(
            "publication-width",
            "trace_families.0.trace.params.line.width",
            4,
            acknowledges=["trace_families.0.trace.params.line.width"],
        ),
        _replace("variant", "metadata.variant", "velocity"),
    )

    resolved = resolve_figure_spec(child, repo_root=tmp_path)

    assert resolved.figure_spec.panels[0].title == "Child"
    assert resolved.figure_spec.trace_families is not None
    assert resolved.figure_spec.trace_families[0].trace.params["line"]["width"] == 4
    assert resolved.composition is not None
    assert [layer.layer_ids for layer in resolved.composition.layers] == [
        ["shared-title", "shared-width"],
        ["publication-title", "publication-width", "variant"],
    ]
    assert resolved.composition.attribution["panels.0.title"].endswith(
        ":publication-title"
    )


@pytest.mark.parametrize(
    ("ancestor_path", "child_path", "acknowledgement"),
    [
        ("metadata", "metadata.variant", "metadata"),
        ("metadata.variant", "metadata", "metadata.variant"),
    ],
)
def test_composition_precedence_acknowledgement_is_prefix_aware_both_directions(
    tmp_path: Path,
    ancestor_path: str,
    child_path: str,
    acknowledgement: str,
) -> None:
    root = _write_payload(
        tmp_path,
        "root.json",
        _base_spec().model_dump(mode="json", exclude_none=True),
    )
    ancestor_value = {"family": "sisu_m2", "variant": "ancestor"}
    parent = _composition(root, _replace("same-id", ancestor_path, ancestor_value))
    parent_ref = _write_payload(
        tmp_path, "parent.json", parent.model_dump(mode="json", exclude_none=True)
    )
    child_value: Any = (
        {"family": "sisu_m2", "variant": "child"}
        if child_path == "metadata"
        else "child"
    )
    child = _composition(
        parent_ref,
        _replace("same-id", child_path, child_value, acknowledges=[acknowledgement]),
    )

    resolved = resolve_figure_spec(child, repo_root=tmp_path)

    assert resolved.composition is not None
    qualified = [
        layer_id
        for layer in resolved.composition.layers
        for layer_id in layer.qualified_layer_ids
    ]
    assert len(qualified) == len(set(qualified)) == 2
    assert all(layer_id.endswith(":same-id") for layer_id in qualified)
    assert set(resolved.composition.attribution.values()).issubset(set(qualified))


@pytest.mark.parametrize(
    ("ancestor_path", "child_path"),
    [
        ("metadata", "metadata.variant"),
        ("metadata.variant", "metadata"),
    ],
)
def test_composition_precedence_rejects_sibling_descendant_acknowledgement(
    tmp_path: Path,
    ancestor_path: str,
    child_path: str,
) -> None:
    root = _write_payload(
        tmp_path,
        "root.json",
        _base_spec().model_dump(mode="json", exclude_none=True),
    )
    parent = _composition(
        root,
        _replace(
            "ancestor",
            ancestor_path,
            {"family": "sisu_m2", "variant": "ancestor"},
        ),
    )
    parent_ref = _write_payload(
        tmp_path, "parent.json", parent.model_dump(mode="json", exclude_none=True)
    )
    child_value: Any = (
        {"family": "sisu_m2", "variant": "child"}
        if child_path == "metadata"
        else "child"
    )
    child = _composition(
        parent_ref,
        _replace(
            "child",
            child_path,
            child_value,
            acknowledges=["metadata.other"],
        ),
    )

    with pytest.raises(ValueError, match="overlaps ancestor-written paths"):
        resolve_figure_spec(child, repo_root=tmp_path)


def test_composition_rejects_unacknowledged_precedence_and_nested_schema_change(
    tmp_path: Path,
) -> None:
    root_ref = _write_payload(
        tmp_path,
        "base.json",
        _base_spec().model_dump(mode="json", exclude_none=True),
    )
    parent = _composition(root_ref, _replace("parent", "panels.0.title", "Parent"))
    parent_ref = _write_payload(
        tmp_path, "parent.json", parent.model_dump(mode="json", exclude_none=True)
    )
    child = _composition(parent_ref, _replace("child", "panels.0.title", "Child"))
    with pytest.raises(ValueError, match="overlaps ancestor-written paths"):
        resolve_figure_spec(child, repo_root=tmp_path)

    nested_schema_change = _composition(
        root_ref,
        _replace(
            "bad-nested-version",
            "trace_families.0.schema_version",
            "feedbax.spec.figure_trace_family.v0",
        ),
    )
    with pytest.raises(ValueError, match="changes schema identity.*without a declared"):
        resolve_figure_spec(nested_schema_change, repo_root=tmp_path)


def test_identity_is_deterministic_and_locator_is_not_authored_identity(tmp_path: Path) -> None:
    payload = _base_spec().model_dump(mode="json", exclude_none=True)
    parent = _write_payload(tmp_path, "base.json", payload)
    spec = _composition(parent, _replace("name", "name", "resolved"))
    relocated = spec.model_copy(
        update={"parent": parent.model_copy(update={"ref": "elsewhere/base.json"})}
    )

    first = resolve_figure_spec(spec, repo_root=tmp_path)
    second = resolve_figure_spec(spec, repo_root=tmp_path)

    assert first == second
    assert first.authored_identity_sha256 == figure_execution.figure_composition_envelope_hash(
        relocated
    )
    assert first.resolved_identity_sha256 == sha256_bytes(
        canonical_json_bytes(first.figure_spec.model_dump(mode="json", exclude_none=True))
    )
    assert figure_manifest_id(first.figure_spec) == figure_manifest_id(second.figure_spec)
    assert first.authored_identity_sha256 != first.resolved_identity_sha256


def test_real_hop_cycle_unknown_parent_and_unknown_template_fail_with_paths(
    tmp_path: Path,
) -> None:
    b = _composition(
        ContentPinnedJsonBase(ref="a.json", sha256="0" * 64),
        _replace("b", "name", "b"),
    )
    b_ref = _write_payload(
        tmp_path, "b.json", b.model_dump(mode="json", exclude_none=True)
    )
    a = _composition(b_ref, _replace("a", "name", "a"))
    a_ref = _write_payload(
        tmp_path, "a.json", a.model_dump(mode="json", exclude_none=True)
    )
    leaf = _composition(a_ref, _replace("leaf", "name", "leaf"))
    with pytest.raises(
        ValueError,
        match=r"figure composition cycle detected: a\.json -> b\.json -> a\.json",
    ):
        resolve_figure_spec(leaf, repo_root=tmp_path)

    unknown_ref = _write_payload(
        tmp_path,
        "unknown.json",
        {
            "schema_id": "example.unknown",
            "schema_version": "example.unknown.v1",
        },
    )
    with pytest.raises(ValueError, match="unknown.*at 'unknown.json'"):
        resolve_figure_spec(
            _composition(unknown_ref, _replace("unknown", "name", "x")),
            repo_root=tmp_path,
        )

    with pytest.raises(ValueError, match="unknown template 'missing-template'"):
        resolve_figure_spec(
            FigureSpec(name="unknown", template="missing-template"),
            registry=FigureRegistry(),
        )


def test_real_loader_failures_and_explicit_durable_identity(tmp_path: Path) -> None:
    valid = _base_spec().model_dump(mode="json", exclude_none=True)
    valid_ref = _write_payload(tmp_path, "valid.json", valid)

    with pytest.raises(ValueError, match="hash mismatch"):
        resolve_figure_spec(
            _composition(
                valid_ref.model_copy(update={"sha256": "f" * 64}),
                _replace("name", "name", "x"),
            ),
            repo_root=tmp_path,
        )
    with pytest.raises(ValueError, match="escapes repo_root"):
        resolve_figure_spec(
            _composition(
                ContentPinnedJsonBase(ref="../valid.json", sha256=valid_ref.sha256),
                _replace("name", "name", "x"),
            ),
            repo_root=tmp_path,
        )

    (tmp_path / "malformed.json").write_text("{", encoding="utf-8")
    (tmp_path / "array.json").write_text("[]", encoding="utf-8")
    for name, message in (
        ("malformed.json", "cannot load"),
        ("array.json", "must contain a JSON object"),
    ):
        with pytest.raises(ValueError, match=message):
            resolve_figure_spec(
                _composition(
                    ContentPinnedJsonBase(ref=name, sha256="0" * 64),
                    _replace("name", "name", "x"),
                ),
                repo_root=tmp_path,
            )

    wrapper = {"figures": [valid]}
    wrapper_ref = _write_payload(tmp_path, "wrapper.json", wrapper).model_copy(
        update={"payload_path": ("figures", "1")}
    )
    with pytest.raises(ValueError, match="array index out of range"):
        resolve_figure_spec(
            _composition(wrapper_ref, _replace("name", "name", "x")),
            repo_root=tmp_path,
        )

    missing = {key: value for key, value in valid.items() if key != "schema_version"}
    with pytest.raises(ValueError, match="explicitly declare schema_version"):
        resolve_figure_spec(missing)
    missing_path = tmp_path / "missing-version.json"
    missing_path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ValueError, match="explicitly declare schema_version"):
        coerce_figure_spec(missing_path)


def test_real_chain_depth_is_bounded(tmp_path: Path) -> None:
    parent = _write_payload(
        tmp_path,
        "root.json",
        _base_spec().model_dump(mode="json", exclude_none=True),
    )
    for index in range(64):
        envelope = _composition(parent, _replace(f"layer-{index}", "name", f"n{index}"))
        parent = _write_payload(
            tmp_path,
            f"layer-{index}.json",
            envelope.model_dump(mode="json", exclude_none=True),
        )
    leaf = _composition(parent, _replace("leaf", "name", "too-deep"))
    with pytest.raises(ValueError, match="exceeds maximum depth 64"):
        resolve_figure_spec(leaf, repo_root=tmp_path)


def test_execution_manifest_uses_the_same_resolved_semantics_as_display(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    direct = FigureSpec(name="base", assembler="feedbax.grid_figure")
    parent = _write_payload(
        tmp_path, "base.json", direct.model_dump(mode="json", exclude_none=True)
    )
    composed = _composition(parent, _replace("name", "name", "composed"))
    displayed = resolve_figure_spec(
        composed,
        repo_root=tmp_path,
        registry=application_registry_bundle.figures,
    )

    manifest, _ = execute_figure_spec(
        composed,
        repo_root=tmp_path,
        root=tmp_path / "runs",
        registry=application_registry_bundle.figures,
    )

    expected = displayed.figure_spec.model_dump(mode="json", exclude_none=True)
    assert manifest.figure_spec.inline == expected
    assert manifest.figure_spec.schema_version == FIGURE_SPEC_SCHEMA_VERSION
    assert FigureSpec.model_validate(manifest.figure_spec.inline) == displayed.figure_spec
    assert manifest.figure_spec.sha256 == displayed.resolved_identity_sha256
    assert manifest.id == figure_manifest_id(displayed.figure_spec)
    assert [payload.kind for payload in manifest.regeneration_specs[:2]] == [
        "FigureCompositionSpec",
        "FigureCompositionProvenance",
    ]


def test_twenty_four_families_share_one_inherited_index_without_copying(
    tmp_path: Path,
) -> None:
    shared_index = {"values": list(range(11))}
    shared_payload = {
        "schema_id": "example.figure_shared_subdocuments",
        "schema_version": "example.figure_shared_subdocuments.v1",
        "conditioning_index": shared_index,
    }
    shared_ref = _write_payload(tmp_path, "shared.json", shared_payload)
    family = _family().model_dump(mode="json", exclude_none=True)
    families: list[dict[str, Any]] = []
    inheritance: list[dict[str, Any]] = []
    for index in range(24):
        item = json.loads(json.dumps(family))
        item["name"] = f"family-{index}"
        item["trace"]["name"] = f"family-{index}-{{index}}"
        item.pop("index")
        families.append(item)
        inheritance.append(
            {
                "target": f"trace_families.{index}.index",
                "parent": {
                    **shared_ref.model_dump(mode="json", exclude_none=True),
                    "payload_path": ["conditioning_index"],
                },
            }
        )
    root = _base_spec().model_dump(mode="json", exclude_none=True)
    root["trace_families"] = families
    root[SOURCE_DOCUMENT_INHERITANCE_KEY] = {
        "schema_id": "feedbax.spec.source_document_inheritance",
        "schema_version": "feedbax.spec.source_document_inheritance.v1",
        "inherit": inheritance,
    }
    root_ref = _write_payload(tmp_path, "root.json", root)
    composed = _composition(root_ref, _replace("name", "name", "inherited"))

    resolved = resolve_figure_spec(composed, repo_root=tmp_path)

    assert resolved.composition is not None
    assert resolved.composition.source_inheritance is not None
    assert len(resolved.composition.source_inheritance.inherit) == 24
    assert len(resolved.composition.inherited_documents) == 1
    assert resolved.composition.inherited_documents[0].ref == "shared.json"
    assert resolved.figure_spec.trace_families is not None
    assert all(
        family.index.values == list(range(11))
        for family in resolved.figure_spec.trace_families
    )
    duplicated = json.loads(json.dumps(root))
    duplicated.pop(SOURCE_DOCUMENT_INHERITANCE_KEY)
    for item in duplicated["trace_families"]:
        item["index"] = shared_index
    duplicated["name"] = "inherited"
    assert resolved.figure_spec.model_dump(mode="json", exclude_none=True) == duplicated
    authored_sources = canonical_json_bytes(root) + canonical_json_bytes(shared_payload)
    repeated_index = canonical_json_bytes(shared_index)
    assert authored_sources.count(repeated_index) == 1
    assert canonical_json_bytes(duplicated).count(repeated_index) == 24


def test_manifest_retains_full_chain_custody_and_distinct_runtime_identities(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    root_ref = _write_payload(
        tmp_path,
        "root.json",
        FigureSpec(name="base", assembler="feedbax.grid_figure").model_dump(
            mode="json", exclude_none=True
        ),
    )
    middle = _composition(root_ref, _replace("middle", "name", "middle"))
    middle_ref = _write_payload(
        tmp_path,
        "middle.json",
        middle.model_dump(mode="json", exclude_none=True),
    )
    leaf = _composition(
        middle_ref,
        _replace("leaf", "name", "leaf", acknowledges=["name"]),
    )

    manifest, _ = execute_figure_spec(
        leaf,
        repo_root=tmp_path,
        root=tmp_path / "runs",
        runtime_metadata={"acceptance": True},
        registry=application_registry_bundle.figures,
    )

    provenance_payload = next(
        payload
        for payload in manifest.regeneration_specs
        if payload.kind == "FigureCompositionProvenance"
    )
    documents = provenance_payload.inline["documents"]
    assert [document["role"] for document in documents] == [
        "root_figure",
        "composition_envelope",
        "authored_leaf",
    ]
    assert [document["order"] for document in documents] == [0, 1, 2]
    for document in documents:
        assert sha256_bytes(canonical_json_bytes(document["inline"])) == document["sha256"]

    runtime = next(
        payload
        for payload in manifest.regeneration_specs
        if payload.kind == "FigureRuntimeBindingSpec"
    )
    assert runtime.inline["authored_figure_source_sha256"] == (
        provenance_payload.inline["authored_envelope_sha256"]
    )
    assert runtime.inline["resolved_figure_spec_sha256"] == manifest.figure_spec.sha256
    assert (
        runtime.inline["authored_figure_source_sha256"]
        != runtime.inline["resolved_figure_spec_sha256"]
    )

    # Re-materialize every referenced source only from manifest custody, then resolve.
    for document in documents[:-1]:
        path = tmp_path / document["ref"]
        path.unlink()
        path.write_text(json.dumps(document["inline"]), encoding="utf-8")
    regenerated = resolve_figure_spec(documents[-1]["selected_inline"], repo_root=tmp_path)
    assert regenerated.resolved_identity_sha256 == manifest.figure_spec.sha256


def test_provenance_rejects_invalid_hash_order_attribution_and_custody(
    tmp_path: Path,
) -> None:
    root_ref = _write_payload(
        tmp_path,
        "root.json",
        _base_spec().model_dump(mode="json", exclude_none=True),
    )
    resolved = resolve_figure_spec(
        _composition(root_ref, _replace("leaf", "name", "leaf")),
        repo_root=tmp_path,
    )
    assert resolved.composition is not None
    payload = resolved.composition.model_dump(mode="json", exclude_none=True)

    invalid_hash = json.loads(json.dumps(payload))
    invalid_hash["authored_envelope_sha256"] = "ABC"
    with pytest.raises(ValueError, match="lowercase sha256"):
        FigureCompositionProvenance.model_validate(invalid_hash)

    invalid_order = json.loads(json.dumps(payload))
    invalid_order["documents"][1]["order"] = 7
    with pytest.raises(ValueError, match="contiguous order"):
        FigureCompositionProvenance.model_validate(invalid_order)

    invalid_attribution = json.loads(json.dumps(payload))
    invalid_attribution["attribution"]["name"] = f"{'0' * 64}:missing"
    with pytest.raises(ValueError, match="unknown layer"):
        FigureCompositionProvenance.model_validate(invalid_attribution)

    invalid_custody = json.loads(json.dumps(payload))
    invalid_custody["documents"][0]["inline"]["name"] = "tampered"
    with pytest.raises(ValueError, match="disagrees with inline document"):
        FigureCompositionProvenance.model_validate(invalid_custody)


def test_analysis_bundle_v5_migrates_to_composable_figure_v6() -> None:
    from feedbax.analysis.bundles import ANALYSIS_BUNDLE_SCHEMA_VERSION

    migrated = default_spec_registry.migrate(
        "AnalysisBundleSpec",
        {
            "schema_id": "feedbax.spec.analysis_bundle",
            "schema_version": "feedbax.spec.analysis_bundle.v5",
            "name": "legacy",
            "stages": [
                {
                    "name": "figure",
                    "kind": "figure",
                    "figure": {"name": "legacy", "assembler": "feedbax.grid_figure"},
                }
            ],
        },
    )
    assert migrated.target_version == ANALYSIS_BUNDLE_SCHEMA_VERSION
    assert migrated.payload["schema_version"] == ANALYSIS_BUNDLE_SCHEMA_VERSION
    assert migrated.payload["stages"][0]["figure"]["schema_id"] == FIGURE_SPEC_SCHEMA_ID
    with pytest.raises(ValueError, match="explicitly declare schema_id, schema_version"):
        from feedbax.analysis.bundles import BundleStageSpec

        BundleStageSpec.model_validate(
            {
                "name": "figure",
                "kind": "figure",
                "figure": {"name": "invalid", "assembler": "feedbax.grid_figure"},
            }
        )


def test_bundle_uses_trusted_root_and_studio_rejects_composition_without_client_root(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    from fastapi import HTTPException

    from feedbax.analysis.bundles import (
        AnalysisBundleSpec,
        BundleStageSpec,
        _execute_figure_stage,
    )
    from feedbax.analysis.execution_context import (
        resolve_staged_execution_context,
        with_staged_repo_root,
    )
    from feedbax.contracts.studio_api import GenerateAnalysisRequest
    from feedbax.web.api.analysis import _figure_spec_for_request

    parent = _write_payload(
        tmp_path,
        "base.json",
        FigureSpec(name="base", assembler="feedbax.grid_figure").model_dump(
            mode="json", exclude_none=True
        ),
    )
    authored = _composition(parent, _replace("name", "name", "composed")).model_dump(
        mode="json", exclude_none=True
    )

    stage = BundleStageSpec.model_validate(
        {"name": "figure", "kind": "figure", "figure": authored}
    )
    bundle = AnalysisBundleSpec(name="composed-bundle", stages=[stage])
    context = with_staged_repo_root(resolve_staged_execution_context(None), tmp_path)
    products = _execute_figure_stage(
        stage,
        [[]],
        root=tmp_path / "runs",
        issues=[],
        bundle=bundle,
        execution_context=context,
        registries=application_registry_bundle,
    )
    assert len(products) == 1
    with pytest.raises(ValueError, match="requires repo_root"):
        _execute_figure_stage(
            stage,
            [[]],
            root=tmp_path / "untrusted",
            issues=[],
            bundle=bundle,
            execution_context=resolve_staged_execution_context(None),
            registries=application_registry_bundle,
        )
    request = GenerateAnalysisRequest(
        node_id="figure",
        job_kind="figure",
        figure_spec=authored,
    )
    with pytest.raises(HTTPException) as excinfo:
        _figure_spec_for_request(request)
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail["code"] == "figure_composition_not_supported_in_studio"


def test_figure_cli_resolve_prints_ordinary_spec_and_optional_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from feedbax.bin import figure as figure_cli

    registry = FigureRegistry()

    async def compose_application():
        await asyncio.sleep(0)
        return SimpleNamespace(bundle=SimpleNamespace(figures=registry))

    monkeypatch.setattr(figure_cli, "compose_application", compose_application)
    parent = _write_payload(
        tmp_path,
        "base.json",
        FigureSpec(name="base", assembler="feedbax.grid_figure").model_dump(
            mode="json", exclude_none=True
        ),
    )
    composed = _composition(parent, _replace("name", "name", "displayed"))
    source = tmp_path / "composed.json"
    source.write_text(
        json.dumps(composed.model_dump(mode="json", exclude_none=True)), encoding="utf-8"
    )

    assert figure_cli.main(["resolve", str(source), "--repo-root", str(tmp_path)]) == 0
    ordinary = json.loads(capsys.readouterr().out)
    assert ordinary["schema_id"] == FIGURE_SPEC_SCHEMA_ID
    assert ordinary["name"] == "displayed"
    assert "composition" not in ordinary

    assert (
        figure_cli.main(
            [
                "resolve",
                str(source),
                "--repo-root",
                str(tmp_path),
                "--with-lineage",
            ]
        )
        == 0
    )
    lineage = json.loads(capsys.readouterr().out)
    assert lineage["figure_spec"] == ordinary
    assert lineage["composition"]["authored_envelope_sha256"] == lineage["authored_identity_sha256"]


def test_composition_authoring_reference_is_indexed_and_names_public_surfaces() -> None:
    docs = Path("docs/api/figures.md").read_text(encoding="utf-8")
    navigation = Path("mkdocs.yml").read_text(encoding="utf-8")

    assert "api/figures.md" in navigation
    for public_surface in (
        "feedbax-figure resolve",
        "feedbax.analysis.figures.resolve_figure_spec",
        "FigureCompositionSpec",
        "coerce_figure_spec",
        "AnalysisBundleSpec` v6",
        "figure_composition_not_supported_in_studio",
        "feedbax.spec.figure_runtime_binding.v2",
        "MatrixCompositionDelta",
        "feedbax.spec.figure.v2",
    ):
        assert public_surface in docs
