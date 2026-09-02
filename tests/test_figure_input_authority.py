from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import plotly.graph_objs as go
import pytest
from pydantic import ValidationError

from feedbax.analysis.analysis import AbstractAnalysis
from feedbax.analysis.context import AnalysisRunContext
from feedbax.analysis.bundles import (
    AnalysisBundleSpec,
    BundleStageSpec,
    execute_staged_analysis_bundle,
)
from feedbax.analysis.execution_context import (
    StagedArtifactProviderRootBinding,
    StagedExecutionContext,
    StagedExecutionContextError,
    StagedParentArtifactProviderBinding,
    StagedParentExecutionLocation,
    with_staged_parent_artifact_provider_bindings,
)
from feedbax.analysis.figures import (
    FigureInputAuthorityError,
    RenderedFigure,
    _figure_expression_context,
    execute_figure_spec,
    resolve_figure_inputs,
)
from feedbax.analysis.manifest_inputs import (
    AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
    AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
    resolve_manifest_input,
)
from feedbax.analysis.specs import (
    AnalysisRecipeResult,
)
from feedbax.contracts.figures import (
    FIGURE_INPUT_ROLE_AUTHORITY_SCHEMA_ID,
    FIGURE_INPUT_ROLE_AUTHORITY_SCHEMA_VERSION,
    FigureArtifactPayload,
    FigureDataProductArtifactPayload,
    FigureInputAuthority,
    FigureInputRoleAuthority,
    FigureSpec,
)
from feedbax.contracts.base import (
    ParentRef,
    canonical_json_bytes,
    sha256_bytes,
)
from feedbax.contracts.manifest import (
    AnalysisDataProduct,
    AnalysisRunManifest,
    AnalysisRunSpec,
    spec_payload,
)
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.contracts.artifact_custody import ImmutableArtifactBlobProviderSpec
from feedbax.contracts.selection import ManifestPredicate
from feedbax.contracts.staged_execution import (
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
    StagedExecutionDescriptor,
)
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from tests.analysis_fixtures import build_toy_analysis_data


_STAGED_PROVIDER_ANALYSIS_TYPE = "feedbax.test.staged_figure_provider"


class _ProviderArtifactAnalysis(AbstractAnalysis):
    """Expose one provider-backed ref for manifest recording."""

    def compute(self, data, artifact, **kwargs):
        return {"artifact": artifact}


def _authority_case(tmp_path: Path, *, explicit_parent: bool = False):
    provider = ImmutableArtifactBlobProvider(tmp_path / "provider")
    certificate = {
        "schema_id": "rlrmp.bridge.certificate",
        "schema_version": "rlrmp.bridge.certificate.v1",
        "values": [1, 2, 3],
    }
    artifact = provider.store_bytes(
        json.dumps(certificate).encode(),
        role="rlrmp-bridge-standard-certificate",
        logical_name="certificate.json",
        media_type="application/json",
    )
    manifest = AnalysisRunManifest(
        id="feedbax-analysis-run:figure-authority",
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec",
            {"analysis_type": "test.figure.authority", "inputs": [], "params": {}},
        ),
        artifacts=[artifact],
        metadata={"uri": "/must/not/reach/constructor", "root": str(tmp_path)},
    )
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_artifact = provider.store_bytes(
        manifest_bytes,
        role="analysis_manifest",
        logical_name="manifest.json",
        media_type="application/json",
    )
    parent = ParentRef(
        kind=manifest.kind,
        id=manifest.id,
        role="grouped_analysis",
        metadata={
            "ref_schema_id": AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
            "ref_schema_version": AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
            "manifest_sha256": sha256_bytes(manifest_bytes),
            "size_bytes": len(manifest_bytes),
        },
    )
    context = StagedExecutionContext(
        descriptor=None,
        opened_artifact_providers={"certificates": provider},
        checkpoint_custody_roots={},
        parent_execution_locations=(
            StagedParentExecutionLocation(
                parent=parent,
                root=provider.root,
                execution_uri=str(provider.canonical_relative_path(manifest_artifact)),
                artifact_provider="certificates",
            ),
        ),
        parent_artifact_provider_bindings=(
            StagedParentArtifactProviderBinding(
                parent,
                "certificates",
                "certificates",
            ),
        ),
    )
    selector = FigureArtifactPayload(
        name="certificate",
        manifest_role="grouped_analysis",
        artifact_role="rlrmp-bridge-standard-certificate",
        artifact_provider="certificates",
        payload_schema_id="rlrmp.bridge.certificate",
        payload_schema_version="rlrmp.bridge.certificate.v1",
    )
    authority = (
        FigureInputAuthority(parent=parent, artifact_payloads=[selector])
        if explicit_parent
        else FigureInputRoleAuthority(
            input_role="grouped_analysis",
            artifact_payloads=[selector],
        )
    )
    spec = FigureSpec(
        name="authority",
        assembler="feedbax.grid_figure",
        inputs=[parent],
        input_authorities=[authority],
    )
    return provider, artifact, certificate, context, spec


def test_role_authority_schema_identity_and_canonical_selector_are_portable() -> None:
    authority = FigureInputRoleAuthority(input_role="grouped_analysis")
    payload = authority.model_dump(mode="json")

    assert payload == {
        "schema_id": FIGURE_INPUT_ROLE_AUTHORITY_SCHEMA_ID,
        "schema_version": FIGURE_INPUT_ROLE_AUTHORITY_SCHEMA_VERSION,
        "input_role": "grouped_analysis",
        "artifact_payloads": [],
    }
    assert b"parent" not in canonical_json_bytes(authority)
    assert default_spec_registry.current_version("FigureInputRoleAuthority") == (
        FIGURE_INPUT_ROLE_AUTHORITY_SCHEMA_VERSION
    )

    with pytest.raises(ValidationError, match="unsupported FigureInputRoleAuthority"):
        FigureInputRoleAuthority(
            schema_version="feedbax.spec.figure_input_role_authority.v0",
            input_role="grouped_analysis",
        )
    with pytest.raises(UnsupportedSpecVersion, match="migration_intentionally_absent=yes"):
        default_spec_registry.migrate(
            "FigureInputRoleAuthority",
            {
                "schema_id": FIGURE_INPUT_ROLE_AUTHORITY_SCHEMA_ID,
                "schema_version": "feedbax.spec.figure_input_role_authority.v0",
                "input_role": "grouped_analysis",
                "artifact_payloads": [],
            },
        )
    with pytest.raises(ValidationError, match="extra_forbidden"):
        FigureInputRoleAuthority.model_validate(
            {
                **payload,
                "root": "/machine/local/provider",
            }
        )


def test_role_authority_resolves_to_the_exact_declared_parent() -> None:
    parent = ParentRef(
        kind="AnalysisRunManifest",
        id="feedbax-analysis-run:exact",
        role="grouped_analysis",
        metadata={"manifest_sha256": "a" * 64, "size_bytes": 42},
    )
    authority = FigureInputRoleAuthority(input_role="grouped_analysis")
    spec = FigureSpec(
        name="role-selector",
        assembler="feedbax.grid_figure",
        inputs=[parent],
        input_authorities=[authority],
    )

    assert authority.resolve_parent(spec.inputs) == parent
    assert authority.resolve_parent(spec.inputs).model_dump(mode="json") == parent.model_dump(
        mode="json"
    )


@pytest.mark.parametrize("failure", ["missing", "ambiguous", "duplicate"])
def test_role_authority_rejects_missing_ambiguous_or_duplicate_selectors(
    failure: str,
) -> None:
    first = ParentRef(
        kind="AnalysisRunManifest",
        id="feedbax-analysis-run:first",
        role="grouped_analysis",
    )
    inputs = [first]
    authorities = [FigureInputRoleAuthority(input_role="grouped_analysis")]
    match = "matches no declared"
    if failure == "missing":
        authorities = [FigureInputRoleAuthority(input_role="absent")]
    elif failure == "ambiguous":
        inputs.append(
            ParentRef(
                kind="AnalysisRunManifest",
                id="feedbax-analysis-run:second",
                role="grouped_analysis",
            )
        )
        match = "is ambiguous"
    elif failure == "duplicate":
        authorities.append(FigureInputRoleAuthority(input_role="grouped_analysis"))
        match = "duplicate exact ParentRef"

    with pytest.raises(ValidationError, match=match):
        FigureSpec(
            name="invalid-role-selector",
            assembler="feedbax.grid_figure",
            inputs=inputs,
            input_authorities=authorities,
        )


def test_explicit_parent_authority_form_remains_supported(tmp_path: Path) -> None:
    _provider, artifact, certificate, context, spec = _authority_case(
        tmp_path,
        explicit_parent=True,
    )

    assert isinstance(spec.input_authorities[0], FigureInputAuthority)
    resolved = resolve_figure_inputs(spec, execution_context=context)
    assert resolved[0].artifact_payloads == {"certificate": certificate}
    assert resolved[0].artifact_refs == (artifact,)


def test_provider_canonical_payload_is_decoded_and_constructor_context_is_sanitized(
    tmp_path: Path,
) -> None:
    _provider, artifact, certificate, context, spec = _authority_case(tmp_path)
    resolved = resolve_figure_inputs(spec, execution_context=context)

    assert resolved[0].artifact_payloads == {"certificate": certificate}
    assert resolved[0].artifact_refs == (artifact,)
    expression_context = _figure_expression_context(spec, resolved)
    payload = expression_context.items["grouped_analysis"].payload
    assert payload["artifact_payloads"]["certificate"] == certificate
    serialized = json.dumps(payload)
    assert str(tmp_path) not in serialized
    assert '"uri"' not in serialized
    assert '"root"' not in serialized


def test_recorded_json_artifact_round_trips_through_public_figure_payload_path(
    tmp_path: Path,
) -> None:
    provider = ImmutableArtifactBlobProvider(tmp_path / "provider")
    certificate = {
        "schema_id": "test.recorded.figure_payload",
        "schema_version": "test.recorded.figure_payload.v1",
        "values": [3, 5, 8],
    }
    analysis_context = AnalysisRunContext(
        spec=AnalysisRunSpec(analysis_type="test.recorded.figure_payload"),
        root=provider.root,
        index_manifest=False,
    )
    recorded = analysis_context.record_json_artifact(
        certificate,
        role="recorded_figure_payload",
        logical_name="figure/payload.json",
    )
    manifest, _manifest_path = analysis_context.finalize()

    manifest_bytes = canonical_json_bytes(manifest)
    manifest_artifact = provider.store_bytes(
        manifest_bytes,
        role="analysis_manifest",
        logical_name="manifest.json",
        media_type="application/json",
    )
    parent = ParentRef(
        kind=manifest.kind,
        id=manifest.id,
        role="grouped_analysis",
        metadata={
            "ref_schema_id": AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
            "ref_schema_version": AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
            "manifest_sha256": sha256_bytes(manifest_bytes),
            "size_bytes": len(manifest_bytes),
        },
    )
    execution_context = StagedExecutionContext(
        descriptor=None,
        opened_artifact_providers={"analysis": provider},
        checkpoint_custody_roots={},
        parent_execution_locations=(
            StagedParentExecutionLocation(
                parent=parent,
                root=provider.root,
                execution_uri=str(provider.canonical_relative_path(manifest_artifact)),
                artifact_provider="analysis",
            ),
        ),
        parent_artifact_provider_bindings=(
            StagedParentArtifactProviderBinding(parent, "analysis", "analysis"),
        ),
    )
    selector = FigureArtifactPayload(
        name="payload",
        manifest_role="grouped_analysis",
        artifact_role="recorded_figure_payload",
        artifact_provider="analysis",
        payload_schema_id=certificate["schema_id"],
        payload_schema_version=certificate["schema_version"],
    )
    figure_spec = FigureSpec(
        name="recorded-payload",
        assembler="feedbax.grid_figure",
        inputs=[parent],
        input_authorities=[FigureInputAuthority(parent=parent, artifact_payloads=[selector])],
    )

    resolved = resolve_figure_inputs(figure_spec, execution_context=execution_context)
    raw_bytes = provider.get_bytes(recorded)
    digest = hashlib.sha256(raw_bytes).hexdigest()

    assert resolved[0].artifact_payloads == {"payload": certificate}
    assert resolved[0].artifact_refs == (recorded,)
    assert raw_bytes == json.dumps(certificate, indent=2, sort_keys=True).encode() + b"\n"
    assert digest == recorded.sha256
    canonical_id = f"artifact://sha256/{digest}"
    assert recorded.artifact_id == canonical_id
    assert recorded.uri == canonical_id
    assert recorded.metadata["local_relative_path"] == str(
        provider.canonical_relative_path(recorded)
    )


def test_direct_execution_records_exact_consumed_artifact(
    tmp_path: Path, application_registry_bundle
) -> None:
    _provider, artifact, _certificate, context, spec = _authority_case(tmp_path)
    manifest, path = execute_figure_spec(
        spec,
        root=tmp_path / "outputs",
        execution_context=context,
        registry=application_registry_bundle.figures,
    )
    assert path.is_file()
    assert manifest.regeneration_specs[0].kind == "FigureRuntimeBindingSpec"
    assert manifest.regeneration_specs[1:] == [artifact]


def test_sparse_authored_mapping_is_preserved_without_expanding_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    application_registry_bundle,
) -> None:
    authored = {
        "schema_id": "feedbax.spec.figure",
        "schema_version": "feedbax.spec.figure.v2",
        "name": "sparse-authored",
        "assembler": "feedbax.grid_figure",
        "metadata": {"identity": "authored"},
    }
    monkeypatch.setattr(
        "feedbax.analysis.figures._build_figures",
        lambda *_args: [RenderedFigure(name="sparse", figure=go.Figure())],
    )

    manifest, _ = execute_figure_spec(
        authored, root=tmp_path / "output", registry=application_registry_bundle.figures
    )
    authored["metadata"]["identity"] = "mutated"

    assert manifest.figure_spec.inline["metadata"] == {"identity": "authored"}


def test_invalid_authored_mapping_is_validated_before_figure_effects(
    tmp_path: Path, application_registry_bundle
) -> None:
    output_root = tmp_path / "output"
    authored = {
        "schema_id": "feedbax.spec.figure",
        "schema_version": "feedbax.spec.figure.v2",
        "name": "invalid-authored",
        "assembler": "feedbax.grid_figure",
        "unsupported": True,
    }

    with pytest.raises(ValidationError, match="extra_forbidden"):
        execute_figure_spec(
            authored, root=output_root, registry=application_registry_bundle.figures
        )

    assert not output_root.exists()


def test_multi_root_runtime_bindings_preserve_authored_figure_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    application_registry_bundle,
) -> None:
    providers = {name: ImmutableArtifactBlobProvider(tmp_path / name) for name in ("left", "right")}
    parents: list[ParentRef] = []
    locations: list[StagedParentExecutionLocation] = []
    authorities: list[FigureInputAuthority] = []
    consumed = []
    for index, (runtime_name, provider) in enumerate(providers.items()):
        payload = {
            "schema_id": "feedbax.test.figure.payload",
            "schema_version": "feedbax.test.figure.payload.v1",
            "value": index,
        }
        artifact = provider.store_bytes(
            json.dumps(payload).encode(),
            role=f"payload-{index}",
            logical_name=f"payload-{index}.json",
            media_type="application/json",
        )
        product = AnalysisDataProduct(
            product_schema_id="science.figure.product",
            product_schema_version="science.figure.product.v1",
            role=f"figure_product_{index}",
            logical_name=f"figure-product-{index}",
            producer_manifest_id=f"feedbax-analysis-run:multi-root-{index}",
            artifacts=[artifact],
        )
        analysis = AnalysisRunManifest(
            id=f"feedbax-analysis-run:multi-root-{index}",
            status="completed",
            analysis_spec=spec_payload(
                "AnalysisRunSpec",
                {"analysis_type": "test.multi_root", "inputs": [], "params": {}},
            ),
            produced_data=[product],
        )
        raw = canonical_json_bytes(analysis)
        manifest_artifact = provider.store_bytes(
            raw,
            role="analysis_manifest",
            logical_name=f"manifest-{index}.json",
            media_type="application/json",
        )
        parent = ParentRef(
            kind=analysis.kind,
            id=analysis.id,
            role=f"analysis_{index}",
            metadata={
                "ref_schema_id": AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
                "ref_schema_version": AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
                "manifest_sha256": sha256_bytes(raw),
                "size_bytes": len(raw),
            },
        )
        parents.append(parent)
        locations.append(
            StagedParentExecutionLocation(
                parent=parent,
                root=provider.root,
                execution_uri=str(provider.canonical_relative_path(manifest_artifact)),
                artifact_provider=runtime_name,
            )
        )
        authorities.append(
            FigureInputAuthority(
                parent=parent,
                artifact_payloads=[
                    FigureDataProductArtifactPayload(
                        name=f"payload_{index}",
                        manifest_role=f"analysis_{index}",
                        product_role=f"figure_product_{index}",
                        product_schema_id="science.figure.product",
                        product_schema_version="science.figure.product.v1",
                        artifact_role=f"payload-{index}",
                        artifact_provider="results",
                        payload_schema_id="feedbax.test.figure.payload",
                        payload_schema_version="feedbax.test.figure.payload.v1",
                    )
                ],
            )
        )
        consumed.append(artifact)
    context = StagedExecutionContext(
        descriptor=None,
        opened_artifact_providers=providers,
        checkpoint_custody_roots={},
        parent_execution_locations=tuple(locations),
    )
    context = with_staged_parent_artifact_provider_bindings(
        context,
        [
            StagedParentArtifactProviderBinding(parent, "results", runtime_name)
            for parent, runtime_name in zip(parents, providers, strict=True)
        ],
    )
    authored_path = (
        Path(__file__).parent / "fixtures" / "figures" / "sisu_m2_pulse_response.figure.v1.json"
    )
    authored = json.loads(authored_path.read_text(encoding="utf-8"))
    authored_payload = spec_payload("FigureSpec", authored)
    expected_authored_sha256 = "dc979c6acbbec4aa6a1cc0ab23f63e145517511fa3d6f94f92a134018fa14a4b"
    assert authored_payload.sha256 == expected_authored_sha256
    retained_before = {
        name: sorted(path.relative_to(provider.root) for path in provider.root.rglob("*"))
        for name, provider in providers.items()
    }
    monkeypatch.setattr(
        "feedbax.analysis.figures._build_figures",
        lambda *_args: [RenderedFigure(name="k5", figure=go.Figure())],
    )

    manifest, _ = execute_figure_spec(
        authored_path,
        runtime_inputs=parents,
        runtime_input_authorities=authorities,
        root=tmp_path / "output",
        execution_context=context,
        registry=application_registry_bundle.figures,
    )

    assert manifest.figure_spec == authored_payload
    assert manifest.figure_spec.inline == authored
    assert manifest.figure_spec.sha256 == expected_authored_sha256
    assert manifest.figure_spec.inline["inputs"] == []
    assert manifest.figure_spec.inline["input_authorities"] == []
    assert manifest.resolved_inputs == parents
    assert manifest.regeneration_specs[0].kind == "FigureRuntimeBindingSpec"
    assert manifest.regeneration_specs[0].inline["authored_figure_source_sha256"] == (
        authored_payload.sha256
    )
    assert manifest.regeneration_specs[0].inline["resolved_figure_spec_sha256"] == sha256_bytes(
        canonical_json_bytes(
            FigureSpec.model_validate(authored).model_dump(mode="json", exclude_none=True)
        )
    )
    assert manifest.regeneration_specs[0].inline["inputs"] == [
        parent.model_dump(mode="json", exclude_none=True) for parent in parents
    ]
    assert len(manifest.regeneration_specs[0].inline["input_authorities"]) == 2
    assert manifest.regeneration_specs[1:] == consumed
    assert retained_before == {
        name: sorted(path.relative_to(provider.root) for path in provider.root.rglob("*"))
        for name, provider in providers.items()
    }
    from feedbax.analysis.reports import (
        ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_ID,
        ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_VERSION,
        ORDERED_FIGURE_REPORT_TYPE,
        execute_report_spec,
    )
    from feedbax.contracts.manifest import ReportSpec

    evidence = ImmutableArtifactBlobProvider(tmp_path / "figure-evidence")
    figure_bytes = canonical_json_bytes(manifest)
    evidence.store_bytes(
        figure_bytes,
        role="figure_manifest",
        logical_name="multi-root-figure.json",
        media_type="application/json",
    )
    figure_parent = ParentRef(
        kind=manifest.kind,
        id=manifest.id,
        role="figure",
        metadata={
            "ref_schema_id": AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
            "ref_schema_version": AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
            "manifest_sha256": sha256_bytes(figure_bytes),
            "size_bytes": len(figure_bytes),
        },
    )
    report, _ = execute_report_spec(
        ReportSpec(
            report_type=ORDERED_FIGURE_REPORT_TYPE,
            inputs=[figure_parent],
            params={
                "schema_id": ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_ID,
                "schema_version": ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_VERSION,
                "title": "Multi-root figure",
                "sections": [
                    {
                        "title": "Evidence",
                        "figures": [
                            {
                                "input_role": "figure",
                                "figure_spec_sha256": expected_authored_sha256,
                                "caption": "Authenticated multi-root figure",
                            }
                        ],
                    }
                ],
            },
        ),
        registry=application_registry_bundle.report_recipes,
        root=tmp_path / "report",
        execution_descriptor=StagedExecutionDescriptor(
            schema_id=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
            schema_version=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
            artifact_providers={"evidence": ImmutableArtifactBlobProviderSpec()},
            checkpoint_custody={},
        ),
        artifact_provider_bindings=[StagedArtifactProviderRootBinding("evidence", evidence.root)],
    )
    assert report.status == "completed"


@pytest.mark.parametrize("failure", ["missing", "wrong_provider", "wrong_parent"])
def test_multi_root_runtime_binding_mismatch_fails_closed(
    tmp_path: Path,
    failure: str,
) -> None:
    provider, _artifact, _certificate, context, authored = _authority_case(tmp_path)
    parent = authored.inputs[0]
    binding_parent = (
        parent.model_copy(update={"id": "feedbax-analysis-run:wrong"})
        if failure == "wrong_parent"
        else parent
    )
    runtime_provider = "missing" if failure == "wrong_provider" else "certificates"
    bindings = (
        []
        if failure == "missing"
        else [
            StagedParentArtifactProviderBinding(
                binding_parent,
                "certificates",
                runtime_provider,
            )
        ]
    )
    if failure == "missing":
        context = StagedExecutionContext(
            descriptor=None,
            opened_artifact_providers={"certificates": provider},
            checkpoint_custody_roots={},
            parent_execution_locations=context.parent_execution_locations,
            parent_artifact_provider_bindings=(
                StagedParentArtifactProviderBinding(
                    parent,
                    "other-authored-label",
                    "certificates",
                ),
            ),
        )
        with pytest.raises(FigureInputAuthorityError):
            resolve_figure_inputs(authored, execution_context=context)
        return
    with pytest.raises(StagedExecutionContextError):
        with_staged_parent_artifact_provider_bindings(context, bindings)


def test_parent_scoped_provider_resolution_rejects_empty_bindings(
    tmp_path: Path,
) -> None:
    provider, _artifact, _certificate, context, spec = _authority_case(tmp_path)
    unbound = StagedExecutionContext(
        descriptor=None,
        opened_artifact_providers={"certificates": provider},
        checkpoint_custody_roots={},
        parent_execution_locations=context.parent_execution_locations,
    )

    with pytest.raises(
        FigureInputAuthorityError,
        match="artifact provider rejected",
    ):
        resolve_figure_inputs(spec, execution_context=unbound)


def test_parent_scoped_provider_binding_rejects_duplicate_and_wrong_root(
    tmp_path: Path,
) -> None:
    provider, _artifact, _certificate, context, spec = _authority_case(tmp_path)
    parent = spec.inputs[0]
    duplicate = StagedParentArtifactProviderBinding(
        parent,
        "certificates",
        "certificates",
    )
    with pytest.raises(
        StagedExecutionContextError,
        match="duplicate exact-parent",
    ):
        StagedExecutionContext(
            descriptor=None,
            opened_artifact_providers={"certificates": provider},
            checkpoint_custody_roots={},
            parent_execution_locations=context.parent_execution_locations,
            parent_artifact_provider_bindings=(duplicate, duplicate),
        )

    wrong_root = tmp_path / "wrong-root"
    wrong_root.mkdir()
    wrong_provider = ImmutableArtifactBlobProvider(wrong_root)
    wrong_root_context = StagedExecutionContext(
        descriptor=None,
        opened_artifact_providers={"certificates": wrong_provider},
        checkpoint_custody_roots={},
        parent_execution_locations=context.parent_execution_locations,
    )
    with pytest.raises(
        StagedExecutionContextError,
        match="execution root",
    ):
        with_staged_parent_artifact_provider_bindings(
            wrong_root_context,
            [duplicate],
        )


def test_parent_scoped_provider_resolution_rejects_wrong_manifest_digest(
    tmp_path: Path,
) -> None:
    provider, _artifact, _certificate, context, spec = _authority_case(tmp_path)
    parent = spec.inputs[0].model_copy(
        update={
            "metadata": {
                **spec.inputs[0].metadata,
                "manifest_sha256": "0" * 64,
            }
        }
    )
    location = context.parent_execution_locations[0]
    corrupted = StagedExecutionContext(
        descriptor=None,
        opened_artifact_providers={"certificates": provider},
        checkpoint_custody_roots={},
        parent_execution_locations=(
            StagedParentExecutionLocation(
                parent=parent,
                root=location.root,
                execution_uri=location.execution_uri,
                artifact_provider=location.artifact_provider,
            ),
        ),
        parent_artifact_provider_bindings=(
            StagedParentArtifactProviderBinding(
                parent,
                "certificates",
                "certificates",
            ),
        ),
    )
    corrupted_spec = spec.model_copy(
        update={
            "inputs": [parent],
            "input_authorities": [
                FigureInputAuthority(
                    parent=parent,
                    artifact_payloads=spec.input_authorities[0].artifact_payloads,
                )
            ],
        },
        deep=True,
    )

    with pytest.raises(
        FigureInputAuthorityError,
        match="authority rejected exact parent",
    ):
        resolve_figure_inputs(corrupted_spec, execution_context=corrupted)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("missing_product", "data product is missing"),
        ("duplicate_product", "data product is duplicated"),
        ("wrong_product_role", "data product is missing"),
        ("wrong_product_schema", "data product schema mismatch"),
        ("wrong_status", "parent status mismatch"),
    ],
)
def test_data_product_selector_failures_precede_figure_effects(
    tmp_path: Path,
    failure: str,
    message: str,
) -> None:
    provider, artifact, _certificate, original_context, original_spec = _authority_case(tmp_path)
    original_parent = original_spec.inputs[0]
    original_manifest = original_context.resolve_manifest_input(original_parent).manifest
    product = AnalysisDataProduct(
        product_schema_id="science.figure.payload",
        product_schema_version="science.figure.payload.v1",
        role="figure_payload",
        logical_name="figure-payload",
        producer_manifest_id=original_manifest.id,
        artifacts=[artifact],
    )
    products = [product]
    if failure == "missing_product":
        products = []
    elif failure == "duplicate_product":
        products.append(
            AnalysisDataProduct(
                product_schema_id=product.product_schema_id,
                product_schema_version=product.product_schema_version,
                role=product.role,
                logical_name="duplicate",
                producer_manifest_id=product.producer_manifest_id,
                artifacts=[artifact],
            )
        )
    status = "failed" if failure == "wrong_status" else "completed"
    manifest = original_manifest.model_copy(
        update={
            "status": status,
            "artifacts": [],
            "produced_data": products,
        },
        deep=True,
    )
    raw = canonical_json_bytes(manifest)
    stored = provider.store_bytes(
        raw,
        role="analysis_manifest",
        logical_name=f"{failure}.json",
        media_type="application/json",
    )
    parent = original_parent.model_copy(
        update={
            "metadata": {
                **original_parent.metadata,
                "manifest_sha256": sha256_bytes(raw),
                "size_bytes": len(raw),
            }
        }
    )
    selector = FigureDataProductArtifactPayload(
        name="payload",
        manifest_role="grouped_analysis",
        product_role=("wrong" if failure == "wrong_product_role" else "figure_payload"),
        product_schema_id=(
            "science.figure.wrong"
            if failure == "wrong_product_schema"
            else "science.figure.payload"
        ),
        product_schema_version="science.figure.payload.v1",
        artifact_role=artifact.role,
        artifact_provider="results",
        payload_schema_id="rlrmp.bridge.certificate",
        payload_schema_version="rlrmp.bridge.certificate.v1",
    )
    spec = FigureSpec(
        name="typed-product-selector",
        assembler="feedbax.grid_figure",
        inputs=[parent],
        input_authorities=[FigureInputAuthority(parent=parent, artifact_payloads=[selector])],
    )
    context = StagedExecutionContext(
        descriptor=None,
        opened_artifact_providers={"retained": provider},
        checkpoint_custody_roots={},
        parent_execution_locations=(
            StagedParentExecutionLocation(
                parent=parent,
                root=provider.root,
                execution_uri=str(provider.canonical_relative_path(stored)),
                artifact_provider="retained",
            ),
        ),
        parent_artifact_provider_bindings=(
            StagedParentArtifactProviderBinding(parent, "results", "retained"),
        ),
    )

    with pytest.raises(FigureInputAuthorityError, match=message):
        resolve_figure_inputs(spec, execution_context=context)


def test_staged_figure_resolves_prior_stage_output_through_executor_context(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    provider = ImmutableArtifactBlobProvider(tmp_path / "provider")
    certificate = {
        "schema_id": "rlrmp.bridge.certificate",
        "schema_version": "rlrmp.bridge.certificate.v1",
        "values": [1, 2, 3],
    }
    artifact = provider.store_bytes(
        json.dumps(certificate).encode(),
        role="rlrmp-bridge-standard-certificate",
        logical_name="certificate.json",
        media_type="application/json",
    )

    def recipe(_spec, _root, _inputs, _execution_context):
        return AnalysisRecipeResult(
            analyses={"provider": _ProviderArtifactAnalysis(variant="provider", cache_result=True)},
            data=build_toy_analysis_data(value=0),
            common_inputs={"artifact": artifact},
        )

    stage_one = BundleStageSpec(
        name="produce",
        kind="analysis",
        mode="grouped",
        analysis_type=_STAGED_PROVIDER_ANALYSIS_TYPE,
        requested_outputs=["provider"],
    )
    descriptor = StagedExecutionDescriptor(
        schema_id=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
        schema_version=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
        artifact_providers={"certificates": ImmutableArtifactBlobProviderSpec()},
        checkpoint_custody={},
    )
    bindings = [StagedArtifactProviderRootBinding("certificates", provider.root)]
    root = tmp_path / "outputs"
    application_registry_bundle.analysis_recipes.register(_STAGED_PROVIDER_ANALYSIS_TYPE, recipe)
    first_execution = execute_staged_analysis_bundle(
        AnalysisBundleSpec(
            name="authority-producer",
            predicate=ManifestPredicate(manifest_kind="TrainingRunManifest"),
            stages=[stage_one],
        ),
        root=root,
        execution_descriptor=descriptor,
        artifact_provider_bindings=bindings,
        registries=application_registry_bundle,
    )
    produced_parent = first_execution.stages[0].manifest_refs[0]
    selector = FigureArtifactPayload(
        name="certificate",
        manifest_role="analysis_run",
        artifact_role="rlrmp-bridge-standard-certificate",
        artifact_provider="certificates",
        payload_schema_id="rlrmp.bridge.certificate",
        payload_schema_version="rlrmp.bridge.certificate.v1",
    )

    def bundle_for(parent: ParentRef) -> AnalysisBundleSpec:
        figure = FigureSpec(
            name="authority",
            assembler="feedbax.grid_figure",
            inputs=[parent],
            input_authorities=[
                FigureInputRoleAuthority(
                    input_role="analysis_run",
                    artifact_payloads=[selector],
                )
            ],
        )
        return AnalysisBundleSpec(
            name="authority-bundle",
            predicate=ManifestPredicate(manifest_kind="TrainingRunManifest"),
            stages=[
                stage_one,
                BundleStageSpec(
                    name="figure",
                    kind="figure",
                    depends_on=["produce"],
                    figure=figure,
                ),
            ],
        )

    bundle = bundle_for(produced_parent)
    authored_figure = bundle.stages[1].figure
    assert authored_figure is not None
    authored_payload = spec_payload(
        "FigureSpec",
        authored_figure.model_dump(mode="json", exclude_none=True),
    )
    execution = execute_staged_analysis_bundle(
        bundle,
        root=root,
        execution_descriptor=descriptor,
        artifact_provider_bindings=bindings,
        registries=application_registry_bundle,
    )
    assert execution.stages[0].manifest_refs == [produced_parent]
    assert execution.stages[1].inputs == [produced_parent]
    staged_manifest = resolve_manifest_input(execution.stages[1].manifest_refs[0], root).manifest
    assert staged_manifest.figure_spec == authored_payload
    assert staged_manifest.figure_spec.inline == authored_figure.model_dump(
        mode="json",
        exclude_none=True,
    )
    assert staged_manifest.figure_spec.sha256 == authored_payload.sha256
    runtime_binding = next(
        payload
        for payload in staged_manifest.regeneration_specs
        if payload.kind == "FigureRuntimeBindingSpec"
    )
    assert runtime_binding.inline["authored_figure_source_sha256"] == authored_payload.sha256
    assert runtime_binding.inline["resolved_figure_spec_sha256"] == authored_payload.sha256
    assert runtime_binding.inline["inputs"] == [
        parent.model_dump(mode="json", exclude_none=True)
        for parent in [*authored_figure.inputs, produced_parent]
    ]
    assert runtime_binding.inline["input_authorities"] == [
        authority.model_dump(mode="json", exclude_none=True)
        for authority in authored_figure.input_authorities
    ]
    assert runtime_binding.inline["runtime_metadata"] == {
        "bundle": {
            "name": bundle.name,
            "stage": "figure",
            "index": 0,
            "schema_id": bundle.schema_id,
            "schema_version": bundle.schema_version,
        }
    }
    assert runtime_binding.inline["artifact_provider_bindings"] == [
        {
            "parent": produced_parent.model_dump(mode="json", exclude_none=True),
            "authored_provider": "certificates",
            "runtime_provider": "certificates",
        }
    ]
    assert artifact in staged_manifest.regeneration_specs

    missing_parent = produced_parent.model_copy(update={"id": f"{produced_parent.id}:missing"})
    with pytest.raises(
        FigureInputAuthorityError,
        match="runtime input/authority binding is invalid",
    ):
        execute_staged_analysis_bundle(
            bundle_for(missing_parent),
            root=root,
            execution_descriptor=descriptor,
            artifact_provider_bindings=bindings,
            registries=application_registry_bundle,
        )


def test_contained_manifest_hardlink_precedes_direct_figure_effects(
    tmp_path: Path, application_registry_bundle
) -> None:
    provider, _artifact, _certificate, context, spec = _authority_case(tmp_path)
    parent = spec.inputs[0]
    source = tmp_path / "contained" / "manifest.json"
    source.parent.mkdir()
    source.write_bytes(context.resolve_manifest_input(parent).raw_bytes)
    os.link(source, tmp_path / "manifest-hardlink.json")
    contained_context = StagedExecutionContext(
        descriptor=None,
        opened_artifact_providers={"certificates": provider},
        checkpoint_custody_roots={},
        parent_execution_locations=(
            StagedParentExecutionLocation(
                parent=parent,
                root=source.parent,
                execution_uri=source.name,
            ),
        ),
    )
    output_root = tmp_path / "outputs"

    with pytest.raises(FigureInputAuthorityError, match="authority rejected exact parent"):
        execute_figure_spec(
            spec,
            root=output_root,
            execution_context=contained_context,
            registry=application_registry_bundle.figures,
        )
    assert not output_root.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "is missing"),
        ("duplicate", "is duplicated"),
        ("wrong_role", "is missing"),
        ("wrong_media", "media type mismatch"),
        ("wrong_schema", "schema_id mismatch"),
        ("wrong_provider", "provider rejected"),
    ],
)
def test_semantic_selector_failures_precede_figure_effects(
    tmp_path: Path,
    mutation: str,
    message: str,
    application_registry_bundle,
) -> None:
    provider, artifact, _certificate, context, spec = _authority_case(tmp_path)
    selector = spec.input_authorities[0].artifact_payloads[0]
    if mutation == "missing":
        selector = selector.model_copy(update={"artifact_role": "absent"})
    elif mutation == "wrong_role":
        selector = selector.model_copy(update={"manifest_role": "wrong"})
        message = "manifest role mismatch"
    elif mutation == "wrong_media":
        selector = selector.model_copy(update={"media_type": "application/cbor"})
    elif mutation == "wrong_schema":
        selector = selector.model_copy(update={"payload_schema_id": "wrong.schema"})
    elif mutation == "wrong_provider":
        selector = selector.model_copy(update={"artifact_provider": "wrong"})
    elif mutation == "duplicate":
        duplicate = artifact.model_copy(update={"logical_name": "duplicate.json"})
        resolved_manifest = context.resolve_manifest_input(spec.inputs[0]).manifest
        replaced = resolved_manifest.model_copy(
            update={"artifacts": [artifact, duplicate]}, deep=True
        )
        raw = canonical_json_bytes(replaced)
        stored = provider.store_bytes(
            raw,
            role="analysis_manifest",
            logical_name="duplicate-manifest.json",
            media_type="application/json",
        )
        parent = spec.inputs[0].model_copy(
            update={
                "metadata": {
                    **spec.inputs[0].metadata,
                    "manifest_sha256": sha256_bytes(raw),
                    "size_bytes": len(raw),
                }
            }
        )
        context = StagedExecutionContext(
            descriptor=None,
            opened_artifact_providers={"certificates": provider},
            checkpoint_custody_roots={},
            parent_execution_locations=(
                StagedParentExecutionLocation(
                    parent=parent,
                    root=provider.root,
                    execution_uri=str(provider.canonical_relative_path(stored)),
                    artifact_provider="certificates",
                ),
            ),
        )
        spec = spec.model_copy(
            update={
                "inputs": [parent],
                "input_authorities": [
                    FigureInputAuthority(parent=parent, artifact_payloads=[selector])
                ],
            },
            deep=True,
        )
    if mutation != "duplicate":
        spec = spec.model_copy(
            update={
                "input_authorities": [
                    FigureInputAuthority(parent=spec.inputs[0], artifact_payloads=[selector])
                ]
            },
            deep=True,
        )
    output_root = tmp_path / "outputs"
    with pytest.raises(FigureInputAuthorityError, match=message):
        execute_figure_spec(
            spec,
            root=output_root,
            execution_context=context,
            registry=application_registry_bundle.figures,
        )
    assert not output_root.exists()
