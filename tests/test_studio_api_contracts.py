from __future__ import annotations

from feedbax.component_registry import ComponentPack
from feedbax.contracts.representation import RepresentationSpec

from feedbax.runtime.components import Gain
from feedbax.runtime.graph import Component
from feedbax.contracts.portable import Ports
from feedbax.studio.component_presentation import PresentationOverride

from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import get_args, get_origin

import plotly.graph_objects as go
import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError, model_validator

from feedbax.analysis.analysis import AbstractAnalysis
from feedbax.analysis.context import AnalysisRunContext
from feedbax.analysis.execution import run_analyses_with_context
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.array_values import (
    ConstantArrayValueSpec,
    SparseCooArrayValueSpec,
    SparseCooEntrySpec,
)
from feedbax.contracts.studio_api import (
    AnalysisPackagesResponse,
    ComponentListResponse,
    STUDIO_API_TRANSPORT_SCHEMA_ID,
    STUDIO_API_TRANSPORT_SCHEMA_VERSION,
    SuccessPayload,
)
from feedbax.contracts.manifest import AnalysisRunSpec, load_manifest
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.contracts.representation import REPRESENTATION_SCHEMA_VERSION
from feedbax.web.app import create_app
from feedbax.web.api import components as components_api
from feedbax.web.api import figures as figures_api
from feedbax.plugins.bootstrap import BootstrapState
from feedbax.plugins.registry import ExperimentRegistry
from feedbax.contracts.training import LrScheduleSpec, OptimizerSpec
from scripts.generate_studio_contracts import (
    AUTHORED_CONTRACT_MODELS,
    CONTRACT_MODEL_NAMES,
    MODEL_TYPES,
    OUTPUT,
    generate,
    emit_interface,
    emit_schema,
    model_dependencies,
)
from tests.support.analysis_fixtures import ToyAnalysis, build_toy_analysis_data


GENERATED_STUDIO_PREFIXES = (
    "/api/analyses",
    "/api/components",
    "/api/domains",
    "/api/graphs",
    "/api/inspection",
    "/api/penzai",
    "/api/projects",
    "/api/runs",
    "/api/trajectories",
)

def _response_model_members(response_model: object) -> list[type[BaseModel]]:
    origin = get_origin(response_model)
    if origin is list:
        response_model = get_args(response_model)[0]

    if isinstance(response_model, type) and issubclass(response_model, BaseModel):
        return [response_model]
    return []


def test_studio_api_openapi_uses_plural_analysis_jobs_route() -> None:
    app = create_app()
    paths = app.openapi()["paths"]

    assert "/api/analyses/jobs" in paths
    assert "/api/analyses/jobs/status/{request_id}" in paths
    assert "/api/analysis/generate" not in paths


def test_analysis_packages_advertise_effective_output_ports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FEEDBAX_WEB_DATA", raising=False)

    class InheritedAnalysis(ToyAnalysis):
        pass

    class ComputeOnlyAnalysis(ToyAnalysis):
        make_figs = AbstractAnalysis.make_figs

    class FigureOnlyAnalysis(ComputeOnlyAnalysis):
        compute = AbstractAnalysis.compute
        _params_to_save = AbstractAnalysis._params_to_save

        def make_figs(self, data, *, result, **kwargs):
            return {"main": go.Figure(go.Scatter(y=[data.states["value"]]))}

    class GenericAnalysis(FigureOnlyAnalysis):
        make_figs = AbstractAnalysis.make_figs

    analyses = {
        "direct": ToyAnalysis(variant="toy"),
        "inherited": InheritedAnalysis(variant="toy"),
        "figure_only": FigureOnlyAnalysis(variant="toy"),
        "compute_only": ComputeOnlyAnalysis(variant="toy"),
        "generic": GenericAnalysis(variant="toy"),
    }
    context = AnalysisRunContext(
        spec=AnalysisRunSpec(analysis_type="toy_analysis", params={"outputs": list(analyses)}),
        root=tmp_path,
    )
    run_analyses_with_context(
        analyses, build_toy_analysis_data(), context, fig_dump_formats=["json"]
    )
    manifest = load_manifest(context.manifest_path)
    published_figures = {
        artifact.metadata["analysis_label"]
        for artifact in manifest.artifacts
        if artifact.role == "figure" and Path(artifact.uri).is_file()
    }
    assert published_figures == {"direct", "inherited", "figure_only"}

    registry = ExperimentRegistry()
    registry.register_package("test", ModuleType("test"), ["toy"], "analysis", "training", "config")
    module = SimpleNamespace(ANALYSES=analyses)
    monkeypatch.setattr(registry, "get_analysis_module", lambda _part: module)

    app = create_app()
    with TestClient(app) as client:
        current = app.state.bootstrap_state
        app.state.bootstrap_state = BootstrapState(
            replace(current.bundle, experiment_packages=registry), current.provenance
        )
        response = client.get("/api/analyses/packages")

    assert response.status_code == 200
    (package,) = AnalysisPackagesResponse.model_validate(response.json()).data.packages
    assert {item.name: item.outputPorts for item in package.analyses} == {
        "ToyAnalysis": ["data", "figure"],
        "InheritedAnalysis": ["data", "figure"],
        "FigureOnlyAnalysis": ["figure"],
        "ComputeOnlyAnalysis": ["data"],
        "GenericAnalysis": ["data"],
    }


def test_figure_registry_api_lists_bootstrapped_constructors() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/api/figures/constructors")

    assert response.status_code == 200
    assert any(item["key"] == "feedbax.grid_figure" for item in response.json())


def test_figure_registry_empty_root_returns_empty_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(figures_api, "default_manifest_root", lambda: tmp_path)

    with TestClient(create_app()) as client:
        response = client.get("/api/figures/")

    assert response.status_code == 200
    assert response.json() == {"items": [], "total": 0, "limit": 50, "offset": 0}


def test_figure_registry_unreadable_manifest_fails_without_exposing_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_path = tmp_path / "manifests" / "figures" / "private-custody.json"
    private_path.parent.mkdir(parents=True)
    private_path.write_text("not-json", encoding="utf-8")
    monkeypatch.setattr(figures_api, "default_manifest_root", lambda: tmp_path)

    with TestClient(create_app()) as client:
        response = client.get("/api/figures/")

    assert response.status_code == 500
    assert response.json() == {"detail": "Figure registry contains an unreadable manifest"}
    assert str(private_path) not in response.text


def test_figure_file_endpoint_returns_content_with_literal_media_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    formats = {
        "json": "application/json",
        "html": "text/html; charset=utf-8",
        "png": "image/png",
        "svg": "image/svg+xml",
        "webp": "image/webp",
        "pdf": "application/pdf",
    }
    artifacts = {}
    for extension in formats:
        artifact_path = tmp_path / f"figure.{extension}"
        artifact_path.write_bytes(b"rendered")
        artifacts[extension] = SimpleNamespace(uri=str(artifact_path))
    monkeypatch.setattr(figures_api, "_figure_manifests", lambda: [SimpleNamespace(id="figure-1")])
    monkeypatch.setattr(figures_api, "_render_artifact", lambda _manifest, fmt: artifacts[fmt])

    with TestClient(create_app()) as client:
        for extension, content_type in formats.items():
            response = client.get(f"/api/figures/figure-1/file?format={extension}")
            assert response.status_code == 200, extension
            assert response.headers["content-type"] == content_type, extension
            assert response.content == b"rendered", extension


def test_figure_file_endpoint_rejects_unsupported_format() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/api/figures/figure-1/file?format=tiff")

    assert response.status_code == 400
    assert response.json() == {
        "detail": (
            "Unsupported format 'tiff'. Allowed formats: "
            "['html', 'json', 'pdf', 'png', 'svg', 'webp']"
        )
    }


def test_analysis_bundle_dry_run_endpoint_returns_stage_status() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/analyses/bundles/dry-run",
        json={
            "bundle": {
                "schema_id": "feedbax.spec.analysis_bundle",
                "schema_version": "feedbax.spec.analysis_bundle.v2",
                "name": "dry-run-test",
                "predicate": {"manifest_kind": "EvaluationRunManifest"},
                "stages": [
                    {
                        "name": "disabled",
                        "kind": "analysis",
                        "skip_reason": "disabled for this bundle",
                    }
                ],
            },
            "preview_limit": 10,
        },
    )

    assert response.status_code == 200
    payload = response.json()["data"]["dry_run"]
    assert payload["bundle_name"] == "dry-run-test"
    assert payload["stages"][0]["status"] == "would_skip"
    assert payload["stages"][0]["reason"] == "disabled for this bundle"


def test_component_api_serves_representation_contract() -> None:
    class ApiRepresentedGain(Gain):
        """API represented component fixture."""

    registry = ComponentRegistry()
    registry.ingest(
        ComponentPack(
            namespace="example",
            components=(ApiRepresentedGain,),
            presentation={
                ApiRepresentedGain: PresentationOverride(
                    representation=RepresentationSpec.model_validate(
                        {
                            "anchors": [
                                {
                                    "id": "endpoint",
                                    "semantic_role": "endpoint",
                                    "interaction_roles": ["selectable"],
                                    "binding": {"kind": "param_path", "path": "gain"},
                                }
                            ],
                            "elements": [
                                {
                                    "id": "glyph",
                                    "archetype": "marker",
                                    "anchors": ["endpoint"],
                                    "frame_provider": {
                                        "kind": "from_input_port",
                                        "input_port": "input",
                                    },
                                }
                            ],
                        }
                    )
                )
            },
        )
    )
    app = create_app()
    with TestClient(app) as client:
        registry.seal()
        current = app.state.bootstrap_state
        app.state.bootstrap_state = BootstrapState(
            replace(current.bundle, components=registry), current.provenance
        )
        response = client.get("/api/components")

    assert response.status_code == 200
    contract = ComponentListResponse.model_validate(response.json())
    represented = next(
        item for item in contract.data.components if item.name == "example.ApiRepresentedGain"
    )
    assert represented.representation is not None
    assert represented.representation.schema_version == REPRESENTATION_SCHEMA_VERSION
    assert represented.representation.elements[0].archetype == "marker"


def test_component_refresh_atomically_swaps_whole_bootstrap_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app()
    with TestClient(app) as client:
        original = app.state.bootstrap_state
        replacement_registry = ComponentRegistry()

        class RefreshOnlyComponent(Component):
            ports = Ports({}, {})

            def __call__(self, inputs, state, *, key):
                return {}, state

        replacement_registry.ingest(
            ComponentPack(
                namespace="example",
                components=(RefreshOnlyComponent,),
            )
        )
        replacement_registry.seal()
        replacement = BootstrapState(
            replace(original.bundle, components=replacement_registry), original.provenance
        )

        async def rebootstrap(**_kwargs):
            return replacement

        monkeypatch.setattr(components_api, "compose_application", rebootstrap)
        response = client.post("/api/components/refresh")

        assert response.status_code == 200
        assert app.state.bootstrap_state is replacement
        assert original.bundle.components.get("RefreshOnlyComponent") is None


def test_component_refresh_failure_preserves_published_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        original = app.state.bootstrap_state

        async def fail_rebootstrap(**_kwargs):
            raise RuntimeError("refresh failed")

        monkeypatch.setattr(components_api, "compose_application", fail_rebootstrap)
        response = client.post("/api/components/refresh")

        assert response.status_code == 500
        assert app.state.bootstrap_state is original


def test_studio_api_transport_models_declare_identity_and_reject_old_or_extra() -> None:
    with pytest.raises(ValidationError, match="literal_error"):
        SuccessPayload.model_validate(
            {
                "schema_id": STUDIO_API_TRANSPORT_SCHEMA_ID,
                "schema_version": "feedbax.spec.studio.api_transport.v0",
                "success": True,
            }
        )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SuccessPayload.model_validate({"success": True, "unexpected": True})

    family = default_spec_registry.resolve("StudioApiTransport")
    assert family.identity == STUDIO_API_TRANSPORT_SCHEMA_ID
    assert family.current_version == STUDIO_API_TRANSPORT_SCHEMA_VERSION
    with pytest.raises(UnsupportedSpecVersion, match="api_transport.v0"):
        default_spec_registry.migrate(
            "StudioApiTransport",
            {"schema_version": "feedbax.spec.studio.api_transport.v0"},
        )



def test_generated_studio_contracts_cover_route_response_models() -> None:
    app = create_app()
    generated_model_names = {model.__name__ for model in (*MODEL_TYPES, *AUTHORED_CONTRACT_MODELS)}
    generated_contract_names = set(CONTRACT_MODEL_NAMES)
    missing: list[str] = []

    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        if not route.path.startswith(GENERATED_STUDIO_PREFIXES):
            continue
        if route.response_model is None:
            continue
        for model in _response_model_members(route.response_model):
            model_name = model.__name__
            if model_name not in generated_model_names:
                missing.append(f"{route.path} response_model={model_name}")
            elif (
                (model_name.endswith(("Response", "Envelope", "Info")) or route.path.startswith("/api/projects"))
                and model_name not in generated_contract_names
            ):
                missing.append(f"{route.path} contractSchemas missing {model_name}")

    assert missing == []


def test_generated_studio_contracts_include_nested_model_dependencies() -> None:
    registered_models = set(MODEL_TYPES).union(AUTHORED_CONTRACT_MODELS)
    missing = {
        model.__name__: sorted(
            dependency.__name__
            for dependency in model_dependencies(model)
            if dependency not in registered_models
        )
        for model in MODEL_TYPES
    }

    assert {name: dependencies for name, dependencies in missing.items() if dependencies} == {}
    assert MODEL_TYPES.index(LrScheduleSpec) < MODEL_TYPES.index(OptimizerSpec)


def test_generated_studio_contracts_are_current() -> None:
    # OpenAPI can annotate recursive root references through endpoint fields.
    create_app().openapi()
    assert OUTPUT.read_text(encoding="utf-8") == generate()


def test_generated_array_value_contracts_preserve_strict_scalar_and_index_types() -> None:
    entry_interface = emit_interface(SparseCooEntrySpec)
    sparse_interface = emit_interface(SparseCooArrayValueSpec)
    constant_interface = emit_interface(ConstantArrayValueSpec)
    entry_schema = emit_schema(SparseCooEntrySpec)
    sparse_schema = emit_schema(SparseCooArrayValueSpec)
    constant_schema = emit_schema(ConstantArrayValueSpec)
    scalar_type = 'boolean | number | "nan" | "+inf" | "-inf"'
    scalar_schema = (
        "z.union([z.boolean(), z.number().int().safe(), z.number().finite(), "
        'z.union([z.literal("nan"), z.literal("+inf"), z.literal("-inf")])])'
    )

    assert "coordinate: number[];" in entry_interface
    assert "shape: number[];" in sparse_interface
    assert "shape: number[];" in constant_interface
    assert f"value: {scalar_type};" in entry_interface
    assert f"fill: {scalar_type};" in sparse_interface
    assert f"value: {scalar_type};" in constant_interface
    assert '"coordinate": z.array(z.number().int().safe().gte(0)).min(1)' in entry_schema
    assert '"shape": z.array(z.number().int().safe().gt(0)).min(1)' in sparse_schema
    assert '"shape": z.array(z.number().int().safe().gt(0)).min(1)' in constant_schema
    assert f'"value": {scalar_schema}' in entry_schema
    assert f'"fill": {scalar_schema}' in sparse_schema
    assert f'"value": {scalar_schema}' in constant_schema
    rendered = "\n".join(
        (
            entry_interface,
            sparse_interface,
            constant_interface,
            entry_schema,
            sparse_schema,
            constant_schema,
        )
    )
    assert "unknown[]" not in rendered
    assert " unknown |" not in rendered
    assert "z.unknown()" not in rendered


def test_generator_fails_for_unregistered_cross_field_constraint() -> None:
    class UnprojectableConstraint(BaseModel):
        left: int
        right: int

        @model_validator(mode="after")
        def validate_order(self) -> "UnprojectableConstraint":
            if self.left > self.right:
                raise ValueError("left must not exceed right")
            return self

    with pytest.raises(ValueError, match="Unprojectable material Pydantic model constraint"):
        emit_schema(UnprojectableConstraint)
