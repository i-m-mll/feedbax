from __future__ import annotations

from collections import Counter
import re

from tests.support.component_owners import register_owners
from feedbax.contracts.representation import RepresentationSpec

from feedbax.builtins.runtime.components import Gain
from feedbax.studio.component_presentation import PresentationOverride

from dataclasses import replace
from typing import get_args, get_origin

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError, model_validator

from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.array_values import (
    ConstantArrayValueSpec,
    SparseCooArrayValueSpec,
    SparseCooEntrySpec,
)
from feedbax.contracts.studio_api import (
    ComponentListResponse,
    STUDIO_API_TRANSPORT_SCHEMA_ID,
    STUDIO_API_TRANSPORT_SCHEMA_VERSION,
    SuccessPayload,
)
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.contracts.representation import REPRESENTATION_SCHEMA_VERSION
from feedbax.web.app import create_app
from feedbax.application.bundle import BootstrapState
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


GENERATED_STUDIO_PREFIXES = (
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


def test_component_api_serves_representation_contract() -> None:
    class ApiRepresentedGain(Gain):
        """API represented component fixture."""

    registry = ComponentRegistry()
    ApiRepresentedGain.component_presentation = PresentationOverride(
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
        ),
    )
    register_owners(registry, "example", ApiRepresentedGain)
    app = create_app()
    with TestClient(app) as client:
        registry.seal()
        current = app.state.bootstrap_state
        app.state.bootstrap_state = BootstrapState(
            replace(current.bundle, components=registry)
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
    rendered = generate()
    types = re.findall(r"^export (?:interface|type) (\w+)\b", rendered, re.MULTILINE)
    validators = re.findall(r"^export const (\w+)Schema:", rendered, re.MULTILINE)
    type_counts = Counter(types)
    validator_counts = Counter(validators)
    referenced_models = {
        dependency.__name__
        for model in MODEL_TYPES
        for dependency in model_dependencies(model)
    }

    assert {
        name: (type_counts[name], validator_counts[name])
        for name in sorted(referenced_models)
        if (type_counts[name], validator_counts[name]) != (1, 1)
    } == {}


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
