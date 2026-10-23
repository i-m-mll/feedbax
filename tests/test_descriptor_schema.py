from __future__ import annotations

import pytest
from pydantic import ValidationError

from feedbax.contracts.descriptors import (
    COMPONENT_DESCRIPTOR_SCHEMA_VERSION,
    COMPONENT_SELECTOR_SYNTAX_SCHEMA_VERSION,
    DESCRIPTOR_BASIS_SCHEMA_VERSION,
    SELECTOR_FALLBACK_POLICY_SCHEMA_VERSION,
    SELECTOR_ROLE_IDENTITY_SCHEMA_VERSION,
    VARIABLE_DESCRIPTOR_SCHEMA_VERSION,
    ComponentDescriptor,
    ComponentSelectorSyntax,
    ComponentSlice,
    DescriptorBasisIdentity,
    DescriptorResolutionError,
    SelectorFallbackPolicyIdentity,
    SelectorRoleIdentity,
    VariableDescriptor,
    descriptor_basis_from_descriptors,
    resolve_descriptor_view,
)
from feedbax.contracts.manifest import AnalysisDataProduct
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.contracts.value_schema import ValueSchema
from feedbax.integrations.provider import provider_manifest


def _value_schema(
    *,
    id: str,
    label: str,
    width: int | None = None,
    units: str = "a.u.",
    frame: str = "controller",
) -> ValueSchema:
    return ValueSchema(
        id=id,
        label=label,
        kind="vector" if width is not None else "scalar",
        dtype="float32",
        shape=[width] if width is not None else [],
        rank=1 if width is not None else 0,
        units=units,
        frame=frame,
        origin="declared",
    )


def _role() -> SelectorRoleIdentity:
    return SelectorRoleIdentity(namespace="rlrmp.feedback", name="controller_feedback")


def _selector_syntax() -> ComponentSelectorSyntax:
    return ComponentSelectorSyntax(namespace="feedbax.descriptor", name="descriptor_uri")


def _fallback_policy() -> SelectorFallbackPolicyIdentity:
    return SelectorFallbackPolicyIdentity(
        namespace="feedbax.selector.fallback",
        name="forbid",
        policy="forbid",
    )


def _variable(width: int = 4, *, scope: dict[str, str] | None = None) -> VariableDescriptor:
    return VariableDescriptor(
        descriptor_id="rlrmp.feedback.controller_visible",
        namespace="rlrmp.feedback",
        label="Controller-visible feedback",
        description="Feedback vector supplied by a downstream package.",
        value_schema=_value_schema(
            id="rlrmp.feedback.vector",
            label="Feedback vector",
            width=width,
        ),
        source_kind="model",
        source_path="states.net.input.feedback",
        timing="per_timestep",
        role=_role(),
        selector_syntax=_selector_syntax(),
        fallback_policy=_fallback_policy(),
        scope=(
            {"graph": "nominal_controller", "slot": "net.input"}
            if scope is None
            else scope
        ),
        tags=["feedback"],
        extensions={"rlrmp": {"basis_family": "controller_feedback"}},
    )


def _components(labels: list[tuple[str, str, str]], *, units: str = "a.u.") -> list[ComponentDescriptor]:
    return [
        ComponentDescriptor(
            descriptor_id=f"rlrmp.feedback.component.{component_id}",
            variable_descriptor_id="rlrmp.feedback.controller_visible",
            component_id=component_id,
            label=label,
            description=description,
            slice=ComponentSlice(start=index, stop=index + 1),
            value_schema=_value_schema(
                id=f"rlrmp.feedback.value.{component_id}",
                label=label,
                units=units,
            ),
            source_kind="model",
            source_path=f"states.net.input.feedback[{index}]",
            timing="per_timestep",
            sign="native",
            transform="target_relative",
            tags=["feedback_component"],
            extensions={"rlrmp": {"label_source": "fixture"}},
        )
        for index, (component_id, label, description) in enumerate(labels)
    ]


def _labels_4d() -> list[tuple[str, str, str]]:
    return [
        ("target_dx", "target dx", "Target-relative x position feedback."),
        ("target_dy", "target dy", "Target-relative y position feedback."),
        ("target_vx", "target vx", "Target-relative x velocity feedback."),
        ("target_vy", "target vy", "Target-relative y velocity feedback."),
    ]


def _labels_6d() -> list[tuple[str, str, str]]:
    return [
        *_labels_4d(),
        ("force_filter_x", "force filter x", "Augmented force/filter x feedback."),
        ("force_filter_y", "force filter y", "Augmented force/filter y feedback."),
    ]


def _basis_4d() -> DescriptorBasisIdentity:
    return descriptor_basis_from_descriptors(
        basis_id="rlrmp.feedback.target_relative_4d",
        variable=_variable(4),
        components=_components(_labels_4d()),
    )


def test_descriptor_records_round_trip_and_keep_value_schema_contract_layer() -> None:
    variable = _variable()
    components = _components(_labels_4d())
    basis = descriptor_basis_from_descriptors(
        basis_id="rlrmp.feedback.target_relative_4d",
        variable=variable,
        components=components,
    )

    round_tripped = DescriptorBasisIdentity.model_validate(basis.model_dump(mode="json"))

    assert variable.value_schema.__class__.__module__ == "feedbax.contracts.value_schema"
    assert round_tripped == basis
    assert round_tripped.descriptor_basis_hash == basis.descriptor_basis_hash
    assert round_tripped.components[0].component_id == "target_dx"
    assert round_tripped.components[0].value_schema.units == "a.u."
    assert variable.role is not None
    assert variable.role.identity == "rlrmp.feedback:controller_feedback:v1"
    assert variable.selector_syntax is not None
    assert variable.selector_syntax.identity == "feedbax.descriptor:descriptor_uri:v1"
    assert variable.fallback_policy is not None
    assert variable.fallback_policy.policy == "forbid"


def test_descriptor_basis_hash_is_stable_and_excludes_nonsemantic_metadata() -> None:
    variable = _variable()
    components = _components(_labels_4d())
    basis = descriptor_basis_from_descriptors(
        basis_id="rlrmp.feedback.target_relative_4d",
        variable=variable,
        components=components,
        metadata={"note": "not hashed"},
    )
    same_semantics = descriptor_basis_from_descriptors(
        basis_id="rlrmp.feedback.target_relative_4d",
        variable=variable,
        components=components,
        metadata={"note": "changed"},
    )

    assert basis.descriptor_basis_hash == same_semantics.descriptor_basis_hash
    assert basis.descriptor_basis_hash is not None
    assert basis.descriptor_basis_hash.startswith("sha256:")


def test_resolved_descriptor_view_requires_descriptor_id_and_matching_basis_hash() -> None:
    variable = _variable()
    components = _components(_labels_4d())
    basis = descriptor_basis_from_descriptors(
        basis_id="rlrmp.feedback.target_relative_4d",
        variable=variable,
        components=components,
    )

    resolved = resolve_descriptor_view(
        descriptor_id=components[2].descriptor_id,
        variable=variable,
        components=components,
        basis=basis,
        descriptor_basis_hash=basis.descriptor_basis_hash or "",
    )

    assert resolved.descriptor_kind == "component"
    assert resolved.component_descriptor == components[2]
    assert resolved.slice == ComponentSlice(start=2, stop=3)

    with pytest.raises(DescriptorResolutionError, match="descriptor basis mismatch"):
        resolve_descriptor_view(
            descriptor_id=components[2].descriptor_id,
            variable=variable,
            components=components,
            basis=basis,
            descriptor_basis_hash="sha256:stale",
        )

    with pytest.raises(DescriptorResolutionError, match="unknown descriptor_id"):
        resolve_descriptor_view(
            descriptor_id="rlrmp.feedback.raw_index.2",
            variable=variable,
            components=components,
            basis=basis,
            descriptor_basis_hash=basis.descriptor_basis_hash or "",
        )


def test_named_negative_width_mismatch_rejected() -> None:
    with pytest.raises(ValidationError, match="width mismatch"):
        descriptor_basis_from_descriptors(
            basis_id="rlrmp.feedback.bad_width",
            variable=_variable(6),
            components=_components(_labels_4d()),
        )


def test_named_negative_same_width_different_order_rejected_on_resolution() -> None:
    variable = _variable()
    components = _components(_labels_4d())
    expected = descriptor_basis_from_descriptors(
        basis_id="rlrmp.feedback.target_relative_4d",
        variable=variable,
        components=components,
    )
    reordered = descriptor_basis_from_descriptors(
        basis_id="rlrmp.feedback.target_relative_4d",
        variable=variable,
        components=[components[1], components[0], components[2], components[3]],
    )

    assert expected.descriptor_basis_hash != reordered.descriptor_basis_hash
    with pytest.raises(DescriptorResolutionError, match="descriptor basis mismatch"):
        resolve_descriptor_view(
            descriptor_id=components[0].descriptor_id,
            variable=variable,
            components=components,
            basis=reordered,
            descriptor_basis_hash=expected.descriptor_basis_hash or "",
        )


def test_named_negative_same_id_different_units_rejected() -> None:
    basis = _basis_4d()
    changed_units = _components(_labels_4d(), units="N")

    with pytest.raises(DescriptorResolutionError, match="value_schema"):
        resolve_descriptor_view(
            descriptor_id=changed_units[0].descriptor_id,
            variable=_variable(),
            components=changed_units,
            basis=basis,
            descriptor_basis_hash=basis.descriptor_basis_hash or "",
        )


def test_named_negative_same_id_different_timing_rejected() -> None:
    basis = _basis_4d()
    changed_timing = [
        component.model_copy(update={"timing": "post_run"})
        for component in _components(_labels_4d())
    ]

    with pytest.raises(DescriptorResolutionError, match="timing"):
        resolve_descriptor_view(
            descriptor_id=changed_timing[0].descriptor_id,
            variable=_variable(),
            components=changed_timing,
            basis=basis,
            descriptor_basis_hash=basis.descriptor_basis_hash or "",
        )


def test_named_negative_duplicate_component_ids_rejected() -> None:
    components = _components(_labels_4d())
    duplicate = components[1].model_copy(update={"component_id": components[0].component_id})

    with pytest.raises(ValidationError, match="duplicate component IDs"):
        descriptor_basis_from_descriptors(
            basis_id="rlrmp.feedback.duplicate",
            variable=_variable(),
            components=[components[0], duplicate, components[2], components[3]],
        )


def test_named_negative_missing_scope_rejected() -> None:
    with pytest.raises(ValidationError, match="scope must not be empty"):
        _variable(scope={})


def test_named_negative_stale_descriptor_version_rejected() -> None:
    with pytest.raises(ValidationError, match="unsupported VariableDescriptor schema_version"):
        VariableDescriptor.model_validate(
            {
                **_variable().model_dump(mode="json"),
                "schema_version": "feedbax.spec.descriptor.variable.v0",
            }
        )

    with pytest.raises(ValidationError, match="unsupported DescriptorBasisIdentity"):
        DescriptorBasisIdentity.model_validate(
            {
                **_basis_4d().model_dump(mode="json"),
                "schema_version": "feedbax.spec.descriptor.basis.v0",
            }
        )


def test_rlrmp_shaped_4d_and_6d_fixtures_are_external_payloads() -> None:
    basis_4d = _basis_4d()
    basis_6d = descriptor_basis_from_descriptors(
        basis_id="rlrmp.feedback.force_filter_augmented_6d",
        variable=_variable(6),
        components=_components(_labels_6d()),
    )

    assert len(basis_4d.components) == 4
    assert len(basis_6d.components) == 6
    assert basis_4d.descriptor_basis_hash != basis_6d.descriptor_basis_hash
    assert basis_6d.components[-1].component_id == "force_filter_y"


def test_descriptor_basis_hash_fills_data_product_contract_slot() -> None:
    basis = _basis_4d()
    product = AnalysisDataProduct.model_validate(
        {
            "product_schema_id": "rlrmp.controller_feedback_scales",
            "product_schema_version": "rlrmp.controller_feedback_scales.v1",
            "role": "controller_feedback_scales",
            "logical_name": "controller_feedback_scales",
            "producer_manifest_id": "analysis-run:selected",
            "descriptor_basis_hash": basis.descriptor_basis_hash,
        }
    )

    assert product.descriptor_basis_hash == basis.descriptor_basis_hash
    assert product.product_identity_hash is not None


def test_descriptor_families_are_registered_and_exported() -> None:
    manifest = provider_manifest()
    policies = default_spec_registry.policy_matrix()
    expected = {
        "VariableDescriptor",
        "ComponentDescriptor",
        "DescriptorBasisIdentity",
        "SelectorRoleIdentity",
        "ComponentSelectorSyntax",
        "SelectorFallbackPolicyIdentity",
    }

    assert expected <= set(manifest.schemas)
    for kind in expected:
        assert policies[kind].owner_module == "feedbax.contracts.descriptors"
        assert "tests/test_descriptor_schema.py" in policies[kind].required_tests

    current_versions = {
        "VariableDescriptor": VARIABLE_DESCRIPTOR_SCHEMA_VERSION,
        "ComponentDescriptor": COMPONENT_DESCRIPTOR_SCHEMA_VERSION,
        "DescriptorBasisIdentity": DESCRIPTOR_BASIS_SCHEMA_VERSION,
        "SelectorRoleIdentity": SELECTOR_ROLE_IDENTITY_SCHEMA_VERSION,
        "ComponentSelectorSyntax": COMPONENT_SELECTOR_SYNTAX_SCHEMA_VERSION,
        "SelectorFallbackPolicyIdentity": SELECTOR_FALLBACK_POLICY_SCHEMA_VERSION,
    }
    for kind, current_version in current_versions.items():
        accepted = default_spec_registry.migrate(kind, {"schema_version": current_version})
        assert accepted.target_version == current_version
        with pytest.raises(UnsupportedSpecVersion):
            default_spec_registry.migrate(kind, {"schema_version": current_version[:-1] + "0"})
