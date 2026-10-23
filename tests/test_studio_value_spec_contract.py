"""Tests for the durable Studio ValueSpec v2 contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from feedbax.contracts.graph import ComponentSpec, StudioValueSpec
from feedbax.contracts.migrations import default_spec_registry


def test_studio_value_spec_migrates_legacy_frontend_v1_literal_sweep() -> None:
    spec = StudioValueSpec.model_validate(
        {
            "schema_version": "feedbax.studio.value.v1",
            "mode": "constant",
            "value": [0.1, 0.2, 0.3],
            "metadata": {"indexed_constant": True},
        }
    )

    assert spec.schema_version == "feedbax.spec.studio.value.v2"
    assert spec.value_form == "literal"
    assert spec.mode == "constant"
    assert spec.variation.scope == "sweep"
    assert spec.variation.enumerable is not None
    assert spec.variation.enumerable.form == "list"
    assert spec.variation.enumerable.values == [0.1, 0.2, 0.3]


def test_studio_value_spec_migrates_legacy_distribution_replicate_semantics() -> None:
    spec = StudioValueSpec.model_validate(
        {
            "schema_version": "feedbax.spec.studio.value.v1",
            "mode": "distribution",
            "distribution": {"family": "normal", "parameters": {"mean": 0.0, "std": 1.0}},
            "sampling_scope": "replicate",
            "metadata": {},
        }
    )

    assert spec.value_form == "distribution"
    assert spec.variation.scope == "replicate"
    assert spec.variation.stochastic_policy == "resample_per_replicate"


def test_studio_value_spec_rejects_unknown_version_and_bad_sweep() -> None:
    with pytest.raises(ValidationError, match="unsupported StudioValueSpec schema_version"):
        StudioValueSpec.model_validate(
            {
                "schema_version": "feedbax.spec.studio.value.v0",
                "mode": "constant",
                "value": 1,
                "metadata": {},
            }
        )

    with pytest.raises(ValidationError, match="sweep variation requires an enumerable"):
        StudioValueSpec.model_validate(
            {
                "schema_version": "feedbax.spec.studio.value.v2",
                "value_form": "literal",
                "mode": "constant",
                "value": 1,
                "variation": {"scope": "sweep"},
                "metadata": {},
            }
        )


def test_component_params_normalize_typed_value_specs() -> None:
    component = ComponentSpec.model_validate(
        {
            "type": "feedbax.test.Component",
            "params": {
                "gain": {
                    "schema_version": "feedbax.spec.studio.value.v1",
                    "mode": "distribution",
                    "distribution": {"family": "uniform", "parameters": {"min": 0.0, "max": 1.0}},
                    "sampling_scope": "run",
                    "metadata": {},
                }
            },
            "input_ports": [],
            "output_ports": [],
        }
    )

    gain = component.params["gain"]
    assert isinstance(gain, dict)
    assert gain["schema_version"] == "feedbax.spec.studio.value.v2"
    assert gain["value_form"] == "distribution"
    assert gain["variation"]["scope"] == "run"
    assert gain["variation"]["stochastic_policy"] == "shared_per_run"


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (
            {
                "schema_version": "feedbax.spec.studio.value.v0",
                "mode": "constant",
                "value": 1,
                "metadata": {},
            },
            "unsupported StudioValueSpec schema_version",
        ),
        (
            {
                "schema_version": "feedbax.studio.value.v0",
                "mode": "constant",
                "value": 1,
                "metadata": {},
            },
            "unsupported StudioValueSpec schema_version",
        ),
        (
            {
                "schema_id": "feedbax.spec.studio.value",
                "schema_version": "third.party.value.v1",
                "mode": "constant",
                "value": 1,
                "metadata": {},
            },
            "unsupported StudioValueSpec schema_version",
        ),
    ],
)
def test_component_params_reject_unsupported_value_spec_like_payloads(
    payload: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValidationError, match=match):
        ComponentSpec.model_validate(
            {
                "type": "feedbax.test.Component",
                "params": {"gain": payload},
                "input_ports": [],
                "output_ports": [],
            }
        )


def test_value_spec_registry_migration_rejects_malformed_legacy_probe_payload() -> None:
    with pytest.raises(ValidationError, match="legacy StudioValueSpec requires mode"):
        default_spec_registry.migrate(
            "StudioValueSpec",
            {"schema_version": "feedbax.spec.studio.value.v1"},
        )


def test_value_spec_schema_registry_declares_v2_migration_policy() -> None:
    family = default_spec_registry.resolve("StudioValueSpec")

    assert family.current_version == "feedbax.spec.studio.value.v2"
    assert family.policy.stance == "migrate"
    assert "feedbax.spec.studio.value.v1" in family.policy.supported_old_versions
    assert "feedbax.studio.value.v1" in family.policy.supported_old_versions
