"""External-style coverage for typed extension-family registration."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from feedbax.analysis.evaluation_compaction import EvaluationBatchFragment
from feedbax.contracts.training import (
    TrainingMethodAuthoringContribution,
    TrainingMethodAuthoringHook,
    standard_supervised_method_descriptor,
)
from feedbax.plugins import (
    EVALUATION_PRODUCT_UNION_FINALIZERS,
    ROW_LOWERERS,
    TRAINING_METHODS,
    BootstrapError,
    BootstrapErrorCode,
    FamilyRequirement,
    PluginDeclaration,
    PluginRegistration,
    bootstrap_application,
    new_registration_context,
)
from feedbax.training import training_method_row_lowerer_registration


def _authoring_contribution(_payload: object) -> TrainingMethodAuthoringContribution:
    return TrainingMethodAuthoringContribution()


def _projection(_payload: object) -> dict[str, object]:
    return {}


def _descriptor_plugin() -> PluginRegistration:
    descriptor = replace(
        standard_supervised_method_descriptor(),
        method_ref="tests/external_authoring/v1",
        payload_schema_id="tests.spec.external_authoring",
        payload_schema_version="tests.spec.external_authoring.v1",
        owner="tests.external",
        package="tests",
        authoring_hook=TrainingMethodAuthoringHook(
            lowerer_id="tests.external_authoring",
            lowerer_version="tests.external_authoring.v1",
            compile=_authoring_contribution,
            graph=_projection,
            task=_projection,
            objective=_projection,
            domain=_projection,
        ),
    )

    def register(context) -> None:
        methods = context.registry(TRAINING_METHODS)
        methods.register_descriptor(descriptor)
        context.registry(ROW_LOWERERS).register(
            training_method_row_lowerer_registration(descriptor, methods)
        )

    return PluginRegistration(
        PluginDeclaration(
            "tests.external_descriptor",
            "1",
            1,
            families=(
                FamilyRequirement(TRAINING_METHODS.family),
                FamilyRequirement(ROW_LOWERERS.family),
            ),
        ),
        register,
    )


def test_external_plugin_uses_public_descriptor_row_lowerer_helper() -> None:
    state = asyncio.run(
        bootstrap_application(
            new_registration_context(local_component_source=None),
            registrations=(_descriptor_plugin(),),
        )
    )

    assert state.registry(TRAINING_METHODS).descriptor("tests/external_authoring/v1") is not None
    assert state.registry(ROW_LOWERERS).available_keys() == (
        (
            "tests.spec.external_authoring",
            "tests.spec.external_authoring.v1",
            "tests.external_authoring",
            "tests.external_authoring.v1",
        ),
    )


def _finalizer_plugin(plugin_id: str) -> PluginRegistration:
    def finalize(_value) -> EvaluationBatchFragment:
        return EvaluationBatchFragment(
            payload={},
            schema_id="tests.spec.union",
            schema_version="tests.spec.union.v1",
            role="union",
        )

    return PluginRegistration(
        PluginDeclaration(
            plugin_id,
            "1",
            1,
            families=(FamilyRequirement(EVALUATION_PRODUCT_UNION_FINALIZERS.family),),
        ),
        lambda context: context.registry(EVALUATION_PRODUCT_UNION_FINALIZERS).register(
            "tests.union",
            "v1",
            finalize,
        ),
    )


def test_finalizer_family_is_isolated_and_collision_is_typed() -> None:
    populated = asyncio.run(
        bootstrap_application(
            new_registration_context(local_component_source=None),
            registrations=(_finalizer_plugin("tests.union.one"),),
        )
    )
    isolated = asyncio.run(
        bootstrap_application(new_registration_context(local_component_source=None))
    )

    assert populated.registry(EVALUATION_PRODUCT_UNION_FINALIZERS).keys() == ("tests.union@v1",)
    assert isolated.registry(EVALUATION_PRODUCT_UNION_FINALIZERS).keys() == ()
    assert populated.provenance[0].registered_keys == {
        EVALUATION_PRODUCT_UNION_FINALIZERS.family: ("tests.union@v1",)
    }
    with pytest.raises(RuntimeError, match="registry is sealed"):
        populated.registry(EVALUATION_PRODUCT_UNION_FINALIZERS).register(
            "tests.late",
            "v1",
            lambda _value: None,  # type: ignore[arg-type]
        )
    with pytest.raises(BootstrapError) as caught:
        asyncio.run(
            bootstrap_application(
                new_registration_context(local_component_source=None),
                registrations=(
                    _finalizer_plugin("tests.union.one"),
                    _finalizer_plugin("tests.union.two"),
                ),
            )
        )
    assert caught.value.code is BootstrapErrorCode.NAMESPACE_COLLISION
