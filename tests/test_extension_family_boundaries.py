"""External-style coverage for typed extension-family registration."""

from __future__ import annotations

import asyncio

import pytest

from feedbax.analysis.evaluation_compaction import EvaluationBatchFragment
from feedbax.plugins import (
    EVALUATION_PRODUCT_UNION_FINALIZERS,
    BootstrapError,
    BootstrapErrorCode,
    FamilyRequirement,
    PluginDeclaration,
    PluginRegistration,
    bootstrap_application,
    new_registration_context,
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
            2,
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
            new_registration_context(),
            registrations=(_finalizer_plugin("tests.union.one"),),
        )
    )
    isolated = asyncio.run(
        bootstrap_application(new_registration_context())
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
                new_registration_context(),
                registrations=(
                    _finalizer_plugin("tests.union.one"),
                    _finalizer_plugin("tests.union.two"),
                ),
            )
        )
    assert caught.value.code is BootstrapErrorCode.NAMESPACE_COLLISION
