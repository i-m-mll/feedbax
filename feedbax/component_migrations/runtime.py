"""Frozen component migration records for runtime."""

from typing import Any, Mapping

from feedbax.component_registry.history import Change, History, LegacyEndpoint
from feedbax.contracts.migrations import UnsupportedComponentMigration
from feedbax.runtime.affine_composer import AffineValueComposer
from feedbax.runtime.channel import Channel
from feedbax.runtime.components import (
    GRU,
    LSTM,
    MLP,
    Activation,
    Constant,
    Damper,
    DeadZone,
    DelayLine,
    Demux,
    ElementwiseAffineModulator,
    Gain,
    Input,
    Linear,
    MatMul,
    Multiply,
    Mux,
    Noise,
    Pulse,
    Ramp,
    RateLimiter,
    Ravel,
    Reshape,
    Saturation,
    Scale,
    Sigmoid,
    Sine,
    Spring,
    Subtract,
    Sum,
    Switch,
)
from feedbax.runtime.feedback import AssembleFeedback, SelectState
from feedbax.runtime.filters import BandPassFilter, FirstOrderFilter, HighPassFilter
from feedbax.runtime.signal_transforms import (
    AddNoise,
    ConstantInput,
    NetworkClamp,
    NetworkConstantInput,
)
from feedbax.runtime.state_feedback import StateFeedbackSelector


def _linear_v1(params):
    return {**params, "initialization": "random"}


def _legacy_clamp(params):
    return "scale" in params


def _drop_clamp_scale(params):
    return {name: value for name, value in params.items() if name != "scale"}


def _legacy_addition(params):
    return set(params) <= {"scale", "active"}


AFFINE_VALUE_COMPOSER_SCHEMA_VERSION_V1 = "feedbax.component.affine_value_composer.v1"


AFFINE_VALUE_COMPOSER_SCHEMA_VERSION_V2 = "feedbax.component.affine_value_composer.v2"


AFFINE_VALUE_COMPOSER_SCHEMA_VERSION_V3 = "feedbax.component.affine_value_composer.v3"


def _affine_v1_evidence(params: Mapping[str, Any]) -> bool:
    return params.get("schema_version") == AFFINE_VALUE_COMPOSER_SCHEMA_VERSION_V1 or (
        "schema_version" not in params and "label" in params
    )


def _affine_v2_evidence(params: Mapping[str, Any]) -> bool:
    return params.get("schema_version") == AFFINE_VALUE_COMPOSER_SCHEMA_VERSION_V2


def _migrate_affine_v1(params: dict[str, Any]) -> dict[str, Any]:
    if "schema_version" in params and not _affine_v1_evidence(params):
        raise UnsupportedComponentMigration(
            f"Conflicting AffineValueComposer v1 marker: {params['schema_version']!r}"
        )
    return {
        **{name: value for name, value in params.items() if name != "label"},
        "schema_version": AFFINE_VALUE_COMPOSER_SCHEMA_VERSION_V2,
    }


def _migrate_affine_v2(params: dict[str, Any]) -> dict[str, Any]:
    if "schema_version" in params and not _affine_v2_evidence(params):
        raise UnsupportedComponentMigration(
            f"Conflicting AffineValueComposer v2 marker: {params['schema_version']!r}"
        )
    return {name: value for name, value in params.items() if name != "schema_version"}


HISTORIES = {
    Gain: History(
        legacy=(LegacyEndpoint("Gain", "1", 1), LegacyEndpoint("Gain", None, 1)),
    ),
    Input: History(
        legacy=(LegacyEndpoint("Input", "1", 1), LegacyEndpoint("Input", None, 1)),
    ),
    Sum: History(
        legacy=(LegacyEndpoint("Sum", "1", 1), LegacyEndpoint("Sum", None, 1)),
    ),
    Subtract: History(
        legacy=(LegacyEndpoint("Subtract", "1", 1), LegacyEndpoint("Subtract", None, 1)),
    ),
    Multiply: History(
        legacy=(LegacyEndpoint("Multiply", "1", 1), LegacyEndpoint("Multiply", None, 1)),
    ),
    Reshape: History(
        legacy=(LegacyEndpoint("Reshape", "1", 1), LegacyEndpoint("Reshape", None, 1)),
    ),
    MatMul: History(
        legacy=(LegacyEndpoint("MatMul", "1", 1), LegacyEndpoint("MatMul", None, 1)),
    ),
    Scale: History(
        legacy=(LegacyEndpoint("Scale", "1", 1), LegacyEndpoint("Scale", None, 1)),
    ),
    Sigmoid: History(
        legacy=(LegacyEndpoint("Sigmoid", "1", 1), LegacyEndpoint("Sigmoid", None, 1)),
    ),
    Activation: History(
        legacy=(LegacyEndpoint("Activation", "1", 1), LegacyEndpoint("Activation", None, 1)),
    ),
    ElementwiseAffineModulator: History(
        legacy=(
            LegacyEndpoint("ElementwiseAffineModulator", "1", 1),
            LegacyEndpoint("ElementwiseAffineModulator", None, 1),
        ),
    ),
    Saturation: History(
        legacy=(LegacyEndpoint("Saturation", "1", 1), LegacyEndpoint("Saturation", None, 1)),
    ),
    Constant: History(
        legacy=(LegacyEndpoint("Constant", "1", 1), LegacyEndpoint("Constant", None, 1)),
    ),
    Ravel: History(
        legacy=(LegacyEndpoint("Ravel", "1", 1), LegacyEndpoint("Ravel", None, 1)),
    ),
    Ramp: History(
        legacy=(LegacyEndpoint("Ramp", "1", 1), LegacyEndpoint("Ramp", None, 1)),
    ),
    Sine: History(
        legacy=(LegacyEndpoint("Sine", "1", 1), LegacyEndpoint("Sine", None, 1)),
    ),
    Pulse: History(
        legacy=(LegacyEndpoint("Pulse", "1", 1), LegacyEndpoint("Pulse", None, 1)),
    ),
    Noise: History(
        legacy=(LegacyEndpoint("Noise", "1", 1), LegacyEndpoint("Noise", None, 1)),
    ),
    Spring: History(
        legacy=(LegacyEndpoint("Spring", "1", 1), LegacyEndpoint("Spring", None, 1)),
    ),
    Damper: History(
        legacy=(LegacyEndpoint("Damper", "1", 1), LegacyEndpoint("Damper", None, 1)),
    ),
    DelayLine: History(
        Change(2, identity=True),
        legacy=(
            LegacyEndpoint("DelayLine", "1", 1),
            LegacyEndpoint("DelayLine", "2", 2),
        ),
    ),
    Linear: History(
        Change(2, migrate=_linear_v1),
        legacy=(
            LegacyEndpoint("Linear", "1", 1),
            LegacyEndpoint("Linear", "2", 2),
        ),
    ),
    MLP: History(
        legacy=(LegacyEndpoint("MLP", "1", 1), LegacyEndpoint("MLP", None, 1)),
    ),
    GRU: History(
        legacy=(LegacyEndpoint("GRU", "1", 1), LegacyEndpoint("GRU", None, 1)),
    ),
    LSTM: History(
        legacy=(LegacyEndpoint("LSTM", "1", 1), LegacyEndpoint("LSTM", None, 1)),
    ),
    Mux: History(
        legacy=(LegacyEndpoint("Mux", "1", 1), LegacyEndpoint("Mux", None, 1)),
    ),
    Demux: History(
        legacy=(LegacyEndpoint("Demux", "1", 1), LegacyEndpoint("Demux", None, 1)),
    ),
    Switch: History(
        legacy=(LegacyEndpoint("Switch", "1", 1), LegacyEndpoint("Switch", None, 1)),
    ),
    DeadZone: History(
        legacy=(LegacyEndpoint("DeadZone", "1", 1), LegacyEndpoint("DeadZone", None, 1)),
    ),
    RateLimiter: History(
        legacy=(LegacyEndpoint("RateLimiter", "1", 1), LegacyEndpoint("RateLimiter", None, 1)),
    ),
    Channel: History(
        Change(2, identity=True),
        legacy=(
            LegacyEndpoint("Channel", "1", 1),
            LegacyEndpoint("Channel", "2", 2),
        ),
    ),
    FirstOrderFilter: History(
        Change(2, identity=True),
        legacy=(
            LegacyEndpoint("FirstOrderFilter", "1", 1),
            LegacyEndpoint("FirstOrderFilter", "2", 2),
        ),
    ),
    HighPassFilter: History(
        legacy=(
            LegacyEndpoint("HighPassFilter", "1", 1),
            LegacyEndpoint("HighPassFilter", None, 1),
        ),
    ),
    BandPassFilter: History(
        legacy=(
            LegacyEndpoint("BandPassFilter", "1", 1),
            LegacyEndpoint("BandPassFilter", None, 1),
        ),
    ),
    SelectState: History(
        legacy=(LegacyEndpoint("SelectState", "1", 1), LegacyEndpoint("SelectState", None, 1)),
    ),
    AssembleFeedback: History(
        legacy=(
            LegacyEndpoint("AssembleFeedback", "1", 1),
            LegacyEndpoint("AssembleFeedback", None, 1),
        ),
    ),
    StateFeedbackSelector: History(
        legacy=(
            LegacyEndpoint("StateFeedbackSelector", "1", 1),
            LegacyEndpoint("StateFeedbackSelector", None, 1),
        ),
    ),
    AddNoise: History(
        legacy=(LegacyEndpoint("AddNoise", "1", 1), LegacyEndpoint("AddNoise", None, 1)),
    ),
    NetworkClamp: History(
        Change(2, migrate=_drop_clamp_scale),
        legacy=(
            LegacyEndpoint("NetworkClamp", "1", 1, matches_unversioned=_legacy_clamp),
            LegacyEndpoint("NetworkClamp", "2", 2),
        ),
    ),
    NetworkConstantInput: History(
        Change(2, identity=True),
        legacy=(
            LegacyEndpoint("NetworkConstantInput", "1", 1, matches_unversioned=_legacy_addition),
            LegacyEndpoint("NetworkConstantInput", "2", 2),
        ),
    ),
    ConstantInput: History(
        Change(2, identity=True),
        legacy=(
            LegacyEndpoint("ConstantInput", "1", 1, matches_unversioned=_legacy_addition),
            LegacyEndpoint("ConstantInput", "2", 2),
        ),
    ),
    AffineValueComposer: History(
        Change(2, migrate=_migrate_affine_v1),
        Change(3, migrate=_migrate_affine_v2),
        legacy=(
            LegacyEndpoint(
                "AffineValueComposer",
                "feedbax.component.affine_value_composer.v1",
                1,
                matches_unversioned=_affine_v1_evidence,
            ),
            LegacyEndpoint(
                "AffineValueComposer",
                "feedbax.component.affine_value_composer.v2",
                2,
                matches_unversioned=_affine_v2_evidence,
            ),
            LegacyEndpoint("AffineValueComposer", "feedbax.component.affine_value_composer.v3", 3),
        ),
    ),
}
