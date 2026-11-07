"""Frozen linear mechanics wire endpoints and label migration."""

from feedbax.component_registry.history import Change, History, LegacyEndpoint
from .linear_state_space import LinearStateSpace, StructuralLinearStateSpace

STRUCTURAL_LINEAR_STATE_SPACE_PARAM_SCHEMA_VERSION_V1 = (
    "feedbax.component.structural_linear_state_space.v1"
)
STRUCTURAL_LINEAR_STATE_SPACE_PARAM_SCHEMA_VERSION = (
    "feedbax.component.structural_linear_state_space.v2"
)


def _drop_structural_label(params):
    return {name: value for name, value in params.items() if name != "label"}


def _materialize_structural_v2_defaults(params):
    return {"dt": 0.01, **params}


def _materialize_linear_v1_defaults(params):
    return {"dt": 0.01, **params}


def _has_legacy_label(params):
    return "label" in params


def _linear_v1_evidence(params):
    return set(params) <= {"A", "B", "B_w", "dt", "initial_state", "pos_slice", "vel_slice"}


def _structural_v2_evidence(params):
    return set(params) <= {
        "A",
        "B",
        "B_w",
        "dt",
        "initial_state",
        "pos_slice",
        "vel_slice",
        "delta_A",
        "scale",
        "active",
    }


HISTORIES = {
    LinearStateSpace: History(
        Change(2, migrate=_materialize_linear_v1_defaults),
        legacy=(LegacyEndpoint("LinearStateSpace", "1", 1, _linear_v1_evidence),),
    ),
    StructuralLinearStateSpace: History(
        Change(2, migrate=_drop_structural_label),
        Change(3, migrate=_materialize_structural_v2_defaults),
        legacy=(
            LegacyEndpoint(
                "StructuralLinearStateSpace",
                STRUCTURAL_LINEAR_STATE_SPACE_PARAM_SCHEMA_VERSION_V1,
                1,
                _has_legacy_label,
            ),
            LegacyEndpoint(
                "StructuralLinearStateSpace",
                STRUCTURAL_LINEAR_STATE_SPACE_PARAM_SCHEMA_VERSION,
                2,
                _structural_v2_evidence,
            ),
        ),
    ),
}
