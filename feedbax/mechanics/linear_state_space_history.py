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


HISTORIES = {
    LinearStateSpace: History(
        Change(2, migrate=_materialize_linear_v1_defaults),
        legacy=(LegacyEndpoint("LinearStateSpace", "1", 1),),
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
                "StructuralLinearStateSpace", STRUCTURAL_LINEAR_STATE_SPACE_PARAM_SCHEMA_VERSION, 2
            ),
        ),
    ),
}
