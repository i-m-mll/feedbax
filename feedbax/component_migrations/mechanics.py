"""Frozen component migration records for mechanics."""

from collections.abc import Mapping
from typing import Any

from feedbax.component_registry.history import Change, History, LegacyEndpoint
from feedbax.mechanics.analytical_component import AnalyticalMusculoskeletalPlant
from feedbax.mechanics.direct_force_components import PointMass, TwoLinkArm
from feedbax.mechanics.force_fields import (
    CurlField,
    DynamicsMatrixPerturb,
    FixedField,
    ThresholdLatchedForce,
)
from feedbax.mechanics.linear_state_space import LinearStateSpace, StructuralLinearStateSpace
from feedbax.mechanics.muscle_graph import JointMuscleGeometry, RadialMuscleGeometry, VectorElements
from feedbax.mechanics.muscles.relu_muscle import ReluMuscle
from feedbax.mechanics.muscles.thelen_muscle import RigidTendonHillMuscleThelen

THRESHOLD_LATCHED_FORCE_SCHEMA_VERSION_V1 = "feedbax.component.threshold_latched_force.v1"


THRESHOLD_LATCHED_FORCE_SCHEMA_VERSION_V2 = "feedbax.component.threshold_latched_force.v2"


THRESHOLD_LATCHED_FORCE_SCHEMA_VERSION = "feedbax.component.threshold_latched_force.v3"


def _threshold_v1_evidence(params: Mapping[str, Any]) -> bool:
    selector = params.get("state_selector")
    return (
        isinstance(selector, Mapping)
        and "path" in selector
        and "kind" not in selector
        and "lateral_force" not in params
    )


def _threshold_v2_evidence(params: Mapping[str, Any]) -> bool:
    selector = params.get("state_selector")
    return isinstance(selector, Mapping) and "kind" in selector and "label" in params


def _migrate_threshold_latched_force_v1(params: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(params)
    selector = dict(migrated["state_selector"])
    selector["kind"] = "fixed"
    migrated["state_selector"] = selector
    migrated["lateral_force"] = 0.0
    return migrated


def _migrate_threshold_latched_force_v2(params: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in params.items() if name != "label"}


def _has_legacy_label(params):
    return "label" in params


def _drop_label(params):
    return {name: value for name, value in params.items() if name != "label"}


def _threshold_v3_evidence(params):
    selector = params.get("state_selector")
    return (
        isinstance(selector, Mapping)
        and "kind" in selector
        and "lateral_force" in params
        and "label" not in params
    )


def _curl_v2_evidence(params):
    return set(params) <= {"scale", "active", "amplitude"}


def _fixed_v2_evidence(params):
    return set(params) <= {"scale", "amplitude", "field", "active"}


def _dynamics_v2_evidence(params):
    return set(params) <= {"scale", "delta_A", "active", "mass"}


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


def _matches_ReluMuscle_v1(params):
    return set(params) <= {
        "dt",
        "tau_deactivation",
        "tau_activation",
        "min_activation",
        "max_isometric_force",
        "initial_activation",
    }


def _materialize_v1_defaults(params):
    return {"n_muscles": 6, "tendon_slack_length": 0.2, **params}


def _matches_RigidTendonHillMuscleThelen_v1(params):
    return set(params) <= {
        "dt",
        "tau_deactivation",
        "tau_activation",
        "n_muscles",
        "max_isometric_force",
        "optimal_muscle_length",
        "initial_activation",
        "tendon_slack_length",
    }


def _matches_VectorElements_v1(params):
    return set(params) <= {"size"}


def _matches_JointMuscleGeometry_v1(params):
    return set(params) <= {"paths"}


def _matches_RadialMuscleGeometry_v1(params):
    return set(params) <= {"n_pairs"}


def _matches_PointMass_v1(params):
    return set(params) <= {"dt", "damping", "mass"}


def _matches_TwoLinkArm_v1(params):
    return set(params) <= {"link_lengths", "dt"}


def _matches_AnalyticalMusculoskeletalPlant_v1(params):
    return set(params) <= {"tau_act", "dt", "tau_deact", "clip_states", "n_steps", "solver_type"}


HISTORIES = {
    FixedField: History(
        Change(2, migrate=_drop_label),
        legacy=(
            LegacyEndpoint("FixedField", "1", 1, _has_legacy_label),
            LegacyEndpoint("FixedField", "2", 2, _fixed_v2_evidence),
        ),
    ),
    DynamicsMatrixPerturb: History(
        Change(2, migrate=_drop_label),
        legacy=(
            LegacyEndpoint("DynamicsMatrixPerturb", "1", 1, _has_legacy_label),
            LegacyEndpoint("DynamicsMatrixPerturb", "2", 2, _dynamics_v2_evidence),
        ),
    ),
    CurlField: History(
        Change(2, migrate=_drop_label),
        legacy=(
            LegacyEndpoint("CurlField", "1", 1, _has_legacy_label),
            LegacyEndpoint("CurlField", "2", 2, _curl_v2_evidence),
        ),
    ),
    ThresholdLatchedForce: History(
        Change(2, migrate=_migrate_threshold_latched_force_v1),
        Change(3, migrate=_migrate_threshold_latched_force_v2),
        legacy=(
            LegacyEndpoint(
                "ThresholdLatchedForce",
                THRESHOLD_LATCHED_FORCE_SCHEMA_VERSION_V1,
                1,
                _threshold_v1_evidence,
            ),
            LegacyEndpoint(
                "ThresholdLatchedForce",
                THRESHOLD_LATCHED_FORCE_SCHEMA_VERSION_V2,
                2,
                _threshold_v2_evidence,
            ),
            LegacyEndpoint("ThresholdLatchedForce", THRESHOLD_LATCHED_FORCE_SCHEMA_VERSION, 3),
        ),
    ),
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
    ReluMuscle: History(legacy=(LegacyEndpoint("ReluMuscle", "1", 1, _matches_ReluMuscle_v1),)),
    RigidTendonHillMuscleThelen: History(
        Change(2, migrate=_materialize_v1_defaults),
        legacy=(
            LegacyEndpoint(
                "RigidTendonHillMuscleThelen", "1", 1, _matches_RigidTendonHillMuscleThelen_v1
            ),
        ),
    ),
    VectorElements: History(
        legacy=(LegacyEndpoint("VectorElements", "1", 1, _matches_VectorElements_v1),)
    ),
    JointMuscleGeometry: History(
        legacy=(LegacyEndpoint("JointMuscleGeometry", "1", 1, _matches_JointMuscleGeometry_v1),)
    ),
    RadialMuscleGeometry: History(
        legacy=(LegacyEndpoint("RadialMuscleGeometry", "1", 1, _matches_RadialMuscleGeometry_v1),)
    ),
    PointMass: History(legacy=(LegacyEndpoint("PointMass", "1", 1, _matches_PointMass_v1),)),
    TwoLinkArm: History(legacy=(LegacyEndpoint("TwoLinkArm", "1", 1, _matches_TwoLinkArm_v1),)),
    AnalyticalMusculoskeletalPlant: History(
        legacy=(
            LegacyEndpoint(
                "AnalyticalMusculoskeletalPlant", "1", 1, _matches_AnalyticalMusculoskeletalPlant_v1
            ),
        )
    ),
}
