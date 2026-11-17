"""Curated representations for the built-in portable owners."""

from feedbax.acausal.multibody import MusclePath, PlanarLink, PointMarker
from feedbax.compiler.muscle_templates import Arm6MuscleRigidTendon, PointMass8MuscleRelu
from feedbax.mechanics.analytical_component import AnalyticalMusculoskeletalPlant
from feedbax.mechanics.direct_force_components import PointMass, TwoLinkArm
from feedbax.mechanics.muscle_config import default_6muscle_2link_segment_lengths

from .acausal_representation import (
    _muscle_path_representation,
    _planar_link_representation,
    _point_marker_representation,
)
from .component_presentation import PresentationOverride
from .direct_force_presentation import _point_mass_representation, _two_link_arm_representation
from .muscle_representation import (
    _default_two_link_muscle_path_geometry,
    _point_mass_muscle_representation,
    _two_link_muscle_representation,
)


def builtin_component_presentation() -> dict[type, PresentationOverride]:
    return {
        PointMass: PresentationOverride(representation=_point_mass_representation()),
        TwoLinkArm: PresentationOverride(representation=_two_link_arm_representation()),
        AnalyticalMusculoskeletalPlant: PresentationOverride(
            representation=_two_link_muscle_representation(
                source="feedbax.mechanics.muscle_config.default_6muscle_2link_muscled_arm_parameters",
                composite=False,
                muscle_path_geometry=_default_two_link_muscle_path_geometry(),
                frame_element_id="links",
                self_link_lengths=[float(x) for x in default_6muscle_2link_segment_lengths()],
            )
        ),
        PlanarLink: PresentationOverride(representation=_planar_link_representation()),
        MusclePath: PresentationOverride(representation=_muscle_path_representation()),
        PointMarker: PresentationOverride(representation=_point_marker_representation()),
        Arm6MuscleRigidTendon: PresentationOverride(
            representation=_two_link_muscle_representation(
                source="feedbax.mechanics.geometry.TwoLinkArmMuscleGeometry.default_six_muscle",
                composite=True,
                muscle_path_geometry=_default_two_link_muscle_path_geometry(),
                frame_input_port="angles",
            )
        ),
        PointMass8MuscleRelu: PresentationOverride(
            representation=_point_mass_muscle_representation()
        ),
    }
