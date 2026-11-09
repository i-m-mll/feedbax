"""Native mechanics acceptance for durable MuJoCo model authoring."""

from __future__ import annotations

import numpy as np
import pytest

from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.mechanics.authored_model import compile_planar_model, compile_spatial_model
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider


mujoco = pytest.importorskip("mujoco")
pytestmark = pytest.mark.optional_mjx


def test_saved_planar_inertials_and_joint_frames_change_native_dynamics():
    def graph(mass=2.0, com=0.5, inertia=0.2, damping=0.1):
        return AcausalGraphSpec.model_validate({
            "physical_domain": "planar_multibody",
            "solver": {"solver_type": "euler", "dt": 0.002},
            "nodes": {
                "world": {"type": "feedbax.WorldFrame"},
                "anchor": {"type": "feedbax.Anchor", "params": {"frame": "a.proximal"}},
                "a": {"type": "feedbax.PlanarLink", "params": {"length": 1}},
                "b": {
                    "type": "feedbax.PlanarLink",
                    "params": {
                        "length": 0.8, "mass": mass, "com": com, "com_y": 0.1, "inertia": inertia,
                    },
                },
                "c": {"type": "feedbax.PlanarLink", "params": {"length": 0.4}},
                "shoulder": {
                    "type": "feedbax.RevoluteJoint",
                    "params": {"parent_frame": "world.frame", "child_frame": "a.proximal"},
                },
                "elbow": {
                    "type": "feedbax.RevoluteJoint",
                    "params": {
                        "parent_frame": "a.distal", "child_frame": "b.proximal",
                        "frame_offset": [0, 0.2], "frame_angle": np.pi / 2,
                        "damping": damping, "stiffness": 0.7, "rest_angle": 0.1,
                    },
                },
                "wrist": {
                    "type": "feedbax.RevoluteJoint",
                    "params": {"parent_frame": "b.distal", "child_frame": "c.proximal"},
                },
                "marker": {
                    "type": "feedbax.PointMarker",
                    "params": {"frame": "c.distal", "marker_name": "effector"},
                },
            },
        })

    def native_reference(mass, com, inertia, damping):
        return mujoco.MjModel.from_xml_string(f"""
          <mujoco>
            <compiler angle="radian"/>
            <option timestep=".002" gravity="0 -9.81 0"/>
            <worldbody>
              <body name="a">
                <inertial pos=".5 0 0" mass="1" diaginertia=".015 .015 .03"/>
                <joint name="shoulder" axis="0 0 1"/>
                <body name="b" pos="1 .2 0" euler="0 0 {np.pi / 2}">
                  <inertial pos="{0.8 * com} .1 0" mass="{mass}"
                            diaginertia="{inertia / 2} {inertia / 2} {inertia}"/>
                  <joint name="elbow" axis="0 0 1" damping="{damping}"
                         stiffness=".7" springref=".1"/>
                  <body name="c" pos=".8 0 0">
                    <inertial pos=".2 0 0" mass="1" diaginertia=".015 .015 .03"/>
                    <joint name="wrist" axis="0 0 1"/>
                    <site name="effector" pos=".4 0 0"/>
                  </body>
                </body>
              </body>
            </worldbody>
          </mujoco>
        """)

    def evaluate(model):
        data = mujoco.MjData(model)
        data.qpos[:] = [0.2, -0.4, 0.1]
        data.qvel[:] = [0.5, 0.3, -0.1]
        mujoco.mj_forward(model, data)
        return data.qacc.copy(), data.site_xpos.copy()

    variants = [(2.0, 0.5, 0.2, 0.1), (3.0, 0.5, 0.2, 0.1),
                (2.0, 0.3, 0.4, 0.1), (2.0, 0.5, 0.2, 0.9)]
    baseline = None
    for variant in variants:
        saved = AcausalGraphSpec.model_validate_json(graph(*variant).model_dump_json())
        descriptor = compile_planar_model(saved)
        actual = evaluate(descriptor.model)
        expected = evaluate(native_reference(*variant))
        np.testing.assert_allclose(actual[0], expected[0], rtol=1e-11)
        np.testing.assert_allclose(actual[1], expected[1], atol=1e-12)
        if baseline is None:
            baseline = descriptor, actual[0]
        else:
            previous, reference_acceleration = baseline
            assert descriptor.identity != previous.identity
            assert descriptor.structural_signature == previous.structural_signature
            assert not np.allclose(actual[0], reference_acceleration)
    assert descriptor.dimension == 2

    disconnected = graph()
    disconnected.nodes["wrist"].params["parent_frame"] = "a.distal"
    with pytest.raises(ValueError, match="Branching"):
        compile_planar_model(disconnected)


def test_saved_spatial_asset_and_overlay_bindings(tmp_path):
    provider = ImmutableArtifactBlobProvider(tmp_path / "content")
    asset = provider.store_object(b"""
      <mujoco>
        <worldbody>
          <site name="fixed" pos="0 0 1"/>
          <body name="arm">
            <joint name="pivot" type="ball"/>
            <inertial pos=".1 0 0" mass="2" diaginertia=".2 .3 .4"/>
            <site name="effector" pos=".3 .1 .2"/>
            <geom name="wrapper" type="sphere" size=".04" pos=".1 0 .12"
                  mass="0" contype="0" conaffinity="0"/>
          </body>
        </worldbody>
        <actuator>
          <general name="drive" joint="pivot" gear="0 1 0 0 0 0"
                   dyntype="filter" dynprm=".05"/>
        </actuator>
      </mujoco>
    """)
    graph = AcausalGraphSpec.model_validate({
        "physical_domain": "spatial_multibody",
        "solver": {"solver_type": "euler", "dt": 0.002},
        "mujoco_asset": {
            "asset": asset.model_dump(),
            "effector_site": "effector",
            "sites": {
                "insertion": {"body": "arm", "position": [0.1, 0, 0.25]},
                "side": {"body": "arm", "position": [0.18, 0, 0.12]},
            },
            "paths": {"flexor": {
                "sites": ["fixed", "insertion"],
                "wraps": [{"after_site": 0, "geom": "wrapper", "side_site": "side"}],
                "law": "feedbax.RigidTendonHillMuscle",
                "state_size": 1,
                "parameters": {"max_isometric_force": 150.0},
            }},
        },
    })
    first = compile_spatial_model(graph, provider)
    graph.mujoco_asset.sites["insertion"].position = (0.1, 0, 0.35)
    restored = AcausalGraphSpec.model_validate_json(graph.model_dump_json())
    edited = compile_spatial_model(restored, provider)

    assert edited.dimension == 3
    assert edited.asset_identity == first.asset_identity == asset.object_id
    assert edited.identity != first.identity
    assert edited.structural_signature == first.structural_signature
    assert (edited.model.nq, edited.model.nv, edited.model.na) == (4, 3, 1)
    assert edited.model.body_mass[edited.body_ids["arm"]] == 2.0
    assert edited.joints[edited.joint_ids["pivot"]].qpos == slice(0, 4)
    assert edited.actuator_ids["drive"] == 0
    assert edited.paths[edited.tendon_ids["flexor"]].parameters["max_isometric_force"] == 150.0
    wrap = edited.model.tendon_adr[edited.tendon_ids["flexor"]] + 1
    assert edited.model.wrap_type[wrap] == mujoco.mjtWrap.mjWRAP_SPHERE
    assert edited.model.wrap_prm[wrap] == edited.site_ids["side"]
    assert edited.model.site_pos[edited.site_ids["insertion"], 2] == 0.35
    assert first.model.site_pos[first.site_ids["insertion"], 2] == 0.25
    with pytest.raises(ValueError, match="needs an artifact provider"):
        compile_spatial_model(restored, None)

    restored.mujoco_asset.asset = provider.store_object(
        b'<mujoco><include file="ambient.xml"/></mujoco>'
    )
    with pytest.raises(ValueError, match="external resource"):
        compile_spatial_model(restored, provider)


def test_saved_planar_damping_modes_are_passive_and_use_physics_substeps():
    from scripts.biomechanics_fixture import planar_interior

    interior = planar_interior(10, direct_torque=True)
    restored = AcausalGraphSpec.model_validate_json(interior.model_dump_json())
    model = compile_planar_model(restored).model
    data = mujoco.MjData(model)
    data.qpos[:] = (-0.4, 1.1)
    data.qvel[:] = (0.6, -0.3)
    mujoco.mj_forward(model, data)

    damping = np.array([[0.05, 0.025], [0.025, 0.05]])
    np.testing.assert_allclose(data.qfrc_passive, -damping @ data.qvel, atol=1e-12)
    np.testing.assert_array_equal(model.opt.gravity, np.zeros(3))
    assert model.opt.timestep == pytest.approx(0.001)
    assert restored.solver.dt == pytest.approx(0.01)
