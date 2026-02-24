"""MuJoCo model construction for configurable planar chains.

Builds MuJoCo models via the MjSpec API (not MJCF XML) from body presets
and chain/simulation configurations.
"""

from __future__ import annotations

import equinox as eqx
import numpy as np
from jaxtyping import Array, Float

from feedbax.mechanics.body import BodyPreset


class ChainConfig(eqx.Module):
    """Physical chain topology configuration.

    Attributes:
        n_joints: Number of rotational joints.
        muscles_per_joint: Number of muscles per joint (typically 2 for antagonist pairs).
    """

    n_joints: int = 3
    muscles_per_joint: int = 2

    @property
    def n_muscles(self) -> int:
        return self.n_joints * self.muscles_per_joint


class SimConfig(eqx.Module):
    """MuJoCo simulation timing configuration.

    Attributes:
        dt: Physics timestep in seconds (default 0.002 = 500 Hz).
        episode_duration: Total episode length in seconds.
    """

    dt: float = 0.002
    episode_duration: float = 2.0


def _set_if_attr(obj: object, name: str, value: object) -> None:
    if hasattr(obj, name):
        setattr(obj, name, value)


def _capsule_volume(length: float, radius: float) -> float:
    cylinder = np.pi * radius * radius * length
    sphere = 4.0 / 3.0 * np.pi * radius**3
    return cylinder + sphere


def build_model(
    preset: BodyPreset,
    chain_config: ChainConfig,
    sim_config: SimConfig,
    muscle_config=None,
):
    """Build a MuJoCo model for a planar chain using MjSpec.

    Constructs an N-link planar chain with hinge joints, capsule geometries,
    and muscle-like actuators from the given body parameterization.

    Args:
        preset: Body parameters (segment lengths, masses, damping, etc.).
        chain_config: Chain topology configuration.
        sim_config: Simulation timing configuration.
        muscle_config: Optional muscle attachment configuration. If None,
            a default flexor/extensor layout is created.

    Returns:
        Compiled ``mujoco.MjModel`` instance.
    """
    import mujoco

    if muscle_config is None:
        from feedbax.mechanics.muscle_config import default_muscle_config
        muscle_config = default_muscle_config(preset, chain_config)

    spec = mujoco.MjSpec()

    _set_if_attr(spec.option, "timestep", float(sim_config.dt))
    _set_if_attr(spec.option, "gravity", [0.0, -9.81, 0.0])

    if hasattr(spec, "compiler") and hasattr(spec.compiler, "angle"):
        spec.compiler.angle = "radian"

    world = spec.worldbody

    # Ground plane for reference.
    try:
        plane = world.add_geom()
        _set_if_attr(plane, "type", mujoco.mjtGeom.mjGEOM_PLANE)
        _set_if_attr(plane, "size", [2.0, 2.0, 0.1])
        _set_if_attr(plane, "pos", [0.0, 0.0, -0.05])
        _set_if_attr(plane, "rgba", [0.9, 0.9, 0.9, 1.0])
    except Exception:
        pass

    body_map: dict[str, object] = {"world": world}
    joint_names: list[str] = []
    joint_objects: list[object] = []

    parent = world
    x_offset = 0.0
    radius = 0.02

    for i in range(chain_config.n_joints):
        length = float(preset.segment_lengths[i])
        mass = float(preset.segment_masses[i])

        body = parent.add_body(name=f"link{i}", pos=[x_offset, 0.0, 0.0])
        body_map[f"link{i}"] = body

        joint = body.add_joint(name=f"joint{i}")
        _set_if_attr(joint, "type", mujoco.mjtJoint.mjJNT_HINGE)
        _set_if_attr(joint, "axis", [0.0, 0.0, 1.0])
        _set_if_attr(joint, "pos", [0.0, 0.0, 0.0])
        _set_if_attr(joint, "limited", True)
        _set_if_attr(joint, "range", [-np.pi, np.pi])
        _set_if_attr(joint, "damping", float(preset.joint_damping[i]))
        _set_if_attr(joint, "stiffness", float(preset.joint_stiffness[i]))
        joint_names.append(f"joint{i}")
        joint_objects.append(joint)

        geom = body.add_geom(name=f"geom{i}")
        _set_if_attr(geom, "type", mujoco.mjtGeom.mjGEOM_CAPSULE)
        _set_if_attr(geom, "fromto", [0.0, 0.0, 0.0, length, 0.0, 0.0])
        _set_if_attr(geom, "size", [radius, 0.0, 0.0])
        if hasattr(geom, "mass"):
            _set_if_attr(geom, "mass", mass)
        else:
            volume = _capsule_volume(length, radius)
            _set_if_attr(geom, "density", mass / max(volume, 1e-6))

        if i == chain_config.n_joints - 1:
            eff = body.add_site(name="effector", pos=[length, 0.0, 0.0])
            _set_if_attr(eff, "size", [0.01, 0.01, 0.01])
            _set_if_attr(eff, "rgba", [0.1, 0.6, 0.9, 1.0])

        parent = body
        x_offset = length

    # Muscle attachment sites.
    for i in range(muscle_config.n_muscles):
        origin_body = body_map[muscle_config.origin_body[i]]
        insert_body = body_map[muscle_config.insertion_body[i]]
        origin_site = origin_body.add_site(name=f"muscle{i}_origin")
        _set_if_attr(origin_site, "pos", muscle_config.origin_pos[i].tolist())
        _set_if_attr(origin_site, "size", [0.005, 0.005, 0.005])
        insert_site = insert_body.add_site(name=f"muscle{i}_insert")
        _set_if_attr(insert_site, "pos", muscle_config.insertion_pos[i].tolist())
        _set_if_attr(insert_site, "size", [0.005, 0.005, 0.005])

    # Joint actuators representing muscles.
    specific_tension = 30.0  # N/cm^2
    moment_arm_scale = 0.03  # meters
    for i in range(chain_config.n_muscles):
        act = spec.add_actuator()
        _set_if_attr(act, "name", f"muscle{i}")
        if hasattr(act, "set_to_motor"):
            act.set_to_motor()
        elif hasattr(mujoco, "mjtActuator"):
            _set_if_attr(act, "type", mujoco.mjtActuator.mjACT_MOTOR)
        joint_name = joint_names[int(muscle_config.joint_index[i])]
        if hasattr(act, "target"):
            act.trntype = mujoco.mjtTrn.mjTRN_JOINT
            act.target = joint_name
        elif hasattr(act, "joint"):
            act.joint = joint_name
        else:
            _set_if_attr(act, "trntype", mujoco.mjtTrn.mjTRN_JOINT)
            joint_obj = joint_objects[int(muscle_config.joint_index[i])]
            if hasattr(act, "trnid") and hasattr(joint_obj, "id"):
                act.trnid = [joint_obj.id, 0]
        pcsa = float(preset.muscle_pcsa[i])
        max_force = pcsa * specific_tension
        gear = float(np.sign(muscle_config.moment_arm[i]) * moment_arm_scale * max_force)
        _set_if_attr(act, "gear", [gear, 0, 0, 0, 0, 0])
        _set_if_attr(act, "ctrlrange", [0.0, 1.0])
        _set_if_attr(act, "ctrllimited", True)

    return spec.compile()


def to_mjx(model):
    """Transfer a compiled MuJoCo model to an MJX device model.

    Args:
        model: ``mujoco.MjModel`` instance (CPU).

    Returns:
        ``mjx.Model`` on the default JAX device.
    """
    from mujoco import mjx

    return mjx.put_model(model)


def get_site_id(model, name: str) -> int:
    """Look up a site index by name on a CPU MuJoCo model.

    Args:
        model: ``mujoco.MjModel`` instance.
        name: Site name.

    Returns:
        Integer site ID, or -1 if not found.
    """
    import mujoco

    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def get_body_id(model, name: str) -> int:
    """Look up a body index by name on a CPU MuJoCo model.

    Args:
        model: ``mujoco.MjModel`` instance.
        name: Body name.

    Returns:
        Integer body ID, or -1 if not found.
    """
    import mujoco

    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
