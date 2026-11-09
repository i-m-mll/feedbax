"""The native model and the layout that Feedbax uses to bind its state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Mapping

import numpy as np

from feedbax.contracts.canonical_json import canonical_json_v2_bytes

if TYPE_CHECKING:
    from mujoco import MjModel


@dataclass(frozen=True)
class JointLayout:
    name: str
    kind: Literal["free", "ball", "slide", "hinge"]
    qpos: slice
    qvel: slice


@dataclass(frozen=True)
class MusclePathBinding:
    """An ordered named site path and the authored law and state layout."""

    name: str
    sites: tuple[str, ...]
    law: str
    state_size: int
    parameters: Mapping[str, object]
    tendon_id: int = -1
    actuator_id: int | None = None
    law_revision: int | None = None

    @property
    def actuator_name(self) -> str:
        return f"feedbax/path/{self.name}"


@dataclass(frozen=True, eq=False)
class CompiledPhysicalModel:
    model: MjModel
    identity: str
    asset_identity: str | None
    structural_signature: str
    dimension: Literal[2, 3]
    effector_site: str | None
    body_ids: Mapping[str, int]
    joint_ids: Mapping[str, int]
    site_ids: Mapping[str, int]
    tendon_ids: Mapping[str, int]
    actuator_ids: Mapping[str, int]
    joints: tuple[JointLayout, ...]
    actuator_state: tuple[slice, ...]
    paths: tuple[MusclePathBinding, ...]
    planar_segment_lengths: tuple[float, ...]


def describe_model(
    model: MjModel,
    *,
    identity: str,
    dimension: Literal[2, 3],
    effector_site: str | None = None,
    asset_identity: str | None = None,
    paths: tuple[MusclePathBinding, ...] = (),
    planar_segment_lengths: tuple[float, ...] = (),
) -> CompiledPhysicalModel:
    """Derive binding and grouping information from the compiled model."""
    import mujoco

    def indices(kind: mujoco.mjtObj, size: int) -> Mapping[str, int]:
        return MappingProxyType({
            name: i for i in range(size)
            if (name := mujoco.mj_id2name(model, kind, i)) is not None
        })

    kinds = {
        mujoco.mjtJoint.mjJNT_FREE: ("free", 7, 6),
        mujoco.mjtJoint.mjJNT_BALL: ("ball", 4, 3),
        mujoco.mjtJoint.mjJNT_SLIDE: ("slide", 1, 1),
        mujoco.mjtJoint.mjJNT_HINGE: ("hinge", 1, 1),
    }
    body_ids = indices(mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    joint_ids = indices(mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    site_ids = indices(mujoco.mjtObj.mjOBJ_SITE, model.nsite)
    if effector_site is not None and effector_site not in site_ids:
        raise ValueError(f"Unknown MuJoCo effector site {effector_site!r}")
    tendon_ids = indices(mujoco.mjtObj.mjOBJ_TENDON, model.ntendon)
    actuator_ids = indices(mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    paths = tuple(
        replace(
            path, tendon_id=tendon_ids[path.name], actuator_id=actuator_ids.get(path.actuator_name),
        )
        for path in paths
    )
    for path in paths:
        sites = [site_ids[site] for site in path.sites]
        for a, b in zip(sites, sites[1:]):
            if (model.site_bodyid[a] == model.site_bodyid[b]
                    and np.array_equal(model.site_pos[a], model.site_pos[b])):
                raise ValueError(f"MusclePath {path.name!r} contains a zero-length fixed segment")
    joint_names = {i: name for name, i in joint_ids.items()}
    joints = []
    for i, (joint_type, qpos, qvel) in enumerate(
        zip(model.jnt_type, model.jnt_qposadr, model.jnt_dofadr, strict=True)
    ):
        kind, nq, nv = kinds[mujoco.mjtJoint(joint_type)]
        joints.append(JointLayout(
            joint_names.get(i, ""), kind,
            slice(int(qpos), int(qpos) + nq), slice(int(qvel), int(qvel) + nv),
        ))

    structure = {
        "observation": [dimension, effector_site],
        "layout": [model.nq, model.nv, model.nu, model.na, model.nmocap],
        "names": [
            dict(index) for index in (body_ids, joint_ids, site_ids, tendon_ids, actuator_ids)
        ],
        "topology": {
            name: getattr(model, name).tolist()
            for name in (
                "body_parentid", "body_mocapid", "jnt_bodyid", "jnt_type", "jnt_limited",
                "jnt_qposadr", "jnt_dofadr", "site_bodyid", "geom_type", "geom_bodyid",
                "geom_contype", "geom_conaffinity", "eq_type", "eq_obj1id", "eq_obj2id",
                "tendon_adr", "tendon_num", "wrap_type", "wrap_objid",
                "actuator_trntype", "actuator_trnid", "actuator_dyntype", "actuator_gaintype",
                "actuator_biastype", "actuator_actadr", "actuator_actnum",
            )
        },
        "paths": [
            [path.name, list(path.sites), path.law, path.state_size]
            if path.law_revision is None else
            [path.name, list(path.sites), path.law, path.state_size, path.law_revision]
            for path in paths
        ],
    }
    signature = sha256(canonical_json_v2_bytes(structure)).hexdigest()
    actuator_state = tuple(
        slice(int(address), int(address) + int(count)) if count else slice(0, 0)
        for address, count in zip(model.actuator_actadr, model.actuator_actnum, strict=True)
    )
    return CompiledPhysicalModel(
        model, identity, asset_identity, signature, dimension, effector_site,
        body_ids, joint_ids, site_ids, tendon_ids, actuator_ids, tuple(joints), actuator_state,
        paths, planar_segment_lengths,
    )
