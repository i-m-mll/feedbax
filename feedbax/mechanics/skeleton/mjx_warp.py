"""Construction of the stock MJX-Warp model and data template."""

from __future__ import annotations

import jax
import numpy as np


def make_warp_model_and_data(model):
    """Allocate a collision-free profile whose capacity is independent of batch size."""
    from mujoco import mjtDisableBit, mjtEnableBit, mjx
    from mujoco.mjx.warp import WARP_INSTALLED

    if not WARP_INSTALLED:
        raise ValueError("MJX-Warp requires the Warp runtime for the installed MuJoCo MJX version")
    if model.opt.enableflags & mjtEnableBit.mjENBL_SLEEP:
        raise ValueError("MJX-Warp continuation does not support sleeping bodies")
    if model.nflex:
        raise ValueError("The MJX-Warp multibody profile supports rigid bodies only")
    if not model.opt.disableflags & mjtDisableBit.mjDSBL_CONTACT:
        contact_types = np.bitwise_or.reduce(model.geom_contype, initial=0)
        contact_affinities = np.bitwise_or.reduce(model.geom_conaffinity, initial=0)
        if model.npair or (model.ngeom > 1 and contact_types & contact_affinities):
            raise ValueError(
                "The MJX-Warp profile requires collisions to be disabled or excluded. "
                "Batched contact capacity has not been qualified."
            )

    device = jax.devices()[0]
    if device.platform != "cpu":
        try:
            cuda_devices = jax.devices("cuda")
        except RuntimeError as exc:
            raise ValueError("MJX-Warp requires a JAX CUDA device") from exc
        if device not in cuda_devices:
            raise ValueError("MJX-Warp requires a JAX CUDA device")

    # Each equality contributes at most six rows. Limits and dry friction
    # contribute at most one row per limited joint/tendon or friction coordinate.
    max_rows = (
        6 * model.neq + np.count_nonzero(model.jnt_limited)
        + np.count_nonzero(model.tendon_limited) + np.count_nonzero(model.dof_frictionloss)
        + np.count_nonzero(model.tendon_frictionloss)
    )
    return (
        mjx.put_model(model, impl="warp", device=device),
        mjx.make_data(
            model, impl="warp", device=device,
            naconmax=1, naccdmax=1, njmax=max(1, int(max_rows)),
        ),
    )
