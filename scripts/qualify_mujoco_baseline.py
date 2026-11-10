"""Reproduce the stock MJX baseline against native MuJoCo on the CPU."""

from importlib.metadata import PackageNotFoundError, version

import jax
import mujoco
import numpy as np
from mujoco import mjx

# The spatial tendon example from upstream issue 3696.
WRAP = """<mujoco>
  <worldbody>
    <site name="origin" pos="0 0 0.3"/>
    <body name="arm">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom name="bone" type="capsule" fromto="0 0 0 0 0 -0.3" size="0.02"/>
      <site name="insert" pos="0.02 0 -0.25"/>
      <geom name="wrapper" type="sphere" size="0.05" pos="0 0 -0.1" contype="0" conaffinity="0"/>
      <site name="side" pos="{side} 0 -0.1" size="0.002"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="muscle" width="0.004">
      <site site="origin"/><geom geom="wrapper" sidesite="side"/><site site="insert"/>
    </spatial>
  </tendon>
</mujoco>"""

STACK = """<mujoco>
  <option iterations="{iterations}" tolerance="1e-13" solver="CG"/>
  <worldbody>
    <geom type="plane" size="1 1 .1" condim="1"/>
    <body pos="0 0 .099">
      <joint type="slide" axis="0 0 1"/>
      <geom type="sphere" size=".1" mass="1" condim="1"/>
    </body>
    <body pos="0 0 .298">
      <joint type="slide" axis="0 0 1"/>
      <geom type="sphere" size=".1" mass="1" condim="1"/>
    </body>
    <body pos="0 0 .497">
      <joint type="slide" axis="0 0 1"/>
      <geom type="sphere" size=".1" mass="1" condim="1"/>
    </body>
  </worldbody>
</mujoco>"""

HINGE = """<mujoco>
  <option timestep=".002" gravity="0 0 0"/>
  <worldbody>
    <body name="rotor">
      <inertial pos="0 0 0" mass="1" diaginertia=".25 .25 .25"/>
      <joint name="shaft" type="hinge"/>
    </body>
  </worldbody>
  <actuator><motor joint="shaft"/></actuator>
</mujoco>"""

def central_fd(fun, x, h=1e-5):
    return (fun(x + h) - fun(x - h)) / (2 * h)


def check_wrap():
    for side in (0.09, 0.01):
        native_model = mujoco.MjModel.from_xml_string(WRAP.format(side=side))
        model = mjx.put_model(native_model, impl="jax")
        data = mjx.make_data(model)

        def cpu(q):
            native_data = mujoco.MjData(native_model)
            native_data.qpos[0] = q
            mujoco.mj_forward(native_model, native_data)
            return native_data.ten_length[0]

        @jax.jit
        def length(q):
            kinematic = mjx.kinematics(model, data.replace(qpos=data.qpos.at[0].set(q)))
            return mjx.tendon(model, kinematic).ten_length[0]

        derivative = jax.grad(length)
        for q in (0.0, 0.3, 0.6, 1.0):
            value = float(length(q))
            slope = float(derivative(q))
            reference = cpu(q)
            reference_slope = central_fd(cpu, q)
            print(f"wrap side={side} q={q}: length={value:.12g} native={reference:.12g} "
                  f"AD={slope:.12g} native FD={reference_slope:.12g}")
            np.testing.assert_allclose(value, reference, rtol=0, atol=1e-12)
            if side == 0.09:
                np.testing.assert_allclose(slope, reference_slope, rtol=0, atol=1e-8)
            else:
                assert np.isnan(slope)


def check_constraint_solver():
    solutions = []
    for iterations in (1, 12):
        native_model = mujoco.MjModel.from_xml_string(STACK.format(iterations=iterations))
        model = mjx.put_model(native_model, impl="jax")
        data = mjx.make_data(model)

        def cpu(f):
            native_data = mujoco.MjData(native_model)
            native_data.qfrc_applied[-1] = f
            mujoco.mj_forward(native_model, native_data)
            return native_data.qacc.copy(), native_data.solver_niter[0]

        @jax.jit
        def acceleration(f):
            state = mjx.forward(model, data.replace(qfrc_applied=data.qfrc_applied.at[-1].set(f)))
            return state.qacc

        force = -1.0
        value = np.asarray(acceleration(force))
        reference, used_iterations = cpu(force)
        reference_slope = central_fd(lambda f: cpu(f)[0][0], force)
        print(f"CG limit={iterations}: qacc={value} native={reference} "
              f"native iterations={used_iterations} native FD={reference_slope:.12g}")
        np.testing.assert_allclose(value, reference, rtol=0, atol=1e-10)
        solutions.append(value)
        derivative = jax.grad(lambda f: acceleration(f)[0])
        if iterations == 1:
            slope = float(derivative(force))
            print(f"CG limit={iterations}: AD={slope:.12g}")
            np.testing.assert_allclose(slope, reference_slope, rtol=0, atol=1e-8)
        else:
            assert 1 < used_iterations < iterations
            try:
                derivative(force)
            except ValueError as error:
                if "Reverse-mode differentiation does not work for lax.while_loop" not in str(error):
                    raise
                print(f"CG limit={iterations}: {error}")
            else:
                raise AssertionError("Expected the stock constrained reverse-mode failure")

    assert np.linalg.norm(solutions[0] - solutions[1]) > 1.0


def check_unconstrained_step():
    native_model = mujoco.MjModel.from_xml_string(HINGE)
    model = mjx.put_model(native_model, impl="jax")
    data = mjx.make_data(model)

    def cpu(u):
        native_data = mujoco.MjData(native_model)
        native_data.ctrl[0] = u
        mujoco.mj_step(native_model, native_data)
        return native_data.qpos[0], native_data.qvel[0]

    @jax.jit
    def step(u):
        updated = mjx.step(model, data.replace(ctrl=data.ctrl.at[0].set(u)))
        return updated.qpos[0], updated.qvel[0]

    torque = 0.3
    dt = float(native_model.opt.timestep)
    speed = torque * dt / 0.25
    value = np.asarray(step(torque))
    reference = cpu(torque)
    slope = float(jax.grad(lambda u: step(u)[0])(torque))
    reference_slope = central_fd(lambda u: cpu(u)[0], torque)
    print(f"unconstrained hinge: (qpos, qvel)={value} native={reference} "
          f"AD={slope:.12g} native FD={reference_slope:.12g}")
    np.testing.assert_allclose(value, (speed * dt, speed), rtol=0, atol=1e-12)
    np.testing.assert_allclose(value, reference, rtol=0, atol=1e-12)
    np.testing.assert_allclose(slope, reference_slope, rtol=0, atol=1e-10)


def main():
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platforms", "cpu")
    for package in ("mujoco", "mujoco-core", "mujoco-mjx", "jax", "jaxlib", "numpy", "scipy"):
        try:
            installed_version = version(package)
        except PackageNotFoundError:
            continue
        print(package, installed_version)
    print(jax.devices())
    check_unconstrained_step()
    check_wrap()
    check_constraint_solver()


if __name__ == "__main__":
    main()
