"""Executable mechanics for a compiled authored MuJoCo model."""

from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.nn import State
from jaxtyping import PRNGKeyArray

from feedbax._atomic import write_bytes_atomic
from feedbax._vendor.jax_cookbook._io import json_dump, load
from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.acausal_interface import derive_acausal_interface, is_acausal_type
from feedbax.mechanics.backend import DiffraxBackend, MJXBackend, PhysicsCapabilities
from feedbax.mechanics.compiled_model import CompiledPhysicalModel
from feedbax.mechanics.mechanics import Mechanics
from feedbax.mechanics.plant import DirectForceInput
from feedbax.mechanics.skeleton.mjx_skeleton import AppliedWrench, MJXSkeleton
from feedbax.mechanics.solver_config import resolve_acausal_solver


class AuthoredMechanics(Mechanics):
    """The graph boundary for planar and spatial MJX models."""

    authored_json: str = eqx.field(static=True)
    descriptor: CompiledPhysicalModel = eqx.field(static=True)
    capabilities: PhysicsCapabilities = eqx.field(static=True)
    _inputs: tuple[str, ...] = eqx.field(static=True)
    _outputs: tuple[tuple[str, str], ...] = eqx.field(static=True)
    _solver_name: str = eqx.field(static=True)

    def __init__(self, graph: AcausalGraphSpec, descriptor: CompiledPhysicalModel):
        from mujoco import mjtDyn, mjtIntegrator, mjtWrap

        interface = derive_acausal_interface(graph)
        if len(interface.input_ports) != 1 or interface.boundary_ports:
            raise ValueError("MuJoCo graphs require one control input and no conserving boundary ports")
        adapters = [
            node for node in graph.nodes.values() if is_acausal_type(node.type, "ActuationInput")
        ]
        if len(adapters) != 1:
            raise ValueError("MuJoCo control must be declared with one ActuationInput")
        source_kind = adapters[0].params.get("source_kind", "force")
        skeleton = MJXSkeleton.from_descriptor(descriptor)
        model = descriptor.model
        self._solver_name = graph.solver.solver_type
        supported_actuation = {
            int(mjtDyn.mjDYN_NONE), int(mjtDyn.mjDYN_INTEGRATOR), int(mjtDyn.mjDYN_FILTER),
        }
        continuous = (
            skeleton.constraint_rows == 0
            and set(model.actuator_dyntype.tolist()) <= supported_actuation
            and not model.actuator_actearly.any()
            and not model.actuator_actlimited.any()
        )
        native = model.opt.integrator in (
            mjtIntegrator.mjINT_EULER, mjtIntegrator.mjINT_RK4, mjtIntegrator.mjINT_IMPLICITFAST,
        )
        wrapped = {int(mjtWrap.mjWRAP_SPHERE), int(mjtWrap.mjWRAP_CYLINDER)}
        self.capabilities = PhysicsCapabilities(
            native_step=native and source_kind == "actuator_control",
            continuous=continuous,
            reverse_mode=skeleton.constraint_rows == 0 and wrapped.isdisjoint(model.wrap_type),
        )
        if self._solver_name == "mujoco":
            if graph.solver.root_finder is not None:
                raise ValueError("Native MuJoCo stepping uses the asset's solver options")
            if not native:
                raise ValueError("The authored MuJoCo integrator is unsupported by MJX-JAX")
            backend = MJXBackend(control_dt=graph.solver.dt, sub_dt=model.opt.timestep)
        else:
            if not continuous:
                raise ValueError(
                    "Continuous MJX requires zero compiled constraint rows and "
                    "unbounded, stage-local actuator dynamics. Use native MuJoCo stepping "
                    "for this model."
                )
            solver = resolve_acausal_solver(
                self._solver_name, graph.solver.dt, physical_domain=graph.physical_domain,
                root_finder=graph.solver.root_finder,
            )
            backend = DiffraxBackend(
                control_dt=graph.solver.dt, sub_dt=model.opt.timestep, solver=solver,
            )
        if source_kind == "actuator_control":
            plant = DirectForceInput(skeleton, clip_states=False)
        elif source_kind == "muscle_excitation_vector" and isinstance(backend, DiffraxBackend):
            from feedbax.mechanics.analytical_plant import AnalyticalMusculoskeletalPlant

            plant = AnalyticalMusculoskeletalPlant.from_authored_planar_model(skeleton)
        else:
            raise ValueError(f"Unsupported MuJoCo control {source_kind!r} for {self._solver_name!r}")

        port = interface.input_ports[0]
        interface.input_payloads[port].validate_shape(port, (plant.input_size,))
        available = {"effector", "position", "velocity", "state"}
        by_port = {
            node.params.get("port_name", "y"): node.params.get("quantity", "position")
            for node in graph.nodes.values() if is_acausal_type(node.type, "SensorOutput")
        }
        if unsupported := set(by_port.values()) - available:
            raise ValueError(f"Unsupported MuJoCo observation quantities: {sorted(unsupported)}")
        self._inputs = interface.input_ports
        self._outputs = tuple((port, by_port[port]) for port in interface.output_ports)
        self.authored_json = graph.model_dump_json(exclude_none=True)
        self.descriptor = descriptor
        super().__init__(plant=plant, backend=backend)

    @property
    def input_ports(self) -> tuple[str, ...]:
        return self._inputs

    @property
    def output_ports(self) -> tuple[str, ...]:
        return tuple(port for port, _ in self._outputs)

    @property
    def supports_reverse_mode(self) -> bool:
        """The stock JAX solver is qualified only with zero allocated constraint rows."""
        return self.capabilities.reverse_mode

    @property
    def workspace_dimension(self) -> int:
        return self.descriptor.dimension

    def _observations(self, values):
        if not values:
            return {}
        effector = values["effector"]
        values = {**values, "position": effector.pos, "velocity": effector.vel}
        return {port: values[quantity] for port, quantity in self._outputs}

    def __call__(self, inputs, state, *, key):
        port = self._inputs[0]
        if port not in inputs:
            raise ValueError(f"Missing required MuJoCo control input {port!r}")
        action = jnp.asarray(inputs[port])
        if action.shape != (self.plant.input_size,):
            raise ValueError(
                f"MuJoCo input {port!r} expects shape ({self.plant.input_size},), got {action.shape}"
            )
        values, state = super().__call__({"force": action}, state, key=key)
        return self._observations(values), state

    def initial_outputs(self, state_value):
        return self._observations(super().initial_outputs(state_value))

    def state_consistency_update(self, state: State) -> State:
        value = self.state_view(state)
        skeleton = self.plant.skeleton.project_state(value.plant.skeleton)
        plant = replace(value.plant, skeleton=skeleton)
        effector = self.plant.skeleton.effector(skeleton)
        return state.set(self.state_index, replace(value, plant=plant, effector=effector))

    def replace_state_value(self, state: State, path, value):
        if path in (("effector",), ("effector", "pos"), ("effector", "vel")):
            position = value.pos if path == ("effector",) else value
            if jnp.shape(position) != (self.workspace_dimension,):
                raise ValueError(f"Cartesian initialization requires {self.workspace_dimension}D coordinates")
            raise ValueError("MuJoCo trial initialization requires generalized coordinates")
        state = super().replace_state_value(state, path, value)
        if path[:2] == ("plant", "skeleton"):
            return self.state_consistency_update(state)
        return state

    def apply_wrench(self, state: State, body: str, wrench: AppliedWrench) -> State:
        """Replace the held load on a named body until the caller replaces it."""
        if body not in self.descriptor.body_ids:
            raise ValueError(f"Unknown MuJoCo body {body!r}")
        value = self.state_view(state)
        skeleton = self.plant.skeleton.with_wrench(
            value.plant.skeleton, self.descriptor.body_ids[body], wrench,
        )
        plant = replace(value.plant, skeleton=skeleton)
        return state.set(self.state_index, replace(value, plant=plant))

    def _state_header(self) -> dict:
        import mujoco

        return {
            "schema": "feedbax.mujoco.continuation.v1",
            "model": self.descriptor.identity,
            "structure": self.descriptor.structural_signature,
            "runtime": ["mjx-jax", mujoco.__version__, jax.__version__],
            "solver": self._solver_name,
            "periods": [self.backend.control_dt, self.backend.sub_dt],
        }

    def save_state(self, path: Path, state: State) -> None:
        """Save this model's physical and integration state with its binding."""
        with BytesIO() as buffer:
            json_dump(buffer, self._state_header())
            eqx.tree_serialise_leaves(buffer, self.state_view(state))
            write_bytes_atomic(path, buffer.getvalue())

    def resume_state(self, path: Path, state: State) -> State:
        """Restore an exact same-runtime snapshot into the current graph state."""
        def template(*, key: PRNGKeyArray, **header):
            if header != self._state_header():
                raise ValueError("MuJoCo checkpoint model, runtime or solver binding differs")
            return self.state_view(state)

        return state.set(self.state_index, load(path, template))

    def transfer_state(self, source: AuthoredMechanics, source_state: State, state: State) -> State:
        """Transfer common physical state and initialize the destination solver buffers."""
        if source.descriptor.identity != self.descriptor.identity:
            raise ValueError("MuJoCo state transfer requires the same physical model")
        old = source.state_view(source_state)
        skeleton = self.plant.skeleton.reset_continuation(old.plant.skeleton)
        plant = replace(old.plant, skeleton=skeleton)
        initial = self.backend.init_state(self.plant, key=jax.random.PRNGKey(0))
        effector = self.plant.skeleton.effector(skeleton)
        return state.set(
            self.state_index,
            replace(self.state_view(state), plant=plant, solver=initial.aux, effector=effector),
        )
