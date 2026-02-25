"""Neural CDE controller module.

Implements a Neural Controlled Differential Equation (CDE) controller as a
drop-in replacement for SimpleStagedNetwork. The CDE update rule is:

    h' = h + f_theta(h) @ (obs - obs_prev)

where f_theta is a learned vector field that maps the hidden state to a matrix,
and (obs - obs_prev) is the observation increment. Actions are read out via a
sigmoid-bounded linear layer for muscle excitations in [0, 1].

:copyright: Copyright 2024-2025 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

import logging
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from equinox import Module, field
from equinox.nn import State, StateIndex
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from feedbax.graph import Component

logger = logging.getLogger(__name__)


class CDENetworkState(Module):
    """State PyTree for CDENetwork.

    Attributes:
        input: The concatenated observation vector from the most recent step.
        hidden: The CDE hidden state vector.
        output: The action output from the most recent step.
        obs_prev: The observation vector from the previous step, used to
            compute the observation increment dX = obs - obs_prev.
    """

    input: Float[Array, "inputs"]
    hidden: Float[Array, "hidden_dim"]
    output: Float[Array, "action_dim"]
    obs_prev: Float[Array, "obs_dim"]


class CDENetwork(Component):
    """Neural CDE controller -- drop-in replacement for SimpleStagedNetwork.

    CDE update (Euler discretization, dt cancels with dX/dt):
        h' = h + f_theta(h) @ (obs - obs_prev)

    Action readout:
        action = sigmoid(readout(h))

    Same input/output ports as SimpleStagedNetwork:
        input_ports = ("input", "feedback")
        output_ports = ("output", "hidden")

    Attributes:
        obs_dim: Dimensionality of the observation vector (static).
        hidden_dim: Dimensionality of the CDE hidden state (static).
        out_size: Dimensionality of the action output. Matches
            SimpleStagedNetwork attribute name for compatibility.
        vector_field: MLP mapping h -> flattened matrix of shape
            (hidden_dim * obs_dim,). Reshaped to (hidden_dim, obs_dim)
            for the CDE step.
        readout: Linear layer mapping h -> action_dim.
        h0: Learned initial hidden state vector.
        state_index: Equinox StateIndex for state management.
    """

    obs_dim: int = field(static=True)
    hidden_dim: int = field(static=True)
    out_size: int
    vector_field: eqx.nn.MLP
    readout: eqx.nn.Linear
    h0: Float[Array, "hidden_dim"]

    state_index: StateIndex
    _initial_state: CDENetworkState = field(static=True)

    input_ports = ("input", "feedback")
    output_ports = ("output", "hidden")

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        out_size: int,
        vf_width: int = 64,
        vf_depth: int = 2,
        *,
        key: PRNGKeyArray,
    ):
        """Construct a CDENetwork.

        Args:
            obs_dim: Dimensionality of the concatenated observation vector
                (task input + feedback).
            hidden_dim: Dimensionality of the CDE hidden state.
            out_size: Dimensionality of the action output (e.g. number of
                muscles or force dimensions).
            vf_width: Width of hidden layers in the vector field MLP.
            vf_depth: Number of hidden layers in the vector field MLP.
            key: PRNG key for parameter initialization.
        """
        key_vf, key_readout, key_h0 = jr.split(key, 3)

        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.out_size = out_size

        # Vector field: h -> matrix(hidden_dim, obs_dim), stored flat
        self.vector_field = eqx.nn.MLP(
            in_size=hidden_dim,
            out_size=hidden_dim * obs_dim,
            width_size=vf_width,
            depth=vf_depth,
            key=key_vf,
        )

        # Readout: h -> action_dim
        self.readout = eqx.nn.Linear(hidden_dim, out_size, key=key_readout)

        # Learned initial hidden state (small random init)
        self.h0 = 0.01 * jr.normal(key_h0, (hidden_dim,))

        # Build initial state for Equinox State management
        init_state = CDENetworkState(
            input=jnp.zeros(obs_dim),
            hidden=self.h0,
            output=jnp.zeros(out_size),
            obs_prev=jnp.zeros(obs_dim),
        )
        self._initial_state = init_state
        self.state_index = StateIndex(init_state)

    def _cde_step(
        self,
        h: Float[Array, "hidden_dim"],
        obs: Float[Array, "obs_dim"],
        obs_prev: Float[Array, "obs_dim"],
    ) -> Float[Array, "hidden_dim"]:
        """Single CDE Euler step.

        Args:
            h: Current hidden state.
            obs: Current observation vector.
            obs_prev: Previous observation vector.

        Returns:
            Updated hidden state h' = h + M @ dX where
            M = vector_field(h).reshape(hidden_dim, obs_dim) and
            dX = obs - obs_prev.
        """
        dX = obs - obs_prev
        M = self.vector_field(h).reshape(self.hidden_dim, self.obs_dim)
        return h + M @ dX

    def _get_action(self, h: Float[Array, "hidden_dim"]) -> Float[Array, "action_dim"]:
        """Compute bounded action from hidden state.

        Args:
            h: Hidden state vector.

        Returns:
            Action in [0, 1] via sigmoid(readout(h)).
        """
        return jax.nn.sigmoid(self.readout(h))

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        """Execute one CDE step.

        Assembles the observation from "input" and "feedback" ports (same
        convention as SimpleStagedNetwork), performs a CDE update, and
        returns the action and updated hidden state.

        Args:
            inputs: Dict with optional "input" and "feedback" port values.
            state: Equinox State container.
            key: PRNG key (unused, but required by Component interface).

        Returns:
            Tuple of (output_dict, updated_state) where output_dict has
            "output" (action) and "hidden" (new hidden state) keys.
        """
        net_state: CDENetworkState = state.get(self.state_index)

        # Assemble observation from input ports, matching SimpleStagedNetwork
        input_value = inputs.get("input", None)
        feedback_value = inputs.get("feedback", None)

        if input_value is None and feedback_value is None:
            raise ValueError("CDENetwork requires at least one input.")

        if input_value is None:
            flat_input = jnp.zeros((0,))
        else:
            flat_input, _ = ravel_pytree(input_value)

        if feedback_value is None:
            flat_feedback = jnp.zeros((0,))
        else:
            flat_feedback, _ = ravel_pytree(feedback_value)

        obs = jnp.concatenate([flat_input, flat_feedback], axis=-1)

        # CDE step
        h_new = self._cde_step(net_state.hidden, obs, net_state.obs_prev)

        # Action readout
        action = self._get_action(h_new)

        # Update state
        new_state = CDENetworkState(
            input=obs,
            hidden=h_new,
            output=action,
            obs_prev=obs,
        )
        state = state.set(self.state_index, new_state)

        return {"output": action, "hidden": h_new}, state
