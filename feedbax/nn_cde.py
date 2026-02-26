"""Neural CDE controller module.

Implements a Neural Controlled Differential Equation (CDE) controller as a
drop-in replacement for SimpleStagedNetwork. The CDE update rule is:

    h' = h + f_theta(h) @ (obs - obs_prev) - decay * h

where f_theta is a learned vector field that maps the hidden state to a matrix,
(obs - obs_prev) is the observation increment, and the decay term provides
LTC-inspired dissipation for hidden state stability. Actions are read out via a
sigmoid-bounded linear layer for muscle excitations in [0, 1].

Optionally, the simple linear decay can be combined with an Anti-NF gated decay
mechanism (Kuleshov et al. 2024, "DeNOTS: Stable Deep Neural ODEs"), where a
GRU cell computes adaptive per-dimension negative feedback on top of the fixed
decay floor:

    gated_feedback = GRU(obs, -h)
    h' = h + f_theta(h) @ dX - decay * h + alpha * gated_feedback

The GRU receives the observation as input and the negated hidden state as its
hidden state, learning input-dependent refinement that adapts to the current
regime. The fixed decay provides an unconditional stability floor, while the
gate can partially counteract the decay for important state dimensions.

:copyright: Copyright 2024-2025 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

import logging
from typing import Optional, Union

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
        h' = h + f_theta(h) @ (obs - obs_prev) - decay * h

    Action readout:
        action = sigmoid(readout(h))

    Same input/output ports as SimpleStagedNetwork:
        input_ports = ("input", "feedback")
        output_ports = ("output", "hidden")

    Attributes:
        obs_dim: Dimensionality of the observation vector (static).
        hidden_dim: Dimensionality of the CDE hidden state (static).
        out_size: Dimensionality of the action output (static). Matches
            SimpleStagedNetwork attribute name for compatibility.
        use_anti_nf: Whether to use Anti-NF gated decay in addition to
            fixed decay (True) or simple linear decay only (False).
            Static field, traced out at JIT time.
        vector_field: MLP mapping h -> flattened matrix of shape
            (hidden_dim * obs_dim,). Reshaped to (hidden_dim, obs_dim)
            for the CDE step.
        readout: Linear layer mapping h -> action_dim.
        h0: Learned initial hidden state vector.
        anti_nf_gate: GRUCell for Anti-NF gated decay, or None when
            use_anti_nf is False. Input is obs, hidden state is -h.
        alpha: Feedback strength scalar for Anti-NF gated decay.
        decay: Hidden state decay rate (LTC-inspired dissipation). Always
            active as an unconditional stability floor.
        state_index: Equinox StateIndex for state management.
    """

    obs_dim: int = field(static=True)
    hidden_dim: int = field(static=True)
    out_size: int = field(static=True)
    use_anti_nf: bool = field(static=True)
    vector_field: eqx.nn.MLP
    readout: eqx.nn.Linear
    h0: Float[Array, "hidden_dim"]
    anti_nf_gate: Union[eqx.nn.GRUCell, None]
    alpha: float

    state_index: StateIndex
    _initial_state: CDENetworkState = field(static=True)

    input_ports = ("input", "feedback")
    output_ports = ("output", "hidden")

    decay: float

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        out_size: int,
        vf_width: int = 64,
        vf_depth: int = 2,
        decay: float = 0.1,
        use_anti_nf: bool = True,
        alpha: float = 1.0,
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
            decay: Hidden state decay rate (LTC-inspired dissipation). Pulls
                h toward zero each step, preventing unbounded drift. Always
                active as an unconditional stability floor.
            use_anti_nf: Whether to add Anti-NF gated decay on top of
                fixed decay (True) or use fixed decay only (False).
            alpha: Feedback strength scalar for Anti-NF gated decay.
            key: PRNG key for parameter initialization.
        """
        key_vf, key_readout, key_h0, key_gate = jr.split(key, 4)

        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.out_size = out_size
        self.decay = decay
        self.use_anti_nf = use_anti_nf
        self.alpha = alpha

        # Anti-NF gated decay (DeNOTS): GRU receives obs as input, -h as
        # hidden state, producing learned input-dependent negative feedback.
        if use_anti_nf:
            self.anti_nf_gate = eqx.nn.GRUCell(
                input_size=obs_dim, hidden_size=hidden_dim, key=key_gate,
            )
        else:
            self.anti_nf_gate = None

        # Vector field: h -> matrix(hidden_dim, obs_dim), stored flat.
        # tanh final activation bounds output to [-1, 1], which bounds dh per
        # step to ||dX|| — Kidger's canonical CDE stability fix.
        self.vector_field = eqx.nn.MLP(
            in_size=hidden_dim,
            out_size=hidden_dim * obs_dim,
            width_size=vf_width,
            depth=vf_depth,
            final_activation=jax.nn.tanh,
            key=key_vf,
        )

        # Readout: h -> action_dim
        self.readout = eqx.nn.Linear(hidden_dim, out_size, key=key_readout)
        # Quiescent start: sigmoid(-5) ≈ 0.007, muscles start near-zero
        self.readout = eqx.tree_at(
            lambda l: l.bias, self.readout, -5.0 * jnp.ones(out_size)
        )

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
        """Single CDE step (hybrid v9b).

        Stability is handled by the vector field's tanh output activation,
        which bounds each element of M to [-1, 1] and thus bounds ||dh|| by
        ||dX|| per step (Kidger's canonical CDE stability approach).

        Fixed decay (``-decay * h``) is always applied as an unconditional
        stability floor that pulls the hidden state toward zero.

        When use_anti_nf is True, GRU-gated adaptive negative feedback
        (DeNOTS Anti-NF) is added on top of the fixed decay. The GRU
        receives the observation as input and -h as hidden state, learning
        input-dependent per-dimension refinement. When the gate is
        well-trained, it can partially counteract the fixed decay for
        important state dimensions, allowing richer dynamics. When gate
        weights get corrupted (e.g. by gradient explosions), the fixed
        decay still pulls the hidden state back, providing recovery that
        the gate alone cannot guarantee.

        Args:
            h: Current hidden state.
            obs: Current observation vector.
            obs_prev: Previous observation vector.

        Returns:
            Updated hidden state.
        """
        dX = obs - obs_prev
        M = self.vector_field(h).reshape(self.hidden_dim, self.obs_dim)
        cde_update = M @ dX

        if self.use_anti_nf:
            # Hybrid v9b: fixed decay floor + Anti-NF adaptive refinement
            # GRU input=obs, hidden=-h -> output is gated negative feedback
            gated_feedback = self.anti_nf_gate(obs, -h)
            h_new = h + cde_update - self.decay * h + self.alpha * gated_feedback
        else:
            # Simple linear decay only (v8 behavior)
            h_new = h + cde_update - self.decay * h

        return h_new

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
