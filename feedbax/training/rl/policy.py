"""Actor-critic policy for PPO training.

Uses a Beta distribution for bounded [0, 1] actions (muscle excitations),
implemented with native JAX (no distrax dependency).
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class BetaParams(NamedTuple):
    """Parameters of a Beta distribution with log_prob and entropy methods.

    Attributes:
        alpha: Concentration parameter alpha, shape ``(..., action_dim)``.
        beta: Concentration parameter beta, shape ``(..., action_dim)``.
    """

    alpha: Float[Array, "... action_dim"]
    beta: Float[Array, "... action_dim"]

    def log_prob(self, x: Float[Array, "... action_dim"]) -> Float[Array, "... action_dim"]:
        """Log probability density of x under the Beta distribution.

        Args:
            x: Values in (0, 1), shape ``(..., action_dim)``.

        Returns:
            Per-dimension log probabilities.
        """
        x = jnp.clip(x, 1e-6, 1.0 - 1e-6)
        return (
            (self.alpha - 1.0) * jnp.log(x)
            + (self.beta - 1.0) * jnp.log(1.0 - x)
            + jax.lax.lgamma(self.alpha + self.beta)
            - jax.lax.lgamma(self.alpha)
            - jax.lax.lgamma(self.beta)
        )

    def entropy(self) -> Float[Array, "... action_dim"]:
        """Differential entropy of the Beta distribution.

        Returns:
            Per-dimension entropy values.
        """
        ab = self.alpha + self.beta
        return (
            jax.lax.lgamma(self.alpha)
            + jax.lax.lgamma(self.beta)
            - jax.lax.lgamma(ab)
            - (self.alpha - 1.0) * jax.scipy.special.digamma(self.alpha)
            - (self.beta - 1.0) * jax.scipy.special.digamma(self.beta)
            + (ab - 2.0) * jax.scipy.special.digamma(ab)
        )

    def sample(self, key: PRNGKeyArray) -> Float[Array, "... action_dim"]:
        """Sample from the Beta distribution.

        Args:
            key: PRNG key.

        Returns:
            Samples in (0, 1), shape ``(..., action_dim)``.
        """
        return jax.random.beta(key, self.alpha, self.beta)


class ActorCritic(eqx.Module):
    """Actor-critic network with Beta action distribution.

    The actor outputs concentration parameters for a Beta distribution
    over [0, 1] actions. The critic estimates state value.

    Attributes:
        actor: MLP mapping observations to 2*action_dim Beta parameters.
        critic: MLP mapping observations to scalar value estimate.
        obs_dim: Observation dimension (static).
        action_dim: Action dimension (static).
    """

    actor: eqx.nn.MLP
    critic: eqx.nn.MLP
    obs_dim: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        key_actor, key_critic = jax.random.split(key)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.actor = eqx.nn.MLP(
            obs_dim,
            2 * action_dim,
            hidden_dim,
            hidden_layers,
            activation=jax.nn.tanh,
            key=key_actor,
        )
        self.critic = eqx.nn.MLP(
            obs_dim,
            1,
            hidden_dim,
            hidden_layers,
            activation=jax.nn.tanh,
            key=key_critic,
        )

    def _apply_mlp(self, mlp: eqx.nn.MLP, obs: Float[Array, "... obs_dim"]) -> Array:
        """Apply MLP, handling batched inputs via vmap."""
        if obs.ndim == 1:
            return mlp(obs)
        return jax.vmap(mlp)(obs)

    def forward(
        self, obs: Float[Array, "... obs_dim"]
    ) -> tuple[
        Float[Array, "... action_dim"],
        Float[Array, "... action_dim"],
        Float[Array, "..."],
    ]:
        """Compute raw Beta parameters and value estimate.

        Args:
            obs: Observation array.

        Returns:
            Tuple of (alpha, beta, value).
        """
        params = self._apply_mlp(self.actor, obs)
        alpha = jax.nn.softplus(params[..., : self.action_dim]) + 1.0
        beta = jax.nn.softplus(params[..., self.action_dim :]) + 1.0
        value = self._apply_mlp(self.critic, obs).squeeze(-1)
        return alpha, beta, value

    def dist_and_value(
        self, obs: Float[Array, "... obs_dim"]
    ) -> tuple[BetaParams, Float[Array, "..."]]:
        """Get action distribution parameters and value estimate.

        Args:
            obs: Observation array.

        Returns:
            Tuple of (BetaParams, value).
        """
        alpha, beta, value = self.forward(obs)
        return BetaParams(alpha, beta), value

    def sample_action(
        self, obs: Float[Array, " obs_dim"], key: PRNGKeyArray
    ) -> tuple[Float[Array, " action_dim"], Float[Array, ""], Float[Array, ""]]:
        """Sample an action from the policy.

        Args:
            obs: Single observation, shape ``(obs_dim,)``.
            key: PRNG key.

        Returns:
            Tuple of (action, log_prob, value).
        """
        dist, value = self.dist_and_value(obs)
        action = dist.sample(key)
        action = jnp.clip(action, 1e-4, 1.0 - 1e-4)
        log_prob = jnp.sum(dist.log_prob(action), axis=-1)
        return action, log_prob, value
