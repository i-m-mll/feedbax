"""Actor-critic policy for PPO training.

Uses a Beta distribution for bounded [0, 1] actions (muscle excitations),
implemented with native JAX (no distrax dependency).

Includes LATTICE (Latent Action Temporally-Correlated Exploration) noise for
structured exploration in latent space rather than action space.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray


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


class LatticeNoiseState(eqx.Module):
    """State for LATTICE temporally-correlated latent-space exploration noise.

    LATTICE injects structured noise into the policy's hidden representation
    rather than the action space, producing temporally correlated exploration
    that is more effective for musculoskeletal control.

    Attributes:
        noise: Current latent noise vector, shape ``(hidden_dim,)``.
        steps_since_resample: Counter tracking steps since last noise resample.
    """

    noise: Float[Array, " hidden_dim"]
    steps_since_resample: Int[Array, ""]


def init_lattice_noise(hidden_dim: int, key: PRNGKeyArray) -> LatticeNoiseState:
    """Initialize LATTICE noise state.

    Samples initial noise from N(0, scale) where scale = 0.1 * hidden_dim^(-0.5),
    keeping the noise magnitude appropriately scaled to the hidden dimension.

    Args:
        hidden_dim: Dimension of the hidden representation to perturb.
        key: PRNG key.

    Returns:
        Initial ``LatticeNoiseState``.
    """
    scale = 0.1 * hidden_dim ** (-0.5)
    noise = jax.random.normal(key, (hidden_dim,)) * scale
    return LatticeNoiseState(
        noise=noise,
        steps_since_resample=jnp.array(0, dtype=jnp.int32),
    )


def maybe_resample_noise(
    state: LatticeNoiseState,
    key: PRNGKeyArray,
    resample_interval: int = 8,
) -> LatticeNoiseState:
    """Conditionally resample LATTICE noise based on step counter.

    Resamples the noise vector when ``steps_since_resample >= resample_interval``,
    otherwise increments the counter. Uses ``jax.lax.cond`` for JIT compatibility.

    Args:
        state: Current noise state.
        key: PRNG key (consumed only on resample).
        resample_interval: Number of steps between resamples.

    Returns:
        Updated ``LatticeNoiseState`` with either fresh noise or incremented counter.
    """
    hidden_dim = state.noise.shape[0]
    scale = 0.1 * hidden_dim ** (-0.5)

    def _resample(_: None) -> LatticeNoiseState:
        new_noise = jax.random.normal(key, (hidden_dim,)) * scale
        return LatticeNoiseState(
            noise=new_noise,
            steps_since_resample=jnp.array(0, dtype=jnp.int32),
        )

    def _increment(_: None) -> LatticeNoiseState:
        return LatticeNoiseState(
            noise=state.noise,
            steps_since_resample=state.steps_since_resample + 1,
        )

    return jax.lax.cond(
        state.steps_since_resample >= resample_interval,
        _resample,
        _increment,
        None,
    )


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
        obs_batched = jnp.atleast_2d(obs)
        out = jax.vmap(mlp)(obs_batched)
        return out.reshape(obs.shape[:-1] + out.shape[1:])

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

    def _actor_hidden(self, obs: Float[Array, " obs_dim"]) -> Float[Array, " hidden_dim"]:
        """Run obs through all actor layers except the last, returning hidden repr.

        Applies activation after each hidden layer, matching ``eqx.nn.MLP`` semantics.

        Args:
            obs: Single observation, shape ``(obs_dim,)``.

        Returns:
            Hidden representation before the final linear layer.
        """
        h = obs
        for layer in self.actor.layers[:-1]:
            h = self.actor.activation(layer(h))
        return h

    def _actor_output_from_hidden(
        self, hidden: Float[Array, " hidden_dim"]
    ) -> Float[Array, " two_action_dim"]:
        """Apply the actor's final linear layer to a hidden representation.

        Args:
            hidden: Hidden representation, shape ``(hidden_dim,)``.

        Returns:
            Raw output params, shape ``(2 * action_dim,)``.
        """
        return self.actor.layers[-1](hidden)


def forward_with_latent_noise(
    policy: ActorCritic,
    obs: Float[Array, " obs_dim"],
    latent_noise: Float[Array, " hidden_dim"],
    use_lattice: bool = True,
) -> tuple[BetaParams, Float[Array, ""]]:
    """Forward pass with LATTICE noise injected into the actor's hidden layer.

    Runs the observation through all actor hidden layers, adds ``latent_noise``
    to the penultimate representation, then passes through the final layer
    to produce Beta distribution parameters. The critic is unaffected by noise.

    Args:
        policy: Actor-critic policy.
        obs: Single observation, shape ``(obs_dim,)``.
        latent_noise: Noise vector, shape ``(hidden_dim,)``. Use zeros for
            behavior identical to the standard forward pass.
        use_lattice: Whether to apply latent noise.

    Returns:
        Tuple of (BetaParams, value).
    """
    # Actor: inject noise into hidden representation
    hidden = policy._actor_hidden(obs)
    hidden_noisy = jnp.where(jnp.asarray(use_lattice), hidden + latent_noise, hidden)
    params = policy._actor_output_from_hidden(hidden_noisy)
    alpha = jax.nn.softplus(params[..., : policy.action_dim]) + 1.0
    beta = jax.nn.softplus(params[..., policy.action_dim :]) + 1.0

    # Critic: no noise
    value = policy.critic(obs).squeeze(-1)

    return BetaParams(alpha, beta), value


def sample_action_with_noise(
    policy: ActorCritic,
    obs: Float[Array, " obs_dim"],
    key: PRNGKeyArray,
    latent_noise: Float[Array, " hidden_dim"],
    use_lattice: bool = True,
) -> tuple[Float[Array, " action_dim"], Float[Array, ""], Float[Array, ""]]:
    """Sample an action from the policy with LATTICE latent-space noise.

    Wrapper around ``forward_with_latent_noise`` that samples from the
    resulting Beta distribution, matching the signature of
    ``ActorCritic.sample_action``.

    Args:
        policy: Actor-critic policy.
        obs: Single observation, shape ``(obs_dim,)``.
        key: PRNG key for action sampling.
        latent_noise: Noise vector, shape ``(hidden_dim,)``.
        use_lattice: Whether to apply latent noise.

    Returns:
        Tuple of (action, log_prob, value).
    """
    dist, value = forward_with_latent_noise(policy, obs, latent_noise, use_lattice)
    action = dist.sample(key)
    action = jnp.clip(action, 1e-4, 1.0 - 1e-4)
    log_prob = jnp.sum(dist.log_prob(action), axis=-1)
    return action, log_prob, value
