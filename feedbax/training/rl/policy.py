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
        # Keep finite scores if the floating-point sampler rounds to an endpoint.
        limits = jnp.finfo(x.dtype)
        x = jnp.clip(x, limits.tiny, 1.0 - limits.epsneg)
        return self._log_density(jnp.log(x), jnp.log1p(-x))

    def _log_density(
        self,
        log_x: Float[Array, "... action_dim"],
        log_one_minus_x: Float[Array, "... action_dim"],
    ) -> Float[Array, "... action_dim"]:
        return (
            (self.alpha - 1.0) * log_x
            + (self.beta - 1.0) * log_one_minus_x
            - jax.scipy.special.betaln(self.alpha, self.beta)
        )

    def joint_log_prob(self, x: Float[Array, "... action_dim"]) -> Float[Array, "..."]:
        """Log density of the full action under independent Beta coordinates."""
        return jnp.sum(self.log_prob(x), axis=-1)

    def joint_log_prob_from_logits(
        self, logits: Float[Array, "... action_dim"]
    ) -> Float[Array, "..."]:
        """Beta log density at the draw represented by logits, in action coordinates."""
        log_x = jax.nn.log_sigmoid(logits)
        log_one_minus_x = jax.nn.log_sigmoid(-logits)
        return jnp.sum(self._log_density(log_x, log_one_minus_x), axis=-1)

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
            Samples in [0, 1], shape ``(..., action_dim)``, with endpoints possible
            from floating-point rounding.
        """
        return _beta_sample_from_logits(self.sample_logits(key))

    def sample_logits(self, key: PRNGKeyArray) -> Float[Array, "... action_dim"]:
        """Sample a Beta draw in logit coordinates to retain both tails."""
        key_a, key_b = jax.random.split(key)
        shape = jnp.broadcast_shapes(self.alpha.shape, self.beta.shape)
        return jax.random.loggamma(key_a, self.alpha, shape) - jax.random.loggamma(
            key_b, self.beta, shape,
        )


def _beta_sample_from_logits(
    logits: Float[Array, "... action_dim"],
) -> Float[Array, "... action_dim"]:
    # Match jax.random.beta's scaled gamma ratio for the same keys and dtype.
    gamma_a = jnp.exp(jnp.minimum(logits, 0.0))
    gamma_b = jnp.exp(jnp.minimum(-logits, 0.0))
    return gamma_a / (gamma_a + gamma_b)


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
        dist, value = self.dist_and_value(obs)
        return dist.alpha, dist.beta, value

    def dist_and_value(
        self,
        obs: Float[Array, "... obs_dim"],
        latent_noise: Float[Array, "... latent_dim"] | None = None,
    ) -> tuple[BetaParams, Float[Array, "..."]]:
        """Return the Beta law conditional on observations and optional latent noise.

        A single noise vector may be shared across a batch. Noise enters at the
        input to the actor output layer, leaving the critic unchanged.
        """
        if latent_noise is None:
            params = self._apply_mlp(self.actor, obs)
        else:
            flat_obs = obs.reshape(-1, obs.shape[-1])
            flat_noise = jnp.broadcast_to(
                latent_noise, obs.shape[:-1] + (latent_noise.shape[-1],),
            ).reshape(-1, latent_noise.shape[-1])
            params = jax.vmap(self._actor_with_noise)(flat_obs, flat_noise)
            params = params.reshape(obs.shape[:-1] + (params.shape[-1],))

        alpha = jax.nn.softplus(params[..., : self.action_dim]) + 1.0
        beta = jax.nn.softplus(params[..., self.action_dim :]) + 1.0
        value = self._apply_mlp(self.critic, obs).squeeze(-1)
        return BetaParams(alpha, beta), value

    def sample_action(
        self, obs: Float[Array, " obs_dim"], key: PRNGKeyArray
    ) -> tuple[Float[Array, " action_dim"], Float[Array, ""], Float[Array, ""]]:
        """Sample an action from the policy.

        Args:
            obs: Single observation, shape ``(obs_dim,)``.
            key: PRNG key.

        Returns:
            Tuple of (clipped action, log density at that action, value).
        """
        return self._sample_clipped_action(obs, key, None)

    def _sample_clipped_action(
        self,
        obs: Float[Array, " obs_dim"],
        key: PRNGKeyArray,
        latent_noise: Float[Array, " latent_dim"] | None,
    ) -> tuple[Float[Array, " action_dim"], Float[Array, ""], Float[Array, ""]]:
        dist, value = self.dist_and_value(obs, latent_noise)
        action = _clip_action(dist.sample(key))
        return action, dist.joint_log_prob(action), value

    def _draw_action_logits(
        self,
        obs: Float[Array, " obs_dim"],
        key: PRNGKeyArray,
        latent_noise: Float[Array, " latent_dim"] | None,
    ) -> tuple[Float[Array, " action_dim"], Float[Array, ""], Float[Array, ""]]:
        dist, value = self.dist_and_value(obs, latent_noise)
        logits = jax.lax.stop_gradient(dist.sample_logits(key))
        return logits, dist.joint_log_prob_from_logits(logits), value

    def _actor_with_noise(
        self, obs: Float[Array, " obs_dim"], latent_noise: Float[Array, " latent_dim"],
    ) -> Float[Array, " two_action_dim"]:
        """Inject noise at the input to the actor output layer."""
        hidden = obs
        for layer in self.actor.layers[:-1]:
            hidden = self.actor.activation(layer(hidden))
        return self.actor.layers[-1](hidden + latent_noise)


def _clip_action(action: Float[Array, "... action_dim"]) -> Float[Array, "... action_dim"]:
    return jnp.clip(action, 1e-4, 1.0 - 1e-4)


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
    return policy.dist_and_value(obs, jnp.where(use_lattice, latent_noise, 0.0))


def sample_action_with_noise(
    policy: ActorCritic,
    obs: Float[Array, " obs_dim"],
    key: PRNGKeyArray,
    latent_noise: Float[Array, " hidden_dim"],
    use_lattice: bool = True,
) -> tuple[Float[Array, " action_dim"], Float[Array, ""], Float[Array, ""]]:
    """Sample an action from the policy with LATTICE latent-space noise.

    Returns a clipped action, matching ``ActorCritic.sample_action``.

    Args:
        policy: Actor-critic policy.
        obs: Single observation, shape ``(obs_dim,)``.
        key: PRNG key for action sampling.
        latent_noise: Noise vector, shape ``(hidden_dim,)``.
        use_lattice: Whether to apply latent noise.

    Returns:
        Tuple of (clipped action, log density at that action, value).
    """
    noise = jnp.where(use_lattice, latent_noise, 0.0)
    return policy._sample_clipped_action(obs, key, noise)
