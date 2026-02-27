"""Adapter that wraps the functional RL environment to satisfy EnvironmentProtocol.

Provides ``RLEnvironmentAdapter``, an equinox Module holding an ``AbstractPlant``
and ``RLEnvConfig``.  It exposes the two methods required by ``EnvironmentProtocol``:

- ``init_env_state(trial_spec, key)`` — delegates to ``rl_env_reset``
- ``step_env(env_state, action, key, t, trial_spec)`` — delegates to ``rl_env_step``
  and maps the functional return values to an ``EnvironmentStep`` NamedTuple.

The import of the protocol types is guarded by ``TYPE_CHECKING`` so the module
remains importable even before ``feedbax/environment.py`` exists.  The class
satisfies the protocol structurally (duck typing) at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray

from feedbax.mechanics.plant import AbstractPlant
from feedbax.training.rl.env import (
    RLEnvConfig,
    RLEnvState,
    rl_env_reset,
    rl_env_step,
)
from feedbax.training.rl.tasks import TaskParams

if TYPE_CHECKING:
    from feedbax.environment import EnvironmentProtocol, EnvironmentStep  # noqa: F401


class RLEnvironmentAdapter(eqx.Module):
    """Wraps the functional RL environment as an ``EnvironmentProtocol``-compatible object.

    Holds an ``AbstractPlant`` and ``RLEnvConfig`` so that JIT/vmap-safe pure
    functions (``rl_env_reset``, ``rl_env_step``) can be called with consistent
    configuration from an object-oriented interface.

    The ``trial_spec`` argument in both methods is expected to be a ``TaskParams``
    instance (the RL equivalent of a task trial specification).

    Attributes:
        plant: The plant model used for physics simulation.
        config: Static environment configuration.
    """

    plant: AbstractPlant
    config: RLEnvConfig

    def init_env_state(
        self,
        trial_spec: TaskParams,
        key: PRNGKeyArray,
    ) -> RLEnvState:
        """Reset the environment and return an initial ``RLEnvState``.

        Maps to ``EnvironmentProtocol.init_env_state``.

        Args:
            trial_spec: ``TaskParams`` describing the episode (start/end
                positions, task type, etc.).  Corresponds to the
                ``EnvironmentProtocol`` ``trial_spec`` argument.
            key: PRNG key for state initialisation.

        Returns:
            Initial ``RLEnvState``.
        """
        return rl_env_reset(self.plant, self.config, trial_spec, key)

    def step_env(
        self,
        env_state: RLEnvState,
        action: Float[Array, " n_muscles"],
        key: PRNGKeyArray,
        t: int,
        trial_spec: TaskParams,
    ) -> "EnvironmentStep":
        """Step the environment forward and return an ``EnvironmentStep``.

        Maps to ``EnvironmentProtocol.step_env``.  The ``key`` and ``t``
        arguments are accepted for protocol compatibility; ``rl_env_step``
        derives the timestep internally from ``env_state.t_index`` and does not
        require a separate PRNG key for the step itself.

        Args:
            env_state: Current ``RLEnvState``.
            action: Muscle excitation commands in ``[0, 1]``, shape
                ``(n_muscles,)``.
            key: PRNG key (unused by the functional step; accepted for
                protocol compatibility).
            t: Current timestep index (unused; ``env_state.t_index`` is the
                authoritative source).
            trial_spec: ``TaskParams`` for the current episode (unused by
                ``rl_env_step``, which reads task from ``env_state.task``).

        Returns:
            ``EnvironmentStep`` with fields:
            - ``obs``: observation vector from the new state
            - ``target``: ``None`` (RL has no supervised target)
            - ``intervene``: ``None``
            - ``reward``: scalar reward
            - ``done``: episode-done flag (0.0 or 1.0)
        """
        # Import here to avoid a hard dependency at module load time.
        # feedbax/environment.py may not exist yet (parallel partition).
        from feedbax.environment import EnvironmentStep  # type: ignore[import]

        new_state, obs, reward, done = rl_env_step(
            self.plant, self.config, env_state, action
        )
        return EnvironmentStep(
            obs=obs,
            target=None,
            intervene=None,
            reward=reward,
            done=done,
        )
