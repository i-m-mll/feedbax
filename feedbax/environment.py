"""Environment and task protocols for feedbax training.

Defines the minimal shared interface that both supervised (TaskTrainer) and
RL training paths use to interact with environments and tasks.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    Optional,
    runtime_checkable,
    Protocol,
)

from jaxtyping import Array, PRNGKeyArray, PyTree

if TYPE_CHECKING:
    from feedbax.loss import TermTree
    from feedbax.misc import BatchInfo
    from feedbax.task import TaskTrialSpec


class EnvironmentStep(NamedTuple):
    """The result of a single environment step.

    Attributes:
        obs: Observation to feed to the agent. Can be any JAX PyTree.
        target: Supervision target for the current timestep. None for RL
            environments that do not use per-step supervised targets.
        intervene: Intervention parameters for the current timestep. None if
            no intervention is active.
        reward: Per-step scalar reward signal. None for supervised training
            environments.
        done: Episode termination flag. None for supervised training
            environments (episodes always run for a fixed number of steps).
    """

    obs: PyTree
    target: Optional[PyTree]
    intervene: Optional[PyTree]
    reward: Optional[Array]
    done: Optional[Array]


@runtime_checkable
class EnvironmentProtocol(Protocol):
    """Per-step environment interface.

    Describes the minimal contract that an environment must fulfill to be
    compatible with both supervised and RL training loops.
    """

    def init_env_state(
        self,
        trial_spec: Any,
        key: PRNGKeyArray,
    ) -> Optional[PyTree]:
        """Initialize environment state for a new episode.

        Args:
            trial_spec: The trial specification for this episode.
            key: A JAX random key.

        Returns:
            The initial environment state, or None for open-loop environments
            that carry no mutable state between timesteps.
        """
        ...

    def step_env(
        self,
        env_state: Optional[PyTree],
        action: Optional[PyTree],
        key: PRNGKeyArray,
        t: int,
        trial_spec: Any,
    ) -> EnvironmentStep:
        """Advance the environment by one timestep.

        Args:
            env_state: The current environment state (None for open-loop
                environments).
            action: The agent's action for this timestep (None if no action is
                required, e.g. during open-loop replay).
            key: A JAX random key.
            t: The current timestep index (0-indexed).
            trial_spec: The trial specification for this episode.

        Returns:
            An EnvironmentStep containing the next observation and optional
            reward, done flag, target, and intervention parameters.
        """
        ...


@runtime_checkable
class TaskProtocol(Protocol):
    """Task interface for trainers.

    Describes the contract that a task object must fulfill so that a trainer
    (TaskTrainer or an RL trainer) can drive a training loop without
    knowing the concrete task type.
    """

    def sample_trial(
        self,
        key: PRNGKeyArray,
        batch_info: Any,
    ) -> "TaskTrialSpec":
        """Generate a trial specification for a single training trial.

        Args:
            key: A JAX random key.
            batch_info: Information about the current training batch (e.g.
                batch size, current iteration). May be None.

        Returns:
            A TaskTrialSpec describing the trial.
        """
        ...

    def episode_length(self, trial_spec: Any) -> int:
        """Return the number of timesteps in a trial.

        Args:
            trial_spec: The trial specification whose length to query.

        Returns:
            The number of timesteps.
        """
        ...

    def compute_loss(
        self,
        states: PyTree,
        trial_spec: Any,
        model: Any,
    ) -> "TermTree":
        """Compute training loss from a completed trajectory.

        Args:
            states: The model state history collected during the episode.
            trial_spec: The trial specification used to generate the episode.
            model: The model that produced the states. Used by some loss
                functions to access model parameters (e.g. for regularisation).

        Returns:
            A TermTree containing scalar loss values and their components.
        """
        ...
