"""Reinforcement learning training infrastructure.

Provides PPO training, task specifications, reward functions, and
functional RL environments that work with any feedbax AbstractPlant.
"""

from feedbax.training.rl.adapter import RLEnvironmentAdapter

__all__ = ["RLEnvironmentAdapter"]
