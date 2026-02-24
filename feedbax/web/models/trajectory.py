"""Pydantic models for trajectory data endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DatasetInfo(BaseModel):
    """Summary info for a trajectory dataset file."""

    name: str
    file_size: int
    modified: float


class TrajectoryMetadata(BaseModel):
    """Metadata extracted from a loaded trajectory dataset."""

    n_trajectories: int
    n_timesteps: int
    n_joints: int
    n_muscles: int
    n_bodies: int
    rollouts_per_body: int
    task_types: list[int]
    body_indices: list[int]
    angle_convention: str = "relative_radians"


class FilterResult(BaseModel):
    """Result of filtering trajectories by body index or task type."""

    indices: list[int]
    count: int
