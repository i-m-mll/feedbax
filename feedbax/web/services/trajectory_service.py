"""Service for loading and querying NPZ trajectory datasets."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from fastapi import HTTPException

from feedbax.web.config import TRAJECTORIES_DIR
from feedbax.web.models.trajectory import DatasetInfo, TrajectoryMetadata

_VALID_DATASET_RE = re.compile(r'^[A-Za-z0-9_-]+$')

# Fields that form a single trajectory row (indexed by trajectory index).
_TRAJECTORY_FIELDS = frozenset({
    'joint_angles',
    'joint_velocities',
    'muscle_activations',
    'effector_pos',
    'task_target',
    'timestamps',
})

# Per-trajectory scalar/1-D metadata fields.
_SCALAR_FIELDS = frozenset({
    'body_preset_flat',
    'body_idx',
    'task_type',
})

_ALL_RETURNABLE_FIELDS = _TRAJECTORY_FIELDS | _SCALAR_FIELDS


class TrajectoryService:
    """Loads NPZ trajectory files, caches handles, and serves queries."""

    def __init__(self, trajectories_dir: Path = TRAJECTORIES_DIR) -> None:
        self._dir = trajectories_dir
        # (resolved_path, mtime) -> NpzFile handle
        self._cache: dict[tuple[str, float], np.lib.npyio.NpzFile] = {}
        # dataset name -> TrajectoryMetadata
        self._metadata_cache: dict[str, TrajectoryMetadata] = {}
        # dataset name -> {body_idx: ndarray, task_type: ndarray}
        self._filter_cache: dict[str, dict[str, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_datasets(self) -> list[DatasetInfo]:
        """Scan the trajectories directory for .npz files."""
        if not self._dir.is_dir():
            return []
        results: list[DatasetInfo] = []
        for path in sorted(self._dir.glob('*.npz')):
            stat = path.stat()
            results.append(DatasetInfo(
                name=path.stem,
                file_size=stat.st_size,
                modified=stat.st_mtime,
            ))
        return results

    def get_metadata(self, dataset: str) -> TrajectoryMetadata:
        """Return metadata for *dataset*, loading and caching as needed."""
        self._validate_dataset_name(dataset)
        npz = self._load(dataset)

        if dataset in self._metadata_cache:
            return self._metadata_cache[dataset]

        joint_angles = npz['joint_angles']
        n_trajectories, n_timesteps, n_joints = joint_angles.shape
        n_muscles = npz['muscle_activations'].shape[2]

        n_bodies = int(npz['n_bodies']) if 'n_bodies' in npz else 0
        rollouts_per_body = int(npz['rollouts_per_body']) if 'rollouts_per_body' in npz else 0

        body_indices = npz['body_idx'] if 'body_idx' in npz else np.array([], dtype=np.int64)
        task_types = npz['task_type'] if 'task_type' in npz else np.array([], dtype=np.int64)

        meta = TrajectoryMetadata(
            n_trajectories=n_trajectories,
            n_timesteps=n_timesteps,
            n_joints=n_joints,
            n_muscles=n_muscles,
            n_bodies=n_bodies,
            rollouts_per_body=rollouts_per_body,
            task_types=sorted(set(int(v) for v in task_types)),
            body_indices=sorted(set(int(v) for v in body_indices)),
        )
        self._metadata_cache[dataset] = meta

        # Also cache the filter arrays while we have them.
        self._filter_cache[dataset] = {
            'body_idx': np.asarray(body_indices),
            'task_type': np.asarray(task_types),
        }

        return meta

    def get_trajectory(
        self,
        dataset: str,
        index: int,
        fields: list[str] | None = None,
    ) -> dict:
        """Return a single trajectory as a plain dict ready for JSON serialisation.

        Uses raw dict construction instead of Pydantic to avoid serialisation
        overhead on large nested lists.
        """
        self._validate_dataset_name(dataset)
        npz = self._load(dataset)

        # Validate index.
        n_trajectories = npz['joint_angles'].shape[0]
        if index < 0 or index >= n_trajectories:
            raise HTTPException(
                status_code=404,
                detail=f'Trajectory index {index} out of range [0, {n_trajectories})',
            )

        # Determine which fields to return.
        requested = _ALL_RETURNABLE_FIELDS
        if fields is not None:
            unknown = set(fields) - _ALL_RETURNABLE_FIELDS
            if unknown:
                raise HTTPException(
                    status_code=400,
                    detail=f'Unknown fields: {", ".join(sorted(unknown))}',
                )
            requested = set(fields)

        result: dict = {}
        for field in requested:
            arr = npz[field]
            if field in _TRAJECTORY_FIELDS:
                result[field] = arr[index].tolist()
            elif field in ('body_idx', 'task_type'):
                result[field] = int(arr[index])
            else:
                # body_preset_flat — 1-D per trajectory
                result[field] = arr[index].tolist()

        return result

    def filter_trajectories(
        self,
        dataset: str,
        body_idx: int | None = None,
        task_type: int | None = None,
    ) -> tuple[list[int], int]:
        """Return indices of trajectories matching the given filters."""
        self._validate_dataset_name(dataset)

        # Ensure filter arrays are cached.
        if dataset not in self._filter_cache:
            self.get_metadata(dataset)

        fc = self._filter_cache[dataset]
        n = len(fc['body_idx'])
        mask = np.ones(n, dtype=bool)

        if body_idx is not None:
            mask &= fc['body_idx'] == body_idx
        if task_type is not None:
            mask &= fc['task_type'] == task_type

        indices = np.nonzero(mask)[0].tolist()
        return indices, len(indices)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_dataset_name(self, name: str) -> None:
        if not _VALID_DATASET_RE.match(name):
            raise HTTPException(
                status_code=400,
                detail='Invalid dataset name. Use alphanumeric characters, hyphens, '
                       'and underscores only.',
            )

    def _resolve_path(self, dataset: str) -> Path:
        return self._dir / f'{dataset}.npz'

    def _load(self, dataset: str) -> np.lib.npyio.NpzFile:
        """Load (or return cached) NpzFile handle for *dataset*.

        Evicts stale cache entries when the file's mtime changes.
        """
        path = self._resolve_path(dataset)
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f'Dataset "{dataset}" not found')

        mtime = path.stat().st_mtime
        key = (str(path), mtime)

        if key in self._cache:
            return self._cache[key]

        # Evict any older handle for the same path but different mtime.
        stale = [k for k in self._cache if k[0] == str(path) and k[1] != mtime]
        for k in stale:
            old = self._cache.pop(k)
            old.close()
        # Also invalidate derived caches.
        self._metadata_cache.pop(dataset, None)
        self._filter_cache.pop(dataset, None)

        handle = np.load(str(path), allow_pickle=False)
        self._cache[key] = handle
        return handle
