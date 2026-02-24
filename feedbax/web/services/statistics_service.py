"""Service for computing trajectory statistics from NPZ datasets."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from feedbax.web.models.statistics import (
    DiagnosticCheck,
    DiagnosticsResponse,
    GroupStatistics,
    HistogramBin,
    HistogramGroup,
    HistogramResponse,
    MetricSummary,
    ScatterPoint,
    ScatterResponse,
    StatisticsResponse,
    TimeseriesPercentiles,
    TimeseriesResponse,
)
from feedbax.web.services.trajectory_service import TrajectoryService

logger = logging.getLogger(__name__)

# Map task_type integers to human-readable labels.
TASK_TYPE_LABELS: dict[int, str] = {
    0: "Reach",
    1: "Hold",
    2: "Track",
    3: "Swing",
}

# Scalar metrics that can be computed per-trajectory.
SCALAR_METRICS = frozenset({
    'final_distance',
    'effort',
    'convergence_time',
    'joint_range_of_motion',
    'peak_activation',
    'movement_amplitude',
    'success_rate',
})

# Time-varying metrics that support percentile timeseries.
TIMESERIES_METRICS = frozenset({
    'distance_to_target',
    'muscle_effort',
})

# Distance threshold for convergence and success.
_CONVERGE_THRESHOLD = 0.05

# Maximum number of timesteps to return for timeseries.
_MAX_TIMESERIES_POINTS = 200


class StatisticsService:
    """Computes statistics over trajectory datasets.

    Delegates NPZ data loading to a ``TrajectoryService`` and caches computed
    scalar metrics keyed by ``(dataset, mtime, group_by)``.
    """

    def __init__(self, trajectory_service: TrajectoryService) -> None:
        self._traj = trajectory_service
        # (dataset, mtime) -> dict[metric_name, ndarray(n_trajectories,)]
        self._scalar_cache: dict[tuple[str, float], dict[str, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summary(self, dataset: str, group_by: str = "none") -> StatisticsResponse:
        """Compute grouped summary statistics for all scalar metrics."""
        scalars = self._get_scalars(dataset)
        npz = self._load_npz(dataset)
        groups = self._build_groups(npz, scalars, group_by)
        return StatisticsResponse(dataset=dataset, group_by=group_by, groups=groups)

    def timeseries(
        self,
        dataset: str,
        metric: str = "distance_to_target",
        group_by: str = "none",
    ) -> TimeseriesResponse:
        """Compute percentile bands for a time-varying metric."""
        if metric not in TIMESERIES_METRICS:
            raise ValueError(
                f"Unknown timeseries metric '{metric}'. "
                f"Available: {sorted(TIMESERIES_METRICS)}"
            )
        npz = self._load_npz(dataset)
        ts_values = self._compute_timeseries_metric(npz, metric)  # (n_traj, n_steps)
        groups_map = self._group_indices(npz, group_by)

        n_steps = ts_values.shape[1]
        if n_steps > _MAX_TIMESERIES_POINTS:
            step_indices = np.linspace(
                0, n_steps - 1, _MAX_TIMESERIES_POINTS, dtype=int,
            )
        else:
            step_indices = np.arange(n_steps)

        ts_downsampled = ts_values[:, step_indices]
        timesteps = step_indices.tolist()

        series: list[TimeseriesPercentiles] = []
        for key, label, indices in groups_map:
            subset = ts_downsampled[indices]
            series.append(TimeseriesPercentiles(
                group_key=key,
                group_label=label,
                timesteps=timesteps,
                p50=_nanpercentile_list(subset, 50),
                p25=_nanpercentile_list(subset, 25),
                p75=_nanpercentile_list(subset, 75),
                p05=_nanpercentile_list(subset, 5),
                p95=_nanpercentile_list(subset, 95),
            ))

        return TimeseriesResponse(
            dataset=dataset, metric=metric, group_by=group_by, series=series,
        )

    def histogram(
        self,
        dataset: str,
        metric: str = "final_distance",
        group_by: str = "none",
        bins: int = 30,
    ) -> HistogramResponse:
        """Bin a scalar metric for each group."""
        if metric not in SCALAR_METRICS:
            raise ValueError(
                f"Unknown scalar metric '{metric}'. "
                f"Available: {sorted(SCALAR_METRICS)}"
            )
        scalars = self._get_scalars(dataset)
        npz = self._load_npz(dataset)
        values = scalars[metric]
        groups_map = self._group_indices(npz, group_by)

        # Compute global bin edges so groups are comparable.
        finite_vals = values[np.isfinite(values)]
        if len(finite_vals) == 0:
            edges = np.linspace(0, 1, bins + 1)
        else:
            lo = float(np.min(finite_vals))
            hi = float(np.max(finite_vals))
            # Bug: c722539 -- identical min/max produces non-monotonic edges
            if lo == hi:
                lo -= 0.5
                hi += 0.5
            edges = np.linspace(lo, hi, bins + 1)

        result_groups: list[HistogramGroup] = []
        for key, label, indices in groups_map:
            subset = values[indices]
            counts, _ = np.histogram(subset[np.isfinite(subset)], bins=edges)
            hist_bins = [
                HistogramBin(lo=float(edges[i]), hi=float(edges[i + 1]), count=int(counts[i]))
                for i in range(len(counts))
            ]
            result_groups.append(HistogramGroup(
                group_key=key, group_label=label, bins=hist_bins,
            ))

        return HistogramResponse(
            dataset=dataset, metric=metric, group_by=group_by, groups=result_groups,
        )

    def scatter(
        self,
        dataset: str,
        x_metric: str = "effort",
        y_metric: str = "final_distance",
    ) -> ScatterResponse:
        """Return per-trajectory (x, y, body_idx, task_type) for two scalar metrics."""
        for m in (x_metric, y_metric):
            if m not in SCALAR_METRICS:
                raise ValueError(
                    f"Unknown scalar metric '{m}'. "
                    f"Available: {sorted(SCALAR_METRICS)}"
                )
        scalars = self._get_scalars(dataset)
        npz = self._load_npz(dataset)

        x_vals = scalars[x_metric]
        y_vals = scalars[y_metric]
        body_idx = npz['body_idx'] if 'body_idx' in npz else np.zeros(len(x_vals), dtype=int)
        task_type = npz['task_type'] if 'task_type' in npz else np.zeros(len(x_vals), dtype=int)

        # Filter out non-finite points.
        mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        points = [
            ScatterPoint(
                x=float(x_vals[i]),
                y=float(y_vals[i]),
                body_idx=int(body_idx[i]),
                task_type=int(task_type[i]),
            )
            for i in np.nonzero(mask)[0]
        ]

        return ScatterResponse(
            dataset=dataset, x_metric=x_metric, y_metric=y_metric, points=points,
        )

    def diagnostics(self, dataset: str) -> DiagnosticsResponse:
        """Run diagnostic checks on a dataset and return pass/warn/fail results."""
        scalars = self._get_scalars(dataset)
        npz = self._load_npz(dataset)
        checks = [
            self._check_movement_collapse(scalars),
            self._check_target_indifference(npz, scalars),
            self._check_identical_behavior(scalars),
            self._check_effort_trap(npz, scalars),
            self._check_non_convergence(scalars),
        ]
        return DiagnosticsResponse(dataset=dataset, checks=checks)

    # ------------------------------------------------------------------
    # Scalar metric computation
    # ------------------------------------------------------------------

    def _get_scalars(self, dataset: str) -> dict[str, np.ndarray]:
        """Return (cached) per-trajectory scalar metrics for *dataset*."""
        mtime = self._get_mtime(dataset)
        key = (dataset, mtime)
        if key in self._scalar_cache:
            return self._scalar_cache[key]

        # Evict stale entries for this dataset.
        stale = [k for k in self._scalar_cache if k[0] == dataset and k[1] != mtime]
        for k in stale:
            del self._scalar_cache[k]

        npz = self._load_npz(dataset)
        scalars = self._compute_all_scalars(npz)
        self._scalar_cache[key] = scalars
        return scalars

    def _compute_all_scalars(self, npz: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
        """Compute all scalar metrics from the raw NPZ arrays."""
        effector_pos = npz['effector_pos']      # (N, T, D)
        task_target = npz['task_target']         # (N, T, D)
        muscle_act = npz['muscle_activations']   # (N, T, M)
        joint_angles = npz['joint_angles']       # (N, T, J)

        n_traj, n_steps, _ = effector_pos.shape

        # Distance from effector to target at each timestep: (N, T)
        dist_to_target = np.linalg.norm(effector_pos - task_target, axis=-1)

        # final_distance: L2 at last timestep
        final_distance = dist_to_target[:, -1]

        # effort: mean over time of sum-of-squared activations across muscles
        effort = np.mean(np.sum(muscle_act ** 2, axis=-1), axis=1)

        # convergence_time: first timestep where distance < threshold
        below_threshold = dist_to_target < _CONVERGE_THRESHOLD  # (N, T)
        # argmax on bool gives first True; if never True, argmax returns 0
        first_below = np.argmax(below_threshold, axis=1)
        ever_converged = np.any(below_threshold, axis=1)
        convergence_time = np.where(ever_converged, first_below, n_steps).astype(float)

        # joint_range_of_motion: mean across joints of (max - min) over time
        joint_rom = np.mean(
            np.max(joint_angles, axis=1) - np.min(joint_angles, axis=1),
            axis=-1,
        )

        # peak_activation: max activation value per trajectory
        peak_activation = np.max(muscle_act.reshape(n_traj, -1), axis=1)

        # movement_amplitude: total effector path length
        step_diffs = np.diff(effector_pos, axis=1)  # (N, T-1, D)
        step_lengths = np.linalg.norm(step_diffs, axis=-1)  # (N, T-1)
        movement_amplitude = np.sum(step_lengths, axis=1)

        # success_rate: binary per trajectory (1 if final_distance < threshold)
        success_rate = (final_distance < _CONVERGE_THRESHOLD).astype(float)

        return {
            'final_distance': final_distance,
            'effort': effort,
            'convergence_time': convergence_time,
            'joint_range_of_motion': joint_rom,
            'peak_activation': peak_activation,
            'movement_amplitude': movement_amplitude,
            'success_rate': success_rate,
        }

    # ------------------------------------------------------------------
    # Timeseries metric computation
    # ------------------------------------------------------------------

    def _compute_timeseries_metric(
        self, npz: np.lib.npyio.NpzFile, metric: str,
    ) -> np.ndarray:
        """Compute a time-varying metric for all trajectories. Returns (N, T)."""
        if metric == 'distance_to_target':
            return np.linalg.norm(
                npz['effector_pos'] - npz['task_target'], axis=-1,
            )
        elif metric == 'muscle_effort':
            return np.sum(npz['muscle_activations'] ** 2, axis=-1)
        else:
            raise ValueError(f"Unknown timeseries metric: {metric}")

    # ------------------------------------------------------------------
    # Grouping helpers
    # ------------------------------------------------------------------

    def _group_indices(
        self, npz: np.lib.npyio.NpzFile, group_by: str,
    ) -> list[tuple[str, str, np.ndarray]]:
        """Return a list of (group_key, group_label, index_array) tuples.

        Each index_array is an integer array of trajectory indices belonging
        to that group.
        """
        n_traj = npz['joint_angles'].shape[0]

        if group_by == "none":
            return [("all", "All", np.arange(n_traj))]

        if group_by == "task_type":
            if 'task_type' not in npz:
                return [("all", "All", np.arange(n_traj))]
            arr = npz['task_type']
            unique = sorted(set(int(v) for v in arr))
            return [
                (
                    f"task_type={v}",
                    TASK_TYPE_LABELS.get(v, f"Task {v}"),
                    np.nonzero(arr == v)[0],
                )
                for v in unique
            ]

        if group_by == "body_idx":
            if 'body_idx' not in npz:
                return [("all", "All", np.arange(n_traj))]
            arr = npz['body_idx']
            unique = sorted(set(int(v) for v in arr))
            return [
                (
                    f"body_idx={v}",
                    f"Body {v}",
                    np.nonzero(arr == v)[0],
                )
                for v in unique
            ]

        if group_by == "body_x_task":
            # Per-trajectory: no aggregation, every trajectory is its own group.
            body_idx = npz['body_idx'] if 'body_idx' in npz else np.zeros(n_traj, dtype=int)
            task_type = npz['task_type'] if 'task_type' in npz else np.zeros(n_traj, dtype=int)
            return [
                (
                    f"body={int(body_idx[i])}_task={int(task_type[i])}",
                    f"Body {int(body_idx[i])} / {TASK_TYPE_LABELS.get(int(task_type[i]), f'Task {int(task_type[i])}')}",
                    np.array([i]),
                )
                for i in range(n_traj)
            ]

        raise ValueError(
            f"Unknown group_by '{group_by}'. "
            f"Available: none, task_type, body_idx, body_x_task"
        )

    def _build_groups(
        self,
        npz: np.lib.npyio.NpzFile,
        scalars: dict[str, np.ndarray],
        group_by: str,
    ) -> list[GroupStatistics]:
        """Build ``GroupStatistics`` for each group."""
        groups_map = self._group_indices(npz, group_by)
        result: list[GroupStatistics] = []
        for key, label, indices in groups_map:
            metrics: dict[str, MetricSummary] = {}
            for metric_name, values in scalars.items():
                subset = values[indices]
                finite = subset[np.isfinite(subset)]
                if len(finite) == 0:
                    metrics[metric_name] = MetricSummary(
                        mean=float('nan'), std=float('nan'),
                        median=float('nan'), q25=float('nan'), q75=float('nan'),
                        min=float('nan'), max=float('nan'), count=0,
                    )
                else:
                    metrics[metric_name] = MetricSummary(
                        mean=float(np.mean(finite)),
                        std=float(np.std(finite)),
                        median=float(np.median(finite)),
                        q25=float(np.percentile(finite, 25)),
                        q75=float(np.percentile(finite, 75)),
                        min=float(np.min(finite)),
                        max=float(np.max(finite)),
                        count=len(finite),
                    )
            result.append(GroupStatistics(
                group_key=key, group_label=label, metrics=metrics,
            ))
        return result

    # ------------------------------------------------------------------
    # Diagnostic checks
    # ------------------------------------------------------------------

    def _check_movement_collapse(self, scalars: dict[str, np.ndarray]) -> DiagnosticCheck:
        """FAIL if median effector path length < 0.05."""
        amp = scalars['movement_amplitude']
        median_amp = float(np.nanmedian(amp))
        passed = median_amp >= 0.05
        return DiagnosticCheck(
            name="movement_collapse",
            status="pass" if passed else "fail",
            reason=(
                "Effector movement is adequate."
                if passed else
                "Median effector path length is near zero -- bodies may not be moving."
            ),
            evidence={"median_path_length": median_amp},
            hint=None if passed else (
                "Check that muscle activations are being applied and that the "
                "simulation is not stuck at the initial state."
            ),
        )

    def _check_target_indifference(
        self, npz: np.lib.npyio.NpzFile, scalars: dict[str, np.ndarray],
    ) -> DiagnosticCheck:
        """FAIL if mean final distance > 0.9 * mean initial distance."""
        dist_ts = np.linalg.norm(
            npz['effector_pos'] - npz['task_target'], axis=-1,
        )  # (N, T)
        mean_initial = float(np.nanmean(dist_ts[:, 0]))
        mean_final = float(np.nanmean(scalars['final_distance']))
        threshold = mean_initial * 0.9
        passed = mean_final <= threshold
        return DiagnosticCheck(
            name="target_indifference",
            status="pass" if passed else "fail",
            reason=(
                "Controllers are reducing distance to targets."
                if passed else
                "Mean final distance is barely reduced from initial -- "
                "controllers may be ignoring targets."
            ),
            evidence={
                "mean_initial_distance": mean_initial,
                "mean_final_distance": mean_final,
                "threshold": threshold,
            },
            hint=None if passed else (
                "The loss function may not be weighting target proximity, or "
                "training did not converge."
            ),
        )

    def _check_identical_behavior(self, scalars: dict[str, np.ndarray]) -> DiagnosticCheck:
        """WARN if std of final distance across trajectories < 0.01."""
        std_fd = float(np.nanstd(scalars['final_distance']))
        passed = std_fd >= 0.01
        return DiagnosticCheck(
            name="identical_behavior",
            status="pass" if passed else "warn",
            reason=(
                "Trajectories show healthy variance."
                if passed else
                "All trajectories end at nearly the same distance -- "
                "the controller may have collapsed to a single strategy."
            ),
            evidence={"std_final_distance": std_fd},
            hint=None if passed else (
                "This can happen when the policy ignores its input and outputs "
                "the same action sequence regardless of target or body."
            ),
        )

    def _check_effort_trap(
        self, npz: np.lib.npyio.NpzFile, scalars: dict[str, np.ndarray],
    ) -> DiagnosticCheck:
        """WARN if mean effort > 0.3 AND mean final distance reduction < 10%."""
        mean_effort = float(np.nanmean(scalars['effort']))
        dist_ts = np.linalg.norm(
            npz['effector_pos'] - npz['task_target'], axis=-1,
        )
        mean_initial = float(np.nanmean(dist_ts[:, 0]))
        mean_final = float(np.nanmean(scalars['final_distance']))

        if mean_initial > 0:
            reduction_frac = (mean_initial - mean_final) / mean_initial
        else:
            reduction_frac = 0.0

        high_effort = mean_effort > 0.3
        low_reduction = reduction_frac < 0.10
        passed = not (high_effort and low_reduction)

        return DiagnosticCheck(
            name="effort_trap",
            status="pass" if passed else "warn",
            reason=(
                "Effort-to-progress ratio is acceptable."
                if passed else
                "High muscle effort with little distance reduction -- "
                "the controller may be co-contracting without useful movement."
            ),
            evidence={
                "mean_effort": mean_effort,
                "mean_initial_distance": mean_initial,
                "mean_final_distance": mean_final,
                "distance_reduction_frac": reduction_frac,
            },
            hint=None if passed else (
                "Consider adding an effort penalty to the loss function, or "
                "check that the plant dynamics allow the effector to reach the target."
            ),
        )

    def _check_non_convergence(self, scalars: dict[str, np.ndarray]) -> DiagnosticCheck:
        """FAIL if fraction with final distance > 0.1 exceeds 90%."""
        fd = scalars['final_distance']
        finite = fd[np.isfinite(fd)]
        if len(finite) == 0:
            frac_far = 1.0
        else:
            frac_far = float(np.mean(finite > 0.1))
        passed = frac_far <= 0.9

        return DiagnosticCheck(
            name="non_convergence",
            status="pass" if passed else "fail",
            reason=(
                "A reasonable fraction of trajectories converge."
                if passed else
                "Over 90% of trajectories end far from their target."
            ),
            evidence={"fraction_final_distance_gt_0.1": frac_far},
            hint=None if passed else (
                "Training may need more iterations, or the task/body "
                "combination may be infeasible with the current architecture."
            ),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_npz(self, dataset: str) -> np.lib.npyio.NpzFile:
        """Load NPZ via the trajectory service (ensures validation + caching)."""
        # Trigger validation and cache-loading inside TrajectoryService.
        self._traj.get_metadata(dataset)
        # Access the cached handle directly.
        path = self._traj._resolve_path(dataset)
        mtime = path.stat().st_mtime
        return self._traj._cache[(str(path), mtime)]

    def _get_mtime(self, dataset: str) -> float:
        """Return the current mtime for *dataset*'s NPZ file.

        Validates that the dataset exists first (delegates to
        ``TrajectoryService.get_metadata`` which raises a 404 if missing).
        """
        # Bug: c722539 -- validate dataset before accessing mtime
        self._traj.get_metadata(dataset)
        path = self._traj._resolve_path(dataset)
        return path.stat().st_mtime


def _nanpercentile_list(arr: np.ndarray, percentile: float) -> list[float]:
    """Compute a percentile along axis 0, returning a list of Python floats.

    Handles NaN values gracefully via ``np.nanpercentile``.
    Returns NaN-filled list if the array has no trajectories (axis-0 is empty).
    """
    # Bug: c722539 -- empty subset crashes np.nanpercentile
    if arr.shape[0] == 0:
        n_cols = arr.shape[1] if arr.ndim >= 2 else 0
        return [float('nan')] * n_cols
    result = np.nanpercentile(arr, percentile, axis=0)
    return [float(v) for v in result]
