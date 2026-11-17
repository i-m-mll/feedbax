# Fixed-seed legacy task and loss references

Captured from signed Feedbax `47fe615970e7c4d320f174d16587385d95433ea1`, tree `a42465cbb79a34f01c391d1e03709b8ca0e3f168`. The [task and loss capture](https://app.mandible.localhost:19420/artifact/artifact-version:c0929f4774da) contains the pinned generation and verification recipes. The [biomechanics capture](https://app.mandible.localhost:19420/artifact/artifact-version:eeb999207c26) contains numerical outputs and stop/resume evidence. The original planar and spatial specifications and assets are stored under `biomechanics/`, including the MJCF and fixed inputs in `fixtures.json`.

- `legacy_task_reference.npz` contains 153 arrays and three legacy task families: SimpleReaches, DelayedReaches and Stabilization. The files include ordered train and validation coordinates, initial states, signals, masks, event and epoch timing, fixed-seed draws, and the real PointMass neural model trajectories and legacy weighted term values used for Step 1 comparison. The 250-trial sequence uses split seed 1. PointMass initialization uses seed 7. Distribution sequences use seeds 16 and 24.
- `legacy_loss_reference.npz` contains 65 arrays with recorded observations and expected per-trial and aggregate results across 16 loss cases. These include fixed-window and empty-window losses, finite differences, both stop-at-goal reductions, both straight-path normalizations, weighting, and the legacy selector-based terminal matrix quadratic.

Read arrays with `numpy.load(archive, allow_pickle=False)`. `reference.json` gives the SHA-256, array names, shapes, dtypes, seeds, fixture settings and comparison tolerances. It also records the lock SHA-256 and Python/JAX/Equinox/NumPy versions.

The document-runtime equality tests read the task archive. The loss archive also preserves the reference cases for future Objective implementations.

## Array interpretation

- Task input signals are indexed by trial, execution step, then feature where present. The stored PointMass state trajectories use the legacy equality fixture's 80-sample aligned observation view. Epoch bounds store half-open intervals. Their fixed-epoch result is length completion. `present/*` is a derived mask from those intervals and the delayed task's target-visibility boundary.
- Loss keys named `per_trial_raw` and `per_trial_weighted` reflect the legacy term tree's leaves. Most leaves are batched. Model-only and some reduced selector leaves are scalars, as specified by their shapes in the index. Straight path normalized by actual distance intentionally yields positive infinity for the closed moving trial.
- The shared-trajectory StopAtGoal cases use window length 2, distance scale 0.025, velocity scale 0.05, velocity weight 1, temperature 0.2 and epsilon 1e-6. The soft-or result there hits its probability floor. The dedicated complete-window fixture supplies finite, non-saturated cases.

The dedicated StopAtGoal archive contains 16 arrays from the exact two complete-window fixture variants. Scored masks, inputs and per-trial raw and weighted values are included. The output passed the existing hand-computed formulas at relative tolerance 1e-6 and absolute tolerance 1e-12 under lease c464b6fc4d61c8cf. The one-off supplementary capture imports utilities from capture.py and does not rerun the task cohorts.
