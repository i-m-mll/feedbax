# MuJoCo numerical baseline 1

The `feedbax[mjx]` extra selects the initial scientific runtime. `pyproject.toml` pins the four provider distributions and `uv.lock` records the resolved dependency versions and artifact hashes. Install the repository baseline with `uv sync --locked --extra mjx --python 3.12`. Feedbax imports the native engine as `mujoco` and the published XLA provider as `mujoco.mjx`. Construct the JAX model with `impl="jax"`. The native distribution is `mujoco-core` because MuJoCo 4.0 splits its simulation and Studio packages. [MuJoCo Core metadata](https://pypi.org/project/mujoco-core/4.0.0/) · [MJX metadata](https://pypi.org/project/mujoco-mjx/4.0.0/)

| Component | Prior tested lock | Baseline 1 | Provenance |
| --- | --- | --- | --- |
| Native MuJoCo | `mujoco==3.5.0` | `mujoco-core==4.0.0` | [4.0.0 source](https://github.com/google-deepmind/mujoco/tree/ac60eae30fa7e0d31315c38d9476752fb81be67e) |
| MJX | `mujoco-mjx==3.5.0` | `mujoco-mjx==4.0.0` | [4.0.0 MJX source](https://github.com/google-deepmind/mujoco/tree/ac60eae30fa7e0d31315c38d9476752fb81be67e/mjx) |
| JAX and JAXLIB | `0.7.2` | `0.7.2` | [JAX 0.7.2](https://github.com/jax-ml/jax/releases/tag/jax-v0.7.2) |

MuJoCo 4.0.0 and JAX 0.12.0.1 were the current published versions when the baseline was selected. The tested pair keeps Feedbax's existing JAX version and adopts the matching native and MJX 4.0.0 releases. This adds public MJX state operations and Warp data conversion while keeping a smaller numerical change. The baseline uses unmodified published artifacts. Changing a provider version or patch invalidates affected evidence until the relevant rows are repeated. [MuJoCo release](https://github.com/google-deepmind/mujoco/releases/tag/4.0.0) · [JAX 0.12.0.1](https://pypi.org/project/jax/0.12.0.1/) · [MJX 4.0.0 state APIs](https://github.com/google-deepmind/mujoco/blob/ac60eae30fa7e0d31315c38d9476752fb81be67e/mjx/mujoco/mjx/_src/io.py)

## Qualified scope

| Target and operation | Baseline evidence |
| --- | --- |
| macOS arm64 JAX CPU, constraint-free hinge | Stock MJX native step agrees with native MuJoCo and the analytical semi-implicit update. The control derivative agrees with native central differences. Feedbax adapter and closed-loop learning qualify separately. |
| macOS arm64 JAX CPU, constrained forward | The three-sphere stack's multi-iteration conjugate-gradient solve agrees with native MuJoCo. Its one-iteration setting produces materially different acceleration. Constrained reverse mode is outside the initial capability. |
| macOS arm64 JAX CPU, tendon wrap primitive | At the four sampled angles, outside-sidesite sphere length and its reverse derivative agree with native MuJoCo and central differences. Inside-sidesite sphere length agrees and its reverse derivative is NaN. Inside-route reverse mode is outside the initial capability. The wrapping delivery owns domain and learning qualification. |
| NVIDIA CUDA, MJX-Warp | Target-specific forward and conversion qualification remains pending. It requires the actual supported CUDA host and compatible Warp/JAX artifacts. |

A constraint-free model has no compiled constraint rows on the MJX-JAX execution path. Joint limits and contact or equality configuration can create rows even at an otherwise regular pose. Capability checks must preserve the authored limits, contact behavior and solver settings. The one-iteration solve and the inactive pose do not establish constrained reverse mode. [MJX solver report](https://github.com/google-deepmind/mujoco/issues/2259) · [MJX inside-wrap report](https://github.com/google-deepmind/mujoco/issues/3696)

## Reproduction

On macOS arm64, run `uv run --no-sync python scripts/qualify_mujoco_baseline.py` after the locked install. The diagnostic enables JAX CPU and float64, uses a central-difference step of `1e-5`, and compares against the independently compiled native MuJoCo implementation. The constrained solve and tendon wrap were reproduced on the prior tested lock and the 4.0.0 candidate. The diagnostic asserts the observed defects to detect a change in provider behavior when reevaluating the baseline.

- At a 2 ms step and 0.3 N·m, the unconstrained hinge advances by `4.8e-6` rad to `0.0024` rad/s. MJX reverse mode and native finite differences both give approximately `1.6e-5` rad per N·m.
- At 0.3 rad, the outside-sidesite tendon length is approximately `0.564100853441` m and its derivative is `0.0155392466` m/rad. The inside-sidesite length is approximately `0.547491292943` m, with native finite difference `-0.0300132564` m/rad and NaN from MJX reverse mode.
- The contact stack's multi-iteration solve takes three iterations in native MuJoCo and gives approximately `[0.820543, 0.949053, 2.210096]` m/s² in both engines. At one iteration both give approximately `[0.491692, -0.317544, 0.459323]` m/s². Only the one-iteration MJX case returns a reverse derivative. The multi-iteration case raises the dynamic-loop error.
