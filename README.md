> [!NOTE]
> **Feedbax is under active development while the eager graph architecture and Studio workflow stabilize.** Feel free to explore, but know that things will continue to change.

# Getting started

Feedbax is a [JAX](https://jax.readthedocs.io/en/latest/beginner_guide.html#beginner-guide) library for optimal feedback control with neural networks.

Feedbax makes it easy to:

- [train](https://docs.lprt.ca/feedbax/examples/0_train_simple) a neural network to control a simulated limb (biomechanical model) to perform movement tasks;
- [intervene](https://docs.lprt.ca/feedbax/examples/3_intervening) on existing models and tasks—for example, to:
    - add force fields that disturb a limb;
    - alter the activity of a single unit in a neural network;
    - perturb the sensory feedback received by a network;
    - add any kind of noise to any part of a model's state;
- schedule an intervention to occur on only a subset of task trials or time steps;
- specify which parts of the model are [trainable](https://docs.lprt.ca/feedbax/examples/1_train/#selecting-part-of-the-model-to-train), and which states are available as sensory feedback;
- train [multiple replicates](https://docs.lprt.ca/feedbax/examples/4_vmap) of a model at once;
- swap out components of models, and write new components.
<!-- - track the progress of a training run in Tensorboard. -->

Feedbax is in active [development](#development). Expect some changes in the near future. The staged model approach has been replaced by an explicit eager graph architecture.

## Feedbax is a JAX library

Feedbax uses JAX and [Equinox](https://docs.kidger.site/equinox/).

[Never used JAX before](https://docs.lprt.ca/feedbax/examples/pytrees/)?

Please also check out:

- [MotorNet](https://github.com/OlivierCodol/MotorNet), a PyTorch library with many similarities to Feedbax.
- [Collimator](https://collimator.ai), a more mature JAX library for composing and optimizing dynamical systems.

## Installation

`pip install feedbax`

Currently requires Python>=3.11.

For best performance, [install JAX](https://jax.readthedocs.io/en/latest/installation.html) with GPU support.

## Documentation

Documentation is available [here](https://docs.lprt.ca/feedbax).

## Web UI

Feedbax includes a web interface for visually constructing and training models. To start both the backend API and frontend dev server:

```bash
./scripts/dev.sh
```

Or run them separately:

```bash
# Backend (FastAPI on port 8000)
uv run uvicorn feedbax.web.app:app --reload --port 8000

# Frontend (Vite/React on port 3008)
cd web && npm run dev
```

Then open http://localhost:3008 in your browser.

## Building application packages on feedbax

Feedbax is a library.  Domain-specific research projects (e.g., **rlrmp**) are
*application packages* that depend on feedbax and extend it with their own
models, tasks, analyses, and training pipelines.

### Editable install (required)

Application packages must be installed as editable installs — not published to
PyPI.  `feedbax.plot.save_figure` resolves output directories relative to the
package's git repository root by walking up from `package.__file__` until it
finds a `.git` directory.  If the package is pip-installed into site-packages
rather than checked out locally, that walk fails and figures would be written
to the wrong place.  Install with:

```bash
uv pip install -e /path/to/your-package
```

### Registration via entry points

Application packages register themselves with the feedbax plugin system via a
`[project.entry-points."feedbax.plugins"]` section in `pyproject.toml` and a
registration function that calls `register_package_from_module_info`.

**`pyproject.toml`:**

```toml
[project.entry-points."feedbax.plugins"]
rlrmp = "rlrmp:register_experiment_package"
```

**`rlrmp/__init__.py`:**

```python
from feedbax.plugins import EXPERIMENT_REGISTRY
from feedbax.plugins.discovery import register_package_from_module_info

def register_experiment_package(registry=None):
    if registry is None:
        registry = EXPERIMENT_REGISTRY
    register_package_from_module_info(
        registry,
        package_name="rlrmp",
        package_module_name="rlrmp",
        parts=["part1", "part2", "part2_5"],
        analysis_module_root="modules.analysis",
        training_module_root="modules.training",
        config_resource_root="config",
        figure_routing={
            "spec_dir_template": "results/{experiment}/figures/{topic}",
            "render_dir_template": "_artifacts/{experiment}/figures/{topic}",
            "spec_format": "json",
            "render_format": "html",
            "create_symlink_in_spec_dir": True,
        },
    )
```

### `figure_routing` config schema

| Key | Type | Description |
|-----|------|-------------|
| `spec_dir_template` | `str` | Path template for the spec JSON directory, relative to repo root. `{experiment}` and `{topic}` are substituted at save time. |
| `render_dir_template` | `str` | Path template for the heavy figure render directory (typically gitignored). |
| `spec_format` | `str` | Always `"json"` for now. |
| `render_format` | `str` | Figure format: `"html"` (default), `"json"` (Plotly JSON), `"png"`, `"svg"`. |
| `create_symlink_in_spec_dir` | `bool` | If `True`, creates a relative symlink in the spec dir pointing at the render file. |

### Saving figures

```python
from feedbax.plot import save_figure

paths = save_figure(
    fig,           # plotly or matplotlib Figure
    spec,          # dict: inputs, transform, plot_kwargs, seed
    package="rlrmp",
    experiment="part2_5",
    topic="adversarial_losses",
)
# paths["spec_path"]    → results/part2_5/figures/adversarial_losses/spec.json
# paths["render_path"]  → _artifacts/part2_5/figures/adversarial_losses/figure.html
# paths["symlink_path"] → results/part2_5/figures/adversarial_losses/figure.html (symlink)
```

The spec JSON receives automatic augmentation: SHA-256 digests for every input
artifact, installed package versions, and a UTC timestamp.

> **Full docs** for the integration pattern are tracked in the documentation tree as the plugin and figure-routing APIs evolve.

## Development

I started to develop Feedbax while learning JAX. My short-term objective has been to support my own use cases—graduate research in the neuroscience of motor control—but I've also tried to design something reusable and general.

I've added GitHub [issues](https://github.com/i-m-mll/feedbax/issues) to document some of my choices and uncertainties. For an overview of major issues in different categories, check out [this GitHub conversation](https://github.com/i-m-mll/feedbax/discussions/27). Refer also to [this page](https://docs.lprt.ca/feedbax/structure) of the docs, for an informal overview of how Feedbax objects relate to each other.

There are many features, especially pre-built models and tasks, that could still be implemented. Some of the models and tasks that are implemented have yet to be fully optimized. So far I've focused more on the overall structure, than on coverage of all the common use cases I can imagine. If there's a particular model, task, or feature you'd like Feedbax to support, [let us know](https://github.com/i-m-mll/feedbax/issues), or contribute some code!

## Acknowledgments

- Thanks to my PhD supervisor Gunnar Blohm and to the rest of our [lab](http://compneurosci.com/), as well as to Dominik Endres and Stephen H. Scott for discussions that have directly influenced this project
- Special thanks to [Patrick Kidger](https://github.com/patrick-kidger), whose JAX libraries and their documentation often serve as examples to me
