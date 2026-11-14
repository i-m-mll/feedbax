> [!NOTE]
> **Feedbax is under active development.** Feel free to explore, but know that things are unstable and likely to change, and that the documentation is deprecated.

<p align="center">
  <img src="docs/assets/feedbax-studio.png" alt="Feedbax Studio logo" width="100" height="100">
</p>

# Feedbax

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

Feedbax includes a web interface for visually constructing and training models. Install the Bun version in `.bun-version` and the locked dependencies.

```bash
uv sync --locked --extra web --extra mjx
bun install --cwd web --frozen-lockfile
./scripts/dev.sh
```

Open http://127.0.0.1:3008 in your browser. Each launch uses a fresh temporary data home. Use `./scripts/dev.sh --home /path/to/studio-data` to keep projects and runs across launches, or add `--preview` to serve a production build at http://127.0.0.1:3009.

Run the isolated browser walkthrough with `cd web && bun run test:studio`. Set `FEEDBAX_STUDIO_E2E_PREVIEW=1` to exercise the production frontend.

## Building application packages on feedbax

Application packages extend Feedbax with their own scientific models, tasks, objectives, perturbations, analyses, and numerical methods. Start with the modules you need; split them as their responsibilities become distinct. There is no required directory scaffold.

Keep reusable problem-domain facilities outside whichever method first used them. A method owns its numerical updates, configuration, and declarations, and can retain its scientific algorithm name. Experiment labels belong in saved projects and study materials.

Keep authored inputs, frozen inputs, and generated outputs distinct. Preserve frozen input bytes and their pinned provenance exactly. Let the compiler produce resolved outputs and locks, and retain execution evidence through the artifact and custody interfaces. Moving source can change source-derived identities: refresh active authoring and regenerate its outputs through the existing public authoring and compilation tools while preserving frozen inputs.

Both editable development installs and installed wheels can provide plugins. Configure resource, output, and custody locations through the relevant public interfaces rather than assuming the installed package is a writable repository checkout. The [downstream stability inventory](docs/design/downstream_interface_stability.md) lists the implemented interfaces during the rebuild and links the historical ratification. The [downstream conformance kit](docs/design/downstream_conformance_kit.md) documents the test helpers available to extension packages.

Declare your plugin entry point in `pyproject.toml`, for example:

```toml
[project.entry-points."feedbax.plugins"]
my_project = "my_project.plugin:PLUGIN_REGISTRATION"
```

The entry point exports a typed `PluginRegistration` with a `PluginDeclaration`, an explicit downstream protocol version, required registry families, and a registration callback. Keep this assembly in a dedicated module loaded by the application's outer layer through the public bootstrap. Numerical modules remain independent of plugin assembly. Register components through explicitly namespaced `ComponentPack` objects; keep component migration records apart from runtime code and associate their histories with current class objects when assembling the pack. A current owner with no history needs no migration record.

## Development

I started to develop Feedbax while learning JAX. My short-term objective has been to support my own use cases—graduate research in the neuroscience of motor control—but I've also tried to design something reusable and general.

I've added GitHub [issues](https://github.com/i-m-mll/feedbax/issues) to document some of my choices and uncertainties. For an overview of major issues in different categories, check out [this GitHub conversation](https://github.com/i-m-mll/feedbax/discussions/27). Refer also to [this page](https://docs.lprt.ca/feedbax/structure) of the docs, for an informal overview of how Feedbax objects relate to each other.

There are many features, especially pre-built models and tasks, that could still be implemented. Some of the models and tasks that are implemented have yet to be fully optimized. So far I've focused more on the overall structure, than on coverage of all the common use cases I can imagine. If there's a particular model, task, or feature you'd like Feedbax to support, [let us know](https://github.com/i-m-mll/feedbax/issues), or contribute some code!

## Acknowledgments

- Thanks to my PhD supervisor Gunnar Blohm and to the rest of our [lab](http://compneurosci.com/), as well as to Dominik Endres and Stephen H. Scott for discussions that have directly influenced this project
- Special thanks to [Patrick Kidger](https://github.com/patrick-kidger), whose JAX libraries and their documentation often serve as examples to me
