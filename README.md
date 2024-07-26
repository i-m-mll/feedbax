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

Feedbax is in active [development](#development). Expect some changes in the near future. In particular, I intend to [replace](https://github.com/i-m-mll/feedbax/discussions/28#discussioncomment-10153895) the staged model approach with a simpler lazy-executing DAG framework.

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

## Development

I started to develop Feedbax while learning JAX. My short-term objective has been to support my own use cases—graduate research in the neuroscience of motor control—but I've also tried to design something reusable and general.

I've added GitHub [issues](https://github.com/i-m-mll/feedbax/issues) to document some of my choices and uncertainties. For an overview of major issues in different categories, check out [this GitHub conversation](https://github.com/i-m-mll/feedbax/discussions/27). Refer also to [this page](https://docs.lprt.ca/feedbax/structure) of the docs, for an informal overview of how Feedbax objects relate to each other.

There are many features, especially pre-built models and tasks, that could still be implemented. Some of the models and tasks that are implemented have yet to be fully optimized. So far I've focused more on the overall structure, than on coverage of all the common use cases I can imagine. If there's a particular model, task, or feature you'd like Feedbax to support, [let us know](https://github.com/i-m-mll/feedbax/issues), or contribute some code!

## Acknowledgments

- Thanks to my PhD supervisor Gunnar Blohm and to the rest of our [lab](http://compneurosci.com/), as well as to Dominik Endres and Stephen H. Scott for discussions that have directly influenced this project
- Special thanks to [Patrick Kidger](https://github.com/patrick-kidger), whose JAX libraries and their documentation often serve as examples to me

