# Getting started

Feedbax is a [JAX](https://jax.readthedocs.io/en/latest/beginner_guide.html#beginner-guide) library for optimal feedback control with neural networks.

Feedbax makes it easy to:

- [train](/feedbax/examples/0_train_simple) a neural network to control a simulated limb (biomechanical model) to perform movement tasks;
- [intervene](/feedbax/examples/3_intervening) on existing models and tasks—for example, to:
    - add force fields that disturb a limb;
    - alter the activity of a single unit in a neural network;
    - perturb the sensory feedback received by a network;
    - add any kind of noise to any part of a model's state;
- schedule an intervention to occur on only a subset of task trials or time steps;
- specify which parts of the model are [trainable](/feedbax/examples/1_train/#selecting-part-of-the-model-to-train), and which parts are available as feedback to a neural network;
- train [multiple replicates](/feedbax/examples/4_vmap) of a model at once;
- swap out components of models, and write new components.
<!-- - track the progress of a training run in Tensorboard. -->

Feedbax was designed for feedback control of biomechanical models, to perform movement tasks in continuous spaces. However, it can be used more generally as a framework for optimal control.

## Feedbax is a JAX library

Feedbax uses JAX and [Equinox](https://docs.kidger.site/equinox/).

[Never used JAX before](/feedbax/examples/pytrees/)?
<!--
One disadvantage of JAX is a lack of GPU support on Windows, though it is possible to use the GPU through the Windows Subsystem for Linux (WSL). -->

For a library that's similar to Feedbax but written in PyTorch, please check out [`MotorNet`](https://github.com/OlivierCodol/MotorNet)!

## Installation

`pip install feedbax`

Currently requires Python>=3.11.

## Development

I've developed Feedbax over the last few months, while learning JAX. My short-term objective has been to serve my own use case—graduate research in the neuroscience of motor control—but I've also tried to design something reusable and general.

I've added GitHub [issues](https://github.com/i-m-mll/feedbax/issues) to document some of my choices and uncertainties, and to open them for discussion. For an overview of major issues in different categories, check out [this GitHub conversation](https://github.com/i-m-mll/feedbax/discussions/27).

There are many features that could be implemented, especially with respect to pre-built models and tasks. So far I've focused more on the overall structure, than on feature-completeness. Anyone can suggest a feature they'd like to see.

You're also welcome to leave feedback on the documentation. GitHub users can comment at the bottom of each page of documentation, or directly on [GitHub](https://github.com/i-m-mll/feedbax/discussions/categories/documentation).

## Acknowledgments

- Thanks to my PhD supervisor Gunnar Blohm and to the rest of our [lab](http://compneurosci.com/), as well as to Dominik Endres and Stephen H. Scott for discussions that have influenced this project
- Special thanks to [Patrick Kidger](https://github.com/patrick-kidger), whose JAX libraries and their documentation often serve as examples to me

