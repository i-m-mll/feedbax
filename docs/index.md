# Getting started

Feedbax is a [JAX](https://jax.readthedocs.io/en/latest/beginner_guide.html#beginner-guide) library for differentiable optimal control.

Feedbax makes it easy to:

- train neural networks to control the movement of simulated limbs (biomechanical models);
- intervene on existing models and tasks—for example, to:
    - add force fields to trials at random;
    - alter the activity of a single unit in a neural network;
    - perturb the sensory feedback received by a network;
    - add any kind of noise to any part of a model's state;
- swap out components of models, and write new components; 
- train multiple replicates of a model in parallel;
- change which parts of the model are trainable, or available to a controller as feedback;
- track the progress of a training run in Tensorboard.

Feedbax was designed for feedback control of biomechanical models by neural networks, to perform movement tasks in continuous spaces. However, it can be used more generally as a framework for optimal control.

## Feedbax is a JAX library

Feedbax uses JAX and [Equinox](https://docs.kidger.site/equinox/) for the structural [advantages](/feedbax/examples/pytrees/) they provide. 

One disadvantage of JAX is a lack of GPU support on Windows, though it is possible to use the GPU through the Windows Subsystem for Linux (WSL).

## Installation

Pip TODO.

## Development

I've developed Feedbax over the last few months, as I've learned JAX. My short-term objective has been to serve my own use case---graduate research in the neuroscience of motor control---but I have also tried to make design choices in pursuit of reusability and generality.

By making the library open source now, I hope to receive some feedback about those decisions, and perhaps make some structural improvements. To make that easier I've created several GitHub issues documenting my choices and uncertainties. The issues largely belong to one of a few categories:

1. Structural issues: these are the most critical.
2. Typing issues: I'm not confident these can be resolved. Mostly I'm still learning to use typing in Python to the fullest.
3. 

Anyone is welcome to ask questions or make suggestions about any part of the code or documentation!

If you are a researcher in optimal control or reinforcement learning, I'd be particularly interested to hear about...

If you are an experienced JAX user...

## Acknowledgments 

Special thanks to Patrick Kidger whose example I've relied on

