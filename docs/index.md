# Getting started

Feedbax is a JAX library for optimal feedback control. 

Feedbax makes it easy to:

- train neural networks (or other controller models) to control the movement of simulated bodies (biomechanical models);
- intervene on existing tasks and models—for example, to
  - add force fields to trials at random;
  - alter the activity of a single unit in a neural network;
  - perturb the sensory feedback received by a network;
- swap out components of models, and write new components; 
- train multiple model replicates in parallel;
- change which parts of the model are trainable, or available to the controller as feedback;
- noise

Feedbax provides common...

- tasks 
- biomechanical models
- plotting functions

## Feedbax is a JAX library

Advantages. PyTrees. NumPy-like. 
Limitations. Windows.

Equinox.

## Installation

Pip TODO.

## Development

I've developed Feedbax over the last few months, as I've learned JAX. In doing so, I've made a number of structural commitments, in pursuit of flexibility and generality. 

By making the library open source now, I hope to receive some feedback about those decisions, and perhaps start with some structural improvements. To make that easier, I've written some documentation, but also a number of GitHub issues documenting my choices, and the shape of my uncertainty. 

You are welcome to ask questions or make suggestions for improvements to any part of the code or documentation.

If you are a researcher in optimal control or reinforcement learning, I'd be particularly interested to hear about...

If you are an experienced JAX user...

