# Getting started

Feedbax is a JAX library for optimal feedback control. Feedbax makes it easy to:

- train neural networks (or other controller models) to control biomechanical models to perform movement tasks;
- intervene on existing tasks and models; for example to
  - add force fields to trials at random
  - alter the activity of a single unit in a neural network
  - perturb the sensory feedback received by a network
- swap out components of models, and write new components; 
- train multiple model replicates in parallel;
- change which parts of the model are trainable;
- change which parts of the state are available as feedback;
- debug

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

## Example usage

See example 0.

## Development

I've developed Feedbax over the last few months, as I've learned JAX.

To achieve the flexibility and generality I've desired, I've made certain design decisions. 

By making the library open source now, I hope to receive some feedback about those decisions, and possibly make some structural improvements. 

To that end, in addition to the primary documentation for Feedbax, I've written some GitHub issues documenting my choices, and the shape of my uncertainties. 

There are several ways in which Feedbax could be more feature-complete. In particular, class of perturbations, neural network models, muscle models... However, before working to proliferate these in detail, I want to address the possibility of structural improvements.

If you are a researcher in optimal control or reinforcement learning, I'd be particularly interested to hear about...

If you are a JAX user...

In any case, you are welcome to ask questions or make suggestions for improvements to any part of the code or documentation.

