# Getting started

Feedbax is a [JAX](https://jax.readthedocs.io/en/latest/beginner_guide.html#beginner-guide) library for optimal (feedback) control.

Feedbax makes it easy to:

- [train](/feedbax/examples/0_train_simple) neural networks to control the movement of simulated limbs (biomechanical models);
- [intervene](/feedbax/examples/3_intervening) on existing models and tasks—for example, to:
    - add force fields to trials at random;
    - alter the activity of a single unit in a neural network;
    - perturb the sensory feedback received by a network;
    - add any kind of noise to any part of a model's state;
- swap out components of models, and write new components; 
- [train replicates](/feedbax/examples/4_vmap) of a model in parallel;
- specify which parts of the model are [trainable](/feedbax/examples/1_train/#selecting-part-of-the-model-to-train), or available to a controller as feedback;
<!-- - track the progress of a training run in Tensorboard. -->

Feedbax was designed for feedback control of biomechanical models by neural networks, performing movement tasks in continuous spaces. However, it can be used more generally as a framework for optimal control.

## Feedbax is a JAX library

Feedbax uses JAX and [Equinox](https://docs.kidger.site/equinox/) for the structural [advantages](/feedbax/examples/pytrees/) they provide. 

One disadvantage of JAX is a lack of GPU support on Windows, though it is possible to use the GPU through the Windows Subsystem for Linux (WSL).

If you prefer to use PyTorch, check out `MotorNet`!

## Installation

Pip TODO.

## Development

I've developed Feedbax over the last few months, as I've learned JAX. My short-term objective has been to serve my own use case—graduate research in the neuroscience of motor control—but I have also tried to make design choices in pursuit of reusability and generality.

By making the library open source now, I hope to receive some feedback about those decisions. To make that easier I've created GitHub [issues](https://github.com/i-m-mll/feedbax/issues) documenting my choices and uncertainties. The issues largely belong to one of a few categories:

1. Structural issues: Perhaps some of the abstractions I've chosen are clumsy. Depending on the effort involved, I'm still willing to initiate major structural changes at this point. 
2. Typing issues: I've tried to err on the side of typing things a bit too much. At least it may still serve as documentation? I'm still learning the limits of typing in Python. 
3. Feature issues: There are many small improvements and additions that could be made, especially pre-built models and tasks.
4. 

If you are a researcher in optimal control or reinforcement learning, I'd be particularly interested to hear what you think about 

- whether you foresee any problems in applying RL formalisms given the way Feedbax is modularized

If you are an experienced Python or JAX user:

- Do
- Any low-hanging fruit re: abstraction
- Anything obviously clumsy I am doing with PyTrees
- Performance issues

!!! Success ""
    Ask questions or make suggestions about any part of the code or documentation!

## Acknowledgments 

Special thanks to [Patrick Kidger](https://github.com/patrick-kidger), whose JAX libraries and their documentation often serve as examples to me.

