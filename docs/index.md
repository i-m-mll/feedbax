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

Feedbax uses JAX and [Equinox](https://docs.kidger.site/equinox/), because of their [features](/feedbax/examples/pytrees/) which are very convenient for scientific analyses. Still, if you've never used JAX before, you might find it (and Feedbax) a little strange at first.
<!--
One disadvantage of JAX is a lack of GPU support on Windows, though it is possible to use the GPU through the Windows Subsystem for Linux (WSL). -->

For a library that's similar to Feedbax but written in PyTorch, please check out [`MotorNet`](https://github.com/OlivierCodol/MotorNet)!

## Installation

Pip TODO.

`python -m pip install`

### Installing from source

## Development

I've developed Feedbax over the last few months, as I've learned JAX. My short-term objective has been to serve my own use case—graduate research in the neuroscience of motor control—but I have also tried to make design choices in pursuit of reusability and generality.

By making the library open source now, I hope to receive some feedback about those decisions. To make that easier I've created GitHub [issues](https://github.com/i-m-mll/feedbax/issues) documenting my choices and uncertainties. The issues largely fall into a few categories:

1. Structure: Some of the abstractions I've chosen are probably clumsy. It would be good to know about that, at this point. Maybe we can make some changes for the better! In approximate order of significance: #19, #24, #12, #1, #5, #21.
2. Features: There are many small additions that could be made, especially to the pre-built models and tasks. There are also a few major improvements which I am anticipating in the near future, such as *online learning* (#21). #10,
3. Typing: Typing in Feedbax is a mess, at the moment. I have been learning to use the typing system recently. However, I haven't been constraining myself with type checker errors. I know I've done some things that probably won't work, and that there may not be clever solutions to some of the issues. See issues: (#7, #8, #9, #11)

If you are an experienced Python or JAX user:

- Anything obviously clumsy I am doing with PyTrees
- Typing
- Performance issues

If you are a researcher in optimal control or reinforcement learning, I'd be particularly interested to hear what you think about

- whether you foresee any problems in applying RL formalisms given the way Feedbax is modularized

!!! Note:
    For comments on the documentation, I have specifically enabled to the Giscus commenting system so that GitHub users can comment on pages directly. You can also participate via the Discussions tab on GitHub.

## Acknowledgments

Special thanks to [Patrick Kidger](https://github.com/patrick-kidger), whose JAX libraries and their documentation often serve as examples to me.

