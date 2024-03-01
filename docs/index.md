# Getting started

Feedbax is a [JAX](https://jax.readthedocs.io/en/latest/beginner_guide.html#beginner-guide) library for optimal (feedback) control.

Feedbax makes it easy to:

- [train](/feedbax/examples/0_train_simple) neural networks to control the movement of simulated limbs (biomechanical models);
- [intervene](/feedbax/examples/3_intervening) on existing models and tasks—for example, to:
    - add force fields to trials at random;
    - alter the activity of a single unit in a neural network;
    - perturb the sensory feedback received by a network;
    - add any kind of noise to any part of a model's state;
- [train replicates](/feedbax/examples/4_vmap) of a model in parallel;
- specify which parts of the model are [trainable](/feedbax/examples/1_train/#selecting-part-of-the-model-to-train), or available to a controller as feedback;
- swap out components of models, and write new components.
<!-- - track the progress of a training run in Tensorboard. -->

Feedbax was designed for feedback control of biomechanical models, to perform movement tasks in continuous spaces. However, it can be used more generally as a framework for optimal control.

## Feedbax is a JAX library

Feedbax uses JAX and [Equinox](https://docs.kidger.site/equinox/), because their features are very convenient for model design and scientific analysis.

[Never used JAX before](/feedbax/examples/pytrees/)?
<!--
One disadvantage of JAX is a lack of GPU support on Windows, though it is possible to use the GPU through the Windows Subsystem for Linux (WSL). -->

For a library that's similar to Feedbax but written in PyTorch, please check out [`MotorNet`](https://github.com/OlivierCodol/MotorNet)!

## Installation

Pip TODO.

`python -m pip install`

### Installing from source

## Development

I've developed Feedbax over the last few months, as I've learned JAX. My short-term objective has been to serve my own use case—graduate research in the neuroscience of motor control—but I have also tried to make design choices in pursuit of reusability and generality.

By making the library open source now, I hope to receive some feedback about those decisions. To make that easier I've created GitHub [issues](https://github.com/i-m-mll/feedbax/issues) documenting my choices and uncertainties. Those issues largely fall into a few categories:

1. Structure: Some of the abstractions I've chosen are probably clumsy. It would be good to find out, at this point—maybe we can make some changes for the better!
   In approximate order of significance: #19, #24, #12, #20, #1, #5, #21.
2. Features: There are many small additions that could be made, especially to the pre-built models and tasks. There are also a few major improvements which I am anticipating in the near future, such as *online learning* (#21). #10,
3. Typing: I am currently working through many pyright errors, trying to make Feedbax compliant. I'm not very experienced with typing in Python, so I may have tried to do things that are heavy-handed or over-clever. See issues: (#7, #8, #9, #11)
4. Performance.

!!! Note:
    For comments on the documentation, I have specifically enabled to the Giscus commenting system so that GitHub users can comment on pages directly. You can also participate via the Discussions tab on GitHub.

## Acknowledgments

Special thanks to [Patrick Kidger](https://github.com/patrick-kidger), whose JAX libraries and their documentation often serve as examples to me.

