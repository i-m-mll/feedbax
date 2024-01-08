# ---
# jupyter:
#   jupytext:
#     formats: ipynb,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: fx
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import torch 


# %% [markdown]
# In PyTorch, we can design our calculations in a modular way by writing Python classes that inherit from `torch.nn.Module`.
#
# For example, here's a module that can compute a simple linear transformation:

# %%
class PytorchModel(torch.nn.Module):
    # note the type of `weights`
    def __init__(self, weights: torch.Tensor): 
        super().__init__()
        self.weights = weights
      
    def forward(self, input_: torch.Tensor):
        return self.weights @ input_  


# %% [markdown]
# Note that `weights` should be a matrix that defines the linear transformation. This is the only parameter of the model.
#
# Let's use this model to apply a linear transformation to a single input example. First, we'll generate some random weights for a linear transformation with 10 inputs and 2 outputs. Let's do this in NumPy so we can re-use the data for a non-PyTorch example in a minute.

# %%
weights = np.random.random((2, 10))
example_input = np.random.random((10,)) 

print("weights:\n", weights, '\n')
print("example input:\n", example_input)

# %% [markdown]
# Now we can 1) make an instance of the model that has the desired weights, 2) apply the model to compute the outputs for our example input

# %%
# need to convert to PyTorch tensors first
weights_tensor = torch.tensor(weights)
example_input_tensor = torch.tensor(example_input)

# %%
model = PytorchModel(weights_tensor)

example_output_tensor = model(example_input_tensor)

example_output_tensor


# %% [markdown]
# Once we build a PyTorch model, we can call it like a function `model(...)`. 
#
# This is not standard behaviour for Python classes!

# %%
# not a subclass of torch.nn.Module!
class NeverCallMe:  
    def __init__(self, weights):
        self.weights = weights
        
    def forward(self, input_):
        return self.weights @ input_


# %%
never_call_me = NeverCallMe(weights)

never_call_me(example_input)


# %% [markdown]
# However, outside of PyTorch we can still tell Python what should happen when we try to call an instance of a class. 
#
# Let's design a class that does the same thing as our PyTorch model from earlier, but using just NumPy:

# %%
class NumpyModel:
    def __init__(self, weights: np.ndarray): 
        self.weights = weights
  
    # __call__ is a "magic method"
    def __call__(self, input_: np.ndarray):
        return self.weights @ input_ 


# %%
model = NumpyModel(weights)

# now this works!
example_output = model(example_input) 

# the same result as with PyTorch
example_output

# %% [markdown]
# ### Dataclasses

# %% [markdown]
# It isn't very nice to read `__init__` methods that may be large and consist only of boilerplate like `self.param1 = param1`, `self.param2 = param2`, etc.
#
# Thankfully there's a prettier way of writing Python classes: dataclasses!

# %%
from dataclasses import dataclass 

@dataclass
class DataclassModel:
    weights: np.ndarray  
    
    def __call__(self, input_: np.ndarray):
        return self.weights @ input_ 


# %% [markdown]
# This has exactly the same behaviour as `NumpyModel`:

# %%
model = DataclassModel(weights)

model(example_input)

# %% [markdown]
# The secret is that Python builds an `__init__` method behind-the-scenes that includes the boilerplate `self.weights = weights`. 
#
# When we write with JAX, we'll use a package called Equinox which allows us to write modules in a PyTorch-like way. However, it automatically lets us write our modules like dataclasses, and we'll use the standard Python `__call__` instead of the PyTorch-specific `forward` method.

# %%
import equinox as eqx 

class DataclassModel(eqx.Module):
    weights: np.ndarray  
    
    def __call__(self, input_: np.ndarray):
        return self.weights @ input_ 


# %% [markdown]
# Note the nice separation between the list of model parameters (just `weights`, in this case) and the computation that the model performs (`__call__`).

# %% [markdown]
# ### JAX is purely functional

# %% [markdown]
# In JAX, we write models that are purely functional. JAX models are *stateless*. That is, they *aren't stateful*. 
#
# Let's see an example of a stateful model so we can figure out what that means.

# %%
@dataclass
class StatefulModel:
  weights: np.ndarray
  
  def __call__(self, input_: np.ndarray):
    output = self.weights @ input_
    
    if np.sum(output) > 0:
      self.weights = -self.weights
    
    return output


# %%
model = StatefulModel(weights)

print("weights:\n", model.weights, '\n')

# %%
output = model(example_input)

print("weights:\n", model.weights, '\n')


# %% [markdown]
# Notice that the model's weights changed *in the background* when we called `model(...)`. 
#
# In other words, the model didn't just cause the input to change into the output, it also had a *side effect* that altered the model itself. If we couldn't read the source code of the model, or if we didn't check that the weights had stayed the same after every model call, we wouldn't even know this had happened. 
#
# **How can we rewrite the same model, but without side effects?**
#
# First, we'll impose a constraint on ourselves that will force us not to unintentionally cause side-effects to happen.

# %%
@dataclass(frozen=True)
class FrozenModel:
    weights: np.ndarray
    
    def __call__(self, input_: np.ndarray):
        output = self.weights @ input_    
        return output


# %% [markdown]
# By passing `frozen=True` to `dataclass`, Python will make sure that any models we build using this class will be *immutable*. That means once a model is built, we can't make any changes to its parameters (in this case, `model.weights`). 
#
# What happens if we try?

# %%
model = FrozenModel(weights)

# try to replace the weights with zeros
model.weights = np.zeros_like(weights)

# %% [markdown]
# This might seem like a problem. Usually, we need to change the parameters of models at some point in our experiments. 
#
# So what do we do? 
#
# When we need to change the parameters, we construct a new model!

# %%
model = FrozenModel(weights)

output = model(example_input)

if np.sum(output) > 0:
    model = FrozenModel(-model.weights)  


# %% [markdown]
# Instead of the model self-modifying in the background like before, now we've insisted that the model has to be explicitly replaced with an updated model.
#
# Of course, if modifying the model is something we want to do repeatedly, we can write a function to do it:

# %%
def update_model(model: FrozenModel, output: np.ndarray) -> FrozenModel:
  
    if np.sum(output) < 0:
        return FrozenModel(-model.weights)
    else:
        return model  


# %% [markdown]
# And our previous code becomes:

# %%
model = FrozenModel(weights)

output = model(example_input)

model = update_model(model, output)


# %% [markdown]
# Notice that the logic of our program is now cleanly separated in two parts: 
#
# 1. `StatelessModel` which defines a linear transformation
# 2. `update_model` which defines how to change one linear transformation into another one, when `output` has a certain value
#
# There are no unexpected side effects, only functions that define transformations from inputs to outputs!

# %% [markdown]
# Why do we care?
#
# 1. **JAX is fast because it can compile your models**, which means it can find ways to convert your code into a form which runs much more efficiently on your hardware. For this to work, **your models must have clearly-defined input-output logic**. That means *no side effects*.
#
# 2. Writing with functional constraints **helps us not to become confused** about what our programs are doing in the background. When we write larger programs with more moving parts, this can become very important.
#
# <div class="alert alert-block alert-warning">
# <i>How can PyTorch make things fast without being purely functional?</i> 
#
# The answer is basically <b>it can't</b>. In the background, PyTorch uses your code to build a computational graph, which is functional.
#
# PyTorch allows for a more flexible coding style, but the lack of constraints means that how we organize our code may end up being not very aligned with the logic of the resulting computational graph. In some cases, this makes it harder to analyze and debug.
# </div>

# %% [markdown]
# Note that Equinox modules automatically set `frozen=True`, so when we're using Equinox we are under these constraints:

# %%
class FrozenModel(eqx.Module):
    weights: np.ndarray
    
    def __call__(self, input_: np.ndarray):
        output = self.weights @ input_    
        return output  
    
    
model = FrozenModel(weights)

model.weights = np.zeros_like(weights)

# %% [markdown]
# ### `jax.numpy`

# %% [markdown]
# JAX arrays and array operations try to mimic the input-output behaviour of NumPy operations as closely as possible. 

# %%
import jax.numpy as jnp

# %% [markdown]
# This gives you access to many of the same operations as `import numpy as np`:

# %%
angles = jnp.array([jnp.pi, 2 * jnp.pi])

angles

# %%
jnp.cos(angles)

# %%
ones = jnp.ones((3, 3))

ones

# %%
jnp.sum(ones, axis=0)

# %%
steps = jnp.arange(5)

steps 

# %%
jnp.concatenate([steps, steps])

# %% [markdown]
# These functions behave like NumPy functions, but they are internally constructed so that JAX can compile them into much faster code, as we'll see shortly.
#
# Finally, note that JAX arrays are interchangeable with NumPy arrays in most cases. We can pass a JAX array to a NumPy or matplotlib function, no problem. 
#
# This contrasts with PyTorch tensors which must be explicitly converted to NumPy arrays.

# %%
# `ones` is a JAX array, but this just works 
np.sum(ones, axis=0)

# %%
ones_tensor = torch.ones((3, 3))

np.sum(ones_tensor, axis=0)

# %%
np.sum(ones_tensor.numpy(), axis=0)
# Sometimes it is necessary to do even grosser things, 
# like `ones_tensor.clone().detach().numpy()`.

# %% [markdown]
# ### `jax.random`
#
# One major difference in how JAX behaves compared to NumPy is (pseudo) random number generation. 
#
# What happens when we generate a random number in NumPy?

# %%
np.random.uniform()

# %%
# different!
np.random.uniform()

# %% [markdown]
# The number was different the second time! That means *np.random is stateful*! When we called `np.random.uniform`, something changed in the background that caused the result of the exact same operation to be different the second time around. 
#
# NumPy does allow us to explicitly refer to a random number generator and tell it what seed to start from. This makes the process *reproducible*:

# %%
seed = 1234

random = np.random.default_rng(seed)
random.uniform()

# %%
# start from same seed -> same result
random = np.random.default_rng(seed)
random.uniform()

# %%
# but it's still stateful!
random.uniform()

# %% [markdown]
# We can control where the random generator starts, but it's still stateful. If we want things to stay reproducible as we edit our code, we need to either:
#
# 1. Make sure the random generator is called the same number of times.
# 2. Explicitly control the seed before every call we make to the generator. 
#
# The situation is similar in PyTorch:

# %%
torch.manual_seed(1234)
torch.rand(1)

# %%
torch.manual_seed(1234)
torch.rand(1)

# %%
torch.rand(1)

# %% [markdown]
# **In JAX, random number generation is purely functional**. That means that the results of an identical function call are always identical. 
#
# It also means we always need to explicitly pass a random key to functions that generate random numbers, either because they return them directly (like `jr.uniform`) or because they use the random numbers as part of some larger computation.

# %%
import jax.random as jr

seed = 1234
key = jr.PRNGKey(seed)

jr.uniform(key)

# %%
# identical!
jr.uniform(key)

# %% [markdown]
# When we want to generate a new random number, we split the key into one or more new keys:

# %%
# Returns 2 keys by default. Both are new, but we just need the first one.
key_next, _ = jr.split(key)

jr.uniform(key_next)


# %% [markdown]
# This forces us to always be clear about the logic of how we are generating random numbers inside our program. 
#
# For example, let's say we want to generate some uniform and Gaussian data:

# %%
def generate_data(key):
    key_uniform, key_normal = jr.split(key)
    data_uniform = jr.uniform(key_uniform, (2, 4))
    data_normal = jr.normal(key_normal, (2, 4))
    return data_uniform, data_normal


# %% [markdown]
# Note that we only need to pass in a single key, and the program logic handles the rest. This is usually what we do: generate a single key at the top level of our program, then let the key splits handle the rest.

# %%
key = jr.PRNGKey(seed)

generate_data(key)

# %%
# we didn't change the key, so this is identical
generate_data(key)

# %% [markdown]
# ### PyTrees

# %% [markdown]
# This is the single feature that really sets JAX apart from other frameworks.
#
# A PyTree is just a way of nesting data, arbitrarily. Any of the standard Python containers (lists, tuples, and dicts), nested in any arbitrarily format, count as PyTrees. 
#
# Here are some examples of PyTrees:

# %%
tree1 = [1, 2, 3]  # a simple list of integers

tree2 = ("boop",) # a tuple with one string element

tree3 = {"a": 1, "b": 2}  # a dictionary

tree4 = {"a": [1, 2, 3], "b": (1, 2, 3)}  # a dictionary with lists and tuples

tree5 = [(jnp.ones, jnp.zeros), 
         (torch.ones, torch.zeros)]  # a list of tuples of functions

tree6 = {"a": {"b": {"c": 0}}}  # a nested dictionary

# and so forth

# %% [markdown]
# We don't need to call any JAX functions to turn these into PyTrees. They are PyTrees simply by virtue of being arbitrary arrangements of Python containers *into trees*. 
#
# What's important is that JAX provides some extremely useful tools for working with *any* kind of PyTree that happens to be convenient for our purposes.
#
# Let's visualize what we mean by a "tree":

# %%
# we'll re-use our old key here because reproducibility doesn't matter
tree = {
    "a": jr.uniform(key, (2,)),
    "b": {
        "c": jr.normal(key, (2,)),
        "d": (1, 2),
    }
}

# Equinox provides a nice function for printing PyTrees more cleanly
print_tree = lambda tree: eqx.tree_pprint(tree, short_arrays=False, width=50)

print_tree(tree)

# %% [markdown]
# By default, JAX considers this dictionary to have the following tree structure:

# %% [markdown]
# <img src="jax_tree_example.png"></img>

# %% [markdown]
# Note that `1` and `2` are *leaves* of the tree. The entire tuple `(1, 2)` is not a leaf, by default. This is because tuples, like other standard Python containers, are taken as the *nodes* of a tree. The nodes are the parts that contain things, while the leaves are the parts that are contained. And by default, NumPy/JAX arrays, numbers (like `1` or `4.5` or `1e-7`), and other non-container types like strings and functions, are considered the leaves of a tree.
#
# JAX can flatten a PyTree and give us back the leaves:

# %%
import jax.tree_util as jtu

jtu.tree_leaves(tree)

# %% [markdown]
# One of the most powerful PyTree operations provided by JAX is `tree_map`:

# %%
summed = jtu.tree_map(jnp.sum, tree)

print_tree(summed)

# %% [markdown]
# In this case, we applied `jnp.sum` to each individual leaf of the tree, and *returned exactly the same tree structure*.
#
# We can also use `tree_map` to apply a function that takes multiple arguments:

# %%
tree1 = [("Hello ", "How "), "Love, "]

tree2 = [("world!", "are you?"), "Moon"]

concatenated = jtu.tree_map(
    lambda x, y: x + y, 
    tree1,
    tree2,
)

print_tree(concatenated)


# %% [markdown]
# Finally, keep in mind that *we can define our own containers that JAX will treat as PyTrees*. 
#
# For example, dataclasses aren't PyTrees in JAX by default, but it is straightforward to tell JAX how to treat them as PyTrees. 
#
# This is exactly what Equinox does. All Equinox modules are PyTrees!

# %%
class NestedModel(eqx.Module):
    weights1: np.ndarray
    weights2: np.ndarray
    frozen_model: FrozenModel
    
    def __call__(self, input_: np.ndarray):
        ...
    
    
nested_model = NestedModel(
    jr.uniform(key, (2, 3)),
    jr.uniform(key, (3, 5)),
    FrozenModel(jr.uniform(key, (1, 3))),
)

print_tree(nested_model)

# %%
# get the shapes of all the parameter arrays
jtu.tree_map(lambda x: x.shape, nested_model)

# %%
# negate *all* the weights
negated_model = jtu.tree_map(lambda x: -x, nested_model)

print_tree(
    negated_model
)

# %% [markdown]
# Much of the power of JAX comes from being able to pass around our data as part of tree structures, and process them with functions like `tree_map`. 

# %% [markdown]
# ### The core JAX transformations

# %% [markdown]
# At last, let's import the core JAX package.

# %%
import jax


# %% [markdown]
# There are three functions that are at the core of how we use JAX for computational models.

# %% [markdown]
# #### `jax.grad` 

# %% [markdown]
# This is how we take gradients, i.e. derivatives.

# %%
def f(x):
    return x ** 2 


# %% [markdown]
# We know that 
#
# $$\frac{d}{dx} x^2 = 2x$$
#
# but what does JAX say?

# %%
grad_f = jax.grad(f)

x = 3.0
grad_f(x)

# %% [markdown]
# OK! And we know the second derivative of $x^2$ is just $2$. 
#
# JAX?

# %%
jax.grad(jax.grad(f))(x)

# %% [markdown]
# In complex models, functions are usually composed. For example, maybe we want to find the square of the sine:

# %%
g = jnp.sin

composed = lambda x: f(g(x))

# %% [markdown]
# By the chain rule, we know 
#
# $$\frac{d}{dx} f(g(x)) = \frac{df}{dx}\left(g(x)\right)\cdot\frac{dg}{dx}(x)$$
#
# And since we know the derivatives of $x^2$ and $\sin(x)$ are respectively $2x$ and $\cos(x)$:
#
# $$\frac{d}{dx} (\sin(x))^2 = 2 \sin(x)\cdot \cos(x)$$
#
# Like other machine learning frameworks, JAX can automatically figure this out, without our needing to do the math every time:

# %%
x = 3.0
jax.grad(composed)(x)

# %%
# and verify with the known solution
2 * jnp.sin(x) * jnp.cos(x)


# %% [markdown]
# Now let's see how this plays with PyTrees. Let's say we write our function in terms of data stored in a PyTree:

# %%
class State(eqx.Module):
    x: float 
    y: float

def f(state):
    return state.x ** 2 + state.y ** 3


# %% [markdown]
# We can use `jax.grad` on this, just like any other function that returns a single number.

# %%
grad_f = jax.grad(f)  

state = State(x=3.0, y=4.0)

gradient = grad_f(state)

print_tree(gradient)

# %% [markdown]
# Notice that JAX returns the gradients in a PyTree with exactly the same structure as the one we used!

# %%
gradient.x  # this is df/dx

# %% [markdown]
# ### `jax.vmap`

# %% [markdown]
# This function allows us to *vectorize* our computations, that is, to apply them to entire batches of data at once.
#
# For example, let's square all the integers from 0 to 9:

# %%
# the integers from 0..9
xs = jnp.arange(10, dtype=float)

f = lambda x: x**2
ys = jax.vmap(f)(xs)

ys

# %% [markdown]
# We can also vmap the gradient:

# %%
grad_f = jax.grad(f)
dy_dxs = jax.vmap(grad_f)(xs)  

# just 2*x
dy_dxs


# %% [markdown]
# PyTrees and vmap play well together. 
#
# For example, let's see a function that generates a model with some random weights

# %%
class Model(eqx.Module):
    weights1: jnp.ndarray 
    weights2: jnp.ndarray
    scalar_param: float = 2.0
    
    def __call__(self, input_):
        ...


def get_model(key):
    
    key1, key2 = jr.split(key)
    
    return Model(
        weights1=jr.uniform(key1, (2, 2)), 
        weights2=jr.normal(key2, (3, 3)),
    )  


# %%
seed = 1234
key = jr.PRNGKey(seed)

model = get_model(key)

model

# %% [markdown]
# We can use vmap to generate several models in parallel:

# %%
n_models = 4
keys = jr.split(key, n_models)
models = jax.vmap(get_model)(keys)

models

# %%
# because we passed a key split, all the weights are different
models.weights1 

# %% [markdown]
# There is no extra overhead: instead of storing multiple `Model` instances, say in a list, JAX is able to look inside the model's PyTree and figure out where to add extra dimensions to the model to represent all the different instantiations.
#
# But look at the PyTree again: `scalar_param` is constant! 

# %%
models.scalar_param

# %% [markdown]
# What's the point of storing it multiple times? We can explicitly tell JAX not to do that, but Equinox provides a function that will conveniently and automatically filter out the leaves of the PyTree that are invariant to the vmap:

# %%
models = eqx.filter_vmap(get_model)(keys)

models


# %% [markdown]
# Finally, `jax.vmap` also plays well with `jax.grad`. 
#
# Note that we now use the filtered Equinox versions of these functions, because they'll be able to treat `scalar_param` as a constant.

# %%
def weights_l2(model):
    return jnp.sum(model.weights1 ** 2) + jnp.sum(model.weights2 ** 2)

grad_weights_l2 = eqx.filter_grad(weights_l2)  

eqx.filter_vmap(grad_weights_l2)(models)


# %% [markdown]
# ### `jax.jit`

# %% [markdown]
# When we want to compile our code and make it run faster, we use `jax.jit`. 
#
# JIT stands for just-in-time compilation.
#
# For example, consider a function that generates and multiplies some matrices.

# %%
def multiply_matrices(key, shape1, shape2):
    key1, key2 = jr.split(key)
    mat1 = jr.normal(key1, shape1)
    mat2 = jr.normal(key2, shape2)
    return jnp.matmul(mat1, mat2)


# %% [markdown]
# We use `jax.vmap` to repeatedly apply this function for many trials, and time how long this takes:

# %%
# shape1 and shape2 are fixed, so we use `in_axes` not to vmap over them
multiply_many_matrices = jax.vmap(multiply_matrices, in_axes=(0, None, None))

n_trials = 100
seed = 1234
shape1 = (10, 500)
shape2 = shape1[::-1]  
keys = jr.split(jr.PRNGKey(seed), n_trials)

# %timeit multiply_many_matrices(keys, shape1, shape2).block_until_ready()

# %% [markdown]
# Note that JAX runs *asynchronously*: it'll pass your instructions to the hardware, then move onto your next line of code, before the hardware finishes its computation. Every JAX array has a method `block_until_ready()` that keeps JAX from moving on until the array's computation is complete---this prevents us from prematurely exiting the timing loop.
#
# Now we'll apply `jax.jit` to speed up this function.

# %%
multiply_many_matrices_jit = jax.jit(multiply_many_matrices, static_argnums=(1, 2))

# %time multiply_many_matrices_jit(keys, shape1, shape2).block_until_ready();

# %% [markdown]
# The function compiles the first time we run it, which is why this first run took much longer than the average runtime of the uncompiled function.
#
# But what if we run the function again, now that it's compiled?

# %%
# %timeit multiply_many_matrices_jit(keys, shape1, shape2).block_until_ready()

# %% [markdown]
# This is much faster than the uncompiled function.
#
# JIT plays well with other JAX transformations, so (for example) we can do
#
# `jit(grad(grad(jit(jit(grad(some_scalar_function))))))`
#
# without issue. However, from a design perspective, when we are designing a large stack of models, we only need to run JIT once, at the top level:
#
# `jit(grad(grad(grad(some_scalar_function))))`
#
# This is sufficient to compile everything that happens inside. Finally, note that there are some important limitations on what can and cannot be compiled by JAX, but that's beyond the scope of this tutorial.

# %% [markdown]
# When training a neural network, a typical place to apply `jax.jit` would be around the function `train_step`, in which:
#
# - The model is vmapped over a batch of training data:    
#     `outputs = jax.vmap(model)(batch_of_train_inputs)`     
#     
# - We take the gradient of the loss function:     
#     `grads = jax.grad(loss_func)(outputs, targets)`    
#     
# - The model is updated based on the gradient information:     
#     `model = update(model, grads)`    
#     

# %% [markdown]
# ### Using `jax.eval_shape`

# %% [markdown]
# Sometimes a function is expensive to compute, and before we start to compute anything, we want to infer what the shapes of the function's return values will be. 
#
# This is what `jax.eval_shape` is for. For example, consider a simple recurrent neural network class:

# %%
import jax
import equinox as eqx
import jax.random as jr
import jax.numpy as jnp


# %%
class RNN(eqx.Module):
    cell: eqx.nn.GRUCell

    def __init__(self, **kwargs):
        self.cell = eqx.nn.GRUCell(**kwargs)

    def __call__(self, xs):
        scan_fn = lambda state, input: (self.cell(input, state), None)
        init_state = jnp.zeros(self.cell.hidden_size)
        final_state, _ = jax.lax.scan(scan_fn, init_state, xs)
        return final_state


# %% [markdown]
# Let's say we want to `vmap` the RNN over many examples. We can use `jax.eval_shape` to figure out the shape of the returned array (`final_state`) without running a potentially expensive computation.

# %%
seed = 1234
key = jr.PRNGKey(seed)
rnn = RNN(input_size=50, hidden_size=1000, key=key)

n_examples = 50000
n_steps = 100
examples = jr.uniform(key, (n_examples, n_steps, 50))
    
rnn_batched = jax.vmap(rnn)

jax.eval_shape(jax.vmap(rnn), examples)

# %% [markdown]
# How much faster is this than running the actual calculation?

# %%
# just run this once since it's kind of slow
# %time rnn_batched(examples).block_until_ready()

# %%
# %timeit jax.eval_shape(rnn_batched, examples)
