

### Basic model structure, purely functional vs. stateful

import numpy as np

weights = np.random.random((2, 10))
input_ = np.random.random((10,)) 

# output = weights @ input_

import torch

class PytorchModel(torch.nn.Module):
  def __init__(self, weights: torch.Tensor): 
    self.weights = weights
  
  def forward(self, input_: torch.Tensor):
    return self.weights @ input_  


model = PytorchModel(torch.tensor(weights))
output = model(torch.tensor(input_))

class Model:
  def __init__(self, weights: np.ndarray): 
    self.weights = weights
  
  def __call__(self, input_: np.ndarray):
    return self.weights @ input_ 

# part of the Python standard library
from dataclasses import dataclass 

@dataclass
class DataclassModel:
  weights: np.ndarray
  
  def __call__(self, input_: np.ndarray):
    return self.weights @ input_ 


model = Model(weights)
output = model(input_)  



class StatefulModel:
  weights: np.ndarray
  
  def __call__(self, input_: np.ndarray):
    output = self.weights @ input_
    
    if np.sum(output) < 0:
      self.weights = -self.weights
    
    return output

model = StatefulModel(weights)
  
# this silently modifies `model.weights`
output = model(input_)  


class StatelessModel:
  weights: np.ndarray
  
  def __call__(self, input_: np.ndarray):
    output = self.weights @ input_    
    return output

model = StatelessModel(weights)

output = model(input_)  

if np.sum(output) < 0:
  model = StatelessModel(-model.weights)  
  
  
def update_model(
  model: StatelessModel, 
  output: np.ndarray
) -> StatelessModel:
  
  if np.sum(output) < 0:
    model = StatelessModel(-model.weights)
  
  return model    


### Equinox and Diffrax

import jax.numpy as jnp


import equinox as eqx


class ImmutableModel(eqx.Module):
  A: jnp.ndarray
  
model = ImmutableModel(A=jnp.zeros((2, 2)))

model.A = jnp.ones((2, 2))  # raises an error


class LTISystem(eqx.Module):
    A: jnp.ndarray
    B: jnp.ndarray
    
    def vector_field(self, t, state, input_):
        d_state = self.A @ state + self.B @ input_
        return d_state
    
    
ORDER = 2
N_DIM = 2

def point_mass(mass=1):
    A = sum([jnp.diagflat(jnp.ones((ORDER - i) * N_DIM), i * N_DIM)
             for i in range(1, ORDER)])
    
    B = jnp.concatenate(
      [
        jnp.zeros((N_DIM, N_DIM)), 
        jnp.eye(N_DIM) / mass
      ], 
    axis=0)
    
    return LTISystem(A, B)


import diffrax as dfx

def solve(solver, vector_field, state0, t0, t1, steps, dt0, input_):
    term = dfx.ODETerm(vector_field)
    saveat = dfx.SaveAt(ts=jnp.linspace(t0, t1, steps))
    sol = dfx.diffeqsolve(
      term, solver, t0, t1, dt0, state0, args=input_, saveat=saveat
    )
    return sol

sys = point_mass(mass=1)

state0 = jnp.array([0., 0., 0.5, 0.])  
input_ = jnp.array([-0.5, 0.5])
t0 = 0 
t1 = 1
dt0 = 0.01  

sol = solve(dfx.Euler, sys.vector_field, state0, t0, t1, dt0, input_)   
# the solution trajectory is accessed by `sol.ts` and `sol.ys`



### jax.random

import jax.random as jr

seed = 1234
key = jr.PRNGKey(seed)

def generate_data(key):
  key_uniform, key_normal = jr.split(key)
  data_uniform = jr.uniform(key_uniform, (1000, 10))
  data_normal = jr.normal(key_normal, (1000, 10))
  return data_uniform, data_normal



### PyTrees


import jax.tree_util as jtu

tree = dict(
  a=jr.uniform(key, (10,)),
  b=dict(
    c=jr.normal(key, (10,)),
    d=(1, 2),
  )
)

jtu.tree_map(jnp.sum, tree)

tree1 = tree2 = None 

jtu.tree_map(lambda x, y: x + y, tree1, tree2)



### ensemble
import jax

def get_model(
    task,
    dt: float = 0.05, 
    hidden_size: int = 50, 
    n_steps: int = 50, 
    feedback_delay: int = 0, 
    *,
    key: jnp.ndarray = None,
):
    ...
    
    return model