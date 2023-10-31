from typing import Protocol, Type, runtime_checkable

import equinox as eqx 
import jax
import jax.numpy as jnp
import jax.random as jr
from typeguard import check_type, typechecked

@runtime_checkable
class RNNCell(Protocol):
    hidden_size: int
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        use_bias: bool, 
        *, 
        key: jax.Array,
        **kwargs, 
    ):
        ...
    
    def __call__(
        self, 
        input: jax.Array, 
        state: jax.Array, 
    ) -> jax.Array:
        ...

class Test(eqx.Module):
    cell: eqx.Module
    
    def __init__(
        self, 
        cell_type: Type[RNNCell], 
        input_size: int, 
        hidden_size: int, 
        **kwargs
    ):
        self.cell = cell_type(input_size, hidden_size, use_bias=False, key=jr.PRNGKey(0), **kwargs)
        
class TestCell(eqx.Module):
    #! no hidden_size field! 
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        use_bias: bool, 
        *, 
        key: jax.Array,
        **kwargs, 
    ):
        return 

@typechecked 
def get_gru_test():
    Test(eqx.nn.GRUCell, 10, 20)
    
@typechecked 
def get_other_test():
    Test(TestCell, 10, 20)
    
get_gru_test() 
get_other_test()  #! this should irritate typeguard

check_type(eqx.nn.GRUCell, Type[RNNCell])