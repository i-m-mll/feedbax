
from dataclasses import dataclass
from typing import Protocol, Tuple, runtime_checkable

import jax.numpy as jnp
from jaxtyping import Array, Float, Int, jaxtyped
from typeguard import check_type, typechecked

# typing: can we use a protocol to require a class with specific fields?

@dataclass 
class DataClass:
    a: int
    b: Float[Array, "2 2"]
    c: Int[Array, "2"]
    
@dataclass 
class DataClassNonConforming: 
    b: int 
    c: int 
    d: int 

@runtime_checkable
class ProtocolClass(Protocol):
    a: int
    b: Float[Array, "2 2"]
    c: Int[Array, "2"]

#@jaxtyped
@typechecked
def test_protocolclass(pc: ProtocolClass):
    # print(isinstance(pc.b, Float[Array, "2 2"]))
    # check_type(pc.c, Int[Array, "2"])
    return pc.b

@typechecked
def test_protocolclass_nonconforming(pc: ProtocolClass):
    return pc.woo  #! this should irritate mypy

# unfortunately mypy seems insensitive to the incorrect shape and dtype of the array
data: Tuple[int, Array, int] = (1, jnp.zeros((2, 2), dtype=float), jnp.ones((2,), dtype=int))
data2: Tuple[int, int, int] = (1, 4, 8)

datacls = DataClass(*data) 
datacls_nc1 = DataClassNonConforming(*data)  #! this should too
datacls_nc = DataClassNonConforming(*data2)  

test_protocolclass(datacls)
test_protocolclass(datacls_nc)  #! this should too
test_protocolclass_nonconforming(datacls_nc)  #! this should too

    
