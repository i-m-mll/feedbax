from dataclasses import dataclass
from typing import NamedTuple

class TestNT(NamedTuple):
    a: int 
    b: float 
    c: jnp.array 

@dataclass 
class TestDC:
    a: int 
    b: float 
    c: jnp.array

data = (1, 2., jnp.zeros(100))    

%timeit TestDC(*data)

%timeit TestNT(*data)

%timeit tuple(data)