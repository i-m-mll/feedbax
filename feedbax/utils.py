
import math

def exp_taylor(x: float, n: int):
    """First `n` terms of the Taylor series for `exp` at the origin.
    """
    return [(x ** i) / math.factorial(i) for i in range(n)]