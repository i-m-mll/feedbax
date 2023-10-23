"""Maybe we can use descriptors to share property computations
between state classes. Here's an example.

This might not make sense since `diffrax` might be confused by the 
extra PyTree leaf. But perhaps some kind of filtering would work?

We could just make the forward kinematics a method of `NLink` and then
inherit into `TwoLink` (because `TwoLink` has methods that can't be implemented
exactly in the n-link case) but we'd have to be careful to respect the
concrete-final rule. So separate `AbstractNLink` and `NLink` classes,
if we wanted a concrete n-link class.
"""

from jaxtyping import Array, Float, PyTree


class CartesianState2D:
    pos: Float[Array, "2"]
    vel: Float[Array, "2"]


class AngularState2D:
    theta: Float[Array, "2"]
    d_theta: Float[Array, "2"]


class NLinkForwardKinematics:
    def __get__(
        self, 
        instance: AngularState2D, 
        owner,
    ) -> CartesianState2D:
        # compute the forward kinematics transformation
        return 


class NLinkState:
    theta: Float[Array, "links"]
    d_theta: Float[Array, "links"]
    effector: CartesianState2D = NLinkForwardKinematics()
    

class TwoLinkState:
    theta: Float[Array, "2"]
    d_theta: Float[Array, "2"]
    effector: CartesianState2D = NLinkForwardKinematics()
