""" """


from typing import Tuple

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Array


class VirtualMuscle(eqx.Module):
    """Muscle model from Brown et al. 1999."""
    beta: float 
    omega: float
    rho: float
    v_max: float
    c_v: Tuple[float, float]
    a_v: Tuple[float, float, float]
    b_v: float
    n_f: Tuple[float, float]
    a_f: float
    c1: float
    c2: float
    k1: float
    k2: float
    l_r1: float
    l_r2: float

    def force(self, l, v, a):
        force_l = self.force_length(l)
        force_v = self.force_velocity(l, v)
        force_pe1, force_pe2 = self.force_passive(l)
        A_f = self.activation_frequency(a, l)
        # assumes 100% fibre recruitment, linear factor R=1:
        force = A_f * (force_l * force_v + force_pe2) + force_pe1  
        return force

    def force_length(self, l):
        return jnp.exp(-jnp.abs((l ** self.beta - 1) / self.omega) ** self.rho)
    
    def force_velocity(self, l, v):
        a_v, b_v, c_v, v_max = self.a_v, self.b_v, self.c_v, self.v_max
        f_lengthen = (b_v - v * (a_v[0] + a_v[1] * l + a_v[2] * l ** 2)) / (b_v + v)
        f_shorten = (v_max - v) / (v_max + v * (c_v[0] + c_v[1] * l))  # TODO: optional Hill approx without l dep
        lengthen_idxs = v > 0
        return lengthen_idxs * f_lengthen + ~lengthen_idxs * f_shorten
    
    def force_passive(self, l):
        c1, k1, l_r1 = self.c1, self.k1, self.l_r1
        f_pe1 = c1 * k1 * jnp.log(jnp.exp((l - l_r1) / k1) + 1)  
        
        c2, k2, l_r2 = self.c2, self.k2, self.l_r2
        f_pe2 = c2 * jnp.exp(k2 * (l - l_r2))  # TODO: optional Hill approx without l dep
        
        return f_pe1, f_pe2
    
    def activation_frequency(self, a, l):
        n_f, a_f = self.n_f, self.a_f
        n_f = n_f[0] + n_f[1] * (1 / l - 1)  # TODO: l_eff filter option (see method _l_eff_field)
        Y = 1  # TODO: Y filter option (see method _Y_field)
        A_f = 1 - jnp.exp(-((a * Y) / (a_f * n_f)) ** n_f)
        return A_f
    
    def _Y_field(self, t, y, args):
        Y = y
        v = args 
        c_Y, tau_Y, v_Y = None, None, None  # TODO: these should be fields
        d_Y = 1 - Y - c_Y * (1 - jnp.exp(-jnp.abs(v) / v_Y)) / tau_Y
        return d_Y
    
    def _l_eff_field(self, t, y, args):
        # TODO: to do this, need to remember A_f from last step...
        l_eff = y
        l, A_f = args 
        tau_l = None
        d_l_eff = (l - l_eff) ** 3 / (tau_l * (1 - A_f))
        return d_l_eff
    
    
#! I think both Lillicrap & Scott & Todorov & Li use averages of slow/fast twitch params
# TODO check!


# TODO: M is a property of the plant, not the muscle?
# TODO: serialize the defaults rather than hardcoding them in subclasses?
class LillicrapScott: #(eqx.Module): 
    M: Array = jnp.array(((2.0, -2.0, 0.0, 0.0, 1.50, -2.0), 
                          (0.0, 0.0, 2.0, -2.0, 2.0, -1.50)))  # [cm], apparently
    theta0: Array = 2 * jnp.pi * jnp.array(((15.0, 4.88, 0.0, 0.0, 4.5, 2.12), 
                                            (0.0, 0.0, 80.86, 109.32, 92.96, 91.52))) / 360. # [deg] TODO: radians
    l0: Array = jnp.array((7.32, 3.26, 6.4, 4.26, 5.95, 4.04))  # [m]
    
    beta = 1.55
    omega = 0.81
    rho = 1.0  # not used
    vmax = -7.39
    cv0 = -3.21
    cv1 = 4.17
    av0 = -3.12
    av1 = 4.21
    av2 = -2.67
    bv = 0.62
    
    #! 1. leaves out the dependency on l in f_shorten
    #! 2. uses A_f = a = u^+ (smoothed rectifier though)
    #! 3. omits passive forces
    
    
class TodorovLi:
    
    pcsa = jnp.array((18, 14, 22, 12, 5, 10))  # [cm^2]
    
    beta = 1.93
    omega = 1.03
    rho = 1.87
    vmax = -5.72
    cv0 = 1.38
    cv1 = 2.09
    av0 = -3.12
    av1 = 4.21
    av2 = -2.67
    bv = 0.62
    
    nf0 = 2.11
    nf1 = 4.16
    a_f = 0.56
    c2 = -0.02
    k2 = -18.7
    l_r2 = 0.79
    tmp1 = -k2 * l_r2
    
    #! 1. omits f_PE1 and slightly modifies f_PE2 (omits the constant term?)
    #! 2. computes A_f but leaves out the Y and l_eff filters
    
    
class ActivationFilter(eqx.Module):
    """First-order filter to model calcium dynamics of muscle activation."""
    tau_act: float = 50  # [ms]
    tau_deact: float = 66  # [ms]
    
    def __init__(self):
        self.tau_diff = self.tau_act - self.tau_deact  # assumes these are constant
    
    def field(self, t, y, args):
        activation = y 
        u = args 
        
        tau = self.tau_deact + jnp.where(u < activation, u, jnp.zeros(1)) * self.tau_diff
        d_activation = (u - activation) / tau
        
        return d_activation