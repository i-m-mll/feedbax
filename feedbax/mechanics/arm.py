""" """


import jax.numpy as jnp
from jaxtyping import Float, Array

from feedbax.utils import sincos_derivative_signs


class TwoLink:
    l: Float[Array, "2"] = jnp.array((0.30, 0.33))  # [m] lengths of arm segments
    m: Float[Array, "2"] = jnp.array((1.4, 1.0))  # [kg] masses of segments
    I: Float[Array, "2"] = jnp.array((0.025, 0.045))  # [kg m^2] moments of inertia of segments
    s: Float[Array, "2"] = jnp.array((0.11, 0.16))  # [m] distance from joint center to segment COM
    B: Float[Array, "2 2"] = jnp.array(((0.05, 0.025),
                                        (0.025, 0.05))) # [kg m^2 s^-1] joint friction matrix
    inertia_gain: float = 1.    
    
    @property
    def a1(self):
        return self.I[0] + self.I[1] + self.m[1] * self.l[0] ** 2 # + m[1]*s[1]**2 + m[0]*s[0]**2
    
    @property
    def a2(self):
        return self.m[1] * self.l[0] * self.s[1]
    
    @property
    def a3(self):
        return self.I[1]  # + m[1] * s[1] ** 2
    
    def field(t, y, args):
        theta, d_theta = y 
        input_torque, twolink = args

        # centripetal and coriolis torques 
        c_vec = jnp.array((
            -d_theta[1] * (2 * d_theta[0] + d_theta[1]),
            d_theta[0] ** 2
        )) * twolink.a2 * jnp.sin(theta[1])  
        
        # inertia matrix that maps torques -> angular accelerations
        cs1 = jnp.cos(theta[1])
        tmp = twolink.a3 + twolink.a2 * cs1
        inertia_mat = jnp.array(((twolink.a1 + 2 * twolink.a2 * cs1, tmp),
                                    (tmp, twolink.a3 * jnp.ones_like(cs1))))
        
        net_torque = input_torque - c_vec.T - jnp.matmul(d_theta, twolink.B.T)
        
        dd_theta = jnp.linalg.inv(inertia_mat) @ net_torque
        
        return d_theta, dd_theta


def nlink_angular_to_cartesian(theta, dtheta, nlink):
    angle_sum = jnp.cumsum(theta)  # links
    length_components = nlink.l * jnp.array([jnp.cos(angle_sum),
                                             jnp.sin(angle_sum)])  # xy, links
    xy_position = jnp.cumsum(length_components, axis=1)  # xy, links
    
    ang_vel_sum = jnp.cumsum(dtheta)  # links
    xy_velocity = jnp.cumsum(jnp.flip(length_components, (0,)) * ang_vel_sum
                             * sincos_derivative_signs(1),
                             axis=1)
    return xy_position, xy_velocity