import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp 
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
import optax 

from feedbax.channel import ChannelState
from feedbax.context import SimpleFeedback, SimpleFeedbackState
import feedbax.loss as fbl
from feedbax.mechanics import Mechanics 
from feedbax.mechanics.linear import point_mass
from feedbax.networks import RNN
from feedbax.plot import plot_loglog_losses, plot_pos_vel_force_2D
from feedbax.recursion import Recursion
from feedbax.task import RandomReaches
from feedbax.trainer import TaskTrainer, save, load