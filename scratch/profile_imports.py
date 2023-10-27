from datetime import datetime
from pathlib import Path
import sys
from typing import Optional

import diffrax as dfx
import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp 
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
import optax 
from torch.utils.tensorboard import SummaryWriter
import tqdm
from tqdm import tqdm

from feedbax.mechanics.arm import (
    nlink_angular_to_cartesian, 
    twolink_effector_pos_to_angles
)
from feedbax.mechanics.muscle import (
    ActivationFilter,
    LillicrapScottVirtualMuscle,
    TodorovLiVirtualMuscle, 
)    
from feedbax.mechanics.muscled_arm import TwoLinkMuscled 
from feedbax.mechanics.system import System
from feedbax.networks import RNN
from feedbax.plot import (
    plot_loglog_losses, 
    plot_2D_joint_positions,
    plot_pos_vel_force_2D,
    plot_activity_heatmap,
)
from feedbax.task import centreout_endpoints, uniform_endpoints
from feedbax.utils import (
    delete_contents,
    internal_grid_points,
    tree_get_idx, 
    tree_set_idx, 
    tree_sum_squares,
)