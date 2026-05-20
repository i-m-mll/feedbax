import plotly.io as pio

from .colors import (
    adjust_color_brightness,
    sample_colorscale_unique,
)
from .io import save_figure, save_figure_with_spec
from .misc import AxesLabels
from .plotly import loss_history, loss_history_compare
from .profiles import profiles
from .trajectories import (
    trajectories,
    trajectories_2D,
    trajectories_3D,
)

pio.templates.default = "plotly_white"
