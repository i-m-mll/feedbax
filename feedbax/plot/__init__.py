import plotly.io as pio

from .colors import (
    adjust_color_brightness,
    sample_colorscale_unique,
)
from .io import save_figure, save_figure_with_spec
from .misc import AxesLabels
from .constructors import (
    constructor_catalog,
    get_figure_constructor,
    get_figure_piece,
    get_figure_template,
    register_figure_constructor,
    register_figure_piece,
    register_figure_template,
    registered_figure_constructors,
    registered_figure_pieces,
    registered_figure_templates,
)
from .plotly import loss_history, loss_history_compare
from .profiles import profiles
from .trajectories import (
    trajectories,
    trajectories_2D,
    trajectories_3D,
)

pio.templates.default = "plotly_white"
