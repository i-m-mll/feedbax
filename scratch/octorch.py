"""Scrap code from OCtorch."""


### octorch.skeletons.arm

def reachable_grid(
        self,
        n_points: int | Sequence[int] = 10,
        dtype: torch.dtype = torch.float32
    ) -> Tensor:  #? TensorType['segment', 'grid_points']
        """Returns """
        if isinstance(n_points, int):
            n_points = [n_points] * self.n_segments
        try:
            n_points[self.n_segments - 1]
        except IndexError:
            raise ValueError('if `n` is a sequence, must have same length as number of arm segments')
        grid_tensors = torch.meshgrid(*[
            torch.linspace(bound_lo, bound_hi, steps=n).to(dtype)
            for (bound_lo, bound_hi), n in zip(self.position_bounds.T, n_points)
        ])
        # flatten out the grid structure; only need to iterate the points
        grid_points = torch.stack([t.reshape(-1) for t in grid_tensors])
        return grid_points


def jacobian_end(self, state):
    angles = state.position
    J_col1 = torch.tensor([-self.l[1] * torch.sin(angles[0] + angles[1]),
                            self.l[1] * torch.cos(angles[0] + angles[1])])
    J_col0 = torch.tensor([-self.l[0] * torch.sin(angles[0]),
                            self.l[0] * torch.cos(angles[0])]) + J_col1
    return torch.stack([J_col0, J_col1], dim=1)


### octorch.data

class CenterOutPoints2D(Dataset[Endpoints2D]):
    """Start points at center, end points outwards in uniform directions.

    The prob

    Args:
        n_directions: Number of evenly-spaced directions around center.
        center: Cartesian start point in every pair of endpoints.
        length: Displacement of end points from center point.
            Can either be a constant, or a callable (e.g. `torch.rand`).
        n_samples: Number of endpoint pairs in the dataset.
            If `None`, includes each direction exactly once.
            If an integer, samples from the directions that many times.
        angle_offset: Rotates all of the evenly-spaced directions (radians).
        p: The probability of sampling each direction.
    """
    
### octorch.integrators 

def euler(
    state: DataTree,
    dstate_dt_fn: Callable,
    dt: float,
):
    """Simple first-order projection."""
    dstate_dt = dstate_dt_fn(state)
    return _step(state, dstate_dt, dt)


def rk4(
    state: DataTree,
    dstate_dt_fn: Callable,
    dt: float,
):
    """Runge-Kutta fourth-order method."""
    # TODO: fix logic to avoid Optional type
    dstate_dt: List[DataTree] = []
    dt_steps = (0.5, 0.5, 1)
    state_ = state
    for i, dt_step in enumerate(dt_steps):
        dstate_dt.append(dstate_dt_fn(state_))
        state_ = _step(state_, dstate_dt[i], dt * dt_step)
    dstate_dt[-1] = dstate_dt_fn(state)

    dstate_dt_tot = (dstate_dt[0] + 2 * (dstate_dt[1] + dstate_dt[2])
                    + dstate_dt[3]) / 6.

    return _step(state, dstate_dt_tot, dt)


### octorch.plot

class _PlotterMeta(_DataclassMeta, ABCMeta):
    pass

class AbstractPlotter(metaclass=_PlotterMeta):
    """Base for plot handlers (using matplotlib).

    I think this should separate the instantiation with plotting parameters
    from passing and plotting the data itself. So, for now, plotting is done
    by calling the instance like a function.

    TODO:
    - Not sure how to manage per-axis parameters. Axis class?
    """

    fig_kw: dict = default_field(dict(
        figsize=None,
        dpi=None,
        constrained_layout=True,  # superior to tight_layout, apparently
    ))
    show: bool = True
    save: bool = False
    label: str = 'plot'
    aspect: str = 'auto'
    axvline: bool = False
    axhline: bool = False

    # by default, figures have a single subplot
    _gridspec_kw = dict(nrows=1, ncols=1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.fig = plt.figure(**self.fig_kw)
        #self.fig.set_size_inches(None, None)
        self._gs = gridspec.GridSpec(**self._gridspec_kw, figure=self.fig)

    @abstractmethod
    def __call__(self):
        """Override in children to determine """
        raise NotImplementedError

    def _show(self):
        if self.show:
            #? makes no difference in jupyter notebook/vscode?
            plt.show()
        if self.save:
            self.fig.savefig(self.label + '.' + self.save)


#! this doesn't need to be specifically for losses.
# TODO: better separation of data/labels and algorithm
class Loss(AbstractPlotter):

    fig_kw: dict = default_field(dict(
        figsize=(8, 5),
    ))
    xscale: str = 'log'
    yscale: str = 'log'
    xlabel: str = 'Training iteration'
    ylabel: str = 'Loss'
    lw: float = 1.5
    fmt: str = 'k'
    terms_lw: float = 0.5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ax = self.fig.add_subplot(1, 1, 1)

        self.ax.set_xscale(self.xscale)
        self.ax.set_yscale(self.yscale)

        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)

    def __call__(self, loss, loss_terms=None):

        self.ax.plot(loss, self.fmt, lw=self.lw)

        if loss_terms is not None:
            for term in loss_terms.values():
                self.ax.plot(term, lw=self.terms_lw)
            self.ax.legend(['Total', *loss_terms.keys()])

        self._show()