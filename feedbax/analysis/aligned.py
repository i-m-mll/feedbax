from collections.abc import Callable, Mapping, Sequence
from functools import cached_property, partial
from types import MappingProxyType
from typing import Optional

import equinox as eqx
import feedbax.plot as fbp
import jax.numpy as jnp
import jax.tree as jt
import jax_cookbook.tree as jtree
import plotly.graph_objects as go
from equinox import Module, field
from feedbax.tasks import AbstractTask
from jax_cookbook import (
    LDict,
    compose,
    is_module,
    is_type,
)
from jax_cookbook._func import compose_
from jaxtyping import Array, Float, PyTree

from feedbax.analysis.analysis import (
    AbstractAnalysis,
    NoPorts,
    get_validation_trial_specs,
)
from feedbax.analysis.plot import ScatterPlots
from feedbax.analysis.state_utils import (
    get_align_epoch_start,
    get_pos_endpoints,
    get_trial_start_positions,
    unsqueezer,
)
from feedbax.analysis.support import _OptionalCallableFieldConverter
from feedbax.config.defaults import EVAL_REACH_LENGTH
from feedbax.config.namespace import TreeNamespace
from feedbax.plot.experiments import add_endpoint_traces
from feedbax.analysis.types import (
    AnalysisInputData,
    Direction,
    Labels,
    ResponseVar,
    VarSpec,
)

VAR_LEVEL_LABEL = "var"
DIRECTION_LEVEL_LABEL = "direction"


def get_reach_directions(task: AbstractTask, *args) -> Array:
    pos_endpoints = get_pos_endpoints(get_validation_trial_specs(task))
    return pos_endpoints[1] - pos_endpoints[0]


def get_trivial_reach_directions(task: AbstractTask, *args) -> Array:
    """Return 'aligns' 'reaches' with the x-y axes; i.e. effectively does nothing.

    The purpose of this is to avoid (for now) trying to bypass `AlignedVars` as a dependency of
    other analyses (e.g. `Profiles`) in cases where it doesn't make sense to align with a certain
    directon (e.g. certain steady-state tasks performed using `SimpleReaches`).
    """
    origins, _ = get_pos_endpoints(get_validation_trial_specs(task))
    return jnp.broadcast_to(jnp.array([1.0, 0.0]), origins.shape)


def _where_force(states, *_):
    filtered = getattr(states.force_filter, "output", None)
    if filtered is not None:
        return filtered
    return states.efferent.output


DEFAULT_VARSET: LDict[str, VarSpec] = LDict.of(VAR_LEVEL_LABEL)(
    {
        ResponseVar.POSITION: VarSpec(
            where=lambda states, *_: states.mechanics.effector.pos,
            labels=Labels("Position", "Pos.", "p"),
            origin=compose(get_trial_start_positions).then(unsqueezer(-2)),
        ),
        ResponseVar.VELOCITY: VarSpec(
            where=lambda states, *_: states.mechanics.effector.vel,
            labels=Labels("Velocity", "Vel.", "v"),
        ),
        ResponseVar.COMMAND: VarSpec(
            where=lambda states, *_: states.net.output,
            labels=Labels("Control command", "Command", "u"),
        ),
        ResponseVar.FORCE: VarSpec(
            where=_where_force,
            labels=Labels("Control force", "Force", "F"),
        ),
    }
)


def get_varset_labels(varset: PyTree[VarSpec]) -> Labels:
    """Get trees of labels for all variables in a tree of specs."""
    return jtree.unzip(
        jt.map(
            lambda spec: spec.labels,
            varset,
            is_leaf=is_type(VarSpec),
        ),
        tuple_cls=Labels,
    )


DIRECTION_IDXS = LDict.of(DIRECTION_LEVEL_LABEL)(
    {
        Direction.PARALLEL: 0,
        Direction.LATERAL: 1,
    }
)


def get_forward_lateral_vel(
    velocity: Float[Array, "*batch conditions time xy=2"],
    pos_endpoints: Float[Array, "point=2 conditions xy=2"],
) -> Float[Array, "*batch conditions time 2"]:
    """Given x-y velocity components, rebase onto components forward and lateral to the line between endpoints.

    Arguments:
        velocity: Trajectories of velocity vectors.
        pos_endpoints: Initial and goal reference positions for each condition, defining reference lines.

    Returns:
        forward: Forward velocity components (parallel to the reference lines).
        lateral: Lateral velocity components (perpendicular to the reference lines).
    """
    init_pos, goal_pos = pos_endpoints
    direction_vec = goal_pos - init_pos

    return project_onto_direction(velocity, direction_vec)


def project_onto_direction(
    var: Float[Array, "*batch conditions time xy=2"],
    direction_vec: Float[Array, "conditions xy=2"],
):
    """Projects components of arbitrary variables into components parallel and orthogonal to a given direction.

    Arguments:
        var: Data with x-y components to be projected.
        direction_vector: Direction vectors.

    Returns:
        projected: Projected components (parallel and lateral).
    """
    # Normalize the line vector
    direction_vec_norm = direction_vec / jnp.linalg.norm(direction_vec, axis=-1, keepdims=True)

    # Broadcast line_vec_norm to match velocity's shape
    direction_vec_norm = direction_vec_norm[:, None]  # Shape: (conditions, 1, xy)

    # Calculate forward component (dot product)
    parallel = jnp.sum(var * direction_vec_norm, axis=-1)

    # Calculate lateral component (cross product)
    lateral = jnp.cross(direction_vec_norm, var)

    return jnp.stack([parallel, lateral], axis=-1)


def get_aligned_vars(vars, directions):
    """Get variables from state PyTree, and project them onto respective reach directions for their trials."""
    return jt.map(
        lambda var: project_onto_direction(var, directions),
        vars,
    )


def get_reach_origins_directions(task: AbstractTask, models: PyTree[Module], hps: TreeNamespace):
    pos_endpoints = get_pos_endpoints(get_validation_trial_specs(task))
    directions = pos_endpoints[1] - pos_endpoints[0]
    origins = pos_endpoints[0]
    return origins, directions


class AlignedVars(AbstractAnalysis[NoPorts]):
    """Align spatial variable (e.g. position and velocity) coordinates with the reach direction."""

    varset: PyTree[VarSpec] = eqx.field(default_factory=lambda: DEFAULT_VARSET)
    directions_fn: Callable = get_reach_directions

    def compute(
        self,
        data: AnalysisInputData,
        **kwargs,
    ) -> PyTree[Array]:
        def _get_aligned_vars_by_task(task, states_by_task, hps_by_task):
            directions = self.directions_fn(task, hps_by_task)

            def _get_aligned_vars(states):
                def _align_var(spec: VarSpec):
                    arr = spec.where(states)
                    if spec.origin is not None:
                        if callable(spec.origin):
                            arr = arr - spec.origin(task)
                        else:
                            # Assume `spec.origin` is a constant array
                            arr = arr - spec.origin
                    #! TODO: Use `ArrayLikeWrapper` to keep var metadata with arrays
                    return project_onto_direction(arr, directions)

                return jt.map(_align_var, self.varset, is_leaf=is_type(VarSpec))

            return jt.map(
                _get_aligned_vars,
                states_by_task,
                is_leaf=is_module,
            )

        result = jt.map(
            _get_aligned_vars_by_task,
            data.tasks,
            data.states,
            data.hps,
            is_leaf=is_module,
        )

        return result


def add_aligned_position_endpoints(
    figs: PyTree[go.Figure], xaxis="x1", yaxis="y1"
) -> PyTree[go.Figure]:
    """Add aligned position endpoints to the figures."""
    #! TODO: Don't hardcode reach length  but use `data.tasks` or `data.hps` per leaf!
    #! (First need to solve: things:///show?id=GGRNzFkx5fUCNnwy9kzfvt)
    return jt.map(
        lambda fig: add_endpoint_traces(
            fig,
            jnp.array([[0.0, 0.0], [EVAL_REACH_LENGTH, 0.0]]),
            xaxis=xaxis,
            yaxis=yaxis,
        ),
        figs,
        is_leaf=is_type(go.Figure),
    )


def get_aligned_trajectories_node(
    colorscale_key: Optional[str] = None,
    pos_endpoints: bool = True,
    varset: PyTree[VarSpec] = DEFAULT_VARSET,
    subplot_level: str = VAR_LEVEL_LABEL,
    align_epoch: Optional[int] = None,
    pre_transform_fns: Sequence[Callable] = (),
) -> ScatterPlots:
    aligned_var_subplot_titles = get_varset_labels(varset).medium
    aligned_var_axes_labels = jt.map(
        lambda l: fbp.AxesLabels(rf"${l}_\parallel$", rf"${l}_\perp$"),
        get_varset_labels(varset).short,
    )
    node = ScatterPlots(
        inputs=ScatterPlots.Ports(input=AlignedVars(varset=varset)),
        colorscale_key=colorscale_key,
        subplot_level=subplot_level,
    ).with_fig_params(
        subplot_titles=aligned_var_subplot_titles,
        #! TODO: Probably leave the individual labels out; just use master labels
        axes_labels=aligned_var_axes_labels,
        # master_axes_labels=AxesLabels2D("Parallel", "Lateral"),
    )
    for transform_fn in pre_transform_fns:
        node = node.after_transform(transform_fn)
    if align_epoch is not None:
        node = node.after_transform(
            lambda vars, *, data: get_align_epoch_start(align_epoch)(vars, data=data)
        )
    if colorscale_key is not None:
        node = node.after_stacking(colorscale_key)
    if pos_endpoints:
        node = node.then_transform_figs(add_aligned_position_endpoints)
    return node


#! TODO: This should not be limited to `ResponseVar`
class Measure(Module):
    """Unified measure class for computing response metrics.

    Attributes:
        response_var: Which response variable to measure (pos, vel, force)
        agg_fn: Function to aggregate over time axis (e.g. jnp.max, jnp.mean)
        direction: Optional direction to extract vector component
        timesteps: Optional slice to select specific timesteps
        transform_fn: Optional function to transform values (e.g. jnp.linalg.norm)
        normalizer: Optional value to divide result by
    """

    response_var: ResponseVar = field(converter=ResponseVar)
    timesteps: Optional[Callable[..., slice]] = field(
        default=None, converter=_OptionalCallableFieldConverter[slice]("Measure.timesteps")
    )
    direction: Optional[Callable[..., Direction]] = field(
        default=None, converter=_OptionalCallableFieldConverter[Direction]("Measure.direction")
    )
    transform_fn: Optional[Callable] = None
    agg_fn: Optional[Callable] = None
    normalizer: Optional[Callable[..., float]] = field(
        default=None, converter=_OptionalCallableFieldConverter[float]("Measure.normalizer")
    )

    @cached_property
    def _methods(self) -> Mapping[str, Callable]:
        return MappingProxyType(
            dict(
                timesteps=self._select_timesteps,
                direction=self._select_direction,
                transform_fn=self._apply_transform,
                agg_fn=self._aggregate,
                normalizer=self._normalize,
            )
        )

    def _call_methods(self, **kwargs) -> list[Callable]:
        return [self._get_response_var] + [
            partial(method, **kwargs)
            for key, method in self._methods.items()
            if getattr(self, key) is not None
        ]

    def _get_response_var(self, input: LDict) -> Float[Array, "..."]:
        """Extract the specified response variable."""
        return input[self.response_var.value]

    def _select_timesteps(self, values: Float[Array, "..."], **kwargs) -> Float[Array, "..."]:
        """Select specified timesteps."""
        assert self.timesteps is not None
        return values[..., self.timesteps(**kwargs), :]

    def _select_direction(self, values: Float[Array, "..."], **kwargs) -> Float[Array, "..."]:
        """Select specified direction component."""
        assert self.direction is not None
        return values[..., DIRECTION_IDXS[self.direction(**kwargs)]]

    def _aggregate(self, values: Float[Array, "..."], **kwargs) -> Float[Array, "..."]:
        """Apply aggregation function over time axis."""
        assert self.agg_fn is not None
        return self.agg_fn(values, axis=-1)

    def _normalize(self, values: Float[Array, "..."], **kwargs) -> Float[Array, "..."]:
        """Apply normalization."""
        assert self.normalizer is not None
        return values / self.normalizer(**kwargs)

    def _apply_transform(self, values: Float[Array, "..."], **kwargs) -> Float[Array, "..."]:
        """Apply custom transformation function."""
        assert self.transform_fn is not None
        return self.transform_fn(values)

    def __call__(self, input: LDict, **kwargs) -> Float[Array, "..."]:
        """Calculate measure for response state.

        Args:
            responses: Response state containing trajectories

        Returns:
            Computed measure values
        """
        return compose_(*self._call_methods(**kwargs))(input)
