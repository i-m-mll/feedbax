from collections.abc import Callable, Mapping, Sequence
from functools import cached_property, partial
from types import MappingProxyType
from typing import Optional, TypeVar

import equinox as eqx
import feedbax.plot as fbp
import jax.numpy as jnp
import jax.tree as jt
import jax_cookbook.tree as jtree
import plotly.graph_objects as go
from equinox import Module, field
from feedbax.task import AbstractTask
from jax_cookbook import (
    compose,
    is_module,
    is_type,
)
from jax_cookbook._func import compose_
from jaxtyping import Array, Float, PyTree

from feedbax._experiments.analysis.analysis import (
    AbstractAnalysis,
    NoPorts,
    get_validation_trial_specs,
)
from feedbax._experiments.analysis.plot import ScatterPlots
from feedbax._experiments.analysis.state_utils import (
    get_align_epoch_start,
    get_pos_endpoints,
    get_trial_start_positions,
    unsqueezer,
)
from feedbax._experiments.constants import EVAL_REACH_LENGTH
from feedbax._experiments.misc import _OptionalCallableFieldConverter
from feedbax._experiments.plot import add_endpoint_traces
from feedbax._experiments.types import (
    AnalysisInputData,
    Direction,
    Labels,
    LDict,
    ResponseVar,
    TreeNamespace,
    VarSpec,
)

T = TypeVar("T")


VAR_LEVEL_LABEL = "var"
DIRECTION_LEVEL_LABEL = "direction"
ENDPOINT_ERROR_STEPS = 10


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


vector_magnitude = partial(jnp.linalg.norm, axis=-1)


def signed_max(x, axis=None, keepdims=False):
    """Return the value with the largest magnitude, positive or negative."""
    abs_x = jnp.abs(x)
    max_idx = jnp.argmax(abs_x, axis=axis)
    if axis is None:
        return x.flatten()[max_idx]
    else:
        return jnp.take_along_axis(x, jnp.expand_dims(max_idx, axis=axis), axis=axis)


# Command measures
initial_command = Measure(
    response_var=ResponseVar.COMMAND,
    transform_fn=vector_magnitude,
    # (there is no command on time step 0)
    timesteps=slice(1, 2),
)
max_net_command = Measure(
    response_var=ResponseVar.COMMAND,
    transform_fn=vector_magnitude,
    agg_fn=jnp.max,
)
sum_net_command = Measure(
    response_var=ResponseVar.COMMAND,
    transform_fn=vector_magnitude,
    agg_fn=jnp.sum,
)


# Force measures
initial_force = Measure(
    response_var=ResponseVar.FORCE,
    transform_fn=vector_magnitude,
    # agg_fn=jnp.mean,
    timesteps=slice(1, 2),
)
max_net_force = Measure(
    response_var=ResponseVar.FORCE,
    transform_fn=vector_magnitude,
    agg_fn=jnp.max,
)
sum_net_force = Measure(
    response_var=ResponseVar.FORCE,
    transform_fn=vector_magnitude,
    agg_fn=jnp.sum,
)
max_parallel_force = Measure(
    response_var=ResponseVar.FORCE,
    direction=Direction.PARALLEL,
    agg_fn=jnp.max,
)
sum_parallel_force = Measure(
    response_var=ResponseVar.FORCE,
    direction=Direction.PARALLEL,
    transform_fn=jnp.abs,
    agg_fn=jnp.sum,
)
max_lateral_force = Measure(
    response_var=ResponseVar.FORCE,
    direction=Direction.LATERAL,
    agg_fn=jnp.max,
)
sum_lateral_force_abs = Measure(
    response_var=ResponseVar.FORCE,
    direction=Direction.LATERAL,
    transform_fn=jnp.abs,
    agg_fn=jnp.sum,
)


# Velocity measures
max_parallel_vel = Measure(
    response_var=ResponseVar.VELOCITY,
    direction=Direction.PARALLEL,
    agg_fn=jnp.max,
)
max_lateral_vel = Measure(
    response_var=ResponseVar.VELOCITY,
    direction=Direction.LATERAL,
    agg_fn=jnp.max,
)
max_lateral_vel_signed = Measure(
    response_var=ResponseVar.VELOCITY,
    direction=Direction.LATERAL,
    agg_fn=signed_max,
)


# Position measures
max_lateral_distance = Measure(
    response_var=ResponseVar.POSITION,
    direction=Direction.LATERAL,
    agg_fn=jnp.max,
    normalizer=EVAL_REACH_LENGTH / 100,
)
largest_lateral_distance = Measure(
    response_var=ResponseVar.POSITION,
    direction=Direction.LATERAL,
    agg_fn=signed_max,
    normalizer=EVAL_REACH_LENGTH / 100,
)
sum_lateral_distance = Measure(
    response_var=ResponseVar.POSITION,
    direction=Direction.LATERAL,
    agg_fn=jnp.sum,
)
sum_lateral_distance_abs = Measure(
    response_var=ResponseVar.POSITION,
    direction=Direction.LATERAL,
    transform_fn=jnp.abs,
    agg_fn=jnp.sum,
)
max_deviation = Measure(
    response_var=ResponseVar.POSITION,
    transform_fn=vector_magnitude,
    agg_fn=jnp.max,
)
sum_deviation = Measure(
    response_var=ResponseVar.POSITION,
    transform_fn=vector_magnitude,
    agg_fn=jnp.sum,
)


def make_end_velocity_error(last_n_steps: int = ENDPOINT_ERROR_STEPS) -> Measure:
    return Measure(
        response_var=ResponseVar.VELOCITY,
        transform_fn=vector_magnitude,
        agg_fn=jnp.mean,
        timesteps=slice(-last_n_steps, None),
    )


def make_end_position_error(
    reach_length: float = EVAL_REACH_LENGTH,
    last_n_steps: int = ENDPOINT_ERROR_STEPS,
) -> Measure:
    """Create measure for endpoint position error."""
    goal_pos = jnp.array([reach_length, 0.0])
    return Measure(
        response_var=ResponseVar.POSITION,
        transform_fn=lambda x: jnp.linalg.norm(x - goal_pos, axis=-1),
        agg_fn=jnp.mean,
        timesteps=slice(-last_n_steps, None),
        normalizer=reach_length / 100,
    )


def reverse_measure(measure: Measure) -> Measure:
    """Create a new measure that inverts the sign of the states before computing.

    For example, use this to turn a measure of the maximum forward velocity into a
    measure of the maximum reverse velocity.
    """
    if measure.transform_fn is not None:
        transform_fn = compose(measure.transform_fn).then(jnp.negative)
    else:
        transform_fn = jnp.negative

    return eqx.tree_at(
        lambda measure: measure.transform_fn,
        measure,
        transform_fn,
        is_leaf=lambda x: x is None,
    )


def set_timesteps(measure: Measure, timesteps) -> Measure:
    return eqx.tree_at(
        lambda measure: measure.timesteps,
        measure,
        timesteps,
        is_leaf=lambda x: x is None,
    )


ALL_MEASURES = LDict.of("measure")(
    dict(
        initial_command=initial_command,
        max_net_command=max_net_command,
        sum_net_command=sum_net_command,
        initial_force=initial_force,
        max_net_force=max_net_force,
        sum_net_force=sum_net_force,
        max_parallel_force_forward=max_parallel_force,
        max_parallel_force_reverse=reverse_measure(max_parallel_force),
        sum_parallel_force=sum_parallel_force,
        max_lateral_force_left=max_lateral_force,
        max_lateral_force_right=reverse_measure(max_lateral_force),
        sum_lateral_force_abs=sum_lateral_force_abs,
        max_parallel_vel_forward=max_parallel_vel,
        max_parallel_vel_reverse=reverse_measure(max_parallel_vel),
        max_lateral_vel_left=max_lateral_vel,
        max_lateral_vel_right=reverse_measure(max_lateral_vel),
        max_lateral_vel_signed=max_lateral_vel_signed,
        max_lateral_distance_left=max_lateral_distance,
        max_lateral_distance_right=reverse_measure(max_lateral_distance),
        largest_lateral_distance=largest_lateral_distance,
        sum_lateral_distance=sum_lateral_distance,
        sum_lateral_distance_abs=sum_lateral_distance_abs,
        max_deviation=max_deviation,
        sum_deviation=sum_deviation,
        end_velocity_error=make_end_velocity_error(),
        end_position_error=make_end_position_error(),
    )
)


MEASURE_LABELS = LDict.of("measure")(
    dict(
        initial_force="Initial control force",
        max_net_force="Max net control force",
        sum_net_force="Sum net control force",
        initial_command="Initial control command",
        max_net_command="Max net control command",
        sum_net_command="Sum net control command",
        max_parallel_force_forward="Max forward force",
        max_parallel_force_reverse="Max reverse force",
        sum_parallel_force="Sum of absolute parallel forces",
        max_lateral_force_left="Max lateral force<br>(left)",
        max_lateral_force_right="Max lateral force<br>(right)",
        sum_lateral_force_abs="Sum of absolute lateral forces",
        max_parallel_vel_forward="Max forward velocity",
        max_parallel_vel_reverse="Max reverse velocity",
        max_lateral_vel_left="Max lateral velocity<br>(left)",
        max_lateral_vel_right="Max lateral velocity<br>(right)",
        max_lateral_vel_signed="Largest lateral velocity",
        max_lateral_distance_left="Max lateral distance<br>(left, % reach length)",
        max_lateral_distance_right="Max lateral distance<br>(right, % reach length)",
        largest_lateral_distance="Largest lateral distance<br>(% reach length)",
        sum_lateral_distance="Sum of signed lateral distances",
        sum_lateral_distance_abs="Sum of absolute lateral distances",
        max_deviation="Max deviation",  # From zero/origin! i.e. stabilization task
        sum_deviation="Sum of deviations",
        end_velocity_error=f"Mean velocity error<br>(last {ENDPOINT_ERROR_STEPS} steps)",
        end_position_error=f"Mean position error<br>(last {ENDPOINT_ERROR_STEPS} steps)",
    )
)


ALL_MEASURE_KEYS = tuple(ALL_MEASURES.keys())
