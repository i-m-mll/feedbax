"""Composable loss function modules operating on state PyTrees.

TODO:

- `LossDict` only computes the total loss once, but when we append a `LossDict`
  for a single timestep to `losses_history` in `TaskTrainer`, we lose the loss
  total for that time step. When it is needed later (e.g. on plotting the loss)
  it will be recomputed, once. It is also not serialized along with the
  `losses_history`. I doubt this is a significant computational loss
  (how many loss terms * training iterations could be involved? 1e6?)
  to have to compute from time to time, but perhaps it would be nice to
  include the total as part of flatten/unflatten. It'd probably just require
  that we allow passing the total on instantiation, however that would be kind
  of weird.
    - Even if we have 6 loss terms with 1e6 iterations, it only takes ~130 ms
    to compute `losses.total`. Given that we only need to compute this once
    per session or so, it shouldn't be a problem.
- The time aggregation could be done in `CompositeLoss`, if we unsqueeze
  terms that don't have a time dimension. This would allow time aggregation
  to be controlled in one place, if for some reason it makes sense to change
  how this aggregation occurs across all loss terms.
- Protocols for all the different `state` types/fields?
    - Alternatively we could make `AbstractLoss` generic over an
      `AbstractState` typevar, however that might not make sense for typing
      the compositions (e.g. `__sum__`) since the composite can support any
      state pytrees that have the right combination of fields, not just pytrees
      that have an identical structure.
- L2 by default, but should allow for other norms

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

#! Can't do this because `AbstractVar` annotations can't be stringified.
# from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property
import logging
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Literal,
    Optional,
    Tuple,
)

import equinox as eqx
from equinox import AbstractVar
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Float, PyTree

from feedbax.misc import get_unique_label, unzip2
from feedbax.state import AbstractState, HasEffectorState

if TYPE_CHECKING:
    from feedbax.task import AbstractTaskTrialSpec


logger = logging.getLogger(__name__)


@jtu.register_pytree_node_class
class LossDict(dict[str, Array]):
    """Dictionary that provides a sum over its values."""

    @cached_property
    def total(self):
        """Elementwise sum over all values in the dictionary."""
        loss_term_values = list(self.values())
        return jax.tree_util.tree_reduce(lambda x, y: x + y, loss_term_values)
        # return jnp.sum(jtu.tree_map(
        #     lambda *args: sum(args),
        #     *loss_term_values
        # ))

    def __setitem__(self, key, value):
        raise TypeError("LossDict does not support item assignment")

    def update(self, other=(), /, **kwargs):
        raise TypeError("LossDict does not support update")

    def tree_flatten(self):
        """The same flatten function used by JAX for `dict`"""
        return unzip2(sorted(self.items()))[::-1]

    @classmethod
    def tree_unflatten(cls, keys, values):
        return LossDict(zip(keys, values))


class AbstractLoss(eqx.Module):
    """Abstract base class for loss functions.

    Instances can be composed by addition and scalar multiplication.
    """

    label: AbstractVar[str]

    def __call__(
        self,
        states: PyTree,
        trial_specs: "AbstractTaskTrialSpec",
    ) -> LossDict:
        return LossDict({self.label: self.term(states, trial_specs)})

    @abstractmethod
    def term(
        self,
        states: PyTree,
        trial_specs: "AbstractTaskTrialSpec",
    ) -> Array:
        """Implement this to calculate a loss term."""
        ...

    def __add__(self, other: "AbstractLoss") -> "CompositeLoss":
        return CompositeLoss(terms=(self, other), weights=(1.0, 1.0))

    def __radd__(self, other: "AbstractLoss") -> "CompositeLoss":
        return self.__add__(other)

    def __sub__(self, other: "AbstractLoss") -> "CompositeLoss":
        # ? I don't know if this even makes sense but it's easy to implement.
        return CompositeLoss(terms=(self, other), weights=(1.0, -1.0))

    def __rsub__(self, other: "AbstractLoss") -> "CompositeLoss":
        return CompositeLoss(terms=(self, other), weights=(-1.0, 1.0))

    def __neg__(self) -> "CompositeLoss":
        return CompositeLoss(terms=(self,), weights=(-1.0,))

    def __mul__(self, other) -> "CompositeLoss":
        """Assume scalar multiplication."""
        if eqx.is_array_like(other):
            if eqx.is_array(other) and not other.shape == ():
                raise ValueError("Can't multiply loss term by non-scalar array")
            return CompositeLoss(terms=(self,), weights=(other,))
        else:
            raise ValueError("Can't multiply loss term by non-numeric type")

    def __rmul__(self, other):
        return self.__mul__(other)


class CompositeLoss(AbstractLoss):
    """Incorporates multiple simple loss terms and their relative weights."""

    terms: Mapping[str, AbstractLoss]
    weights: Mapping[str, float]
    label: str

    def __init__(
        self,
        terms: Mapping[str, AbstractLoss] | Sequence[AbstractLoss],
        weights: Optional[Mapping[str, float] | Sequence[float]] = None,
        label: str = "",
        user_labels: bool = True,
    ):
        """
        !!! Note
            During construction the user may pass dictionaries and/or sequences
            of `AbstractLoss` instances (`terms`) and weights.

            Any `CompositeLoss` instances in `terms` are flattened, and their
            simple terms incorporated directly into the new composite loss,
            with the weights of those simple terms multiplied by the weight
            given in `weights` for their parent composite term.

            If a composite term has a user-specified label, that label will be
            prepended to the labels of its component terms, on flattening. If
            the flattened terms still do not have unique labels, they will be
            suffixed with the lowest integer that makes them unique.

        Arguments:
            terms: The sequence or mapping of loss terms to be included.
            weights: A float PyTree of the same structure as `terms`, giving
                the scalar term weights. By default, all terms have equal weight.
            label: The label for the composite loss.
            user_labels: If `True`, the keys in `terms`---if it is a mapping---
                are used as term labels, instead of the `label` field of each term.
                This is useful because it may be convenient for the user to match up
                the structure of `terms` and `weights` in a PyTree such as a dict,
                which provides labels, yet continue to use the default labels.
        """
        self.label = label

        if isinstance(terms, dict):
            if user_labels:
                labels, terms = list(zip(*terms.items()))
            else:
                labels = [term.label for term in terms.values()]
                terms = list(terms.values())
        else:
            # TODO: if `terms` is a dict, this fails!
            labels = [term.label for term in terms]

        if weights is None:
            weights = jax.tree_map(lambda _: 1.0, terms)
        elif not len(terms) == len(weights):
            raise ValueError(
                "Mismatch between number of loss terms " + "and number of term weights"
            )

        if isinstance(weights, dict):
            weights = list(weights.values())

        # Split into lists of data for simple and composite terms.
        term_tuples_split: Tuple[
            Sequence[Tuple[str, AbstractLoss, float]],
            Sequence[Tuple[str, AbstractLoss, float]],
        ]
        term_tuples_split = eqx.partition(
            list(zip(labels, terms, weights)),
            lambda x: not isinstance(x[1], CompositeLoss),
            is_leaf=lambda x: isinstance(x, tuple),
        )

        # Removes the `None` values from the lists.
        term_tuples_leaves = jax.tree_map(
            lambda x: jtu.tree_leaves(x, is_leaf=lambda x: isinstance(x, tuple)),
            term_tuples_split,
            is_leaf=lambda x: isinstance(x, list),
        )

        # Start with the simple terms, if there are any.
        if term_tuples_leaves[0] == []:
            all_labels, all_terms, all_weights = (), (), ()
        else:
            all_labels, all_terms, all_weights = zip(*term_tuples_leaves[0])

        # Make sure the simple term labels are unique.
        for i, label in enumerate(all_labels):
            label = get_unique_label(label, all_labels[:i])
            all_labels = all_labels[:i] + (label,) + all_labels[i + 1 :]

        # Flatten the composite terms, assuming they have the usual dict
        # attributes. We only need to flatten one level, because this `__init__`
        # (and the immutability of `eqx.Module`) ensures no deeper nestings
        # are ever constructed except through extreme hacks.
        for group_label, composite_term, group_weight in term_tuples_leaves[1]:
            labels = composite_term.terms.keys()

            # If a unique label for the composite term is available, use it to
            # format the labels of the flattened terms.
            if group_label != "":
                labels = [f"{group_label}_{label}" for label in labels]
            elif composite_term.label != "":
                labels = [f"{composite_term.label}_{label}" for label in labels]

            # Make sure the labels are unique.
            for label in labels:
                label = get_unique_label(label, all_labels)
                all_labels += (label,)

            all_terms += tuple(composite_term.terms.values())
            all_weights += tuple(
                [group_weight * weight for weight in composite_term.weights.values()]
            )

        self.terms = dict(zip(all_labels, all_terms))
        self.weights = dict(zip(all_labels, all_weights))

    def __or__(self, other: "CompositeLoss") -> "CompositeLoss":
        """Merge two composite losses, overriding terms with the same label."""
        return CompositeLoss(
            terms=self.terms | other.terms,
            weights=self.weights | other.weights,
            label=other.label,
        )

    @jax.named_scope("fbx.CompositeLoss")
    def __call__(
        self,
        states: AbstractState,
        trial_specs: "AbstractTaskTrialSpec",
    ) -> LossDict:
        """Evaluate, weight, and return all component terms.

        Arguments:
            states: Trajectories of system states for a set of trials.
            trial_specs: Task specifications for the set of trials.
        """
        # Evaluate all loss terms
        losses = jax.tree_map(
            lambda loss: loss.term(states, trial_specs),
            self.terms,
            is_leaf=lambda x: isinstance(x, AbstractLoss),
        )

        # aggregate over batch
        losses = jax.tree_map(lambda x: jnp.mean(x, axis=0), losses)

        if self.weights is not None:
            # term scaling
            losses = jax.tree_map(
                lambda term, weight: term * weight, dict(losses), self.weights
            )

        return LossDict(losses)

    def term(
        self,
        states: PyTree,
        trial_specs: "AbstractTaskTrialSpec",
    ) -> Array:
        return self(states, trial_specs).total


class EffectorPositionLoss(AbstractLoss):
    """Penalizes the effector's squared distance from the target position
    across the trial.

    Attributes:
        label: The label for the loss term.
        discount_func: Returns a trajectory with which to weight (discount)
            the loss values calculated for each time step of the trial.
            Defaults to a power-law curve that puts most of the weight on
            time steps near the end of the trial.

    !!! Note
        If the return value of `discount_func` is shaped such that it gives
        non-zero weight to the position error during the fixation period of
        (say) a delayed reach task, then typically the target will be specified
        as the fixation point during that period, and `EffectorPositionLoss`
        will also act as a fixation loss.

        On the other hand, when using certain kinds of goal error discounting
        (e.g. exponential, favouring errors near the end of the trial) then the
        fixation loss may not be weighed into `EffectorPositionLoss`, and it
        may be appropriate to add `EffectorFixationLoss` to the composite loss.
        However, in that case the same result could still be achieved using a
        single instance of `EffectorPositionLoss`, by passing a `discount`
        that's the sum of the goal error discount (say, non-zero only near the
        end of the trial) and the hold signal (non-zero only during the
        fixation period) scaled by the relative weights of the goal and
        fixation error losses.
    """

    label: str = "Effector position"
    discount_func: Optional[Callable[[int], Float[Array, "time"]]] = (
        lambda n_steps: power_discount(n_steps, discount_exp=6)[None, :]
    )

    def term(
        self,
        states: HasEffectorState,
        trial_specs: "AbstractTaskTrialSpec",
    ) -> Array:

        # Sum over X, Y, giving the squared Euclidean distance
        loss = jnp.sum(
            (states.mechanics.effector.pos - trial_specs.target.pos) ** 2, axis=-1
        )

        # temporal discount
        if self.discount_func is not None:
            loss = loss * self.discount(loss.shape[-1])

        # sum over time
        loss = jnp.sum(loss, axis=-1)

        return loss

    # @cache
    def discount(self, n_steps):
        # Can't use a cache because of JIT.
        # But we only need to run this once per training iteration...
        return self.discount_func(n_steps)


class EffectorStraightPathLoss(AbstractLoss):
    """Penalizes non-straight paths followed by the effector between initial
    and final position.

    !!! Info ""
        Calculates the length of the paths followed, and normalizes by the
        Euclidean (straight-line) distance between the initial and final state.

    Attributes:
        label: The label for the loss term.
        normalize_by: Controls whether to normalize by the distance between the
            initial position & actual final position, or the initial position
            & task-specified goal position.
    """

    label: str = "Effector path straightness"
    normalize_by: Literal["actual", "goal"] = "actual"

    def term(
        self,
        states: HasEffectorState,
        trial_specs: "AbstractTaskTrialSpec",
    ) -> Array:

        effector_pos = states.mechanics.effector.pos
        pos_diff = jnp.diff(effector_pos, axis=1)
        piecewise_lengths = jnp.linalg.norm(pos_diff, axis=-1)
        path_length = jnp.sum(piecewise_lengths, axis=1)
        if self.normalize_by == "actual":
            final_pos = effector_pos[:, -1]
        elif self.normalize_by == "goal":
            final_pos = trial_specs.target.pos
        init_final_diff = final_pos - effector_pos[:, 0]
        straight_length = jnp.linalg.norm(init_final_diff, axis=-1)

        loss = path_length / straight_length

        return loss


class EffectorFixationLoss(AbstractLoss):
    """Penalizes the effector's squared distance from the fixation position.

    !!! Info ""
        Similar to `EffectorPositionLoss`, but only penalizes the position
        error during the part of the trial where `trial_specs.inputs.hold`
        is non-zero/`True`.

    Attributes:
        label: The label for the loss term.
    """

    label: str = "Effector maintains fixation"

    def term(
        self,
        states: PyTree,
        trial_specs: "AbstractTaskTrialSpec",
    ) -> Array:

        loss = jnp.sum(
            (states.mechanics.effector.pos - trial_specs.target.pos) ** 2, axis=-1
        )

        loss = loss * jnp.squeeze(trial_specs.inputs.hold)

        # sum over time
        loss = jnp.sum(loss, axis=-1)

        return loss


class EffectorFinalVelocityLoss(AbstractLoss):
    """Penalizes the squared difference between the effector's final velocity
    and the goal velocity (typically zero) on the final timestep.

    Attributes:
        label: The label for the loss term.
    """

    label: str = "Effector final velocity"

    def term(
        self,
        states: PyTree,
        trial_specs: "AbstractTaskTrialSpec",
    ) -> Array:

        loss = jnp.sum(
            (states.mechanics.effector.vel[:, -1] - trial_specs.target.vel[:, -1]) ** 2,
            axis=-1,
        )

        return loss


class NetworkOutputLoss(AbstractLoss):
    """Penalizes the squared values of the network's outputs.

    Attributes:
        label: The label for the loss term.
    """

    label: str = "NN output"

    def term(
        self,
        states: PyTree,
        trial_specs: "AbstractTaskTrialSpec",
    ) -> Array:

        # Sum over output channels
        loss = jnp.sum(states.net.output**2, axis=-1)

        # Sum over time
        loss = jnp.sum(loss, axis=-1)

        return loss


class NetworkActivityLoss(AbstractLoss):
    """Penalizes the squared values of the network's hidden activity.

    Attributes:
        label: The label for the loss term.
    """

    label: str = "NN hidden activity"

    def term(
        self,
        states: PyTree,
        trial_specs: "AbstractTaskTrialSpec",
    ) -> Array:

        # Sum over hidden units
        loss = jnp.sum(states.net.hidden**2, axis=-1)

        # sum over time
        loss = jnp.sum(loss, axis=-1)

        return loss


def power_discount(n_steps: int, discount_exp: int = 6):
    """A power-law vector that puts most of the weight on its later elements.

    Arguments:
        n_steps: The number of time steps in the trajectory to be weighted.
        discount_exp: The exponent of the power law.
    """
    if discount_exp == 0:
        return 1.0
    else:
        return jnp.linspace(1.0 / n_steps, 1.0, n_steps) ** discount_exp


def mse(x, y):
    """Mean squared error."""
    return jax.tree_map(
        lambda x, y: jnp.mean((x - y) ** 2),
        x,
        y,
    )
