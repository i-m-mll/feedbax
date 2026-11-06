"""Composable loss function modules operating on state PyTrees.

TODO:

- The time aggregation could be done in `CompositeLoss`, if we unsqueeze
  terms that don't have a time dimension. This would allow time aggregation
  to be controlled in one place, if for some reason it makes sense to change
  how this aggregation occurs across all loss terms.
  - Actually I think the time aggregation should probably be left to the
    specific implementation; consider that we might want to do dynamic slicing
    and aggregation which involves vmapping over a function in which the time
    aggregation occurs.
- Protocols for all the different `state` types/fields?
    - Alternatively we could make `AbstractLoss` generic over a
      `StateT` typevar, however that might not make sense for typing
      the compositions (e.g. `__sum__`) since the composite can support any
      state pytrees that have the right combination of fields, not just pytrees
      that have an identical structure.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

#! Can't do this because `AbstractVar` annotations can't be stringified.
# from __future__ import annotations

import functools as ft
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    Self,
    Tuple,
    TypeVar,
)

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
import jax.tree_util as jtu
from equinox import AbstractVar, Module
from feedbax._vendor.jax_cookbook.misc import softmin
from jaxtyping import Array, ArrayLike, Float, PyTree

from feedbax._numerics import sqrt_with_zero_grad
from feedbax.runtime.graph import Component
from feedbax.runtime.mapping import WhereDict
from feedbax.runtime.naming import get_unique_label
from feedbax.runtime.state import State

if TYPE_CHECKING:
    from feedbax.models.feedback import SimpleFeedbackState
    from feedbax.tasks import TaskTrialSpec


logger = logging.getLogger(__name__)


U = TypeVar("U")


def nan_safe_mse(input: Array, target: Array) -> Float[Array, ""]:
    """Mean squared error over finite target entries only."""
    finite = jnp.isfinite(target)
    difference = jnp.where(finite, input, 0) - jnp.where(finite, target, 0)
    squared_error = jnp.square(difference)
    count = jnp.maximum(jnp.sum(finite), 1)
    return jnp.sum(squared_error) / count


@jtu.register_pytree_node_class
@dataclass
class TermTree[T](Mapping[str, "TermTree"]):
    label: str
    names: Tuple[str, ...]
    children: Tuple[Self, ...] = ()
    value: Optional[Array] = None  # only for leaf
    weight: float = 1.0
    leaf_fn: Callable[[Array], Array] = jnp.mean
    originator: Optional[T] = None

    def tree_flatten(self):
        """Flatten for PyTree; only include numeric/JAX parts dynamically."""
        # dynamic leaves (these are mapped by vmap/jit):
        # weight is included here (not in aux) so that changing the weight value
        # does NOT change the treedef. This allows loss_update_func to update
        # weights without breaking tree_set(history.loss, new_losses, idx).
        # weight is kept as-is (no jnp.asarray) so that Python float weights
        # stay as Python non-array leaves, avoiding traced-bool issues in
        # filter_jit's partition(out, is_array) output processing.
        children = (self.children, self.value, self.weight)
        # static metadata (these are carried in aux data, not mapped):
        aux = (self.label, self.names, self.leaf_fn, self.originator)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        label, names, leaf_fn, originator = aux_data
        kids, value, weight = children
        return cls(
            label=label,
            names=names,
            children=kids,
            value=value,
            weight=weight,
            leaf_fn=leaf_fn,
            originator=originator,
        )

    @staticmethod
    def leaf(
        label: str,
        value: Array,
        *,
        weight: float = 1.0,
        leaf_fn: Callable[[Array], Array] = jnp.mean,
        originator: Optional[T] = None,
    ) -> "TermTree":
        return TermTree(
            label=label,
            names=(),
            children=(),
            value=value,
            leaf_fn=leaf_fn,
            weight=weight,  # originator=originator
        )

    @staticmethod
    def branch(
        label: str,
        mapping: Mapping[str, "TermTree"],
        *,
        weight: float = 1.0,
        leaf_fn: Callable[[Array], Array] = jnp.mean,
        originator: Optional[T] = None,  #! I don't think this should use the class generic T
    ) -> "TermTree":
        # Freeze order for JIT-stability
        names = tuple(mapping.keys())
        kids = tuple(mapping[k] for k in names)
        return TermTree(
            label=label,
            names=names,
            children=kids,
            value=None,
            weight=weight,
            leaf_fn=leaf_fn,
            # originator=originator,
        )

    def with_weight(self, weight: float) -> Self:
        return TermTree(
            label=self.label,
            names=self.names,
            children=self.children,
            value=self.value,
            leaf_fn=self.leaf_fn,
            weight=weight,
        )

    def fold(
        self,
        *,
        on_leaf: Callable[["TermTree"], U],
        on_branch: Callable[["TermTree", Tuple[U, ...]], U],
    ) -> U:
        if self.value is not None:
            return on_leaf(self)
        child_vals = tuple(c.fold(on_leaf=on_leaf, on_branch=on_branch) for c in self.children)
        return on_branch(self, child_vals)

    def aggregate(
        self,
        *,
        leaf_fn: Optional[
            Callable[[Array], Array]
        ] = None,  # uniform override; None → use node.leaf_fn
        plus: Callable[[Array, Array], Array] = jnp.add,  # how to combine siblings
        times: Callable[[Array, Array], Array] = jnp.multiply,  # how to apply weights
        zero: Array | float = 0.0,  # identity for `plus`
    ) -> Array:
        # (a) how to evaluate a leaf:
        def _on_leaf(node: "TermTree") -> Array:
            f = node.leaf_fn if leaf_fn is None else leaf_fn
            v = f(node.value)  # scalar (or broadcastable) is expected
            return times(jnp.asarray(node.weight), v)

        # (b) how to evaluate a branch:
        def _on_branch(node: "TermTree", kids: Tuple[Array, ...]) -> Array:
            acc = ft.reduce(plus, kids, jnp.asarray(zero))
            return times(jnp.asarray(node.weight), acc)

        return self.fold(on_leaf=_on_leaf, on_branch=_on_branch)

    @property
    def total(self) -> Array:
        """Return the node-weighted scalar sum of all leaves in this tree."""
        return self.aggregate()

    def map(self, fn: Callable[[Array], ArrayLike], check_value: Callable = eqx.is_array) -> Self:
        """Apply a function to all `value` arrays in this `TermTree`.

        Handle the recursion explicitly rather than using `jt.map` and excluding `weights` etc.
        """
        if self.value is not None:
            # Leaf node
            new_value = fn(self.value) if check_value(self.value) else self.value
            return eqx.tree_at(lambda t: t.value, self, new_value)
        else:
            # Branch node - recursively apply to children
            new_children = tuple(child.map(fn) for child in self.children)
            return eqx.tree_at(lambda t: t.children, self, new_children)

    def _walk_leaves(
        self,
        *,
        include_root: bool,
        apply_weights: bool,
    ):
        """Yield (path:str, node:TermTree, cum_w:float)."""

        def visit(node: "TermTree", path_parts: list[str], cum_w: float):
            if node.value is not None:
                # leaf
                path = "/".join(path_parts) if path_parts else node.label
                yield path, node, (cum_w * node.weight if apply_weights else 1.0)
                return

            next_cum_w = cum_w * (node.weight if apply_weights else 1.0)
            base_parts = [node.label] if include_root and not path_parts else path_parts
            for name, child in zip(node.names, node.children):
                yield from visit(child, base_parts + [name], next_cum_w)

        yield from visit(self, [], 1.0)

    def flatten(
        self,
        *,
        apply_weights: bool = True,
        apply_leaf_fn: bool = False,
        include_root: bool = False,
    ) -> dict[str, Array]:
        """
        Returns {'path/to/leaf': Array} for all leaves, WITHOUT applying `node.leaf_fn`.

        If `apply_weights=True`, multiplies the raw leaf arrays by the cumulative
        product of weights along the path (including the leaf's own `weight`).
        """
        out: dict[str, Array] = {}
        for path, node, w in self._walk_leaves(
            include_root=include_root, apply_weights=apply_weights
        ):
            #! TODO: Make this `dict[str, Array | None]` and keep `None` leaves
            arr = node.value if node.value is not None else jnp.asarray(0.0)
            arr = self.leaf_fn(arr) if apply_leaf_fn else arr
            out[path] = (w * arr) if apply_weights else arr
        return out

    def iter_items(self, *, include_root: bool = False):
        """Yield (path, leaf_node) for all leaves (no weights, no reduction)."""
        for path, node, _ in self._walk_leaves(include_root=include_root, apply_weights=False):
            if node.value is not None:
                yield path, node

    def __len__(self):
        return len(self.children)

    def __iter__(self):
        return iter(self.names)

    def __getitem__(self, k: str) -> Self:
        i = self.names.index(k)
        return self.children[i]


def is_termtree(x):
    return isinstance(x, TermTree)


class AbstractLoss(Module):
    """Abstract base class for loss functions.

    Instances can be composed by addition and scalar multiplication.

    For leaf loss terms, subclass this and implement the `term` method.

    For nodes composed of multiple weighted terms, instantiate `CompositeLoss`.

    For a composite leaf that includes multiple weighted sub-terms that all
    depend on shared variables, instantiate `FuncTermsLoss`.
    """

    label: AbstractVar[str]

    def __call__(
        self,
        states: PyTree,
        trial_specs: "TaskTrialSpec",
        model: Component,
    ) -> TermTree["AbstractLoss"]:
        return TermTree.leaf(
            self.label,
            self.term(states, trial_specs, model),
            originator=self,
        )

    def term(
        self,
        states: Optional[PyTree],
        trial_specs: Optional["TaskTrialSpec"],
        model: Optional[Component],
    ) -> Array:
        """Implement this to calculate a loss term."""
        raise NotImplementedError

    # TODO
    # def __init_subclass__(cls, **kwargs):
    #     """Enforce that subclasses implement exactly one of `term` (leaf nodes) or `__call__`."""
    #     super().__init_subclass__(**kwargs)
    #     if cls is AbstractLoss:
    #         return  # don't check the root

    #     base_call = AbstractLoss.__call__
    #     base_term = AbstractLoss.term

    #     # effective implementations after MRO resolution
    #     call_impl = getattr(cls, "__call__", None)
    #     term_impl = getattr(cls, "term", None)

    #     call_overridden = call_impl is not base_call
    #     term_overridden = term_impl is not base_term

    #     # allow abstract intermediates that implement neither
    #     if not call_overridden and not term_overridden:
    #         #! Doesn't work, even if we subclass ABC; need to check if abstract `eqx.Module`
    #         if not inspect.isabstract(cls):
    #             raise TypeError(
    #                 f"{cls.__name__} must override exactly one of '__call__' or 'term'."
    #             )
    #         return

    #     # forbid overriding both
    #     if call_overridden and term_overridden:
    #         raise TypeError(
    #             f"{cls.__name__} must override exactly one of '__call__' or 'term', not both."
    #         )

    def skeleton(self, batch_dims: tuple[int, ...]) -> TermTree:
        return TermTree.leaf(self.label, jnp.empty(batch_dims))

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


class AbstractTermedLoss(AbstractLoss):
    terms: AbstractVar[Mapping[str, Any]]
    weights: AbstractVar[Mapping[str, float]]

    def flatten_weights(
        self,
        *,
        prefix: str = "",
        apply_weights: bool = True,
        parent_weight: float = 1.0,
        include_self_label: bool = False,
    ) -> dict[str, float]:
        """
        Return a flattened mapping { 'path/to/leaf': effective_weight }.

        If apply_weights=True, each value includes the product of all weights
        along the path from the root down to that leaf. Otherwise, only the
        local (child) weights are returned.
        """
        base = prefix
        if include_self_label:
            base += self.label
        out: dict[str, float] = {}
        if base:
            base += "/"

        for name, subloss in self.terms.items():
            local_w = self.weights.get(name, 1.0)
            path_prefix = f"{base}{name}"

            if isinstance(subloss, AbstractTermedLoss):
                # Recurse into sub-composites
                sub = subloss.flatten_weights(
                    prefix=path_prefix,
                    apply_weights=apply_weights,
                    parent_weight=(parent_weight * local_w if apply_weights else 1.0),
                )
                out.update(sub)
            else:
                # Leaf: record effective weight
                effective = parent_weight * local_w if apply_weights else local_w
                out[path_prefix] = float(effective)

        return out

    def skeleton(self, batch_dims: tuple[int, ...]) -> TermTree:
        children: dict[str, TermTree] = {}
        for name, term in self.terms.items():
            if isinstance(term, AbstractLoss):
                child = term.skeleton(batch_dims)
            else:
                child = TermTree.leaf(name, jnp.empty(batch_dims))
            child = child.with_weight(self.weights.get(name, 1.0))
            children[name] = child
        return TermTree.branch(self.label, children, originator=self)


class FuncTermsLoss[T](AbstractTermedLoss):
    """A leaf loss node with multiple weighted terms that depend on shared context."""

    label: str
    build_context: Callable[
        [State, "TaskTrialSpec", Component], T
    ]  # (states, trial_specs, model) -> Ctx
    terms: Mapping[str, Callable[[T], Array]]
    weights: Mapping[str, float]

    @jax.named_scope("fbx.FuncTermsLoss")
    def __call__(
        self,
        states: State,
        trial_specs: "TaskTrialSpec",
        model: Component,
    ) -> TermTree[AbstractLoss]:
        ctx = self.build_context(states, trial_specs, model)
        children = {}
        for name, fn in self.terms.items():
            v = fn(ctx)
            # v = v if v.shape == () else self.reduce(v)  # enforce scalar per component
            leaf = TermTree.leaf(name, v).with_weight(self.weights.get(name, 1.0))
            children[name] = leaf
        return TermTree.branch(self.label, children, originator=self)


class CompositeLoss(AbstractTermedLoss):
    """A loss node that composes multiple loss nodes and their weights."""

    label: str
    terms: dict[str, AbstractLoss]
    weights: dict[str, float]

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
            weights: Scalar term weights. A sequence matches the term order.
                A mapping must contain exactly the input term keys, or the unique
                term labels when `terms` is a sequence. Defaults to 1.0 per term.
            label: The label for the composite loss.
            user_labels: If `True`, the keys in `terms`---if it is a mapping---
                are used as term labels, instead of the `label` field of each term.
                This is useful because it may be convenient for the user to match up
                the structure of `terms` and `weights` in a PyTree such as a dict,
                which provides labels, yet continue to use the default labels.
        """
        self.label = label

        if isinstance(terms, Mapping):
            weight_keys = tuple(terms)
            loss_terms = tuple(terms[key] for key in weight_keys)
            labels = weight_keys if user_labels else tuple(term.label for term in loss_terms)
        elif isinstance(terms, Sequence):
            loss_terms = tuple(terms)
            labels = weight_keys = tuple(term.label for term in loss_terms)
        else:
            raise ValueError("terms must be a mapping or sequence of AbstractLoss")

        if isinstance(weights, Mapping):
            expected = set(weight_keys)
            if len(expected) != len(weight_keys):
                raise ValueError("Mapping weights for a sequence of loss terms require unique labels")
            provided = set(weights)
            missing = expected - provided
            unknown = provided - expected
            if missing or unknown:
                raise ValueError(
                    f"Loss weight keys must match term keys. "
                    f"Missing: {sorted(missing, key=repr)}. Unknown: {sorted(unknown, key=repr)}."
                )
            weight_values = (weights[key] for key in weight_keys)
        elif isinstance(weights, Sequence):
            if len(weights) != len(loss_terms):
                raise ValueError("Mismatch between number of loss terms and number of term weights")
            weight_values = iter(weights)
        elif weights is None:
            weight_values = (1.0 for _ in loss_terms)
        else:
            raise ValueError("weights must be a mapping or sequence of scalar weights")

        self.terms = {}
        self.weights = {}
        composites = []
        for term_label, term, weight in zip(labels, loss_terms, weight_values, strict=True):
            if isinstance(term, CompositeLoss):
                composites.append((term_label, term, weight))
            else:
                name = get_unique_label(term_label, self.terms.keys())
                self.terms[name] = term
                self.weights[name] = weight

        for group_label, composite, group_weight in composites:
            group_label = group_label or composite.label
            for child_name, child in composite.terms.items():
                label = f"{group_label}_{child_name}" if group_label else child_name
                name = get_unique_label(label, self.terms.keys())
                self.terms[name] = child
                self.weights[name] = group_weight * composite.weights[child_name]

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
        states: State,
        trial_specs: "TaskTrialSpec",
        model: Component,
    ) -> TermTree[AbstractLoss]:
        """Evaluate, weight, and return all component terms.

        Arguments:
            states: Trajectories of system states for a set of trials.
            trial_specs: Task specifications for the set of trials.
        """
        children = {}
        for name, loss in self.terms.items():
            node = loss(states, trial_specs, model)
            children[name] = node.with_weight(self.weights[name])
        return TermTree.branch(self.label, children, originator=self)

    def without(self, *keys: str, label: Optional[str] = None) -> "CompositeLoss":
        """Return a new `CompositeLoss` without the specified terms."""
        return CompositeLoss(
            terms={k: v for k, v in self.terms.items() if k not in keys},
            weights={k: v for k, v in self.weights.items() if k not in keys},
            label=self.label if label is None else label,
        )

    def with_weights(self, new_weights: Mapping[str, float]) -> "CompositeLoss":
        """Return a new loss with selected weights updated by term name.

        Keys must name existing terms. Omitted weights retain their values.
        """
        unknown = set(new_weights).difference(self.terms)
        if unknown:
            raise ValueError(f"Unknown loss weight keys: {sorted(unknown, key=repr)}")
        return eqx.tree_at(lambda loss: loss.weights, self, {**self.weights, **new_weights})


# Maybe rename TargetValueSpec; I feel like a "`TargetSpec`" would include a `where` field
class TargetSpec(Module):
    """Associate a state's target value with time indices and discounting factors."""

    # `value` may be `None` when we specify default values for the other fields
    value: Optional[PyTree[Array]] = None
    time_idxs: Optional[Array | Callable] = None
    time_mask: Optional[Array | Callable] = None
    discount: Optional[Array | Callable] = None  # field(default_factory=lambda: jnp.array([1.0]))

    def __and__(self, other):
        # Allows user to do `target_zero & target_final_state`, for example.
        return eqx.combine(self, other)

    def __rand__(self, other):
        # Necessary for edge case of `None & spec`
        return eqx.combine(other, self)

    def get_time_mask(self, n: int) -> Optional[Array | Callable]:
        if self.time_idxs is None:
            return None
        mask = jnp.zeros((n,), dtype=bool)

        if callable(self.time_idxs):

            def mask_fn(trial_spec):
                idxs = self.time_idxs(trial_spec)  # type: ignore
                return mask.at[idxs].set(True)

            return mask_fn

        else:
            mask = jnp.zeros((n,), dtype=bool)
            mask = mask.at[self.time_idxs].set(True)
            return mask

    @property
    def batch_axes(self) -> PyTree[None | int]:
        """Target values map over trials. Time selectors are shared or callable."""
        return TargetSpec(value=0 if self.value is not None else None)


"""Useful partial target specs"""
target_final_state = TargetSpec(None, jnp.array([-1], dtype=int), None)
target_zero = TargetSpec(jnp.array(0.0), None, None)


class TargetStateLoss(AbstractLoss):
    """Penalize a state variable in comparison to a target value.

    !!! Note ""
        Currently only supports `where` functions that select a
        single state array, not a `PyTree[Array]`.

    Arguments:
        label: The label for the loss term.
        where: Function that takes the PyTree of model states, and
            returns the substate to be penalized.
        norm: Function which takes the difference between
            the substate and the target, and transforms it into a distance. For example,
            if the substate is effector position, then the substate-target difference
            gives the difference between the $x$ and $y$ position components, and the
            default `norm` function (`jnp.linalg.norm` on `axis=-1`) returns the
            Euclidean distance between the actual and target positions.
        spec: Gives default/constant values for the substate target, discount, and
            time index.
    """

    label: str
    where: Callable
    norm: Callable = lambda x: jnp.sum(x**2, axis=-1)
    # norm: Callable = lambda x: jnp.linalg.norm(x, axis=-1)  # Spatial distance
    spec: Optional[TargetSpec] = None  # Default/constant values.

    @cached_property
    def key(self):
        return WhereDict.key_transform(self.where)

    def term(
        self,
        states: Optional[PyTree],
        trial_specs: Optional["TaskTrialSpec"],
        model: Optional[Component],
    ) -> Array:
        """
        Arguments:
            trial_specs: Trial-by-trial information. In particular, if
                `trial_specs.targets` contains a `TargetSpec` entry mapped by
                `self.key`, the values of that `TargetSpec` instance will
                take precedence over the defaults specified by `self.spec`.
                This allows `AbstractTask` subclasses to specify trial-by-trial
                targets, where appropriate.
        """
        assert states is not None, "TargetStateLoss requires states, but states is None"
        assert trial_specs is not None, (
            "TargetStateLoss requires trial_specs, but trial_specs is None"
        )

        state, target, selectors = _retained_target_inputs(self, states, trial_specs)
        density = _target_density(self, state, target)
        return _sum_target_density(density, selectors, trial_specs)


def _resolve_target_spec(
    loss: TargetStateLoss, trial_specs: "TaskTrialSpec"
) -> tuple[TargetSpec, int | None]:
    """Resolve the target and the declared batch axis of its value."""
    trial_target = trial_specs.targets.get(loss.key)
    if isinstance(trial_target, Mapping):
        trial_target = trial_target[loss.label]
    if trial_target is not None and not isinstance(trial_target, TargetSpec):
        raise ValueError(f"TargetStateLoss '{loss.label}' expected a TargetSpec.")
    if trial_target is None:
        target = loss.spec
    elif loss.spec is None:
        target = trial_target
    else:
        target = eqx.combine(trial_target, loss.spec)
    if target is None or target.value is None:
        raise ValueError(
            f"TargetStateLoss '{loss.label}' requires a target value in its "
            "defaults or trial specifications."
        )
    trial_axis = (
        trial_target.batch_axes.value
        if trial_target is not None and trial_target.value is not None
        else None
    )
    return target, trial_axis


def _align_target_shape(
    target_shape: tuple[int, ...], state_shape: tuple[int, ...]
) -> tuple[int, ...]:
    """Align a single trial's target to (time, *features) without broadcasting it."""
    if len(target_shape) <= len(state_shape):
        aligned_shape = (1,) * (len(state_shape) - len(target_shape)) + target_shape
        if jnp.broadcast_shapes(aligned_shape, state_shape) == state_shape:
            return aligned_shape
    raise ValueError(f"Target shape {target_shape} cannot broadcast to {state_shape}.")


def _target_weight_selectors(target: TargetSpec, n_steps: int) -> tuple["WeightsSpec", ...]:
    mask = target.time_mask
    if mask is None:
        mask = target.get_time_mask(n_steps)
    return tuple(weight for weight in (mask, target.discount) if weight is not None)


def _norm_on_support(
    norm: Callable,
    residual: Array,
    support: Array | None,
    *,
    time_axis: int,
    density_time_axis: int,
) -> Array:
    if support is None:
        return norm(residual)

    def mask_along_time(ndim: int, axis: int) -> Array:
        shape = [1] * ndim
        shape[0] = support.shape[0]
        shape[axis] = support.shape[1]
        return support.reshape(shape)

    # A nonzero residual keeps Euclidean norm derivatives defined at inactive steps.
    residual = jnp.where(mask_along_time(residual.ndim, time_axis), residual, 1)
    density = norm(residual)
    return jnp.where(mask_along_time(density.ndim, density_time_axis), density, 0)


def _target_density(
    loss: TargetStateLoss, state: Array, target: Array, *, support: Array | None = None
) -> Array:
    if jnp.broadcast_shapes(state.shape, jnp.shape(target)) != state.shape:
        raise ValueError(
            f"TargetStateLoss '{loss.label}' target shape {jnp.shape(target)} "
            f"cannot broadcast to selected state shape {state.shape}."
        )
    return _norm_on_support(loss.norm, state - target, support, time_axis=1, density_time_axis=-1)


_LEGACY_TARGET_START_STEP = 1


def _target_start_step(trial_specs: "TaskTrialSpec") -> int:
    timeline = getattr(trial_specs, "timeline", None)
    return getattr(timeline, "target_start_step", _LEGACY_TARGET_START_STEP)


def _retained_target_inputs(
    loss: TargetStateLoss, states: PyTree, trial_specs: "TaskTrialSpec"
) -> tuple[Array, Array, tuple["WeightsSpec", ...]]:
    state = loss.where(states)[:, _target_start_step(trial_specs):]
    target, trial_axis = _resolve_target_spec(loss, trial_specs)
    value = _align_retained_target_value(jnp.asarray(target.value), trial_axis, state.shape[1:])
    return state, value, _target_weight_selectors(target, state.shape[1])


def _align_retained_target_value(
    value: Array, trial_axis: int | None, trial_state_shape: tuple[int, ...]
) -> Array:
    if trial_axis is not None and value.ndim:
        value = jnp.moveaxis(value, trial_axis, 0)
        return value.reshape(
            (value.shape[0], *_align_target_shape(value.shape[1:], trial_state_shape))
        )
    return value.reshape(_align_target_shape(value.shape, trial_state_shape))


def _sum_target_density(
    density: Array,
    selectors: Sequence["WeightsSpec"],
    trial_specs: "TaskTrialSpec",
    *,
    support: Array | None = None,
) -> Array:
    if not selectors:
        return density.sum(axis=-1)
    return reduce_over_time_with_weights(
        arr=density,
        trial_specs=trial_specs,
        time_axis=-1,
        trial_axis=0,
        trial_axis_specs=0,
        masks=selectors,
        support=support,
    )


WeightsSpec = float | Array | Callable[[object], float | Array]
# Callable takes a single trial's spec (PyTree leaf view) and returns scalar or (T,) weights


def _move_trial_axis_pytree(tree, trial_axis: int):
    def _move(x):
        if not isinstance(x, Array) or x.ndim < 2:
            return x
        return jnp.moveaxis(x, trial_axis, 0)

    return jt.map(_move, tree)


def _prepare_time_weight(selector: WeightsSpec, trial_spec, n_steps: int) -> Array:
    weight = jnp.asarray(selector(trial_spec) if callable(selector) else selector)
    if weight.ndim != 0 and weight.shape != (n_steps,):
        raise ValueError(
            f"Time weight must be scalar or shape ({n_steps},), got {weight.shape}."
        )
    return jnp.broadcast_to(weight, (n_steps,))


def _multiply_time_weights(weights: Sequence[Array], dtype) -> Array:
    return ft.reduce(jnp.multiply, (weight.astype(dtype) for weight in weights))


def _per_trial_weights(selector: WeightsSpec, specs, N: int, T: int) -> Array:
    if not callable(selector):
        return jnp.broadcast_to(_prepare_time_weight(selector, specs, T), (N, T))

    def one(spec):
        return _prepare_time_weight(selector, spec, T)

    axes = getattr(specs, "batch_axes", None)
    if axes is not None:
        axes = jt.broadcast(axes, specs, is_leaf=lambda x: x is None)
        axes = jt.map(
            lambda axis, leaf: axis if eqx.is_array(leaf) and leaf.ndim else None,
            axes,
            specs,
            is_leaf=lambda x: x is None,
        )
    else:
        axes = jt.map(lambda leaf: 0 if eqx.is_array(leaf) and leaf.ndim else None, specs)
    return eqx.filter_vmap(one, in_axes=(axes,), axis_size=N)(specs)


def _trial_time_perm(ndim: int, trial_axis: int, time_axis: int):
    """
    Permutation that moves TRIAL -> 0 and TIME -> -1 while preserving the
    relative order of all other axes.
    """
    ta = trial_axis if trial_axis >= 0 else ndim + trial_axis
    ti = time_axis if time_axis >= 0 else ndim + time_axis
    others = [ax for ax in range(ndim) if ax not in (ta, ti)]
    # Preserve original order of the non-(trial,time) axes
    return [ta, *others, ti]


def _inv_perm(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def reduce_over_time_with_weights(
    *,
    label: str = "",
    arr: Array,  # shape: (..., T, F?) pre-norm
    trial_specs,  # PyTree; each leaf has a trial axis
    time_axis: int,  # where T lives in `arr`
    trial_axis: int,  # where trial axis lives in `arr`
    trial_axis_specs: int,  # where trial axis lives in `trial_specs` leaves
    masks: Sequence[WeightsSpec],  # any number of scalar/(T,) or callable(spec_i)->scalar/(T,)
    support: Array | None = None,  # Boolean (trial, time) support of the input density.
) -> Array:
    """
    1) Transpose so: (trial, ..., time).
    2) Build combined per-trial weights W (N, T) by multiplying all masks.
    3) Weighted sum over time, restore trial axis, drop time axis.

    If provided, support zeros each weight before multiplication. The caller
    supplies an input density that is already zero outside that support.
    """
    # Put trial at 0 and time at -1 in one explicit transpose
    perm = _trial_time_perm(arr.ndim, trial_axis, time_axis)
    inv_perm = _inv_perm(perm)
    arr_rt = jnp.transpose(arr, perm)  # (N, ..., T)
    specs_T0 = _move_trial_axis_pytree(trial_specs, trial_axis_specs)

    T = arr_rt.shape[-1]
    dtype = arr_rt.dtype

    N = arr_rt.shape[0]
    if masks:
        prepared = [_per_trial_weights(selector, specs_T0, N, T) for selector in masks]
        if support is not None:
            prepared = [jnp.where(support, weight, 0) for weight in prepared]
        weights = _multiply_time_weights(prepared, dtype)
        weight_shape = (N,) + (1,) * (arr_rt.ndim - 2) + (T,)
        arr_rt = arr_rt * weights.reshape(weight_shape)
    reduced = arr_rt.sum(axis=-1)

    # Restore the trial axis to its original slot in the *output* (time removed)
    # inv_perm maps (trial, others..., time) -> original. Drop the 'time' entry,
    # then move axis 0 to where 'trial' originally lived.
    inv_perm_wo_time = [p for i, p in enumerate(inv_perm) if i != (len(inv_perm) - 1)]
    target_trial_axis = inv_perm_wo_time.index(0)
    out = jnp.moveaxis(reduced, 0, target_trial_axis)
    return out


"""Penalizes the effector's squared distance from the target position
across the trial."""
effector_pos_loss = TargetStateLoss(
    "Effector position",
    where=lambda state: state.mechanics.effector.pos,
    # Euclidean distance
    norm=lambda *args, **kwargs: (jnp.linalg.norm(*args, axis=-1, **kwargs) ** 2),
)


effector_vel_loss = TargetStateLoss(
    "Effector position",
    where=lambda state: state.mechanics.effector.vel,
    # Euclidean distance
    norm=lambda *args, **kwargs: (jnp.linalg.norm(*args, axis=-1, **kwargs) ** 2),
    spec=target_final_state,
)


class StopAtGoalLoss(AbstractLoss):
    """Encourages the effector to stop at the goal at least once.

    Each trial scores complete windows of ``window_len`` consecutive steps starting at
    or after the timeline's target start step. The mean cost of each window combines
    squared position error and squared velocity. ``softmin`` approximates the lowest
    window cost. ``soft-or`` penalizes a low probability that any window succeeds,
    with ``eps`` in ``(0, 1]`` setting a probability floor.

    The position target may be a shared feature vector, a per-trial feature vector,
    or a per-trial trajectory defined on the scored steps. The returned ``TermTree``
    contains one loss per trial and takes their mean in its total.
    """

    label: str = "effector_stop_at_goal"
    method: Literal["softmin", "soft-or"] = "softmin"
    std_dist: float = 0.025  # how close to goal is "at goal"
    std_vel: float = 0.05
    vel_weight: float = 1.0
    window_len: int = eqx.field(default=5, static=True)
    # softmin: how soft the min is (small -> credit focused on a single window)
    # soft-or: how close to the goal we need to be to register as a success
    tau: float = 0.2
    eps: float = 1e-6  # probability floor for soft-or

    def __check_init__(self):
        if self.window_len < 1:
            raise ValueError("StopAtGoalLoss window_len must be >= 1")

    def term(
        self,
        states: Optional[PyTree],
        trial_specs: Optional["TaskTrialSpec"],
        model: Optional[Component],
    ) -> Array:
        assert states is not None, "StopAtGoalLoss requires states"
        assert trial_specs is not None, "StopAtGoalLoss requires trial_specs"
        effector = states.mechanics.effector
        start = _target_start_step(trial_specs)
        pos = effector.pos[:, start:]
        vel = effector.vel[:, start:]
        if self.window_len > pos.shape[1]:
            raise ValueError(
                f"StopAtGoalLoss window_len ({self.window_len}) exceeds the number "
                f"of scored steps ({pos.shape[1]})."
            )

        goal = trial_specs.targets["mechanics.effector.pos"]
        value = jnp.asarray(goal.value)
        trial_axis = None if value.ndim == 1 else goal.batch_axes.value
        target = _align_retained_target_value(value, trial_axis, pos.shape[1:])
        target = jnp.broadcast_to(target, pos.shape)
        dtype = jnp.result_type(pos, vel, target, jnp.float32)
        dist_sq = jnp.sum((pos.astype(dtype) - target.astype(dtype)) ** 2, axis=-1)
        vel_sq = jnp.sum(vel.astype(dtype) ** 2, axis=-1)
        cost = dist_sq / self.std_dist**2 + self.vel_weight * vel_sq / self.std_vel**2
        window_cost = jax.lax.reduce_window(
            cost, 0.0, jax.lax.add, (1, self.window_len), (1, 1), "VALID"
        ) / self.window_len

        if self.method == "softmin":
            return softmin(window_cost, self.tau)
        elif self.method == "soft-or":
            log_no_success = jax.nn.log_sigmoid(window_cost / self.tau).sum(axis=-1)
            p_exist = -jnp.expm1(log_no_success)
            return -jnp.log(self.eps + (1 - self.eps) * p_exist)
        else:
            raise ValueError("method must be 'softmin' or 'soft-or'")


class ModelLoss(AbstractLoss):
    """Wrapper for functions that take a model, and return a scalar."""

    label: str
    loss_fn: Callable[[Component], Array]

    def term(
        self,
        states: Optional[PyTree],
        trial_specs: Optional["TaskTrialSpec"],
        model: Optional[Component],
    ) -> Array:
        assert model is not None, "ModelLoss requires a model, but model is None"
        return self.loss_fn(model)


class CrossTimestepLoss(AbstractLoss):
    """Marker base class for losses that consume neighbouring timesteps."""

    order: AbstractVar[int]
    where: AbstractVar[Callable]
    time_axis: AbstractVar[int]
    norm: AbstractVar[Callable]


def _squared_difference_norm(delta: Array, *, time_axis: int = 1) -> Array:
    feature_axes = tuple(axis for axis in range(1, delta.ndim) if axis != time_axis)
    return jnp.sum(delta**2, axis=feature_axes)


class NthDifferenceLoss(CrossTimestepLoss):
    """Penalize the squared N-th finite difference of a state variable.

    Computes ``mean(||diff(x, n=order)||²)`` over the time axis (default
    ``axis=1``, matching feedbax's ``(trial, time, ...)`` rollout layout). The
    selected substate may be a single ``Array`` or a ``PyTree`` of arrays: each
    leaf is differenced and squared-summed over its non-time/non-trial axes,
    then the contributions are summed across leaves.

    Trial is axis 0 unless time is on axis 0, which represents an unbatched
    trajectory. An explicit ``norm`` receives the differenced array in its
    original layout and must preserve the position of the time axis.

    ``order=1`` reproduces :class:`StateDerivativeLoss`; ``order=2`` reproduces
    :class:`OutputJerkLoss` when applied to effector velocity.
    """

    label: str = "nth_difference"
    order: int = eqx.field(default=1, static=True)
    where: Callable = lambda state: state
    time_axis: int = 1
    norm: Callable = _squared_difference_norm

    def __check_init__(self):
        if self.order < 1:
            raise ValueError("NthDifferenceLoss order must be >= 1")

    def _validate_length(self, n_steps: int) -> None:
        if n_steps <= self.order:
            raise ValueError(
                f"{type(self).__name__} order {self.order} requires more than "
                f"{self.order} timesteps, got {n_steps}."
            )

    def _resolve_time_axis(self, state: Array) -> int:
        return range(state.ndim)[self.time_axis]

    def _difference_density(
        self, state: Array, time_axis: int, *, support: Array | None = None
    ) -> tuple[Array, int]:
        """Return the per-step density and the position of its time axis."""
        self._validate_length(state.shape[time_axis])
        difference = jnp.diff(state, n=self.order, axis=time_axis)
        norm = self.norm
        density_time_axis = time_axis
        if norm is _squared_difference_norm:
            norm = ft.partial(_squared_difference_norm, time_axis=time_axis)
            density_time_axis = min(time_axis, 1)
        density = _norm_on_support(
            norm, difference, support, time_axis=time_axis, density_time_axis=density_time_axis
        )
        return density, density_time_axis

    def term(
        self,
        states: Optional[PyTree],
        trial_specs: Optional["TaskTrialSpec"],
        model: Optional[Component],
    ) -> Array:
        assert states is not None, f"{type(self).__name__} requires states"

        substate = self.where(states)

        def _per_leaf(arr: Array) -> Array:
            density, axis = self._difference_density(arr, self._resolve_time_axis(arr))
            return jnp.mean(density, axis=axis)

        per_leaf = jt.map(_per_leaf, substate)
        leaves = jt.leaves(per_leaf)
        if not leaves:
            raise ValueError(
                f"{type(self).__name__}('{self.label}'): `where` returned no array "
                "leaves to penalise."
            )
        return ft.reduce(jnp.add, leaves)


class StateDerivativeLoss(NthDifferenceLoss):
    """Penalize the squared first-difference of a state variable along time.

    Canonical semantic specialization of ``NthDifferenceLoss(order=1)`` for
    controller-state smoothness, with the default selector ``state.net.hidden``.

    Bug: efc4d68
    """

    label: str = "state_derivative"
    order: int = eqx.field(default=1, static=True)
    where: Callable = lambda state: state.net.hidden


class OutputJerkLoss(NthDifferenceLoss):
    """Penalise the squared discrete second-difference of a state variable along time.

    Computes ``mean(||x_{t+1} - 2 x_t + x_{t-1}||²)`` over the time axis (default
    ``axis=1``, matching feedbax's ``(trial, time, ...)`` rollout layout). When
    applied to a velocity trajectory (the default), the discrete second
    difference is the discrete jerk (third derivative of position) up to a
    constant ``dt`` factor — hence the class name.

    The selected substate may be a single ``Array`` or a ``PyTree`` of arrays —
    each leaf is second-differenced and squared-summed over its
    non-time/non-trial axes, then the contributions are summed across leaves.

    This term mirrors :class:`StateDerivativeLoss` but uses a cross-timestep
    order of 2 (needs ``t-1, t, t+1``) rather than 1 (needs only ``t-1, t``).
    Most commonly used as an output-jerk regulariser (Shahbazi, Codol, Michaels
    & Gribble 2025, Eq. 1) at large weight (e.g. ``1e5``) to suppress
    high-frequency motor commands.

    Attributes:
        label: Name for the loss term.
        where: Function ``state -> Array | PyTree[Array]`` selecting the
            substate to penalise. Default selects the effector velocity, so
            second-differencing yields jerk. Supplying a position-substate
            instead would yield acceleration penalty, not jerk.
        time_axis: Axis along which to take the second difference. Defaults to
            ``1`` (feedbax convention: trial=0, time=1, features...).
        norm: Function applied to the time-second-differenced state to produce a
            scalar-per-(trial,time) loss density before time/trial averaging.
            Default: squared L2 norm over feature axes.

    Note:
        This class declares a cross-timestep order of 2 (``order = 2``). It is
        invariant to the recurrent cell type and vmap-friendly for the same
        reason as :class:`StateDerivativeLoss` — it operates structurally on
        the rollout PyTree rather than knowing about cell internals.

    Bug: 7e1d257
    """

    label: str = "output_jerk"
    order: int = eqx.field(default=2, static=True)
    where: Callable = lambda state: state.mechanics.effector.vel


class EpochMaskedLoss(AbstractLoss):
    """Compositional time-mask wrapper that restricts a base loss to specified epochs.

    The mask selects per-(trial, time) contributions from
    ``trial_specs.timeline.epoch_bounds`` and the configured ``epoch_indices``.
    Inactive contributions are zero. The wrapped norm acts independently on
    each trial and timestep and should have a finite value and derivative on
    the all-ones residual used at inactive steps.

    For example, to penalise pre-go controller output on a `DelayedReaches`
    task with ``epoch_names=("hold", "target_on", "movement")``, wrap a
    ``TargetStateLoss`` selecting the controller output with
    ``epoch_indices=(0, 1)`` — the mask is then 1 during the hold and
    target_on epochs and 0 once the go cue arrives.

    Supported base losses: :class:`TargetStateLoss` and
    :class:`NthDifferenceLoss` family terms such as :class:`StateDerivativeLoss`
    (``order=1``) and :class:`OutputJerkLoss` (``order=2``). For cross-timestep
    base losses the mask is aligned to the
    differenced time axis: order ``k`` consumes ``k`` timesteps so a length-T
    mask becomes length ``T - k`` after differencing. We follow the
    "right-edge" convention: the mask sample at index ``i`` of the differenced
    array gates the contribution involving timesteps ``[i, i+1, …, i+k]``,
    which is mask-of-length-T sliced as ``mask[k:]``. This is the convention
    that makes "include this timestep's *outgoing* transition iff this
    timestep is in the masked set" come out cleanly.

    Attributes:
        base_loss: The wrapped per-step or cross-timestep loss term.
        epoch_indices: Tuple of epoch indices (into ``epoch_names``) to
            INCLUDE — the loss only contributes inside these epochs.
        label: Name for this loss term. Defaults to
            ``f"{base_loss.label}_epoch_masked"``.

    Bug: efc4d68 (rlrmp consumer)
    """

    label: str = ""
    base_loss: AbstractLoss = eqx.field(default=None)
    # Static so it doesn't end up as a traced PyTree leaf and so we can use it
    # as a Python tuple at trace-time. Tuples of small ints are JIT-stable.
    epoch_indices: tuple[int, ...] = eqx.field(default=(), static=True)

    def __check_init__(self):
        if self.base_loss is None:
            raise ValueError("EpochMaskedLoss requires a base_loss")
        if not self.label:
            # Equinox modules are immutable; resolve a sensible default via
            # eqx.tree_at-style assignment isn't possible here, so we fall
            # back to relying on the default-string via __post_init__-like
            # logic. Users are encouraged to pass an explicit label.
            object.__setattr__(self, "label", f"{self.base_loss.label}_epoch_masked")

    # ----------------------------------------------------------------------
    # Mask construction
    # ----------------------------------------------------------------------

    def _build_mask(self, trial_specs: "TaskTrialSpec", T: int) -> Array:
        """Build a per-trial time mask of shape ``(N, T)`` from epoch bounds.

        ``epoch_bounds`` is shape ``(N, E+1)`` (batched) or ``(E+1,)`` (per-trial).
        For each listed epoch index ``k``, the mask is 1 in
        ``[bounds[..., k], bounds[..., k+1])``.
        """
        timeline = trial_specs.timeline
        if timeline.epoch_bounds is None:
            raise ValueError(
                f"EpochMaskedLoss('{self.label}') requires "
                "trial_specs.timeline.epoch_bounds to be present."
            )
        bounds = timeline.epoch_bounds  # (N, E+1) or (E+1,)
        # Promote to (N, E+1) for uniform handling. The state arrays the base
        # loss reads will have a leading trial dim of size N, so we always
        # construct an (N, T) mask. If `bounds` is unbatched we broadcast.
        if bounds.ndim == 1:
            bounds = bounds[None, :]  # (1, E+1)
        # t-axis indices.
        t = jnp.arange(T, dtype=bounds.dtype)  # (T,)
        # Accumulate boolean OR over the listed epochs.
        mask = jnp.zeros((bounds.shape[0], T), dtype=jnp.bool_)
        for k in self.epoch_indices:
            s = bounds[:, k:k + 1]            # (N, 1)
            e = bounds[:, k + 1:k + 2]        # (N, 1)
            mask = mask | ((t[None, :] >= s) & (t[None, :] < e))
        return mask  # (N, T) bool

    def _mask_n_steps(self, trial_specs: "TaskTrialSpec", fallback: int) -> int:
        timeline_n_steps = getattr(trial_specs.timeline, "n_steps", None)
        if timeline_n_steps is None:
            return fallback
        return int(timeline_n_steps)

    def _align_mask_to_loss_time_axis(self, mask: Array, expected_len: int) -> Array:
        """Right-align a timeline mask to a loss density time axis."""
        mask_len = mask.shape[-1]
        if mask_len < expected_len:
            raise ValueError(
                f"EpochMaskedLoss('{self.label}') built a mask with {mask_len} "
                f"timesteps for a loss density with {expected_len} timesteps."
            )
        if mask_len > expected_len:
            return mask[:, mask_len - expected_len:]
        return mask

    # ----------------------------------------------------------------------
    # Dispatch
    # ----------------------------------------------------------------------

    @jax.named_scope("fbx.EpochMaskedLoss")
    def __call__(
        self,
        states: PyTree,
        trial_specs: "TaskTrialSpec",
        model: Component,
    ) -> TermTree["AbstractLoss"]:
        base = self.base_loss

        if isinstance(base, TargetStateLoss):
            value = self._call_target_state(base, states, trial_specs)
        elif isinstance(base, NthDifferenceLoss):
            value = self._call_cross_timestep(base, states, trial_specs)
        else:
            raise NotImplementedError(
                f"EpochMaskedLoss does not yet support wrapping "
                f"{type(base).__name__}. Supported base losses: "
                "TargetStateLoss and NthDifferenceLoss."
            )

        return TermTree.leaf(self.label, value, originator=self)

    def skeleton(self, batch_dims: tuple[int, ...]) -> TermTree:
        return TermTree.leaf(self.label, jnp.empty(batch_dims))

    # --- per-step base losses -------------------------------------------------

    def _call_target_state(
        self,
        base: "TargetStateLoss",
        states: PyTree,
        trial_specs: "TaskTrialSpec",
    ) -> Array:
        assert states is not None
        assert trial_specs is not None

        state, target, selectors = _retained_target_inputs(base, states, trial_specs)
        T = state.shape[1]
        n_steps = self._mask_n_steps(trial_specs, T + _target_start_step(trial_specs))
        support = self._align_mask_to_loss_time_axis(self._build_mask(trial_specs, n_steps), T)
        density = _target_density(base, state, target, support=support)
        return _sum_target_density(density, selectors, trial_specs, support=support)

    # --- cross-timestep base losses ------------------------------------------

    def _call_cross_timestep(
        self,
        base: NthDifferenceLoss,
        states: PyTree,
        trial_specs: "TaskTrialSpec",
    ) -> Array:
        """Apply per-trial mask to a cross-timestep loss of given `order`.

        Computes ``mean( m_t * ||Δ^order x||² )`` over time, where the mask
        ``m`` has length ``T - order`` (right-edge alignment).
        """
        assert states is not None
        assert trial_specs is not None

        leaves = jt.leaves(base.where(states))
        if not leaves:
            raise ValueError(
                f"EpochMaskedLoss('{self.label}'): wrapped {type(base).__name__} "
                "selected no array leaves to penalise."
            )
        T_full = leaves[0].shape[base._resolve_time_axis(leaves[0])]
        base._validate_length(T_full)
        T_diff = T_full - base.order
        n_steps = self._mask_n_steps(trial_specs, T_full)
        epoch_mask = self._build_mask(trial_specs, n_steps)
        # Full-rollout timelines drop the consumed initial samples; target-length
        # timelines already share the right edge with first-difference densities.
        support = self._align_mask_to_loss_time_axis(epoch_mask, T_diff)

        def _per_leaf(arr: Array) -> Array:
            density, axis = base._difference_density(
                arr, base._resolve_time_axis(arr), support=support
            )
            return jnp.mean(density, axis=axis)

        return ft.reduce(jnp.add, map(_per_leaf, leaves))


class EffectorStraightPathLoss(AbstractLoss):
    """Penalizes non-straight paths followed by the effector between initial
    and final position.

    !!! Info ""
        Divides the path length by the straight-line distance from the initial
        position to the selected endpoint. Differentiation uses a zero gradient
        for zero-length segments. With a zero reference distance, a stationary
        path scores zero and a moving path scores positive infinity, which can
        make the batch mean infinite. Both zero-reference cases use a zero
        gradient by convention.

    Attributes:
        label: The label for the loss term.
        normalize_by: Controls whether to normalize by the distance between the
            initial position & actual final position, or the initial position
            & task-specified goal position. For position target trajectories,
            the goal is the last target position.
    """

    label: str = "effector_path_straightness"
    normalize_by: Literal["actual", "goal"] = "actual"

    def term(
        self,
        states: Optional["SimpleFeedbackState"],
        trial_specs: Optional["TaskTrialSpec"],
        model: Optional[Component],
    ) -> Array:
        assert states is not None, "EffectorStraightPathLoss requires states"
        assert trial_specs is not None, "EffectorStraightPathLoss requires trial_specs"

        effector_pos = states.mechanics.effector.pos
        initial_pos = effector_pos[:, 0]
        pos_diff = jnp.diff(effector_pos, axis=1)
        piecewise_lengths = sqrt_with_zero_grad(jnp.sum(jnp.square(pos_diff), axis=-1))
        path_length = jnp.sum(piecewise_lengths, axis=1)
        if self.normalize_by == "actual":
            final_pos = effector_pos[:, -1]
        elif self.normalize_by == "goal":
            target = trial_specs.targets["mechanics.effector.pos"]
            final_pos = jnp.asarray(target.value)
            trial_axis = target.batch_axes.value
            time_axis = 0
            if trial_axis is not None and final_pos.ndim >= initial_pos.ndim:
                final_pos = jnp.moveaxis(final_pos, trial_axis, 0)
                time_axis = 1
            if final_pos.ndim == initial_pos.ndim + time_axis:
                if final_pos.shape[time_axis] == 0:
                    raise ValueError("EffectorStraightPathLoss requires a nonempty goal trajectory")
                final_pos = jnp.moveaxis(final_pos, time_axis, 0)[-1]
            final_pos = jnp.broadcast_to(final_pos, initial_pos.shape)
        else:
            raise ValueError("normalize_by must be 'actual' or 'goal'")
        init_final_diff = final_pos - initial_pos
        straight_length = sqrt_with_zero_grad(jnp.sum(jnp.square(init_final_diff), axis=-1))

        zero_reference = straight_length == 0
        # Keep the division's derivative finite at a zero reference distance.
        ratio = path_length / jnp.where(zero_reference, 1, straight_length)
        return jnp.where(zero_reference & (path_length > 0), jnp.inf, ratio)


def power_discount(n_steps: int, discount_exp: int = 6) -> Array:
    """A power-law vector that puts most of the weight on its later elements.

    Arguments:
        n_steps: The number of time steps in the trajectory to be weighted.
        discount_exp: The exponent of the power law.
    """
    if discount_exp == 0:
        return jnp.array(1.0)
    else:
        return jnp.linspace(1.0 / n_steps, 1.0, n_steps) ** discount_exp
