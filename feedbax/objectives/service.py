"""Service for converting loss specifications to feedbax loss objects."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import jax.numpy as jnp

from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.training import LossTermSpec, ObjectiveSlotSpec, TimeAggregationSpec
from feedbax.objectives.loss import AbstractLoss, CompositeLoss
from feedbax.objectives.spec import (
    ConstantScheduleSpec,
    EpochMaskSpec,
    FiniteDifferenceLossSpec,
    MatrixPayloadSpec,
    MatrixQuadraticLossSpec,
    MetricSpec,
    MovementEpochRampScheduleSpec,
    ObjectiveExecutionRequirements,
    ObjectiveSpec,
    ObjectiveTermSpec,
    PowerLawScheduleSpec,
    ReductionSpec,
    SelectorAddressSpec,
    TargetStateLossSpec,
    TargetValueSpec,
    TaskTimelineSpec,
    validate_objective_spec,
)


@dataclass
class ProbeInfo:
    """Information about a probe available for loss computation."""

    id: str
    label: str
    node: str
    timing: Literal["input", "output"]
    selector: str
    description: Optional[str] = None


@dataclass
class TimeRange:
    """A range of time indices."""

    start: Optional[int]
    end: Optional[int]


@dataclass
class TargetSpecResult:
    """Result of building a target specification."""

    mode: str
    time_range: Optional[TimeRange] = None
    segment_name: Optional[str] = None
    time_idxs: Optional[List[int]] = None
    discount_type: Optional[str] = None
    discount_exp: Optional[float] = None


NORM_FUNCTIONS: Dict[str, str] = {
    "squared_l2": "feedbax.loss.norms.squared_l2",
    "squared": "feedbax.loss.norms.squared_l2",
    "l2": "feedbax.loss.norms.l2",
    "l1": "feedbax.loss.norms.l1",
    "absolute": "feedbax.loss.norms.l1",
    "huber": "feedbax.loss.norms.huber",
}
_LEGACY_LOSS_TYPE_ALIASES = {
    "TargetStateLoss": "target_state",
    "target_state": "target_state",
    "MatrixQuadraticLoss": "matrix_quadratic",
    "matrix_quadratic": "matrix_quadratic",
}
_LEGACY_TIME_REDUCTION = {
    "all": "mean",
    "mean": "mean",
    "sum": "sum",
    "final": "final",
}


class ObjectiveLoweringError(ValueError):
    """Path-addressed objective lowering failure."""

    def __init__(self, path: str, message: str):
        self.path = path
        self.detail = message
        super().__init__(f"{path}: {message}")


@dataclass(frozen=True)
class LoweredObjective:
    """Executable objective bundle emitted by the shared lowering service."""

    loss: AbstractLoss
    requirements: ObjectiveExecutionRequirements
    source_kind: Literal["loss_term", "objective_spec"]


class SelectorObjectiveLoss(AbstractLoss):
    """Executable selector-addressed loss term lowered from durable specs."""

    label: str
    selector: str
    value_dtype: str | None
    target_selector: str | None
    target_value: Any
    norm: str
    huber_delta: float | None
    matrix: Any
    matrix_kind: Literal["dense", "diagonal"] | None
    reduction: ReductionSpec
    mask: EpochMaskSpec | None
    schedule: ConstantScheduleSpec | PowerLawScheduleSpec | MovementEpochRampScheduleSpec | None
    timeline: TaskTimelineSpec | None
    difference_order: int
    path: str

    def term(self, states: Any, trial_specs: Any, model: Any) -> Any:
        source = _select_runtime_value(
            states,
            trial_specs,
            model,
            self.selector,
            path=f"{self.path}/selector",
        )
        source = _cast_declared_dtype(
            source,
            self.value_dtype,
            path=f"{self.path}/selector/value_dtype",
        )
        if self.target_selector is not None:
            target = _select_runtime_value(
                states,
                trial_specs,
                model,
                self.target_selector,
                path=f"{self.path}/target_selector",
            )
        else:
            target = _resolve_target_value(
                self.target_value,
                trial_specs,
                path=f"{self.path}/target",
            )

        diff = jnp.asarray(source) - jnp.asarray(target)
        if self.difference_order:
            diff = jnp.diff(diff, n=self.difference_order, axis=1)
        values = _metric_values(
            diff,
            norm=self.norm,
            huber_delta=self.huber_delta,
            matrix=self.matrix,
            matrix_kind=self.matrix_kind,
            path=self.path,
        )
        values = _apply_time_weights(
            values,
            mask=self.mask,
            schedule=self.schedule,
            timeline=self.timeline,
            path=self.path,
        )
        return _reduce_objective_values(
            values,
            self.reduction,
            norm=self.norm,
            path=f"{self.path}/reduction",
        )


class LossService:
    """Service for managing loss configuration and conversion."""

    def get_available_probes(self, graph: GraphSpec) -> List[ProbeInfo]:
        """Extract all available probes from a graph specification.

        Probes can come from:
        1. Barnacles attached to nodes
        2. Taps in the graph
        3. Output ports of nodes (implicit probes)

        Args:
            graph: The graph specification to extract probes from.

        Returns:
            List of ProbeInfo objects describing available probes.
        """
        probes: List[ProbeInfo] = []

        # Extract probes from barnacles
        if graph.barnacles:
            for node_name, barnacles in graph.barnacles.items():
                for barnacle in barnacles:
                    if barnacle.kind == "probe":
                        probe = ProbeInfo(
                            id=barnacle.id,
                            label=barnacle.label,
                            node=node_name,
                            timing=barnacle.timing,
                            selector=f"probe:{barnacle.id}",
                            description=f"Probe on {node_name} ({barnacle.timing})",
                        )
                        probes.append(probe)

        # Extract probes from taps
        if graph.taps:
            for tap in graph.taps:
                if tap.type == "probe":
                    after_node = tap.position.get("afterNode", "unknown")
                    probe = ProbeInfo(
                        id=tap.id,
                        label=tap.id,
                        node=after_node,
                        timing="output",
                        selector=f"probe:{tap.id}",
                        description=f"Tap probe after {after_node}",
                    )
                    probes.append(probe)

        # Extract implicit probes from output ports
        for node_name, node_spec in graph.nodes.items():
            for output_port in node_spec.output_ports:
                probe = ProbeInfo(
                    id=f"{node_name}.{output_port}",
                    label=f"{node_name}.{output_port}",
                    node=node_name,
                    timing="output",
                    selector=f"port:{node_name}.{output_port}",
                    description=f"Output port {output_port} of {node_name}",
                )
                probes.append(probe)

        return probes

    def resolve_probe_selector(
        self, selector: str, graph: GraphSpec
    ) -> Optional[Dict[str, Any]]:
        """Resolve a probe selector string to a probe specification.

        Selector formats:
        - "probe:<id>" - Reference a barnacle or tap probe by ID
        - "port:<node>.<port>" - Reference a node output port
        - "path:<dotted.path>" - Reference a state path directly

        Args:
            selector: The selector string.
            graph: The graph specification.

        Returns:
            Dictionary with probe information, or None if not found.
        """
        if not selector:
            return None

        if selector.startswith("probe:"):
            probe_id = selector[6:]
            # Check barnacles
            if graph.barnacles:
                for node_name, barnacles in graph.barnacles.items():
                    for barnacle in barnacles:
                        if barnacle.id == probe_id:
                            return {
                                "type": "barnacle",
                                "node": node_name,
                                "barnacle_id": probe_id,
                                "timing": barnacle.timing,
                                "read_paths": barnacle.read_paths,
                            }
            # Check taps
            if graph.taps:
                for tap in graph.taps:
                    if tap.id == probe_id:
                        return {
                            "type": "tap",
                            "tap_id": probe_id,
                            "position": tap.position,
                            "paths": tap.paths,
                        }
            return None

        if selector.startswith("port:"):
            port_ref = selector[5:]
            if "." in port_ref:
                node_name, port_name = port_ref.rsplit(".", 1)
                if node_name in graph.nodes:
                    node_spec = graph.nodes[node_name]
                    if port_name in node_spec.output_ports:
                        return {
                            "type": "port",
                            "node": node_name,
                            "port": port_name,
                        }
            return None

        if selector.startswith("path:"):
            path = selector[5:]
            return {
                "type": "path",
                "path": path,
            }

        return None

    def build_time_aggregation(
        self, time_agg: Optional[TimeAggregationSpec]
    ) -> TargetSpecResult:
        """Build a target specification from time aggregation settings.

        Args:
            time_agg: The time aggregation specification.

        Returns:
            TargetSpecResult with the processed time aggregation.
        """
        if time_agg is None:
            return TargetSpecResult(mode="all")

        result = TargetSpecResult(mode=time_agg.mode)

        if time_agg.mode == "range":
            result.time_range = TimeRange(start=time_agg.start, end=time_agg.end)
        elif time_agg.mode == "segment":
            result.segment_name = time_agg.segment_name
        elif time_agg.mode == "custom":
            result.time_idxs = time_agg.time_idxs

        if time_agg.discount and time_agg.discount != "none":
            result.discount_type = time_agg.discount
            result.discount_exp = time_agg.discount_exp

        return result

    def get_norm_function(self, norm: Optional[str]) -> Optional[str]:
        """Get the norm function path for a given norm name.

        Args:
            norm: The norm name (squared_l2, l2, l1, huber).

        Returns:
            The fully qualified path to the norm function, or None.
        """
        if norm is None:
            return None
        return NORM_FUNCTIONS.get(norm)

    def validate_loss_spec(
        self, spec: LossTermSpec, graph: GraphSpec
    ) -> List[Dict[str, Any]]:
        """Validate a loss specification against a graph.

        Args:
            spec: The loss term specification to validate.
            graph: The graph specification.

        Returns:
            List of validation errors (empty if valid).
        """
        errors: List[Dict[str, Any]] = []

        def validate_term(
            term: LossTermSpec, path: List[str]
        ) -> None:
            # Validate selector if present
            if term.selector:
                resolved = self.resolve_probe_selector(term.selector, graph)
                if resolved is None:
                    errors.append({
                        "path": path,
                        "field": "selector",
                        "message": f"Could not resolve selector: {term.selector}",
                    })

            # Validate time aggregation
            if term.time_agg:
                if term.time_agg.mode == "range":
                    if term.time_agg.start is None or term.time_agg.end is None:
                        errors.append({
                            "path": path,
                            "field": "time_agg",
                            "message": "Range mode requires start and end indices",
                        })
                    elif term.time_agg.start > term.time_agg.end:
                        errors.append({
                            "path": path,
                            "field": "time_agg",
                            "message": "Start index must be less than or equal to end index",
                        })
                elif term.time_agg.mode == "segment":
                    if not term.time_agg.segment_name:
                        errors.append({
                            "path": path,
                            "field": "time_agg",
                            "message": "Segment mode requires segment_name",
                        })
                elif term.time_agg.mode == "custom":
                    if not term.time_agg.time_idxs:
                        errors.append({
                            "path": path,
                            "field": "time_agg",
                            "message": "Custom mode requires time_idxs",
                        })

                if term.time_agg.discount == "power" and term.time_agg.discount_exp is None:
                    errors.append({
                        "path": path,
                        "field": "time_agg",
                        "message": "Power discount requires discount_exp",
                    })

            # Validate norm
            if term.norm and term.norm not in NORM_FUNCTIONS:
                errors.append({
                    "path": path,
                    "field": "norm",
                    "message": f"Unknown norm function: {term.norm}",
                })

            # Validate weight
            if term.weight < 0:
                errors.append({
                    "path": path,
                    "field": "weight",
                    "message": "Weight must be non-negative",
                })

            # Recursively validate children
            if term.children:
                for child_key, child_term in term.children.items():
                    validate_term(child_term, path + [child_key])

        validate_term(spec, [])
        return errors

    def spec_to_loss_config(
        self, spec: LossTermSpec, graph: GraphSpec
    ) -> Dict[str, Any]:
        """Convert a loss specification to a loss configuration dictionary.

        This produces a configuration that can be used to instantiate
        feedbax loss objects.

        Args:
            spec: The loss term specification.
            graph: The graph specification.

        Returns:
            Dictionary configuration for the loss.
        """
        config: Dict[str, Any] = {
            "type": spec.type,
            "label": spec.label,
            "weight": spec.weight,
        }

        if spec.selector:
            resolved = self.resolve_probe_selector(spec.selector, graph)
            if resolved:
                config["probe"] = resolved

        if spec.norm:
            config["norm"] = self.get_norm_function(spec.norm)

        if spec.matrix is not None:
            config["matrix"] = spec.matrix
            config["matrix_kind"] = spec.matrix_kind or "dense"

        if spec.time_agg:
            target_spec = self.build_time_aggregation(spec.time_agg)
            time_agg_config: Dict[str, Any] = {
                "mode": target_spec.mode,
            }
            if target_spec.time_range:
                time_agg_config["start"] = target_spec.time_range.start
                time_agg_config["end"] = target_spec.time_range.end
            if target_spec.segment_name:
                time_agg_config["segment_name"] = target_spec.segment_name
            if target_spec.time_idxs:
                time_agg_config["time_idxs"] = target_spec.time_idxs
            if target_spec.discount_type:
                time_agg_config["discount"] = target_spec.discount_type
                time_agg_config["discount_exp"] = target_spec.discount_exp
            config["time_aggregation"] = time_agg_config

        if spec.children:
            config["children"] = {
                key: self.spec_to_loss_config(child, graph)
                for key, child in spec.children.items()
            }

        return config

    def lower_objective_slot(
        self,
        slot: ObjectiveSlotSpec,
        *,
        graph: GraphSpec | None = None,
        trial_axis: str = "batch",
        path: str = "/objective",
    ) -> LoweredObjective:
        """Lower a training-run objective slot through the shared service."""

        if slot.kind == "loss_term":
            if slot.loss is None:
                raise ObjectiveLoweringError(f"{path}/loss", "loss_term slot is missing loss")
            return self.lower_loss_term_spec(
                slot.loss,
                graph=graph,
                trial_axis=trial_axis,
                path=f"{path}/loss",
            )
        if slot.kind == "objective_spec":
            if slot.payload is None:
                raise ObjectiveLoweringError(f"{path}/payload", "objective_spec slot is missing payload")
            return self.lower_objective_spec(
                slot.payload,
                trial_axis=trial_axis,
                path=f"{path}/payload",
            )
        raise ObjectiveLoweringError(
            path,
            f"objective kind {slot.kind!r} is external and cannot be lowered by Feedbax",
        )

    def lower_loss_term_spec(
        self,
        spec: LossTermSpec,
        *,
        graph: GraphSpec | None = None,
        trial_axis: str = "batch",
        path: str = "/loss",
    ) -> LoweredObjective:
        """Lower a legacy ``LossTermSpec`` to an executable loss object."""

        if graph is not None:
            errors = self.validate_loss_spec(spec, graph)
            if errors:
                first = errors[0]
                error_path = _join_path(path, *(str(part) for part in first["path"]))
                raise ObjectiveLoweringError(
                    f"{error_path}/{first['field']}",
                    first["message"],
                )
        objective = loss_term_spec_to_objective_spec(spec, path=path)
        lowered = self.lower_objective_spec(
            objective,
            trial_axis=trial_axis,
            path=path,
        )
        return LoweredObjective(
            loss=lowered.loss,
            requirements=lowered.requirements,
            source_kind="loss_term",
        )

    def lower_objective_spec(
        self,
        spec: ObjectiveSpec | dict[str, Any],
        *,
        trial_axis: str = "batch",
        path: str = "/objective",
    ) -> LoweredObjective:
        """Lower a durable ``ObjectiveSpec`` to an executable composite loss."""

        objective = validate_objective_spec(spec)
        terms: dict[str, AbstractLoss] = {}
        weights: dict[str, float] = {}
        trial_reductions: list[str] = []
        for index, term in enumerate(objective.terms):
            term_path = f"{path}/terms/{index}"
            terms[term.label] = self._lower_objective_term(
                term,
                timeline=objective.timeline,
                path=term_path,
            )
            weights[term.label] = term.weight
            trial_reductions.append(term.reduction.trial)

        loss: AbstractLoss
        if not terms:
            raise ObjectiveLoweringError(f"{path}/terms", "objective must contain at least one term")
        if len(terms) == 1:
            loss = CompositeLoss(terms=terms, weights=weights, label="objective")
        else:
            loss = CompositeLoss(terms=terms, weights=weights, label="objective")
        requirements = _requirements_from_reductions(trial_reductions, trial_axis=trial_axis)
        return LoweredObjective(loss=loss, requirements=requirements, source_kind="objective_spec")

    def _lower_objective_term(
        self,
        term: ObjectiveTermSpec,
        *,
        timeline: TaskTimelineSpec | None,
        path: str,
    ) -> AbstractLoss:
        lowerer = _OBJECTIVE_TERM_LOWERERS.get(type(term))
        if lowerer is None:
            raise ObjectiveLoweringError(f"{path}/type", f"unknown objective term {type(term).__name__}")
        return lowerer(term, timeline=timeline, path=path)


def loss_term_spec_to_objective_spec(
    spec: LossTermSpec | Mapping[str, Any],
    *,
    path: str = "/loss",
) -> ObjectiveSpec:
    """Adapt a legacy ``LossTermSpec`` tree to modern objective/reduction specs."""

    root = spec if isinstance(spec, LossTermSpec) else LossTermSpec.model_validate(spec)
    terms: list[ObjectiveTermSpec] = []
    used_labels: set[str] = set()

    def visit(term: LossTermSpec, keys: tuple[str, ...], weight: float, term_path: str) -> None:
        if term.children:
            for child_key, child in term.children.items():
                visit(
                    child,
                    (*keys, child_key),
                    weight * child.weight,
                    f"{term_path}/children/{child_key}",
                )
            return

        label = _unique_objective_label("_".join(keys) if keys else term.label, used_labels)
        used_labels.add(label)
        terms.append(
            _legacy_leaf_to_objective_term(
                term,
                label=label,
                weight=weight,
                path=term_path,
            )
        )

    visit(root, (), 1.0, path)
    if not terms:
        raise ObjectiveLoweringError(f"{path}/children", "loss spec must contain at least one leaf term")
    return ObjectiveSpec(
        terms=terms,
        metadata={
            "lowered_from": "feedbax.spec.training.loss_term",
            "source_schema_version": root.schema_version,
        },
    )


def _legacy_leaf_to_objective_term(
    term: LossTermSpec,
    *,
    label: str,
    weight: float,
    path: str,
) -> ObjectiveTermSpec:
    if not term.selector:
        raise ObjectiveLoweringError(f"{path}/selector", f"loss leaf {term.label!r} requires selector")
    if term.target_selector is not None and term.target_value is not None:
        raise ObjectiveLoweringError(
            path,
            "loss leaf cannot specify both target_selector and target_value",
        )
    term_type = _LEGACY_LOSS_TYPE_ALIASES.get(term.type)
    if term_type is None:
        raise ObjectiveLoweringError(f"{path}/type", f"unknown loss term type {term.type!r}")

    reduction = _legacy_reduction_spec(term.time_agg, path=f"{path}/time_agg")
    metadata = _legacy_term_metadata(term)
    selector = _selector_address(term.selector, matrix=term_type == "matrix_quadratic")

    if term_type == "target_state":
        if term.target_selector is None and term.target_value is None:
            raise ObjectiveLoweringError(
                path,
                "loss leaf requires either target_selector or target_value",
            )
        return TargetStateLossSpec(
            label=label,
            weight=weight,
            selector=selector,
            target=_legacy_target_spec(term, path=path),
            metric=_legacy_metric_spec(term.norm),
            reduction=reduction,
            metadata=metadata,
        )

    _validate_matrix_payload(term.matrix, term.matrix_kind, path=path)
    if term.matrix is None:
        raise ObjectiveLoweringError(f"{path}/matrix", "matrix_quadratic loss requires matrix")
    return MatrixQuadraticLossSpec(
        label=label,
        weight=weight,
        selector=selector,
        target=_legacy_target_spec(term, path=path, default_value=0.0),
        matrix=MatrixPayloadSpec(kind=term.matrix_kind or "dense", value=term.matrix),
        metric=MetricSpec(kind="squared_l2"),
        reduction=reduction,
        metadata=metadata,
    )


def _legacy_reduction_spec(
    time_agg: TimeAggregationSpec | None,
    *,
    path: str,
) -> ReductionSpec:
    if time_agg is not None and time_agg.discount not in (None, "none"):
        raise ObjectiveLoweringError(
            f"{path}/discount",
            "legacy time_agg.discount has no executable ObjectiveSpec equivalent",
        )
    mode = "mean" if time_agg is None else time_agg.mode
    time = _LEGACY_TIME_REDUCTION.get(mode)
    if time is None:
        raise ObjectiveLoweringError(
            f"{path}/mode",
            f"legacy time aggregation {mode!r} has no ObjectiveSpec equivalent",
        )
    return ReductionSpec(time=time, trial="mean", feature="sum")


def _legacy_target_spec(
    term: LossTermSpec,
    *,
    path: str,
    default_value: Any = None,
) -> TargetValueSpec:
    if term.target_selector is not None:
        return TargetValueSpec(
            kind="selector",
            selector=_selector_address(term.target_selector),
        )
    value = default_value if term.target_value is None else term.target_value
    return TargetValueSpec(kind="constant", value=value)


def _legacy_metric_spec(norm: str | None) -> MetricSpec:
    kind = norm or "squared_l2"
    if kind == "huber":
        return MetricSpec(kind="huber", huber_delta=1.0)
    return MetricSpec(kind=kind)


def _legacy_term_metadata(term: LossTermSpec) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "legacy_loss_type": term.type,
    }
    if term.retention is not None:
        metadata["legacy_retention"] = term.retention.model_dump(mode="json", exclude_none=True)
    return metadata


def _selector_address(selector: str, *, matrix: bool = False) -> SelectorAddressSpec:
    kind = "state"
    if selector.startswith("port:"):
        kind = "port"
    elif selector.startswith("probe:"):
        kind = "probe"
    elif selector.startswith(("trial_specs.", "task_data.")):
        kind = "task_data"
    return SelectorAddressSpec(
        selector=selector,
        kind=kind,
        temporal_axis="time",
        feature_axis="coordinate" if matrix else None,
    )


def _unique_objective_label(label: str, used: set[str]) -> str:
    if label not in used:
        return label
    index = 1
    while f"{label}_{index}" in used:
        index += 1
    return f"{label}_{index}"


def _lower_selector_objective_term(
    term: TargetStateLossSpec | MatrixQuadraticLossSpec | FiniteDifferenceLossSpec,
    *,
    timeline: TaskTimelineSpec | None,
    path: str,
) -> AbstractLoss:
    target = term.target if isinstance(term, (TargetStateLossSpec, MatrixQuadraticLossSpec)) else None
    target_selector, target_value = _objective_target_payload(target, path=path)
    matrix = term.matrix.value if isinstance(term, MatrixQuadraticLossSpec) else None
    matrix_kind = term.matrix.kind if isinstance(term, MatrixQuadraticLossSpec) else None
    _validate_matrix_payload(matrix, matrix_kind, path=path)
    if isinstance(term.schedule, PowerLawScheduleSpec) and term.schedule.window == "epoch":
        raise ObjectiveLoweringError(
            f"{path}/schedule",
            "epoch power-law schedule requires executor timeline binding",
        )
    return SelectorObjectiveLoss(
        label=term.label,
        selector=term.selector.selector,
        value_dtype=term.selector.value_dtype,
        target_selector=target_selector,
        target_value=target_value,
        norm=term.metric.kind,
        huber_delta=term.metric.huber_delta,
        matrix=matrix,
        matrix_kind=matrix_kind,
        reduction=term.reduction,
        mask=term.mask,
        schedule=term.schedule,
        timeline=timeline,
        difference_order=term.order if isinstance(term, FiniteDifferenceLossSpec) else 0,
        path=path,
    )


_OBJECTIVE_TERM_LOWERERS: dict[type[ObjectiveTermSpec], Callable[..., AbstractLoss]] = {
    TargetStateLossSpec: _lower_selector_objective_term,
    MatrixQuadraticLossSpec: _lower_selector_objective_term,
    FiniteDifferenceLossSpec: _lower_selector_objective_term,
}


def _join_path(base: str, *parts: str) -> str:
    suffix = "/".join(part for part in parts if part)
    return f"{base}/{suffix}" if suffix else base


def _requirements_from_reductions(
    reductions: tuple[str, ...] | list[str],
    *,
    trial_axis: str,
) -> ObjectiveExecutionRequirements:
    semantics = {reduction for reduction in reductions if reduction not in {"all", "none"}}
    if not semantics:
        return ObjectiveExecutionRequirements()
    if len(semantics) > 1:
        raise ObjectiveLoweringError(
            "/objective/reduction/trial",
            f"objective terms disagree on trial-axis aggregation semantics {sorted(semantics)!r}",
        )
    semantic = semantics.pop()
    return ObjectiveExecutionRequirements(
        requires_axes=[trial_axis],
        aggregation_semantics={trial_axis: semantic},
    )


def _objective_target_payload(target: TargetValueSpec | None, *, path: str) -> tuple[str | None, Any]:
    if target is None:
        return None, 0.0
    if target.kind == "constant":
        return None, target.value
    if target.kind == "selector":
        if target.selector is None:
            raise ObjectiveLoweringError(f"{path}/target/selector", "selector target is missing selector")
        return target.selector.selector, None
    if target.kind == "task_target":
        return f"trial_specs.targets.{target.target_key}", None
    raise ObjectiveLoweringError(f"{path}/target/kind", f"unsupported target kind {target.kind!r}")


def _select_runtime_value(states: Any, trial_specs: Any, model: Any, selector: str, *, path: str) -> Any:
    root: Any
    parts: list[str]
    if selector.startswith("state."):
        root = states
        parts = selector.split(".")[1:]
    elif selector.startswith("states."):
        root = states
        parts = selector.split(".")[1:]
    elif selector.startswith("trial_specs."):
        root = trial_specs
        parts = selector.split(".")[1:]
    elif selector.startswith("task_data."):
        root = trial_specs
        parts = selector.split(".")
    elif selector.startswith("model."):
        root = model
        parts = selector.split(".")[1:]
    elif selector.startswith("path:"):
        root = states
        parts = selector[5:].split(".")
    elif selector.startswith("port:") or selector.startswith("probe:"):
        raise ObjectiveLoweringError(
            path,
            f"selector {selector!r} requires retained-observable executor binding",
        )
    else:
        root = states
        parts = selector.split(".")

    value = root
    for part in parts:
        if part == "":
            continue
        if isinstance(value, dict):
            if part not in value:
                raise ObjectiveLoweringError(path, f"selector {selector!r} missing key {part!r}")
            value = value[part]
        else:
            try:
                value = getattr(value, part)
            except AttributeError as exc:
                raise ObjectiveLoweringError(
                    path,
                    f"selector {selector!r} missing attribute {part!r}",
                ) from exc
    return value


def _resolve_target_value(target: Any, trial_specs: Any, *, path: str) -> Any:
    if isinstance(target, str) and target.startswith("trial_specs."):
        return _select_runtime_value(None, trial_specs, None, target, path=path)
    if hasattr(target, "value"):
        return target.value
    return target


def _cast_declared_dtype(value: Any, dtype: str | None, *, path: str) -> Any:
    if dtype is None:
        return value
    try:
        return jnp.asarray(value, dtype=jnp.dtype(dtype))
    except TypeError as exc:
        raise ObjectiveLoweringError(path, f"unsupported selector value_dtype {dtype!r}") from exc


def _metric_values(
    values: Any,
    *,
    norm: str,
    huber_delta: float | None,
    matrix: Any,
    matrix_kind: Literal["dense", "diagonal"] | None,
    path: str,
) -> Any:
    arr = jnp.asarray(values)
    if matrix is not None:
        return _matrix_quadratic_values(arr, matrix, matrix_kind or "dense", path=f"{path}/matrix")
    if norm in {"squared_l2", "squared"}:
        return jnp.square(arr)
    if norm in {"l1", "absolute"}:
        return jnp.abs(arr)
    if norm == "l2":
        return jnp.square(arr)
    if norm == "huber":
        if huber_delta is None:
            raise ObjectiveLoweringError(f"{path}/metric", "huber metric requires huber_delta")
        abs_arr = jnp.abs(arr)
        delta = jnp.asarray(huber_delta, dtype=arr.dtype)
        return jnp.where(
            abs_arr <= delta,
            0.5 * jnp.square(arr),
            delta * (abs_arr - 0.5 * delta),
        )
    raise ObjectiveLoweringError(f"{path}/metric", f"unsupported norm {norm!r}")


def _matrix_quadratic_values(
    arr: Any,
    matrix: Any,
    matrix_kind: Literal["dense", "diagonal"],
    *,
    path: str,
) -> Any:
    mat = jnp.asarray(matrix)
    if arr.ndim == 0:
        if mat.ndim != 0:
            raise ObjectiveLoweringError(path, "scalar quadratic requires scalar matrix")
        return mat * jnp.square(arr)
    if matrix_kind == "diagonal":
        if mat.ndim != 1:
            raise ObjectiveLoweringError(path, "diagonal matrix must be rank 1")
        if mat.shape[0] != arr.shape[-1]:
            raise ObjectiveLoweringError(
                path,
                "diagonal matrix length must match selected feature dimension",
            )
        return jnp.sum(jnp.square(arr) * mat, axis=-1)
    if matrix_kind == "dense":
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ObjectiveLoweringError(path, "dense matrix must be square rank 2")
        if mat.shape[0] != arr.shape[-1]:
            raise ObjectiveLoweringError(
                path,
                "dense matrix shape must match selected feature dimension",
            )
        return jnp.einsum("...i,ij,...j->...", arr, mat, arr)
    raise ObjectiveLoweringError(path, f"unsupported matrix kind {matrix_kind!r}")


def _validate_matrix_payload(matrix: Any, matrix_kind: str | None, *, path: str) -> None:
    if matrix is None:
        return
    mat = jnp.asarray(matrix)
    kind = matrix_kind or "dense"
    if kind == "dense" and (mat.ndim != 2 or mat.shape[0] != mat.shape[1]):
        raise ObjectiveLoweringError(f"{path}/matrix", "dense matrix must be square rank 2")
    if kind == "diagonal" and mat.ndim != 1:
        raise ObjectiveLoweringError(f"{path}/matrix", "diagonal matrix must be rank 1")
    if kind not in {"dense", "diagonal"}:
        raise ObjectiveLoweringError(f"{path}/matrix_kind", f"unsupported matrix kind {kind!r}")


def _apply_time_weights(
    values: Any,
    *,
    mask: EpochMaskSpec | None,
    schedule: ConstantScheduleSpec | PowerLawScheduleSpec | MovementEpochRampScheduleSpec | None,
    timeline: TaskTimelineSpec | None,
    path: str,
) -> Any:
    arr = jnp.asarray(values)
    if arr.ndim < 2:
        if mask is not None or not isinstance(schedule, (type(None), ConstantScheduleSpec)):
            raise ObjectiveLoweringError(path, "time masks and schedules require a time axis")
        return arr
    weights = jnp.ones((arr.shape[1],), dtype=arr.dtype)
    if mask is not None:
        weights = weights * _timeline_mask(mask, timeline, arr.shape[1], path=f"{path}/mask")
    if schedule is not None:
        weights = weights * _schedule_weights(schedule, timeline, arr.shape[1], path=f"{path}/schedule")
    shape = [1] * arr.ndim
    shape[1] = arr.shape[1]
    return arr * weights.reshape(shape)


def _timeline_mask(
    mask: EpochMaskSpec,
    timeline: TaskTimelineSpec | None,
    n_steps: int,
    *,
    path: str,
) -> Any:
    if timeline is None:
        raise ObjectiveLoweringError(path, "epoch mask requires objective timeline")
    epoch_by_name = {epoch.name: epoch for epoch in timeline.epochs}
    weights = jnp.zeros((n_steps,), dtype=bool)
    for epoch_name in mask.epochs:
        epoch = epoch_by_name.get(epoch_name)
        if epoch is None:
            raise ObjectiveLoweringError(path, f"unknown epoch {epoch_name!r}")
        if epoch.length_range is None:
            raise ObjectiveLoweringError(
                path,
                f"epoch {epoch_name!r} requires length_range for static lowering",
            )
        start = sum((item.length_range or (0, 0))[0] for item in timeline.epochs[:epoch.index])
        end = min(start + epoch.length_range[0], n_steps)
        weights = weights.at[start:end].set(True)
    return weights if mask.mode == "include" else jnp.logical_not(weights)


def _schedule_weights(
    schedule: ConstantScheduleSpec | PowerLawScheduleSpec | MovementEpochRampScheduleSpec,
    timeline: TaskTimelineSpec | None,
    n_steps: int,
    *,
    path: str,
) -> Any:
    if isinstance(schedule, ConstantScheduleSpec):
        return jnp.full((n_steps,), schedule.value)
    if isinstance(schedule, PowerLawScheduleSpec):
        if schedule.window == "epoch":
            raise ObjectiveLoweringError(path, "epoch power-law schedule requires executor timeline binding")
        x = jnp.linspace(0.0, 1.0, n_steps)
        curve = x ** schedule.exponent
        return schedule.start_value + (schedule.end_value - schedule.start_value) * curve
    if isinstance(schedule, MovementEpochRampScheduleSpec):
        if timeline is None:
            raise ObjectiveLoweringError(path, "movement_epoch_ramp requires objective timeline")
        epoch_by_name = {epoch.name: epoch for epoch in timeline.epochs}
        epoch = epoch_by_name.get(schedule.anchor_epoch)
        if epoch is None or epoch.length_range is None:
            raise ObjectiveLoweringError(
                path,
                f"movement ramp anchor {schedule.anchor_epoch!r} needs length_range",
            )
        start = sum((item.length_range or (0, 0))[0] for item in timeline.epochs[:epoch.index])
        local = jnp.clip((jnp.arange(n_steps) - start) / schedule.duration_steps, 0.0, 1.0)
        if schedule.shape == "linear":
            curve = local
        elif schedule.shape == "cosine":
            curve = 0.5 - 0.5 * jnp.cos(jnp.pi * local)
        elif schedule.shape == "power":
            curve = local ** float(schedule.power)
        else:
            raise ObjectiveLoweringError(path, f"unsupported ramp shape {schedule.shape!r}")
        if not schedule.hold_after:
            curve = jnp.where(jnp.arange(n_steps) <= start + schedule.duration_steps, curve, 0.0)
        return schedule.start_value + (schedule.end_value - schedule.start_value) * curve
    raise ObjectiveLoweringError(path, f"unsupported schedule {type(schedule).__name__}")


def _reduce_objective_values(
    values: Any,
    reduction: ReductionSpec,
    *,
    norm: str,
    path: str,
) -> Any:
    arr = jnp.asarray(values)
    if arr.ndim >= 3:
        arr = _apply_reduction(
            arr,
            axis=-1,
            kind=reduction.feature,
            tail_fraction=reduction.tail_fraction,
            path=f"{path}/feature",
        )
        if norm == "l2":
            arr = jnp.sqrt(arr)
    elif norm == "l2":
        arr = jnp.sqrt(arr)
    if arr.ndim >= 2:
        if reduction.time == "final":
            arr = jnp.take(arr, -1, axis=1)
        else:
            arr = _apply_reduction(
                arr,
                axis=1,
                kind=reduction.time,
                tail_fraction=reduction.tail_fraction,
                path=f"{path}/time",
            )
    if arr.ndim >= 1:
        arr = _apply_reduction(
            arr,
            axis=0,
            kind=reduction.trial,
            tail_fraction=reduction.tail_fraction,
            path=f"{path}/trial",
        )
    return arr


def _apply_reduction(
    values: Any,
    *,
    axis: int,
    kind: str,
    tail_fraction: float,
    path: str,
) -> Any:
    if kind == "none":
        return values
    if kind == "mean":
        return jnp.mean(values, axis=axis)
    if kind == "sum":
        return jnp.sum(values, axis=axis)
    if kind == "tail":
        size = values.shape[axis]
        if size <= 0:
            raise ObjectiveLoweringError(path, "tail reduction requires non-empty axis")
        k = max(1, math.ceil(size * tail_fraction))
        sorted_values = jnp.sort(values, axis=axis)
        tail = jnp.take(sorted_values, jnp.arange(size - k, size), axis=axis)
        return jnp.mean(tail, axis=axis)
    raise ObjectiveLoweringError(path, f"unsupported reduction {kind!r}")


# Singleton instance
loss_service = LossService()
