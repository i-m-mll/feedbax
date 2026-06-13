"""Retained-observable and executable loss-plan lowering.

This module is deliberately independent of the Studio worker loop. It defines
the contract that turns graph-authored retained observables and loss selectors
into a validated retention plan, then evaluates lowered loss terms against a
trace map produced by graph execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Optional

import jax.numpy as jnp

from feedbax.contracts.graph import (
    AnalysisInputRequirement,
    GraphSpec,
    RetainedObservableSpec,
    RetainedObservableTargetSpec,
    RetentionPolicySpec,
)
from feedbax.task_timeline_masks import (
    TaskTimelineMaskError,
    align_time_mask,
    build_task_timeline_mask,
)
from feedbax.contracts.training import LossTermSpec, TimeAggregationSpec, TrainingSpec


SelectorKind = Literal[
    "port",
    "edge",
    "graph_output",
    "recurrent_carry",
    "state_path",
    "task_data",
]
RetentionMode = Literal["stream", "window", "trajectory"]

_SELECTOR_PREFIX_KINDS: dict[str, SelectorKind] = {
    "port": "port",
    "edge": "edge",
    "graph_output": "graph_output",
    "recurrent_carry": "recurrent_carry",
    "path": "state_path",
    "state_path": "state_path",
    "task_data": "task_data",
}
_RETENTION_RANK: dict[RetentionMode, int] = {
    "stream": 0,
    "window": 1,
    "trajectory": 2,
}


@dataclass(frozen=True)
class SelectorRef:
    """Canonical selector reference used by retention and loss plans."""

    selector: str
    kind: SelectorKind
    node_id: Optional[str] = None
    port: Optional[str] = None
    edge_id: Optional[str] = None
    path: Optional[str] = None
    timing: Optional[str] = None
    source: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetentionPolicyPlan:
    """Merged retention requirement for a selector."""

    mode: RetentionMode = "trajectory"
    window_size: Optional[int] = None
    order: int = 0
    reasons: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetainedObservablePlan:
    """Executable retained-observable request."""

    id: str
    label: str
    selector: SelectorRef
    retention: RetentionPolicyPlan
    value_schema: Optional[Mapping[str, Any]] = None
    explicit: bool = False
    sources: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LossTermPlan:
    """Executable loss leaf over retained trace values."""

    key: str
    type: str
    label: str
    weight: float
    source: SelectorRef
    target: Optional[SelectorRef] = None
    target_value: Any = None
    norm: Literal["squared_l2", "l2", "l1", "huber"] = "squared_l2"
    time_agg: TimeAggregationSpec = field(default_factory=TimeAggregationSpec)
    time_mask: Optional[Any] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetentionPlan:
    """Lowered observable and loss plan for a training/eval run."""

    observables: tuple[RetainedObservablePlan, ...]
    loss_terms: tuple[LossTermPlan, ...] = ()

    @property
    def by_selector(self) -> dict[str, RetainedObservablePlan]:
        return {plan.selector.selector: plan for plan in self.observables}


class RetentionPlanError(ValueError):
    """Pathful lowering/validation error."""

    def __init__(self, message: str, *, path: str, selector: Optional[str] = None) -> None:
        super().__init__(message)
        self.path = path
        self.selector = selector

    def to_issue(self) -> dict[str, Any]:
        issue: dict[str, Any] = {"message": str(self), "path": self.path}
        if self.selector is not None:
            issue["selector"] = self.selector
        return issue


def normalize_selector_ref(
    value: str | Mapping[str, Any] | RetainedObservableTargetSpec | SelectorRef,
    *,
    path: str = "/selector",
    kind: Optional[str] = None,
) -> SelectorRef:
    """Normalize compact selector strings, Studio refs, or target dicts."""

    if isinstance(value, SelectorRef):
        return value
    if isinstance(value, RetainedObservableTargetSpec):
        raw = value.model_dump(mode="json", exclude_none=True)
    elif isinstance(value, str):
        raw = {"selector": value}
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raise RetentionPlanError(
            f"Selector reference must be a string or object, got {type(value).__name__}",
            path=path,
        )

    if isinstance(raw.get("target"), Mapping):
        target = dict(raw["target"])
        raw = {**target, **{k: v for k, v in raw.items() if k != "target"}}

    selector = raw.get("compact") or raw.get("selector")
    selector_kind = kind or raw.get("kind") or raw.get("namespace")
    if not isinstance(selector, str) or not selector:
        raw_path = raw.get("path")
        if selector_kind == "task_data" and isinstance(raw_path, str) and raw_path:
            selector = f"task_data:{raw_path.removeprefix('task_data:')}"
        elif selector_kind in {"state_path", "path"} and isinstance(raw_path, str) and raw_path:
            selector = raw_path if raw_path.startswith("path:") else f"path:{raw_path}"
        else:
            raise RetentionPlanError("Selector reference is missing a compact selector", path=path)

    inferred_kind = _infer_selector_kind(selector, path=path)
    if selector_kind in {"path", "state_path"}:
        normalized_kind: SelectorKind = "state_path"
    elif selector_kind in _SELECTOR_PREFIX_KINDS.values():
        normalized_kind = selector_kind  # type: ignore[assignment]
    elif isinstance(selector_kind, str) and selector_kind in _SELECTOR_PREFIX_KINDS:
        normalized_kind = _SELECTOR_PREFIX_KINDS[selector_kind]
    else:
        normalized_kind = inferred_kind

    if inferred_kind != normalized_kind and not (
        normalized_kind == "recurrent_carry" and inferred_kind == "edge"
    ):
        raise RetentionPlanError(
            f"Selector {selector!r} has kind {inferred_kind!r}, not {normalized_kind!r}",
            path=path,
            selector=selector,
        )

    parsed = _parse_selector(selector, normalized_kind, path)
    return SelectorRef(
        selector=selector,
        kind=normalized_kind,
        node_id=raw.get("node_id") if isinstance(raw.get("node_id"), str) else parsed.node_id,
        port=raw.get("port") if isinstance(raw.get("port"), str) else parsed.port,
        edge_id=raw.get("edge_id") if isinstance(raw.get("edge_id"), str) else parsed.edge_id,
        path=raw.get("path") if isinstance(raw.get("path"), str) else parsed.path,
        timing=raw.get("timing") if isinstance(raw.get("timing"), str) else parsed.timing,
        source={k: v for k, v in raw.items() if k not in {"selector", "compact"}},
    )


def lower_retention_plan(
    graph: GraphSpec,
    training: Optional[TrainingSpec | LossTermSpec] = None,
    *,
    task_spec: Optional[Mapping[str, Any] | Any] = None,
    artifact_selectors: Iterable[str | Mapping[str, Any] | SelectorRef] = (),
    analysis_input_requirements: Iterable[AnalysisInputRequirement | Mapping[str, Any]] = (),
) -> RetentionPlan:
    """Lower explicit observables, loss selectors, artifacts, and analysis inputs."""

    requirements: dict[str, _ObservableRequirement] = {}
    for index, observable in enumerate(graph.retained_observables or []):
        _merge_requirement(
            requirements,
            _requirement_from_observable(observable, f"/graph/retained_observables/{index}"),
        )

    loss_spec: Optional[LossTermSpec]
    if isinstance(training, TrainingSpec):
        loss_spec = training.loss
    else:
        loss_spec = training
    loss_terms: list[LossTermPlan] = []
    if loss_spec is not None:
        loss_terms.extend(_lower_loss_terms(loss_spec, "/loss", requirements, task_spec=task_spec))

    for index, selector_value in enumerate(artifact_selectors):
        selector = normalize_selector_ref(selector_value, path=f"/artifacts/{index}/selector")
        _merge_requirement(
            requirements,
            _ObservableRequirement(
                id=_stable_observable_id(selector.selector),
                label=selector.selector,
                selector=selector,
                retention=RetentionPolicyPlan(
                    mode="trajectory",
                    reasons=("artifact",),
                ),
                explicit=False,
                sources=(f"/artifacts/{index}",),
            ),
        )

    for index, requirement_value in enumerate(analysis_input_requirements):
        _merge_requirement(
            requirements,
            _requirement_from_analysis_input(
                requirement_value,
                f"/analysis/input_requirements/{index}",
            ),
        )

    observables = tuple(
        _requirement_to_plan(_validate_requirement(graph, requirement))
        for requirement in sorted(requirements.values(), key=lambda item: item.selector.selector)
    )
    return RetentionPlan(observables=observables, loss_terms=tuple(loss_terms))


def validate_retention_plan(
    graph: GraphSpec,
    training: Optional[TrainingSpec | LossTermSpec] = None,
    *,
    task_spec: Optional[Mapping[str, Any] | Any] = None,
    analysis_input_requirements: Iterable[AnalysisInputRequirement | Mapping[str, Any]] = (),
) -> list[RetentionPlanError]:
    """Return lowering errors without raising."""

    try:
        lower_retention_plan(
            graph,
            training,
            task_spec=task_spec,
            analysis_input_requirements=analysis_input_requirements,
        )
    except RetentionPlanError as exc:
        return [exc]
    return []


def evaluate_loss_plan(
    loss_terms: Iterable[LossTermPlan],
    trace: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Evaluate lowered loss terms over a selector-keyed trace map."""

    total = jnp.asarray(0.0)
    terms: dict[str, Any] = {}
    for term in loss_terms:
        value = evaluate_loss_term(term, trace)
        weighted = value * term.weight
        terms[term.key] = weighted
        total = total + weighted
    return total, terms


def evaluate_loss_term(term: LossTermPlan, trace: Mapping[str, Any]) -> Any:
    """Evaluate one lowered loss term over retained source/target arrays."""

    source = _lookup_trace(trace, term.source.selector, f"/loss/{term.key}/selector")
    if term.target is not None:
        target = _lookup_trace(trace, term.target.selector, f"/loss/{term.key}/target_selector")
    elif term.target_value is not None:
        target = term.target_value
    else:
        target = 0.0

    diff = jnp.asarray(source) - jnp.asarray(target)
    per_step = _norm_values(diff, term.norm)
    return _aggregate_time(
        per_step,
        term.time_agg,
        f"/loss/{term.key}/time_agg",
        time_mask=term.time_mask,
    )


def retention_plan_to_json(plan: RetentionPlan) -> dict[str, Any]:
    """Serialize a retention plan for manifests/artifacts."""

    return {
        "observables": [
            {
                "id": item.id,
                "label": item.label,
                "selector": item.selector.selector,
                "target": {
                    "kind": item.selector.kind,
                    "node_id": item.selector.node_id,
                    "port": item.selector.port,
                    "edge_id": item.selector.edge_id,
                    "path": item.selector.path,
                    "timing": item.selector.timing,
                },
                "retention": {
                    "mode": item.retention.mode,
                    "window_size": item.retention.window_size,
                    "order": item.retention.order,
                    "reasons": list(item.retention.reasons),
                    "metadata": dict(item.retention.metadata),
                },
                "value_schema": dict(item.value_schema) if item.value_schema else None,
                "explicit": item.explicit,
                "sources": list(item.sources),
                "metadata": dict(item.metadata),
            }
            for item in plan.observables
        ],
        "loss_terms": [
            {
                "key": item.key,
                "type": item.type,
                "label": item.label,
                "weight": item.weight,
                "selector": item.source.selector,
                "target_selector": item.target.selector if item.target is not None else None,
                "target_value": item.target_value,
                "norm": item.norm,
                "time_agg": item.time_agg.model_dump(mode="json", exclude_none=True),
                "time_mask": (
                    jnp.asarray(item.time_mask, dtype=bool).tolist()
                    if item.time_mask is not None
                    else None
                ),
                "metadata": dict(item.metadata),
            }
            for item in plan.loss_terms
        ],
    }


@dataclass
class _ObservableRequirement:
    id: str
    label: str
    selector: SelectorRef
    retention: RetentionPolicyPlan
    value_schema: Optional[Mapping[str, Any]] = None
    explicit: bool = False
    sources: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _requirement_from_observable(
    observable: RetainedObservableSpec,
    path: str,
) -> _ObservableRequirement:
    target = observable.target
    selector_value: Any = target if target is not None else observable.selector
    if selector_value is None:
        selector_value = observable.id
    selector = normalize_selector_ref(selector_value, path=f"{path}/target")
    return _ObservableRequirement(
        id=observable.id or _stable_observable_id(selector.selector),
        label=observable.label or selector.selector,
        selector=selector,
        retention=_policy_to_plan(
            observable.retention,
            reason="explicit",
            path=f"{path}/retention",
        ),
        value_schema=observable.value_schema,
        explicit=True,
        sources=(path,),
        metadata=observable.metadata,
    )


def _requirement_from_analysis_input(
    requirement_value: AnalysisInputRequirement | Mapping[str, Any],
    path: str,
) -> _ObservableRequirement:
    requirement = (
        requirement_value
        if isinstance(requirement_value, AnalysisInputRequirement)
        else AnalysisInputRequirement.model_validate(requirement_value)
    )
    selector_value: Any = requirement.target if requirement.target is not None else requirement.selector
    if selector_value is None:
        raise RetentionPlanError(
            "Analysis input requirement is missing a selector",
            path=f"{path}/selector",
        )
    selector = normalize_selector_ref(selector_value, path=f"{path}/target")
    consumer = requirement.consumer.model_dump(mode="json", exclude_none=True)
    return _ObservableRequirement(
        id=requirement.id or _stable_observable_id(selector.selector),
        label=requirement.label or selector.selector,
        selector=selector,
        retention=_policy_to_plan(
            requirement.retention,
            reason="analysis_input",
            path=f"{path}/retention",
        ),
        value_schema=requirement.value_schema,
        explicit=False,
        sources=(path,),
        metadata={
            "source": "analysis_input",
            "consumer": consumer,
            **requirement.metadata,
        },
    )


def _lower_loss_terms(
    term: LossTermSpec,
    path: str,
    requirements: dict[str, _ObservableRequirement],
    *,
    task_spec: Optional[Mapping[str, Any] | Any],
) -> list[LossTermPlan]:
    if term.children:
        lowered: list[LossTermPlan] = []
        for child_key, child in term.children.items():
            lowered.extend(
                _lower_loss_terms(
                    child,
                    f"{path}/children/{child_key}",
                    requirements,
                    task_spec=task_spec,
                )
            )
        return lowered

    if not term.selector:
        raise RetentionPlanError(
            f"Loss leaf {term.label!r} requires a selector",
            path=f"{path}/selector",
        )
    source = normalize_selector_ref(term.selector, path=f"{path}/selector")
    retention = _policy_to_plan(
        term.retention or RetentionPolicySpec(mode="trajectory"),
        reason=f"loss:{path}",
        path=f"{path}/retention",
    )
    _merge_requirement(
        requirements,
        _ObservableRequirement(
            id=_stable_observable_id(source.selector),
            label=term.label or source.selector,
            selector=source,
            retention=retention,
            sources=(path,),
        ),
    )

    target = None
    if term.target_selector:
        target = normalize_selector_ref(term.target_selector, path=f"{path}/target_selector")
        _merge_requirement(
            requirements,
            _ObservableRequirement(
                id=_stable_observable_id(target.selector),
                label=f"{term.label} target",
                selector=target,
                retention=retention,
                sources=(f"{path}/target_selector",),
            ),
        )
    if term.target_selector is not None and term.target_value is not None:
        raise RetentionPlanError(
            "Loss leaf cannot specify both target_selector and target_value",
            path=path,
            selector=term.selector,
        )
    time_agg = term.time_agg or TimeAggregationSpec(mode="all")
    time_mask = None
    metadata: dict[str, Any] = {}
    if time_agg.mode == "segment":
        try:
            timeline_mask = build_task_timeline_mask(
                task_spec,
                segment_name=time_agg.segment_name or "",
                n_steps=_task_n_steps(task_spec),
                path="/task_spec/timeline",
            )
        except TaskTimelineMaskError as exc:
            raise RetentionPlanError(
                str(exc),
                path=exc.path,
                selector=term.selector,
            ) from exc
        time_mask = timeline_mask.mask
        metadata["time_mask"] = {
            "segment_name": timeline_mask.segment_name,
            "epoch_ids": list(timeline_mask.epoch_ids),
            "n_steps": timeline_mask.n_steps,
            **dict(timeline_mask.metadata),
        }
        retention = _merge_retention(
            retention,
            RetentionPolicyPlan(
                mode="trajectory",
                reasons=("segment_time_aggregation",),
                metadata={"segment_name": time_agg.segment_name},
            ),
        )

    key = _loss_key(path)
    return [
        LossTermPlan(
            key=key,
            type=term.type,
            label=term.label,
            weight=term.weight,
            source=source,
            target=target,
            target_value=term.target_value,
            norm=term.norm or "squared_l2",
            time_agg=time_agg,
            time_mask=time_mask,
            metadata=metadata,
        )
    ]


def _task_n_steps(task_spec: Optional[Mapping[str, Any] | Any]) -> int:
    if task_spec is None:
        return 0
    if isinstance(task_spec, Mapping):
        task = task_spec
    elif hasattr(task_spec, "model_dump"):
        task = task_spec.model_dump(mode="json", exclude_none=True)
    else:
        task = {}
    timeline = task.get("timeline")
    if isinstance(timeline, Mapping):
        metadata = timeline.get("metadata")
        if isinstance(metadata, Mapping) and metadata.get("n_steps") is not None:
            return int(metadata["n_steps"])
    params = task.get("params")
    if isinstance(params, Mapping) and params.get("n_steps") is not None:
        return int(params["n_steps"])
    if isinstance(params, Mapping) and params.get("n_control_stages") is not None:
        return int(params["n_control_stages"])
    return 0


def _policy_to_plan(
    policy: RetentionPolicySpec,
    *,
    reason: str,
    path: str,
) -> RetentionPolicyPlan:
    if policy.mode == "window" and (policy.window_size is None or policy.window_size <= 0):
        raise RetentionPlanError(
            "Window retention requires a positive window_size",
            path=f"{path}/window_size",
        )
    order = int(policy.order or 0)
    return RetentionPolicyPlan(
        mode=policy.mode,
        window_size=policy.window_size,
        order=order,
        reasons=(policy.reason or reason,),
        metadata=policy.metadata,
    )


def _merge_requirement(
    requirements: dict[str, _ObservableRequirement],
    incoming: _ObservableRequirement,
) -> None:
    current = requirements.get(incoming.selector.selector)
    if current is None:
        requirements[incoming.selector.selector] = incoming
        return

    retention = _merge_retention(current.retention, incoming.retention)
    requirements[incoming.selector.selector] = _ObservableRequirement(
        id=current.id,
        label=current.label or incoming.label,
        selector=current.selector,
        retention=retention,
        value_schema=current.value_schema or incoming.value_schema,
        explicit=current.explicit or incoming.explicit,
        sources=tuple(dict.fromkeys((*current.sources, *incoming.sources))),
        metadata={**dict(incoming.metadata), **dict(current.metadata)},
    )


def _merge_retention(
    left: RetentionPolicyPlan,
    right: RetentionPolicyPlan,
) -> RetentionPolicyPlan:
    mode = left.mode if _RETENTION_RANK[left.mode] >= _RETENTION_RANK[right.mode] else right.mode
    window_size = max(
        (size for size in (left.window_size, right.window_size) if size is not None),
        default=None,
    )
    return RetentionPolicyPlan(
        mode=mode,
        window_size=window_size,
        order=max(left.order, right.order),
        reasons=tuple(dict.fromkeys((*left.reasons, *right.reasons))),
        metadata={**dict(left.metadata), **dict(right.metadata)},
    )


def _validate_requirement(graph: GraphSpec, requirement: _ObservableRequirement) -> _ObservableRequirement:
    selector = requirement.selector
    _validate_executable_retention(requirement)
    if selector.kind == "port":
        _validate_port(
            graph,
            selector.node_id,
            selector.port,
            selector.selector,
            path=_selector_validation_path(requirement, default="/graph"),
        )
    elif selector.kind == "edge":
        _validate_edge(
            graph,
            selector,
            path=_selector_validation_path(requirement, default="/graph/wires"),
        )
    elif selector.kind == "graph_output":
        output_name = selector.path or selector.selector.removeprefix("graph_output:")
        if output_name not in graph.output_bindings:
            raise RetentionPlanError(
                f"Unknown graph output selector {selector.selector!r}",
                path=_source_path(requirement),
                selector=selector.selector,
            )
    elif selector.kind == "recurrent_carry":
        _validate_edge(
            graph,
            selector,
            recurrent=True,
            path=_selector_validation_path(requirement, default="/graph/wires"),
        )
    elif selector.kind == "state_path":
        if not (selector.path or selector.selector.removeprefix("path:")):
            raise RetentionPlanError(
                f"State-path selector {selector.selector!r} has no path",
                path=_source_path(requirement),
                selector=selector.selector,
            )
    elif selector.kind == "task_data":
        if not (selector.path or selector.selector.removeprefix("task_data:")):
            raise RetentionPlanError(
                f"Task-data selector {selector.selector!r} has no path",
                path=_source_path(requirement),
                selector=selector.selector,
            )
    return requirement


def _validate_executable_retention(requirement: _ObservableRequirement) -> None:
    mode = requirement.retention.mode
    if mode == "trajectory":
        return
    raise RetentionPlanError(
        (
            f"{mode!r} retention for selector {requirement.selector.selector!r} is not "
            "supported by the current graph worker; request trajectory retention instead"
        ),
        path=f"{_source_path(requirement)}/retention/mode",
        selector=requirement.selector.selector,
    )


def _requirement_to_plan(requirement: _ObservableRequirement) -> RetainedObservablePlan:
    return RetainedObservablePlan(
        id=requirement.id,
        label=requirement.label,
        selector=requirement.selector,
        retention=requirement.retention,
        value_schema=requirement.value_schema,
        explicit=requirement.explicit,
        sources=requirement.sources,
        metadata=requirement.metadata,
    )


def _validate_port(
    graph: GraphSpec,
    node_id: Optional[str],
    port: Optional[str],
    selector: str,
    *,
    path: str = "/graph",
) -> None:
    if not node_id or node_id not in graph.nodes:
        raise RetentionPlanError(
            f"Unknown node in selector {selector!r}",
            path=path,
            selector=selector,
        )
    node = graph.nodes[node_id]
    known_ports = set(node.input_ports) | set(node.output_ports)
    if not port or port not in known_ports:
        raise RetentionPlanError(
            f"Unknown port in selector {selector!r}",
            path=f"/graph/nodes/{node_id}",
            selector=selector,
        )


def _validate_edge(
    graph: GraphSpec,
    selector: SelectorRef,
    *,
    recurrent: bool = False,
    path: str = "/graph/wires",
) -> None:
    source_node, source_port, target_node, target_port = _edge_parts(selector)
    for wire in graph.wires:
        if (
            wire.source_node == source_node
            and wire.source_port == source_port
            and wire.target_node == target_node
            and wire.target_port == target_port
        ):
            if recurrent and wire.temporality != "recurrent":
                raise RetentionPlanError(
                    f"Selector {selector.selector!r} targets an instant edge, not a recurrent carry",
                    path=path,
                    selector=selector.selector,
                )
            return
    raise RetentionPlanError(
        f"Unknown edge selector {selector.selector!r}",
        path=path,
        selector=selector.selector,
    )


def _edge_parts(selector: SelectorRef) -> tuple[str, str, str, str]:
    edge = selector.edge_id or selector.selector
    if edge.startswith(("edge:", "recurrent_carry:")):
        edge = edge.split(":", 1)[1]
    source, separator, target = edge.partition("->")
    if not separator:
        raise RetentionPlanError(
            f"Edge selector {selector.selector!r} must use source.port->target.port",
            path="/selector",
            selector=selector.selector,
        )
    source_node, source_port = _node_port(source, selector.selector)
    target_node, target_port = _node_port(target, selector.selector)
    return source_node, source_port, target_node, target_port


def _parse_selector(selector: str, kind: SelectorKind, path: str) -> SelectorRef:
    if kind == "port":
        node_id, port = _node_port(selector.removeprefix("port:"), selector)
        return SelectorRef(selector=selector, kind=kind, node_id=node_id, port=port)
    if kind in {"edge", "recurrent_carry"}:
        edge_id = selector
        source_node, source_port, _target_node, _target_port = _edge_parts(
            SelectorRef(selector=selector, kind=kind, edge_id=edge_id)
        )
        return SelectorRef(
            selector=selector,
            kind=kind,
            node_id=source_node,
            port=source_port,
            edge_id=edge_id,
            timing="step",
        )
    if kind == "graph_output":
        return SelectorRef(
            selector=selector,
            kind=kind,
            path=selector.removeprefix("graph_output:"),
        )
    if kind == "state_path":
        normalized = selector.removeprefix("path:").removeprefix("state_path:")
        return SelectorRef(selector=selector, kind=kind, path=normalized)
    if kind == "task_data":
        return SelectorRef(selector=selector, kind=kind, path=selector.removeprefix("task_data:"))
    raise RetentionPlanError(f"Unsupported selector kind {kind!r}", path=path, selector=selector)


def _infer_selector_kind(selector: str, *, path: str) -> SelectorKind:
    prefix, separator, _rest = selector.partition(":")
    if not separator:
        raise RetentionPlanError(
            f"Selector {selector!r} must include a namespace prefix",
            path=path,
            selector=selector,
        )
    kind = _SELECTOR_PREFIX_KINDS.get(prefix)
    if kind is None:
        raise RetentionPlanError(
            f"Unsupported selector namespace {prefix!r}",
            path=path,
            selector=selector,
        )
    return kind


def _node_port(value: str, selector: str) -> tuple[str, str]:
    node, separator, port = value.partition(".")
    if not node or not separator or not port:
        raise RetentionPlanError(
            f"Selector {selector!r} must include node.port",
            path="/selector",
            selector=selector,
        )
    return node, port


def _stable_observable_id(selector: str) -> str:
    safe = (
        selector.replace(":", "_")
        .replace(".", "_")
        .replace("->", "__")
        .replace("/", "_")
    )
    return f"observable:{safe}"


def _loss_key(path: str) -> str:
    key = path.strip("/").replace("/", ".")
    return key or "loss"


def _source_path(requirement: _ObservableRequirement) -> str:
    return requirement.sources[0] if requirement.sources else "/retention"


def _selector_validation_path(requirement: _ObservableRequirement, *, default: str) -> str:
    if requirement.metadata.get("source") == "analysis_input":
        return _source_path(requirement)
    return default


def _lookup_trace(trace: Mapping[str, Any], selector: str, path: str) -> Any:
    if selector not in trace:
        raise RetentionPlanError(
            f"Trace is missing retained selector {selector!r}",
            path=path,
            selector=selector,
        )
    return trace[selector]


def _norm_values(values: Any, norm: str) -> Any:
    arr = jnp.asarray(values)
    if norm == "squared_l2":
        per_element = jnp.square(arr)
    elif norm == "l2":
        per_element = jnp.square(arr)
    elif norm == "l1":
        per_element = jnp.abs(arr)
    elif norm == "huber":
        abs_arr = jnp.abs(arr)
        per_element = jnp.where(abs_arr <= 1.0, 0.5 * jnp.square(arr), abs_arr - 0.5)
    else:
        raise RetentionPlanError(
            f"Unsupported loss norm {norm!r}",
            path="/loss/norm",
        )

    if arr.ndim <= 1:
        reduced = per_element
    else:
        reduced = jnp.sum(per_element, axis=tuple(range(1, arr.ndim)))
    if norm == "l2":
        return jnp.sqrt(reduced)
    return reduced


def _aggregate_time(
    values: Any,
    time_agg: TimeAggregationSpec,
    path: str,
    *,
    time_mask: Optional[Any] = None,
) -> Any:
    arr = jnp.asarray(values)
    mode = time_agg.mode
    if mode in {"all", "mean"}:
        return jnp.mean(arr)
    if mode == "sum":
        return jnp.sum(arr)
    if mode == "final":
        return arr[-1] if arr.ndim > 0 else arr
    if mode == "range":
        if time_agg.start is None or time_agg.end is None:
            raise RetentionPlanError(
                "Range time aggregation requires start and end",
                path=path,
            )
        if time_agg.start < 0 or time_agg.end <= time_agg.start:
            raise RetentionPlanError(
                "Range time aggregation must satisfy 0 <= start < end",
                path=path,
            )
        selected = arr[time_agg.start : time_agg.end]
        if selected.shape[0] == 0:
            raise RetentionPlanError(
                "Range time aggregation selected no timesteps",
                path=path,
            )
        return jnp.mean(selected)
    if mode == "custom":
        if not time_agg.time_idxs:
            raise RetentionPlanError("Custom time aggregation requires time_idxs", path=path)
        return jnp.mean(arr[jnp.asarray(time_agg.time_idxs)])
    if mode == "segment":
        if time_mask is None:
            raise RetentionPlanError(
                "Segment time aggregation requires task timeline masks",
                path=path,
            )
        try:
            mask = align_time_mask(time_mask, arr, path=path)
        except TaskTimelineMaskError as exc:
            raise RetentionPlanError(str(exc), path=exc.path) from exc
        return jnp.mean(arr[mask])
    raise RetentionPlanError(f"Unsupported time aggregation {mode!r}", path=path)
