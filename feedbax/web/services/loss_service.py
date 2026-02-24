"""Service for converting loss specifications to feedbax loss objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional

from feedbax.web.models.graph import GraphSpec, BarnacleSpec
from feedbax.web.models.training import LossTermSpec, TimeAggregationSpec


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
    "l2": "feedbax.loss.norms.l2",
    "l1": "feedbax.loss.norms.l1",
    "huber": "feedbax.loss.norms.huber",
}


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


# Singleton instance
loss_service = LossService()
