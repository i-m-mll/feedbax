"""Shared Python/Zod cross-field refinement declarations for Studio contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CrossFieldRefinement:
    """One stable rule with equivalent Python and generated TypeScript predicates."""

    rule_id: str
    message: str
    python_invalid: Callable[[Any], bool] | None
    typescript_invalid: str


CROSS_FIELD_REFINEMENTS: dict[str, tuple[CrossFieldRefinement, ...]] = {
    "AdditiveGraphChannelTargetSpec": (
        CrossFieldRefinement(
            "additive-target.edge-source",
            "edge additive channel adapters require source_node and source_port",
            lambda value: value.kind == "edge"
            and (value.source_node is None or value.source_port is None),
            "value.kind === 'edge' && (value.source_node == null || value.source_port == null)",
        ),
        CrossFieldRefinement(
            "additive-target.input-source",
            "input additive channel adapters must not set source_node/source_port",
            lambda value: value.kind == "input"
            and (value.source_node is not None or value.source_port is not None),
            "value.kind === 'input' && (value.source_node != null || value.source_port != null)",
        ),
    ),
    "StudioValueEnumerableSpec": (
        CrossFieldRefinement(
            "value-enumerable.list-values",
            "sweep list enumerables require at least one value",
            lambda value: value.form == "list" and not value.values,
            "value.form === 'list' && (!value.values || value.values.length === 0)",
        ),
        CrossFieldRefinement(
            "value-enumerable.range-fields",
            "sweep range enumerables require start, stop, and count",
            lambda value: value.form == "range"
            and (value.start is None or value.stop is None or value.count is None),
            "value.form === 'range' && (value.start == null || value.stop == null || value.count == null)",
        ),
        CrossFieldRefinement(
            "value-enumerable.range-count",
            "sweep range count must be positive",
            lambda value: value.form == "range" and value.count is not None and value.count < 1,
            "value.form === 'range' && value.count != null && value.count < 1",
        ),
        CrossFieldRefinement(
            "value-enumerable.log-range",
            "log sweep ranges require positive start and stop",
            lambda value: value.form == "range"
            and value.scale == "log"
            and value.start is not None
            and value.stop is not None
            and (value.start <= 0 or value.stop <= 0),
            "value.form === 'range' && value.scale === 'log' && value.start != null && value.stop != null && (value.start <= 0 || value.stop <= 0)",
        ),
        CrossFieldRefinement(
            "value-enumerable.sampler-fields",
            "sweep sampler enumerables require sampler and n",
            lambda value: value.form == "sampler" and (value.sampler is None or value.n is None),
            "value.form === 'sampler' && (value.sampler == null || value.n == null)",
        ),
        CrossFieldRefinement(
            "value-enumerable.sampler-count",
            "sweep sampler n must be positive",
            lambda value: value.form == "sampler" and value.n is not None and value.n < 1,
            "value.form === 'sampler' && value.n != null && value.n < 1",
        ),
    ),
    "StudioValueVariationSpec": (
        CrossFieldRefinement(
            "value-variation.sweep-enumerable",
            "sweep variation requires an enumerable list, range, or sampler+n",
            lambda value: value.scope == "sweep" and value.enumerable is None,
            "value.scope === 'sweep' && value.enumerable == null",
        ),
        CrossFieldRefinement(
            "value-variation.non-sweep-enumerable",
            "enumerable payloads are only valid for sweep variation",
            lambda value: value.scope != "sweep" and value.enumerable is not None,
            "value.scope !== 'sweep' && value.enumerable != null",
        ),
        CrossFieldRefinement(
            "value-variation.run-policy",
            "run variation samples once and shares within the run",
            lambda value: value.scope == "run"
            and value.stochastic_policy not in (None, "shared_per_run"),
            "value.scope === 'run' && value.stochastic_policy != null && value.stochastic_policy !== 'shared_per_run'",
        ),
        CrossFieldRefinement(
            "value-variation.replicate-policy",
            "replicate variation resamples per replicate",
            lambda value: value.scope == "replicate"
            and value.stochastic_policy not in (None, "resample_per_replicate"),
            "value.scope === 'replicate' && value.stochastic_policy != null && value.stochastic_policy !== 'resample_per_replicate'",
        ),
    ),
    "StudioValueSpec": (
        CrossFieldRefinement(
            "value.schema-version",
            "unsupported StudioValueSpec schema version",
            lambda value: value.schema_version != "feedbax.spec.studio.value.v2",
            "value.schema_version !== 'feedbax.spec.studio.value.v2'",
        ),
        CrossFieldRefinement(
            "value.mode-alias",
            "StudioValueSpec mode must match value_form compatibility alias",
            lambda value: value.mode
            != ("constant" if value.value_form == "literal" else value.value_form),
            "value.mode !== (value.value_form === 'literal' ? 'constant' : value.value_form)",
        ),
        CrossFieldRefinement(
            "value.literal-sampling-scope",
            "literal fixed values must not carry a sampling_scope",
            lambda value: value.value_form == "literal"
            and value.variation.scope != "sweep"
            and value.sampling_scope not in (None, "fixed"),
            "value.value_form === 'literal' && value.variation.scope !== 'sweep' && value.sampling_scope != null && value.sampling_scope !== 'fixed'",
        ),
        CrossFieldRefinement(
            "value.distribution-run-policy",
            "run distribution specs sample once and share",
            lambda value: value.value_form == "distribution"
            and value.variation.scope == "run"
            and value.variation.stochastic_policy not in (None, "shared_per_run"),
            "value.value_form === 'distribution' && value.variation.scope === 'run' && value.variation.stochastic_policy != null && value.variation.stochastic_policy !== 'shared_per_run'",
        ),
        CrossFieldRefinement(
            "value.distribution-replicate-policy",
            "replicate distribution specs resample per replicate",
            lambda value: value.value_form == "distribution"
            and value.variation.scope == "replicate"
            and value.variation.stochastic_policy not in (None, "resample_per_replicate"),
            "value.value_form === 'distribution' && value.variation.scope === 'replicate' && value.variation.stochastic_policy != null && value.variation.stochastic_policy !== 'resample_per_replicate'",
        ),
    ),
    "StudioEpochValueSpec": (
        CrossFieldRefinement(
            "epoch-value.finite",
            "timeline epoch values must contain only finite numbers",
            None,
            "containsNonFiniteNumber(value.value_spec)",
        ),
        CrossFieldRefinement(
            "epoch-value.mode",
            "timeline epoch values support constant, function, or distribution modes",
            None,
            "!['constant', 'function', 'distribution'].includes(value.value_spec.mode)",
        ),
        CrossFieldRefinement(
            "epoch-value.required-payload",
            "timeline epoch value is missing its required mode payload",
            None,
            "(value.value_spec.mode === 'constant' && value.value_spec.value == null) || (value.value_spec.mode === 'function' && !value.value_spec.function_id) || (value.value_spec.mode === 'distribution' && (value.value_spec.distribution == null || typeof value.value_spec.distribution !== 'object' || value.value_spec.distribution.parameters == null || typeof value.value_spec.distribution.parameters !== 'object'))",
        ),
        CrossFieldRefinement(
            "epoch-value.distribution",
            "timeline distribution must be uniform min/max or normal mean/std with valid bounds",
            None,
            "value.value_spec.mode === 'distribution' && (() => { const distribution = value.value_spec.distribution as Record<string, unknown>; const parameters = distribution?.parameters as Record<string, unknown>; const first = parameters?.[distribution?.family === 'uniform' ? 'min' : 'mean']; const second = parameters?.[distribution?.family === 'uniform' ? 'max' : 'std']; if (distribution?.family !== 'uniform' && distribution?.family !== 'normal') return true; if (typeof first !== 'number' || !Number.isFinite(first) || typeof second !== 'number' || !Number.isFinite(second)) return true; return distribution.family === 'uniform' ? second < first : second < 0; })()",
        ),
    ),
    "StudioTaskTimelineSpec": (
        CrossFieldRefinement(
            "task-timeline.identity",
            "unsupported Studio task timeline schema identity",
            lambda value: value.schema_id != "feedbax.spec.studio.task_timeline"
            or value.schema_version != "feedbax.spec.studio.task_timeline.v2",
            "value.schema_id !== 'feedbax.spec.studio.task_timeline' || value.schema_version !== 'feedbax.spec.studio.task_timeline.v2'",
        ),
        CrossFieldRefinement(
            "task-timeline.epochs",
            "timeline epochs must have unique ids and contiguous indexes",
            None,
            "hasDuplicate((value.epochs ?? []).map((epoch) => epoch.id)) || (value.epochs ?? []).map((epoch) => epoch.index).slice().sort((a, b) => a - b).some((index, position) => index !== position)",
        ),
        CrossFieldRefinement(
            "task-timeline.targets",
            "timeline signal target ids must be unique",
            None,
            "hasDuplicate((value.signals ?? []).map((signal) => signal.task_data_id ?? signal.id))",
        ),
        CrossFieldRefinement(
            "task-timeline.epoch-values",
            "timeline epoch values must name known targets and epochs without overlap",
            None,
            "(value.epoch_value_specs ?? []).some((entry) => !(value.signals ?? []).some((signal) => (signal.task_data_id ?? signal.id) === entry.target_id) || !(value.epochs ?? []).some((epoch) => epoch.id === entry.epoch_id)) || hasDuplicate((value.epoch_value_specs ?? []).map((entry) => `${entry.target_id}\\u0000${entry.epoch_id}`))",
        ),
        CrossFieldRefinement(
            "task-timeline.canonical-order",
            "timeline epoch values must be ordered by target_id then epoch index",
            None,
            "(value.epoch_value_specs ?? []).some((entry, index, entries) => index > 0 && (entries[index - 1].target_id > entry.target_id || (entries[index - 1].target_id === entry.target_id && (value.epochs ?? []).findIndex((epoch) => epoch.id === entries[index - 1].epoch_id) > (value.epochs ?? []).findIndex((epoch) => epoch.id === entry.epoch_id))))",
        ),
    ),
    "StudioScenarioSpec": (
        CrossFieldRefinement(
            "studio-scenario.task-spec",
            "Studio scenario task_spec must satisfy the generated TaskSpec contract",
            None,
            "value.task_spec != null && !TaskSpecSchema.safeParse(value.task_spec).success",
        ),
    ),
    "WorkspaceDocument": (
        CrossFieldRefinement(
            "workspace-document.semantic-root",
            "WorkspaceDocument semantic_root must target /graph",
            lambda value: value.semantic_root.authored_path != "/graph",
            "value.semantic_root.authored_path !== '/graph'",
        ),
        CrossFieldRefinement(
            "workspace-document.anchor-revision",
            "WorkspaceDocument semantic anchors must target the semantic root revision",
            lambda value: any(
                anchor.semantic_document_sha256 != value.semantic_root.semantic_document_sha256
                for anchor in value.semantic_anchors.values()
            ),
            "Object.values(value.semantic_anchors).some((anchor) => anchor.semantic_document_sha256 !== value.semantic_root.semantic_document_sha256)",
        ),
    ),
}


def validate_cross_field_refinements(model_name: str, value: Any) -> None:
    """Run the registered Python side of one generated-contract rule set."""

    for refinement in CROSS_FIELD_REFINEMENTS[model_name]:
        if refinement.python_invalid is not None and refinement.python_invalid(value):
            raise ValueError(refinement.message)


def _typescript_refinement(
    rule_id: str,
    message: str,
    typescript_invalid: str,
) -> CrossFieldRefinement:
    return CrossFieldRefinement(rule_id, message, None, typescript_invalid)


CROSS_FIELD_REFINEMENTS.update(
    {
        "SparseCooArrayValueSpec": (
            _typescript_refinement(
                "array.sparse-entries",
                "sparse COO entries must match shape, coordinates, and dtype",
                "arrayScalarInvalid(value.fill, value.dtype, value.nonfinite) || value.entries.some((entry) => entry.coordinate.length !== value.shape.length || entry.coordinate.some((coordinate, axis) => coordinate >= value.shape[axis]) || arrayScalarInvalid(entry.value, value.dtype, value.nonfinite)) || new Set(value.entries.map((entry) => JSON.stringify(entry.coordinate))).size !== value.entries.length",
            ),
        ),
        "ConstantArrayValueSpec": (
            _typescript_refinement(
                "array.constant-value",
                "constant array value must match dtype and non-finite policy",
                "arrayScalarInvalid(value.value, value.dtype, value.nonfinite)",
            ),
        ),
        "AcausalConnectionSpec": (
            _typescript_refinement(
                "acausal-connection.distinct-endpoints",
                "AcausalConnectionSpec endpoints must be distinct",
                "value.a[0] === value.b[0] && value.a[1] === value.b[1]",
            ),
        ),
        "AcausalGraphSpec": (
            _typescript_refinement(
                "acausal-graph.no-array-params",
                "AcausalGraphSpec does not support component-param array value schemas",
                "Object.values(value.nodes).some((node) => Object.values(node.params).some(containsArrayValueEnvelope))",
            ),
        ),
        "GraphSpec": (
            _typescript_refinement(
                "graph.subgraph-family",
                "GraphSpec subgraphs must use a supported graph schema identity",
                "value.subgraphs != null && Object.values(value.subgraphs).some((subgraph) => subgraph.schema_id !== 'feedbax.spec.graph' && subgraph.schema_id !== 'feedbax.spec.acausal_graph')",
            ),
        ),
        "ComponentSpec": (
            _typescript_refinement(
                "component.typed-param-envelopes",
                "typed component parameter envelopes must satisfy their declared schema",
                "Object.values(value.params).some(invalidTypedParamEnvelope)",
            ),
        ),
        "AnalysisDataProductRequirement": (
            _typescript_refinement(
                "analysis-product.schema-id",
                "unsupported AnalysisDataProductRequirement schema identity",
                "value.schema_id !== 'feedbax.spec.analysis_data_product_requirement' || value.schema_version !== 'feedbax.spec.analysis_data_product_requirement.v1'",
            ),
            _typescript_refinement(
                "analysis-product.required-names",
                "AnalysisDataProductRequirement role and product_schema_id must not be empty",
                "value.role.trim().length === 0 || value.product_schema_id.trim().length === 0",
            ),
        ),
        "DynamicPortPolicy": (
            _typescript_refinement(
                "dynamic-port.generated-template",
                "generated_name_template must contain only the {index} replacement field",
                "!validGeneratedPortTemplate(value.generated_name_template)",
            ),
            _typescript_refinement(
                "dynamic-port.fixed-names",
                "fixed dynamic port names must be unique and non-empty",
                "hasDuplicate(value.fixed_input_ports) || hasDuplicate(value.fixed_output_ports) || value.fixed_input_ports.some((name) => name.length === 0) || value.fixed_output_ports.some((name) => name.length === 0)",
            ),
        ),
        "RepresentationStateAnchorSelectorBinding": (
            _typescript_refinement(
                "representation.anchor-subpath",
                "representation anchor_subpath requires an object selector namespace",
                "value.anchor_subpath != null && !['mechanics_object', 'biomechanics_object', 'task_object'].includes(value.selector.namespace)",
            ),
        ),
        "RepresentationPlanarChainSpec": (
            _typescript_refinement(
                "representation.planar-chain",
                "planar-chain frame ids and reference pose must be coherent",
                "hasDuplicate(value.frame_ids) || value.frame_ids[0] !== 'world' || (value.reference_pose != null && value.reference_pose.values.length !== value.frame_ids.length - 1)",
            ),
        ),
        "RepresentationMusclePathGeometrySpec": (
            _typescript_refinement(
                "representation.unique-muscle-paths",
                "muscle path ids must be unique",
                "hasDuplicate(value.paths.map((path) => path.id))",
            ),
        ),
        "RepresentationFrameProvider": (
            _typescript_refinement(
                "representation.frame-provider",
                "frame provider kind requires its matching reference",
                "(value.kind === 'fixed' && !value.frame) || (value.kind === 'from_input_port' && !value.input_port) || (value.kind === 'from_representation_element' && !value.element_id) || (value.kind === 'registered_renderer' && !value.renderer_id)",
            ),
        ),
        "RepresentationStyleSpec": (
            _typescript_refinement(
                "representation.style-source",
                "representation style channels require value or binding",
                "value.value == null && value.binding == null",
            ),
        ),
        "RepresentationElementSpec": (
            _typescript_refinement(
                "representation.element-capability",
                "representation element capabilities must match their archetype",
                "(value.archetype === 'registered_renderer' && !value.renderer_id) || (value.planar_chain != null && value.archetype !== 'planar_chain')",
            ),
        ),
        "RepresentationSpec": (
            _typescript_refinement(
                "representation.identity-graph",
                "representation ids and anchor references must be valid",
                "hasDuplicate(value.anchors.map((anchor) => anchor.id)) || hasDuplicate(value.elements.map((element) => element.id)) || (value.reachability != null && !value.anchors.some((anchor) => anchor.id === value.reachability?.origin_anchor)) || value.elements.some((element) => element.anchors.some((anchor) => !value.anchors.some((candidate) => candidate.id === anchor)))",
            ),
        ),
        "DomainMeta": (
            _typescript_refinement(
                "domain-meta.identifiers",
                "domain ids must use the feedbax.domain namespace with non-empty parts",
                "!validDomainId(value.id) || value.nestable_domains.some((domainId) => !validDomainId(domainId))",
            ),
        ),
        "DomainCompileReport": (
            _typescript_refinement(
                "domain-report.status",
                "domain report status must match diagnostic severities",
                "(value.status === 'ok' && value.diagnostics.some((diagnostic) => diagnostic.severity === 'error' || diagnostic.severity === 'warning')) || (value.status === 'ok_with_warnings' && value.diagnostics.some((diagnostic) => diagnostic.severity === 'error')) || (value.status === 'error' && !value.diagnostics.some((diagnostic) => diagnostic.severity === 'error'))",
            ),
        ),
        "WorkspaceReplayTrialIdentity": (
            _typescript_refinement(
                "workspace-replay.identity-source",
                "trial identity source requires stable_id unless index_fallback",
                "!value.stable_id && value.source !== 'index_fallback'",
            ),
        ),
        "WorkspaceReplaySampleAxis": (
            _typescript_refinement(
                "workspace-replay.axis-length",
                "time axis values length must equal length",
                "value.values != null && value.values.length !== value.length",
            ),
        ),
        "WorkspaceReplayTrack": (
            _typescript_refinement(
                "workspace-replay.track-dim",
                "track samples must match dim",
                "value.samples.some((sample) => sample.length !== value.dim)",
            ),
        ),
        "WorkspaceReplayTrial": (
            _typescript_refinement(
                "workspace-replay.trial-length",
                "track and overlay sample counts must match the time axis",
                "value.tracks.some((track) => track.samples.length !== value.time.length) || value.overlays.some((overlay) => overlay.samples.length !== value.time.length)",
            ),
        ),
        "WorkspaceReplayProduct": (
            _typescript_refinement(
                "workspace-replay.source-mode",
                "workspace replay source mode must match imported artifact and warnings",
                "(value.source_mode === 'resolved_scene' && value.imported_artifact != null) || (value.source_mode === 'imported_artifact' && (value.imported_artifact == null || value.warnings.length === 0))",
            ),
        ),
        "BatchScheduleOriginSpec": (
            _typescript_refinement(
                "batch-schedule.origin",
                "batch is required only for an absolute schedule origin",
                "(value.kind === 'absolute' && value.batch == null) || (value.kind !== 'absolute' && value.batch != null)",
            ),
        ),
        "LrScheduleSpec": (
            _typescript_refinement(
                "lr-schedule.identity",
                "unsupported LrScheduleSpec schema identity",
                "value.schema_id !== 'feedbax.spec.training.lr_schedule' || value.schema_version !== 'feedbax.spec.training.lr_schedule.v2'",
            ),
            _typescript_refinement(
                "lr-schedule.shape",
                "learning-rate schedule fields must match its kind",
                "(value.kind !== 'constant' && value.total_steps == null) || (value.kind === 'warmup_cosine' && (value.constant_lr_iterations < 1 || (value.total_steps != null && value.constant_lr_iterations >= value.total_steps))) || (value.kind === 'delayed_cosine' && value.total_steps != null && value.constant_lr_iterations >= value.total_steps)",
            ),
        ),
        "SelectionSpec": (
            _typescript_refinement(
                "selection.identity",
                "unsupported SelectionSpec schema identity",
                "value.schema_id !== 'feedbax.spec.selection' || value.schema_version !== 'feedbax.spec.selection.v2'",
            ),
            _typescript_refinement(
                "selection.mode",
                "selection fields must match selection mode",
                "(value.mode === 'explicit' && (value.query != null || value.frozen_refs.length > 0 || value.frozen_at != null)) || (value.mode === 'query' && (value.query == null || value.ids.length > 0 || value.frozen_refs.length > 0 || value.frozen_at != null)) || (value.mode === 'frozen' && (value.query == null || value.frozen_refs.length === 0 || value.ids.length > 0))",
            ),
        ),
    }
)


REGISTERED_CONSTRAINT_VALIDATORS: dict[str, frozenset[str]] = {
    "SparseCooArrayValueSpec": frozenset({"validate_entries"}),
    "ConstantArrayValueSpec": frozenset({"validate_value"}),
    "AcausalConnectionSpec": frozenset({"validate_and_canonicalize"}),
    "AcausalGraphSpec": frozenset(
        {"validate_subgraph_family", "reject_component_param_array_values"}
    ),
    "AdditiveGraphChannelTargetSpec": frozenset({"validate_target"}),
    "AnalysisDataProductRequirement": frozenset({"validate_schema_identity"}),
    "GraphSpec": frozenset({"validate_subgraph_schema_family"}),
    "StudioValueEnumerableSpec": frozenset({"validate_form_payload"}),
    "StudioValueVariationSpec": frozenset({"validate_scope_payload"}),
    "StudioValueSpec": frozenset({"validate_value_variation"}),
    "StudioEpochValueSpec": frozenset({"validate_safe_value"}),
    "StudioTaskTimelineSpec": frozenset({"validate_epoch_values"}),
    "StudioScenarioSpec": frozenset({"admit_task_spec"}),
    "WorkspaceDocument": frozenset({"validate_semantic_anchors"}),
    "DynamicPortPolicy": frozenset({"validate_port_namespace"}),
    "RepresentationStateAnchorSelectorBinding": frozenset({"validate_anchor_subpath_namespace"}),
    "RepresentationPlanarChainSpec": frozenset({"validate_frame_ids"}),
    "RepresentationMusclePathGeometrySpec": frozenset({"validate_unique_ids"}),
    "RepresentationFrameProvider": frozenset({"validate_provider"}),
    "RepresentationStyleSpec": frozenset({"validate_style_source"}),
    "RepresentationElementSpec": frozenset({"validate_registered_renderer"}),
    "RepresentationSpec": frozenset({"canonicalize_and_validate"}),
    "DomainMeta": frozenset({"_validate_id", "_validate_nestable_domains"}),
    "DomainCompileReport": frozenset({"validate_status_matches_diagnostics"}),
    "WorkspaceReplayTrialIdentity": frozenset({"validate_identity_source"}),
    "WorkspaceReplaySampleAxis": frozenset({"validate_values_length"}),
    "WorkspaceReplayTrack": frozenset({"validate_samples_shape"}),
    "WorkspaceReplayTrial": frozenset({"validate_trial_lengths"}),
    "WorkspaceReplayProduct": frozenset({"validate_source_mode"}),
    "BatchScheduleOriginSpec": frozenset({"_validate_origin"}),
    "LrScheduleSpec": frozenset({"_validate_schedule_shape"}),
    "SelectionSpec": frozenset({"_validate_selection_spec"}),
}


__all__ = [
    "CROSS_FIELD_REFINEMENTS",
    "CrossFieldRefinement",
    "REGISTERED_CONSTRAINT_VALIDATORS",
    "validate_cross_field_refinements",
]
