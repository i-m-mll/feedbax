import {
  selectorSchemaMetadata,
  selectorValueSchema,
} from '@/features/scenario/selectors';
import {
  graphPortEntityId,
  objectiveEntityId,
} from '@/features/scenario/entities';
import type { ComponentSpec, WireSpec } from '@/types/graph';
import {
  LOSS_TERM_SPEC_SCHEMA_ID,
  LOSS_TERM_SPEC_SCHEMA_VERSION,
  type LossTermSpec,
  type NormFunction,
  type TimeAggregationSpec,
} from '@/types/training';
import type {
  StudioObjectiveSpec,
  StudioObjectiveTermSpec,
  StudioScenarioEntity,
  StudioScenarioEntityRegistry,
  StudioSelectorRef,
} from '@/types/workspace';

const OBJECTIVE_SCHEMA_VERSION = 'feedbax.studio.objective.v1';
const VALID_NORMS = new Set<NormFunction>(['squared_l2', 'l2', 'l1', 'huber']);

export const OBJECTIVE_PENALTY_OPTIONS: Array<{ value: NormFunction; label: string }> = [
  { value: 'squared_l2', label: 'Squared L2' },
  { value: 'l2', label: 'L2' },
  { value: 'l1', label: 'L1' },
  { value: 'huber', label: 'Huber' },
];

export const OBJECTIVE_TEMPORAL_MODE_OPTIONS: Array<{
  value: TimeAggregationSpec['mode'];
  label: string;
}> = [
  { value: 'all', label: 'Full trajectory' },
  { value: 'mean', label: 'Mean' },
  { value: 'sum', label: 'Sum' },
  { value: 'final', label: 'Final step' },
];

export const OBJECTIVE_DISCOUNT_OPTIONS: Array<{
  value: NonNullable<TimeAggregationSpec['discount']>;
  label: string;
}> = [
  { value: 'none', label: 'None' },
];

export function isStudioObjectiveSpec(value: unknown): value is StudioObjectiveSpec {
  return Boolean(value && typeof value === 'object' && Array.isArray((value as StudioObjectiveSpec).terms));
}

export function emptyObjectiveSpec(): StudioObjectiveSpec {
  return {
    schema_version: OBJECTIVE_SCHEMA_VERSION,
    terms: [],
    legacy_loss_spec: null,
    metadata: {},
  };
}

export function ensureObjectiveSpec(value: unknown): StudioObjectiveSpec {
  if (isStudioObjectiveSpec(value)) return value;
  return emptyObjectiveSpec();
}

function createId(label: string): string {
  const slug = label
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 48);
  const suffix =
    typeof crypto !== 'undefined' && 'randomUUID' in crypto
      ? crypto.randomUUID().slice(0, 8)
      : `${Date.now()}`;
  return `${slug || 'objective'}_${suffix}`;
}

function uniqueTermId(spec: StudioObjectiveSpec, label: string): string {
  const existing = new Set(spec.terms.map((term) => term.id));
  let id = createId(label);
  let counter = 1;
  while (existing.has(id)) {
    id = `${id}_${counter}`;
    counter += 1;
  }
  return id;
}

export function objectiveTermEnabled(term: StudioObjectiveTermSpec): boolean {
  return term.metadata.disabled !== true;
}

export function createObjectiveTerm({
  spec,
  label,
  sourceSelector,
  targetSelector = null,
}: {
  spec: StudioObjectiveSpec;
  label: string;
  sourceSelector?: StudioSelectorRef | null;
  targetSelector?: StudioSelectorRef | null;
}): StudioObjectiveTermSpec {
  return enrichObjectiveTermWithSelectorSchema({
    id: uniqueTermId(spec, label),
    type_id: 'TargetStateLoss',
    label,
    role: 'loss',
    source_selector: sourceSelector ?? null,
    target_selector: targetSelector,
    operator: 'distance',
    penalty: 'squared_l2',
    temporal_selector: { mode: 'all' },
    weight: 1,
    units: selectorValueSchema(sourceSelector)?.units ?? sourceSelector?.units ?? null,
    validation: null,
    metadata: {
      authored_in: 'scenario_projection_workspace',
    },
  });
}

function prefixedSelectorMetadata(
  prefix: 'source' | 'target',
  selector: StudioSelectorRef | null | undefined
): Record<string, unknown> {
  const schemaMetadata = selectorSchemaMetadata(selector);
  return Object.fromEntries(
    Object.entries({
      selector: selector ?? null,
      selector_compact: selector?.compact ?? null,
      ...schemaMetadata,
    }).map(([key, value]) => [`${prefix}_${key}`, value])
  );
}

export function enrichObjectiveTermWithSelectorSchema(
  term: StudioObjectiveTermSpec
): StudioObjectiveTermSpec {
  const sourceUnits =
    term.units ??
    selectorValueSchema(term.source_selector)?.units ??
    term.source_selector?.units ??
    selectorValueSchema(term.target_selector)?.units ??
    term.target_selector?.units ??
    null;
  return {
    ...term,
    units: sourceUnits,
    metadata: {
      ...term.metadata,
      ...prefixedSelectorMetadata('source', term.source_selector),
      ...prefixedSelectorMetadata('target', term.target_selector),
      temporal_selector: term.temporal_selector ?? null,
      retention: term.retention ?? term.metadata.retention ?? null,
    },
  };
}

export function addObjectiveTerm(
  spec: StudioObjectiveSpec,
  term: StudioObjectiveTermSpec
): StudioObjectiveSpec {
  return {
    ...spec,
    terms: [...spec.terms, term],
    metadata: { ...spec.metadata, edited_from: 'objective_authoring' },
  };
}

export function updateObjectiveTerm(
  spec: StudioObjectiveSpec,
  termId: string,
  updates: Partial<StudioObjectiveTermSpec>
): StudioObjectiveSpec {
  return {
    ...spec,
    terms: spec.terms.map((term) =>
      term.id === termId
        ? enrichObjectiveTermWithSelectorSchema({
            ...term,
            ...updates,
            id: term.id,
            units:
              updates.source_selector !== undefined || updates.target_selector !== undefined
                ? updates.units
                : updates.units ?? term.units,
            metadata: {
              ...term.metadata,
              ...(updates.metadata ?? {}),
            },
          })
        : term
    ),
    metadata: { ...spec.metadata, edited_from: 'objective_authoring' },
  };
}

export function setObjectiveTermEnabled(
  spec: StudioObjectiveSpec,
  termId: string,
  enabled: boolean
): StudioObjectiveSpec {
  const term = spec.terms.find((candidate) => candidate.id === termId);
  if (!term) return spec;
  return updateObjectiveTerm(spec, termId, {
    metadata: {
      ...term.metadata,
      disabled: !enabled,
    },
  });
}

export function removeObjectiveTerm(
  spec: StudioObjectiveSpec,
  termId: string
): StudioObjectiveSpec {
  return {
    ...spec,
    terms: spec.terms.filter((term) => term.id !== termId),
    metadata: { ...spec.metadata, edited_from: 'objective_authoring' },
  };
}

function sanitizedLossKey(term: StudioObjectiveTermSpec, fallbackIndex: number): string {
  const raw = term.id || term.label || `objective_${fallbackIndex}`;
  const key = raw
    .replace(/^objective_term:/, '')
    .replace(/^objective:/, '')
    .replace(/[^a-zA-Z0-9_]+/g, '_')
    .replace(/^_+|_+$/g, '');
  return key || `objective_${fallbackIndex}`;
}

function normFromPenalty(penalty: string | null | undefined): NormFunction | undefined {
  return penalty && VALID_NORMS.has(penalty as NormFunction) ? (penalty as NormFunction) : undefined;
}

export function objectiveGraphPortTarget(
  selector: StudioSelectorRef | null | undefined
): { nodeId: string; direction: 'input' | 'output'; port: string } | null {
  if (!selector) return null;
  if (selector.namespace === 'graph_port' && selector.target_id && selector.path) {
    return {
      nodeId: selector.target_id,
      direction: selector.metadata.direction === 'input' ? 'input' : 'output',
      port: selector.path,
    };
  }

  const nodeId =
    typeof selector.metadata.graph_port_node_id === 'string'
      ? selector.metadata.graph_port_node_id
      : null;
  const port =
    typeof selector.metadata.graph_port_name === 'string'
      ? selector.metadata.graph_port_name
      : null;
  const direction = selector.metadata.graph_port_direction === 'input' ? 'input' : 'output';
  if (!nodeId || !port) return null;
  return { nodeId, direction, port };
}

export function objectiveSelectorSubpath(
  selector: StudioSelectorRef | null | undefined
): string {
  return typeof selector?.metadata.subpath === 'string' ? selector.metadata.subpath : '';
}

function statePathForPortSubpath(
  portSelector: StudioSelectorRef,
  subpath: string
): string {
  const nodeId = portSelector.target_id ?? 'node';
  const port = portSelector.path ?? 'output';
  const normalized = subpath.trim().replace(/^\.|\.$/g, '');
  if (nodeId === 'mechanics' && port === 'effector') {
    if (normalized === 'position') return 'states.mechanics.effector.pos';
    if (normalized === 'velocity') return 'states.mechanics.effector.vel';
  }
  if (nodeId === 'network' && port === 'hidden' && (!normalized || normalized === 'hidden')) {
    return 'states.net.hidden';
  }
  return ['states', nodeId, port, normalized].filter(Boolean).join('.');
}

export function selectorWithSubpath(
  portSelector: StudioSelectorRef | null | undefined,
  subpath: string
): StudioSelectorRef | null {
  if (!portSelector) return null;
  const normalized = subpath.trim().replace(/^\.|\.$/g, '');
  if (!normalized) return portSelector;
  const nodeId = portSelector.target_id ?? null;
  const port = portSelector.path ?? null;
  return {
    namespace: 'state_path',
    compact: `path:${statePathForPortSubpath(portSelector, normalized)}`,
    target_id: nodeId,
    path: statePathForPortSubpath(portSelector, normalized),
    role: portSelector.role,
    expected_shape: portSelector.expected_shape,
    dtype: portSelector.dtype,
    units: portSelector.units,
    frame: portSelector.frame,
    metadata: {
      ...portSelector.metadata,
      source: 'port_substate_selector',
      subpath: normalized,
      graph_port_node_id: nodeId,
      graph_port_name: port,
      graph_port_direction: portSelector.metadata.direction === 'input' ? 'input' : 'output',
      graph_port_compact: portSelector.compact,
    },
  };
}

function timeAggregationFromObjective(
  value: StudioObjectiveTermSpec['temporal_selector']
): TimeAggregationSpec | undefined {
  if (!value || typeof value !== 'object' || !('mode' in value)) return undefined;
  return value as TimeAggregationSpec;
}

function retentionFromObjective(term: StudioObjectiveTermSpec): LossTermSpec['retention'] | undefined {
  if (term.retention) return term.retention;
  const metadataRetention = term.metadata.retention;
  if (
    metadataRetention &&
    typeof metadataRetention === 'object' &&
    'mode' in metadataRetention
  ) {
    return metadataRetention as LossTermSpec['retention'];
  }
  return undefined;
}

export function lossSpecFromObjectiveSpec(spec: StudioObjectiveSpec): LossTermSpec {
  const children: Record<string, LossTermSpec> = {};
  spec.terms.forEach((term, index) => {
    if (!objectiveTermEnabled(term)) return;
    if (!['loss', 'regularizer', 'reward', 'constraint'].includes(term.role)) return;
    children[sanitizedLossKey(term, index)] = {
      schema_id: LOSS_TERM_SPEC_SCHEMA_ID,
      schema_version: LOSS_TERM_SPEC_SCHEMA_VERSION,
      type: term.type_id || 'TargetStateLoss',
      label: term.label,
      weight: term.weight,
      selector: term.source_selector?.compact,
      target_selector: term.target_selector?.compact ?? null,
      ...(term.target_value !== undefined ? { target_value: term.target_value } : {}),
      retention: retentionFromObjective(term) ?? null,
      norm: normFromPenalty(term.penalty),
      ...(term.matrix !== undefined ? { matrix: term.matrix } : {}),
      ...(term.matrix_kind !== undefined ? { matrix_kind: term.matrix_kind } : {}),
      time_agg: timeAggregationFromObjective(term.temporal_selector),
    };
  });
  return {
    schema_id: LOSS_TERM_SPEC_SCHEMA_ID,
    schema_version: LOSS_TERM_SPEC_SCHEMA_VERSION,
    type: 'Composite',
    label: 'scenario_objective',
    weight: 1,
    children,
  };
}

function selectorFromGraphPortEntity(entity: StudioScenarioEntity): StudioSelectorRef | null {
  return entity.selector ?? null;
}

function selectorFromGraphEdgeEntity(entity: StudioScenarioEntity): StudioSelectorRef | null {
  if (entity.selector) return entity.selector;
  const wire = entity.metadata.wire as WireSpec | undefined;
  if (!wire) return null;
  return {
    namespace: 'graph_port',
    compact: `port:${wire.source_node}.${wire.source_port}`,
    target_id: wire.source_node,
    path: wire.source_port,
    role: 'observed',
    metadata: { inferred_from: entity.id, direction: 'output' },
  };
}

export function sourceSelectorForEntity(
  entity: StudioScenarioEntity | null | undefined,
  _registry: StudioScenarioEntityRegistry
): StudioSelectorRef | null {
  if (!entity) return null;
  if (entity.kind === 'graph_port') return selectorFromGraphPortEntity(entity);
  if (entity.kind === 'graph_edge') return selectorFromGraphEdgeEntity(entity);
  if (entity.kind === 'probe') return entity.selector ?? null;
  if (entity.kind === 'task_data') return entity.selector ?? null;
  return null;
}

export function targetSelectorForEntity(
  entity: StudioScenarioEntity | null | undefined
): StudioSelectorRef | null {
  if (!entity) return null;
  if (entity.kind !== 'task_object') return null;
  return entity.selector ?? null;
}

export function relatedObjectiveEntityIds(
  term: StudioObjectiveTermSpec
): string[] {
  const ids: string[] = [];
  const portTarget = objectiveGraphPortTarget(term.source_selector);
  if (portTarget) {
    ids.push(graphPortEntityId(portTarget.nodeId, portTarget.direction, portTarget.port));
  }
  ids.push(objectiveEntityId(term.id));
  return ids;
}

export function componentOutputSelector(
  nodeId: string,
  component: ComponentSpec | undefined
): StudioSelectorRef | null {
  const firstOutput = component?.output_ports[0];
  if (!firstOutput) return null;
  return {
    namespace: 'graph_port',
    compact: `port:${nodeId}.${firstOutput}`,
    target_id: nodeId,
    path: firstOutput,
    role: 'observed',
    metadata: { direction: 'output' },
  };
}
