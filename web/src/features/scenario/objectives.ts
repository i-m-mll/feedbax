import {
  graphPortEntityId,
  objectiveEntityId,
} from '@/features/scenario/entities';
import type { ComponentSpec, WireSpec } from '@/types/graph';
import type { LossTermSpec, NormFunction, TimeAggregationSpec } from '@/types/training';
import type {
  StudioObjectiveSpec,
  StudioObjectiveTermSpec,
  StudioScenarioEntity,
  StudioScenarioEntityRegistry,
  StudioSelectorRef,
} from '@/types/workspace';

const OBJECTIVE_SCHEMA_VERSION = 'feedbax.studio.objective.v1';
const VALID_NORMS = new Set<NormFunction>(['squared_l2', 'l2', 'l1', 'huber']);

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
  return {
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
    units: null,
    validation: null,
    metadata: {
      authored_in: 'scenario_projection_workspace',
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
        ? {
            ...term,
            ...updates,
            id: term.id,
            metadata: {
              ...term.metadata,
              ...(updates.metadata ?? {}),
            },
          }
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

function timeAggregationFromObjective(
  value: StudioObjectiveTermSpec['temporal_selector']
): TimeAggregationSpec | undefined {
  if (!value || typeof value !== 'object' || !('mode' in value)) return undefined;
  return value as TimeAggregationSpec;
}

export function lossSpecFromObjectiveSpec(spec: StudioObjectiveSpec): LossTermSpec {
  const children: Record<string, LossTermSpec> = {};
  spec.terms.forEach((term, index) => {
    if (!objectiveTermEnabled(term)) return;
    if (!['loss', 'regularizer', 'reward', 'constraint'].includes(term.role)) return;
    children[sanitizedLossKey(term, index)] = {
      type: term.type_id || 'TargetStateLoss',
      label: term.label,
      weight: term.weight,
      selector: term.source_selector?.compact,
      norm: normFromPenalty(term.penalty),
      time_agg: timeAggregationFromObjective(term.temporal_selector),
    };
  });
  return {
    type: 'Composite',
    label: 'scenario_objective',
    weight: 1,
    children,
  };
}

function selectorFromGraphPortEntity(entity: StudioScenarioEntity): StudioSelectorRef | null {
  return entity.selector ?? null;
}

function selectorFromGraphNodeEntity(entity: StudioScenarioEntity): StudioSelectorRef | null {
  const outputs = Array.isArray(entity.metadata.output_ports)
    ? (entity.metadata.output_ports as unknown[])
    : [];
  const firstOutput = outputs.find((value): value is string => typeof value === 'string');
  const nodeId = typeof entity.metadata.node_id === 'string' ? entity.metadata.node_id : null;
  if (!nodeId || !firstOutput) return null;
  return {
    namespace: 'graph_port',
    compact: `port:${nodeId}.${firstOutput}`,
    target_id: nodeId,
    path: firstOutput,
    role: 'observed',
    metadata: { inferred_from: entity.id },
  };
}

function selectorFromGraphEdgeEntity(entity: StudioScenarioEntity): StudioSelectorRef | null {
  const wire = entity.metadata.wire as WireSpec | undefined;
  if (!wire) return null;
  return {
    namespace: 'graph_port',
    compact: `port:${wire.source_node}.${wire.source_port}`,
    target_id: wire.source_node,
    path: wire.source_port,
    role: 'observed',
    metadata: { inferred_from: entity.id },
  };
}

function selectorFromMechanicsEntity(
  entity: StudioScenarioEntity,
  registry: StudioScenarioEntityRegistry
): StudioSelectorRef | null {
  const graphNodeRelation = entity.relations.find((relation) => relation.kind === 'binds');
  const graphNode = graphNodeRelation ? registry.entities[graphNodeRelation.entity_id] : null;
  if (graphNode?.kind === 'graph_node') {
    return selectorFromGraphNodeEntity(graphNode);
  }
  return entity.selector ?? null;
}

export function sourceSelectorForEntity(
  entity: StudioScenarioEntity | null | undefined,
  registry: StudioScenarioEntityRegistry
): StudioSelectorRef | null {
  if (!entity) return null;
  if (entity.kind === 'graph_port') return selectorFromGraphPortEntity(entity);
  if (entity.kind === 'graph_node') return selectorFromGraphNodeEntity(entity);
  if (entity.kind === 'graph_edge') return selectorFromGraphEdgeEntity(entity);
  if (entity.kind === 'probe') return entity.selector ?? null;
  if (entity.kind === 'mechanics_object') return selectorFromMechanicsEntity(entity, registry);
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
  if (term.source_selector?.namespace === 'graph_port' && term.source_selector.target_id && term.source_selector.path) {
    ids.push(graphPortEntityId(term.source_selector.target_id, 'output', term.source_selector.path));
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
    metadata: {},
  };
}
