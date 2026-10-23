import { STATE_FIELD_TREE, type StateFieldNode } from '@/types/analysis';
import type {
  SelectorTargetSchema,
  StudioObjectiveSpec,
  StudioScenarioEntityRegistry,
  StudioSchemaOrigin,
  StudioSchemaRegistry,
  StudioSelectorRef,
  ValueSchema,
} from '@/types/workspace';

export type SelectorOptionGroup =
  | 'ports'
  | 'state'
  | 'task'
  | 'mechanics'
  | 'probes'
  | 'observables'
  | 'analysis';

export interface StudioSelectorOption {
  id: string;
  group: SelectorOptionGroup;
  label: string;
  detail: string | null;
  selector: StudioSelectorRef;
  source_entity_id: string | null;
  used_by_objective_ids: string[];
  origin: StudioSchemaOrigin | 'entity_registry' | 'state_browser';
  schema_target_id?: string | null;
}

const GROUP_LABELS: Record<SelectorOptionGroup, string> = {
  ports: 'Graph ports',
  state: 'State paths',
  task: 'Task',
  mechanics: 'Mechanics',
  probes: 'Probes',
  observables: 'Retained observables',
  analysis: 'Analysis',
};

const STATE_HINTS: Record<
  string,
  {
    label: string;
    detail: string;
    target_id?: string;
    graph_port_node_id?: string;
    graph_port_name?: string;
    graph_port_direction?: 'input' | 'output';
    subpath?: string;
    expected_shape?: unknown[];
    units?: string;
  }
> = {
  'states.mechanics.effector.pos': {
    label: 'Effector position',
    detail: 'trajectory · x/y',
    target_id: 'mechanics',
    graph_port_node_id: 'mechanics',
    graph_port_name: 'effector',
    graph_port_direction: 'output',
    subpath: 'position',
    expected_shape: ['time', 2],
  },
  'states.mechanics.effector.vel': {
    label: 'Effector velocity',
    detail: 'trajectory · x/y',
    target_id: 'mechanics',
    graph_port_node_id: 'mechanics',
    graph_port_name: 'effector',
    graph_port_direction: 'output',
    subpath: 'velocity',
    expected_shape: ['time', 2],
  },
  'states.net.hidden': {
    label: 'Network hidden state',
    detail: 'trajectory · units',
    target_id: 'network',
    graph_port_node_id: 'network',
    graph_port_name: 'hidden',
    graph_port_direction: 'output',
    subpath: 'hidden',
    expected_shape: ['time', 'units'],
  },
  'states.net.output': {
    label: 'Network output',
    detail: 'trajectory',
    target_id: 'network',
    expected_shape: ['time', 'channels'],
  },
  'states.efferent.output': {
    label: 'Motor command',
    detail: 'trajectory · actuators',
    target_id: 'mechanics',
    expected_shape: ['time', 'actuators'],
  },
  'states.feedback.noise': {
    label: 'Feedback noise',
    detail: 'trajectory',
    target_id: 'feedback',
    expected_shape: ['time', 'channels'],
  },
  'task.validation_trials.targets': {
    label: 'Validation targets',
    detail: 'task targets',
    target_id: 'task',
    expected_shape: ['trials', 2],
  },
};

export function selectorGroupLabel(group: SelectorOptionGroup): string {
  return GROUP_LABELS[group];
}

function stringValue(value: unknown): string | null {
  return typeof value === 'string' && value.length > 0 ? value : null;
}

function directionValue(value: unknown): 'input' | 'output' | null {
  return value === 'input' || value === 'output' ? value : null;
}

function isSelectorRef(value: unknown): value is StudioSelectorRef {
  if (!value || typeof value !== 'object') return false;
  const candidate = value as Partial<StudioSelectorRef>;
  return (
    typeof candidate.namespace === 'string' &&
    typeof candidate.compact === 'string' &&
    typeof candidate.metadata === 'object' &&
    candidate.metadata !== null
  );
}

function isValueSchema(value: unknown): value is ValueSchema {
  if (!value || typeof value !== 'object') return false;
  const candidate = value as Partial<ValueSchema>;
  return (
    typeof candidate.id === 'string' &&
    typeof candidate.label === 'string' &&
    typeof candidate.kind === 'string' &&
    typeof candidate.origin === 'string' &&
    typeof candidate.metadata === 'object' &&
    candidate.metadata !== null
  );
}

export function selectorValueSchema(
  selector: StudioSelectorRef | null | undefined
): ValueSchema | null {
  return isValueSchema(selector?.metadata.value_schema) ? selector.metadata.value_schema : null;
}

export function selectorSchemaMetadata(
  selector: StudioSelectorRef | null | undefined
): Record<string, unknown> {
  const valueSchema = selectorValueSchema(selector);
  return {
    value_schema: valueSchema,
    value_schema_id: valueSchema?.id ?? null,
    value_schema_kind: valueSchema?.kind ?? null,
    schema_origin:
      valueSchema?.origin ??
      (typeof selector?.metadata.schema_origin === 'string'
        ? selector.metadata.schema_origin
        : null),
    dtype: selector?.dtype ?? valueSchema?.dtype ?? null,
    shape: selector?.expected_shape ?? valueSchema?.shape ?? null,
    rank: valueSchema?.rank ?? null,
    units: selector?.units ?? valueSchema?.units ?? null,
    frame: selector?.frame ?? valueSchema?.frame ?? null,
  };
}

export function selectorBase(selector: StudioSelectorRef | null | undefined): StudioSelectorRef | null {
  if (!selector) return null;
  const base = selector.metadata.base_selector;
  return isSelectorRef(base) ? base : selector;
}

export function selectorAccessExpression(
  selector: StudioSelectorRef | null | undefined
): string {
  if (!selector) return '';
  const expression = selector.metadata.access_expression;
  return typeof expression === 'string' ? expression : '';
}

export function selectorWithAccessExpression(
  selector: StudioSelectorRef | null | undefined,
  expression: string
): StudioSelectorRef | null {
  const base = selectorBase(selector);
  if (!base) return null;
  const trimmed = expression.trim();
  if (!trimmed) return base;
  const label = `${selectorDisplayLabel(base)}${trimmed}`;
  return {
    namespace: 'custom',
    compact: `${base.compact}${trimmed}`,
    target_id: base.target_id ?? null,
    path: `${base.path ?? base.compact}${trimmed}`,
    role: base.role ?? 'observed',
    expected_shape: null,
    dtype: base.dtype ?? null,
    units: base.units ?? null,
    frame: base.frame ?? null,
    metadata: {
      label,
      detail: 'custom subpath / slice',
      source: 'selector_expression',
      base_selector: base,
      access_expression: trimmed,
    },
  };
}

export function selectorDisplayLabel(selector: StudioSelectorRef | null | undefined): string {
  if (!selector) return 'None';
  const base = selector.metadata.base_selector;
  const expression = selectorAccessExpression(selector);
  if (expression && isSelectorRef(base)) return `${selectorDisplayLabel(base)}${expression}`;
  const label = typeof selector.metadata.label === 'string' ? selector.metadata.label : null;
  if (label) return label;
  const path = selector.path ?? selector.compact.replace(/^path:/, '');
  const hint = selector.namespace === 'state_path' ? STATE_HINTS[path] : null;
  if (hint) return hint.label;
  if (selector.namespace === 'graph_port') {
    return selector.target_id && selector.path
      ? `${selector.target_id}.${selector.path}`
      : selector.compact;
  }
  if (selector.namespace === 'task_data') {
    return selector.path ? `task.${selector.path}` : selector.compact;
  }
  if (selector.namespace === 'state_path') {
    return path;
  }
  return selector.path ?? selector.compact;
}

export function selectorDetail(selector: StudioSelectorRef | null | undefined): string | null {
  if (!selector) return null;
  const path = selector.path ?? selector.compact.replace(/^path:/, '');
  const hint = selector.namespace === 'state_path' ? STATE_HINTS[path] : null;
  const parts = [
    selector.units ?? selectorValueSchema(selector)?.units ?? hint?.units,
    selector.dtype ?? selectorValueSchema(selector)?.dtype,
    Array.isArray(selector.expected_shape ?? selectorValueSchema(selector)?.shape ?? hint?.expected_shape)
      ? (selector.expected_shape ?? selectorValueSchema(selector)?.shape ?? hint?.expected_shape)?.join(' x ')
      : null,
    typeof selector.metadata.detail === 'string' ? selector.metadata.detail : hint?.detail ?? null,
  ];
  return parts.filter(Boolean).join(' · ') || null;
}

function graphPortDirection(selector: StudioSelectorRef): string | null {
  return typeof selector.metadata.direction === 'string' ? selector.metadata.direction : null;
}

function optionMatchesGraphPort(option: StudioSelectorOption, selector: StudioSelectorRef): boolean {
  if (selector.namespace !== 'graph_port') return false;
  const metadata = option.selector.metadata;
  const direction = graphPortDirection(selector);
  return (
    metadata.graph_port_node_id === selector.target_id &&
    metadata.graph_port_name === selector.path &&
    (!direction || metadata.graph_port_direction === direction)
  );
}

export function selectorOptionsForGraphPort(
  selector: StudioSelectorRef | null | undefined,
  options: StudioSelectorOption[]
): StudioSelectorOption[] {
  if (!selector || selector.namespace !== 'graph_port') return [];
  return options.filter(
    (option) => option.group !== 'ports' && optionMatchesGraphPort(option, selector)
  );
}

export function preferredSelectorForGraphPort(
  selector: StudioSelectorRef | null | undefined,
  options: StudioSelectorOption[]
): StudioSelectorRef | null {
  if (!selector || selector.namespace !== 'graph_port') return selector ?? null;
  return selectorOptionsForGraphPort(selector, options)[0]?.selector ?? selector;
}

function flattenStateFields(nodes: StateFieldNode[]): StateFieldNode[] {
  return nodes.flatMap((node) => [node, ...(node.children ? flattenStateFields(node.children) : [])]);
}

function objectiveUsesByCompact(objectiveSpec: StudioObjectiveSpec | null | undefined) {
  const byCompact = new Map<string, string[]>();
  for (const term of objectiveSpec?.terms ?? []) {
    for (const selector of [term.source_selector, term.target_selector]) {
      if (!selector?.compact) continue;
      byCompact.set(selector.compact, [...(byCompact.get(selector.compact) ?? []), term.id]);
    }
  }
  return byCompact;
}

function stateSelectorForPath(path: string, label: string): StudioSelectorRef {
  const hint = STATE_HINTS[path];
  return {
    namespace: 'state_path',
    compact: `path:${path}`,
    target_id: hint?.target_id ?? null,
    path,
    role: 'observed',
    expected_shape: hint?.expected_shape ?? null,
    dtype: null,
    units: hint?.units ?? null,
    frame: null,
    metadata: {
      label: hint?.label ?? label,
      detail: hint?.detail ?? null,
      source: 'curated_state_hint',
      schema_origin: 'curated_fallback',
      subpath: hint?.subpath,
      graph_port_node_id: hint?.graph_port_node_id,
      graph_port_name: hint?.graph_port_name,
      graph_port_direction: hint?.graph_port_direction,
    },
  };
}

function parsePortSelector(selector: string): { nodeId: string; port: string } | null {
  const match = /^port:([^.\s]+)\.(.+)$/.exec(selector);
  if (!match) return null;
  return { nodeId: match[1], port: match[2] };
}

function parseEdgeSelector(
  selector: string
): { sourceNode: string; sourcePort: string; targetNode: string; targetPort: string; edgeId: string } | null {
  const match = /^edge:([^.\s]+)\.([^-\s]+)->([^.\s]+)\.(.+)$/.exec(selector);
  if (!match) return null;
  const edgeId = `${match[1]}:${match[2]}->${match[3]}:${match[4]}`;
  return {
    sourceNode: match[1],
    sourcePort: match[2],
    targetNode: match[3],
    targetPort: match[4],
    edgeId,
  };
}

function schemaPortForTarget(
  target: SelectorTargetSchema,
  schemaRegistry: StudioSchemaRegistry | null | undefined
) {
  const sourcePortId = stringValue(target.source.port_id);
  return sourcePortId
    ? schemaRegistry?.ports.find((port) => port.id === sourcePortId) ?? null
    : null;
}

function schemaTaskDataForTarget(
  target: SelectorTargetSchema,
  schemaRegistry: StudioSchemaRegistry | null | undefined
) {
  const sourceTaskDataId = stringValue(target.source.task_data_id)?.replace(/^task_data:/, '');
  const selectorPath = target.selector.startsWith('task_data:')
    ? target.selector.replace(/^task_data:/, '')
    : null;
  return (
    schemaRegistry?.task_data.find(
      (data) =>
        data.id.replace(/^task_data:/, '') === sourceTaskDataId ||
        data.path === selectorPath
    ) ?? null
  );
}

function schemaTargetGraphMetadata(
  target: SelectorTargetSchema,
  schemaRegistry: StudioSchemaRegistry | null | undefined
): Record<string, unknown> {
  const path = target.selector.startsWith('path:') ? target.selector.replace(/^path:/, '') : null;
  const hint = path ? STATE_HINTS[path] : null;
  const port = schemaPortForTarget(target, schemaRegistry);
  const parsedPort = parsePortSelector(target.selector);
  const graphPortNodeId =
    stringValue(target.source.graph_port_node_id) ??
    port?.node_id ??
    parsedPort?.nodeId ??
    stringValue(target.source.node_id) ??
    stringValue(target.source.source_node) ??
    hint?.graph_port_node_id ??
    null;
  const graphPortName =
    stringValue(target.source.graph_port_name) ??
    port?.port ??
    parsedPort?.port ??
    stringValue(target.source.port) ??
    stringValue(target.source.source_port) ??
    hint?.graph_port_name ??
    null;
  const graphPortDirection =
    directionValue(target.source.graph_port_direction) ??
    directionValue(target.source.direction) ??
    port?.direction ??
    hint?.graph_port_direction ??
    null;

  return {
    ...(graphPortNodeId ? { graph_port_node_id: graphPortNodeId } : {}),
    ...(graphPortName ? { graph_port_name: graphPortName } : {}),
    ...(graphPortDirection ? { graph_port_direction: graphPortDirection } : {}),
  };
}

function schemaTargetSelectorRef(
  target: SelectorTargetSchema,
  schemaRegistry: StudioSchemaRegistry | null | undefined
): StudioSelectorRef {
  const valueSchema = target.value_schema;
  const port = schemaPortForTarget(target, schemaRegistry);
  const taskData = schemaTaskDataForTarget(target, schemaRegistry);
  const parsedPort = parsePortSelector(target.selector);
  const parsedEdge = parseEdgeSelector(target.selector);
  const graphMetadata = schemaTargetGraphMetadata(target, schemaRegistry);
  const pathSelector = target.selector.startsWith('path:')
    ? target.selector.replace(/^path:/, '')
    : null;
  const graphOutput = target.selector.startsWith('graph_output:')
    ? target.selector.replace(/^graph_output:/, '')
    : null;
  const taskDataPath = target.selector.startsWith('task_data:')
    ? target.selector.replace(/^task_data:/, '')
    : null;
  const probeId = target.selector.startsWith('probe:')
    ? target.selector.replace(/^probe:/, '')
    : stringValue(target.source.probe_id);
  const hint = pathSelector ? STATE_HINTS[pathSelector] : null;
  const graphPortDirection =
    directionValue(graphMetadata.graph_port_direction) ??
    directionValue(target.source.direction);

  let namespace: StudioSelectorRef['namespace'] = 'custom';
  let targetId: string | null = stringValue(target.source.target_id);
  let path: string | null = null;
  let role: StudioSelectorRef['role'] = 'observed';

  if (target.kind === 'port' || parsedPort) {
    namespace = 'graph_port';
    targetId = port?.node_id ?? parsedPort?.nodeId ?? targetId;
    path = port?.port ?? parsedPort?.port ?? null;
    role = graphPortDirection === 'input' ? 'editable' : 'observed';
  } else if (target.kind === 'edge' || parsedEdge) {
    namespace = 'graph_edge';
    targetId = stringValue(target.source.edge_id) ?? parsedEdge?.edgeId ?? targetId;
    path = null;
  } else if (target.kind === 'recurrent_carry') {
    namespace = 'recurrent_carry';
    targetId = stringValue(target.source.edge_id) ?? parsedEdge?.edgeId ?? targetId;
    path = null;
  } else if (target.kind === 'graph_output' || graphOutput) {
    namespace = 'graph_output';
    targetId = graphOutput ?? stringValue(target.source.output_name) ?? targetId;
    path = graphOutput ?? stringValue(target.source.output_name);
  } else if (target.kind === 'task_data' || taskDataPath) {
    namespace = 'task_data';
    targetId = schemaRegistry?.scenario_id ?? targetId;
    path = taskData?.path ?? taskDataPath;
    role = taskData?.bindable ? 'editable' : 'observed';
  } else if (target.kind === 'probe' || probeId) {
    namespace = 'probe';
    targetId = probeId;
    path = null;
  } else if (pathSelector) {
    namespace = 'state_path';
    targetId = hint?.target_id ?? targetId;
    path = pathSelector;
  }

  return {
    namespace,
    compact: target.selector,
    target_id: targetId,
    path,
    role,
    expected_shape: valueSchema.shape ?? hint?.expected_shape ?? null,
    dtype: valueSchema.dtype ?? null,
    units: valueSchema.units ?? hint?.units ?? null,
    frame: valueSchema.frame ?? null,
    metadata: {
      ...target.metadata,
      ...graphMetadata,
      label: target.label,
      detail: stringValue(target.metadata.detail),
      source: 'studio_schema_registry',
      schema_target_id: target.id,
      schema_kind: target.kind,
      schema_origin: target.origin,
      selector_source: target.source,
      value_schema: valueSchema,
      ...(parsedEdge ? { edge_id: parsedEdge.edgeId } : {}),
      ...(hint?.subpath ? { subpath: hint.subpath } : {}),
      ...(graphPortDirection ? { direction: graphPortDirection } : {}),
    },
  };
}

function schemaTargetGroup(
  target: SelectorTargetSchema,
  selector: StudioSelectorRef
): SelectorOptionGroup {
  if (target.kind === 'port' || selector.namespace === 'graph_port') return 'ports';
  if (
    target.kind === 'edge' ||
    target.kind === 'graph_output' ||
    target.kind === 'recurrent_carry' ||
    target.kind === 'retained_observable' ||
    selector.namespace === 'graph_edge' ||
    selector.namespace === 'graph_output' ||
    selector.namespace === 'recurrent_carry' ||
    selector.namespace === 'retained_observable'
  ) {
    return 'observables';
  }
  if (target.kind === 'task_data' || selector.namespace === 'task_data') return 'task';
  if (target.kind === 'probe' || selector.namespace === 'probe') return 'probes';
  if (target.kind === 'objective') return 'analysis';
  if (selector.path?.startsWith('task.')) return 'task';
  return 'state';
}

function optionForSelector({
  group,
  label,
  detail,
  selector,
  sourceEntityId,
  usedByObjectiveIds,
  origin,
  schemaTargetId = null,
}: {
  group: SelectorOptionGroup;
  label: string;
  detail: string | null;
  selector: StudioSelectorRef;
  sourceEntityId: string | null;
  usedByObjectiveIds: string[];
  origin: StudioSelectorOption['origin'];
  schemaTargetId?: string | null;
}): StudioSelectorOption {
  return {
    id: `${group}:${selector.compact}`,
    group,
    label,
    detail,
    selector,
    source_entity_id: sourceEntityId,
    used_by_objective_ids: usedByObjectiveIds,
    origin,
    schema_target_id: schemaTargetId,
  };
}

export function selectorOptionsForRegistry({
  registry,
  schemaRegistry,
  objectiveSpec,
}: {
  registry: StudioScenarioEntityRegistry;
  schemaRegistry?: StudioSchemaRegistry | null;
  objectiveSpec?: StudioObjectiveSpec | null;
}): StudioSelectorOption[] {
  const usedByCompact = objectiveUsesByCompact(objectiveSpec);
  const options: StudioSelectorOption[] = [];
  const schemaTargets = schemaRegistry?.selector_targets ?? [];

  for (const target of schemaTargets) {
    const selector = schemaTargetSelectorRef(target, schemaRegistry);
    options.push(
      optionForSelector({
        group: schemaTargetGroup(target, selector),
        label: target.label,
        detail: selectorDetail(selector),
        selector,
        sourceEntityId: null,
        usedByObjectiveIds: usedByCompact.get(selector.compact) ?? [],
        origin: target.origin,
        schemaTargetId: target.id,
      })
    );
  }

  for (const entity of Object.values(registry.entities)) {
    if (!entity.selector) continue;
    if (entity.kind === 'graph_port') {
      options.push(
        optionForSelector({
          group: 'ports',
          label: entity.label,
          detail: entity.summary ?? null,
          selector: {
            ...entity.selector,
            metadata: { ...entity.selector.metadata, label: entity.label },
          },
          sourceEntityId: entity.id,
          usedByObjectiveIds: usedByCompact.get(entity.selector.compact) ?? [],
          origin: 'entity_registry',
        })
      );
    }
    if (entity.kind === 'probe') {
      options.push(
        optionForSelector({
          group: 'probes',
          label: entity.label,
          detail: entity.summary ?? null,
          selector: {
            ...entity.selector,
            metadata: { ...entity.selector.metadata, label: entity.label },
          },
          sourceEntityId: entity.id,
          usedByObjectiveIds: usedByCompact.get(entity.selector.compact) ?? [],
          origin: 'entity_registry',
        })
      );
    }
    if (entity.kind === 'graph_edge') {
      options.push(
        optionForSelector({
          group: entity.selector.namespace === 'state_path' ? 'state' : 'observables',
          label: entity.label,
          detail: entity.summary ?? null,
          selector: {
            ...entity.selector,
            metadata: { ...entity.selector.metadata, label: entity.label },
          },
          sourceEntityId: entity.id,
          usedByObjectiveIds: usedByCompact.get(entity.selector.compact) ?? [],
          origin: 'entity_registry',
        })
      );
    }
    if (
      entity.kind === 'task_object' ||
      entity.kind === 'task_data' ||
      entity.kind === 'mechanics_object'
    ) {
      options.push(
        optionForSelector({
          group: entity.kind === 'mechanics_object' ? 'mechanics' : 'task',
          label: entity.label,
          detail: entity.summary ?? null,
          selector: {
            ...entity.selector,
            metadata: { ...entity.selector.metadata, label: entity.label },
          },
          sourceEntityId: entity.id,
          usedByObjectiveIds: usedByCompact.get(entity.selector.compact) ?? [],
          origin: 'entity_registry',
        })
      );
    }
  }

  if (schemaTargets.length === 0) {
    for (const node of flattenStateFields(STATE_FIELD_TREE)) {
      if (!STATE_HINTS[node.path]) continue;
      const selector = stateSelectorForPath(node.path, node.label);
      options.push(
        optionForSelector({
          group: node.path.startsWith('task.') ? 'task' : 'state',
          label: selectorDisplayLabel(selector),
          detail: selectorDetail(selector),
          selector,
          sourceEntityId: null,
          usedByObjectiveIds: usedByCompact.get(selector.compact) ?? [],
          origin: 'state_browser',
        })
      );
    }
  }

  const seen = new Set<string>();
  return options
    .filter((option) => {
      if (seen.has(option.id)) return false;
      seen.add(option.id);
      return true;
    })
    .sort((a, b) => {
      const groupOrder =
        ['ports', 'observables', 'state', 'task', 'mechanics', 'probes', 'analysis'].indexOf(a.group) -
        ['ports', 'observables', 'state', 'task', 'mechanics', 'probes', 'analysis'].indexOf(b.group);
      return groupOrder || a.label.localeCompare(b.label);
    });
}
