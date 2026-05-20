import { STATE_FIELD_TREE, type StateFieldNode } from '@/types/analysis';
import type {
  StudioObjectiveSpec,
  StudioScenarioEntityRegistry,
  StudioSelectorRef,
} from '@/types/workspace';

export type SelectorOptionGroup =
  | 'ports'
  | 'state'
  | 'task'
  | 'mechanics'
  | 'probes'
  | 'analysis';

export interface StudioSelectorOption {
  id: string;
  group: SelectorOptionGroup;
  label: string;
  detail: string | null;
  selector: StudioSelectorRef;
  source_entity_id: string | null;
  used_by_objective_ids: string[];
}

const GROUP_LABELS: Record<SelectorOptionGroup, string> = {
  ports: 'Graph ports',
  state: 'State paths',
  task: 'Task',
  mechanics: 'Mechanics',
  probes: 'Probes',
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
    selector.units ?? hint?.units,
    selector.dtype,
    Array.isArray(selector.expected_shape ?? hint?.expected_shape)
      ? (selector.expected_shape ?? hint?.expected_shape)?.join(' x ')
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
      source: 'state_browser',
      subpath: hint?.subpath,
      graph_port_node_id: hint?.graph_port_node_id,
      graph_port_name: hint?.graph_port_name,
      graph_port_direction: hint?.graph_port_direction,
    },
  };
}

function optionForSelector({
  group,
  label,
  detail,
  selector,
  sourceEntityId,
  usedByObjectiveIds,
}: {
  group: SelectorOptionGroup;
  label: string;
  detail: string | null;
  selector: StudioSelectorRef;
  sourceEntityId: string | null;
  usedByObjectiveIds: string[];
}): StudioSelectorOption {
  return {
    id: `${group}:${selector.compact}`,
    group,
    label,
    detail,
    selector,
    source_entity_id: sourceEntityId,
    used_by_objective_ids: usedByObjectiveIds,
  };
}

export function selectorOptionsForRegistry({
  registry,
  objectiveSpec,
}: {
  registry: StudioScenarioEntityRegistry;
  objectiveSpec?: StudioObjectiveSpec | null;
}): StudioSelectorOption[] {
  const usedByCompact = objectiveUsesByCompact(objectiveSpec);
  const options: StudioSelectorOption[] = [];

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
        })
      );
    }
    if (entity.kind === 'graph_edge') {
      options.push(
        optionForSelector({
          group: 'state',
          label: entity.label,
          detail: entity.summary ?? null,
          selector: {
            ...entity.selector,
            metadata: { ...entity.selector.metadata, label: entity.label },
          },
          sourceEntityId: entity.id,
          usedByObjectiveIds: usedByCompact.get(entity.selector.compact) ?? [],
        })
      );
    }
    if (
      entity.kind === 'task_object' ||
      entity.kind === 'task_data' ||
      entity.kind === 'mechanics_object'
    ) {
      const group = entity.kind === 'task_object' ? 'task' : 'mechanics';
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
        })
      );
    }
  }

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
      })
    );
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
        ['ports', 'state', 'task', 'mechanics', 'probes', 'analysis'].indexOf(a.group) -
        ['ports', 'state', 'task', 'mechanics', 'probes', 'analysis'].indexOf(b.group);
      return groupOrder || a.label.localeCompare(b.label);
    });
}
