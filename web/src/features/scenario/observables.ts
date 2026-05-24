import { selectorDetail, selectorDisplayLabel, selectorValueSchema } from './selectors';
import type {
  RetainedObservableSpec,
  RetainedObservableTargetSpec,
  RetentionPolicySpec,
} from '@/types/graph';
import type { StudioSelectorRef } from '@/types/workspace';

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
  return `obs:${slug || 'observable'}_${suffix}`;
}

function edgeIdFromSelector(selector: StudioSelectorRef): string | null {
  if (typeof selector.metadata.edge_id === 'string') return selector.metadata.edge_id;
  const match = /^edge:([^.\s]+)\.([^-\s]+)->([^.\s]+)\.(.+)$/.exec(selector.compact);
  if (!match) return null;
  return `${match[1]}:${match[2]}->${match[3]}:${match[4]}`;
}

function explicitObservableRetentionPolicy(): RetentionPolicySpec {
  return {
    mode: 'trajectory',
    window_size: null,
    order: null,
    reason: 'explicit_observable_authoring',
  };
}

export function selectorToRetainedObservableTarget(
  selector: StudioSelectorRef
): RetainedObservableTargetSpec | null {
  if (selector.namespace === 'graph_port') {
    return {
      kind: 'port',
      selector: selector.compact,
      node_id: selector.target_id ?? null,
      port: selector.path ?? null,
      timing: selector.metadata.direction === 'input' ? 'input' : 'output',
      metadata: {
        selector_namespace: selector.namespace,
        selector_detail: selectorDetail(selector),
      },
    };
  }
  if (selector.namespace === 'graph_edge' || selector.namespace === 'recurrent_carry') {
    return {
      kind: selector.namespace === 'recurrent_carry' ? 'recurrent_carry' : 'edge',
      selector: selector.compact,
      edge_id: edgeIdFromSelector(selector),
      timing: 'step',
      metadata: {
        selector_namespace: selector.namespace,
        selector_detail: selectorDetail(selector),
      },
    };
  }
  if (selector.namespace === 'graph_output') {
    return {
      kind: 'graph_output',
      selector: selector.compact,
      node_id:
        typeof selector.metadata.graph_port_node_id === 'string'
          ? selector.metadata.graph_port_node_id
          : null,
      port:
        typeof selector.metadata.graph_port_name === 'string'
          ? selector.metadata.graph_port_name
          : null,
      path: selector.path ?? selector.compact.replace(/^graph_output:/, ''),
      timing: 'step',
      metadata: {
        selector_namespace: selector.namespace,
        selector_detail: selectorDetail(selector),
      },
    };
  }
  if (selector.namespace === 'state_path') {
    return {
      kind: 'state_path',
      selector: selector.compact,
      node_id: selector.target_id ?? null,
      path: selector.path ?? selector.compact.replace(/^path:/, ''),
      timing: 'step',
      metadata: {
        selector_namespace: selector.namespace,
        selector_detail: selectorDetail(selector),
      },
    };
  }
  if (selector.namespace === 'task_data') {
    return {
      kind: 'task_data',
      selector: selector.compact,
      path: selector.path ?? selector.compact.replace(/^task_data:/, ''),
      timing: 'step',
      metadata: {
        selector_namespace: selector.namespace,
        selector_detail: selectorDetail(selector),
      },
    };
  }
  return null;
}

export function createRetainedObservable({
  selector,
  label = selectorDisplayLabel(selector),
  retention = explicitObservableRetentionPolicy(),
  existingIds = new Set<string>(),
}: {
  selector: StudioSelectorRef;
  label?: string;
  retention?: RetentionPolicySpec;
  existingIds?: Set<string>;
}): RetainedObservableSpec | null {
  const target = selectorToRetainedObservableTarget(selector);
  if (!target) return null;
  let id = createId(label);
  let counter = 1;
  while (existingIds.has(id)) {
    id = `${id}_${counter}`;
    counter += 1;
  }
  return {
    id,
    label,
    selector: selector.compact,
    target,
    retention,
    value_schema: selectorValueSchema(selector),
    metadata: {
      authored_in: 'studio_observables',
      selector_label: selectorDisplayLabel(selector),
      selector_detail: selectorDetail(selector),
    },
  };
}

export function retainedObservableSelectorPatch(
  selector: StudioSelectorRef
): Pick<RetainedObservableSpec, 'selector' | 'target' | 'value_schema' | 'metadata'> | null {
  const target = selectorToRetainedObservableTarget(selector);
  if (!target) return null;
  return {
    selector: selector.compact,
    target,
    value_schema: selectorValueSchema(selector),
    metadata: {
      authored_in: 'studio_observables',
      selector_label: selectorDisplayLabel(selector),
      selector_detail: selectorDetail(selector),
    },
  };
}

export function retainedObservableTargetKindLabel(
  target: RetainedObservableTargetSpec | null | undefined
): string {
  switch (target?.kind) {
    case 'port':
      return 'Port';
    case 'edge':
      return 'Edge';
    case 'graph_output':
      return 'Graph output';
    case 'recurrent_carry':
      return 'Recurrent carry';
    case 'state_path':
      return 'State path';
    case 'task_data':
      return 'Task data';
    default:
      return 'Selector';
  }
}
