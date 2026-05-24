import { describe, expect, it } from 'vitest';
import { selectorTreeFromScenarioOptions } from './DataSourceNode';
import type { StudioSelectorOption } from '@/features/scenario/selectors';
import type { StudioSelectorRef } from '@/types/workspace';

function selector(
  namespace: StudioSelectorRef['namespace'],
  compact: string,
  overrides: Partial<StudioSelectorRef> = {},
): StudioSelectorRef {
  const { metadata, ...rest } = overrides;
  const base: StudioSelectorRef = {
    namespace,
    compact,
    target_id: null,
    path: null,
    role: 'observed',
    expected_shape: null,
    dtype: null,
    units: null,
    frame: null,
    metadata: {},
    ...rest,
  };
  return { ...base, metadata: { ...metadata } };
}

function option(
  label: string,
  group: StudioSelectorOption['group'],
  selectorRef: StudioSelectorRef,
  overrides: Partial<StudioSelectorOption> = {},
): StudioSelectorOption {
  return {
    id: `${group}:${selectorRef.compact}`,
    group,
    label,
    detail: 'schema detail',
    selector: selectorRef,
    source_entity_id: null,
    used_by_objective_ids: [],
    origin: 'entity_registry',
    schema_target_id: null,
    ...overrides,
  };
}

describe('selectorTreeFromScenarioOptions', () => {
  it('groups graph ports and state paths under model variable owners', () => {
    const tree = selectorTreeFromScenarioOptions([
      option(
        'cell.input',
        'ports',
        selector('graph_port', 'port:cell.input', {
          target_id: 'cell',
          path: 'input',
          metadata: { direction: 'input' },
        }),
      ),
      option(
        'Network hidden state',
        'state',
        selector('state_path', 'path:states.net.hidden', {
          target_id: 'cell',
          path: 'states.net.hidden',
          metadata: { graph_port_node_id: 'cell' },
        }),
      ),
      option(
        'Task',
        'task',
        selector('task_object', 'task:train', {
          target_id: 'train',
          path: null,
        }),
      ),
    ]);

    expect(tree.map((node) => node.label)).toEqual(['Model variables']);
    expect(tree[0]?.children?.map((node) => node.label)).toEqual(['Cell']);
    expect(tree[0]?.children?.[0]?.children).toEqual([
      expect.objectContaining({
        label: 'Hidden state',
        detail: 'schema detail',
        path: 'path:states.net.hidden',
      }),
      expect.objectContaining({
        label: 'Input',
        path: 'port:cell.input',
      }),
    ]);
  });

  it('filters legacy model owners that are not present in the current graph', () => {
    const tree = selectorTreeFromScenarioOptions(
      [
        option(
          'Cell output',
          'ports',
          selector('graph_port', 'port:cell.output', {
            target_id: 'cell',
            path: 'output',
            metadata: { direction: 'output' },
          }),
        ),
        option(
          'Network output',
          'state',
          selector('state_path', 'path:states.net.output', {
            target_id: 'network',
            path: 'states.net.output',
            metadata: { graph_port_node_id: 'network' },
          }),
        ),
        option(
          'Effector position',
          'state',
          selector('state_path', 'path:states.mechanics.effector.pos', {
            target_id: 'mechanics',
            path: 'states.mechanics.effector.pos',
            metadata: { graph_port_node_id: 'mechanics' },
          }),
        ),
        option(
          'cell -> cell',
          'state',
          selector('state_path', 'path:cell->cell', {
            target_id: 'cell',
            path: 'state',
            metadata: { source: 'state_flow_edge' },
          }),
        ),
      ],
      new Set(['input_mux', 'cell', 'readout']),
    );

    expect(tree[0]?.label).toBe('Model variables');
    expect(tree[0]?.children?.map((node) => node.label)).toEqual(['Cell']);
    expect(tree[0]?.children?.[0]?.children?.map((node) => node.label)).toEqual(['Output']);
  });

  it('deduplicates task data and omits the top-level task object row', () => {
    const tree = selectorTreeFromScenarioOptions([
      option(
        'Task',
        'task',
        selector('task_object', 'task:train', {
          target_id: 'train',
        }),
      ),
      option(
        'Inputs object',
        'task',
        selector('task_data', 'task_data:inputs', {
          target_id: 'train',
          path: 'inputs',
        }),
      ),
      option(
        'Inputs schema',
        'task',
        selector('task_data', 'task_data:inputs', {
          target_id: 'train',
          path: 'inputs',
        }),
        { origin: 'declared', schema_target_id: 'selector:task_data:inputs' },
      ),
    ]);

    expect(tree.map((node) => node.label)).toEqual(['Task data']);
    expect(tree[0]?.children).toEqual([
      expect.objectContaining({
        label: 'Inputs schema',
        path: 'task_data:inputs',
        selector: expect.objectContaining({ compact: 'task_data:inputs' }),
      }),
    ]);
  });
});
