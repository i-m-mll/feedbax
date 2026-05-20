import { describe, expect, it } from 'vitest';
import {
  createDefaultTaskBindingSpec,
  defaultTaskData,
  ensureTaskBindingSpec,
} from '@/features/scenario/taskBindings';
import type { GraphSpec } from '@/types/graph';
import type { TaskSpec } from '@/types/training';
import type { StudioTaskBindingSpec } from '@/types/workspace';

const graphWithNetworkInput: GraphSpec = {
  nodes: {
    network: {
      type: 'Network',
      params: {},
      input_ports: ['input'],
      output_ports: ['output'],
    },
  },
  wires: [],
  input_ports: [],
  output_ports: [],
  input_bindings: {},
  output_bindings: {},
};

const graphWithTaskMux: GraphSpec = {
  ...graphWithNetworkInput,
  nodes: {
    task_mux: {
      type: 'Mux',
      params: { n_inputs: 3 },
      input_ports: ['in_0', 'in_1', 'in_2'],
      output_ports: ['output'],
    },
    network: {
      type: 'Network',
      params: {},
      input_ports: ['input'],
      output_ports: ['output'],
    },
  },
  wires: [
    {
      source_node: 'task_mux',
      source_port: 'output',
      target_node: 'network',
      target_port: 'input',
    },
  ],
};

const delayedTask: TaskSpec = { type: 'DelayedReaches', params: {} };

describe('task data bindings', () => {
  it('exposes delayed-reach inputs as named bindable task data', () => {
    const data = defaultTaskData(delayedTask);

    expect(data.map((item) => [item.id, item.label, item.bindable])).toEqual([
      ['target_position', 'Target position', true],
      ['hold', 'Hold/go cue', true],
      ['target_on', 'Target shown', true],
      ['movement_target', 'Movement target', false],
      ['inits', 'Initial state', false],
      ['intervene', 'Intervention', false],
    ]);
    expect(data.find((item) => item.id === 'target_position')).toMatchObject({
      path: 'inputs.effector_target.pos',
      role: 'model_input',
      expected_shape: ['time', 2],
    });
    expect(data.find((item) => item.id === 'movement_target')).toMatchObject({
      path: 'targets.effector',
      role: 'target',
      bindable: false,
    });
  });

  it('does not seed a single generic network input binding for delayed reaches', () => {
    const spec = createDefaultTaskBindingSpec(graphWithNetworkInput, delayedTask);

    expect(spec.exposed_data.map((item) => item.id)).toContain('target_position');
    expect(spec.exposed_data.map((item) => item.id)).not.toContain('inputs');
    expect(spec.bindings).toEqual([]);
  });

  it('seeds named delayed-reach task data bindings into the task mux', () => {
    const spec = createDefaultTaskBindingSpec(graphWithTaskMux, delayedTask);

    expect(spec.bindings.map((binding) => [binding.source_data_id, binding.target_port])).toEqual([
      ['target_position', 'in_0'],
      ['hold', 'in_1'],
      ['target_on', 'in_2'],
    ]);
    expect(spec.bindings.every((binding) => binding.target_node_id === 'task_mux')).toBe(true);
  });

  it('drops legacy generic Inputs bindings when delayed-reach task data is normalized', () => {
    const legacySpec: StudioTaskBindingSpec = {
      schema_version: 'feedbax.studio.task_bindings.v2',
      exposed_data: [
        {
          id: 'inputs',
          label: 'Inputs',
          kind: 'signal',
          role: 'model_input',
          path: 'inputs',
          bindable: true,
          metadata: {},
        },
      ],
      bindings: [
        {
          id: 'task:inputs->network:input',
          source_data_id: 'inputs',
          target_node_id: 'network',
          target_port: 'input',
          role: 'model_input',
          metadata: {},
        },
      ],
      metadata: {},
    };

    const spec = ensureTaskBindingSpec(legacySpec, graphWithNetworkInput, delayedTask);

    expect(spec.exposed_data.map((item) => item.id)).not.toContain('inputs');
    expect(spec.bindings).toEqual([]);
  });
});
