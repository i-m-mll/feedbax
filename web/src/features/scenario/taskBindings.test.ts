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
