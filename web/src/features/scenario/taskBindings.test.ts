import { describe, expect, it } from 'vitest';
import {
  createDefaultTaskBindingSpec,
  defaultTaskData,
  ensureTaskBindingSpec,
  removeTaskBindingsForTargetNodes,
  retargetTaskBindingsForNodeRename,
  scopedTaskBindingSpec,
  taskBindingId,
  targetInputOccupied,
  GRAPH_BINDABLE_TASK_DATA_ROLES,
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

const graphWithNamedMux: GraphSpec = {
  ...graphWithTaskMux,
  nodes: {
    mux: {
      type: 'Mux',
      params: { n_inputs: 3 },
      input_ports: ['in_0', 'in_1', 'in_2'],
      output_ports: ['output'],
    },
    network: graphWithTaskMux.nodes.network,
  },
  wires: [
    {
      source_node: 'mux',
      source_port: 'output',
      target_node: 'network',
      target_port: 'input',
    },
  ],
};

const delayedTask: TaskSpec = { type: 'DelayedReaches', params: {} };

describe('task data bindings', () => {
  it('includes component parameters in graph-bindable roles', () => {
    expect(GRAPH_BINDABLE_TASK_DATA_ROLES.has('component_parameter')).toBe(true);
    expect(GRAPH_BINDABLE_TASK_DATA_ROLES.has('intervention')).toBe(false);
  });

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
      path: 'inputs.effector_target',
      role: 'model_input',
      expected_shape: ['time', 4],
      value_spec: {
        mode: 'function',
        function_id: 'delayed_reach_target_position',
      },
    });
    expect(data.find((item) => item.id === 'hold')?.value_spec).toMatchObject({
      mode: 'constant',
      value: { active: 1, inactive: 0 },
    });
    expect(data.find((item) => item.id === 'movement_target')).toMatchObject({
      path: 'targets.effector',
      role: 'target',
      bindable: false,
      value_spec: {
        mode: 'function',
        function_id: 'delayed_reach_movement_target',
      },
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

  it('discovers delayed-reach task muxes by graph role rather than node id', () => {
    const spec = createDefaultTaskBindingSpec(graphWithNamedMux, delayedTask);

    expect(spec.bindings.map((binding) => [binding.source_data_id, binding.target_node_id, binding.target_port])).toEqual([
      ['target_position', 'mux', 'in_0'],
      ['hold', 'mux', 'in_1'],
      ['target_on', 'mux', 'in_2'],
    ]);
  });

  it('refreshes canonical delayed-reach task data schemas in saved specs', () => {
    const savedSpec = createDefaultTaskBindingSpec(graphWithNamedMux, delayedTask);
    savedSpec.exposed_data = savedSpec.exposed_data.map((data) =>
      data.id === 'target_position'
        ? {
            ...data,
            path: 'inputs.effector_target.pos',
            expected_shape: ['time', 2],
            value_spec: data.value_spec
              ? { ...data.value_spec, shape: ['time', 2] }
              : data.value_spec,
          }
        : data
    );

    const normalized = ensureTaskBindingSpec(savedSpec, graphWithNamedMux, delayedTask);
    const targetPosition = normalized.exposed_data.find((data) => data.id === 'target_position');

    expect(targetPosition).toMatchObject({
      path: 'inputs.effector_target',
      expected_shape: ['time', 4],
      value_spec: { shape: ['time', 4] },
    });
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

  it('treats task data occupancy as target-scoped so one datum can fan out', () => {
    const spec: StudioTaskBindingSpec = {
      schema_version: 'feedbax.studio.task_bindings.v2',
      exposed_data: [
        {
          id: 'target_on',
          label: 'Target shown',
          kind: 'signal',
          role: 'model_input',
          path: 'inputs.target_on',
          bindable: true,
          metadata: {},
        },
      ],
      bindings: [
        {
          id: 'task:target_on->task_mux:in_2',
          source_data_id: 'target_on',
          target_node_id: 'task_mux',
          target_port: 'in_2',
          role: 'model_input',
          metadata: {},
        },
      ],
      metadata: {},
    };

    expect(targetInputOccupied(graphWithTaskMux, spec, 'task_mux', 'in_2')).toBe(true);
    expect(targetInputOccupied(graphWithTaskMux, spec, 'task_mux', 'in_3')).toBe(false);
  });

  it('retargets task binding IDs and target nodes when model nodes are renamed', () => {
    const spec = createDefaultTaskBindingSpec(graphWithNetworkInput, {
      type: 'ReachingTask',
      params: {},
    });

    const renamed = retargetTaskBindingsForNodeRename(spec, 'network', 'controller');

    expect(renamed.bindings).toEqual([
      {
        id: 'task:inputs->controller:input',
        source_data_id: 'inputs',
        target_node_id: 'controller',
        target_port: 'input',
        role: 'model_input',
        metadata: {},
      },
    ]);
  });

  it('keeps same local node IDs isolated by graph path', () => {
    const spec = createDefaultTaskBindingSpec(graphWithTaskMux, delayedTask);
    const scopedBinding = {
      id: taskBindingId('hold', 'task_mux', 'in_1', ['network']),
      source_data_id: 'hold',
      target_graph_path: ['network'],
      target_node_id: 'task_mux',
      target_port: 'in_1',
      role: 'model_input',
      metadata: {},
    };
    const withSubgraphBinding: StudioTaskBindingSpec = {
      ...spec,
      bindings: [...spec.bindings, scopedBinding],
    };

    expect(scopedTaskBindingSpec(withSubgraphBinding, []).bindings).toEqual(spec.bindings);
    expect(scopedTaskBindingSpec(withSubgraphBinding, ['network']).bindings).toEqual([
      scopedBinding,
    ]);
    expect(targetInputOccupied(graphWithTaskMux, scopedTaskBindingSpec(withSubgraphBinding, []), 'task_mux', 'in_1')).toBe(true);
    expect(targetInputOccupied(graphWithTaskMux, scopedTaskBindingSpec(withSubgraphBinding, ['network']), 'task_mux', 'in_1')).toBe(true);
  });

  it('removes deleted-node task bindings only in the active graph path', () => {
    const spec = createDefaultTaskBindingSpec(graphWithTaskMux, delayedTask);
    const subgraphBinding = {
      id: taskBindingId('hold', 'task_mux', 'in_1', ['network']),
      source_data_id: 'hold',
      target_graph_path: ['network'],
      target_node_id: 'task_mux',
      target_port: 'in_1',
      role: 'model_input',
      metadata: {},
    };
    const withSubgraphBinding: StudioTaskBindingSpec = {
      ...spec,
      bindings: [...spec.bindings, subgraphBinding],
    };

    const pruned = removeTaskBindingsForTargetNodes(withSubgraphBinding, ['task_mux'], ['network']);

    expect(pruned.bindings).toEqual(spec.bindings);
  });

  it('removes bindings that target deleted model nodes', () => {
    const spec = createDefaultTaskBindingSpec(graphWithTaskMux, delayedTask);

    const pruned = removeTaskBindingsForTargetNodes(spec, ['task_mux']);

    expect(pruned.bindings).toEqual([]);
    expect(pruned.exposed_data.map((data) => data.id)).toContain('target_position');
  });
});
