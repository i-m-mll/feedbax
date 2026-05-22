import { describe, expect, it } from 'vitest';
import { validateGraph } from '@/features/graph/validation';
import { projectStudioSchema, validateConnectionAgainstSchema } from './project';
import {
  applyBoundaryOverrides,
  deriveDimensionConstraints,
  deriveSubgraphBoundaryOverrides,
} from './dimensions';
import type { ComponentDefinition } from '@/types/components';
import type { GraphSpec } from '@/types/graph';
import type { StudioTaskBindingSpec } from '@/types/workspace';

const components: ComponentDefinition[] = [
  {
    name: 'VectorSource',
    category: 'Test',
    description: '',
    param_schema: [],
    input_ports: ['input'],
    output_ports: ['output'],
    icon: 'Circle',
    default_params: {},
    port_types: {
      inputs: { input: { dtype: 'vector' } },
      outputs: { output: { dtype: 'vector' } },
    },
  },
  {
    name: 'ScalarSink',
    category: 'Test',
    description: '',
    param_schema: [],
    input_ports: ['excitation'],
    output_ports: ['force'],
    icon: 'Circle',
    default_params: {},
    port_types: {
      inputs: { excitation: { dtype: 'scalar' } },
      outputs: { force: { dtype: 'scalar' } },
    },
  },
  {
    name: 'Mux',
    category: 'Test',
    description: '',
    param_schema: [],
    input_ports: ['in_0', 'in_1'],
    output_ports: ['output'],
    icon: 'GitMerge',
    default_params: { n_inputs: 2 },
    port_types: {
      inputs: { in_0: { dtype: 'vector' }, in_1: { dtype: 'vector' } },
      outputs: { output: { dtype: 'vector' } },
    },
  },
  {
    name: 'Network',
    category: 'Test',
    description: '',
    param_schema: [],
    input_ports: ['input', 'hidden'],
    output_ports: ['output', 'hidden'],
    icon: 'Circle',
    default_params: { input_size: 1, hidden_size: 100, out_size: 1 },
    port_types: {
      inputs: { input: { dtype: 'vector' }, hidden: { dtype: 'vector' } },
      outputs: { output: { dtype: 'vector' }, hidden: { dtype: 'vector' } },
    },
  },
  {
    name: 'GRU',
    category: 'Test',
    description: '',
    param_schema: [],
    input_ports: ['input', 'hidden'],
    output_ports: ['output', 'hidden'],
    icon: 'Circle',
    default_params: { input_size: 1, hidden_size: 100 },
    port_types: {
      inputs: { input: { dtype: 'vector' }, hidden: { dtype: 'vector' } },
      outputs: { output: { dtype: 'vector' }, hidden: { dtype: 'vector' } },
    },
  },
  {
    name: 'Arm6MuscleRigidTendon',
    category: 'Test',
    description: '',
    param_schema: [],
    input_ports: ['excitation'],
    output_ports: ['forces'],
    icon: 'Circle',
    default_params: {},
    port_types: {
      inputs: { excitation: { dtype: 'vector' } },
      outputs: { forces: { dtype: 'vector' } },
    },
  },
];

const graph: GraphSpec = {
  nodes: {
    source: {
      type: 'VectorSource',
      params: {},
      input_ports: ['input'],
      output_ports: ['output'],
    },
    sink: {
      type: 'ScalarSink',
      params: {},
      input_ports: ['excitation'],
      output_ports: ['force'],
    },
  },
  wires: [],
  input_ports: [],
  output_ports: [],
  input_bindings: {},
  output_bindings: {},
};

function taskBindingSpec(sourceDataId = 'inputs'): StudioTaskBindingSpec {
  return {
    schema_version: 'feedbax.studio.task_bindings.v2',
    exposed_data: [
      {
        id: 'inputs',
        label: 'Inputs',
        kind: 'signal',
        role: 'model_input',
        path: 'inputs',
        bindable: true,
        dtype: 'vector',
        metadata: {},
      },
      {
        id: 'targets',
        label: 'Targets',
        kind: 'target',
        role: 'target',
        path: 'targets',
        bindable: false,
        dtype: 'scalar',
        metadata: {},
      },
    ],
    bindings: [
      {
        id: `task:${sourceDataId}->sink:excitation`,
        source_data_id: sourceDataId,
        target_node_id: 'sink',
        target_port: 'excitation',
        role: 'model_input',
        metadata: {},
      },
    ],
    metadata: {},
  };
}

describe('projectStudioSchema', () => {
  it('validates graph wires against projected port schemas', () => {
    const registry = projectStudioSchema(
      {
        ...graph,
        wires: [
          {
            source_node: 'source',
            source_port: 'output',
            target_node: 'sink',
            target_port: 'excitation',
          },
          {
            source_node: 'source',
            source_port: 'input',
            target_node: 'sink',
            target_port: 'excitation',
          },
        ],
      },
      components
    );
    const issueTypes = new Set(registry.issues.map((issue) => issue.type));

    expect(issueTypes).toContain('graph_wire_dtype_mismatch');
    expect(issueTypes).toContain('wrong_source_port_direction');
    expect(issueTypes).toContain('graph_input_occupied');
  });

  it('validates task-data binding conflicts against projected schemas', () => {
    const registry = projectStudioSchema(
      {
        ...graph,
        wires: [
          {
            source_node: 'source',
            source_port: 'output',
            target_node: 'sink',
            target_port: 'excitation',
          },
        ],
      },
      components,
      taskBindingSpec('targets')
    );
    const issueTypes = new Set(registry.issues.map((issue) => issue.type));

    expect(issueTypes).toContain('task_data_not_bindable');
    expect(issueTypes).toContain('task_binding_target_occupied');
  });

  it('validates task binding identity as part of the binding contract', () => {
    const spec = taskBindingSpec();
    spec.bindings = [
      {
        ...spec.bindings[0],
        id: 'not-canonical',
      },
      {
        ...spec.bindings[0],
      },
      {
        ...spec.bindings[0],
      },
    ];

    const registry = projectStudioSchema(graph, components, spec);
    const issueTypes = new Set(registry.issues.map((issue) => issue.type));

    expect(issueTypes).toContain('task_binding_id_mismatch');
    expect(issueTypes).toContain('duplicate_task_binding');
  });

  it('treats task-data bindings as graph input connectivity', () => {
    const spec = taskBindingSpec();
    spec.exposed_data[0].dtype = 'float32';
    spec.exposed_data[0].expected_shape = ['time', 'channels'];
    spec.exposed_data[0].value_spec = {
      schema_version: 'feedbax.studio.value.v1',
      mode: 'reference',
      dtype: 'float32',
      shape: ['time', 'channels'],
      metadata: {},
    };
    spec.bindings[0] = {
      ...spec.bindings[0],
      target_node_id: 'source',
      target_port: 'input',
    };
    const registry = projectStudioSchema(graph, components, spec);
    const validation = validateGraph(graph, registry);

    expect(registry.issues.map((issue) => issue.type)).not.toContain(
      'task_binding_dtype_mismatch'
    );
    expect(
      validation.errors.find((error) => error.message === "Input port 'source.input' is not connected")
    ).toBeUndefined();
  });

  it('compares task-data trajectory bindings using their per-step sample view', () => {
    const spec = taskBindingSpec();
    spec.exposed_data[0].dtype = 'float32';
    spec.exposed_data[0].expected_shape = ['time', 2];
    spec.bindings[0] = {
      ...spec.bindings[0],
      target_node_id: 'network',
      target_port: 'input',
    };
    const registry = projectStudioSchema(
      {
        ...graph,
        nodes: {
          network: {
            type: 'Network',
            params: { input_size: 2, out_size: 6 },
            input_ports: ['input'],
            output_ports: ['output'],
          },
        },
      },
      components,
      spec
    );

    const inputs = registry.task_data.find((data) => data.id === 'task_data:inputs');
    const networkInput = registry.ports.find((port) => port.id === 'port:network.input:input');
    expect(inputs?.value_schema.shape).toEqual(['time', 2]);
    expect(inputs?.value_schema.metadata.sample_shape).toEqual([2]);
    expect(networkInput?.value_schema.shape).toBeNull();
    expect(registry.issues.map((issue) => issue.type)).not.toContain(
      'task_binding_rank_mismatch'
    );
    expect(registry.issues.map((issue) => issue.type)).not.toContain(
      'task_binding_shape_mismatch'
    );
  });

  it('keeps protocol Task Data out of graph-facing task bindings', () => {
    const spec = taskBindingSpec('targets');
    spec.exposed_data[1].bindable = true;

    const registry = projectStudioSchema(graph, components, spec);
    const targetData = registry.task_data.find((data) => data.path === 'targets');
    const issueTypes = new Set(registry.issues.map((issue) => issue.type));

    expect(targetData?.role).toBe('target');
    expect(targetData?.bindable).toBe(false);
    expect(targetData?.metadata.task_data_surface).toBe('protocol');
    expect(issueTypes).toContain('task_data_bindable_role_mismatch');
    expect(issueTypes).toContain('task_data_protocol_path_bindable');
    expect(issueTypes).toContain('task_data_not_bindable');
  });

  it('exposes connection validation for canvas guards', () => {
    const registry = projectStudioSchema(graph, components);
    const issues = validateConnectionAgainstSchema(
      registry,
      'source',
      'output',
      'sink',
      'excitation'
    );

    expect(issues.map((issue) => issue.type)).toContain('graph_wire_dtype_mismatch');
  });

  it('projects dynamic mux input ports with the mux input schema', () => {
    const muxTaskBindings: StudioTaskBindingSpec = {
      schema_version: 'feedbax.studio.task_bindings.v2',
      exposed_data: [
        {
          id: 'target_on',
          label: 'Target shown',
          kind: 'signal',
          role: 'model_input',
          path: 'inputs.target_on',
          bindable: true,
          dtype: 'float32',
          expected_shape: ['time', 1],
          metadata: {},
        },
      ],
      bindings: [
        {
          id: 'task:target_on->mux:in_2',
          source_data_id: 'target_on',
          target_node_id: 'mux',
          target_port: 'in_2',
          role: 'model_input',
          metadata: {},
        },
      ],
      metadata: {},
    };
    const registry = projectStudioSchema(
      {
        ...graph,
        nodes: {
          mux: {
            type: 'Mux',
            params: { n_inputs: 2 },
            input_ports: ['in_0', 'in_1'],
            output_ports: ['output'],
          },
        },
      },
      components,
      muxTaskBindings
    );

    const dynamicPort = registry.ports.find((port) => port.id === 'port:mux.in_2:input');
    expect(dynamicPort?.value_schema.dtype).toBe('vector');
    expect(dynamicPort?.origin).toBe('declared');
    expect(registry.issues.map((issue) => issue.type)).not.toContain(
      'unknown_task_binding_target_port'
    );
  });

  it('infers mux output width from bound task-data sample shapes', () => {
    const muxTaskBindings: StudioTaskBindingSpec = {
      schema_version: 'feedbax.studio.task_bindings.v2',
      exposed_data: [
        {
          id: 'target_position',
          label: 'Target position',
          kind: 'signal',
          role: 'model_input',
          path: 'inputs.effector_target',
          bindable: true,
          dtype: 'float32',
          expected_shape: ['time', 4],
          metadata: {},
        },
        {
          id: 'hold',
          label: 'Hold/go cue',
          kind: 'signal',
          role: 'model_input',
          path: 'inputs.hold',
          bindable: true,
          dtype: 'float32',
          expected_shape: ['time', 1],
          metadata: {},
        },
      ],
      bindings: [
        {
          id: 'task:target_position->mux:in_0',
          source_data_id: 'target_position',
          target_node_id: 'mux',
          target_port: 'in_0',
          role: 'model_input',
          metadata: {},
        },
        {
          id: 'task:hold->mux:in_1',
          source_data_id: 'hold',
          target_node_id: 'mux',
          target_port: 'in_1',
          role: 'model_input',
          metadata: {},
        },
      ],
      metadata: {},
    };
    const registry = projectStudioSchema(
      {
        ...graph,
        nodes: {
          mux: {
            type: 'Mux',
            params: { n_inputs: 2 },
            input_ports: ['in_0', 'in_1'],
            output_ports: ['output'],
          },
        },
      },
      components,
      muxTaskBindings
    );

    const muxOutput = registry.ports.find((port) => port.id === 'port:mux.output:output');
    expect(muxOutput?.value_schema.shape).toEqual([5]);
    expect(muxOutput?.value_schema.metadata.dimension_source).toBe('mux_concat_inputs');
  });

  it('derives network size constraints from task muxes and mechanics consumers', () => {
    const spec: StudioTaskBindingSpec = {
      schema_version: 'feedbax.studio.task_bindings.v2',
      exposed_data: [
        {
          id: 'target_position',
          label: 'Target position',
          kind: 'signal',
          role: 'model_input',
          path: 'inputs.effector_target',
          bindable: true,
          dtype: 'float32',
          expected_shape: ['time', 4],
          metadata: {},
        },
        {
          id: 'hold',
          label: 'Hold/go cue',
          kind: 'signal',
          role: 'model_input',
          path: 'inputs.hold',
          bindable: true,
          dtype: 'float32',
          expected_shape: ['time', 1],
          metadata: {},
        },
      ],
      bindings: [
        {
          id: 'task:target_position->task_mux:in_0',
          source_data_id: 'target_position',
          target_node_id: 'task_mux',
          target_port: 'in_0',
          role: 'model_input',
          metadata: {},
        },
        {
          id: 'task:hold->task_mux:in_1',
          source_data_id: 'hold',
          target_node_id: 'task_mux',
          target_port: 'in_1',
          role: 'model_input',
          metadata: {},
        },
      ],
      metadata: {},
    };
    const schemaGraph: GraphSpec = {
      ...graph,
      nodes: {
        task_mux: {
          type: 'Mux',
          params: { n_inputs: 2 },
          input_ports: ['in_0', 'in_1'],
          output_ports: ['output'],
        },
        network: {
          type: 'Network',
          params: { input_size: 4, out_size: 2 },
          input_ports: ['input'],
          output_ports: ['output'],
        },
        arm: {
          type: 'Arm6MuscleRigidTendon',
          params: {},
          input_ports: ['excitation'],
          output_ports: ['forces'],
        },
      },
      wires: [
        {
          source_node: 'task_mux',
          source_port: 'output',
          target_node: 'network',
          target_port: 'input',
        },
        {
          source_node: 'network',
          source_port: 'output',
          target_node: 'arm',
          target_port: 'excitation',
        },
      ],
    };
    const registry = projectStudioSchema(schemaGraph, components, spec);
    const constraints = deriveDimensionConstraints(schemaGraph, registry);

    expect(constraints.find((item) => item.param === 'input_size')).toMatchObject({
      node_id: 'network',
      inferred_value: 5,
      current_value: 4,
      status: 'conflict',
    });
    expect(constraints.find((item) => item.param === 'out_size')).toMatchObject({
      node_id: 'network',
      inferred_value: 6,
      current_value: 2,
      status: 'conflict',
    });
  });

  it('uses subgraph boundary schemas for parent port compatibility', () => {
    const spec: StudioTaskBindingSpec = {
      schema_version: 'feedbax.studio.task_bindings.v2',
      exposed_data: [
        {
          id: 'target_position',
          label: 'Target position',
          kind: 'signal',
          role: 'model_input',
          path: 'inputs.effector_target',
          bindable: true,
          dtype: 'float32',
          expected_shape: ['time', 4],
          metadata: {},
        },
        {
          id: 'hold',
          label: 'Hold/go cue',
          kind: 'signal',
          role: 'model_input',
          path: 'inputs.hold',
          bindable: true,
          dtype: 'float32',
          expected_shape: ['time', 1],
          metadata: {},
        },
        {
          id: 'target_on',
          label: 'Target shown',
          kind: 'signal',
          role: 'model_input',
          path: 'inputs.target_on',
          bindable: true,
          dtype: 'float32',
          expected_shape: ['time', 1],
          metadata: {},
        },
      ],
      bindings: [
        {
          id: 'task:target_position->task_mux:in_0',
          source_data_id: 'target_position',
          target_node_id: 'task_mux',
          target_port: 'in_0',
          role: 'model_input',
          metadata: {},
        },
        {
          id: 'task:hold->task_mux:in_1',
          source_data_id: 'hold',
          target_node_id: 'task_mux',
          target_port: 'in_1',
          role: 'model_input',
          metadata: {},
        },
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
    const childGraph: GraphSpec = {
      ...graph,
      nodes: {
        cell: {
          type: 'GRU',
          params: { input_size: 4, hidden_size: 100 },
          input_ports: ['input', 'hidden'],
          output_ports: ['output', 'hidden'],
        },
        readout: {
          type: 'Linear',
          params: { input_size: 100, output_size: 2 },
          input_ports: ['input'],
          output_ports: ['output'],
        },
      },
      wires: [
        {
          source_node: 'cell',
          source_port: 'output',
          target_node: 'readout',
          target_port: 'input',
        },
      ],
      input_ports: ['input', 'hidden'],
      output_ports: ['output', 'hidden'],
      input_bindings: { input: ['cell', 'input'], hidden: ['cell', 'hidden'] },
      output_bindings: { output: ['readout', 'output'], hidden: ['cell', 'output'] },
    };
    const parentGraph: GraphSpec = {
      ...graph,
      nodes: {
        task_mux: {
          type: 'Mux',
          params: { n_inputs: 3 },
          input_ports: ['in_0', 'in_1', 'in_2'],
          output_ports: ['output'],
        },
        network: {
          type: 'Network',
          params: { input_size: 4, hidden_size: 100, out_size: 2 },
          input_ports: ['input', 'hidden'],
          output_ports: ['output', 'hidden'],
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
      subgraphs: { network: childGraph },
    };
    const registry = projectStudioSchema(parentGraph, components, spec);
    const hiddenInput = registry.ports.find((port) => port.id === 'port:network.hidden:input');
    const issues = validateConnectionAgainstSchema(
      registry,
      'task_mux',
      'output',
      'network',
      'hidden'
    );

    expect(registry.ports.find((port) => port.id === 'port:task_mux.output:output')?.value_schema.shape).toEqual([6]);
    expect(hiddenInput?.value_schema.shape).toEqual([100]);
    expect(issues.map((issue) => issue.type)).toContain('graph_wire_shape_mismatch');

    const overrides = deriveSubgraphBoundaryOverrides(parentGraph, 'network', registry);
    const childRegistry = applyBoundaryOverrides(
      projectStudioSchema(childGraph, components),
      overrides
    );
    const childConstraints = deriveDimensionConstraints(childGraph, childRegistry);
    expect(
      childConstraints.find((item) => item.node_id === 'cell' && item.param === 'input_size')
    ).toMatchObject({
      inferred_value: 6,
      current_value: 4,
      status: 'conflict',
    });
  });

  it('uses graph boundary schemas as dimension sources for child nodes', () => {
    const childGraph: GraphSpec = {
      ...graph,
      nodes: {
        cell: {
          type: 'GRU',
          params: { input_size: 4, hidden_size: 100 },
          input_ports: ['input', 'hidden'],
          output_ports: ['output', 'hidden'],
        },
      },
      input_ports: ['input'],
      output_ports: ['hidden'],
      input_bindings: { input: ['cell', 'input'] },
      output_bindings: { hidden: ['cell', 'output'] },
    };
    const registry = projectStudioSchema(childGraph, components);
    const overridden = {
      ...registry,
      ports: registry.ports.map((port) =>
        port.id === 'port:graph.input:input'
          ? { ...port, value_schema: { ...port.value_schema, shape: [6], rank: 1 } }
          : port
      ),
    };
    const constraints = deriveDimensionConstraints(childGraph, overridden);

    expect(constraints.find((item) => item.node_id === 'cell' && item.param === 'input_size')).toMatchObject({
      inferred_value: 6,
      current_value: 4,
      status: 'conflict',
    });
  });

  it('validates typed intervention taps against selector schemas', () => {
    const registry = projectStudioSchema(
      {
        ...graph,
        taps: [
          {
            id: 'good-clamp',
            type: 'intervention',
            position: { afterNode: 'source' },
            paths: {},
            transform: {
              type: 'intervention',
              params: {},
              intervention: {
                operation: 'clamp',
                target_selector: {
                  namespace: 'graph_port',
                  compact: 'port:source.output',
                  target_id: 'source',
                  path: 'output',
                  role: 'observed',
                  metadata: {},
                },
                bounds: { min: 0, max: 1 },
                metadata: {},
              },
            },
          },
          {
            id: 'bad-offset',
            type: 'intervention',
            position: { afterNode: 'sink' },
            paths: {},
            transform: {
              type: 'intervention',
              params: {},
              intervention: {
                operation: 'offset',
                target_selector: {
                  namespace: 'custom',
                  compact: 'path:missing',
                  role: 'observed',
                  metadata: {},
                },
                metadata: {},
              },
            },
          },
        ],
      },
      components
    );
    const issueTypes = new Set(registry.issues.map((issue) => issue.type));

    expect(issueTypes).toContain('intervention_unknown_target');
    expect(issueTypes).not.toContain('intervention_missing_bounds');
  });
});
