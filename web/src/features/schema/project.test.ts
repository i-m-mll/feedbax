import { describe, expect, it } from 'vitest';
import { projectStudioSchema, validateConnectionAgainstSchema } from './project';
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
        path: 'inputs',
        bindable: true,
        dtype: 'vector',
        metadata: {},
      },
      {
        id: 'targets',
        label: 'Targets',
        kind: 'target',
        path: 'targets',
        bindable: false,
        dtype: 'scalar',
        metadata: {},
      },
    ],
    bindings: [
      {
        id: 'binding',
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
});
