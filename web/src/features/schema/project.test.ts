import { describe, expect, it } from 'vitest';
import { validateGraph } from '@/features/graph/validation';
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
