import { describe, expect, it } from 'vitest';
import {
  normalizeGraphAuthoringTypes,
  normalizeGraphForStudioAuthoring,
  normalizeTaskBindingSpecForStudioAuthoring,
} from '@/features/graph/normalization';
import type { GraphSpec } from '@/types/graph';

const runtimeGraph: GraphSpec = {
  nodes: {
    network: {
      type: 'SimpleStagedNetwork',
      params: { input_size: 4, hidden_size: 100, output_size: 2 },
      input_ports: ['target', 'feedback'],
      output_ports: ['output'],
    },
    task_mux: {
      type: 'Mux',
      params: { n_inputs: 2 },
      input_ports: ['in_0', 'in_1'],
      output_ports: ['output'],
    },
  },
  wires: [
    {
      source_node: 'task_mux',
      source_port: 'output',
      target_node: 'network',
      target_port: 'target',
    },
  ],
  input_ports: ['target'],
  output_ports: ['output'],
  input_bindings: { target: ['network', 'target'] },
  output_bindings: { output: ['network', 'output'] },
  subgraphs: {
    child: {
      nodes: {
        inner: {
          type: 'SimpleStagedNetwork',
          params: { output_size: 3 },
          input_ports: ['target'],
          output_ports: ['output'],
        },
      },
      wires: [],
      input_ports: ['target'],
      output_ports: [],
      input_bindings: { target: ['inner', 'target'] },
      output_bindings: {},
    },
  },
  barnacles: {
    network: [
      {
        id: 'probe:legacy',
        kind: 'probe',
        timing: 'output',
        label: 'Legacy probe',
        read_paths: ['output'],
        write_paths: [],
        transform: '',
      },
    ],
  },
  metadata: {
    name: 'Runtime graph',
    created_at: '2026-05-21T00:00:00Z',
    updated_at: '2026-05-21T00:00:00Z',
    version: '1.0.0',
  },
};

describe('graph authoring normalization', () => {
  it('maps runtime network nodes, target ports, subgraphs, and legacy taps to Studio authoring shape', () => {
    const normalized = normalizeGraphAuthoringTypes(runtimeGraph);

    expect(normalized.nodes.network).toMatchObject({
      type: 'Network',
      params: { input_size: 4, hidden_size: 100, out_size: 2 },
      input_ports: ['input', 'feedback'],
    });
    expect(normalized.wires[0].target_port).toBe('input');
    expect(normalized.input_ports).toEqual(['input']);
    expect(normalized.input_bindings).toEqual({ input: ['network', 'input'] });
    expect(normalized.subgraphs?.child.nodes.inner).toMatchObject({
      type: 'Network',
      input_ports: ['input'],
    });
    expect(normalized.subgraphs?.child.input_bindings).toEqual({ input: ['inner', 'input'] });
    expect(normalized.subgraphs?.network.nodes.cell).toMatchObject({
      type: 'GRU',
      params: { input_size: 4, hidden_size: 100 },
    });
    expect(normalized.subgraphs?.network.nodes.readout).toMatchObject({
      type: 'Linear',
      params: { output_size: 2 },
    });
    expect(normalized.taps).toEqual([
      {
        id: 'probe:legacy',
        type: 'probe',
        position: { afterNode: 'network' },
        paths: { output: 'output' },
        transform: undefined,
      },
    ]);
    expect(normalized.barnacles).toBeUndefined();
  });

  it('normalizes dynamic mux ports after task-data binding ingress', () => {
    const normalized = normalizeGraphForStudioAuthoring(runtimeGraph, {
      schema_version: 'feedbax.studio.task_bindings.v2',
      exposed_data: [],
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
    });

    expect(normalized.nodes.task_mux.input_ports).toEqual(['in_0', 'in_1', 'in_2']);
    expect(normalized.nodes.task_mux.params.n_inputs).toBe(3);
  });

  it('retargets task-data bindings from legacy Network target ports to input ports', () => {
    const graph = normalizeGraphForStudioAuthoring(runtimeGraph);
    const normalized = normalizeTaskBindingSpecForStudioAuthoring(
      {
        schema_version: 'feedbax.studio.task_bindings.v2',
        exposed_data: [],
        bindings: [
          {
            id: 'task:inputs->network:target',
            source_data_id: 'inputs',
            target_node_id: 'network',
            target_port: 'target',
            role: 'model_input',
            metadata: {},
          },
        ],
        metadata: {},
      },
      graph
    );

    expect(normalized?.bindings).toEqual([
      {
        id: 'task:inputs->network:input',
        source_data_id: 'inputs',
        target_node_id: 'network',
        target_port: 'input',
        role: 'model_input',
        metadata: {},
      },
    ]);
  });
});
