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
  it('does not generate hidden network topology while normalizing legacy metadata', () => {
    const normalized = normalizeGraphAuthoringTypes(runtimeGraph);

    expect(normalized.nodes.network).toMatchObject({
      type: 'SimpleStagedNetwork',
      params: { input_size: 4, hidden_size: 100, output_size: 2 },
      input_ports: ['target', 'feedback'],
    });
    expect(normalized.wires[0].target_port).toBe('target');
    expect(normalized.input_ports).toEqual(['input']);
    expect(normalized.input_bindings).toEqual({ input: ['network', 'target'] });
    expect(normalized.subgraphs?.child.nodes.inner).toMatchObject({
      type: 'SimpleStagedNetwork',
      input_ports: ['target'],
    });
    expect(normalized.subgraphs?.child.input_bindings).toEqual({ input: ['inner', 'target'] });
    expect(normalized.subgraphs?.network).toBeUndefined();
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

  it('does not turn unsupported runtime params into generated graph ports', () => {
    const graph: GraphSpec = {
      nodes: {
        network: {
          type: 'SimpleStagedNetwork',
          params: {
            input_size: 4,
            hidden_size: 3,
            output_size: 2,
            unsupported_gating: 'multiplicative',
            unsupported_gain: [0.1, -0.2, 0.3],
          },
          input_ports: ['target', 'feedback'],
          output_ports: ['output', 'hidden'],
        },
      },
      wires: [],
      input_ports: ['target', 'feedback'],
      output_ports: ['output', 'hidden'],
      input_bindings: {
        target: ['network', 'target'],
        feedback: ['network', 'feedback'],
      },
      output_bindings: {
        output: ['network', 'output'],
        hidden: ['network', 'hidden'],
      },
    };

    const normalized = normalizeGraphAuthoringTypes(graph);

    expect(normalized.subgraphs?.network).toBeUndefined();
    expect(normalized.nodes.network.type).toBe('SimpleStagedNetwork');
    expect(normalized.nodes.network.params.modulator_input).toBeUndefined();
  });

  it('preserves task-data bindings when runtime networks are not wrapped', () => {
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
        id: 'task:inputs->network:target',
        source_data_id: 'inputs',
        target_node_id: 'network',
        target_port: 'target',
        role: 'model_input',
        metadata: {},
      },
    ]);
  });

  it('retargets saved mux task bindings after network subgraph normalization', () => {
    const graph: GraphSpec = {
      nodes: {
        input_mux: {
          type: 'Mux',
          params: { n_inputs: 2 },
          input_ports: ['in_0', 'in_1'],
          output_ports: ['output'],
        },
      },
      wires: [],
      input_ports: ['input', 'feedback'],
      output_ports: ['output'],
      input_bindings: {
        input: ['input_mux', 'in_0'],
        feedback: ['input_mux', 'in_1'],
      },
      output_bindings: {},
    };
    const normalized = normalizeTaskBindingSpecForStudioAuthoring(
      {
        schema_version: 'feedbax.studio.task_bindings.v2',
        exposed_data: [],
        bindings: [
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
      },
      graph
    );

    expect(normalized?.bindings).toEqual([
      {
        id: 'task:hold->input_mux:in_1',
        source_data_id: 'hold',
        target_node_id: 'input_mux',
        target_port: 'in_1',
        role: 'model_input',
        metadata: {},
      },
    ]);
  });

  it('preserves explicit legacy Network model wrappers without topology repair', () => {
    const legacyInner: GraphSpec = {
      nodes: {
        cell: {
          type: 'GRU',
          params: { input_size: 4, hidden_size: 100 },
          input_ports: ['input', 'hidden'],
          output_ports: ['output', 'hidden'],
        },
      },
      wires: [],
      input_ports: ['input'],
      output_ports: ['output'],
      input_bindings: { input: ['cell', 'input'] },
      output_bindings: { output: ['cell', 'output'] },
    };
    const graph: GraphSpec = {
      nodes: {
        network: {
          type: 'Network',
          params: {},
          input_ports: ['input', 'feedback'],
          output_ports: ['output'],
        },
      },
      wires: [],
      input_ports: ['input'],
      output_ports: ['output'],
      input_bindings: { input: ['network', 'input'] },
      output_bindings: { output: ['network', 'output'] },
      subgraphs: {
        network: {
          nodes: {
            model: {
              type: 'Subgraph',
              params: {},
              input_ports: ['input', 'feedback'],
              output_ports: ['output'],
            },
          },
          wires: [],
          input_ports: ['input', 'feedback'],
          output_ports: ['output'],
          input_bindings: {
            input: ['model', 'input'],
            feedback: ['model', 'feedback'],
          },
          output_bindings: {
            output: ['model', 'output'],
          },
          subgraphs: { model: legacyInner },
        },
      },
    };

    const normalized = normalizeGraphAuthoringTypes(graph);
    const subgraph = normalized.subgraphs!.network;

    expect(subgraph.nodes.model?.type).toBe('Subgraph');
    expect(subgraph.subgraphs?.model.nodes.cell.type).toBe('GRU');
    expect(subgraph.output_bindings).toEqual({ output: ['model', 'output'] });
  });

  it('does not rewrite legacy Network feedback cycle cuts', () => {
    const graph: GraphSpec = {
      nodes: {
        network: {
          type: 'Network',
          params: {},
          input_ports: ['input', 'feedback'],
          output_ports: ['output'],
        },
        mechanics: {
          type: 'PointMass',
          params: {},
          input_ports: ['force'],
          output_ports: ['effector'],
        },
        feedback: {
          type: 'FeedbackChannels',
          params: {},
          input_ports: ['input'],
          output_ports: ['output'],
        },
      },
      wires: [
        {
          source_node: 'network',
          source_port: 'output',
          target_node: 'mechanics',
          target_port: 'force',
        },
        {
          source_node: 'mechanics',
          source_port: 'effector',
          target_node: 'feedback',
          target_port: 'input',
        },
        {
          source_node: 'feedback',
          source_port: 'output',
          target_node: 'network',
          target_port: 'feedback',
        },
      ],
      input_ports: [],
      output_ports: [],
      input_bindings: {},
      output_bindings: {},
    };

    expect(normalizeGraphAuthoringTypes(graph).wires[2]).toMatchObject({
      source_node: 'feedback',
      target_node: 'network',
      target_port: 'feedback',
    });
    expect(normalizeGraphAuthoringTypes(graph).wires[2].temporality).toBeUndefined();
  });
});
