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
      input_ports: ['input', 'feedback'],
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
    expect(normalized.subgraphs?.network.wires).toContainEqual(
      expect.objectContaining({
        source_node: 'cell',
        source_port: 'hidden',
        target_node: 'cell',
        target_port: 'hidden',
        temporality: 'recurrent',
        recurrent_initializer: expect.objectContaining({ kind: 'zeros', shape: [100] }),
      })
    );
    expect(normalized.subgraphs?.network.output_bindings.hidden).toEqual(['cell', 'hidden']);
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

  it('generates explicit SISU input routing for multiplicative Network authoring', () => {
    const graph: GraphSpec = {
      nodes: {
        network: {
          type: 'SimpleStagedNetwork',
          params: {
            input_size: 4,
            hidden_size: 3,
            output_size: 2,
            sisu_gating: 'multiplicative',
            sisu_alpha: [0.1, -0.2, 0.3],
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
    const subgraph = normalized.subgraphs!.network;

    expect(normalized.nodes.network.input_ports).toEqual(['input', 'feedback', 'sisu']);
    expect(subgraph.nodes.sisu_modulator).toMatchObject({
      type: 'ElementwiseAffineModulator',
      params: {
        signal_shape: [3],
        gain_init: [0.1, -0.2, 0.3],
      },
    });
    expect(subgraph.nodes.cell.params.input_size).toBe(3);
    expect(subgraph.input_bindings.sisu).toEqual(['sisu_modulator', 'modulator']);
    expect(subgraph.output_bindings.hidden).toEqual(['sisu_modulator', 'output']);
    expect(subgraph.wires).toContainEqual(
      expect.objectContaining({
        source_node: 'sisu_modulator',
        source_port: 'output',
        target_node: 'cell',
        target_port: 'hidden',
        temporality: 'recurrent',
      })
    );
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

  it('retargets saved mux task bindings after network subgraph normalization', () => {
    const graph = normalizeGraphForStudioAuthoring(runtimeGraph).subgraphs!.network;
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

  it('flattens legacy Network model wrapper and upgrades hidden recurrence', () => {
    const normalizedRuntime = normalizeGraphAuthoringTypes(runtimeGraph);
    const legacyInner = {
      ...normalizedRuntime.subgraphs!.network,
      wires: normalizedRuntime.subgraphs!.network.wires.map((wire) => ({
        ...wire,
        temporality: 'instant' as const,
        recurrent_initializer: null,
      })),
      output_ports: ['output'],
      output_bindings: { output: ['readout', 'output'] as [string, string] },
    };
    const graph: GraphSpec = {
      ...normalizedRuntime,
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
    const hiddenWire = subgraph.wires.find(
      (wire) =>
        wire.source_node === 'cell' &&
        wire.source_port === 'hidden' &&
        wire.target_node === 'cell' &&
        wire.target_port === 'hidden'
    );

    expect(subgraph.nodes.model).toBeUndefined();
    expect(subgraph.nodes.cell.type).toBe('GRU');
    expect(subgraph.output_bindings.hidden).toEqual(['cell', 'hidden']);
    expect(hiddenWire).toMatchObject({
      temporality: 'recurrent',
      recurrent_initializer: expect.objectContaining({ kind: 'zeros' }),
    });
  });

  it('marks legacy Network feedback cycle cuts recurrent', () => {
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
      temporality: 'recurrent',
      recurrent_initializer: {
        kind: 'zeros',
        scope: 'trial',
        source: 'state_initializer',
        state_slot: 'feedback',
      },
    });
  });
});
