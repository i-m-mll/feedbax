/**
 * Model graph specification for RLRMP projects.
 *
 * Represents the point-mass reaching model used in the RLRMP research:
 *   - SimpleStagedNetwork: GRU-based recurrent network (100-180 hidden units)
 *   - PointMass: 2D point-mass plant mechanics
 *   - FeedbackChannels: delayed, noisy proprioceptive feedback (position + velocity)
 *   - SimpleReaches: center-out reaching task
 *
 * This is a simplified visual representation of the model architecture.
 * The actual model is built from Python Equinox modules; this graph
 * shows the high-level component structure for visual reference.
 */

import type { GraphSpec, GraphUIState } from '@/types/graph';

/**
 * Create the RLRMP model graph with node positions for visual layout.
 *
 * @param projectName - Name to assign to the graph metadata.
 */
export function createRlrmpModelGraph(projectName: string): {
  graph: GraphSpec;
  uiState: GraphUIState;
} {
  const now = new Date().toISOString();

  const graph: GraphSpec = {
    nodes: {
      task: {
        type: 'SimpleReaches',
        params: {
          n_steps: 100,
          workspace: [
            [-0.15, -0.15],
            [0.15, 0.15],
          ],
          eval_n_directions: 8,
          eval_reach_length: 0.08,
          eval_grid_n: 1,
        },
        input_ports: [],
        output_ports: ['inputs', 'targets', 'inits', 'intervene'],
      },
      network: {
        type: 'SimpleStagedNetwork',
        params: {
          hidden_size: 100,
          input_size: 4,
          out_size: 2,
          hidden_type: 'GRUCell',
          out_nonlinearity: 'tanh',
          n_stages: 1,
        },
        input_ports: ['input', 'feedback'],
        output_ports: ['output', 'hidden'],
      },
      mechanics: {
        type: 'PointMass',
        params: {
          mass: 1.0,
          dt: 0.01,
          damping: 0.0,
        },
        input_ports: ['force'],
        output_ports: ['effector', 'state'],
      },
      feedback: {
        type: 'FeedbackChannels',
        params: {
          delay: 5,
          noise_std: 0.02,
          channels: ['position', 'velocity'],
        },
        input_ports: ['input'],
        output_ports: ['output'],
      },
    },
    wires: [
      {
        source_node: 'task',
        source_port: 'inputs',
        target_node: 'network',
        target_port: 'input',
      },
      {
        source_node: 'feedback',
        source_port: 'output',
        target_node: 'network',
        target_port: 'feedback',
      },
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
    ],
    input_ports: [],
    output_ports: ['effector'],
    input_bindings: {},
    output_bindings: {
      effector: ['mechanics', 'effector'],
    },
    taps: [],
    subgraphs: {},
    metadata: {
      name: projectName,
      description: 'Point-mass reaching model with GRU network, delayed noisy feedback',
      created_at: now,
      updated_at: now,
      version: '1.0.0',
    },
  };

  const uiState: GraphUIState = {
    viewport: { x: 0, y: 0, zoom: 1 },
    node_states: {
      task: { position: { x: 100, y: 200 }, collapsed: false, selected: false },
      network: { position: { x: 380, y: 200 }, collapsed: false, selected: false },
      mechanics: { position: { x: 660, y: 200 }, collapsed: false, selected: false },
      feedback: { position: { x: 520, y: 420 }, collapsed: false, selected: false },
    },
  };

  return { graph, uiState };
}
