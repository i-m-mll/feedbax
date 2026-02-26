/**
 * CDE subgraph templates: preset internal node/edge layouts for known CDE architectures.
 *
 * Each template captures the DAG within one timestep of the CDE step function.
 * Nodes and edges are in React Flow format (type: 'component') so they render
 * directly inside the SubgraphNode nested preview canvas.
 *
 * Input ports and output ports declare what the overall subgraph node exposes
 * to the parent canvas.
 */

import type { Node, Edge } from '@xyflow/react';
import type { GraphNodeData, ParamValue } from '@/types/graph';

export interface CdeSubgraphTemplate {
  /** Human-readable name shown in the component palette. */
  name: string;
  /** Palette category label. */
  category: string;
  /** Short description shown in the palette card. */
  description: string;
  /** Lucide icon name to use in the palette card. */
  icon: string;
  /** Ports the subgraph node exposes on the parent canvas — inputs. */
  inputPorts: string[];
  /** Ports the subgraph node exposes on the parent canvas — outputs. */
  outputPorts: string[];
  /** Internal React Flow nodes for the preview canvas. */
  nodes: Node<GraphNodeData>[];
  /** Internal React Flow edges for the preview canvas. */
  edges: Edge[];
}

// ---------------------------------------------------------------------------
// Shared layout constants
// ---------------------------------------------------------------------------

const COL0 = 20;   // obs / obs_prev inputs
const COL1 = 160;  // Subtract (dX)
const COL2 = 320;  // VectorField / reshape / h_prev
const COL3 = 480;  // MatMul
const COL4 = 640;  // Sum → h_new
const COL5 = 800;  // Linear → action
const COL6 = 960;  // Sigmoid → output

const ROW_TOP = 40;
const ROW_MID = 160;
const ROW_BOT = 280;

function makeNode(
  id: string,
  type: string,
  label: string,
  x: number,
  y: number,
  inputPorts: string[],
  outputPorts: string[],
  params: Record<string, ParamValue> = {}
): Node<GraphNodeData> {
  return {
    id,
    type: 'component',
    position: { x, y },
    data: {
      label,
      spec: {
        type,
        params,
        input_ports: inputPorts,
        output_ports: outputPorts,
      },
      collapsed: false,
    },
  };
}

function makeEdge(
  sourceId: string,
  sourceHandle: string,
  targetId: string,
  targetHandle: string
): Edge {
  return {
    id: `${sourceId}:${sourceHandle}->${targetId}:${targetHandle}`,
    source: sourceId,
    target: targetId,
    sourceHandle,
    targetHandle,
    type: 'routed',
  };
}

// ---------------------------------------------------------------------------
// Template: standardCDE
// VectorField(MLP) → M, Subtract(obs,obs_prev) → dX, MatMul(M,dX) → h_new,
// Linear → Sigmoid → action
// No decay, no feedback gate.
// ---------------------------------------------------------------------------

export const standardCDE: CdeSubgraphTemplate = {
  name: 'CDE Standard',
  category: 'CDE Controllers',
  description: 'Basic CDE step: VectorField × dX → h_new, Linear → Sigmoid → action.',
  icon: 'BrainCircuit',
  inputPorts: ['obs', 'obs_prev', 'h_prev'],
  outputPorts: ['h_new', 'action'],
  nodes: [
    makeNode('obs_in',     'Input',    'obs',      COL0, ROW_TOP, [],           ['obs']),
    makeNode('obs_prev_in','Input',    'obs_prev', COL0, ROW_MID, [],           ['obs_prev']),
    makeNode('h_prev_in',  'Input',    'h_prev',   COL0, ROW_BOT, [],           ['h_prev']),

    makeNode('subtract',   'Subtract', 'dX',       COL1, ROW_TOP, ['a','b'],    ['out']),
    makeNode('vf',         'MLP',      'VectorField', COL2, ROW_BOT, ['input'], ['output'],
      { hidden_size: 128, activation: 'tanh' }),
    makeNode('reshape',    'Reshape',  'Reshape',  COL3, ROW_BOT, ['input'],    ['output']),
    makeNode('matmul',     'MatMul',   'MatMul',   COL3, ROW_TOP, ['a','b'],    ['out']),
    makeNode('sum_h',      'Sum',      'Sum→h_new', COL4, ROW_MID, ['a','b'],   ['out']),

    makeNode('linear',     'Linear',   'Linear',   COL5, ROW_MID, ['input'],   ['output']),
    makeNode('sigmoid',    'Sigmoid',  'Sigmoid',  COL6, ROW_MID, ['input'],   ['output']),
  ],
  edges: [
    // dX = obs - obs_prev
    makeEdge('obs_in',      'obs',     'subtract',  'a'),
    makeEdge('obs_prev_in', 'obs_prev','subtract',  'b'),
    // VectorField: h_prev → MLP → Reshape → M
    makeEdge('h_prev_in',  'h_prev',  'vf',        'input'),
    makeEdge('vf',         'output',  'reshape',   'input'),
    // MatMul: M × dX
    makeEdge('reshape',    'output',  'matmul',    'a'),
    makeEdge('subtract',   'out',     'matmul',    'b'),
    // Sum: h_prev + cde_update → h_new
    makeEdge('h_prev_in',  'h_prev',  'sum_h',     'a'),
    makeEdge('matmul',     'out',     'sum_h',     'b'),
    // action path
    makeEdge('sum_h',      'out',     'linear',    'input'),
    makeEdge('linear',     'output',  'sigmoid',   'input'),
  ],
};

// ---------------------------------------------------------------------------
// Template: cdeWithDecay
// Standard CDE + exponential decay floor on h_prev before summation.
// h_new = h_prev + cde_update + (-decay * h_prev)
// ---------------------------------------------------------------------------

export const cdeWithDecay: CdeSubgraphTemplate = {
  name: 'CDE + Decay',
  category: 'CDE Controllers',
  description: 'CDE with exponential decay: h_new = h_prev + M×dX − decay×h_prev.',
  icon: 'TrendingUp',
  inputPorts: ['obs', 'obs_prev', 'h_prev'],
  outputPorts: ['h_new', 'action'],
  nodes: [
    makeNode('obs_in',     'Input',    'obs',       COL0, ROW_TOP, [],           ['obs']),
    makeNode('obs_prev_in','Input',    'obs_prev',  COL0, ROW_MID, [],           ['obs_prev']),
    makeNode('h_prev_in',  'Input',    'h_prev',    COL0, ROW_BOT, [],           ['h_prev']),

    makeNode('subtract',   'Subtract', 'dX',        COL1, ROW_TOP, ['a','b'],    ['out']),
    makeNode('vf',         'MLP',      'VectorField', COL2, ROW_BOT, ['input'], ['output'],
      { hidden_size: 128, activation: 'tanh' }),
    makeNode('reshape',    'Reshape',  'Reshape',   COL3, ROW_BOT, ['input'],    ['output']),
    makeNode('matmul',     'MatMul',   'MatMul',    COL3, ROW_TOP, ['a','b'],    ['out']),
    makeNode('decay',      'Scale',    'decay×h',   COL2, ROW_MID, ['input'],   ['output'],
      { scale: -0.1 }),
    makeNode('sum_h',      'Sum',      'Sum→h_new', COL4, ROW_MID, ['a','b','c'], ['out']),

    makeNode('linear',     'Linear',   'Linear',    COL5, ROW_MID, ['input'],   ['output']),
    makeNode('sigmoid',    'Sigmoid',  'Sigmoid',   COL6, ROW_MID, ['input'],   ['output']),
  ],
  edges: [
    makeEdge('obs_in',      'obs',      'subtract', 'a'),
    makeEdge('obs_prev_in', 'obs_prev', 'subtract', 'b'),
    makeEdge('h_prev_in',  'h_prev',   'vf',        'input'),
    makeEdge('vf',         'output',   'reshape',   'input'),
    makeEdge('reshape',    'output',   'matmul',    'a'),
    makeEdge('subtract',   'out',      'matmul',    'b'),
    // decay term
    makeEdge('h_prev_in',  'h_prev',   'decay',     'input'),
    // sum: h_prev + cde_update + decay_term
    makeEdge('h_prev_in',  'h_prev',   'sum_h',     'a'),
    makeEdge('matmul',     'out',      'sum_h',     'b'),
    makeEdge('decay',      'output',   'sum_h',     'c'),
    // action
    makeEdge('sum_h',      'out',      'linear',    'input'),
    makeEdge('linear',     'output',   'sigmoid',   'input'),
  ],
};

// ---------------------------------------------------------------------------
// Template: cdeAntiNF
// Standard CDE + Anti-NF (Neural Feedback) gate via GRUCell.
// No fixed decay; gated feedback modulates h update.
// ---------------------------------------------------------------------------

const COL_GATE = 340;

export const cdeAntiNF: CdeSubgraphTemplate = {
  name: 'CDE + Anti-NF',
  category: 'CDE Controllers',
  description: 'CDE with Anti-NF gate: GRU(obs, −h) × α provides gated feedback correction.',
  icon: 'BrainCog',
  inputPorts: ['obs', 'obs_prev', 'h_prev'],
  outputPorts: ['h_new', 'action'],
  nodes: [
    makeNode('obs_in',     'Input',    'obs',        COL0, ROW_TOP,  [],           ['obs']),
    makeNode('obs_prev_in','Input',    'obs_prev',   COL0, ROW_MID,  [],           ['obs_prev']),
    makeNode('h_prev_in',  'Input',    'h_prev',     COL0, ROW_BOT,  [],           ['h_prev']),

    makeNode('subtract',   'Subtract', 'dX',         COL1, ROW_TOP,  ['a','b'],    ['out']),
    makeNode('negate_h',   'Scale',    '−h_prev',    COL1, ROW_BOT,  ['input'],    ['output'],
      { scale: -1.0 }),
    makeNode('vf',         'MLP',      'VectorField', COL2, ROW_MID, ['input'],   ['output'],
      { hidden_size: 128, activation: 'tanh' }),
    makeNode('reshape',    'Reshape',  'Reshape',    COL3, ROW_TOP,  ['input'],    ['output']),
    makeNode('matmul',     'MatMul',   'MatMul',     COL3, ROW_TOP+80, ['a','b'], ['out']),
    makeNode('gru_gate',   'GRUCell',  'Anti-NF GRU', COL_GATE, ROW_BOT, ['input','hx'], ['output'],
      { hidden_size: 64 }),
    makeNode('alpha',      'Scale',    'α gate',     COL4, ROW_BOT,  ['input'],    ['output'],
      { scale: 0.1 }),
    makeNode('sum_h',      'Sum',      'Sum→h_new',  COL4+80, ROW_MID, ['a','b','c'], ['out']),

    makeNode('linear',     'Linear',   'Linear',     COL5, ROW_MID,  ['input'],   ['output']),
    makeNode('sigmoid',    'Sigmoid',  'Sigmoid',    COL6, ROW_MID,  ['input'],   ['output']),
  ],
  edges: [
    makeEdge('obs_in',      'obs',      'subtract',  'a'),
    makeEdge('obs_prev_in', 'obs_prev', 'subtract',  'b'),
    makeEdge('h_prev_in',  'h_prev',   'vf',         'input'),
    makeEdge('vf',         'output',   'reshape',    'input'),
    makeEdge('reshape',    'output',   'matmul',     'a'),
    makeEdge('subtract',   'out',      'matmul',     'b'),
    // Anti-NF gate: GRU(obs, -h_prev)
    makeEdge('h_prev_in',  'h_prev',   'negate_h',  'input'),
    makeEdge('obs_in',     'obs',      'gru_gate',  'input'),
    makeEdge('negate_h',   'output',   'gru_gate',  'hx'),
    makeEdge('gru_gate',   'output',   'alpha',     'input'),
    // sum: h_prev + cde_update + gated_feedback
    makeEdge('h_prev_in',  'h_prev',   'sum_h',     'a'),
    makeEdge('matmul',     'out',      'sum_h',     'b'),
    makeEdge('alpha',      'output',   'sum_h',     'c'),
    // action
    makeEdge('sum_h',      'out',      'linear',    'input'),
    makeEdge('linear',     'output',   'sigmoid',   'input'),
  ],
};

// ---------------------------------------------------------------------------
// Template: cdeHybridV9b
// Full hybrid — fixed decay floor + Anti-NF gate (production architecture).
// h_new = h_prev + M×dX + (-decay×h_prev) + GRU(obs,-h_prev)×α
// ---------------------------------------------------------------------------

export const cdeHybridV9b: CdeSubgraphTemplate = {
  name: 'CDE Hybrid v9b',
  category: 'CDE Controllers',
  description: 'Production architecture: fixed-decay floor + Anti-NF gate (v9b hybrid).',
  icon: 'Sparkles',
  inputPorts: ['obs', 'obs_prev', 'h_prev'],
  outputPorts: ['h_new', 'action'],
  nodes: [
    // Inputs
    makeNode('obs_in',     'Input',    'obs',        COL0,       ROW_TOP,      [],           ['obs']),
    makeNode('obs_prev_in','Input',    'obs_prev',   COL0,       ROW_MID,      [],           ['obs_prev']),
    makeNode('h_prev_in',  'Input',    'h_prev',     COL0,       ROW_BOT,      [],           ['h_prev']),

    // dX path
    makeNode('subtract',   'Subtract', 'dX',         COL1,       ROW_TOP,      ['a','b'],    ['out']),

    // VectorField path
    makeNode('vf',         'MLP',      'VectorField', COL1,      ROW_BOT,      ['input'],   ['output'],
      { hidden_size: 128, activation: 'tanh' }),
    makeNode('reshape',    'Reshape',  'Reshape',    COL2,       ROW_BOT,      ['input'],    ['output']),
    makeNode('matmul',     'MatMul',   'MatMul',     COL2,       ROW_TOP,      ['a','b'],    ['out']),

    // Decay path
    makeNode('decay',      'Scale',    '−decay×h',   COL2,       ROW_MID,      ['input'],   ['output'],
      { scale: -0.1 }),

    // Anti-NF gate
    makeNode('negate_h',   'Scale',    '−h_prev',    COL2,       ROW_BOT+80,   ['input'],   ['output'],
      { scale: -1.0 }),
    makeNode('gru_gate',   'GRUCell',  'Anti-NF GRU', COL3,     ROW_BOT+80,   ['input','hx'], ['output'],
      { hidden_size: 64 }),
    makeNode('alpha',      'Scale',    'α gate',     COL4,       ROW_BOT+80,   ['input'],   ['output'],
      { scale: 0.1 }),

    // Sum → h_new
    makeNode('sum_h',      'Sum',      'Sum→h_new',  COL4,       ROW_MID,      ['a','b','c','d'], ['out']),

    // Output path
    makeNode('linear',     'Linear',   'Linear',     COL5,       ROW_MID,      ['input'],   ['output']),
    makeNode('sigmoid',    'Sigmoid',  'Sigmoid',    COL6,       ROW_MID,      ['input'],   ['output']),
  ],
  edges: [
    // dX
    makeEdge('obs_in',      'obs',      'subtract',  'a'),
    makeEdge('obs_prev_in', 'obs_prev', 'subtract',  'b'),
    // VectorField
    makeEdge('h_prev_in',  'h_prev',   'vf',        'input'),
    makeEdge('vf',         'output',   'reshape',   'input'),
    makeEdge('reshape',    'output',   'matmul',    'a'),
    makeEdge('subtract',   'out',      'matmul',    'b'),
    // Decay
    makeEdge('h_prev_in',  'h_prev',   'decay',     'input'),
    // Anti-NF gate
    makeEdge('h_prev_in',  'h_prev',   'negate_h',  'input'),
    makeEdge('obs_in',     'obs',      'gru_gate',  'input'),
    makeEdge('negate_h',   'output',   'gru_gate',  'hx'),
    makeEdge('gru_gate',   'output',   'alpha',     'input'),
    // Sum: h_prev + cde_update + decay_term + gated_feedback
    makeEdge('h_prev_in',  'h_prev',   'sum_h',     'a'),
    makeEdge('matmul',     'out',      'sum_h',     'b'),
    makeEdge('decay',      'output',   'sum_h',     'c'),
    makeEdge('alpha',      'output',   'sum_h',     'd'),
    // Action
    makeEdge('sum_h',      'out',      'linear',    'input'),
    makeEdge('linear',     'output',   'sigmoid',   'input'),
  ],
};

// ---------------------------------------------------------------------------
// All templates as an ordered list for the palette
// ---------------------------------------------------------------------------

export const CDE_TEMPLATES: CdeSubgraphTemplate[] = [
  standardCDE,
  cdeWithDecay,
  cdeAntiNF,
  cdeHybridV9b,
];
