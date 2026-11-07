import { afterEach, expect, it } from 'vitest';
import { ComponentDefinitionSchema, GraphSpecSchema } from '@/generated/studioContracts';
import { upsertRootTaskBinding } from '@/services/studioTaskBindings';
import { useGraphStore } from '@/stores/graphStore';
import { defaultTaskSpec, defaultTrainingSpec } from '@/stores/trainingStore';
import { buildWorkspaceSnapshot, getTrainingScenario, useWorkspaceStore } from '@/stores/workspaceStore';
import { visibleDynamicInputPorts } from './dynamicPorts';
import type { GraphSpec } from '@/types/graph';

const initialGraph = useGraphStore.getState();
const initialWorkspace = useWorkspaceStore.getState();

afterEach(() => {
  useGraphStore.setState(initialGraph, true);
  useWorkspaceStore.setState(initialWorkspace, true);
});

it('keeps an external component’s wired and task-bound ports through graph reload', () => {
  const component = ComponentDefinitionSchema.parse({
    name: 'example.Collector',
    category: 'External',
    description: '',
    input_ports: ['signal-7'],
    output_ports: ['value'],
    default_params: { slots: 1 },
    dynamic_port_policy: {
      count_param: 'slots',
      count_mode: 'integer',
      direction: 'input',
      fixed_output_ports: ['value'],
      generated_name_template: 'signal-{index}',
      generated_index_origin: 7,
      dynamic_port_type: { dtype: 'vector' },
      auto_resize_minimum: 1,
    },
  });
  const graph: GraphSpec = {
    nodes: { source: { type: 'feedbax.Constant', params: {}, input_ports: [], output_ports: ['output'] } },
    wires: [],
    input_ports: [],
    output_ports: [],
    input_bindings: {},
    output_bindings: {},
  };
  const uiState = { viewport: { x: 0, y: 0, zoom: 1 }, node_states: {} };
  const editor = useGraphStore.getState();
  editor.setComponentRegistry([component]);
  editor.hydrateGraph(graph, uiState, 'external');
  useWorkspaceStore.getState().restoreWorkspace(buildWorkspaceSnapshot({
    workspace: null, graph, uiState, trainingSpec: defaultTrainingSpec, taskSpec: defaultTaskSpec,
    analysisSnapshot: null, projectName: 'External policy',
  }));
  editor.addNodeFromComponent(component, { x: 200, y: 100 });
  const target = Object.keys(useGraphStore.getState().graph.nodes).find((id) => id !== 'source')!;
  editor.onConnect({ source: 'source', sourceHandle: 'output', target, targetHandle: 'signal-7' });
  const state = useGraphStore.getState();
  expect(visibleDynamicInputPorts(state.graph, target, state._componentRegistry)?.nextPort)
    .toBe('signal-8');

  const bound = upsertRootTaskBinding('inputs', target, 'signal-8');
  expect(bound.ok).toBe(true);
  const snapshot = editor.capturePersistedGraph();
  editor.hydrateGraph(GraphSpecSchema.parse(JSON.parse(JSON.stringify(snapshot.graph))), snapshot.uiState, 'external');
  expect(useGraphStore.getState().graph.nodes[target]).toMatchObject({
    params: { slots: 2 }, input_ports: ['signal-7', 'signal-8'],
  });
  const bindings = getTrainingScenario(useWorkspaceStore.getState().workspace)!.task_binding_spec!;
  const edge = useGraphStore.getState().edges[0];
  editor.onEdgesChange([{ type: 'remove', id: edge.id }], bindings);
  expect(useGraphStore.getState().graph.nodes[target].input_ports).toEqual(['signal-7', 'signal-8']);
  editor.resizeDynamicInputs({ ...bindings, bindings: [] });
  expect(useGraphStore.getState().graph.nodes[target].input_ports).toEqual(['signal-7']);
});

it('edits configured port names and preserves canonical revision through save and reload', () => {
  const component = ComponentDefinitionSchema.parse({
    name: 'example.NamedAdapter', revision: 3, supported_revisions: [1, 2, 3],
    category: 'External', description: '',
    default_params: { input_port: 'value', output_port: 'result' },
    input_ports: ['value'], output_ports: ['result'],
    dynamic_port_policy: {
      count_mode: 'names', input_name_params: ['input_port'], output_name_params: ['output_port'],
    },
  });
  const editor = useGraphStore.getState();
  editor.setComponentRegistry([component]);
  editor.hydrateGraph({ nodes: {}, wires: [], input_ports: [], output_ports: [], input_bindings: {}, output_bindings: {} },
    { viewport: { x: 0, y: 0, zoom: 1 }, node_states: {} }, 'named');
  editor.addNodeFromComponent(component, { x: 0, y: 0 });
  editor.updateNodeParams('namedAdapter', 'input_port', 'signal');
  editor.updateNodeParams('namedAdapter', 'output_port', 'response');
  const saved = editor.capturePersistedGraph();
  editor.hydrateGraph(GraphSpecSchema.parse(JSON.parse(JSON.stringify(saved.graph))), saved.uiState, 'named');
  expect(useGraphStore.getState().graph.nodes.namedAdapter).toMatchObject({
    type: 'example.NamedAdapter', revision: 3,
    params: { input_port: 'signal', output_port: 'response' },
    input_ports: ['signal'], output_ports: ['response'],
  });
});
