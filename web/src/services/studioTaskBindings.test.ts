import { afterEach, beforeEach, expect, it } from 'vitest';
import { upsertRootTaskBinding } from '@/services/studioTaskBindings';
import { useGraphStore } from '@/stores/graphStore';
import {
  buildWorkspaceSnapshot,
  getTrainingScenario,
  useWorkspaceStore,
} from '@/stores/workspaceStore';
import { defaultTaskSpec, defaultTrainingSpec } from '@/stores/trainingStore';
import { StudioTaskBindingSpecSchema } from '@/generated/studioContracts';
import { ROOT_TASK_BINDING_MESSAGE } from '@/features/scenario/taskBindings';
import type { GraphSpec, GraphUIState } from '@/types/graph';

const initialGraphState = useGraphStore.getState();
const initialWorkspaceState = useWorkspaceStore.getState();

beforeEach(() => {
  const interior: GraphSpec = {
    schema_id: 'feedbax.spec.graph', schema_version: 'feedbax.spec.graph.v5',
    nodes: {
      gain: { type: 'Gain', params: { gain: 1 }, input_ports: ['input'], output_ports: ['output'] },
    },
    wires: [], input_ports: ['input'], output_ports: ['output'],
    input_bindings: { input: ['gain', 'input'] },
    output_bindings: { output: ['gain', 'output'] },
  };
  const graph: GraphSpec = {
    ...interior,
    nodes: {
      controller: { type: 'Subgraph', params: {}, input_ports: ['input'], output_ports: ['output'] },
    },
    input_ports: [], output_ports: [], input_bindings: {}, output_bindings: {},
    subgraphs: { controller: interior },
  };
  const uiState: GraphUIState = { viewport: { x: 0, y: 0, zoom: 1 }, node_states: {} };
  useGraphStore.getState().setComponentRegistry([{
    name: 'Subgraph', category: 'Structure', description: '',
    input_ports: ['input'], output_ports: ['output'], default_params: {},
    is_composite: true, domain: 'feedbax.domain.causal', interior_domain: 'feedbax.domain.causal',
  }]);
  useGraphStore.getState().hydrateGraph(graph, uiState, 'root-graph');
  useWorkspaceStore.getState().restoreWorkspace(buildWorkspaceSnapshot({
    workspace: null, graph, uiState,
    trainingSpec: defaultTrainingSpec, taskSpec: defaultTaskSpec,
    analysisSnapshot: null, projectName: 'Root bindings',
  }));
  const spec = getTrainingScenario(useWorkspaceStore.getState().workspace)?.task_binding_spec;
  if (!spec) throw new Error('Root binding fixture requires a task binding spec');
  useWorkspaceStore.getState().updateActiveScenarioTaskBindingSpec({ ...spec, bindings: [] });
});

afterEach(() => {
  useGraphStore.setState(initialGraphState, true);
  useWorkspaceStore.setState(initialWorkspaceState, true);
});

it('round-trips a binding to an explicitly exposed root composite input', () => {
  const outcome = upsertRootTaskBinding('inputs', 'controller', 'input');
  if (outcome.ok === false) throw new Error(outcome.message);
  expect(outcome.created).toBe(true);
  const stored = getTrainingScenario(useWorkspaceStore.getState().workspace)?.task_binding_spec;
  expect(stored).toEqual(outcome.spec);
  const restored = StudioTaskBindingSpecSchema.parse(JSON.parse(JSON.stringify(stored)));
  expect(restored.bindings).toEqual([{
    id: 'task:inputs->controller:input', source_data_id: 'inputs',
    target_node_id: 'controller', target_port: 'input', role: 'model_input', metadata: {},
  }]);
  expect(useGraphStore.getState().capturePersistedGraph().graph.subgraphs?.controller)
    .toMatchObject({ input_bindings: { input: ['gain', 'input'] } });
});

it('refuses a stale nested creation attempt before any document or selection mutation', () => {
  const createFromEarlierRootView = () => upsertRootTaskBinding('inputs', 'gain', 'input');
  useGraphStore.getState().enterSubgraph('controller');
  const graphState = useGraphStore.getState();
  const workspaceState = useWorkspaceStore.getState();

  expect(createFromEarlierRootView()).toEqual({ ok: false, message: ROOT_TASK_BINDING_MESSAGE });
  expect(useGraphStore.getState()).toBe(graphState);
  expect(useWorkspaceStore.getState()).toBe(workspaceState);
});
