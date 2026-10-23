// @vitest-environment jsdom

import { act, cleanup, fireEvent, render } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AnalysisPanel } from '@/components/panels/AnalysisPanel';
import { ensureTaskBindingSpec } from '@/features/scenario/taskBindings';
import { useAppShortcuts } from '@/hooks/useShortcuts';
import { useAnalysisStore } from '@/stores/analysisStore';
import { useGraphStore } from '@/stores/graphStore';
import {
  buildWorkspaceSnapshot,
  getTrainingScenario,
  useWorkspaceStore,
} from '@/stores/workspaceStore';
import type { AnalysisClassDef } from '@/types/analysis';
import type { GraphSpec, GraphUIState } from '@/types/graph';
import type { TaskSpec, TrainingSpec } from '@/types/training';

vi.mock('@/components/analysis/AnalysisCanvas', () => ({
  AnalysisCanvas: () => <div aria-label="analysis canvas" />,
}));

vi.mock('@/components/panels/AnalysisBundlePanel', () => ({
  AnalysisBundlePanel: () => <div aria-label="analysis bundle panel" />,
}));

vi.mock('@/components/panels/AnalysisPageSettings', () => ({
  AnalysisPageSettings: () => <div aria-label="analysis page settings" />,
}));

vi.mock('@/api/analysisAPI', () => ({
  fetchAnalysisClasses: vi.fn(async () => []),
}));

vi.mock('@/hooks/useFigureGenerationStatus', () => ({
  useFigureGenerationStatus: vi.fn(),
}));

function AnalysisShortcutHarness() {
  useAppShortcuts();
  return <AnalysisPanel />;
}

const graph: GraphSpec = {
  nodes: {
    a: {
      type: 'Gain',
      params: {},
      input_ports: ['input'],
      output_ports: ['output'],
    },
    b: {
      type: 'Gain',
      params: {},
      input_ports: ['input'],
      output_ports: ['output'],
    },
  },
  wires: [],
  input_ports: [],
  output_ports: [],
  input_bindings: {},
  output_bindings: {},
};

const uiState: GraphUIState = {
  viewport: { x: 0, y: 0, zoom: 1 },
  node_states: {
    a: { position: { x: 0, y: 0 }, collapsed: false, selected: false },
    b: { position: { x: 160, y: 0 }, collapsed: false, selected: false },
  },
};

const trainingSpec: TrainingSpec = {
  optimizer: { type: 'adam', params: {} },
  loss: { type: 'Composite', label: 'loss', weight: 1, children: {} },
  n_batches: 1,
  batch_size: 1,
};

const taskSpec: TaskSpec = {
  type: 'ReachingTask',
  params: {},
};

const analysisClass: AnalysisClassDef = {
  name: 'ActivityPlot',
  description: 'Plot activity',
  category: 'Figures',
  inputPorts: ['series'],
  outputPorts: [],
  defaultParams: { title: 'Activity' },
  icon: 'LineChart',
};

function addSelectedAnalysisNode() {
  useAnalysisStore.getState().addAnalysisNode(analysisClass, { x: 240, y: 0 });
  const node = useAnalysisStore.getState().nodes.find((item) => item.type === 'analysis')!;
  useAnalysisStore.getState().setSelectedNode(node.id);
  return node;
}

function addSelectedTaskBinding() {
  const workspace = useWorkspaceStore.getState().workspace!;
  const trainingScenario = getTrainingScenario(workspace)!;
  const taskBindingSpec = ensureTaskBindingSpec(
    trainingScenario.task_binding_spec,
    graph,
    taskSpec,
  );
  const bindingId = 'binding:analysis-owner-boundary';
  trainingScenario.task_binding_spec = {
    ...taskBindingSpec,
    bindings: [
      {
        id: bindingId,
        source_data_id: taskBindingSpec.exposed_data[0].id,
        target_node_id: 'a',
        target_port: 'input',
        role: 'model_input',
        metadata: {},
      },
    ],
  };
  useWorkspaceStore.getState().selectTopPaneEntity(`task_binding:${bindingId}`);
  return bindingId;
}

describe('AnalysisPanel interaction ownership', () => {
  beforeEach(() => {
    useGraphStore.getState().hydrateGraph(graph, uiState);
    useWorkspaceStore.getState().setWorkspace(buildWorkspaceSnapshot({
      workspace: null,
      graph,
      uiState,
      trainingSpec,
      taskSpec,
      analysisSnapshot: null,
      projectName: 'Analysis interaction owner test',
    }));
    useAnalysisStore.getState().resetAnalysis();
    useAnalysisStore.getState().setAnalysisClasses([analysisClass]);
    useAnalysisStore.getState().addPage('Page 1');
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    useAnalysisStore.getState().resetAnalysis();
    useGraphStore.getState().resetGraph();
    useWorkspaceStore.setState({
      workspace: null,
      lastTrainingExecutionPreparation: null,
      lastPipelineMaterializationResult: null,
    });
  });

  it('keeps Delete on the real inspector Remove control out of model deletion', () => {
    useGraphStore.getState().setSelectedNode('a');
    addSelectedAnalysisNode();
    const { getByRole } = render(<AnalysisShortcutHarness />);
    let wasCanceled = false;

    act(() => {
      wasCanceled = !fireEvent.keyDown(
        getByRole('button', { name: 'Remove' }),
        { key: 'Delete' },
      );
    });

    expect(wasCanceled).toBe(true);
    expect(useGraphStore.getState().graph.nodes.a).toBeDefined();
    expect(useGraphStore.getState().nodes.find((node) => node.id === 'a')?.selected).toBe(true);
  });

  it('keeps Backspace on the real page control out of task-binding deletion', () => {
    const bindingId = addSelectedTaskBinding();
    const { getByTitle } = render(<AnalysisShortcutHarness />);
    let wasCanceled = false;

    act(() => {
      wasCanceled = !fireEvent.keyDown(
        getByTitle('Close page'),
        { key: 'Backspace' },
      );
    });

    expect(wasCanceled).toBe(true);
    expect(getTrainingScenario(useWorkspaceStore.getState().workspace)?.task_binding_spec?.bindings)
      .toEqual([expect.objectContaining({ id: bindingId })]);
  });

  it('preserves explicit analysis deletion through the inspector Remove control', async () => {
    const node = addSelectedAnalysisNode();
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true);
    const user = userEvent.setup();
    const { getByRole } = render(<AnalysisShortcutHarness />);

    await user.click(getByRole('button', { name: 'Remove' }));

    expect(confirm).toHaveBeenCalledWith(
      'Delete this analysis node and its connected wires? This cannot currently be undone.',
    );
    expect(useAnalysisStore.getState().graphSpec?.nodes[node.id]).toBeUndefined();
  });

  it('keeps Backspace editable in the real page-name input without deleting the model', async () => {
    useGraphStore.getState().setSelectedNode('a');
    const user = userEvent.setup();
    const { getByText, getByRole } = render(<AnalysisShortcutHarness />);

    await user.dblClick(getByText('Page 1'));
    const input = getByRole('textbox');
    await user.clear(input);
    await user.type(input, 'Edited');
    await user.keyboard('{Backspace}');

    expect((input as HTMLInputElement).value).toBe('Edite');
    expect(useGraphStore.getState().graph.nodes.a).toBeDefined();
  });
});
