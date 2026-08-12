// @vitest-environment jsdom

import { act, cleanup, render } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useAppShortcuts, FIT_VIEW_SHORTCUT_EVENT } from '@/hooks/useShortcuts';
import { ensureTaskBindingSpec } from '@/features/scenario/taskBindings';
import { useGraphStore } from '@/stores/graphStore';
import {
  buildWorkspaceSnapshot,
  getTrainingScenario,
  useWorkspaceStore,
} from '@/stores/workspaceStore';
import type { GraphSpec, GraphUIState } from '@/types/graph';
import type { TaskSpec, TrainingSpec } from '@/types/training';

vi.mock('@/hooks/useGraphs', () => ({
  useSaveGraph: () => ({
    mutateAsync: vi.fn(),
  }),
}));

function ShortcutHarness() {
  useAppShortcuts();
  return (
    <>
      <input aria-label="editable target" />
      <div
        aria-label="analysis shortcut target"
        data-studio-interaction-owner="analysis"
        tabIndex={-1}
      />
    </>
  );
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

describe('useAppShortcuts', () => {
  beforeEach(() => {
    useGraphStore.getState().hydrateGraph(graph, uiState);
    useWorkspaceStore.getState().setWorkspace(buildWorkspaceSnapshot({
      workspace: null,
      graph,
      uiState,
      trainingSpec,
      taskSpec,
      analysisSnapshot: null,
      projectName: 'Shortcut test',
    }));
  });

  afterEach(() => {
    cleanup();
    useGraphStore.getState().resetGraph();
    useWorkspaceStore.setState({
      workspace: null,
      lastTrainingExecutionPreparation: null,
      lastTrainingLocalRunResult: null,
      lastPipelineMaterializationResult: null,
    });
  });

  it('selects all graph nodes and clears selection from global shortcuts', () => {
    render(<ShortcutHarness />);

    act(() => {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'a', metaKey: true }));
    });
    expect(useGraphStore.getState().nodes.every((node) => node.selected)).toBe(true);

    act(() => {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    });
    expect(useGraphStore.getState().nodes.every((node) => !node.selected)).toBe(true);
  });

  it('ignores editable targets and emits zoom-to-fit events', () => {
    const { getByLabelText } = render(<ShortcutHarness />);
    const fitListener = vi.fn();
    window.addEventListener(FIT_VIEW_SHORTCUT_EVENT, fitListener);

    act(() => {
      getByLabelText('editable target').dispatchEvent(
        new KeyboardEvent('keydown', { key: 'a', metaKey: true, bubbles: true })
      );
    });
    expect(useGraphStore.getState().nodes.every((node) => !node.selected)).toBe(true);

    act(() => {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: '0', metaKey: true }));
    });
    expect(fitListener).toHaveBeenCalledTimes(1);

    window.removeEventListener(FIT_VIEW_SHORTCUT_EVENT, fitListener);
  });

  it('does not delete a selected model node from an analysis-owned key path', () => {
    useGraphStore.getState().setSelectedNode('a');
    const { getByLabelText } = render(<ShortcutHarness />);

    act(() => {
      getByLabelText('analysis shortcut target').dispatchEvent(
        new KeyboardEvent('keydown', { key: 'Delete', bubbles: true }),
      );
    });

    expect(useGraphStore.getState().graph.nodes.a).toBeDefined();
    expect(useGraphStore.getState().nodes.find((node) => node.id === 'a')?.selected).toBe(true);
  });

  it('does not delete a selected task binding from an analysis-owned key path', () => {
    const workspace = useWorkspaceStore.getState().workspace!;
    const trainingScenario = getTrainingScenario(workspace)!;
    const taskBindingSpec = ensureTaskBindingSpec(
      trainingScenario.task_binding_spec,
      graph,
      taskSpec,
    );
    const bindingId = 'binding:shortcut-test';
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
    const { getByLabelText } = render(<ShortcutHarness />);

    act(() => {
      getByLabelText('analysis shortcut target').dispatchEvent(
        new KeyboardEvent('keydown', { key: 'Backspace', bubbles: true }),
      );
    });

    expect(getTrainingScenario(useWorkspaceStore.getState().workspace)?.task_binding_spec?.bindings)
      .toEqual([expect.objectContaining({ id: bindingId })]);
  });
});
