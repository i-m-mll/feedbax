// @vitest-environment jsdom

import { act, cleanup, render } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useAppShortcuts, FIT_VIEW_SHORTCUT_EVENT } from '@/hooks/useShortcuts';
import { ensureTaskBindingSpec, taskBindingId } from '@/features/scenario/taskBindings';
import { useGraphStore } from '@/stores/graphStore';
import {
  buildWorkspaceSnapshot,
  getStageByKind,
  getTrainingScenario,
  useWorkspaceStore,
} from '@/stores/workspaceStore';
import type { GraphSpec, GraphUIState } from '@/types/graph';
import type { TaskSpec, TrainingSpec } from '@/types/training';

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

  it('replays node deletion against its original training scenario after reassignment', () => {
    const mux = {
      type: 'Mux',
      params: { n_inputs: 2 },
      input_ports: ['in_0', 'in_1'],
      output_ports: ['output'],
    };
    const boundGraph = { ...graph, nodes: { ...graph.nodes, a: mux } };
    const editor = useGraphStore.getState();
    editor.hydrateGraph(boundGraph, uiState);

    const store = useWorkspaceStore.getState();
    const workspace = store.workspace!;
    const stage = getStageByKind(workspace, 'train')!;
    const scenario = getTrainingScenario(workspace)!;
    const spec = ensureTaskBindingSpec(scenario.task_binding_spec, boundGraph, taskSpec);
    const source = spec.exposed_data[0].id;
    const removed = {
      id: taskBindingId(source, 'a', 'in_2'),
      source_data_id: source,
      target_node_id: 'a',
      target_port: 'in_2',
      role: 'model_input',
      metadata: { label: 'Original binding' },
    } as const;
    const other = {
      ...removed,
      id: taskBindingId(source, 'b', 'input'),
      target_node_id: 'b',
      target_port: 'input',
      metadata: { label: 'Independent binding' },
    };
    const alternateId = 'alternate-training';
    const alternateSpec = { ...spec, bindings: [other] };
    const originalSpec = { ...spec, bindings: [removed, other] };
    store.setWorkspace({
      ...workspace,
      scenarios: {
        ...workspace.scenarios,
        [scenario.id]: { ...scenario, task_binding_spec: originalSpec },
        [alternateId]: { ...scenario, id: alternateId, task_binding_spec: alternateSpec },
      },
    });
    editor.updateNodeParams('a', 'n_inputs', 3, originalSpec);
    editor.toggleNodeCollapse('b');

    const getSpec = (scenarioId: string) =>
      useWorkspaceStore.getState().workspace?.scenarios[scenarioId]?.task_binding_spec;
    const press = (key: string, modifiers: KeyboardEventInit = {}) => {
      act(() => { window.dispatchEvent(new KeyboardEvent('keydown', { key, ...modifiers })); });
    };
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true);
    try {
      editor.setSelectedNode('a');
      render(<ShortcutHarness />);
      press('Delete');
      const edited = { ...other, metadata: { label: 'Edited while the node was deleted' } };
      act(() => {
        store.updateScenarioDraft(scenario.id, {
          task_binding_spec: { ...getSpec(scenario.id)!, bindings: [edited] },
        });
        store.updateStageDraft(stage.id, { scenario_id: alternateId });
      });

      press('z', { metaKey: true });
      expect(getSpec(scenario.id)?.bindings).toEqual([removed, edited]);
      expect(getSpec(alternateId)).toEqual(alternateSpec);

      press('z', { metaKey: true });
      expect(useGraphStore.getState().uiState.node_states.b.collapsed).toBe(false);
      expect(useGraphStore.getState().graph.nodes.a.input_ports).toContain('in_2');

      press('z', { metaKey: true, shiftKey: true });
      expect(useGraphStore.getState().uiState.node_states.b.collapsed).toBe(true);
      expect(getSpec(scenario.id)?.bindings).toEqual([removed, edited]);

      press('z', { metaKey: true, shiftKey: true });
      expect(getSpec(scenario.id)?.bindings).toEqual([edited]);
      expect(getSpec(alternateId)).toEqual(alternateSpec);
    } finally {
      confirm.mockRestore();
    }
  });
});
