import { afterEach, describe, expect, it, vi } from 'vitest';
import type { GraphSpec, GraphUIState } from '@/types/graph';

const LOCAL_PROJECTS_STORAGE_KEY = 'feedbax:studio-local-tabs';

function makeStorage(initial: Record<string, string> = {}): Storage {
  const values = new Map(Object.entries(initial));
  return {
    get length() {
      return values.size;
    },
    clear: vi.fn(() => values.clear()),
    getItem: vi.fn((key: string) => values.get(key) ?? null),
    key: vi.fn((index: number) => Array.from(values.keys())[index] ?? null),
    removeItem: vi.fn((key: string) => {
      values.delete(key);
    }),
    setItem: vi.fn((key: string, value: string) => {
      values.set(key, value);
    }),
  };
}

const savedTabsPayload = JSON.stringify({
  version: 1,
  activeTabId: 'local-tab',
  tabs: [
    {
      tabId: 'local-tab',
      label: 'Unsaved local tab',
      graphSnapshot: {
        graph: {
          nodes: {},
          wires: [],
          input_ports: [],
          output_ports: [],
          input_bindings: {},
          output_bindings: {},
          metadata: {
            name: 'Unsaved local tab',
            created_at: '2026-05-21T00:00:00Z',
            updated_at: '2026-05-21T00:00:00Z',
            version: '1.0.0',
          },
        },
        uiState: { viewport: { x: 0, y: 0, zoom: 1 }, node_states: {} },
        graphId: null,
        isDirty: true,
        lastSavedAt: null,
        graphStack: [],
        currentGraphLabel: 'Unsaved local tab',
        currentContext: 'top-level',
        edgeStyle: 'bezier',
        past: [],
        future: [],
        selectedTapId: null,
        selectedEdgeId: null,
        pendingStateMerge: null,
      },
      trainingSnapshot: {
        trainingSpec: {
          optimizer: { type: 'adam', params: { learning_rate: 0.001 } },
          loss: { type: 'Composite', label: 'loss', weight: 1, children: {} },
          n_batches: 10,
          batch_size: 4,
        },
        taskSpec: { type: 'ReachingTask', params: {} },
        selectedLossPath: null,
        lossValidationErrors: [],
        highlightedProbeSelector: null,
      },
      analysisSnapshot: { pages: [], activePageId: null },
      workspaceSnapshot: null,
    },
  ],
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('projectsStore local restore state', () => {
  it('marks restored local tabs so last-project autoload can stay idle', async () => {
    vi.resetModules();
    vi.stubGlobal(
      'window',
      { localStorage: makeStorage({ [LOCAL_PROJECTS_STORAGE_KEY]: savedTabsPayload }) },
    );
    vi.stubGlobal('crypto', { randomUUID: () => 'generated-tab' });

    const { useProjectsStore } = await import('@/stores/projectsStore');

    expect(useProjectsStore.getState().hasRestoredLocalTabs).toBe(true);
    expect(useProjectsStore.getState().activeTabId).toBe('local-tab');
  });

  it('does not mark fresh sessions as restored local tabs', async () => {
    vi.resetModules();
    vi.stubGlobal('window', { localStorage: makeStorage() });
    vi.stubGlobal('crypto', { randomUUID: () => 'generated-tab' });

    const { useProjectsStore } = await import('@/stores/projectsStore');

    expect(useProjectsStore.getState().hasRestoredLocalTabs).toBe(false);
    expect(useProjectsStore.getState().tabs).toHaveLength(1);
  });

  it('normalizes restored local tab aliases without synthesizing network subgraphs', async () => {
    const runtimePayload = JSON.parse(savedTabsPayload);
    runtimePayload.tabs[0].graphSnapshot.graph.nodes = {
      network: {
        type: 'SimpleStagedNetwork',
        params: { input_size: 4, hidden_size: 100, output_size: 2 },
        input_ports: ['target'],
        output_ports: ['output'],
      },
    };
    runtimePayload.tabs[0].graphSnapshot.graph.input_ports = ['target'];
    runtimePayload.tabs[0].graphSnapshot.graph.input_bindings = {
      target: ['network', 'target'],
    };

    vi.resetModules();
    vi.stubGlobal(
      'window',
      {
        localStorage: makeStorage({
          [LOCAL_PROJECTS_STORAGE_KEY]: JSON.stringify(runtimePayload),
        }),
      },
    );
    vi.stubGlobal('crypto', { randomUUID: () => 'generated-tab' });

    const { useProjectsStore } = await import('@/stores/projectsStore');
    const { useGraphStore } = await import('@/stores/graphStore');

    expect(useProjectsStore.getState().tabs[0].graphSnapshot.graph.nodes.network.type).toBe(
      'SimpleStagedNetwork'
    );
    expect(useGraphStore.getState().graph.nodes.network.type).toBe('SimpleStagedNetwork');
    expect(useGraphStore.getState().graph.subgraphs?.network).toBeUndefined();
    expect(useGraphStore.getState().graph.input_bindings).toEqual({
      input: ['network', 'target'],
    });
  });

  it('can replace the startup placeholder when autoloading a saved project', async () => {
    vi.resetModules();
    vi.stubGlobal('window', { localStorage: makeStorage() });
    vi.stubGlobal('crypto', { randomUUID: vi.fn(() => 'generated-tab') });

    const { useProjectsStore } = await import('@/stores/projectsStore');
    const { useGraphStore } = await import('@/stores/graphStore');

    expect(useProjectsStore.getState().tabs).toHaveLength(1);
    expect(useGraphStore.getState().currentGraphLabel).toBe('Reaching Task Model');

    useProjectsStore.getState().openProjectInTab(
      'movement-ramp-project',
      {
        nodes: {},
        wires: [],
        input_ports: [],
        output_ports: [],
        input_bindings: {},
        output_bindings: {},
        metadata: {
          name: 'RLRMP movement-ramp training runs',
          created_at: '2026-05-21T00:00:00Z',
          updated_at: '2026-05-21T00:00:00Z',
          version: '1.0.0',
        },
      },
      { viewport: { x: 0, y: 0, zoom: 1 }, node_states: {} },
      'RLRMP movement-ramp training runs',
      { pages: [], activePageId: null },
      null,
      { replaceActiveTab: true },
    );

    expect(useProjectsStore.getState().tabs).toHaveLength(1);
    expect(useProjectsStore.getState().tabs[0].graphSnapshot.graphId).toBe(
      'movement-ramp-project'
    );
    expect(useProjectsStore.getState().tabs[0].label).toBe(
      'RLRMP movement-ramp training runs'
    );
    expect(useGraphStore.getState().graphId).toBe('movement-ramp-project');
  });

  it('reopens a saved project inside its persisted subgraph path', async () => {
    vi.resetModules();
    vi.stubGlobal('window', { localStorage: makeStorage() });
    vi.stubGlobal('crypto', { randomUUID: vi.fn(() => 'generated-tab') });

    const { useProjectsStore } = await import('@/stores/projectsStore');
    const { useGraphStore } = await import('@/stores/graphStore');
    const { buildWorkspaceSnapshot } = await import('@/stores/workspaceStore');
    const { defaultTrainingSpec, defaultTaskSpec } = await import('@/stores/trainingStore');

    const graph: GraphSpec = {
      nodes: {
        network: {
          type: 'Network',
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
      metadata: {
        name: 'Nested project',
        created_at: '2026-06-11T00:00:00Z',
        updated_at: '2026-06-11T00:00:00Z',
        version: '1.0.0',
      },
      subgraphs: {
        network: {
          nodes: {
            inner: {
              type: 'Subgraph',
              params: {},
              input_ports: ['input'],
              output_ports: ['output'],
            },
          },
          wires: [],
          input_ports: ['input'],
          output_ports: ['output'],
          input_bindings: { input: ['inner', 'input'] },
          output_bindings: { output: ['inner', 'output'] },
          subgraphs: {
            inner: {
              nodes: {
                core: {
                  type: 'Gain',
                  params: { gain: 2 },
                  input_ports: ['input'],
                  output_ports: ['output'],
                },
              },
              wires: [],
              input_ports: ['input'],
              output_ports: ['output'],
              input_bindings: { input: ['core', 'input'] },
              output_bindings: { output: ['core', 'output'] },
            },
          },
        },
      },
    };
    const uiState: GraphUIState = {
      viewport: { x: 0, y: 0, zoom: 1 },
      node_states: {
        network: { position: { x: 0, y: 0 }, collapsed: false, selected: false },
      },
      subgraph_states: {
        network: {
          viewport: { x: 20, y: 30, zoom: 0.9 },
          node_states: {
            inner: { position: { x: 100, y: 50 }, collapsed: false, selected: false },
          },
          subgraph_states: {
            inner: {
              viewport: { x: 40, y: 60, zoom: 0.8 },
              node_states: {
                core: { position: { x: 240, y: 120 }, collapsed: false, selected: false },
              },
            },
          },
        },
      },
    };
    const workspace = buildWorkspaceSnapshot({
      workspace: null,
      graph,
      uiState,
      trainingSpec: defaultTrainingSpec,
      taskSpec: defaultTaskSpec,
      analysisSnapshot: { pages: [], activePageId: null },
      projectName: 'Nested project',
      graphStackPath: ['network', 'inner'],
    });

    useProjectsStore.getState().openProjectInTab(
      'nested-project',
      graph,
      uiState,
      'Nested project',
      { pages: [], activePageId: null },
      workspace,
      { replaceActiveTab: true },
    );

    expect(useProjectsStore.getState().tabs[0].label).toBe('Nested project');
    expect(useGraphStore.getState().graphStack.map((layer) => layer.childNodeId)).toEqual([
      'network',
      'inner',
    ]);
    expect(useGraphStore.getState().currentGraphLabel).toBe('inner');
    expect(useGraphStore.getState().graph.nodes.core.params).toEqual({ gain: 2 });
  });

  it('drops restored clean startup placeholders when a saved project tab exists', async () => {
    const payload = JSON.parse(savedTabsPayload);
    payload.activeTabId = 'placeholder';
    payload.tabs = [
      {
        tabId: 'placeholder',
        label: 'Reaching Task Model',
        graphSnapshot: {
          graph: {
            nodes: {},
            wires: [],
            input_ports: [],
            output_ports: [],
            input_bindings: {},
            output_bindings: {},
            metadata: {
              name: 'Reaching Task Model',
              created_at: '2026-05-21T00:00:00Z',
              updated_at: '2026-05-21T00:00:00Z',
              version: '1.0.0',
            },
          },
          uiState: { viewport: { x: 0, y: 0, zoom: 1 }, node_states: {} },
          graphId: null,
          isDirty: false,
          lastSavedAt: null,
          graphStack: [],
          currentGraphLabel: 'Reaching Task Model',
          currentContext: 'top-level',
          edgeStyle: 'bezier',
          past: [],
          future: [],
          selectedTapId: null,
          selectedEdgeId: null,
          pendingStateMerge: null,
        },
        trainingSnapshot: {
          trainingSpec: {
            optimizer: { type: 'adam', params: { learning_rate: 0.001 } },
            loss: { type: 'Composite', label: 'loss', weight: 1, children: {} },
            n_batches: 10,
            batch_size: 4,
          },
          taskSpec: { type: 'SimpleReaches', params: {} },
          selectedLossPath: null,
          lossValidationErrors: [],
          highlightedProbeSelector: null,
        },
        analysisSnapshot: { pages: [], activePageId: null },
        workspaceSnapshot: null,
      },
      {
        ...payload.tabs[0],
        tabId: 'movement-ramp',
        label: 'RLRMP movement-ramp training runs',
        graphSnapshot: {
          ...payload.tabs[0].graphSnapshot,
          graphId: 'movement-ramp-project',
          currentGraphLabel: 'RLRMP movement-ramp training runs',
          graph: {
            ...payload.tabs[0].graphSnapshot.graph,
            metadata: {
              ...payload.tabs[0].graphSnapshot.graph.metadata,
              name: 'RLRMP movement-ramp training runs',
            },
          },
        },
        trainingSnapshot: {
          ...payload.tabs[0].trainingSnapshot,
          taskSpec: { type: 'DelayedReaches', params: {} },
        },
      },
    ];

    vi.resetModules();
    vi.stubGlobal(
      'window',
      {
        localStorage: makeStorage({
          [LOCAL_PROJECTS_STORAGE_KEY]: JSON.stringify(payload),
        }),
      },
    );
    vi.stubGlobal('crypto', { randomUUID: () => 'generated-tab' });

    const { useProjectsStore } = await import('@/stores/projectsStore');

    expect(useProjectsStore.getState().tabs).toHaveLength(1);
    expect(useProjectsStore.getState().tabs[0].label).toBe(
      'RLRMP movement-ramp training runs'
    );
    expect(useProjectsStore.getState().activeTabId).toBe('movement-ramp');
  });

  it('does not schedule local persistence for selection, viewport, or history-only churn', async () => {
    vi.useFakeTimers();
    vi.resetModules();
    const storage = makeStorage();
    vi.stubGlobal('window', { localStorage: storage });
    vi.stubGlobal('crypto', { randomUUID: () => 'generated-tab' });

    await import('@/stores/projectsStore');
    const { useGraphStore } = await import('@/stores/graphStore');
    const { useTrainingStore } = await import('@/stores/trainingStore');
    const { useAnalysisStore } = await import('@/stores/analysisStore');
    const { useWorkspaceStore } = await import('@/stores/workspaceStore');

    useGraphStore.getState().setSelectedNode('input_mux');
    useTrainingStore.getState().setSelectedLossPath(['reach_loss', 'position']);
    useAnalysisStore.getState().setViewport({ x: 24, y: 36, zoom: 0.9 });
    useWorkspaceStore.getState().selectTopPaneEntity('graph_node:input_mux');
    useGraphStore.setState((state) => ({
      uiState: {
        ...state.uiState,
        viewport: { x: 120, y: 80, zoom: 0.75 },
        node_states: {
          ...state.uiState.node_states,
          input_mux: {
            ...state.uiState.node_states.input_mux,
            position: { x: 320, y: 240 },
          },
        },
      },
    }));
    useGraphStore.setState((state) => {
      const historyEntry = { graph: state.graph, uiState: state.uiState };
      return {
        past: [...state.past, historyEntry],
        future: [historyEntry],
      };
    });

    vi.advanceTimersByTime(300);

    expect(storage.setItem).not.toHaveBeenCalled();
  });

  it('omits undo and redo history from persisted local tab payloads', async () => {
    vi.resetModules();
    const storage = makeStorage();
    vi.stubGlobal('window', { localStorage: storage });
    vi.stubGlobal('crypto', { randomUUID: () => 'generated-tab' });

    const { persistLocalProjectTabs } = await import('@/stores/projectsStore');
    const { useGraphStore } = await import('@/stores/graphStore');

    useGraphStore.setState((state) => {
      const historyEntry = { graph: state.graph, uiState: state.uiState };
      return {
        past: [historyEntry],
        future: [historyEntry],
      };
    });

    expect(persistLocalProjectTabs()).toBe(true);
    const raw = storage.getItem(LOCAL_PROJECTS_STORAGE_KEY);
    expect(raw).not.toBeNull();
    const payload = JSON.parse(raw as string);
    const graphSnapshot = payload.tabs[0].graphSnapshot;

    expect(graphSnapshot).not.toHaveProperty('past');
    expect(graphSnapshot).not.toHaveProperty('future');
    expect(graphSnapshot).not.toHaveProperty('graphHistory');
  });
});
