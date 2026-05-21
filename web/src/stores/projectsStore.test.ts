import { describe, expect, it, vi } from 'vitest';

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

  it('normalizes restored local tab graphs before exposing project state', async () => {
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
      'Network'
    );
    expect(useGraphStore.getState().graph.nodes.network.type).toBe('Network');
    expect(useGraphStore.getState().graph.subgraphs?.network.nodes.cell.type).toBe('GRU');
    expect(useGraphStore.getState().graph.input_bindings).toEqual({
      input: ['network', 'input'],
    });
  });
});
