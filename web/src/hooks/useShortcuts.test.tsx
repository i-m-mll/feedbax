// @vitest-environment jsdom

import { act, cleanup, render } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useAppShortcuts, FIT_VIEW_SHORTCUT_EVENT } from '@/hooks/useShortcuts';
import { useGraphStore } from '@/stores/graphStore';
import type { GraphSpec, GraphUIState } from '@/types/graph';

vi.mock('@/hooks/useGraphs', () => ({
  useSaveGraph: () => ({
    mutateAsync: vi.fn(),
  }),
}));

function ShortcutHarness() {
  useAppShortcuts();
  return <input aria-label="editable target" />;
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

describe('useAppShortcuts', () => {
  beforeEach(() => {
    useGraphStore.getState().hydrateGraph(graph, uiState);
  });

  afterEach(() => {
    cleanup();
    useGraphStore.getState().resetGraph();
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
});
