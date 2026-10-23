// @vitest-environment jsdom

import { act, cleanup, renderHook } from '@testing-library/react';
import { afterEach, expect, it } from 'vitest';
import { useRootGraph } from '@/hooks/useRootGraph';
import { useGraphStore } from '@/stores/graphStore';
import type { GraphSpec } from '@/types/graph';
import { isAcausalGraphSpec } from '@/types/graph';
import { AcausalGraphSpecSchema } from '@/generated/studioContracts';
import { currentDraftSpecHashesForScenario } from '@/utils/pipelineCollections';

const initialState = useGraphStore.getState();

afterEach(() => {
  cleanup();
  useGraphStore.setState(initialState, true);
});

it('keeps root identity across acausal navigation and updates it for an interior edit', () => {
  const graph: GraphSpec = {
    schema_id: 'feedbax.spec.graph',
    schema_version: 'feedbax.spec.graph.v5',
    nodes: {
      system: {
        type: 'AcausalSystem', params: {}, input_ports: [], output_ports: [],
      },
    },
    wires: [], input_ports: [], output_ports: [], input_bindings: {}, output_bindings: {},
    subgraphs: {
      system: {
        schema_id: 'feedbax.spec.acausal_graph',
        schema_version: 'feedbax.spec.acausal_graph.v1',
        physical_domain: 'translational',
        nodes: {
          mass: { type: 'Mass', params: { mass: 1 }, input_ports: [], output_ports: [] },
        },
        connections: [{ a: ['mass', 'left'], b: ['mass', 'right'] }],
        solver: { solver_type: 'euler', dt: 0.01 },
      },
    },
  };
  useGraphStore.getState().setComponentRegistry([{
    name: 'AcausalSystem', category: 'Mechanics', description: '',
    param_schema: [], input_ports: [], output_ports: [], default_params: {},
    is_composite: true, domain: 'feedbax.domain.causal',
    interior_domain: 'feedbax.domain.acausal',
  }]);
  useGraphStore.getState().hydrateGraph(
    graph, { viewport: { x: 0, y: 0, zoom: 1 }, node_states: {} }, 'root-graph'
  );
  const { result } = renderHook(() => useRootGraph());
  const hash = () => currentDraftSpecHashesForScenario(null, result.current).hashes.graph_spec;
  const initialHash = hash();

  act(() => useGraphStore.getState().enterSubgraph('system'));
  expect(useGraphStore.getState().currentContext).toBe('feedbax.domain.acausal');
  expect(result.current.schema_id).toBe('feedbax.spec.graph');
  expect(hash()).toBe(initialHash);

  act(() => useGraphStore.getState().updateNodeParams('mass', 'mass', 2));
  expect(result.current.subgraphs?.system.nodes.mass.params.mass).toBe(2);
  expect(result.current.subgraphs?.system).toMatchObject({
    connections: [{ a: ['mass', 'left'], b: ['mass', 'right'] }],
    solver: { solver_type: 'euler', dt: 0.01 },
  });
  expect(hash()).not.toBe(initialHash);
  const editedHash = hash();

  act(() => useGraphStore.getState().exitToBreadcrumb(0));
  expect(hash()).toBe(editedHash);
  act(() => useGraphStore.getState().enterSubgraph('system'));
  expect(useGraphStore.getState().graph.nodes.mass.params.mass).toBe(2);
  expect(hash()).toBe(editedHash);
  act(() => {
    useGraphStore.getState().selectAll();
    useGraphStore.getState().duplicateSelected();
    useGraphStore.getState().renameNode('mass2', 'load');
  });
  const duplicated = useGraphStore.getState().graph;
  if (!isAcausalGraphSpec(duplicated)) throw new Error('Expected the acausal interior');
  expect(duplicated.connections).toContainEqual({ a: ['load', 'left'], b: ['load', 'right'] });
  const renamedHash = hash();
  expect(renamedHash).not.toBe(editedHash);
  act(() => useGraphStore.getState().onNodesChange([{ type: 'remove', id: 'load' }]));
  const edited = useGraphStore.getState().graph;
  expect(AcausalGraphSpecSchema.safeParse(edited).success).toBe(true);
  if (!isAcausalGraphSpec(edited)) throw new Error('Expected the acausal interior');
  expect(edited.connections).toEqual([{ a: ['mass', 'left'], b: ['mass', 'right'] }]);
  expect(edited.nodes.load).toBeUndefined();
  expect(hash()).toBe(editedHash);
  act(() => useGraphStore.getState().exitToBreadcrumb(0));
  expect(hash()).toBe(editedHash);
});
