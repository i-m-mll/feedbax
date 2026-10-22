// @vitest-environment jsdom

import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, expect, it } from 'vitest';
import { EdgeInspector } from '@/components/panels/ScenarioInspectorPanel';
import { buildScenarioEntityRegistry, graphEdgeEntityId } from '@/features/scenario/entities';
import { acausalConnectionId } from '@/features/domains/acausal';
import type { AcausalGraphSpec } from '@/types/graph';

afterEach(cleanup);

it('renders conserving endpoints without causal temporality or tap actions', () => {
  const graph: AcausalGraphSpec = {
    schema_id: 'feedbax.spec.acausal_graph',
    schema_version: 'feedbax.spec.acausal_graph.v1',
    physical_domain: 'translational',
    nodes: {
      mass: { type: 'Mass', params: {}, input_ports: [], output_ports: [] },
      spring: { type: 'Spring', params: {}, input_ports: [], output_ports: [] },
    },
    connections: [{ a: ['mass', 'flange'], b: ['spring', 'flange'] }],
    solver: { solver_type: 'euler', dt: 0.01 },
  };
  const registry = buildScenarioEntityRegistry({ scenario: null, graph });
  const id = graphEdgeEntityId(acausalConnectionId(graph.connections[0]));
  render(<EdgeInspector entity={registry.entities[id]} />);

  expect(screen.getByText('Conserving connection')).toBeDefined();
  expect(screen.getByText('mass.flange ↔ spring.flange')).toBeDefined();
  expect(screen.queryByText(/Instant|Recurrent t\+1/)).toBeNull();
  expect(screen.queryByRole('button', { name: /tap/i })).toBeNull();
});
