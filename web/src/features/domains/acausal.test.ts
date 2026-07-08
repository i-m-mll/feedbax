import { describe, expect, it } from 'vitest';
import type { ComponentDefinition } from '@/types/components';
import type { AcausalGraphSpec, DomainCompileReport } from '@/generated/studioContracts';
import {
  acausalEdgesFromGraph,
  compileStatusForReport,
  evaluateAcausalConnection,
  rollupCompileStatus,
  stableGraphHash,
} from './acausal';

const registry = new Map<string, ComponentDefinition>([
  [
    'Mass',
    {
      name: 'Mass',
      category: 'Mechanics',
      description: 'Mass',
      input_ports: ['flange'],
      output_ports: [],
      default_params: {},
      port_types: {
        inputs: {
          flange: { kind: 'conserving', physical_domain: 'translational' },
        },
      },
    },
  ],
  [
    'ActuationInput',
    {
      name: 'ActuationInput',
      category: 'Mechanics',
      description: 'Actuation adapter',
      input_ports: ['signal'],
      output_ports: ['flange'],
      default_params: {},
      port_types: {
        inputs: { signal: { kind: 'signal' } },
        outputs: { flange: { kind: 'conserving', physical_domain: 'translational' } },
      },
    },
  ],
]);

function acausalGraph(): AcausalGraphSpec {
  return {
    schema_id: 'feedbax.spec.acausal_graph',
    schema_version: 'feedbax.spec.acausal_graph.v1',
    physical_domain: 'translational',
    solver: { solver_type: 'implicit_euler', dt: 0.01 },
    nodes: {
      mass: { type: 'Mass', params: {}, input_ports: ['flange'], output_ports: [] },
      adapter: {
        type: 'ActuationInput',
        params: {},
        input_ports: ['signal'],
        output_ports: ['flange'],
      },
    },
    connections: [],
  };
}

function report(graph: AcausalGraphSpec, status: DomainCompileReport['status']): DomainCompileReport {
  return {
    schema_id: 'feedbax.spec.domain_compile_report',
    schema_version: 'feedbax.spec.domain_compile_report.v1',
    status,
    interior_content_hash: stableGraphHash(graph),
    diagnostics:
      status === 'error'
        ? [{ severity: 'error', code: 'unbalanced', message: 'Unbalanced', node_ids: ['mass'] }]
        : [],
    summary: { equations: 2, unknowns: 2 },
  };
}

describe('acausal domain helpers', () => {
  it('matches the backend canonical acausal content hash', () => {
    expect(stableGraphHash(acausalGraph())).toBe(
      '88c5a546717d4f4059079e03b5df8f75d9ade7e63e39b1e60116b22cbe7ab727'
    );
  });

  it('blocks signal-to-conserving connections with adapter-oriented text', () => {
    const verdict = evaluateAcausalConnection(acausalGraph(), registry, {
      source: 'adapter',
      sourceHandle: 'signal',
      target: 'mass',
      targetHandle: 'flange',
    });

    expect(verdict).toEqual({
      allowed: false,
      reason: 'Signal and conserving ports connect through ActuationInput or SensorOutput adapters.',
    });
  });

  it('projects conserving connections as arrowless multi-edge-compatible edges', () => {
    const graph = acausalGraph();
    graph.connections = [
      { a: ['adapter', 'flange'], b: ['mass', 'flange'] },
      { a: ['adapter', 'flange'], b: ['mass', 'flange_2'] },
    ];

    const edges = acausalEdgesFromGraph(graph);

    expect(edges).toHaveLength(2);
    expect(edges.every((edge) => edge.type === 'conserving')).toBe(true);
    expect(edges[0].markerEnd).toBeUndefined();
  });

  it('derives stale and rolled-up compile status from reports and current hashes', () => {
    const graph = acausalGraph();
    const staleReport = report(graph, 'ok');
    graph.nodes!.extra = { type: 'Mass', params: {}, input_ports: ['flange'], output_ports: [] };

    expect(compileStatusForReport(staleReport, stableGraphHash(graph))).toBe('stale');

    const child = acausalGraph();
    const parent: AcausalGraphSpec = { ...acausalGraph(), subgraphs: { child } };
    expect(
      rollupCompileStatus({
        graph: parent,
        path: ['plant'],
        reports: {
          plant: report(parent, 'ok'),
          'plant/child': report(child, 'error'),
        },
      })
    ).toBe('error');
  });
});
