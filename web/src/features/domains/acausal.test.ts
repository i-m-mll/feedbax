import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import type { ComponentDefinition } from '@/types/components';
import type { AcausalGraphSpec, DomainCompileReport } from '@/generated/studioContracts';
import {
  acausalEdgesFromGraph,
  compileStatusForReport,
  evaluateAcausalConnection,
  graphPathKey,
  rollupCompileStatus,
  stableGraphHash,
} from './acausal';

const registry = new Map<string, ComponentDefinition>([
  [
    'Mass',
    {
      name: 'feedbax.Mass',
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
      name: 'feedbax.ActuationInput',
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
      mass: { type: 'feedbax.Mass', params: {}, input_ports: ['flange'], output_ports: [] },
      adapter: {
        type: 'feedbax.ActuationInput',
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
  it('matches the endpoint subject across authored and defaulted fields', () => {
    const { interior, subject } = JSON.parse(readFileSync(
      new URL('../../../../tests/fixtures/acausal_report_subject.json', import.meta.url), 'utf8',
    )) as { interior: AcausalGraphSpec; subject: string };
    const node = interior.nodes!.plant;
    const child = interior.subgraphs!.plant;
    const [{ a, b }] = child.connections!;
    expect(stableGraphHash(interior)).toBe(subject);
    expect(stableGraphHash({
      ...interior,
      schema_id: 'feedbax.spec.acausal_graph',
      nodes: { plant: { ...node, revision: null, input_ports: [], output_ports: [] } },
      connections: [],
      solver: {
        ...interior.solver,
        root_finder: { method: 'newton', rtol: 1e-6, atol: 1e-6, max_steps: 32 },
      },
      subgraphs: { plant: { ...child, connections: [{ a: b, b: a }], metadata: undefined } },
    })).toBe(subject);
    expect(stableGraphHash({
      ...interior,
      solver: { ...interior.solver, root_finder: null },
    })).not.toBe(subject);
    const authored = node.params.authored as unknown[];
    expect(stableGraphHash({
      ...interior,
      nodes: { plant: { ...node, params: { ...node.params, authored: authored.slice(1) } } },
    })).not.toBe(subject);
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
    graph.nodes!.extra = { type: 'feedbax.Mass', params: {}, input_ports: ['flange'], output_ports: [] };

    expect(compileStatusForReport(staleReport, stableGraphHash(graph))).toBe('stale');

    const child = acausalGraph();
    const parent: AcausalGraphSpec = { ...acausalGraph(), subgraphs: { child } };
    const reports = {
      [graphPathKey(['plant'])]: report(parent, 'ok'),
      [graphPathKey(['plant', 'child'])]: report(child, 'error'),
    };
    expect(
      rollupCompileStatus({
        graph: parent,
        path: ['plant'],
        statusForLayer: (path, hash) => compileStatusForReport(reports[graphPathKey(path)], hash),
      })
    ).toBe('error');
  });
});
