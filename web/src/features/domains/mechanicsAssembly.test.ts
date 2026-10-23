import { describe, expect, it } from 'vitest';
import type { AcausalGraphSpec, DomainDiagnostic } from '@/generated/studioContracts';
import {
  ASSEMBLY_INVALID_FRAME_REBIND,
  ASSEMBLY_MISSING_INTERIOR,
  editLinkLength,
  isAssemblyEditFailure,
  projectMechanicsAssembly,
  rebindJointFrame,
} from './mechanicsAssembly';

function twoLinkArm(): AcausalGraphSpec {
  return {
    schema_id: 'feedbax.spec.acausal_graph',
    schema_version: 'feedbax.spec.acausal_graph.v1',
    physical_domain: 'planar_multibody',
    nodes: {
      world: { type: 'WorldFrame', params: {}, input_ports: ['frame'], output_ports: [] },
      anchor: {
        type: 'Anchor',
        params: { world_frame: 'world.frame', frame: 'upper.proximal' },
        input_ports: ['frame', 'world'],
        output_ports: [],
      },
      upper: {
        type: 'PlanarLink',
        params: { length: 0.33, mass: 1, com: 0.5, inertia: 0.03 },
        input_ports: ['proximal', 'distal'],
        output_ports: [],
      },
      shoulder: {
        type: 'RevoluteJoint',
        params: { parent_frame: 'world.frame', child_frame: 'upper.proximal', damping: 0.1 },
        input_ports: ['parent', 'child'],
        output_ports: [],
      },
      forearm: {
        type: 'PlanarLink',
        params: { length: 0.27, mass: 1, com: 0.5, inertia: 0.02 },
        input_ports: ['proximal', 'distal'],
        output_ports: [],
      },
      elbow: {
        type: 'RevoluteJoint',
        params: { parent_frame: 'upper.distal', child_frame: 'forearm.proximal', damping: 0.1 },
        input_ports: ['parent', 'child'],
        output_ports: [],
      },
      ...Object.fromEntries(
        Array.from({ length: 6 }, (_, index) => [
          `muscle_${index}`,
          {
            type: 'MusclePath',
            params: {
              path_points: [
                { frame: 'world.frame', offset: [0, 0] },
                { frame: 'upper.distal', offset: [0, 0] },
                { frame: 'forearm.distal', offset: [0, 0] },
              ],
            },
            input_ports: [],
            output_ports: [],
          },
        ])
      ),
    },
    connections: [
      { a: ['world', 'frame'], b: ['anchor', 'world'] },
      { a: ['anchor', 'frame'], b: ['upper', 'proximal'] },
      { a: ['world', 'frame'], b: ['shoulder', 'parent'] },
      { a: ['shoulder', 'child'], b: ['upper', 'proximal'] },
      { a: ['upper', 'distal'], b: ['elbow', 'parent'] },
      { a: ['elbow', 'child'], b: ['forearm', 'proximal'] },
    ],
    solver: { solver_type: 'euler', dt: 0.01 },
  };
}

describe('mechanics assembly projection', () => {
  it('derives the two-link arm tree and muscle attachments from one acausal spec', () => {
    const projection = projectMechanicsAssembly(twoLinkArm());
    const structuralLabels = projection.rows
      .filter((row) => row.kind !== 'muscle')
      .map((row) => row.label);

    expect(structuralLabels).toEqual([
      'Anchor',
      'upper-arm link',
      'shoulder joint',
      'forearm link',
      'elbow joint',
    ]);
    expect(projection.rows.filter((row) => row.kind === 'muscle')).toHaveLength(6);
    expect(projection.rows.find((row) => row.id === 'upper')?.attachment_count).toBeUndefined();
    expect(projection.rows.find((row) => row.id === 'anchor')?.attachment_count).toBe(0);
  });

  it('round-trips link length edits through the same graph spec', () => {
    const graph = twoLinkArm();
    const result = editLinkLength(graph, 'upper', 0.45);

    expect(result.ok).toBe(true);
    if (!result.ok) throw new Error('expected length edit to succeed');
    expect(result.graph.nodes.upper.params?.length).toBe(0.45);
    expect(graph.nodes.upper.params?.length).toBe(0.33);
    expect(projectMechanicsAssembly(result.graph).rows.find((row) => row.id === 'upper')?.length)
      .toBe(0.45);
  });

  it('rebinds joint sockets by updating acausal connections and frame params', () => {
    const result = rebindJointFrame(twoLinkArm(), 'elbow', 'child', 'forearm.distal');

    expect(result.ok).toBe(true);
    if (!result.ok) throw new Error('expected rebind to succeed');
    expect(result.graph.nodes.elbow.params?.child_frame).toBe('forearm.distal');
    expect(
      result.graph.connections.some(
        (connection) =>
          `${connection.a[0]}.${connection.a[1]}|${connection.b[0]}.${connection.b[1]}` ===
            'elbow.child|forearm.distal' ||
          `${connection.a[0]}.${connection.a[1]}|${connection.b[0]}.${connection.b[1]}` ===
            'forearm.distal|elbow.child'
      )
    ).toBe(true);
  });

  it('rejects invalid frame rebinds with a named diagnostic', () => {
    const result = rebindJointFrame(twoLinkArm(), 'elbow', 'child', 'ghost.frame');

    expect(isAssemblyEditFailure(result)).toBe(true);
    if (!isAssemblyEditFailure(result)) throw new Error('expected rebind to fail');
    expect(result.diagnostic.code).toBe(ASSEMBLY_INVALID_FRAME_REBIND);
    expect(result.diagnostic.node_ids).toEqual(['elbow']);
  });

  it('surfaces backend diagnostics as row badges without duplicating the source', () => {
    const diagnostics: DomainDiagnostic[] = [
      {
        severity: 'error',
        code: 'mechanics.unanchored_chain',
        message: 'Unanchored chain: upper',
        node_ids: ['upper'],
      },
    ];

    const projection = projectMechanicsAssembly(twoLinkArm(), diagnostics);

    expect(projection.rows.find((row) => row.id === 'upper')?.diagnostic_codes).toEqual([
      'mechanics.unanchored_chain',
    ]);
  });

  it('does not synthesize a tree when a composite has no materialized subgraph', () => {
    const projection = projectMechanicsAssembly(null, [], 'plant');

    expect(projection.rows).toEqual([]);
    expect(projection.diagnostics[0]).toMatchObject({
      code: ASSEMBLY_MISSING_INTERIOR,
      severity: 'error',
    });
  });
});
