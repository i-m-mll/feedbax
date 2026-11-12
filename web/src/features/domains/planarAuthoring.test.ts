import { expect, it } from 'vitest';
import { applyAuthoredEdits, type AuthoredProject } from '@/features/projectFolder/authoredEdits';
import { AcausalGraphSpecSchema, ComponentSpecSchema } from '@/generated/studioContracts';
import { planPlanarOperation, readPlanarDraft, type PlanarOperation } from './planarAuthoring';

const occurrence = { document_id: 'model', node_path: ['mechanics'] };
const address = { kind: 'document', id: 'model' } as const;

function initial({ explicit = false, wired = false } = {}): AuthoredProject {
  return { documents: { model: { schema: 'feedbax.spec.graph', payload: {
    nodes: { mechanics: { type: 'feedbax.AcausalSystem' } },
    wires: wired ? [{ target_node: 'mechanics', target_port: 'force' }] : [],
    subgraphs: { mechanics: {
      physical_domain: 'planar_multibody', solver: { solver_type: 'euler', dt: 0.002 },
      nodes: {
        world: { type: 'feedbax.WorldFrame' },
        anchor: { type: 'feedbax.Anchor', params: { frame: 'upper.proximal' } },
        upper: { type: 'feedbax.PlanarLink', params: { length: 0.4 } },
        shoulder: { type: 'feedbax.RevoluteJoint', params: { parent_frame: 'world.frame', child_frame: 'upper.proximal' } },
        drive: { type: 'feedbax.ActuationInput', params: { source_kind: 'actuator_control', size: 1, port_name: 'force' } },
      },
      ...(explicit ? { connections: [
        { a: ['anchor', 'world'], b: ['world', 'frame'] },
        { a: ['anchor', 'frame'], b: ['upper', 'proximal'] },
        { a: ['shoulder', 'parent'], b: ['world', 'frame'] },
        { a: ['shoulder', 'child'], b: ['upper', 'proximal'] },
      ] } : {}),
    } },
  } } } };
}

function edit(project: AuthoredProject, operation: PlanarOperation) {
  const planned = planPlanarOperation(project, occurrence, operation);
  if (planned.ok === false) throw new Error(planned.reason);
  return applyAuthoredEdits(project, address, planned.value.patches, true);
}

function graph(project: AuthoredProject) {
  const selected = readPlanarDraft(project, occurrence);
  if (selected.ok === false) throw new Error(selected.reason);
  return selected.value.graph;
}

const append = { kind: 'append-link', node: 'forearm', joint: 'elbow', length: 0.32 } as const;

it('keeps the authored connection convention and inverts an ordered structural edit', () => {
  for (const explicit of [false, true]) {
    const project = initial({ explicit });
    const extended = edit(project, append).project;
    expect(graph(extended).nodes.drive.params?.size).toBe(2);
    expect(graph(extended).connections?.length).toBe(explicit ? 6 : undefined);
    const withJointFrame = edit(extended, { kind: 'bind-joint', node: 'elbow', socket: 'child', frame: 'forearm.distal' }).project;
    expect(graph(withJointFrame).nodes.elbow.params?.child_frame).toBe('forearm.distal');
    expect(graph(withJointFrame).connections?.length).toBe(explicit ? 6 : undefined);
    const removed = edit(withJointFrame, { kind: 'remove-tail', node: 'forearm' });
    expect(graph(removed.project).nodes.drive.params?.size).toBe(1);
    const restored = applyAuthoredEdits(removed.project, address, removed.inverse).project;
    expect(JSON.stringify(restored)).toBe(JSON.stringify(withJointFrame));
  }
});

it('refuses to resize wired inputs or delete/rebind to break references', () => {
  expect(planPlanarOperation(initial({ wired: true }), occurrence, append)).toMatchObject({ ok: false, reason: /dependent controller/ });
  const extended = edit(initial(), append).project;
  const attached = edit(extended, { kind: 'add-muscle', node: 'flexor', points: [
    { frame: 'world.frame', offset: [0, 0.1] },
    { frame: 'forearm.distal', offset: [0, 0.1] },
  ] }).project;
  expect(planPlanarOperation(attached, occurrence, { kind: 'remove-tail', node: 'forearm' })).toMatchObject({ ok: false, reason: /flexor/ });
  expect(planPlanarOperation(attached, occurrence, { kind: 'bind-joint', node: 'elbow', socket: 'parent', frame: 'forearm.proximal' }))
    .toMatchObject({ ok: false, reason: /grounded serial chain/ });
  expect(planPlanarOperation(attached, occurrence, { kind: 'set-attachment', node: 'flexor', index: 1, expectedFrame: 'forearm.distal',
    attachment: { frame: 'missing.distal', offset: [0, 0] } })).toMatchObject({ ok: false, reason: /missing frame/ });
  const referenced: AuthoredProject = { ...extended, documents: { ...extended.documents, analysis: {
    schema: 'analysis', payload: { body: { document: 'model', pointer: '/subgraphs/mechanics/nodes/forearm/params/length' } },
  } } };
  expect(planPlanarOperation(referenced, occurrence, { kind: 'remove-tail', node: 'forearm' })).toMatchObject({ ok: false, reason: /authored references/ });
});

it('refuses to rebase an interior replacement over independently authored fields', () => {
  const project = initial();
  const added = planPlanarOperation(project, occurrence, append);
  if (added.ok === false) throw new Error(added.reason);
  const external = edit(project, { kind: 'set-link', node: 'upper', field: 'mass', value: 2 }).project;
  expect(() => applyAuthoredEdits(external, address, added.value.patches, true, project)).toThrow('authored structure');
  expect(graph(external).nodes.upper.params?.mass).toBe(2);
});

it('keeps omitted authored maps and validates typed parameter envelopes', () => {
  const world = { type: 'feedbax.WorldFrame' };
  expect(ComponentSpecSchema.parse(world)).toEqual(world);
  const empty = { physical_domain: 'planar_multibody', solver: { solver_type: 'euler', dt: 0.002 } };
  expect(AcausalGraphSpecSchema.parse(empty)).toEqual(empty);
  expect(AcausalGraphSpecSchema.parse({ ...empty, nodes: { world } })).toEqual({ ...empty, nodes: { world } });
  const badParameter = { type: 'feedbax.PlanarLink', params: { length: {
    schema_id: 'feedbax.spec.component_param.array_value', values: ['invalid'],
  } } };
  expect(AcausalGraphSpecSchema.safeParse({ ...empty, nodes: { link: badParameter } }).success).toBe(false);
});
