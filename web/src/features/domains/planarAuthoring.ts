import { AcausalGraphSpecSchema, type AcausalGraphSpec, type ComponentSpec, type JsonValue, type ProjectValueEdit } from '@/generated/studioContracts';
import type { AuthoredProject } from '@/features/projectFolder/authoredEdits';
import type { ProjectEditController, ProjectEditValidation } from '@/features/projectFolder/projectEditController';
import { editLinkLength, frameOptionsForGraph, rebindJointFrame, type JointSocketKind } from './mechanicsAssembly';

type JsonObject = { [key: string]: JsonValue };
type Point = [number, number];
export interface PlanarOccurrence { document_id: string; node_path: readonly string[] }
export interface PlanarAttachment { frame: string; offset: Point }
type LinkField = 'length' | 'mass' | 'inertia' | 'com' | 'com_y';
type JointField = 'damping' | 'stiffness' | 'rest_angle' | 'frame_angle';
type MuscleField = 'max_isometric_force' | 'optimal_muscle_length' | 'tendon_slack_length' | 'tau_act' | 'tau_deact';

export type PlanarOperation =
  | { kind: 'set-link'; node: string; field: LinkField; value: number }
  | { kind: 'set-joint'; node: string; field: JointField; value: number }
  | { kind: 'set-joint-offset'; node: string; offset: Point }
  | { kind: 'bind-joint'; node: string; socket: JointSocketKind; frame: string }
  | { kind: 'set-muscle'; node: string; field: MuscleField; value: number }
  | { kind: 'set-attachment'; node: string; index: number; expectedFrame: string; attachment: PlanarAttachment }
  | { kind: 'insert-guide'; node: string; index: number; attachment: PlanarAttachment }
  | { kind: 'remove-guide'; node: string; index: number }
  | { kind: 'add-muscle'; node: string; points: readonly PlanarAttachment[] }
  | { kind: 'remove-muscle'; node: string }
  | { kind: 'append-link'; node: string; joint: string; length: number }
  | { kind: 'remove-tail'; node: string };

type Graph = AcausalGraphSpec & { nodes: Record<string, ComponentSpec> };
type ErrorResult = { ok: false; reason: string; path: string; node: string | null };
type Result<T> = { ok: true; value: T } | ErrorResult;
type Joint = { node: string; body: string };
type Draft = { graph: Graph; parent: JsonObject; path: string };
type Plan = { name: string; patches: ProjectValueEdit[] };

const ok = <T>(value: T): Result<T> => ({ ok: true, value });
const refuse = (reason: string, path: string, node: string | null = null): ErrorResult => ({ ok: false, reason, path, node });
const object = (value: unknown): value is JsonObject => value !== null && typeof value === 'object' && !Array.isArray(value);
const own = (value: object, key: string) => Object.prototype.hasOwnProperty.call(value, key);
const token = (value: string) => value.replace(/~/g, '~0').replace(/\//g, '~1');
const kind = (node: ComponentSpec | undefined, type: string) => node?.type === `feedbax.${type}`;
const pair = (value: unknown): value is Point => Array.isArray(value) && value.length === 2 && value.every(Number.isFinite);
const finitePositive = (value: number) => Number.isFinite(value) && value > 0;

export function readPlanarDraft(project: AuthoredProject, occurrence: PlanarOccurrence): Result<Draft> {
  const doc = project.documents?.[occurrence.document_id];
  if (doc?.schema !== 'feedbax.spec.graph' || !object(doc.payload) || !occurrence.node_path.length) {
    return refuse('Choose a named planar model document.', '/payload');
  }
  let current = doc.payload;
  let parent = current;
  let path = '/payload';
  for (const name of occurrence.node_path) {
    const subgraphs = current.subgraphs;
    path += `/subgraphs/${token(name)}`;
    if (!object(subgraphs) || !own(subgraphs, name) || !object(subgraphs[name])) {
      return refuse('The selected model occurrence no longer exists.', path, name);
    }
    parent = current;
    current = subgraphs[name];
  }
  const parsed = AcausalGraphSpecSchema.safeParse(current);
  if (!parsed.success || parsed.data.physical_domain !== 'planar_multibody' || !parsed.data.nodes || Object.keys(parsed.data.subgraphs ?? {}).length) {
    return refuse('The selected occurrence is not a supported planar interior.', path);
  }
  return ok({ graph: current as unknown as Graph, path, parent });
}

function frames(graph: Graph) {
  return new Map(frameOptionsForGraph(graph).map((frame) => [frame.id, [frame.node_id, frame.port] as PointFrame]));
}
type PointFrame = [string, string];

function chain(graph: Graph): Joint[] | null {
  const world = Object.entries(graph.nodes).filter(([, node]) => kind(node, 'WorldFrame'));
  const anchors = Object.entries(graph.nodes).filter(([, node]) => kind(node, 'Anchor'));
  const joints = Object.entries(graph.nodes).filter(([, node]) => kind(node, 'RevoluteJoint'));
  const links = Object.values(graph.nodes).filter((node) => kind(node, 'PlanarLink'));
  if (world.length !== 1 || anchors.length !== 1 || !links.length || joints.length !== links.length) return null;
  const worldFrame = `${world[0][0]}.frame`;
  const anchor = anchors[0][1].params;
  if ((anchor?.world_frame ?? 'world.frame') !== worldFrame) return null;
  const byFrame = frames(graph);
  const root = typeof anchor?.frame === 'string' ? byFrame.get(anchor.frame) : undefined;
  if (!root || root[1] === 'frame') return null;
  const edge = (left: PointFrame, right: PointFrame) => JSON.stringify([JSON.stringify(left), JSON.stringify(right)].sort());
  const expectedEdges = new Set([
    edge([anchors[0][0], 'world'], [world[0][0], 'frame']), edge([anchors[0][0], 'frame'], root),
  ]);
  const outgoing = new Map<string, Joint>();
  for (const [id, joint] of joints) {
    const { parent_frame: parentFrame, child_frame: childFrame } = joint.params ?? {};
    const parent = typeof parentFrame === 'string' ? byFrame.get(parentFrame) : undefined;
    const child = typeof childFrame === 'string' ? byFrame.get(childFrame) : undefined;
    if (!parent || !child || child[1] === 'frame' || outgoing.has(parent[0]) ||
        (parent[1] === 'frame' && childFrame !== anchor?.frame)) return null;
    outgoing.set(parent[0], { node: id, body: child[0] });
    expectedEdges.add(edge([id, 'parent'], parent));
    expectedEdges.add(edge([id, 'child'], child));
  }
  if (graph.connections?.length) {
    const actualEdges = new Set(graph.connections.map(({ a, b }) => edge(a, b)));
    if (actualEdges.size !== graph.connections.length || actualEdges.size !== expectedEdges.size ||
        [...actualEdges].some((item) => !expectedEdges.has(item))) return null;
  }
  const ordered: Joint[] = [];
  const visited = new Set<string>();
  let parent = world[0][0];
  for (let i = 0; i < links.length; i += 1) {
    const joint = outgoing.get(parent);
    if (!joint || visited.has(joint.body)) return null;
    ordered.push(joint);
    visited.add(joint.body);
    parent = joint.body;
  }
  return ordered;
}

export function attachmentPoints(node: ComponentSpec): PlanarAttachment[] | null {
  const value = node.params?.path_points;
  if (!Array.isArray(value) || value.length < 2) return null;
  const resolved: PlanarAttachment[] = [];
  for (const point of value) {
    if (!object(point) || typeof point.frame !== 'string' || !pair(point.offset ?? [0, 0])) return null;
    resolved.push({ frame: point.frame, offset: (point.offset ?? [0, 0]) as Point });
  }
  return resolved;
}

export function fixedPath(node: ComponentSpec | undefined) {
  return node && kind(node, 'MusclePath') && !(Array.isArray(node.params?.wraps) && node.params.wraps.length) && node.params?.moment_arm == null;
}

function invalidPath(graph: Graph) {
  const byFrame = frames(graph);
  for (const [name, node] of Object.entries(graph.nodes)) {
    if (!kind(node, 'MusclePath')) continue;
    const path = attachmentPoints(node);
    if (!path) return `Muscle ${name} requires an origin and insertion with finite local offsets.`;
    let previous: { body: string; x: number; y: number } | undefined;
    for (const point of path) {
      const frame = byFrame.get(point.frame);
      if (!frame) return `Muscle ${name} refers to missing frame ${point.frame}.`;
      const [body, port] = frame;
      const x = point.offset[0] + (port === 'distal' ? Number(graph.nodes[body].params?.length ?? 0.3) : 0);
      const y = point.offset[1];
      if (previous?.body === body && previous.x === x && previous.y === y) return `Muscle ${name} contains a zero-length fixed segment.`;
      previous = { body, x, y };
    }
  }
  return null;
}

function replaceParameter(graph: Graph, node: string, field: string, value: JsonValue): Graph {
  const original = graph.nodes[node];
  return { ...graph, nodes: { ...graph.nodes, [node]: { ...original, params: { ...original.params, [field]: value } } } };
}

function paramPatch(draft: Draft, node: string, field: string, value: JsonValue): ProjectValueEdit {
  const params = draft.graph.nodes[node].params;
  const path = `${draft.path}/nodes/${token(node)}/params`;
  return params
    ? { op: own(params, field) ? 'replace' : 'add', path: `${path}/${token(field)}`, value }
    : { op: 'add', path, value: { [field]: value } };
}

function validId(nodes: Graph['nodes'], id: string) {
  return /^[A-Za-z_][A-Za-z0-9_-]*$/.test(id) && id !== '__proto__' && !own(nodes, id);
}

function resizeActuation(draft: Draft, occurrence: PlanarOccurrence, nodes: Graph['nodes'], delta: number, changing: 'links' | 'muscles') {
  if (Object.values(nodes).some((node) => kind(node, 'MusclePath') && node.params?.moment_arm != null)) {
    return 'Remove the constant moment arm before changing the link or muscle layout.';
  }
  const controls = Object.entries(nodes).filter(([, node]) => kind(node, 'ActuationInput'));
  if (controls.length !== 1) return 'The chain requires one declared control input.';
  const [id, control] = controls[0];
  const params = control.params;
  const source = params?.source_kind;
  if (source !== 'actuator_control' && source !== 'muscle_excitation_vector') return 'This control input cannot be resized by Scene.';
  if (source === 'muscle_excitation_vector' && changing === 'muscles' && Object.values(nodes).every((node) => !kind(node, 'MusclePath'))) {
    return 'Keep at least one muscle for the excitation input.';
  }
  const occurrenceId = occurrence.node_path[occurrence.node_path.length - 1];
  const statePorts = new Set(Object.values(nodes).filter((node) => kind(node, 'SensorOutput') && node.params?.quantity === 'state')
    .map((node) => node.params?.port_name ?? 'y'));
  const stateConsumer = (Array.isArray(draft.parent.wires) && draft.parent.wires.some((wire) =>
    object(wire) && wire.source_node === occurrenceId && statePorts.has(wire.source_port))) ||
    (object(draft.parent.output_bindings) && Object.values(draft.parent.output_bindings).some((binding) =>
      Array.isArray(binding) && binding[0] === occurrenceId && statePorts.has(binding[1])));
  if (stateConsumer) return 'Disconnect dependent state outputs before changing the model layout.';
  if ((source === 'actuator_control') !== (changing === 'links')) return null;
  const port = params?.port_name ?? 'u';
  const wired = Array.isArray(draft.parent.wires) && draft.parent.wires.some((wire) =>
    object(wire) && wire.target_node === occurrenceId && wire.target_port === port);
  const exposed = object(draft.parent.input_bindings) && Object.values(draft.parent.input_bindings).some((binding) =>
    Array.isArray(binding) && binding[0] === occurrenceId && binding[1] === port);
  const adapters = Array.isArray(draft.parent.additive_channel_adapters) && draft.parent.additive_channel_adapters.some((adapter) =>
    object(adapter) && object(adapter.target) && adapter.target.target_node === occurrenceId && adapter.target.target_port === port);
  if (wired || adapters || (exposed && occurrence.node_path.length > 1)) return 'Disconnect the dependent controller before changing its input width.';
  if (source === 'actuator_control') {
    const size = params?.size;
    if (typeof size !== 'number' || !Number.isInteger(size) || size + delta < 1) return 'The actuator width is invalid.';
    nodes[id] = { ...control, params: { ...params, size: size + delta } };
  }
  return null;
}

function referencedPaths(project: AuthoredProject, occurrence: PlanarOccurrence, path: string, relatives: readonly string[][]): boolean {
  const pointerPaths = relatives.map((segments) => path.slice('/payload'.length) + '/' + segments.map(token).join('/'));
  const parts = occurrence.node_path.flatMap((name) => ['subgraphs', name]);
  const patchPaths = relatives.map((segments) => ['documents', occurrence.document_id, 'payload', ...parts, ...segments].join('.'));
  const under = (value: unknown, prefixes: string[], delimiter: string) =>
    typeof value === 'string' && prefixes.some((prefix) => value === prefix || value.startsWith(prefix + delimiter));
  function visit(value: unknown): boolean {
    if (Array.isArray(value)) return value.some(visit);
    if (!object(value)) return false;
    if (value.document === occurrence.document_id && Object.keys(value).every((key) => key === 'document' || key === 'pointer') &&
        under(value.pointer, pointerPaths, '/')) return true;
    const patches = value.patches;
    if (object(patches) && Object.keys(patches).some((key) => patches[key] !== null && under(key, patchPaths, '.'))) return true;
    return Object.values(value).some(visit);
  }
  return visit(project);
}

function tailDependent(graph: Graph, tail: Joint) {
  const frameIds = new Set([`${tail.body}.proximal`, `${tail.body}.distal`]);
  const owner = Object.entries(graph.nodes).find(([id, node]) => id !== tail.node && id !== tail.body &&
    (frameIds.has(String(node.params?.frame)) || frameIds.has(String(node.params?.parent_frame)) ||
      frameIds.has(String(node.params?.child_frame)) || attachmentPoints(node)?.some((point) => frameIds.has(point.frame))));
  if (owner) return owner[0];
  const wrap = Object.entries(graph.wrap_geometries ?? {}).find(([, value]) => frameIds.has(value.frame));
  if (wrap) return wrap[0];
  return Object.entries(graph.planar_dynamics?.joint_dampers ?? {}).find(([, damper]) => own(damper.coefficients, tail.node))?.[0];
}

export function planPlanarOperation(project: AuthoredProject, occurrence: PlanarOccurrence, operation: PlanarOperation): Result<Plan> {
  const selected = readPlanarDraft(project, occurrence);
  if (selected.ok === false) return selected;
  const draft = selected.value;
  const { graph, path } = draft;
  const ordered = chain(graph);
  const node = graph.nodes[operation.node];
  const bad = (reason: string) => refuse(reason, `${path}/nodes/${token(operation.node)}`, operation.node);
  if (!ordered) return bad('The model must be one grounded serial chain.');
  const parameter = (name: string, field: string, value: JsonValue) => ok({ name, patches: [paramPatch(draft, operation.node, field, value)] });
  const structural = (name: string, next: Graph) => {
    if (!chain(next)) return bad('The edit would break the grounded serial chain.');
    const issue = invalidPath(next);
    return issue ? bad(issue) : ok({ name, patches: [{ op: 'replace' as const, path, value: next as unknown as JsonValue }] });
  };

  switch (operation.kind) {
    case 'set-link': {
      const { field, value } = operation;
      if (!kind(node, 'PlanarLink') || !Number.isFinite(value) ||
          ((field === 'length' || field === 'mass' || field === 'inertia') && value <= 0)) return bad('Enter a supported finite link value.');
      if (field === 'length') {
        const edited = editLinkLength(graph, operation.node, value);
        if (edited.ok === false) return bad(edited.diagnostic.message);
        const issue = invalidPath(edited.graph as Graph);
        if (issue) return bad(issue);
      }
      return parameter(field === 'length' ? 'Set link length' : 'Set link property', field, value);
    }
    case 'set-joint':
      if (!kind(node, 'RevoluteJoint') || !Number.isFinite(operation.value) ||
          ((operation.field === 'damping' || operation.field === 'stiffness') && operation.value < 0)) return bad('Enter a supported finite joint value.');
      return parameter('Set joint property', operation.field, operation.value);
    case 'set-joint-offset':
      return kind(node, 'RevoluteJoint') && pair(operation.offset)
        ? parameter('Set joint offset', 'frame_offset', operation.offset) : bad('Enter a finite joint offset.');
    case 'bind-joint': {
      const edited = rebindJointFrame(graph, operation.node, operation.socket, operation.frame);
      if (edited.ok === false) return bad(edited.diagnostic.message);
      const after = chain(edited.graph as Graph);
      if (after?.some((joint, index) => joint.node !== ordered[index]?.node || joint.body !== ordered[index]?.body)) {
        return bad('Rebinding must preserve the ordered body and joint identities.');
      }
      return structural('Bind joint frame', edited.graph as Graph);
    }
    case 'set-muscle': {
      const params = node?.params;
      if (!kind(node, 'MusclePath') || !['feedbax.RigidTendonHillMuscle', 'RigidTendonHillMuscle'].includes(String(params?.law ?? 'feedbax.RigidTendonHillMuscle')) ||
          !finitePositive(operation.value)) return bad('Enter a positive value for the supported rigid tendon muscle law.');
      return object(params?.law_parameters)
        ? parameter('Set muscle property', 'law_parameters', { ...params.law_parameters, [operation.field]: operation.value })
        : parameter('Set muscle property', operation.field, operation.value);
    }
    case 'set-attachment':
    case 'insert-guide':
    case 'remove-guide': {
      const attachments = fixedPath(node) ? attachmentPoints(node) : null;
      if (!attachments || !Number.isInteger(operation.index)) return bad('Choose a fixed muscle attachment.');
      const { index } = operation;
      const existing = attachments[index];
      if (operation.kind === 'set-attachment') {
        if (!existing || existing.frame !== operation.expectedFrame || !pair(operation.attachment.offset)) return bad('The attachment frame changed. Choose the point again.');
        if (existing.frame === operation.attachment.frame && existing.offset.every((value, axis) => value === operation.attachment.offset[axis])) {
          return ok({ name: 'Edit attachment', patches: [] });
        }
        const original = node.params!.path_points as JsonObject[];
        const replacement = { ...original[index], ...operation.attachment };
        const next = [...original];
        next[index] = replacement;
        const issue = invalidPath(replaceParameter(graph, operation.node, 'path_points', next));
        if (issue) return bad(issue);
        const editPath = `${path}/nodes/${token(operation.node)}/params/path_points/${index}`;
        return ok({ name: 'Edit attachment', patches: [{ op: 'replace', path: editPath, value: replacement }] });
      }
      if (index < 1 || index >= attachments.length - (operation.kind === 'remove-guide' ? 1 : 0)) return bad('Origin and insertion must remain at the ends of the path.');
      const shifted = attachments.slice(index).map((_, offset) => ['nodes', operation.node, 'params', 'path_points', String(index + offset)]);
      if (referencedPaths(project, occurrence, path, shifted)) return bad('Rebind the authored references to this path before shifting its guides.');
      const updated = [...node.params!.path_points as JsonObject[]];
      if (operation.kind === 'insert-guide') {
        if (!pair(operation.attachment.offset)) return bad('Enter a finite guide offset.');
        updated.splice(index, 0, { ...operation.attachment });
      } else updated.splice(index, 1);
      const issue = invalidPath(replaceParameter(graph, operation.node, 'path_points', updated));
      return issue ? bad(issue) : parameter(operation.kind === 'insert-guide' ? 'Insert fixed guide' : 'Remove fixed guide', 'path_points', updated);
    }
    case 'append-link': {
      if (!validId(graph.nodes, operation.node) || !validId(graph.nodes, operation.joint) || operation.node === operation.joint || !finitePositive(operation.length)) {
        return bad('Choose unused link and joint names and a positive length.');
      }
      const tail = ordered[ordered.length - 1];
      const nodes = { ...graph.nodes,
        [operation.node]: { type: 'feedbax.PlanarLink', params: { length: operation.length } },
        [operation.joint]: { type: 'feedbax.RevoluteJoint', params: { parent_frame: `${tail.body}.distal`, child_frame: `${operation.node}.proximal` } },
      };
      const issue = resizeActuation(draft, occurrence, nodes, 1, 'links');
      if (issue) return bad(issue);
      return structural('Append link and joint', { ...graph, nodes, ...(graph.connections?.length ? { connections: [
        ...graph.connections,
        { a: [operation.joint, 'parent'], b: [tail.body, 'distal'] },
        { a: [operation.joint, 'child'], b: [operation.node, 'proximal'] },
      ] } : {}) });
    }
    case 'remove-tail': {
      const tail = ordered[ordered.length - 1];
      if (ordered.length === 1 || tail.body !== operation.node) return bad('Remove only the terminal link of a chain with at least two links.');
      const dependency = tailDependent(graph, tail);
      if (dependency) return bad(`Rebind ${dependency} before removing ${tail.body}.`);
      if (referencedPaths(project, occurrence, path, [['nodes', tail.body], ['nodes', tail.node]])) {
        return bad('Rebind the authored references to the terminal link or joint before removing it.');
      }
      const nodes = { ...graph.nodes };
      delete nodes[tail.body];
      delete nodes[tail.node];
      const issue = resizeActuation(draft, occurrence, nodes, -1, 'links');
      if (issue) return bad(issue);
      const connections = graph.connections?.filter(({ a, b }) => a[0] !== tail.node && b[0] !== tail.node);
      return structural('Remove terminal link and joint', { ...graph, nodes, ...(connections ? { connections } : {}) });
    }
    case 'add-muscle':
    case 'remove-muscle': {
      const adding = operation.kind === 'add-muscle';
      const nodes = { ...graph.nodes };
      if (adding) {
        if (!validId(nodes, operation.node) || operation.points.length < 2 || operation.points.some((point) => !pair(point.offset))) {
          return bad('Choose an unused muscle name and an ordered fixed path.');
        }
        nodes[operation.node] = { type: 'feedbax.MusclePath', params: { path_points: operation.points.map((point) => ({ ...point })) } };
      } else {
        if (!fixedPath(node)) return bad('Choose a fixed muscle path to remove.');
        if (graph.connections?.some(({ a, b }) => a[0] === operation.node || b[0] === operation.node)) return bad('Disconnect the dependent muscle references first.');
        if (referencedPaths(project, occurrence, path, [['nodes', operation.node]])) {
          return bad('Rebind the authored references to this muscle before removing it.');
        }
        delete nodes[operation.node];
      }
      const issue = resizeActuation(draft, occurrence, nodes, adding ? 1 : -1, 'muscles');
      return issue ? bad(issue) : structural(adding ? 'Add muscle path' : 'Remove muscle path', { ...graph, nodes });
    }
  }
}

function structuralValidation(occurrence: PlanarOccurrence, operation: PlanarOperation): ProjectEditValidation | undefined {
  switch (operation.kind) {
    case 'append-link':
    case 'remove-tail':
    case 'add-muscle':
    case 'remove-muscle':
    case 'insert-guide':
    case 'remove-guide': {
      const guide = operation.kind === 'insert-guide' || operation.kind === 'remove-guide'
        ? { node: operation.node, index: operation.index } : null;
      return (before, after) => {
        const old = readPlanarDraft(before, occurrence);
        const next = readPlanarDraft(after, occurrence);
        if (old.ok === false || next.ok === false) throw new Error('The authored Scene occurrence is unavailable.');
        const { path, graph } = next.value;
        const removed = Object.keys(old.value.graph.nodes).filter((id) => !own(graph.nodes, id)).map((id) => ['nodes', id]);
        if (guide) {
          const original = attachmentPoints(old.value.graph.nodes[guide.node]) ?? [];
          removed.push(...original.slice(guide.index).map((_, offset) =>
            ['nodes', guide.node, 'params', 'path_points', String(guide.index + offset)]));
        }
        if (removed.length && referencedPaths(after, occurrence, path, removed)) {
          throw new Error('An authored reference still uses an element changed by this edit.');
        }
      };
    }
    default:
      return undefined;
  }
}

export function commitPlanarOperation(editor: ProjectEditController, occurrence: PlanarOccurrence, operation: PlanarOperation): Result<Plan> {
  const planned = planPlanarOperation(editor.getSnapshot().draft, occurrence, operation);
  if (!planned.ok || !planned.value.patches.length) return planned;
  try {
    const target = { document_id: occurrence.document_id, node_path: [...occurrence.node_path] };
    editor.edit({ kind: 'document', id: target.document_id }, planned.value.name, planned.value.patches, structuralValidation(target, operation));
    return planned;
  } catch (error) {
    return refuse(error instanceof Error ? error.message : String(error), planned.value.patches[0].path, operation.node);
  }
}
