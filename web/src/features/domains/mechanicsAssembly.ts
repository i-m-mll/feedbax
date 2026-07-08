import type {
  AcausalConnectionSpec,
  AcausalGraphSpec,
  ComponentSpec,
  DomainDiagnostic,
} from '@/generated/studioContracts';

export const ASSEMBLY_MISSING_INTERIOR = 'mechanics.assembly.missing_interior';
export const ASSEMBLY_UNSUPPORTED_DOMAIN = 'mechanics.assembly.unsupported_domain';
export const ASSEMBLY_INVALID_FRAME_REBIND = 'mechanics.assembly.invalid_frame_rebind';

type Endpoint = [string, string];

export type AssemblyViewMode = 'graph' | 'assembly' | 'split';
export type AssemblyRowKind = 'anchor' | 'link' | 'joint' | 'muscle' | 'marker' | 'adapter';
export type JointSocketKind = 'parent' | 'child';

export interface AssemblyFrameOption {
  id: string;
  node_id: string;
  port: string;
  label: string;
}

export interface AssemblyJointSocket {
  kind: JointSocketKind;
  frame_id: string | null;
  options: AssemblyFrameOption[];
  diagnostic_codes: string[];
}

export interface AssemblyRow {
  id: string;
  kind: AssemblyRowKind;
  node_id: string;
  label: string;
  depth: number;
  frame_id?: string | null;
  attachment_count?: number;
  length?: number | null;
  sockets?: AssemblyJointSocket[];
  diagnostic_codes: string[];
}

export interface AssemblyProjection {
  rows: AssemblyRow[];
  frames: AssemblyFrameOption[];
  diagnostics: DomainDiagnostic[];
}

export type AssemblyEditResult =
  | { ok: true; graph: AcausalGraphSpec }
  | { ok: false; diagnostic: DomainDiagnostic };

export function isAssemblyEditFailure(
  result: AssemblyEditResult
): result is Extract<AssemblyEditResult, { ok: false }> {
  return result.ok === false;
}

function endpointKey(endpoint: Endpoint): string {
  return `${endpoint[0]}.${endpoint[1]}`;
}

function parseFrameId(frameId: string): Endpoint | null {
  const [nodeId, port, ...rest] = frameId.split('.');
  if (!nodeId || !port || rest.length > 0) return null;
  return [nodeId, port];
}

function sameEndpoint(left: Endpoint, right: Endpoint): boolean {
  return left[0] === right[0] && left[1] === right[1];
}

function endpointFromConnection(
  connection: AcausalConnectionSpec,
  nodeId: string,
  port: string
): Endpoint | null {
  const endpoint: Endpoint = [nodeId, port];
  if (sameEndpoint(connection.a, endpoint)) return connection.b;
  if (sameEndpoint(connection.b, endpoint)) return connection.a;
  return null;
}

function nodeLabel(nodeId: string, node: ComponentSpec): string {
  if (node.type === 'Anchor') return 'Anchor';
  if (node.type === 'PlanarLink') {
    if (nodeId === 'upper') return 'upper-arm link';
    if (nodeId === 'forearm') return 'forearm link';
    return `${nodeId} link`;
  }
  if (node.type === 'RevoluteJoint') return `${nodeId} joint`;
  if (node.type === 'MusclePath') return nodeId.replace(/_/g, ' ');
  if (node.type === 'PointMarker') return `${nodeId.replace(/_/g, ' ')} marker`;
  return nodeId;
}

function rowKind(node: ComponentSpec): AssemblyRowKind {
  if (node.type === 'Anchor') return 'anchor';
  if (node.type === 'PlanarLink') return 'link';
  if (node.type === 'RevoluteJoint') return 'joint';
  if (node.type === 'MusclePath') return 'muscle';
  if (node.type === 'PointMarker') return 'marker';
  return 'adapter';
}

function sortNodeIds(graph: AcausalGraphSpec, type: string): string[] {
  return Object.entries(graph.nodes ?? {})
    .filter(([, node]) => node.type === type)
    .map(([nodeId]) => nodeId)
    .sort((left, right) => left.localeCompare(right, undefined, { numeric: true }));
}

export function frameOptionsForGraph(graph: AcausalGraphSpec): AssemblyFrameOption[] {
  return Object.entries(graph.nodes ?? {})
    .flatMap(([nodeId, node]) => {
      if (node.type === 'WorldFrame') {
        return [{ id: `${nodeId}.frame`, node_id: nodeId, port: 'frame', label: `${nodeId}.frame` }];
      }
      if (node.type === 'PlanarLink') {
        return ['proximal', 'distal'].map((port) => ({
          id: `${nodeId}.${port}`,
          node_id: nodeId,
          port,
          label: `${nodeLabel(nodeId, node)} ${port}`,
        }));
      }
      return [];
    })
    .sort((left, right) => left.id.localeCompare(right.id));
}

function connectionFrame(graph: AcausalGraphSpec, nodeId: string, port: string): string | null {
  for (const connection of graph.connections ?? []) {
    const other = endpointFromConnection(connection, nodeId, port);
    if (other) return endpointKey(other);
  }
  return null;
}

function frameParam(node: ComponentSpec, name: string): string | null {
  const value = node.params?.[name];
  return typeof value === 'string' && value ? value : null;
}

function childLinkForJoint(graph: AcausalGraphSpec, jointId: string): string | null {
  const node = graph.nodes?.[jointId];
  const frame = frameParam(node, 'child_frame') ?? connectionFrame(graph, jointId, 'child');
  const parsed = frame ? parseFrameId(frame) : null;
  return parsed && graph.nodes?.[parsed[0]]?.type === 'PlanarLink' ? parsed[0] : null;
}

function parentLinkForJoint(graph: AcausalGraphSpec, jointId: string): string | null {
  const node = graph.nodes?.[jointId];
  const frame = frameParam(node, 'parent_frame') ?? connectionFrame(graph, jointId, 'parent');
  const parsed = frame ? parseFrameId(frame) : null;
  return parsed && graph.nodes?.[parsed[0]]?.type === 'PlanarLink' ? parsed[0] : null;
}

function diagnosticCodesByNode(diagnostics: DomainDiagnostic[]): Map<string, string[]> {
  const result = new Map<string, string[]>();
  for (const diagnostic of diagnostics) {
    for (const nodeId of diagnostic.node_ids ?? []) {
      result.set(nodeId, [...(result.get(nodeId) ?? []), diagnostic.code]);
    }
  }
  return result;
}

function muscleAttachmentCount(graph: AcausalGraphSpec, frameId: string): number {
  return sortNodeIds(graph, 'MusclePath').filter((nodeId) => {
    const points = graph.nodes[nodeId].params?.path_points;
    return Array.isArray(points) && points.some((point) => {
      if (!point || typeof point !== 'object') return false;
      return (point as { frame?: unknown }).frame === frameId;
    });
  }).length;
}

export function projectMechanicsAssembly(
  graph: AcausalGraphSpec | null | undefined,
  diagnostics: DomainDiagnostic[] = [],
  ownerLabel = 'composite'
): AssemblyProjection {
  if (!graph) {
    return {
      rows: [],
      frames: [],
      diagnostics: [
        {
          severity: 'error',
          code: ASSEMBLY_MISSING_INTERIOR,
          message: `Cannot show assembly for ${ownerLabel}: this composite has no materialized acausal subgraph.`,
          node_ids: [],
        },
      ],
    };
  }
  if (graph.physical_domain !== 'planar_multibody') {
    return {
      rows: [],
      frames: [],
      diagnostics: [
        {
          severity: 'error',
          code: ASSEMBLY_UNSUPPORTED_DOMAIN,
          message: `Assembly view supports planar multibody interiors, not ${graph.physical_domain}.`,
          node_ids: [],
        },
      ],
    };
  }

  const frames = frameOptionsForGraph(graph);
  const byNode = diagnosticCodesByNode(diagnostics);
  const rows: AssemblyRow[] = [];
  const seen = new Set<string>();
  const jointsByParent = new Map<string, string[]>();
  const inboundJointByChild = new Map<string, string[]>();
  for (const jointId of sortNodeIds(graph, 'RevoluteJoint')) {
    const parentLink = parentLinkForJoint(graph, jointId);
    const childLink = childLinkForJoint(graph, jointId);
    if (parentLink) jointsByParent.set(parentLink, [...(jointsByParent.get(parentLink) ?? []), jointId]);
    if (childLink) inboundJointByChild.set(childLink, [...(inboundJointByChild.get(childLink) ?? []), jointId]);
  }

  const addNodeRow = (nodeId: string, depth: number, frameId?: string | null) => {
    if (seen.has(nodeId)) return;
    const node = graph.nodes[nodeId];
    if (!node) return;
    seen.add(nodeId);
    const length = node.type === 'PlanarLink' && typeof node.params?.length === 'number'
      ? node.params.length
      : null;
    const sockets = node.type === 'RevoluteJoint'
      ? (['parent', 'child'] as const).map((kind) => {
          const connected = connectionFrame(graph, nodeId, kind);
          const declared = frameParam(node, `${kind}_frame`);
          const frame_id = connected ?? declared;
          const missing = frame_id && !frames.some((frame) => frame.id === frame_id);
          return {
            kind,
            frame_id,
            options: frames,
            diagnostic_codes: missing ? [ASSEMBLY_INVALID_FRAME_REBIND] : [],
          };
        })
      : undefined;
    rows.push({
      id: nodeId,
      kind: rowKind(node),
      node_id: nodeId,
      label: nodeLabel(nodeId, node),
      depth,
      frame_id: frameId ?? null,
      attachment_count: frameId ? muscleAttachmentCount(graph, frameId) : undefined,
      length,
      sockets,
      diagnostic_codes: byNode.get(nodeId) ?? [],
    });
  };

  const visitLink = (linkId: string, depth: number) => {
    addNodeRow(linkId, depth);
    for (const jointId of inboundJointByChild.get(linkId) ?? []) addNodeRow(jointId, depth + 1);
    for (const jointId of jointsByParent.get(linkId) ?? []) {
      const childLink = childLinkForJoint(graph, jointId);
      if (childLink) {
        visitLink(childLink, depth + 1);
        addNodeRow(jointId, depth + 2);
      } else {
        addNodeRow(jointId, depth + 1);
      }
    }
  };

  const anchorIds = sortNodeIds(graph, 'Anchor');
  for (const anchorId of anchorIds) {
    const anchor = graph.nodes[anchorId];
    const rootFrame = frameParam(anchor, 'frame') ?? connectionFrame(graph, anchorId, 'frame');
    addNodeRow(anchorId, 0, rootFrame);
    const root = rootFrame ? parseFrameId(rootFrame)?.[0] : null;
    if (root && graph.nodes[root]?.type === 'PlanarLink') visitLink(root, 1);
  }
  for (const linkId of sortNodeIds(graph, 'PlanarLink')) visitLink(linkId, rows.length ? 1 : 0);

  for (const markerId of sortNodeIds(graph, 'PointMarker')) {
    addNodeRow(markerId, 1, frameParam(graph.nodes[markerId], 'frame'));
  }
  for (const muscleId of sortNodeIds(graph, 'MusclePath')) {
    addNodeRow(muscleId, 1);
  }

  return { rows, frames, diagnostics };
}

function replaceConnectionFrame(
  connections: AcausalConnectionSpec[],
  jointId: string,
  socket: JointSocketKind,
  frame: Endpoint
): AcausalConnectionSpec[] {
  const jointEndpoint: Endpoint = [jointId, socket];
  let replaced = false;
  const next = connections.map((connection) => {
    if (sameEndpoint(connection.a, jointEndpoint)) {
      replaced = true;
      return { a: frame, b: jointEndpoint };
    }
    if (sameEndpoint(connection.b, jointEndpoint)) {
      replaced = true;
      return { a: jointEndpoint, b: frame };
    }
    return connection;
  });
  return replaced ? next : [...next, { a: jointEndpoint, b: frame }];
}

export function editLinkLength(
  graph: AcausalGraphSpec,
  linkId: string,
  length: number
): AssemblyEditResult {
  const node = graph.nodes?.[linkId];
  if (!node || node.type !== 'PlanarLink' || !Number.isFinite(length) || length <= 0) {
    return {
      ok: false,
      diagnostic: {
        severity: 'error',
        code: 'mechanics.assembly.invalid_link_length',
        message: `Link length for ${linkId} must be a positive finite number.`,
        node_ids: [linkId],
      },
    };
  }
  return {
    ok: true,
    graph: {
      ...graph,
      nodes: {
        ...graph.nodes,
        [linkId]: {
          ...node,
          params: { ...(node.params ?? {}), length },
        },
      },
    },
  };
}

export function rebindJointFrame(
  graph: AcausalGraphSpec,
  jointId: string,
  socket: JointSocketKind,
  frameId: string
): AssemblyEditResult {
  const joint = graph.nodes?.[jointId];
  const frame = parseFrameId(frameId);
  const validFrame = frameOptionsForGraph(graph).some((option) => option.id === frameId);
  if (!joint || joint.type !== 'RevoluteJoint' || !frame || !validFrame) {
    return {
      ok: false,
      diagnostic: {
        severity: 'error',
        code: ASSEMBLY_INVALID_FRAME_REBIND,
        message: `Cannot bind ${jointId}.${socket} to missing frame ${frameId}.`,
        node_ids: joint ? [jointId] : [],
        details: { joint_id: jointId, socket, frame_id: frameId },
      },
    };
  }
  return {
    ok: true,
    graph: {
      ...graph,
      nodes: {
        ...graph.nodes,
        [jointId]: {
          ...joint,
          params: { ...(joint.params ?? {}), [`${socket}_frame`]: frameId },
        },
      },
      connections: replaceConnectionFrame(graph.connections ?? [], jointId, socket, frame),
    },
  };
}
