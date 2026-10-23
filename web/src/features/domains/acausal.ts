import type { Edge } from '@xyflow/react';
import { Position, type Connection } from '@xyflow/react';
import type { ComponentDefinition, PortType } from '@/types/components';
import type {
  AcausalConnectionSpec,
  AcausalGraphSpec,
  DomainCompileReport,
} from '@/generated/studioContracts';
import type { GraphEdgeData, GraphNodeData } from '@/types/graph';

export type CompileStatus =
  | 'never_compiled'
  | 'stale'
  | 'compiling'
  | 'ok'
  | 'ok_with_warnings'
  | 'error';

export type AcausalConnectionVerdict =
  | { allowed: true; message: string }
  | { allowed: false; reason: string };

type Endpoint = [string, string];

const ADAPTER_MESSAGE =
  'Signal and conserving ports connect through ActuationInput or SensorOutput adapters.';

function endpointKey(endpoint: Endpoint): string {
  return `${endpoint[0]}:${endpoint[1]}`;
}

export function canonicalAcausalConnectionId(a: Endpoint, b: Endpoint): string {
  const [left, right] = [endpointKey(a), endpointKey(b)].sort();
  return `acausal:${left}|${right}`;
}

export function acausalConnectionId(connection: AcausalConnectionSpec): string {
  return canonicalAcausalConnectionId(connection.a, connection.b);
}

export function graphPathKey(path: string[]): string {
  return path.join('/');
}

function rightRotate(value: number, amount: number): number {
  return (value >>> amount) | (value << (32 - amount));
}

function sha256(text: string): string {
  const bytes = new TextEncoder().encode(text);
  const bitLength = bytes.length * 8;
  const paddedLength = (((bytes.length + 9 + 63) >> 6) << 6);
  const data = new Uint8Array(paddedLength);
  data.set(bytes);
  data[bytes.length] = 0x80;
  const view = new DataView(data.buffer);
  view.setUint32(paddedLength - 4, bitLength, false);

  const k = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
    0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
    0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
    0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
    0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
    0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2,
  ];
  const h = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ];
  const w = new Array<number>(64);
  for (let offset = 0; offset < data.length; offset += 64) {
    for (let i = 0; i < 16; i += 1) w[i] = view.getUint32(offset + i * 4, false);
    for (let i = 16; i < 64; i += 1) {
      const s0 = rightRotate(w[i - 15], 7) ^ rightRotate(w[i - 15], 18) ^ (w[i - 15] >>> 3);
      const s1 = rightRotate(w[i - 2], 17) ^ rightRotate(w[i - 2], 19) ^ (w[i - 2] >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }
    let [a, b, c, d, e, f, g, hh] = h;
    for (let i = 0; i < 64; i += 1) {
      const s1 = rightRotate(e, 6) ^ rightRotate(e, 11) ^ rightRotate(e, 25);
      const ch = (e & f) ^ (~e & g);
      const temp1 = (hh + s1 + ch + k[i] + w[i]) >>> 0;
      const s0 = rightRotate(a, 2) ^ rightRotate(a, 13) ^ rightRotate(a, 22);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = (s0 + maj) >>> 0;
      hh = g;
      g = f;
      f = e;
      e = (d + temp1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temp1 + temp2) >>> 0;
    }
    h[0] = (h[0] + a) >>> 0;
    h[1] = (h[1] + b) >>> 0;
    h[2] = (h[2] + c) >>> 0;
    h[3] = (h[3] + d) >>> 0;
    h[4] = (h[4] + e) >>> 0;
    h[5] = (h[5] + f) >>> 0;
    h[6] = (h[6] + g) >>> 0;
    h[7] = (h[7] + hh) >>> 0;
  }
  return h.map((value) => value.toString(16).padStart(8, '0')).join('');
}

function canonicalAcausalPayload(graph: AcausalGraphSpec): unknown {
  const normalize = (item: unknown): unknown => {
    if (item === null || item === undefined) return undefined;
    if (Array.isArray(item)) return item.map(normalize).filter((value) => value !== undefined);
    if (typeof item !== 'object') return item;
    const entries = Object.entries(item as Record<string, unknown>)
      .flatMap(([key, value]) => {
        const normalized = normalize(value);
        return normalized === undefined ? [] : [[key, normalized] as const];
      })
      .sort(([left], [right]) => left.localeCompare(right));
    return Object.fromEntries(entries);
  };
  const payload = normalize(graph) as Record<string, unknown>;
  payload.connections = [...((payload.connections as AcausalConnectionSpec[] | undefined) ?? [])].sort(
    (left, right) =>
      `${left.a[0]}:${left.a[1]}|${left.b[0]}:${left.b[1]}`.localeCompare(
        `${right.a[0]}:${right.a[1]}|${right.b[0]}:${right.b[1]}`
      )
  );
  if (graph.subgraphs) {
    payload.subgraphs = Object.fromEntries(
      Object.keys(graph.subgraphs)
        .sort()
        .map((key) => [key, canonicalAcausalPayload(graph.subgraphs![key])])
    );
  }
  return payload;
}

export function stableGraphHash(graph: AcausalGraphSpec): string {
  return sha256(JSON.stringify(canonicalAcausalPayload(graph)));
}

export function compileStatusForReport(
  report: DomainCompileReport | null | undefined,
  currentHash: string,
  compiling = false
): CompileStatus {
  if (compiling) return 'compiling';
  if (!report) return 'never_compiled';
  if (report.interior_content_hash !== currentHash) return 'stale';
  return report.status;
}

function statusRank(status: CompileStatus): number {
  if (status === 'error') return 5;
  if (status === 'stale') return 4;
  if (status === 'compiling') return 3;
  if (status === 'ok_with_warnings') return 2;
  if (status === 'never_compiled') return 1;
  return 0;
}

export function worstCompileStatus(statuses: CompileStatus[]): CompileStatus {
  return statuses.reduce<CompileStatus>(
    (worst, status) => (statusRank(status) > statusRank(worst) ? status : worst),
    'ok'
  );
}

export function rollupCompileStatus(args: {
  graph: AcausalGraphSpec;
  path: string[];
  reports: Record<string, DomainCompileReport>;
  compilingPaths?: Set<string>;
}): CompileStatus {
  const pathKey = graphPathKey(args.path);
  const own = compileStatusForReport(
    args.reports[pathKey],
    stableGraphHash(args.graph),
    args.compilingPaths?.has(pathKey) ?? false
  );
  const childStatuses = Object.entries(args.graph.subgraphs ?? {}).map(([nodeId, child]) =>
    rollupCompileStatus({
      ...args,
      graph: child,
      path: [...args.path, nodeId],
    })
  );
  return worstCompileStatus([own, ...childStatuses]);
}

function portTypeForEndpoint(
  graph: AcausalGraphSpec,
  registry: Map<string, ComponentDefinition>,
  endpoint: Endpoint
): PortType | null {
  const [nodeId, port] = endpoint;
  const node = graph.nodes?.[nodeId];
  if (!node) return null;
  const component = registry.get(node.type);
  return component?.port_types?.inputs?.[port] ?? component?.port_types?.outputs?.[port] ?? null;
}

export function portIsConserving(
  graph: AcausalGraphSpec,
  registry: Map<string, ComponentDefinition>,
  nodeId: string,
  port: string
): boolean {
  return portTypeForEndpoint(graph, registry, [nodeId, port])?.kind === 'conserving';
}

export function evaluateAcausalConnection(
  graph: AcausalGraphSpec,
  registry: Map<string, ComponentDefinition>,
  connection: Connection
): AcausalConnectionVerdict {
  if (!connection.source || !connection.sourceHandle || !connection.target || !connection.targetHandle) {
    return { allowed: false, reason: 'Choose two conserving ports.' };
  }
  if (connection.source === connection.target) {
    return { allowed: false, reason: 'A conserving connection cannot join a node to itself.' };
  }
  const source: Endpoint = [connection.source, connection.sourceHandle];
  const target: Endpoint = [connection.target, connection.targetHandle];
  const sourceType = portTypeForEndpoint(graph, registry, source);
  const targetType = portTypeForEndpoint(graph, registry, target);
  if (sourceType?.kind !== 'conserving' || targetType?.kind !== 'conserving') {
    return { allowed: false, reason: ADAPTER_MESSAGE };
  }
  if (
    sourceType.physical_domain &&
    targetType.physical_domain &&
    sourceType.physical_domain !== targetType.physical_domain
  ) {
    return {
      allowed: false,
      reason: `Conserving ports must share a physical domain; ${sourceType.physical_domain} cannot connect to ${targetType.physical_domain}.`,
    };
  }
  return { allowed: true, message: 'Conserving ports can connect here' };
}

export function connectionFromReactFlow(connection: Connection): AcausalConnectionSpec | null {
  if (!connection.source || !connection.sourceHandle || !connection.target || !connection.targetHandle) {
    return null;
  }
  const a: Endpoint = [connection.source, connection.sourceHandle];
  const b: Endpoint = [connection.target, connection.targetHandle];
  return endpointKey(a) <= endpointKey(b) ? { a, b } : { a: b, b: a };
}

export function acausalEdgesFromGraph(graph: AcausalGraphSpec): Edge<GraphEdgeData>[] {
  return (graph.connections ?? []).map((connection) => ({
    id: acausalConnectionId(connection),
    source: connection.a[0],
    sourceHandle: connection.a[1],
    target: connection.b[0],
    targetHandle: connection.b[1],
    sourcePosition: Position.Right,
    targetPosition: Position.Left,
    type: 'conserving',
    zIndex: 1,
    data: {
      physical_domain: graph.physical_domain,
    },
  }));
}

export function acausalConnectionsFromEdges(
  edges: Edge<GraphEdgeData>[]
): AcausalConnectionSpec[] {
  return edges.flatMap((edge) => {
    if (edge.type !== 'conserving' || !edge.source || !edge.sourceHandle || !edge.target || !edge.targetHandle) {
      return [];
    }
    return connectionFromReactFlow({
      source: edge.source,
      sourceHandle: edge.sourceHandle as string,
      target: edge.target,
      targetHandle: edge.targetHandle as string,
    }) ?? [];
  });
}

export function acausalNodeDataPatch(args: {
  graph: AcausalGraphSpec;
  registry: Map<string, ComponentDefinition>;
  reports: Record<string, DomainCompileReport>;
  path: string[];
  compilingPaths?: Set<string>;
}): Record<string, Partial<GraphNodeData>> {
  return Object.fromEntries(
    Object.entries(args.graph.nodes ?? {}).map(([nodeId]) => {
      const child = args.graph.subgraphs?.[nodeId];
      const status = child
        ? rollupCompileStatus({
            graph: child,
            path: [...args.path, nodeId],
            reports: args.reports,
            compilingPaths: args.compilingPaths,
          })
        : undefined;
      return [
        nodeId,
        {
          current_domain: 'feedbax.domain.acausal',
          status,
        },
      ];
    })
  );
}
