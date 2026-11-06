import { isCausalGraphSpec, type GraphSpec, type GraphSubgraphSpec, type WireSpec } from '@/types/graph';

interface CausalInputDriver {
  node: string;
  port: string;
  owner: string;
  path: string;
}

type Driver = CausalInputDriver & (
  | { kind: 'external'; name: string }
  | { kind: 'wire' }
);

export function wireInputOwner(
  sourceNode: string,
  sourcePort: string,
  temporality: WireSpec['temporality'] = 'instant'
): string {
  return `${temporality} wire ${sourceNode}.${sourcePort}`;
}

export function* causalInputDrivers(graph: GraphSpec): Generator<Driver> {
  for (const [name, [node, port]] of Object.entries(graph.input_bindings)) {
    yield {
      kind: 'external', name, node, port,
      owner: `external input '${name}'`,
      path: `input_bindings.${name}`,
    };
  }
  for (const [index, wire] of graph.wires.entries()) {
    yield {
      kind: 'wire',
      node: wire.target_node,
      port: wire.target_port,
      owner: wireInputOwner(wire.source_node, wire.source_port, wire.temporality),
      path: `wires.${index}`,
    };
  }
}

export function findCausalInputDriver(
  graph: GraphSpec,
  node: string,
  port: string,
  accepts: (driver: Driver) => boolean = () => true
): Driver | undefined {
  for (const driver of causalInputDrivers(graph)) {
    if (driver.node === node && driver.port === port
      && accepts(driver)) {
      return driver;
    }
  }
}

export function findWireConnectionDriver(
  graph: GraphSpec,
  node: string,
  port: string,
  parent?: { graph: GraphSubgraphSpec; childNodeId?: string }
): Driver | undefined {
  return findCausalInputDriver(graph, node, port, (driver) => {
    if (driver.kind === 'wire' || !parent) return true;
    // A new interior wire replaces an automatically exposed, unused subgraph input.
    return Boolean(
      parent.childNodeId && isCausalGraphSpec(parent.graph) &&
      findCausalInputDriver(parent.graph, parent.childNodeId, driver.name)
    );
  });
}

export function inputDriverConflict(node: string, port: string, first: string, second: string): string {
  return `Input ${node}.${port} has multiple drivers: ${first} and ${second}`;
}
