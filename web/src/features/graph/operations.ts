import type { GraphSpec, WireSpec, ComponentSpec } from '@/types/graph';

export function addNode(
  graph: GraphSpec,
  name: string,
  component: ComponentSpec
): GraphSpec {
  if (name in graph.nodes) {
    throw new Error(`Node '${name}' already exists`);
  }
  return {
    ...graph,
    nodes: { ...graph.nodes, [name]: component },
  };
}

export function removeNode(graph: GraphSpec, name: string): GraphSpec {
  if (!(name in graph.nodes)) {
    throw new Error(`Node '${name}' does not exist`);
  }

  const newNodes = { ...graph.nodes };
  delete newNodes[name];

  const newWires = graph.wires.filter(
    (w) => w.source_node !== name && w.target_node !== name
  );

  const newInputBindings = { ...graph.input_bindings };
  const newOutputBindings = { ...graph.output_bindings };

  for (const [key, [nodeName]] of Object.entries(newInputBindings)) {
    if (nodeName === name) delete newInputBindings[key];
  }
  for (const [key, [nodeName]] of Object.entries(newOutputBindings)) {
    if (nodeName === name) delete newOutputBindings[key];
  }

  return {
    ...graph,
    nodes: newNodes,
    wires: newWires,
    input_bindings: newInputBindings,
    output_bindings: newOutputBindings,
  };
}

export function addWire(graph: GraphSpec, wire: WireSpec): GraphSpec {
  const sourceNode = graph.nodes[wire.source_node];
  if (!sourceNode) {
    throw new Error(`Source node '${wire.source_node}' does not exist`);
  }
  if (!sourceNode.output_ports.includes(wire.source_port)) {
    throw new Error(
      `Source port '${wire.source_port}' does not exist on '${wire.source_node}'`
    );
  }

  const targetNode = graph.nodes[wire.target_node];
  if (!targetNode) {
    throw new Error(`Target node '${wire.target_node}' does not exist`);
  }
  if (!targetNode.input_ports.includes(wire.target_port)) {
    throw new Error(
      `Target port '${wire.target_port}' does not exist on '${wire.target_node}'`
    );
  }

  const exists = graph.wires.some(
    (w) =>
      w.source_node === wire.source_node &&
      w.source_port === wire.source_port &&
      w.target_node === wire.target_node &&
      w.target_port === wire.target_port
  );
  if (exists) {
    throw new Error('Wire already exists');
  }

  return {
    ...graph,
    wires: [...graph.wires, wire],
  };
}

export function removeWire(graph: GraphSpec, wire: WireSpec): GraphSpec {
  return {
    ...graph,
    wires: graph.wires.filter(
      (w) =>
        !(
          w.source_node === wire.source_node &&
          w.source_port === wire.source_port &&
          w.target_node === wire.target_node &&
          w.target_port === wire.target_port
        )
    ),
  };
}

export function insertBetween(
  graph: GraphSpec,
  nodeName: string,
  component: ComponentSpec,
  wire: WireSpec,
  inputPort: string = 'input',
  outputPort: string = 'output'
): GraphSpec {
  let newGraph = removeWire(graph, wire);
  newGraph = addNode(newGraph, nodeName, component);
  newGraph = addWire(newGraph, {
    source_node: wire.source_node,
    source_port: wire.source_port,
    target_node: nodeName,
    target_port: inputPort,
  });
  newGraph = addWire(newGraph, {
    source_node: nodeName,
    source_port: outputPort,
    target_node: wire.target_node,
    target_port: wire.target_port,
  });
  return newGraph;
}
