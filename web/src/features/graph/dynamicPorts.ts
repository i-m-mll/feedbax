import type { ComponentDefinition, DynamicPortPolicy } from '@/generated/studioContracts';
import { isAcausalGraphSpec, type ComponentSpec, type GraphSpec, type GraphSubgraphSpec } from '@/types/graph';
import type { StudioTaskBindingSpec } from '@/types/workspace';

export type ComponentCatalogue = ReadonlyMap<string, ComponentDefinition>;

type PortDirection = 'input' | 'output';

function generatedPort(policy: DynamicPortPolicy, offset: number): string {
  return policy.generated_name_template!.split('{index}').join(
    String((policy.generated_index_origin ?? 0) + offset)
  );
}

export function generatedPortOffset(policy: DynamicPortPolicy, port: string): number | null {
  if (policy.count_mode === 'names') return null;
  const fixed = policy.direction === 'input' ? policy.fixed_input_ports : policy.fixed_output_ports;
  if (fixed?.includes(port)) return null;
  const parts = policy.generated_name_template!.split('{index}');
  if (parts.length < 2 || !port.startsWith(parts[0])) return null;
  const token = /^-?\d+/.exec(port.slice(parts[0].length))?.[0];
  if (!token) return null;

  // The suffix may start with a digit, so verify the entire generated name.
  for (let end = token.length; end > 0; end -= 1) {
    const index = Number(token.slice(0, end));
    const offset = index - (policy.generated_index_origin ?? 0);
    if (Number.isSafeInteger(offset) && offset >= 0 && generatedPort(policy, offset) === port) {
      return offset;
    }
  }
  return null;
}

export function componentPortType(
  component: ComponentDefinition | undefined,
  direction: PortDirection,
  port: string,
) {
  const explicit = component?.port_types?.[direction === 'input' ? 'inputs' : 'outputs']?.[port];
  const policy = component?.dynamic_port_policy;
  return explicit ?? (
    policy && (policy.count_mode === 'names'
      ? !(policy[direction === 'input' ? 'fixed_input_ports' : 'fixed_output_ports'] ?? []).includes(port)
      : policy.direction === direction && generatedPortOffset(policy, port) !== null)
      ? policy.dynamic_port_type
      : undefined
  );
}

function occupiedInputs(graph: GraphSpec, taskBindings?: StudioTaskBindingSpec | null) {
  const occupied = new Map<string, Set<string>>();
  const add = (node: string, port: string) => {
    let ports = occupied.get(node);
    if (!ports) occupied.set(node, ports = new Set());
    ports.add(port);
  };
  for (const wire of graph.wires) add(wire.target_node, wire.target_port);
  for (const [node, port] of Object.values(graph.input_bindings)) add(node, port);
  for (const binding of taskBindings?.bindings ?? []) add(binding.target_node_id, binding.target_port);
  return occupied;
}

function samePorts(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((port, i) => port === b[i]);
}

function authoredCount(spec: ComponentSpec, component: ComponentDefinition, policy: DynamicPortPolicy) {
  const value = Object.prototype.hasOwnProperty.call(spec.params, policy.count_param!)
    ? spec.params[policy.count_param!]
    : component.default_params[policy.count_param!];
  if (policy.count_mode === 'sequence_length') return Array.isArray(value) ? value.length : null;
  return typeof value === 'number' && Number.isSafeInteger(value) ? value : null;
}

function inputCountFromReferences(policy: DynamicPortPolicy, occupied: Iterable<string>): number {
  let count = policy.auto_resize_minimum ?? 0;
  for (const port of occupied) {
    const offset = generatedPortOffset(policy, port);
    if (offset !== null) count = Math.max(count, offset + 1);
  }
  return count;
}

export function normalizeDynamicPortNode(
  spec: ComponentSpec,
  component?: ComponentDefinition,
  occupied?: ReadonlySet<string>,
  resize = false,
): ComponentSpec {
  const policy = component?.dynamic_port_policy;
  if (!component || !policy) return spec;
  if (policy.count_mode === 'names') {
    const params = { ...component.default_params, ...spec.params };
    const names = (fields: string[]) => fields.map((field) => params[field]);
    const inputs = [...(policy.fixed_input_ports ?? []), ...names(policy.input_name_params ?? [])];
    const outputs = [...(policy.fixed_output_ports ?? []), ...names(policy.output_name_params ?? [])];
    const valid = (ports: unknown[]): ports is string[] =>
      ports.every((port) => typeof port === 'string' && port.length > 0) &&
      new Set(ports).size === ports.length;
    if (!valid(inputs) || !valid(outputs)) return spec;
    return samePorts(spec.input_ports, inputs) && samePorts(spec.output_ports, outputs)
      ? spec : { ...spec, input_ports: inputs, output_ports: outputs };
  }
  let count = authoredCount(spec, component, policy);
  let params = spec.params;
  if (policy.auto_resize_minimum != null) {
    const required = inputCountFromReferences(policy, occupied ?? []);
    count = resize ? required : Math.max(count ?? 0, required);
    if (count !== params[policy.count_param!]) params = { ...params, [policy.count_param!]: count };
  }
  if (count === null || count < (policy.minimum_count ?? 1)) return spec;
  const generated = Array.from({ length: count }, (_, offset) => generatedPort(policy, offset));
  const fixedInputs = policy.fixed_input_ports ?? [];
  const fixedOutputs = policy.fixed_output_ports ?? [];
  const input_ports = policy.direction === 'input' ? [...fixedInputs, ...generated] : fixedInputs;
  const output_ports = policy.direction === 'output' ? [...fixedOutputs, ...generated] : fixedOutputs;
  if (new Set(input_ports).size !== input_ports.length || new Set(output_ports).size !== output_ports.length) {
    return spec;
  }
  if (params === spec.params && samePorts(spec.input_ports, input_ports) && samePorts(spec.output_ports, output_ports)) {
    return spec;
  }
  return { ...spec, params, input_ports: [...input_ports], output_ports: [...output_ports] };
}

export function normalizeDynamicPorts<T extends GraphSubgraphSpec>(
  graph: T,
  components: ComponentCatalogue,
  taskBindings?: StudioTaskBindingSpec | null,
  resize = false,
): T {
  if (isAcausalGraphSpec(graph)) return graph;
  const occupied = occupiedInputs(graph, taskBindings);
  let nodes = graph.nodes;
  for (const [nodeId, spec] of Object.entries(graph.nodes)) {
    const next = normalizeDynamicPortNode(spec, components.get(spec.type), occupied.get(nodeId), resize);
    if (spec !== next) {
      if (nodes === graph.nodes) nodes = { ...nodes };
      nodes[nodeId] = next;
    }
  }
  return nodes === graph.nodes ? graph : { ...graph, nodes };
}

export function visibleDynamicInputPorts(
  graph: GraphSubgraphSpec,
  nodeId: string,
  components: ComponentCatalogue,
  taskBindings?: StudioTaskBindingSpec | null,
): { ports: string[]; nextPort: string | null } | null {
  if (isAcausalGraphSpec(graph)) return null;
  const spec = graph.nodes[nodeId];
  if (!spec) return null;
  const component = components.get(spec.type);
  const policy = component?.dynamic_port_policy;
  if (!policy || policy.direction !== 'input') return null;
  const occupied = occupiedInputs(graph, taskBindings).get(nodeId);
  const normalized = normalizeDynamicPortNode(spec, component, occupied);
  const ports = normalized.input_ports;
  const count = normalized.params[policy.count_param!];
  if (policy.auto_resize_minimum == null || typeof count !== 'number' ||
    ports.some((port) => generatedPortOffset(policy, port) !== null && !occupied?.has(port))) {
    return { ports, nextPort: null };
  }
  const nextPort = generatedPort(policy, count);
  if (ports.includes(nextPort)) return { ports, nextPort: null };
  return { ports: [...ports, nextPort], nextPort };
}
