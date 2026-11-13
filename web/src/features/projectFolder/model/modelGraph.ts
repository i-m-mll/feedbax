import type {
  ComponentDefinition, CompositeComponent, LeafComponent, ModelAddress, ModelDocument, ModelPorts,
  PhysicalInterior, PortElement, PortType, SignalType,
} from '@/generated/studioContracts';
import { canonicalJsonV2 } from '@/utils/canonicalJsonV2';

export const modelSchema = 'feedbax.spec.model';
export type Component = LeafComponent | CompositeComponent;
export type GraphOwner = ModelDocument | CompositeComponent;
export type Direction = 'input' | 'output' | 'physical';
export type Port = {
  readonly end: PortElement['end'];
  readonly direction: Direction;
  readonly name: string;
  readonly type: SignalType | string;
  readonly mark?: 'Task' | 'Objective';
  readonly taskSignal?: string;
  readonly growing?: boolean;
};
export type Registry = ReadonlyMap<string, ComponentDefinition>;

export function addressKey(address: ModelAddress) {
  return canonicalJsonV2(address);
}

export function at(graph: ModelAddress, element: ModelAddress['element']): ModelAddress {
  return { ...graph, element };
}

export function modelPath(model: ModelDocument, path: readonly string[]) {
  let owner: GraphOwner = model;
  const entries: { id: string; name: string }[] = [];
  for (const id of path) {
    const component: Component | undefined = owner.interior.components?.[id];
    if (component?.kind !== 'composite') break;
    owner = component;
    entries.push({ id, name: component.name });
  }
  return { owner, entries };
}

export function modelRegistry(definitions: readonly ComponentDefinition[]): Registry {
  return new Map(definitions.flatMap((item) => [
    [item.name, item] as const,
    [item.identity?.type_id ?? item.name, item] as const,
  ]));
}

export function shortName(type: string) {
  return type.split('.').at(-1) ?? type;
}

export function accepts(definition: ComponentDefinition, domain: PhysicalInterior['physical_domain'] | null) {
  if (definition.is_composite) return false;
  if (!domain) return definition.domain === 'feedbax.domain.causal';
  return definition.domain === 'feedbax.domain.acausal' ||
    ((domain === 'planar_multibody' || domain === 'spatial_multibody') && definition.domain === 'feedbax.domain.mechanics');
}

function signalType(type: PortType | undefined): SignalType {
  return { base: type?.dtype, shape: type?.shape };
}

function leafPorts(id: string, component: LeafComponent, definition: ComponentDefinition | undefined): Port[] {
  if (!definition) return [];
  const policy = definition.dynamic_port_policy;
  const fields = { ...definition.default_params, ...component.fields };
  return (['input', 'output'] as const).flatMap((direction) => {
    let names = (direction === 'input' ? definition.input_ports : definition.output_ports) ?? [];
    const types = direction === 'input' ? definition.port_types?.inputs : definition.port_types?.outputs;
    const fixed = (direction === 'input' ? policy?.fixed_input_ports : policy?.fixed_output_ports) ?? [];
    if (policy?.count_mode === 'names') {
      const keys = (direction === 'input' ? policy.input_name_params : policy.output_name_params) ?? [];
      names = [...new Set<string>([...fixed, ...keys.flatMap((key): string[] => {
        const value = fields[key];
        if (typeof value === 'string') return [value];
        if (!Array.isArray(value)) return [];
        return (value as unknown[]).filter((name): name is string => typeof name === 'string');
      })])];
    } else if (policy?.direction === direction && policy.count_param && policy.generated_name_template) {
      const value = component.port_counts?.[policy.count_param] ?? fields[policy.count_param];
      const count = Array.isArray(value) ? value.length : typeof value === 'number' ? value : policy.minimum_count ?? 0;
      const template = policy.generated_name_template;
      names = [...fixed, ...Array.from({ length: count }, (_, index) =>
        template.replace('{index}', String(index + (policy.generated_index_origin ?? 0))))];
    }
    const grow = direction === 'input' && policy?.direction === 'input' && policy.auto_resize_minimum != null &&
      policy.generated_name_template ? policy.generated_name_template.replace(
        '{index}', String(names.length - fixed.length + (policy.generated_index_origin ?? 0)),
      ) : undefined;
    return (grow ? [...names, grow] : names).map((port) => {
      const type = types?.[port] ?? policy?.dynamic_port_type;
      const kind = type?.kind === 'conserving' ? 'physical' : direction;
      return {
        end: { owner: 'component', component: id, direction: kind, port },
        direction: kind, name: grow === port ? 'Add Input' : port, growing: grow === port,
        type: kind === 'physical' ? type?.physical_domain ?? '' : signalType(type),
      } satisfies Port;
    });
  }).filter((port, index, all) => port.direction !== 'physical' || all.findIndex((item) =>
    item.direction === 'physical' && item.name === port.name) === index);
}

export function componentPorts(id: string, component: Component, registry: Registry): Port[] {
  if (component.kind === 'leaf') return leafPorts(id, component, registry.get(component.type));
  return ([
    ['input', component.ports.inputs], ['output', component.ports.outputs], ['physical', component.ports.physical],
  ] as const).flatMap(([direction, ports]) => (ports ?? []).map((port): Port => ({
    end: { owner: 'component', component: id, port: port.id, direction }, direction, name: port.name, type: port.type,
  })));
}

export function boundaryPorts(ports: ModelPorts): Port[] {
  return [
    ...(ports.inputs ?? []).map(({ id, name, type }) => ({
      end: { owner: 'inputs', port: id }, direction: 'output', name, type,
    }) satisfies Port),
    ...(ports.outputs ?? []).map(({ id, name, type }) => ({
      end: { owner: 'outputs', port: id }, direction: 'input', name, type,
    }) satisfies Port),
    ...(ports.physical ?? []).map(({ id, name, type }) => ({
      end: { owner: 'boundary', port: id }, direction: 'physical', name, type,
    }) satisfies Port),
  ];
}

export function portId(port: Pick<Port, 'direction' | 'end'>) {
  return `${port.direction}:${port.end.port}`;
}

export function wireEnd(port: Port) {
  const end = port.end;
  return end.owner === 'component'
    ? { owner: end.owner, component: end.component, port: end.port } as const
    : end;
}

export function endKey(end: ReturnType<typeof wireEnd>) {
  return end.owner === 'component' ? `component:${end.component}:${end.port}` : `${end.owner}:${end.port}`;
}

export function fieldAddress(address: ModelAddress, ...path: (string | number)[]): ModelAddress {
  if (address.element.kind === 'field') throw new Error('A field must name its owning element');
  return at(address, { kind: 'field', parent: address.element, path });
}

export function graphKind(component: CompositeComponent) {
  const domain = component.interior.physical_domain;
  return domain === 'planar_multibody' || domain === 'spatial_multibody' ? 'Mechanics' : domain ? 'Equations' : 'Graph';
}
