import type {
  ComponentDefinition, CompositeComponent, GraphInterior, LeafComponent, ModelAddress, ModelDocument,
  ModelPhysicalConnection, ModelPorts, ModelRemovedElement, ModelWire, PhysicalInterior, PortElement, PortType, SignalType,
} from '@/generated/studioContracts';
import { canonicalJsonV2 } from '@/utils/canonicalJsonV2';
import { studioName } from '@/components/activity/studioVocabulary';
import { humanizeLabel } from '@/components/values/humanizeLabel';

export const modelSchema = 'feedbax.spec.model';
export type Component = LeafComponent | CompositeComponent;
export type GraphOwner = ModelDocument | CompositeComponent;
export type Direction = 'input' | 'output' | 'physical';
export type Port = {
  readonly end: PortElement['end'];
  readonly direction: Direction;
  readonly name: string;
  readonly type: SignalType | string;
  readonly mark?: 'task' | 'objective';
  readonly taskSignal?: string;
  readonly growing?: boolean;
};
export type Registry = ReadonlyMap<string, ComponentDefinition>;
type ModelAttachment = NonNullable<ModelWire['attachments']>[number];

/** The kind, size and unit of a port's type, as its card shows it beside the port's name; a real value's kind goes unsaid. */
export function portTypeName({ type }: Port) {
  if (typeof type === 'string') return humanizeLabel(type);
  if (type.space) return [studioName(type.space === 'point' ? 'pointType' : 'vectorType'), type.unit].filter(Boolean).join(' ');
  return [type.base !== 'real' && type.base && humanizeLabel(type.base), shapeText(type), type.unit].filter(Boolean).join(' ');
}

const shapeText = (type: SignalType) => type.shape?.length ? `[${type.shape.join(', ')}]` : null;

export function portTypeLabel(port: Port) {
  const { type } = port;
  if (typeof type === 'string') return humanizeLabel(type);
  return [type.space ? portTypeName(port) : type.unit, type.base, shapeText(type)].filter(Boolean).join(' ');
}

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

/**
 * Whether `definition` is a template that places a composite. A leaf can carry a template too, its own expansion, and is
 * still placed as a leaf.
 */
export function isTemplate(definition: ComponentDefinition) {
  return !!definition.is_composite && !!definition.template_kind;
}

export type EquationsDomain = NonNullable<ComponentDefinition['equations_domains']>[number];
type Domain = PhysicalInterior['physical_domain'] | null;

/**
 * A row of the Library: a definition placed as itself, or, for a definition that starts Equations composites, a new
 * composite of one of its domains.
 */
export type LibraryEntry = {
  readonly id: string;
  readonly name: string;
  readonly group: string;
  readonly definition: ComponentDefinition;
  readonly equations?: EquationsDomain;
};

/** Whether `definition`'s conserving ports all fit `domain`: a port that states no domain fits any. */
function conservesIn(definition: ComponentDefinition, domain: Domain) {
  const types = definition.port_types;
  return [...Object.values(types?.inputs ?? {}), ...Object.values(types?.outputs ?? {})]
    .every((type) => type.kind !== 'conserving' || !type.physical_domain || type.physical_domain === domain);
}

/**
 * The Library's rows in a graph of `domain`. A causal graph holds causal definitions, templates and new Equations
 * composites; an Equations interior holds the elements whose conserving ports are of its domain, and an Equations
 * composite of the same domain.
 */
export function libraryEntries(definitions: readonly ComponentDefinition[], domain: Domain): LibraryEntry[] {
  return definitions.flatMap((definition): LibraryEntry[] => {
    const own = { id: definition.name, name: shortName(definition.name), definition };
    const domains = definition.equations_domains ?? [];
    if (domains.length) {
      return domains.filter((each) => !domain || each === domain).map((equations) => ({
        id: `${definition.name}:${equations}`, name: humanizeLabel(equations), group: studioName('equations'), definition, equations,
      }));
    }
    if (definition.is_composite && !isTemplate(definition)) return [];
    const group = isTemplate(definition) ? studioName('templates') : definition.category;
    if (!domain) return definition.domain === 'feedbax.domain.causal' ? [{ ...own, group }] : [];
    if (definition.is_composite) return [];
    const mechanics = domain === 'planar_multibody' || domain === 'spatial_multibody';
    if (mechanics) return definition.domain === 'feedbax.domain.acausal' || definition.domain === 'feedbax.domain.mechanics'
      ? [{ ...own, group }] : [];
    return definition.domain === 'feedbax.domain.acausal' && conservesIn(definition, domain) ? [{ ...own, group }] : [];
  });
}

/** The component an entry places while its add is pending, and, for a new Equations composite, the one it sends. */
export function newEquations(entry: LibraryEntry & { equations: EquationsDomain }): CompositeComponent {
  return {
    kind: 'composite', name: entry.name, ports: { inputs: [], outputs: [], physical: [] },
    interior: { physical_domain: entry.equations, components: {}, wires: {}, connections: {} },
  };
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
        direction: kind, name: grow === port ? studioName('addInput') : port, growing: grow === port,
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

export function attachmentAddress(owner: ModelAddress, id: string): ModelAddress {
  if (owner.element.kind !== 'wire' && owner.element.kind !== 'port') throw new Error('Expected an attachment owner');
  return at(owner, { kind: 'attachment', parent: owner.element, id });
}

export function fieldAddress(address: ModelAddress, ...path: (string | number)[]): ModelAddress {
  const parent = address.element;
  if (parent.kind === 'field' || parent.kind === 'inputs' || parent.kind === 'outputs') {
    throw new Error('A field must name its owning element');
  }
  return at(address, { kind: 'field', parent, path });
}

export function graphKind(component: CompositeComponent) {
  const domain = component.interior.physical_domain;
  return domain === 'planar_multibody' || domain === 'spatial_multibody' ? 'Mechanics' : domain ? studioName('equations') : 'Graph';
}

type Element = ModelAddress['element'];

function owners(element: Element): Element[] {
  if (element.kind === 'field' || element.kind === 'attachment') return [element, ...owners(element.parent)];
  if (element.kind === 'port' && element.end.owner === 'component') {
    return [element, { kind: 'component', id: element.end.component }];
  }
  return [element];
}

/** Whether `address` is `holder` or lies inside it: a field of it, an attachment on it, a port of it, or an element of a composite's interior. */
export function holds(holder: ModelAddress, address: ModelAddress) {
  const depth = holder.composites.length;
  if (!holder.composites.every((id, index) => address.composites[index] === id)) return false;
  if (holder.element.kind === 'component' && address.composites[depth] === holder.element.id) return true;
  if (address.composites.length !== depth) return false;
  const key = canonicalJsonV2(holder.element);
  return owners(address.element).some((element) => canonicalJsonV2(element) === key);
}

/** `model` with each of `removed` put back where it was, so that what a document or value removes can be drawn in place. */
export function withRemoved(model: ModelDocument, removed: readonly ModelRemovedElement[]): ModelDocument {
  if (!removed.length) return model;
  const drawn = structuredClone(model);
  for (const { address, definition, wires, connections } of removed) {
    const reached = modelPath(drawn, address.composites);
    if (reached.entries.length !== address.composites.length) continue;
    const interior: Omit<GraphInterior, 'physical_domain'> & Partial<Pick<PhysicalInterior, 'connections'>> = reached.owner.interior;
    const element = address.element;
    if (element.kind === 'component') {
      interior.components = { ...interior.components, [element.id]: definition as Component };
      interior.wires = { ...interior.wires, ...wires };
      if (connections && Object.keys(connections).length) interior.connections = { ...interior.connections, ...connections };
    } else if (element.kind === 'wire') {
      interior.wires = { ...interior.wires, [element.id]: definition as ModelWire };
    } else if (element.kind === 'physical_connection') {
      interior.connections = { ...interior.connections, [element.id]: definition as ModelPhysicalConnection };
    } else if (element.kind === 'attachment') {
      const parent = element.parent;
      const item = definition as ModelAttachment;
      if (parent.kind === 'wire') {
        const wire = interior.wires?.[parent.id];
        if (wire) wire.attachments = [...wire.attachments ?? [], item];
        continue;
      }
      const end = parent.end;
      const holder = end.owner === 'component' ? interior.components?.[end.component] : interior;
      if (!holder || end.owner === 'boundary' || end.owner === 'component' && end.direction === 'physical') continue;
      const store = holder.attachments ??= {};
      const side = end.owner === 'inputs' || end.owner === 'component' && end.direction === 'output' ? 'sources' : 'sinks';
      const chains = (store[side] ??= {}) as Record<string, ModelAttachment[]>;
      chains[end.port] = [...chains[end.port] ?? [], item];
    }
  }
  return drawn;
}
