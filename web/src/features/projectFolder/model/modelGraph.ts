import { joinText, referentName, type ShownText } from '@/components/ui/ShownText';
import type {
  AddComponent, AddModelAttachment, ComponentDefinition, ModelAddress, ModelPhysicalConnection, RegisteredComponent, ModelRemovedElement, PortElement, PortType, SignalType,
  StoredCompositeComponent, StoredDelayAttachment, StoredGraphInterior, StoredHoldAttachment, StoredInterventionAttachment,
  StoredLeafComponent, StoredModelDocument, StoredModelPorts, StoredModelWire, StoredPhysicalInterior, StoredProbeAttachment,
} from '@/generated/studioContracts';
import { canonicalJsonV2 } from '@/utils/canonicalJsonV2';
import { studioName } from '@/components/activity/studioVocabulary';
import { humanizeLabel } from '@/components/values/humanizeLabel';

export const modelSchema = 'feedbax.spec.model.v2';
export type Component = StoredLeafComponent | StoredCompositeComponent;
export type GraphOwner = StoredModelDocument | StoredCompositeComponent;
export type GraphInterior = StoredGraphInterior;
export type PhysicalInterior = StoredPhysicalInterior;
export type ModelWire = StoredModelWire;
/** An attachment as a chain stores it, under its key. */
export type StoredAttachment = StoredDelayAttachment | StoredHoldAttachment | StoredInterventionAttachment | StoredProbeAttachment;
/** An attachment of a chain, with its key as `id`, in the chain's order. */
export type Attachment = StoredAttachment & { readonly id: string };
/** A chain of attachments, keyed and in order. */
export type AttachmentChain = Readonly<Record<string, StoredAttachment>>;
export type Direction = 'input' | 'output' | 'physical';
export type Port = {
  readonly end: PortElement['end'];
  readonly direction: Direction;
  readonly name: string;
  readonly type: SignalType | string;
  readonly mark?: 'task' | 'objective' | 'intervention';
  readonly taskSignal?: string;
  readonly growing?: boolean;
};
export type Registry = ReadonlyMap<string, ComponentDefinition>;

/** The attachment `item` as its chain stores it, under its key. */
export function storedAttachment({ id: _key, ...item }: Attachment): StoredAttachment {
  return item;
}

/** The attachments of `chain` in its order, each with its key. */
export function chainItems(chain: AttachmentChain | undefined): Attachment[] {
  return Object.entries(chain ?? {}).map(([id, item]) => ({ ...item, id }));
}

/** The kind, size and unit of a port's type, as its card shows it beside the port's name; a unit says a value is real. */
export function portTypeName({ type }: Port) {
  if (typeof type === 'string') return humanizeLabel(type);
  if (type.space) return [studioName(type.space === 'point' ? 'pointType' : 'vectorType'), type.unit].filter(Boolean).join(' ');
  const kind = type.base && !(type.base === 'real' && type.unit) ? humanizeLabel(type.base) : null;
  return [kind, shapeText(type), type.unit].filter(Boolean).join(' ');
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

export function modelPath(model: StoredModelDocument, path: readonly string[]) {
  let owner: GraphOwner = model;
  const entries: { id: string; name: string }[] = [];
  for (const id of path) {
    const component: Component | undefined = owner.interior.components?.[id];
    if (component?.kind !== 'composite') break;
    owner = component;
    entries.push({ id, name: component.label });
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
export function newEquations(entry: LibraryEntry & { equations: EquationsDomain }): StoredCompositeComponent {
  return {
    kind: 'composite', label: entry.name, ports: {},
    interior: { physical_domain: entry.equations, components: {}, wires: {}, connections: {} },
  };
}

function signalType(type: PortType | undefined): SignalType {
  return { base: type?.dtype, shape: type?.shape };
}

function leafPorts(id: string, component: StoredLeafComponent, definition: ComponentDefinition | undefined): Port[] {
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
  ] as const).flatMap(([direction, ports]) => Object.entries(ports ?? {}).map(([key, port]): Port => ({
    end: { owner: 'component', component: id, port: key, direction }, direction, name: port.label, type: port.type,
  })));
}

export function boundaryPorts(ports: StoredModelPorts): Port[] {
  return [
    ...Object.entries(ports.inputs ?? {}).map(([id, { label, type }]) => ({
      end: { owner: 'inputs', port: id }, direction: 'output', name: label, type,
    }) satisfies Port),
    ...Object.entries(ports.outputs ?? {}).map(([id, { label, type }]) => ({
      end: { owner: 'outputs', port: id }, direction: 'input', name: label, type,
    }) satisfies Port),
    ...Object.entries(ports.physical ?? {}).map(([id, { label, type }]) => ({
      end: { owner: 'boundary', port: id }, direction: 'physical', name: label, type,
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

export function graphKind(component: StoredCompositeComponent) {
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

/**
 * `component` as an edit sends it. It stands in until the Service takes stored components, keyed by the edit's address.
 */
export const sentComponent = <Sent extends Component | RegisteredComponent>(component: Sent) =>
  component as unknown as Sent extends RegisteredComponent ? RegisteredComponent : Exclude<AddComponent['component'], RegisteredComponent>;

/**
 * `item` as an edit sends it, its key in the edit's address. It stands in until the Service takes stored attachments.
 */
export const sentAttachment = (item: Attachment) => storedAttachment(item) as unknown as AddModelAttachment['attachment'];

/** A removed element as its document stored it. */
export interface StoredRemoved {
  readonly address: ModelAddress;
  readonly definition: Component | ModelWire | ModelPhysicalConnection | StoredAttachment;
  readonly wires?: Readonly<Record<string, ModelWire>>;
  readonly connections?: Readonly<Record<string, ModelPhysicalConnection>>;
}

/**
 * `removed` in its stored form. It stands in until the Service sends removed definitions stored, the member key in the
 * address.
 */
export const storedRemoved = (removed: ModelRemovedElement): StoredRemoved => removed as unknown as StoredRemoved;

/** `model` with each of `removed` put back where it was, so that what a document or value removes can be drawn in place. */
export function withRemoved(model: StoredModelDocument, removed: readonly ModelRemovedElement[]): StoredModelDocument {
  if (!removed.length) return model;
  const drawn = structuredClone(model);
  for (const { address, definition, wires, connections } of removed.map(storedRemoved)) {
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
      const item = definition as StoredAttachment;
      if (parent.kind === 'wire') {
        const wire = interior.wires?.[parent.id];
        if (wire) wire.attachments = { ...wire.attachments, [element.id]: item };
        continue;
      }
      const end = parent.end;
      const holder = end.owner === 'component' ? interior.components?.[end.component] : interior;
      if (!holder || end.owner === 'boundary' || end.owner === 'component' && end.direction === 'physical') continue;
      const store = holder.attachments ??= {};
      const side = end.owner === 'inputs' || end.owner === 'component' && end.direction === 'output' ? 'sources' : 'sinks';
      const chains = (store[side] ??= {}) as Record<string, AttachmentChain>;
      chains[end.port] = { ...chains[end.port], [element.id]: item };
    }
  }
  return drawn;
}

/** An intervention of a Model, by its address, with the name its shelf row reads: its operation and where it acts. */
export interface ModelIntervention {
  readonly address: ModelAddress;
  readonly name: ShownText;
}

/**
 * Every intervention of `model`, on a wire or a source port of any of its graphs, nested ones included, its ports named
 * as `registry` names its components' ports.
 */
export function modelInterventions(model: StoredModelDocument, registry: Registry): ModelIntervention[] {
  const found: ModelIntervention[] = [];
  const walk = (owner: GraphOwner, composites: string[]) => {
    const components = owner.interior.components ?? {};
    const boundary = boundaryPorts(owner.ports);
    const endName = (end: { owner: string; component?: string; port: string }): ShownText => {
      if (end.owner !== 'component') return [referentName(boundary.find((item) => item.end.owner === end.owner && item.end.port === end.port), end.port)];
      const component = components[end.component!];
      const port = component && componentPorts(end.component!, component, registry).find((item) => item.end.port === end.port);
      return joinText([[component ? component.label : { missing: end.component! }], [referentName(port, end.port)]], '.');
    };
    const add = (parent: ModelAddress['element'], where: ShownText, chain: AttachmentChain | undefined) => {
      for (const item of chainItems(chain)) {
        if (item.kind !== 'intervention') continue;
        found.push({
          address: { composites, element: { kind: 'attachment', parent, id: item.id } as ModelAddress['element'] },
          name: joinText([humanizeLabel(item.operation ?? ''), where], ' '),
        });
      }
    };
    if (!('physical_domain' in owner.interior) || !owner.interior.physical_domain) {
      const interior = owner.interior;
      for (const [id, wire] of Object.entries(interior.wires ?? {})) {
        add({ kind: 'wire', id }, joinText([endName(wire.source), endName(wire.target)], ' → '), wire.attachments);
      }
      for (const [port, items] of Object.entries(interior.attachments?.sources ?? {})) {
        add({ kind: 'port', end: { owner: 'inputs', port } }, endName({ owner: 'inputs', port }), items);
      }
    }
    for (const [id, component] of Object.entries(components)) {
      for (const [port, items] of Object.entries(component.attachments?.sources ?? {})) {
        add({ kind: 'port', end: { owner: 'component', component: id, port, direction: 'output' } },
          endName({ owner: 'component', component: id, port }), items);
      }
      if (component.kind === 'composite') walk(component, [...composites, id]);
    }
  };
  walk(model, []);
  return found;
}
