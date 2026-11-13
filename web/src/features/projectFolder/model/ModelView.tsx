import { useQuery } from '@tanstack/react-query';
import { MarkerType, type OnConnectStart } from '@xyflow/react';
import { useEffect, useMemo, useRef, useState, type MouseEvent } from 'react';
import { useStore } from 'zustand';
import { fetchComponents } from '@/api/components';
import { editEvaluateAttachment, readEvaluateAttachments, reviewModelConnections } from '@/api/projectFolders';
import { ApiRequestError, apiErrorMessage } from '@/api/request';
import { Dialog } from '@/components/ui/Dialog';
import { canvasViewportCommands, type CanvasViewport } from '@/components/canvas/StudioCanvas';
import { bindStudioCommand, commandAttributes } from '@/features/projectWindow/studioCommands';
import type {
  ComponentDefinition, DocumentPresentation, EvaluateAttachmentReadRequest, EvaluateAttachmentRequest, JsonValue,
  ModelAddress, ModelClipboard, ModelDocument, ModelEditCommand, ModelPosition, StudioProjectSnapshot,
} from '@/generated/studioContracts';
import { ModelEditProblemSchema } from '@/generated/studioContracts';
import type { ProjectEditController } from '../projectEditController';
import { authoredPointer, presentationAddress } from '../authoredEdits';
import { ModelCanvas, type ModelCard, type ModelEdge } from './ModelCanvas';
import { AttachmentFields, ComponentFields, Field, ModelFields, PortFields, type Attachment } from './ModelInspector';
import {
  accepts, addressKey, at, boundaryPorts, componentPorts, endKey, fieldAddress, graphKind, modelPath,
  modelRegistry, portId, shortName, wireEnd, type Port,
} from './modelGraph';
import { selectModel, type ModelSelectionStore } from './modelSelection';

type Operation = ModelEditCommand['operation'];
type Difference = EvaluateAttachmentRequest['difference'];
type Props = {
  model?: { id: string; value: ModelDocument };
  saved: StudioProjectSnapshot;
  editor: ProjectEditController;
  window: ModelSelectionStore;
  active: boolean;
  editable: boolean;
  evaluate?: Omit<EvaluateAttachmentReadRequest, 'revision'>;
  problem?: { address: ModelAddress; reason: string };
  path: readonly string[];
  setPath(path: string[]): void;
};
const button = 'rounded px-2 py-1 text-left hover:bg-shell-hover disabled:opacity-50';
const panel = 'grid max-h-[70vh] w-[min(28rem,90vw)] gap-2 overflow-auto rounded border border-shell-divider bg-shell-chrome p-3 shadow-xl';
const emptyPath: string[] = [];
const inputCard = '@inputs';
const outputCard = '@outputs';
const emptyPorts: Port[] = [];
let clipboard: ModelClipboard | undefined;

function idFor(end: { owner: string; component?: string }) {
  return end.owner === 'component' ? end.component! : end.owner === 'outputs' ? outputCard : inputCard;
}

function currentGraph(document: string, composites: string[]): ModelAddress {
  return { document, composites, element: { kind: 'graph' } };
}

function inGraph(address: ModelAddress, graph: ModelAddress) {
  return address.document === graph.document && address.composites.length === graph.composites.length &&
    address.composites.every((id, index) => id === graph.composites[index]);
}

function positionsOf(saved: StudioProjectSnapshot, document: string) {
  return saved.presentation?.documents?.[document]?.positions;
}

function positionAt(index: number): ModelPosition {
  return { x: 245 + 235 * (index % 4), y: 38 + 160 * Math.floor(index / 4) };
}

function attachmentAddress(owner: ModelAddress, id: string): ModelAddress {
  if (owner.element.kind !== 'wire' && owner.element.kind !== 'port') throw new Error('Expected an attachment owner');
  return at(owner, { kind: 'attachment', parent: owner.element, id });
}

function stepCount(item: Attachment, step: number) {
  return item.kind === 'delay' ? String(item.delay.unit === 'st' ? item.delay.value : Math.round(item.delay.value / step))
    : item.kind === 'probe' ? '⌖' : '◇';
}

function attachmentName(item: Attachment) {
  return item.kind === 'probe' ? item.name : item.kind === 'delay' ? 'Delay' : item.operation.toLowerCase() === 'clamp' ? 'Clamp' : 'Scale';
}

function searchable(definition: ComponentDefinition, query: string) {
  return (shortName(definition.name) + ' ' + definition.category).toLowerCase().includes(query.toLowerCase().trim());
}

function LeafChoices({ definitions, query, choose, disabled, browse, inspect }: {
  definitions: readonly ComponentDefinition[]; query: string; choose(definition: ComponentDefinition): void;
  disabled: boolean; browse?: boolean; inspect?: (definition: ComponentDefinition) => void;
}) {
  const groups = new Map<string, ComponentDefinition[]>();
  for (const definition of definitions) {
    if (!searchable(definition, query)) continue;
    const group = groups.get(definition.category) ?? [];
    if (!groups.has(definition.category)) groups.set(definition.category, group);
    group.push(definition);
  }
  return <>{[...groups].map(([category, items]) => <section key={category} className="space-y-0.5">
    <h3 className="px-2 pt-1 text-xs text-shell-muted">{category}</h3>
    {items.map((definition) => <button key={definition.name} className={`${button} block w-full`} type="button"
      aria-label={`Add ${shortName(definition.name)}`} disabled={disabled && !browse}
      draggable={browse && !disabled} onDragStart={(event) => {
        event.dataTransfer.setData('application/x-feedbax-component', definition.name);
      }} onClick={() => browse ? inspect?.(definition) : choose(definition)}
      onDoubleClick={() => { if (browse && !disabled) choose(definition); }}
      onKeyDown={(event) => {
        if (browse && !disabled && event.key === 'Enter') { event.preventDefault(); choose(definition); }
      }}>{shortName(definition.name)}</button>)}
  </section>)}</>;
}

export function useStoredModelView({ model, saved, editor, window: state, active, editable, evaluate, problem, path, setPath }: Props) {
  const selected = useStore(state, (current) => current.selection);
  const pick = useStore(state, (current) => current.pick);
  const reveal = useStore(state, (current) => current.reveal);
  const viewport = useRef<CanvasViewport>(null);
  const cursor = useRef<ModelPosition>();
  const dragSource = useRef<Port>();
  const connectionReview = useRef<AbortController>();
  const requestBusy = useRef(false);
  const [busy, setBusy] = useState(false);
  const [issue, setIssue] = useState<{ address?: ModelAddress; reason: string }>();
  const [drawer, setDrawer] = useState<'library' | 'outline'>('library');
  const [query, setQuery] = useState('');
  const [libraryItem, setLibraryItem] = useState<ComponentDefinition>();
  const [search, setSearch] = useState<ModelPosition>();
  const [searchText, setSearchText] = useState('');
  const [locked, setLocked] = useState(false);
  const [rename, setRename] = useState<string>();
  const [addPort, setAddPort] = useState<'input' | 'output'>();
  const [menu, setMenu] = useState<{ address: ModelAddress; x: number; y: number; point: ModelPosition }>();
  const [verdict, setVerdict] = useState<ReadonlyMap<string, string | null>>();
  const [held, setHeld] = useState<ReadonlyMap<string, ModelPosition>>();

  const registryQuery = useQuery({ queryKey: ['components'], queryFn: fetchComponents, enabled: active && !!model, staleTime: 300_000 });
  const registry = useMemo(() => modelRegistry(registryQuery.data ?? []), [registryQuery.data]);
  const resolved = useMemo(() => model && modelPath(model.value, path), [model?.value, path]);
  const currentPath = useMemo(() => resolved?.entries.map(({ id }) => id) ?? emptyPath, [resolved]);
  const graph = useMemo(() => currentGraph(model?.id ?? '', currentPath), [model?.id, currentPath]);
  const graphId = addressKey(graph);
  const owner = resolved?.owner;
  const interior = owner?.interior;
  const physical = interior?.physical_domain ?? null;
  const readBody = evaluate && { ...evaluate, revision: saved.revision };
  const attachments = useQuery({
    queryKey: ['model-evaluate-attachments', saved.project_id, readBody],
    queryFn: ({ signal }) => readEvaluateAttachments(saved.project_id, readBody!, signal),
    enabled: active && !!model && !!readBody, retry: false,
  });
  const editTopology = editable && !busy && !evaluate;
  const selectedHere = selected.filter((item) => inGraph(item, graph));
  const selectedIds = new Set(selectedHere.map(addressKey));
  const selection = selectedHere[0];
  const portMap = useMemo(() => {
    const ports = owner ? boundaryPorts(owner.ports) : emptyPorts;
    const map = new Map<string, Port[]>([
      [inputCard, ports.filter((port) => idFor(port.end) === inputCard)],
      [outputCard, ports.filter((port) => idFor(port.end) === outputCard)],
    ]);
    for (const [id, component] of Object.entries(interior?.components ?? {})) {
      map.set(id, componentPorts(id, component, registry));
    }
    return map;
  }, [owner, interior, registry]);
  const definitions = useMemo(() => (registryQuery.data ?? []).filter((definition) => accepts(definition, physical)), [registryQuery.data, physical]);
  const persisted = model && positionsOf(saved, model.id);
  const presentation = model && saved.presentation?.documents?.[model.id];
  const collapsed = presentation?.collapsed;
  const flipped = presentation?.flipped;
  const waypoints = presentation?.waypoints;
  const objects = Object.entries(interior?.components ?? {});
  const wires = Object.entries(interior?.wires ?? {});
  const mountKey = `${saved.project_id}:${graphId}`;
  const handledReveal = useRef<ModelAddress>();
  useEffect(() => {
    if (!model || !active || !reveal || reveal.document !== model.id || handledReveal.current === reveal) return;
    handledReveal.current = reveal;
    if (!state.getState().reveal) return;
    state.setState({ reveal: null });
    setPath(reveal.composites);
  }, [active, model?.id, reveal, setPath, state]);
  useEffect(() => {
    if (model && active && path.length !== currentPath.length) setPath(currentPath);
  }, [model?.id, active, currentPath, path, setPath]);
  const problemAddress = problem?.address;
  useEffect(() => {
    if (!model || !active || !problemAddress) return;
    const address = problemAddress.element.kind === 'field' ? at(problemAddress, problemAddress.element.parent) : problemAddress;
    setPath(address.composites);
    selectModel(state, address);
  }, [model?.id, active, problemAddress, state]);
  useEffect(() => () => connectionReview.current?.abort(), [graphId]);

  async function apply(name: string, operation: Operation, choose = true) {
    if (!model || !editTopology || requestBusy.current) return false;
    requestBusy.current = true;
    const reviewed = editor.getSnapshot().saved;
    setBusy(true);
    setIssue(undefined);
    try {
      const result = await editor.editModel(name, model.id, { operation }, reviewed);
      if (result.status === 'refused') {
        setIssue({ address: result.problem.address, reason: result.problem.detail });
        return false;
      }
      if (result.clipboard) clipboard = result.clipboard;
      if (choose && result.changed.length) {
        state.setState({ selection: operation.op === 'delete' || operation.op === 'cut' ? [] : result.changed });
      }
      return true;
    } catch (error) {
      const problem = ModelEditProblemSchema.safeParse(error instanceof ApiRequestError ? error.detail : undefined);
      setIssue(problem.success ? { address: problem.data.address, reason: problem.data.detail } : { reason: apiErrorMessage(error) });
      return false;
    } finally {
      requestBusy.current = false;
      setBusy(false);
    }
  }

  async function applyEvaluate(name: string, difference: Difference) {
    if (!readBody || !attachments.data || attachments.data.revision !== saved.revision || requestBusy.current) return;
    requestBusy.current = true;
    setBusy(true);
    setIssue(undefined);
    try {
      const result = await editEvaluateAttachment(saved.project_id, { ...readBody, difference }, editor.signal);
      if (result.revision !== saved.revision) throw new Error('The selected Evaluate revision changed.');
      editor.editFiles(name, [result.file], {
        reviewedRevision: saved.revision,
        dependencies: model ? [{ kind: 'document', id: model.id }] : [],
      });
      await editor.flush();
      selectModel(state, difference.op === 'remove' ? at(result.changed, { kind: 'graph' }) : result.changed);
    } catch (error) {
      setIssue({ reason: apiErrorMessage(error) });
    } finally {
      requestBusy.current = false;
      setBusy(false);
    }
  }

  function changeLayout<K extends 'collapsed' | 'flipped' | 'waypoints'>(name: string, key: K, next: DocumentPresentation[K]) {
    if (!model || !saved.presentation || requestBusy.current) return;
    const documents = saved.presentation.documents;
    const patch = !documents ? { op: 'add', path: '/documents', value: { [model.id]: { [key]: next } } } as const
      : !presentation ? { op: 'add', path: authoredPointer(['documents', model.id]), value: { [key]: next } } as const
        : { op: presentation[key] === undefined ? 'add' : 'replace', path: authoredPointer(['documents', model.id, key]), value: next } as const;
    try {
      editor.editFiles(name, [{ address: presentationAddress, patches: [patch] }], {
        reviewedRevision: saved.revision, reviewedPresentationRevision: saved.presentation_revision,
      });
      void editor.flush().catch((error) => setIssue({ reason: apiErrorMessage(error) }));
    } catch (error) {
      setIssue({ reason: apiErrorMessage(error) });
    }
  }

  function select(address: ModelAddress, extend = false) {
    setLibraryItem(undefined);
    selectModel(state, address, extend);
  }
  function clear() {
    const armed = state.getState().pick;
    state.setState(armed ? { pick: null } : { selection: [] });
    setLibraryItem(undefined);
  }
  function go(path: string[]) {
    setSearch(undefined);
    setMenu(undefined);
    setPath(path);
  }
  function openSearch(point = cursor.current ?? positionAt(objects.length)) {
    if (!editTopology) return;
    setSearchText('');
    setSearch(point);
  }
  function add(definition: ComponentDefinition, point?: ModelPosition) {
    const position = point ?? positionAt(objects.length);
    void apply('Add Component', {
      op: 'add_component', graph, position,
      component: {
        kind: 'leaf', name: shortName(definition.name), type: definition.identity?.type_id ?? definition.name,
        revision: definition.revision, fields: Object.fromEntries(Object.entries(definition.default_params ?? {})
          .filter(([name]) => name !== 'dt' && name !== 'control_dt')) as Record<string, JsonValue>,
      },
    }).then((ok) => { if (ok) setSearch(undefined); });
  }
  function portAt(node: string, handle: string | null) {
    return handle ? portMap.get(node)?.find((port) => portId(port) === handle) : undefined;
  }

  const connectStart: OnConnectStart = (_, { nodeId, handleId }) => {
    if (!editTopology || locked || !nodeId) return;
    const port = portAt(nodeId, handleId);
    dragSource.current = port;
    if (!port || port.direction !== 'output') return;
    const source = wireEnd(port);
    if (source.owner !== 'inputs' && source.owner !== 'component') return;
    connectionReview.current?.abort();
    const request = new AbortController();
    connectionReview.current = request;
    void reviewModelConnections(saved.project_id, { revision: saved.revision, graph, source }, request.signal)
      .then(({ targets }) => {
        if (!request.signal.aborted) setVerdict(new Map(targets.map(({ target, reason }) => [endKey(target), reason])));
      })
      .catch((error) => { if (!request.signal.aborted) setIssue({ reason: apiErrorMessage(error) }); });
  };
  function connect(sourceNode: string, sourceHandle: string | null, targetNode: string, targetHandle: string | null) {
    const a = portAt(sourceNode, sourceHandle), b = portAt(targetNode, targetHandle);
    if (!a || !b || locked) return;
    const left = wireEnd(a), right = wireEnd(b);
    if (a.direction === 'physical' && b.direction === 'physical' && physical &&
        (left.owner === 'component' || left.owner === 'boundary') && (right.owner === 'component' || right.owner === 'boundary')) {
      void apply('Connect Ports', { op: 'connect_physical', graph, connection: { a: left, b: right } });
    } else if (a.direction === 'output' && b.direction === 'input' &&
        (left.owner === 'component' || left.owner === 'inputs') && (right.owner === 'component' || right.owner === 'outputs') &&
        !verdict?.get(endKey(right))) {
      void apply('Connect Ports', { op: 'connect', graph, source: left, target: right });
    }
  }

  const effective = new Map<string, Attachment[]>();
  for (const [id, wire] of wires) effective.set(addressKey(at(graph, { kind: 'wire', id })), [...(wire.attachments ?? [])]);
  const portOwners = new Map<string, ModelAddress>();
  for (const ports of portMap.values()) for (const port of ports) {
    if (port.direction === 'physical') continue;
    const address = at(graph, { kind: 'port', end: port.end });
    portOwners.set(addressKey(address), address);
    const end = port.end;
    const store = end.owner === 'component' ? interior?.components?.[end.component]?.attachments : interior?.attachments;
    effective.set(addressKey(address), [...((port.direction === 'output' ? store?.sources : store?.sinks)?.[end.port] ?? [])]);
  }
  for (const difference of attachments.data?.effective ?? []) {
    const owner = difference.op === 'add' ? difference.owner :
      difference.address.element.kind === 'attachment' ? at(difference.address, difference.address.element.parent) : undefined;
    const list = owner && effective.get(addressKey(owner));
    if (!list) continue;
    if (difference.op === 'add') {
      const index = difference.after === null ? -1 : list.findIndex(({ id }) => id === difference.after);
      if (difference.after === null || index >= 0) list.splice(index + 1, 0, difference.attachment);
    } else if (difference.address.element.kind === 'attachment') {
      const id = difference.address.element.id;
      const index = list.findIndex((item) => item.id === id);
      if (index < 0) continue;
      if (difference.op === 'remove') list.splice(index, 1);
      else if (difference.after === undefined) list[index] = difference.attachment;
      else {
        list.splice(index, 1);
        const after = list.findIndex((item) => item.id === difference.after);
        list.splice(after + 1, 0, difference.attachment);
      }
    }
  }

  function move(nodes: ModelCard[]) {
    const placements = nodes.filter((node) => node.data.address.element.kind === 'component' && !node.data.locked)
      .map((node) => ({ address: node.data.address, position: node.position }));
    if (!placements.length) return;
    setHeld(new Map(placements.map(({ address, position }) => [addressKey(address), position])));
    void apply('Move Component', { op: 'move_components', placements }, false).finally(() => setHeld(undefined));
  }

  function makeCard(id: string, index: number, definition?: typeof objects[number][1]): ModelCard {
    const address = definition ? at(graph, { kind: 'component', id }) : graph;
    const key = addressKey(address);
    const type = definition?.kind === 'leaf' ? registry.get(definition.type) : undefined;
    const value = definition?.kind === 'leaf' ? definition.fields?.hidden_size ?? definition.fields?.output_size : undefined;
    const name = definition?.name ?? (id === inputCard ? 'inputs' : 'outputs');
    const caption = definition?.kind === 'leaf'
      ? [shortName(type?.name ?? definition.type), typeof value === 'number' ? value : undefined].filter((item) => item !== undefined).join(' · ')
      : definition?.kind === 'composite' ? graphKind(definition) : undefined;
    const ports = portMap.get(id) ?? emptyPorts;
    return {
      id, type: 'modelCard', ariaLabel: name,
      position: held?.get(key) ?? (definition ? persisted?.[key] ?? positionAt(index) :
        { x: id === inputCard ? 18 : 245 + 235 * Math.max(1, Math.min(objects.length, 4)), y: 38 }),
      selected: !!definition && selectedIds.has(key),
      draggable: !!definition && editTopology && !locked && !pick,
      selectable: !!definition && !pick,
      data: {
        address, name, caption, ports, collapsed: !!definition && !!collapsed?.[graphId], flipped: flipped?.[key],
        locked: locked || !editTopology, stacked: definition?.kind === 'composite',
        selected: !!definition && selectedIds.has(key),
        holdsSelection: definition?.kind === 'composite' && selected.some((item) => item.document === graph.document &&
          currentPath.every((parent, offset) => item.composites[offset] === parent) && item.composites[currentPath.length] === id),
        problem: issue?.address && addressKey(issue.address) === key ? issue.reason :
          problemAddress && addressKey(problemAddress) === key ? problem?.reason : undefined,
        onPort: (port) => select(at(graph, { kind: 'port', end: port.end })),
        canPick: pick ? (port) => pick.accepts(at(graph, { kind: 'port', end: port.end }), port) : undefined,
        pickPort: pick ? (port) => { pick.choose(at(graph, { kind: 'port', end: port.end })); state.setState({ pick: null }); } : undefined,
        portReasons: verdict && new Map(ports.filter((port) => port.direction === 'input').map((port) =>
          [portId(port), verdict.get(endKey(wireEnd(port))) ?? null])),
        enter: definition?.kind === 'composite' ? () => go([...currentPath, id]) : undefined,
        renaming: rename === key, endRename: () => setRename(undefined),
        rename: (name) => { void apply('Rename', { op: 'rename', address, name }).then((ok) => { if (ok) setRename(undefined); }); },
        addPort: !definition && editTopology ? () => setAddPort(id === inputCard ? 'input' : 'output') : undefined,
      },
    };
  }

  const nodes = [makeCard(inputCard, 0), ...objects.map(([id, component], index) => makeCard(id, index, component)), makeCard(outputCard, 0)];
  const selfLoops = new Map<string, number>();
  const makeEdge = (id: string, from: Parameters<typeof idFor>[0] & { port: string },
    to: Parameters<typeof idFor>[0] & { port: string }, type: ModelEdge['type']): ModelEdge => {
    const isPhysical = type === 'physical';
    const address = at(graph, { kind: isPhysical ? 'physical_connection' : 'wire', id });
    const key = addressKey(address);
    const source = idFor(from);
    const target = idFor(to);
    const self = !isPhysical && source === target ? selfLoops.get(source) ?? 0 : undefined;
    if (self !== undefined) selfLoops.set(source, self + 1);
    return {
      id, source, target, type,
      sourceHandle: `${isPhysical ? 'physical' : 'output'}:${from.port}`,
      targetHandle: `${isPhysical ? 'physical' : 'input'}:${to.port}`,
      markerEnd: isPhysical ? undefined : { type: MarkerType.Arrow, width: 12, height: 12, color: '#555860' },
      selected: selectedIds.has(key),
      data: {
        address, selected: selectedIds.has(key), waypoints: waypoints?.[key],
        self,
        attachments: effective.get(key)?.map((item) => ({
          id: item.id, label: attachmentName(item), mark: stepCount(item, model?.value.step_seconds ?? 0.01),
          delay: item.kind === 'delay', select: () => select(attachmentAddress(address, item.id)),
        })),
      },
    };
  };
  const edges = [
    ...wires.map(([id, wire]) => makeEdge(id, wire.source, wire.target, 'signal')),
    ...('connections' in (interior ?? {}) ? Object.entries((interior as { connections?: Record<string, { a: Parameters<typeof makeEdge>[1]; b: Parameters<typeof makeEdge>[2] }> }).connections ?? {})
      .map(([id, connection]) => makeEdge(id, connection.a, connection.b, 'physical')) : []),
  ];

  const element = selection?.element;
  const component = element?.kind === 'component' ? interior?.components?.[element.id] : undefined;
  const port = element?.kind === 'port' ? portMap.get(idFor(element.end))?.find((item) =>
    addressKey(at(graph, { kind: 'port', end: item.end })) === addressKey(selection!)) : undefined;
  const attachmentOwner = element?.kind === 'attachment' && selection ? at(selection, element.parent) : undefined;
  const attachment = element?.kind === 'attachment' && attachmentOwner
    ? effective.get(addressKey(attachmentOwner))?.find(({ id }) => id === element.id) : undefined;
  const ownerAddress = selection && (element?.kind === 'wire' || element?.kind === 'port') ? selection : attachmentOwner;
  const chain = ownerAddress && effective.get(addressKey(ownerAddress));
  const canEvaluate = Boolean(readBody && attachments.data?.revision === saved.revision);
  const authored = attachments.data?.records ?? [];
  function isAuthored(address: ModelAddress) {
    return authored.some((record) => addressKey(record.op === 'add' ? attachmentAddress(record.owner, record.attachment.id) : record.address) === addressKey(address));
  }
  function updateAttachment(address: ModelAddress, item: Attachment) {
    if (evaluate) {
      void applyEvaluate(attachmentName(item), { op: 'change', address, attachment: item });
      return;
    }
    if (address.element.kind !== 'attachment') return;
    if (address.element.parent.kind === 'wire') {
      const wire = at(address, address.element.parent);
      const previous = effective.get(addressKey(wire)) ?? [];
      void apply(attachmentName(item), {
        op: 'set_field', address: fieldAddress(wire, 'attachments'),
        value: previous.map((entry) => entry.id === item.id ? item : entry) as unknown as JsonValue,
      }, false);
    } else {
      const fields = item.kind === 'intervention' ? [['operation', item.operation], ['fields', item.fields ?? {}]] as const
        : item.kind === 'probe' ? [['name', item.name]] as const : [];
      if (fields.length === 1) void apply(attachmentName(item), {
        op: 'set_field', address: fieldAddress(address, fields[0][0]), value: fields[0][1] as JsonValue,
      }, false);
    }
  }
  function addAttachment(address: ModelAddress, kind: 'delay' | 'intervention' | 'probe') {
    const list = effective.get(addressKey(address)) ?? [];
    const id = crypto.randomUUID();
    const item: Attachment = kind === 'delay' ? { id, kind, delay: { unit: 'st', value: 1 }, initial: 0 }
      : kind === 'intervention' ? { id, kind, operation: 'scale', fields: { factor: 1 } }
      : { id, kind, name: port?.name ?? 'Probe' };
    if (evaluate) {
      void applyEvaluate(`Add ${attachmentName(item)}`, { op: 'add', owner: address, after: list.at(-1)?.id ?? null, attachment: item });
    } else if (address.element.kind === 'wire') {
      void apply(`Add ${attachmentName(item)}`, {
        op: 'set_field', address: fieldAddress(address, 'attachments'), value: [...list, item] as unknown as JsonValue,
      }, false).then((ok) => { if (ok) select(attachmentAddress(address, item.id)); });
    }
  }
  function removeAttachment(address: ModelAddress) {
    if (evaluate) void applyEvaluate('Remove Attachment', { op: 'remove', address });
    else void apply('Remove Attachment', { op: 'delete', selection: [address] });
  }
  function armInitial(address: ModelAddress, item: Attachment) {
    if (item.kind !== 'delay') return;
    state.setState({ pick: {
      accepts: (candidate, port) => inGraph(candidate, address) && port.direction === 'output' && port.end.owner === 'component',
      choose: (candidate) => updateAttachment(address, { ...item, initial: { kind: 'name', reference: addressKey(candidate) } }),
    } });
  }
  function labelOfPort(address: ModelAddress) {
    if (address.element.kind !== 'port') return '';
    const end = address.element.end;
    const candidate = portMap.get(idFor(end))?.find((item) =>
      addressKey(at(graph, { kind: 'port', end: item.end })) === addressKey(address));
    return [end.owner === 'component' ? interior?.components?.[end.component]?.name : end.owner, candidate?.name].filter(Boolean).join('.');
  }
  const shownAttachments = (address: ModelAddress, list: readonly Attachment[]) => <ul>
    {list.map((item) => {
      const itemAddress = attachmentAddress(address, item.id);
      return <li key={item.id} className="flex gap-1">
        <button className={button} type="button" aria-pressed={selectedIds.has(addressKey(itemAddress))}
          onClick={() => select(itemAddress)}>
          {stepCount(item, model?.value.step_seconds ?? 0.01)} · {attachmentName(item)}{isAuthored(itemAddress) ? ' ◇' : ''}
        </button>
      </li>;
    })}
  </ul>;
  function attachmentOptions(address: ModelAddress) {
    const wire = address.element.kind === 'wire';
    const source = address.element.kind === 'port' &&
      (address.element.end.owner === 'inputs' || address.element.end.owner === 'component' && address.element.end.direction === 'output');
    if (!(wire || source) || !(editTopology && wire || canEvaluate) || busy) return null;
    return <div className="flex flex-wrap gap-1">
      {wire && editTopology && <button className={button} type="button" onClick={() => addAttachment(address, 'delay')}>Add Delay</button>}
      {canEvaluate && <button className={button} type="button" onClick={() => addAttachment(address, 'intervention')}>Add Intervention</button>}
      {(wire && editTopology || canEvaluate) && <button className={button} type="button" onClick={() => addAttachment(address, 'probe')}>Add Probe</button>}
    </div>;
  }

  const componentsOnly = selectedHere.length > 0 && selectedHere.every((item) => item.element.kind === 'component');
  const viewportCommands = canvasViewportCommands(() => viewport.current, !!model);
  const commands = [
    bindStudioCommand('addComponent', () => openSearch(), { enabled: editTopology && definitions.length > 0 }),
    bindStudioCommand('modelSelectAll', () => state.setState({
      selection: objects.map(([id]) => at(graph, { kind: 'component', id })),
    }), { enabled: !!objects.length }),
    bindStudioCommand('modelRemove', () => {
      if (selection?.element.kind === 'attachment') removeAttachment(selection);
      else void apply('Delete', { op: 'delete', selection: [...selectedHere] });
    }, { enabled: selectedHere.length > 0 && (editTopology || canEvaluate && !!attachment && !busy) }),
    bindStudioCommand('modelRename', () => { if (selection) setRename(addressKey(selection)); }, { enabled: editTopology && !!component }),
    bindStudioCommand('modelDuplicate', () => void apply('Duplicate', { op: 'duplicate', selection: [...selectedHere] }),
      { enabled: editTopology && componentsOnly }),
    bindStudioCommand('modelGroup', () => void apply('Group', { op: 'group', selection: [...selectedHere], name: 'Group' }),
      { enabled: editTopology && componentsOnly && !physical }),
    bindStudioCommand('modelUngroup', () => { if (selection) void apply('Ungroup', { op: 'ungroup', address: selection }); },
      { enabled: editTopology && component?.kind === 'composite' && !component.interior.physical_domain }),
    bindStudioCommand('modelCopy', () => void apply('Copy', { op: 'copy', selection: [...selectedHere] }, false),
      { enabled: editTopology && componentsOnly }),
    bindStudioCommand('modelCut', () => void apply('Cut', { op: 'cut', selection: [...selectedHere] }),
      { enabled: editTopology && componentsOnly }),
    bindStudioCommand('modelPaste', () => { if (clipboard) void apply('Paste', { op: 'paste', graph, clipboard }); },
      { enabled: editTopology && !!clipboard }),
    ...viewportCommands,
  ];
  function contextual(event: MouseEvent, address: ModelAddress, point: ModelPosition) {
    if (address.element.kind === 'graph') return;
    event.preventDefault();
    event.stopPropagation();
    if (!selectedIds.has(addressKey(address))) select(address);
    setMenu({ address, x: event.clientX, y: event.clientY, point });
  }
  const inspector = libraryItem ? <section className="grid gap-2 p-2 text-sm">
    <strong>{shortName(libraryItem.name)}</strong><p>{libraryItem.description}</p>
    <ul>{[...(libraryItem.input_ports ?? []), ...(libraryItem.output_ports ?? [])].map((name, index) => <li key={index}>{name}</li>)}</ul>
    {editTopology && <button className={button} type="button" onClick={() => add(libraryItem)}>Add {shortName(libraryItem.name)}</button>}
  </section> : model && <section className="grid gap-2 p-2 text-sm">
    {component && selection ? <>
      <strong>{component.name}</strong>
      <ComponentFields component={component} definition={component.kind === 'leaf' ? registry.get(component.type) : undefined}
        change={editTopology ? (path, value) => void apply('Set Value', {
          op: 'set_field', address: fieldAddress(selection, ...path), value,
        }, false) : undefined} />
    </> : port ? <>
      <strong>{port.name}</strong><PortFields port={port} />
      {selection && editTopology && port.end.owner !== 'component' && <Field label="Name" value={port.name}
        parse={(raw) => raw.trim() || undefined} commit={(name) => void apply('Rename', { op: 'rename', address: selection, name: String(name) }, false)} />}
    </> : attachment && selection ? <>
      <strong>{attachmentName(attachment)}</strong>
      <AttachmentFields item={attachment} step={model.value.step_seconds}
        change={editTopology || canEvaluate && !busy ? (item) => updateAttachment(selection, item) : undefined}
        arm={attachment.kind === 'delay' && editTopology ? () => armInitial(selection, attachment) : undefined}
        linkLabel={labelOfPort} />
      {(editTopology || canEvaluate && !busy) && <button className={button} type="button"
        onClick={() => removeAttachment(selection)}>Remove Attachment</button>}
      {canEvaluate && chain && chain.length > 1 && isAuthored(selection) && <label>After
        <select value={chain[chain.indexOf(attachment) - 1]?.id ?? ''} onChange={(event) => void applyEvaluate(
          'Move Attachment', { op: 'change', address: selection, attachment, after: event.currentTarget.value || null },
        )}>
          <option value="">Source</option>
          {chain.filter((item) => item.id !== attachment.id).map((item) => <option key={item.id} value={item.id}>{attachmentName(item)}</option>)}
        </select>
      </label>}
    </> : ownerAddress ? <strong>Wire</strong>
      : <ModelFields model={model.value} change={editTopology && !currentPath.length ? (path, value) => void apply('Step', {
        op: 'set_field', address: fieldAddress(graph, ...path), value,
      }, false) : undefined} />}
    {ownerAddress && !attachment && chain && <div className="border-t border-shell-divider pt-1">
      {shownAttachments(ownerAddress, chain)}
      {attachmentOptions(ownerAddress)}
    </div>}
  </section>;

  const outline = <ul className="space-y-0.5">
    {objects.map(([id, component]) => {
      const address = at(graph, { kind: 'component', id });
      return <li key={id} className="flex">
        {component.kind === 'composite' && <button className={button} type="button"
          aria-label={`Enter ${component.name}`} onClick={() => go([...currentPath, id])}>›</button>}
        <button className={button} type="button" aria-pressed={selectedIds.has(addressKey(address))}
          onClick={(event) => select(address, event.shiftKey)}>{component.name}</button>
      </li>;
    })}
    {wires.map(([id, wire]) => {
      const address = at(graph, { kind: 'wire', id });
      const title = [wire.source, wire.target].map((end) =>
        `${end.owner === 'component' ? interior?.components?.[end.component]?.name ?? end.component : end.owner}.${end.port}`).join(' → ');
      return <li key={id}>
        <button type="button" className={button} aria-pressed={selectedIds.has(addressKey(address))} onClick={() => select(address)}>{title}</button>
        {shownAttachments(address, effective.get(addressKey(address)) ?? [])}
      </li>;
    })}
    {[...portOwners].map(([key, address]) => {
      const items = effective.get(key);
      return items?.length ? <li key={key}>{shownAttachments(address, items)}</li> : null;
    })}
  </ul>;

  return {
    documentId: model?.id,
    header: model && <nav className="flex min-w-0 items-center gap-1" aria-label="Model Path">
      <button type="button" className={button} onClick={() => go([])}>{model.id}</button>
      {resolved?.entries.map(({ id, name }, index) => <button key={id} type="button" className={button}
        onClick={() => go(currentPath.slice(0, index + 1))}>› {name}</button>)}
    </nav>,
    body: model ? <div className="flex h-full min-h-0 flex-col" onKeyDown={(event) => {
      if (event.key === 'Escape' && state.getState().pick) { event.stopPropagation(); clear(); }
    }}>
      {(issue || problem || attachments.error) && <p className="px-2 text-sm" role="alert">
        {issue?.reason ?? problem?.reason ?? apiErrorMessage(attachments.error)}
      </p>}
      <div className="min-h-0 flex-1">
        <ModelCanvas key={mountKey} nodes={nodes} edges={edges} physical={!!physical} locked={locked} canEdit={editTopology}
          collapsed={!!collapsed?.[graphId]} viewportRef={viewport}
          onToggleLock={() => setLocked((current) => !current)}
          onToggleCollapse={() => changeLayout('Collapse All', 'collapsed', { ...collapsed, [graphId]: !collapsed?.[graphId] })}
          onCanvasPoint={(point) => { cursor.current = point; }}
          onSearch={openSearch} onNode={(node, extend) => {
            if (!pick && node.data.address.element.kind === 'component') select(node.data.address, extend);
          }} onEdge={(edge) => { if (!pick && edge.data) select(edge.data.address); }}
          onClear={clear} onBox={(cards) => state.setState({
            selection: cards.filter((node) => node.data.address.element.kind === 'component').map((node) => node.data.address),
          })} onMove={move} onContext={contextual} onConnectStart={connectStart} onConnect={connect}
          onConnectEnd={(point, valid) => {
            connectionReview.current?.abort();
            setVerdict(undefined);
            if (!valid && !dragSource.current) cursor.current = point;
            dragSource.current = undefined;
          }} onLibraryDrop={(type, point) => {
            const definition = registry.get(type);
            if (definition && definitions.includes(definition)) add(definition, point);
          }} />
      </div>
    </div> : null,
    drawer: model && <section className="space-y-2 p-1 text-sm">
      <div className="flex" aria-label="Model Drawer">
        <button className={button} type="button" aria-pressed={drawer === 'library'} onClick={() => setDrawer('library')}>Library</button>
        <button className={button} type="button" aria-pressed={drawer === 'outline'} onClick={() => setDrawer('outline')}>Outline</button>
      </div>
      {registryQuery.error && <p role="alert">{apiErrorMessage(registryQuery.error)}</p>}
      {drawer === 'library' ? <>
        <input aria-label="Search Library" className="w-full border border-shell-divider px-1 py-0.5"
          value={query} onChange={(event) => setQuery(event.currentTarget.value)} />
        <LeafChoices definitions={definitions} query={query} choose={add} inspect={setLibraryItem} disabled={!editTopology} browse />
      </> : outline}
    </section>,
    inspector,
    commands,
    overlay: search ? <Dialog ariaLabel="Add Component" onClose={() => setSearch(undefined)} manageFocus panelClassName={panel}>
      <input data-dialog-autofocus aria-label="Components" className="border border-shell-divider px-2 py-1"
        value={searchText} onChange={(event) => setSearchText(event.currentTarget.value)} />
      <LeafChoices definitions={definitions} query={searchText} choose={(item) => add(item, search)} disabled={!editTopology} />
    </Dialog> : addPort ? <Dialog ariaLabel="Add Port" onClose={() => setAddPort(undefined)} manageFocus panelClassName={panel}>
      <Field label="Name" value="" parse={(raw) => raw.trim() || undefined} commit={(name) => {
        void apply('Add Port', { op: 'add_port', graph, direction: addPort, port: { id: crypto.randomUUID(), name: String(name), type: {} } }, false)
          .then((ok) => { if (ok) setAddPort(undefined); });
      }} />
      {addPort === 'input' && currentPath.length > 0 && (() => {
        const outerPath = currentPath.slice(0, -1);
        const outer = model && modelPath(model.value, outerPath).owner;
        const candidates = outer && [
          ...boundaryPorts(outer.ports).filter(({ direction }) => direction === 'output'),
          ...Object.entries(outer.interior.components ?? {}).filter(([id]) => id !== currentPath.at(-1)).flatMap(([id, component]) =>
            componentPorts(id, component, registry).filter(({ direction }) => direction === 'output')),
        ];
        return candidates?.map((port) => {
          const source = wireEnd(port);
          if (source.owner !== 'inputs' && source.owner !== 'component') return null;
          return <button key={endKey(source)} className={button} type="button" onClick={() => {
            void apply('Bring Signal In', { op: 'bring_signal_in', source_graph: currentGraph(graph.document, outerPath),
              source, destination: graph }, false).then((ok) => { if (ok) setAddPort(undefined); });
          }}>{source.owner === 'component' ? `${outer.interior.components?.[source.component]?.name}.` : ''}{port.name}</button>;
        });
      })()}
    </Dialog> : menu && <Dialog ariaLabel="Model Commands" onClose={() => setMenu(undefined)} manageFocus className="fixed inset-0">
      <div role="menu" className="fixed grid rounded border border-shell-divider bg-shell-chrome p-1 text-sm shadow-lg"
        style={{ left: Math.min(menu.x, window.innerWidth - 160), top: Math.min(menu.y, window.innerHeight - 240) }}>
        {commands.filter((command) => command.enabled && command.id.startsWith('model')).map((command) =>
          <button className={button} role="menuitem" type="button" key={command.id} {...commandAttributes(command)}
            onClick={() => { setMenu(undefined); command.invoke(); }}>{command.label}</button>)}
        {menu.address.element.kind === 'component' && editTopology && <button className={button} role="menuitem" type="button"
          onClick={() => { const key = addressKey(menu.address); changeLayout('Flip', 'flipped', { ...flipped, [key]: !flipped?.[key] }); setMenu(undefined); }}>Flip</button>}
        {menu.address.element.kind === 'wire' && editTopology && <>
          <button className={button} role="menuitem" type="button" onClick={() => {
            const key = addressKey(menu.address);
            changeLayout('Add Waypoint', 'waypoints', { ...waypoints, [key]: [...(waypoints?.[key] ?? []), menu.point] });
            setMenu(undefined);
          }}>Add Waypoint</button>
          {!!waypoints?.[addressKey(menu.address)]?.length && <button className={button} role="menuitem" type="button" onClick={() => {
            const key = addressKey(menu.address);
            changeLayout('Remove Waypoint', 'waypoints', { ...waypoints, [key]: waypoints![key].slice(0, -1) });
            setMenu(undefined);
          }}>Remove Waypoint</button>}
        </>}
      </div>
    </Dialog>,
  };
}
