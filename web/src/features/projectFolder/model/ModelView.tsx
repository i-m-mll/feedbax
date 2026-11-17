import { useQueries, useQuery } from '@tanstack/react-query';
import { MarkerType, type OnConnectStart } from '@xyflow/react';
import { useEffect, useId, useMemo, useRef, useState, type MouseEvent } from 'react';
import { useStore } from 'zustand';
import { fetchComponents } from '@/api/components';
import { editEvaluateAttachment, projectModelFields, readEvaluateAttachments, readModelProjection, reviewModelConnections } from '@/api/projectFolders';
import { ApiRequestError, apiErrorMessage } from '@/api/request';
import { StudioRegion, studioReadState } from '@/components/activity/StudioRegion';
import { studioAccessibleName, studioFormat, studioName, studioReason } from '@/components/activity/studioVocabulary';
import { ChevronRight, Crosshair, House, Search, X } from 'lucide-react';
import { Button, IconButton } from '@/components/ui/Button';
import { Dialog } from '@/components/ui/Dialog';
import { Menu, MenuContent, MenuGroup, MenuGroupLabel, MenuItem } from '@/components/ui/Menu';
import { Inspector, InspectorAddRow } from '@/components/ui/Inspector';
import { Tree, type TreeNode } from '@/components/ui/Tree';
import { SegmentedControl } from '@/components/ui/SegmentedControl';
import { Select } from '@/components/ui/Select';
import { TextField } from '@/components/ui/TextField';
import { ValueField } from '@/components/values/ValueField';
import { fromFieldView } from '@/components/values/adapters/fieldView';
import { rawField } from '@/components/values/adapters/raw';
import type { FieldModel } from '@/components/values/fieldModel';
import { canvasViewportCommands, type CanvasViewport } from '@/components/canvas/StudioCanvas';
import { bindStudioCommand, commandAttributes } from '@/features/projectWindow/studioCommands';
import type {
  ComponentDefinition, EvaluateAttachmentReadRequest, EvaluateAttachmentRequest, JsonValue,
  ModelAddress, ModelClipboard, ModelConnectionReview, ModelDocument, ModelEditCommand, ModelFieldRequest, ModelPosition, StudioProjectSnapshot, StudioSavedTrainKey,
} from '@/generated/studioContracts';
import { ModelAddressSchema, ModelEditProblemSchema } from '@/generated/studioContracts';
import type { ProjectEditController } from '../projectEditController';
import { ModelCanvas, type ModelCard, type ModelEdge } from './ModelCanvas';
import { attachmentFields, componentFields, initialField, modelFields, sharedFields, type Attachment, type FieldChange } from './ModelInspector';
import {
  accepts, addressKey, at, boundaryPorts, componentPorts, endKey, fieldAddress, graphKind, modelPath,
  modelRegistry, portId, portTypeLabel, shortName, wireEnd, type Component, type Port,
} from './modelGraph';
import { selectElement, useModelSelection, type ElementSelectionStore } from '@/features/projectWindow/elementSelection';
import { sweepValueRuns, type ConfirmRunRemoval } from '../runRemoval';

type Operation = ModelEditCommand['operation'];
type Difference = EvaluateAttachmentRequest['difference'];
type Props = {
  model?: { id: string; value: ModelDocument };
  saved: StudioProjectSnapshot;
  editor: ProjectEditController;
  elements: ElementSelectionStore;
  active: boolean;
  editable: boolean;
  fieldEditable: boolean;
  fieldLitSteps?: ReadonlySet<string>;
  fieldRows?: readonly string[];
  fieldAxes?: Record<string, string[]>;
  selectedSweepValues?: ReadonlyMap<string, ReadonlySet<string>>;
  confirmRunRemoval: ConfirmRunRemoval;
  evaluate?: Omit<EvaluateAttachmentReadRequest, 'revision'>;
  problem?: { address: ModelAddress; reason: string };
  problems?: readonly { address: ModelAddress; reason: string }[];
  projectionRun?: StudioSavedTrainKey;
  path: readonly string[];
  setPath(path: string[]): void;
};
const sourceAttachment = '@source';
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

function linkedAddress(value: string) {
  try {
    const parsed = ModelAddressSchema.safeParse(JSON.parse(value));
    return parsed.success ? parsed.data : undefined;
  } catch (error) {
    if (error instanceof SyntaxError) return undefined;
    throw error;
  }
}

function stepCount(item: Attachment, step: number) {
  return item.kind === 'delay' ? String(item.delay.unit === 'st' ? item.delay.value : Math.round(item.delay.value / step))
    : item.kind === 'probe' ? '⌖' : '◇';
}

function attachmentName(item: Attachment) {
  return item.kind === 'probe' ? item.name : studioName(item.kind === 'delay' ? 'delay' : item.operation.toLowerCase() === 'clamp' ? 'clamp' : 'scale');
}

function searchable(definition: ComponentDefinition, query: string) {
  return (shortName(definition.name) + ' ' + definition.category).toLowerCase().includes(query.toLowerCase().trim());
}

function signalFits(source: Port, target: Port) {
  if (typeof source.type === 'string' || typeof target.type === 'string') return false;
  const left = source.type;
  const right = target.type;
  if (left.base && right.base && left.base !== right.base || left.unit && right.unit && left.unit !== right.unit) return false;
  const shape = right.shape;
  return !left.shape || !shape || (left.shape.length === shape.length && left.shape.every((axis, index) =>
    typeof axis !== 'number' || typeof shape[index] !== 'number' || axis === shape[index]));
}

function LeafChoices({ definitions, query, choose, disabled, browse, inspect }: {
  definitions: readonly ComponentDefinition[]; query: string; choose(definition: ComponentDefinition): void;
  disabled: boolean; browse?: boolean; inspect?: (definition: ComponentDefinition) => void;
}) {
  const categoryId = useId();
  const groups = new Map<string, ComponentDefinition[]>();
  for (const definition of definitions) {
    if (!searchable(definition, query)) continue;
    const group = groups.get(definition.category) ?? [];
    if (!groups.has(definition.category)) groups.set(definition.category, group);
    group.push(definition);
  }
  return <>{[...groups].map(([category, items], index) => <section key={category} aria-labelledby={`${categoryId}-${index}`} className="space-y-0.5">
    <h3 id={`${categoryId}-${index}`} className="px-2 pt-1 text-xs text-shell-muted">{category}</h3>
    {items.map((definition) => <Button key={definition.name} variant="ghost" align="start" className="w-full"
      aria-label={studioAccessibleName('add', shortName(definition.name))} disabled={disabled && !browse}
      draggable={browse && !disabled} onDragStart={(event) => {
        event.dataTransfer.setData('application/x-feedbax-component', definition.name);
      }} onClick={() => browse ? inspect?.(definition) : choose(definition)}
      onDoubleClick={() => { if (browse && !disabled) choose(definition); }}
      onKeyDown={(event) => {
        if (browse && !disabled && event.key === 'Enter') { event.preventDefault(); choose(definition); }
      }}>{shortName(definition.name)}</Button>)}
  </section>)}</>;
}

export function useStoredModelView({ model, saved, editor, elements: state, active, editable, fieldEditable, fieldLitSteps, fieldRows, fieldAxes, selectedSweepValues, confirmRunRemoval, evaluate, problem, problems, projectionRun, path, setPath }: Props) {
  const selected = useModelSelection(state);
  const pick = useStore(state, (current) => current.pick);
  const reveal = useStore(state, (current) => current.reveal);
  const viewport = useRef<CanvasViewport>(null);
  const cursor = useRef<ModelPosition>();
  const dragSource = useRef<Port>();
  const connectionReview = useRef<AbortController>();
  const keyboardReview = useRef<AbortController>();
  const libraryReview = useRef<Promise<ModelConnectionReview>>();
  const requestBusy = useRef(false);
  const [busy, setBusy] = useState(false);
  const [issue, setIssue] = useState<{ address?: ModelAddress; reason: string }>();
  const [drawer, setDrawer] = useState<'library' | 'outline'>('library');
  const [query, setQuery] = useState('');
  const [libraryItem, setLibraryItem] = useState<ComponentDefinition>();
  const [search, setSearch] = useState<ModelPosition>();
  const [searchSource, setSearchSource] = useState<Port>();
  const [searchLibrary, setSearchLibrary] = useState<ModelConnectionReview['library']>();
  const [searchText, setSearchText] = useState('');
  const [locked, setLocked] = useState(false);
  const [rename, setRename] = useState<string>();
  const [addPort, setAddPort] = useState<'input' | 'output'>();
  const [menu, setMenu] = useState<{ address: ModelAddress; x: number; y: number; point: ModelPosition }>();
  const [verdict, setVerdict] = useState<ReadonlyMap<string, string | null>>();
  const [held, setHeld] = useState<ReadonlyMap<string, ModelPosition>>();

  const registryQuery = useQuery({ queryKey: ['components'], queryFn: fetchComponents, enabled: active && !!model, staleTime: 300_000 });
  const registry = useMemo(() => modelRegistry(registryQuery.data ?? []), [registryQuery.data]);
  const readBody = evaluate && { ...evaluate, revision: saved.revision };
  const projectionBody = projectionRun && model && { run: projectionRun, document: model.id, revision: saved.revision };
  const projectionRunId = projectionRun && JSON.stringify(projectionRun);
  const projection = useQuery({
    queryKey: ['model-projection', saved.project_id, projectionBody],
    queryFn: ({ signal }) => readModelProjection(saved.project_id, projectionBody!, signal),
    enabled: active && !!projectionBody && !evaluate,
  });
  const projectedPorts = useMemo(() => new Map(projection.data?.ports?.map(({ address, type }) => [addressKey(address), type])), [projection.data]);
  const markedPorts = useMemo(() => new Map([
    ...(projection.data?.task_inputs ?? []).map((address) => [addressKey(address), 'task'] as const),
    ...(projection.data?.objective_ports ?? []).map((address) => [addressKey(address), 'objective'] as const),
  ]), [projection.data]);
  const attachments = useQuery({
    queryKey: ['model-evaluate-attachments', saved.project_id, readBody],
    queryFn: ({ signal }) => readEvaluateAttachments(saved.project_id, readBody!, signal),
    enabled: active && !!model && !!readBody,
  });
  const effectiveModel = evaluate ? attachments.data?.resolved_model : projection.data?.resolved_model;
  const shownModel = evaluate ? effectiveModel : model?.value;
  const resolved = useMemo(() => shownModel && modelPath(shownModel, path), [shownModel, path]);
  const currentPath = useMemo(() => resolved?.entries.map(({ id }) => id) ?? emptyPath, [resolved]);
  const graph = useMemo(() => currentGraph(model?.id ?? '', currentPath), [model?.id, currentPath]);
  const graphId = addressKey(graph);
  const owner = resolved?.owner;
  const interior = owner?.interior;
  const physical = interior?.physical_domain ?? null;
  const editTopology = editable && !busy && !evaluate;
  const selectedHere = selected.filter((item) => inGraph(item, graph));
  const selectedIds = new Set(selectedHere.map(addressKey));
  const selection = selectedHere[0];
  const fieldOwners = selectedHere.length > 1
    ? selectedHere.every((item) => item.element.kind === 'component') ? selectedHere : []
    : selection?.element.kind === 'attachment' || selection?.element.kind === 'component' ? [selection]
    : !selection ? [graph] : [];
  const canEvaluate = Boolean(fieldEditable && readBody && attachments.data?.editable &&
    attachments.data.revision === saved.revision);
  const canEditField = evaluate ? !busy && canEvaluate : fieldEditable;
  const fieldRun = evaluate?.run ?? projectionRun;
  const fieldBodies = shownModel && !libraryItem ? fieldOwners.map((owner) => ({
    owner, revision: saved.revision, run: fieldRun, evaluate_scope: evaluate?.scope,
    rows: evaluate ? undefined : [...fieldRows ?? []],
    axes: evaluate ? undefined : fieldAxes,
  })) : [];
  type FieldBody = typeof fieldBodies[number];
  const fieldQueries = useQueries({ queries: fieldBodies.map((fieldBody) => ({
    queryKey: ['model-fields', saved.project_id, fieldBody],
    queryFn: ({ signal }: { signal: AbortSignal }) => projectModelFields(saved.project_id, fieldBody, signal),
    enabled: active,
    placeholderData: (previous: Awaited<ReturnType<typeof projectModelFields>> | undefined, query?: { queryKey: readonly unknown[] }) => {
      const previousBody = query?.queryKey[2] as FieldBody | undefined;
      return previousBody && addressKey(previousBody.owner) === addressKey(fieldBody.owner)
        && JSON.stringify(previousBody.evaluate_scope) === JSON.stringify(fieldBody.evaluate_scope)
        && JSON.stringify(previousBody.rows) === JSON.stringify(fieldBody.rows) ? previous : undefined;
    },
  })) });
  const fieldsPlaceholder = fieldQueries.some((query) => query.isPlaceholderData);
  const fieldsError = fieldQueries.find((query) => query.error)?.error ?? null;
  const fieldsData = fieldQueries.every((query) => query.data) ? fieldQueries.map((query) => query.data!) : undefined;
  async function editField(fieldBody: FieldBody, address: ModelAddress, edit: NonNullable<ModelFieldRequest['edit']>) {
    if (!canEditField || fieldsPlaceholder) return studioReason('unavailable', studioName('field'));
    try {
      const result = await projectModelFields(saved.project_id, {
        ...fieldBody, address, edit, lit_steps: [...fieldLitSteps ?? []],
      }, editor.signal);
      if (result.problems?.length) return result.problems[0].reason;
      if (result.command) {
        const applied = await editor.editModel('Set Value', address.document, result.command, saved);
        return applied.status === 'refused' ? applied.problem.detail : undefined;
      }
      if (result.files?.length && await confirmRunRemoval(sweepValueRuns(result.removed_values))) {
        const variant = result.files.find(({ address }) => address.kind === 'variant')?.address.id;
        const dependencies = variant ? saved.file_revisions.map(({ address: file }) => file).filter((file) =>
          (file.kind === 'document' && file.id === address.document) ||
          (file.kind === 'step' && fieldLitSteps?.has(file.id)) ||
          (file.kind === 'variant' && file.id !== variant))
          : [{ kind: 'document' as const, id: address.document },
            ...(fieldBody.run ? [{ kind: 'step' as const, id: fieldBody.run.step }] : []),
            ...(attachments.data?.source ? [{ kind: 'step' as const, id: attachments.data.source.step }] : [])];
        editor.editFiles('Set Field', result.files, { reviewedRevision: fieldBody.revision, dependencies });
      }
      return undefined;
    } catch (error) {
      return apiErrorMessage(error);
    }
  }
  const fieldName = (address: ModelAddress) => address.element.kind === 'field' ? address.element.path.map(String).join('.') : addressKey(address);
  const projectedLists = fieldBodies.map((fieldBody, index) => (fieldsData?.[index]?.fields ?? []).map((field) => {
    const model = fromFieldView(field, {
      id: addressKey(field.address),
      planSources: field.sources,
      source: attachments.data?.field_sources?.[addressKey(field.address)] ??
        attachments.data?.field_sources?.[addressKey(fieldBody.owner)] ??
        attachments.data?.source ?? undefined,
      commit: canEditField && !fieldsPlaceholder ? (edit) => editField(fieldBody, field.address, edit) : undefined,
      rowSelection: !!fieldRows?.length,
      multipleRows: !evaluate && (fieldRows?.length ?? 0) > 1,
      selectedMembers: field.sweep ? selectedSweepValues?.get(field.sweep.id) : undefined,
    });
    return { ...model, id: fieldName(field.address) };
  }));
  const projectedFields = sharedFields(projectedLists);
  const fieldScope = JSON.stringify([saved.project_id, fieldRows, fieldOwners.map(addressKey)]);
  const ownNames = projectedLists.map((list) => new Set(list.map(({ id }) => id.split('.').at(-1)!)));
  const projectedNames = new Set([...ownNames[0] ?? []].filter((name) => ownNames.every((names) => names.has(name))));
  const addressedProblems = new Map<string, string>();
  for (const item of issue?.address ? [...(problems ?? []), { address: issue.address, reason: issue.reason }] : problems ?? []) {
    let address = item.address;
    if (address.document !== graph.document || !currentPath.every((id, index) => address.composites[index] === id)) continue;
    const child = address.composites[currentPath.length];
    if (child) address = at(graph, { kind: 'component', id: child });
    else if (address.element.kind === 'field') address = at(address, address.element.parent);
    if (address.element.kind === 'attachment') address = at(address, address.element.parent);
    addressedProblems.set(addressKey(address), item.reason);
  }
  const portMap = useMemo(() => {
    const ports = owner ? boundaryPorts(owner.ports) : emptyPorts;
    const unused = currentPath.length ? [] : (projection.data?.task_signals ?? []).filter((signal) =>
      !owner?.ports.inputs?.some((input) => input.name === signal.id)).map((signal): Port => ({
      end: { owner: 'inputs', port: `@task:${signal.id}` }, direction: 'output', name: signal.label,
      type: signal.type, taskSignal: signal.id,
    }));
    const map = new Map<string, Port[]>([
      [inputCard, [...ports.filter((port) => idFor(port.end) === inputCard), ...unused]],
      [outputCard, ports.filter((port) => idFor(port.end) === outputCard)],
    ]);
    for (const [id, component] of Object.entries(interior?.components ?? {})) {
      map.set(id, componentPorts(id, component, registry));
    }
    for (const [id, items] of map) map.set(id, items.map((port) => {
      const key = addressKey(at(graph, { kind: 'port', end: port.end }));
      const type = projectedPorts.get(key);
      return type || markedPorts.has(key) ? { ...port, type: type ?? port.type, mark: markedPorts.get(key) } : port;
    }));
    return map;
  }, [owner, interior, registry, graph, projectedPorts, markedPorts, currentPath.length, projection.data]);
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
    if (resolved && active && path.length !== currentPath.length) setPath(currentPath);
  }, [resolved, active, currentPath, path, setPath]);
  const problemAddress = problem?.address;
  useEffect(() => {
    if (!model || !active || !problemAddress) return;
    const address = problemAddress.element.kind === 'field' ? at(problemAddress, problemAddress.element.parent) : problemAddress;
    setPath(address.composites);
    selectElement(state, address);
  }, [model?.id, active, problemAddress, state]);
  useEffect(() => () => connectionReview.current?.abort(), [graphId]);
  useEffect(() => {
    if (!active) return;
    return () => {
      keyboardReview.current?.abort();
      state.setState({ pick: null });
    };
  }, [active, graphId, projectionRunId, saved.revision, state]);

  /** Applies a Model edit; a refusal is answered with its reason and the element it concerns. */
  async function attempt(name: string, operation: Operation, choose = true): Promise<{ address?: ModelAddress; reason: string } | null> {
    if (!model || !editTopology || requestBusy.current) return { reason: studioReason('unavailable', name) };
    requestBusy.current = true;
    const reviewed = editor.getSnapshot().saved;
    setBusy(true);
    setIssue(undefined);
    try {
      const result = await editor.editModel(name, model.id, { operation }, reviewed);
      if (result.status === 'refused') return { address: result.problem.address, reason: result.problem.detail };
      if (result.clipboard) clipboard = result.clipboard;
      if (choose && result.changed.length) {
        const selection = operation.op === 'delete' || operation.op === 'cut' || operation.op === 'ungroup' ? []
          : operation.op === 'group' ? result.changed.slice(-1) : result.changed;
        state.setState({ selection });
      }
      return null;
    } catch (error) {
      const problem = ModelEditProblemSchema.safeParse(error instanceof ApiRequestError ? error.detail : undefined);
      return problem.success ? { address: problem.data.address, reason: problem.data.detail } : { reason: apiErrorMessage(error) };
    } finally {
      requestBusy.current = false;
      setBusy(false);
    }
  }
  async function apply(name: string, operation: Operation, choose = true) {
    if (!model || !editTopology || requestBusy.current) return false;
    const refusal = await attempt(name, operation, choose);
    if (refusal) setIssue(refusal);
    return !refusal;
  }

  async function applyEvaluate(name: string, difference: Difference, onField = false): Promise<string | undefined> {
    if (!readBody || !attachments.data || attachments.data.revision !== saved.revision || requestBusy.current) {
      return studioReason('unavailable', name);
    }
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
      if (difference.op !== 'set_field') {
        selectElement(state, difference.op === 'remove' ? at(result.changed, { kind: 'graph' }) : result.changed);
      }
      return undefined;
    } catch (error) {
      if (onField) return apiErrorMessage(error);
      setIssue({ reason: apiErrorMessage(error) });
      return undefined;
    } finally {
      requestBusy.current = false;
      setBusy(false);
    }
  }

  function route(address: ModelAddress, points: ModelPosition[], name: string) {
    void apply(name, { op: 'route_wire', address, waypoints: points }, false);
  }

  function select(address: ModelAddress, extend = false) {
    setLibraryItem(undefined);
    selectElement(state, address, extend);
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
  function enterComposite(id: string, subject?: string) {
    return bindStudioCommand('enter', () => go([...currentPath, id]), { subject });
  }
  function keyboardConnect(port: Port) {
    if (locked || !editTopology || port.direction !== 'output') return;
    if (port.taskSignal && !projectionRun) return;
    const source = port.taskSignal && projectionRun
      ? { owner: 'task' as const, run: projectionRun, signal: port.taskSignal } : wireEnd(port);
    if (source.owner !== 'inputs' && source.owner !== 'component' && source.owner !== 'task') return;
    const request = new AbortController();
    keyboardReview.current?.abort();
    keyboardReview.current = request;
    void reviewModelConnections(saved.project_id, { revision: saved.revision, graph, source }, request.signal).then(({ targets }) => {
      if (request.signal.aborted) return;
      const verdicts = new Map(targets.map(({ target, reason }) => [endKey(target), reason]));
      state.setState({ pick: {
        accepts: (address, candidate) => inGraph(address, graph) && candidate.direction === 'input' &&
          !(port.taskSignal && candidate.growing) &&
          verdicts.get(endKey(wireEnd(candidate))) === null,
        choose: (address) => {
          if (address.element.kind !== 'port') return;
          const { end } = address.element;
          const id = idFor(end);
          const target = portMap.get(id)?.find((item) => item.direction === 'input' && item.end.port === end.port);
          if (target) connect(idFor(port.end), portId(port), id, portId(target));
        },
      } });
    }).catch((error) => { if (!request.signal.aborted) setIssue({ reason: apiErrorMessage(error) }); })
      .finally(() => { if (keyboardReview.current === request) keyboardReview.current = undefined; });
  }
  function openSearch(point = cursor.current ?? positionAt(objects.length), source?: Port) {
    if (!editTopology) return;
    setSearchText('');
    setSearch(point);
    setSearchSource(source);
    setSearchLibrary(undefined);
    if (source) void libraryReview.current?.then((result) => setSearchLibrary(result.library)).catch((error) => {
      if (!(error instanceof DOMException && error.name === 'AbortError')) setIssue({ reason: apiErrorMessage(error) });
    });
  }
  function add(definition: ComponentDefinition, point?: ModelPosition, from?: Port, to?: string) {
    const occupied = objects.map(([id], index) => persisted?.[addressKey(at(graph, { kind: 'component', id }))] ?? positionAt(index));
    let index = 0;
    let available = positionAt(index);
    while (occupied.some(({ x, y }) => Math.abs(x - available.x) < 196 && Math.abs(y - available.y) < 150)) {
      available = positionAt(++index);
    }
    const position = point ?? available;
    const source = from && wireEnd(from);
    void apply(studioName('addComponent'), {
      op: 'add_component', graph, position,
      ...(source && to && (source.owner === 'inputs' || source.owner === 'component')
        ? { connect_from: source, connect_to_port: to } : {}),
      component: leaf(definition),
    }).then((ok) => { if (ok) setSearch(undefined); });
  }
  function leaf(definition: ComponentDefinition) {
    return {
      kind: 'leaf', name: shortName(definition.name), type: definition.identity?.type_id ?? definition.name,
      revision: definition.revision, fields: Object.fromEntries(Object.entries(definition.default_params ?? {})
        .filter(([name]) => name !== 'dt' && name !== 'control_dt')) as Record<string, JsonValue>,
    } as const;
  }
  function matchesSource(definition: ComponentDefinition, name: string, source: Port) {
    const target = definition.port_types?.inputs?.[name];
    if (typeof source.type === 'string' || target?.kind === 'conserving') return false;
    if (target?.dtype && source.type.base && target.dtype !== source.type.base) return false;
    const shape = source.type.shape;
    return !shape || !target?.shape || (shape.length === target.shape.length &&
      shape.every((axis, index) => typeof axis !== 'number' || axis === target.shape?.[index]));
  }
  function insert(definition: ComponentDefinition, point: ModelPosition, edge: ModelEdge, flipped: boolean) {
    const address = edge.data?.address;
    if (address?.element.kind !== 'wire') return;
    const source = portAt(edge.source, edge.sourceHandle ?? null);
    const target = portAt(edge.target, edge.targetHandle ?? null);
    const candidate = componentPorts('new', leaf(definition), registry);
    const input = source && candidate.find((port) => port.direction === 'input' && matchesSource(definition, port.end.port, source));
    const output = target && candidate.find((port) => port.direction === 'output' && signalFits(port, target));
    if (!input || !output) {
      setIssue({ address, reason: 'This component has no matching ports for the wire.' });
      return;
    }
    void apply(studioName('insertComponent'), {
      op: 'insert_component', address, component: leaf(definition), position: point, flipped,
      input_port: input.end.port, output_port: output.end.port,
    });
  }
  function portAt(node: string, handle: string | null) {
    return handle ? portMap.get(node)?.find((port) => portId(port) === handle) : undefined;
  }

  const connectStart: OnConnectStart = (_, { nodeId, handleId }) => {
    if (!editTopology || locked || !nodeId) return;
    const port = portAt(nodeId, handleId);
    dragSource.current = port;
    if (!port || port.direction !== 'output' || port.taskSignal) return;
    const source = wireEnd(port);
    if (source.owner !== 'inputs' && source.owner !== 'component') return;
    connectionReview.current?.abort();
    const request = new AbortController();
    connectionReview.current = request;
    const reviewed = reviewModelConnections(saved.project_id, { revision: saved.revision, graph, source }, request.signal);
    libraryReview.current = reviewed;
    void reviewed
      .then(({ targets }) => {
        if (!request.signal.aborted) setVerdict(new Map(targets.map(({ target, reason }) => [endKey(target), reason])));
      })
      .catch((error) => { if (!request.signal.aborted) setIssue({ reason: apiErrorMessage(error) }); });
  };
  function connect(sourceNode: string, sourceHandle: string | null, targetNode: string, targetHandle: string | null) {
    const a = portAt(sourceNode, sourceHandle), b = portAt(targetNode, targetHandle);
    if (!a || !b || locked) return;
    const left = wireEnd(a), right = wireEnd(b);
    if (a.taskSignal && projectionRun && b.direction === 'input' && (right.owner === 'component' || right.owner === 'outputs')) {
      void apply(studioName('bindSignal'), { op: 'connect_task_signal', graph, run: projectionRun, signal: a.taskSignal, target: right });
      return;
    }
    if (a.direction === 'physical' && b.direction === 'physical' && physical &&
        (left.owner === 'component' || left.owner === 'boundary') && (right.owner === 'component' || right.owner === 'boundary')) {
      void apply(studioName('connectPorts'), { op: 'connect_physical', graph, connection: { a: left, b: right } });
    } else if (a.direction === 'output' && b.direction === 'input' &&
        (left.owner === 'component' || left.owner === 'inputs') && (right.owner === 'component' || right.owner === 'outputs') &&
        !verdict?.get(endKey(right))) {
      void apply(studioName('connectPorts'), { op: 'connect', graph, source: left, target: right, ...(b.growing ? { grow_target: true } : {}) });
    }
  }

  const effectiveInterior = effectiveModel ? modelPath(effectiveModel, currentPath).owner.interior : interior;
  const effective = new Map<string, readonly Attachment[]>();
  for (const [id, wire] of Object.entries(effectiveInterior?.wires ?? {})) {
    effective.set(addressKey(at(graph, { kind: 'wire', id })), wire.attachments ?? []);
  }
  const portOwners = new Map<string, ModelAddress>();
  for (const ports of portMap.values()) for (const port of ports) {
    if (port.direction === 'physical' || port.taskSignal || port.growing) continue;
    const address = at(graph, { kind: 'port', end: port.end });
    portOwners.set(addressKey(address), address);
    const end = port.end;
    const store = end.owner === 'component' ? effectiveInterior?.components?.[end.component]?.attachments : effectiveInterior?.attachments;
    effective.set(addressKey(address), (port.direction === 'output' ? store?.sources : store?.sinks)?.[end.port] ?? []);
  }

  function move(nodes: ModelCard[]) {
    const placements = nodes.filter((node) => node.data.address.element.kind === 'component' && !node.data.locked)
      .map((node) => ({ address: node.data.address, position: node.position }));
    if (!placements.length) return;
    setHeld(new Map(placements.map(({ address, position }) => [addressKey(address), position])));
    void apply(studioName('moveComponent'), { op: 'move_components', placements }, false).finally(() => setHeld(undefined));
  }

  function makeCard(id: string, index: number, definition?: typeof objects[number][1]): ModelCard {
    const address = definition ? at(graph, { kind: 'component', id }) : graph;
    const key = addressKey(address);
    const type = definition?.kind === 'leaf' ? registry.get(definition.type) : undefined;
    const value = definition?.kind === 'leaf' ? definition.fields?.hidden_size ?? definition.fields?.output_size : undefined;
    const name = definition?.name ?? studioName(id === inputCard ? 'inputs' : 'outputs');
    const caption = definition?.kind === 'leaf'
      ? [shortName(type?.name ?? definition.type), typeof value === 'number' ? value : undefined].filter((item) => item !== undefined).join(' · ')
      : definition?.kind === 'composite' ? graphKind(definition) : undefined;
    const ports = portMap.get(id) ?? emptyPorts;
    const problem = addressedProblems.get(key);
    const enter = definition?.kind === 'composite' ? enterComposite(id).invoke : undefined;
    return {
      id, type: 'modelCard', ariaLabel: name,
      domAttributes: {
        'aria-description': [caption, problem].filter(Boolean).join(' · ') || undefined,
        ...(enter && { onKeyDownCapture: (event) => {
          if (event.target !== event.currentTarget || event.key !== 'Enter') return;
          event.preventDefault();
          event.stopPropagation();
          enter();
        } }),
      },
      position: held?.get(key) ?? (definition ? persisted?.[key] ?? positionAt(index) :
        { x: id === inputCard ? 18 : 245 + 235 * Math.max(1, Math.min(objects.length, 4)), y: 38 }),
      selected: !!definition && selectedIds.has(key),
      draggable: !!definition && editTopology && !locked && !pick,
      selectable: !!definition && !pick,
      data: {
        address, name, caption, ports, collapsed: !!definition && !!collapsed?.[graphId], flipped: flipped?.[key],
        locked: locked || !editTopology, physicalConnections: !!physical, stacked: definition?.kind === 'composite',
        selected: !!definition && selectedIds.has(key),
        holdsSelection: definition?.kind === 'composite' && selected.some((item) => item.document === graph.document &&
          currentPath.every((parent, offset) => item.composites[offset] === parent) && item.composites[currentPath.length] === id),
        problem,
        onPort: (port) => { if (!port.taskSignal && !port.growing) select(at(graph, { kind: 'port', end: port.end })); },
        keyboardConnect,
        canPick: pick ? (port) => pick.accepts(at(graph, { kind: 'port', end: port.end }), port) : undefined,
        cancelPick: pick ? clear : undefined,
        pickPort: pick ? (port) => { pick.choose(at(graph, { kind: 'port', end: port.end })); state.setState({ pick: null }); } : undefined,
        portReasons: verdict && new Map(ports.filter((port) => port.direction === 'input').map((port) =>
          [portId(port), verdict.get(endKey(wireEnd(port)))]).filter((entry): entry is [string, string | null] => entry[1] !== undefined)),
        portProblems: new Map(ports.flatMap((port) => {
          const reason = addressedProblems.get(addressKey(at(graph, { kind: 'port', end: port.end })));
          return reason ? [[portId(port), reason] as const] : [];
        })),
        enter,
        renaming: rename === key, endRename: () => setRename(undefined),
        rename: (name) => { void apply(studioName('rename'), { op: 'rename', address, name }).then((ok) => { if (ok) setRename(undefined); }); },
        addPort: !definition && editTopology ? () => setAddPort(id === inputCard ? 'input' : 'output') : undefined,
      },
    };
  }

  const nodes = [makeCard(inputCard, 0), ...objects.map(([id, component], index) => makeCard(id, index, component)), makeCard(outputCard, 0)];
  const selfLoops = new Map<string, number>();
  const bundled = new Map<string, ModelEdge>();
  const makeEdge = (id: string, from: Parameters<typeof idFor>[0] & { port: string },
    to: Parameters<typeof idFor>[0] & { port: string }, type: ModelEdge['type']): ModelEdge => {
    const isPhysical = type === 'physical';
    const address = at(graph, { kind: isPhysical ? 'physical_connection' : 'wire', id });
    const key = addressKey(address);
    const source = idFor(from);
    const target = idFor(to);
    const endpoint = (end: typeof from) => {
      const port = portMap.get(idFor(end))?.find((candidate) => candidate.end.port === end.port);
      const owner = end.owner === 'component' ? interior?.components?.[end.component!]?.name
        : studioName(end.owner === 'outputs' ? 'outputs' : 'inputs');
      return [owner, port?.name ?? end.port].filter(Boolean).join('.');
    };
    const wireLabel = `${endpoint(from)} → ${endpoint(to)}`;
    const self = !isPhysical && source === target ? selfLoops.get(source) ?? 0 : undefined;
    if (self !== undefined) selfLoops.set(source, self + 1);
    return {
      id, source, target, type, ariaLabel: studioAccessibleName('wire', wireLabel),
      sourceHandle: `${isPhysical ? 'physical' : 'output'}:${from.port}`,
      targetHandle: `${isPhysical ? 'physical' : 'input'}:${to.port}`,
      markerEnd: isPhysical ? undefined : { type: MarkerType.Arrow, width: 12, height: 12, color: 'var(--text-muted)' },
      selected: selectedIds.has(key),
      data: {
        address, selected: selectedIds.has(key), waypoints: waypoints?.[key], problem: addressedProblems.get(key),
        self,
        moveWaypoint: editTopology && !locked ? (index, point) => {
          route(address, (waypoints?.[key] ?? []).map((previous, offset) => offset === index ? point : previous), 'Move Waypoint');
        } : undefined,
        attachments: effective.get(key)?.map((item) => ({
          id: item.id, label: attachmentName(item), mark: stepCount(item, model?.value.step_seconds ?? 0.01),
          description: item.kind === 'delay' ? `${item.delay.value} ${item.delay.unit} · ${wireLabel}` : wireLabel,
          delay: item.kind === 'delay', select: () => select(attachmentAddress(address, item.id)),
          drag: editTopology && !locked ? attachmentAddress(address, item.id) : undefined,
        })),
      },
    };
  };
  const connections = [
    ...wires.map(([id, wire]) => makeEdge(id, wire.source, wire.target, 'signal')),
    ...('connections' in (interior ?? {}) ? Object.entries((interior as { connections?: Record<string, { a: Parameters<typeof makeEdge>[1]; b: Parameters<typeof makeEdge>[2] }> }).connections ?? {})
      .map(([id, connection]) => makeEdge(id, connection.a, connection.b, 'physical')) : []),
  ];
  const edges = connections.flatMap((edge) => {
    if (!collapsed?.[graphId] || edge.source === edge.target || edge.source.startsWith('@') || edge.target.startsWith('@')) return [edge];
    const endpoints = edge.type === 'physical' ? [edge.source, edge.target].sort() : [edge.source, edge.target];
    const key = JSON.stringify([edge.type, ...endpoints]);
    const representative = bundled.get(key);
    if (!representative) {
      bundled.set(key, edge);
      return [edge];
    }
    if (representative.data) representative.data.bundleCount = (representative.data.bundleCount ?? 1) + 1;
    return [];
  });

  const element = selection?.element;
  const component = element?.kind === 'component' ? interior?.components?.[element.id] : undefined;
  const port = element?.kind === 'port' ? portMap.get(idFor(element.end))?.find((item) =>
    addressKey(at(graph, { kind: 'port', end: item.end })) === addressKey(selection!)) : undefined;
  const attachmentOwner = element?.kind === 'attachment' && selection ? at(selection, element.parent) : undefined;
  const attachment = element?.kind === 'attachment' && attachmentOwner
    ? effective.get(addressKey(attachmentOwner))?.find(({ id }) => id === element.id) : undefined;
  const ownerAddress = selection && (element?.kind === 'wire' || element?.kind === 'port') ? selection : attachmentOwner;
  const chain = ownerAddress && effective.get(addressKey(ownerAddress));
  const authored = attachments.data?.records ?? [];
  function isAuthored(address: ModelAddress) {
    return authored.some((record) => addressKey(record.op === 'add' ? attachmentAddress(record.owner, record.attachment.id) : record.address) === addressKey(address));
  }
  async function updateAttachment(address: ModelAddress, item: Attachment) {
    const difference = { op: 'change', address, attachment: item } as const;
    if (evaluate) return applyEvaluate(attachmentName(item), difference, true);
    return (await attempt(attachmentName(item), { op: 'set_attachment', difference }, false))?.reason;
  }
  function addAttachment(address: ModelAddress, kind: 'delay' | 'intervention' | 'probe') {
    const list = effective.get(addressKey(address)) ?? [];
    const id = crypto.randomUUID();
    const item: Attachment = kind === 'delay' ? { id, kind, delay: { unit: 'st', value: 1 }, initial: 0 }
      : kind === 'intervention' ? { id, kind, operation: 'scale', fields: { factor: 1 } }
      : { id, kind, name: port?.name ?? 'Probe' };
    const difference = { op: 'add', owner: address, after: list.at(-1)?.id ?? null, attachment: item } as const;
    if (evaluate) void applyEvaluate(studioAccessibleName('add', attachmentName(item)), difference);
    else void apply(studioAccessibleName('add', attachmentName(item)), { op: 'set_attachment', difference });
  }
  function removeAttachment(address: ModelAddress) {
    if (evaluate) void applyEvaluate(studioName('removeAttachment'), { op: 'remove', address });
    else void apply(studioName('removeAttachment'), { op: 'delete', selection: [address] });
  }
  function armInitial(address: ModelAddress, item: Attachment) {
    if (item.kind !== 'delay') return;
    state.setState({ pick: {
      accepts: (candidate, port) => {
        if (!inGraph(candidate, address) || port.direction !== 'output' || port.end.owner !== 'component') return false;
        const component = interior?.components?.[port.end.component];
        return component?.kind === 'leaf' && component.fields?.stateful !== false &&
          !!registry.get(component.type)?.reset_outputs?.includes(port.end.port);
      },
      choose: (candidate) => { void updateAttachment(address, { ...item, initial: { kind: 'name', reference: addressKey(candidate) } }); },
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
        <Button variant="ghost" pressed={selectedIds.has(addressKey(itemAddress))}
          onClick={() => select(itemAddress)}>
          {stepCount(item, model?.value.step_seconds ?? 0.01)} · {attachmentName(item)}{isAuthored(itemAddress) ? ' ◇' : ''}
        </Button>
      </li>;
    })}
  </ul>;
  function attachmentCommands(address: ModelAddress) {
    const wire = address.element.kind === 'wire';
    const source = address.element.kind === 'port' &&
      (address.element.end.owner === 'inputs' || address.element.end.owner === 'component' && address.element.end.direction === 'output');
    const signalPort = address.element.kind === 'port' &&
      (address.element.end.owner !== 'component' || address.element.end.direction !== 'physical') &&
      address.element.end.owner !== 'boundary';
    if (!(wire || signalPort) || !(editTopology || canEvaluate) || busy) return [];
    return [
      ...(wire && editTopology ? [bindStudioCommand('addDelay', () => addAttachment(address, 'delay'))] : []),
      ...(wire || source ? [bindStudioCommand('addIntervention', () => addAttachment(address, 'intervention'))] : []),
      bindStudioCommand('addProbe', () => addAttachment(address, 'probe')),
    ];
  }

  const componentsOnly = selectedHere.length > 0 && selectedHere.every((item) => item.element.kind === 'component');
  const viewportCommands = canvasViewportCommands(() => viewport.current, !!model);
  const isCollapsed = !!collapsed?.[graphId];
  const collapse = bindStudioCommand('collapseAll', () => {
    void apply(studioName(isCollapsed ? 'expandAll' : 'collapseAll'), { op: 'collapse_graph', graph, collapsed: !isCollapsed }, false);
  }, { name: isCollapsed ? 'expandAll' : 'collapseAll', enabled: editTopology });
  const lock = bindStudioCommand('lock', () => setLocked((current) => !current), {
    name: locked ? 'unlock' : 'lock', enabled: editTopology,
  });
  const leave = currentPath.length ? bindStudioCommand('leave', () => go(currentPath.slice(0, -1))) : undefined;
  const attachmentsForSelection = ownerAddress && !attachment ? attachmentCommands(ownerAddress) : [];
  function topologyCommands(address: ModelAddress, point = cursor.current) {
    const targetElement = address.element;
    if (targetElement.kind === 'component') return [
      bindStudioCommand('flip', () => {
        void apply(studioName('flip'), { op: 'flip_component', address }, false);
      }, { enabled: editTopology }),
      ...(interior?.components?.[targetElement.id]?.kind === 'composite' ? [enterComposite(targetElement.id)] : []),
    ];
    if (targetElement.kind !== 'wire') return [];
    const routePoints = waypoints?.[addressKey(address)] ?? [];
    return [
      bindStudioCommand('addWaypoint', () => {
        if (point) {
          route(address, [...routePoints, point], studioName('addWaypoint'));
          return;
        }
        const edge = connections.find((item) => item.id === targetElement.id);
        const source = nodes.find(({ id }) => id === edge?.source);
        const target = nodes.find(({ id }) => id === edge?.target);
        if (!source || !target) return;
        route(address, [...routePoints, {
          x: (source.position.x + target.position.x) / 2 + 90,
          y: (source.position.y + target.position.y) / 2 + 32,
        }], studioName('addWaypoint'));
      }, { enabled: editTopology }),
      bindStudioCommand('removeWaypoint', () => {
        route(address, routePoints.slice(0, -1), studioName('removeWaypoint'));
      }, { enabled: editTopology && routePoints.length > 0 }),
    ];
  }
  const commands = [
    bindStudioCommand('addComponent', () => openSearch(), { enabled: editTopology && definitions.length > 0 }),
    ...(port?.direction === 'output' ? [bindStudioCommand('connectPorts', () => keyboardConnect(port),
      { enabled: editTopology && !locked })] : []),
    bindStudioCommand('modelSelectAll', () => state.setState({
      selection: objects.map(([id]) => at(graph, { kind: 'component', id })),
    }), { enabled: !!objects.length }),
    bindStudioCommand('modelRemove', () => {
      if (selection?.element.kind === 'attachment') removeAttachment(selection);
      else void apply(studioName('modelRemove'), { op: 'delete', selection: [...selectedHere] });
    }, { enabled: selectedHere.length > 0 && (editTopology || canEvaluate && !!attachment && !busy) }),
    bindStudioCommand('modelRename', () => { if (selection) setRename(addressKey(selection)); }, { enabled: editTopology && !!component }),
    bindStudioCommand('modelDuplicate', () => void apply(studioName('modelDuplicate'), { op: 'duplicate', selection: [...selectedHere] }),
      { enabled: editTopology && componentsOnly }),
    bindStudioCommand('modelGroup', () => void apply(studioName('modelGroup'), { op: 'group', selection: [...selectedHere], name: studioName('modelGroup') }),
      { enabled: editTopology && componentsOnly && !physical }),
    bindStudioCommand('modelUngroup', () => { if (selection) void apply(studioName('modelUngroup'), { op: 'ungroup', address: selection }); },
      { enabled: editTopology && component?.kind === 'composite' && !component.interior.physical_domain }),
    bindStudioCommand('modelCopy', () => void apply(studioName('modelCopy'), { op: 'copy', selection: [...selectedHere] }, false),
      { enabled: editTopology && componentsOnly }),
    bindStudioCommand('modelCut', () => void apply(studioName('modelCut'), { op: 'cut', selection: [...selectedHere] }),
      { enabled: editTopology && componentsOnly }),
    bindStudioCommand('modelPaste', () => { if (clipboard) void apply(studioName('modelPaste'), { op: 'paste', graph, clipboard }); },
      { enabled: editTopology && !!clipboard }),
    ...attachmentsForSelection,
    ...(selection ? topologyCommands(selection) : []),
    ...(leave ? [leave] : []),
    collapse, lock,
    ...viewportCommands,
  ];
  function contextual(event: MouseEvent, address: ModelAddress, point: ModelPosition) {
    if (address.element.kind === 'graph') return;
    event.preventDefault();
    event.stopPropagation();
    if (!selectedIds.has(addressKey(address))) select(address);
    setMenu({ address, x: event.clientX, y: event.clientY, point });
  }
  const fieldEditing = editTopology || canEvaluate && !busy;
  function rawChange(address: ModelAddress): FieldChange | undefined {
    if (!fieldEditing) return undefined;
    return async (path, value) => {
      const difference = { op: 'set_field', address: fieldAddress(address, ...path), value } as const;
      if (evaluate) return applyEvaluate(studioName('setValue'), difference, true);
      return (await attempt(studioName('setValue'), difference, false))?.reason;
    };
  }
  function endLabel(end: { owner: string; component?: string; port: string }) {
    const port = portMap.get(idFor(end))?.find((candidate) => candidate.end.port === end.port);
    const name = end.owner === 'component' ? interior?.components?.[end.component!]?.name
      : studioName(end.owner === 'outputs' ? 'outputs' : 'inputs');
    return [name, port?.name ?? end.port].filter(Boolean).join('.');
  }
  const selectionKeys = selectedHere.map(addressKey);
  function attachmentRows(rows: readonly { owner: ModelAddress; item: Attachment; port?: string }[]) {
    if (!rows.length) return null;
    const byKey = new Map(rows.map((row) => [addressKey(attachmentAddress(row.owner, row.item.id)), row]));
    const nodes: TreeNode[] = [...byKey].map(([id, { owner, item, port }]) => ({
      id, name: attachmentName(item) + (isAuthored(attachmentAddress(owner, item.id)) ? ' ◇' : ''),
      trailing: <>{port && <span>{port}</span>}<span>{stepCount(item, model?.value.step_seconds ?? 0.01)}</span></>,
    }));
    return <Tree aria-label={studioName('attachment')} nodes={nodes} selected={selectionKeys.filter((key) => byKey.has(key))}
      onSelectedChange={(ids) => {
        const row = ids.length ? byKey.get(ids.at(-1)!) : undefined;
        if (row) select(attachmentAddress(row.owner, row.item.id));
      }} />;
  }
  function addRow(groups: readonly { label?: string; commands: ReturnType<typeof attachmentCommands> }[]) {
    const shown = groups.filter(({ commands }) => commands.length);
    if (!shown.length) return null;
    const items = (commands: ReturnType<typeof attachmentCommands>) => commands.map((command) =>
      <MenuItem key={command.id} {...commandAttributes(command)} onClick={command.invoke}>{command.label}</MenuItem>);
    return <InspectorAddRow label={studioName('addAttachment')}>
      {shown.map(({ label, commands }, index) => label
        ? <MenuGroup key={index}><MenuGroupLabel>{label}</MenuGroupLabel>{items(commands)}</MenuGroup>
        : items(commands))}
    </InspectorAddRow>;
  }
  const rows = (scope: string, models: readonly FieldModel[]) =>
    models.map((field) => <ValueField key={JSON.stringify([scope, field.id])} model={field} />);
  const fieldRegion = fieldBodies.length > 0 && <StudioRegion state={studioReadState(
    'field', fieldsData, fieldsError, () => { for (const query of fieldQueries) void query.refetch(); },
  )} />;
  const projectedRows = rows(fieldScope, projectedFields);
  const definitionOf = (item: Component | undefined) => item?.kind === 'leaf' ? registry.get(item.type) : undefined;
  const typeOf = (item: Component) => item.kind === 'composite' ? graphKind(item)
    : shortName(definitionOf(item)?.name ?? item.type);
  const shownType = (name: string, type: string) => type === name ? undefined : type;
  const components = componentsOnly ? selectedHere.flatMap((address) => {
    const item = address.element.kind === 'component' ? interior?.components?.[address.element.id] : undefined;
    return item ? [{ address, item }] : [];
  }) : [];

  function componentInspector(address: ModelAddress, item: Component) {
    const signalPorts = (portMap.get(address.element.kind === 'component' ? address.element.id : '') ?? [])
      .filter((candidate) => candidate.direction !== 'physical' && !candidate.taskSignal && !candidate.growing);
    const ports = signalPorts.map((candidate) => ({ port: candidate, address: at(graph, { kind: 'port', end: candidate.end }) }));
    const attached = ports.flatMap(({ port: candidate, address: owner }) =>
      (effective.get(addressKey(owner)) ?? []).map((entry) => ({ owner, item: entry, port: candidate.name })));
    const add = addRow(ports.map(({ port: candidate, address: owner }) => ({ label: candidate.name, commands: attachmentCommands(owner) })));
    return <Inspector name={item.name} type={shownType(item.name, typeOf(item))}
      attachments={(attached.length || add) && <>{attachmentRows(attached)}{add}</>}>
      {fieldRegion}
      {projectedRows}
      {rows(addressKey(address), componentFields(item, definitionOf(item), rawChange(address), projectedNames))}
    </Inspector>;
  }

  function attachmentInspector(address: ModelAddress, item: Attachment) {
    const change = fieldEditing ? (next: Attachment) => updateAttachment(address, next) : undefined;
    const initial = item.kind === 'delay' ? (() => {
      const value = item.initial;
      const reference = value && typeof value === 'object' && !Array.isArray(value) &&
        value.kind === 'name' && typeof value.reference === 'string' ? value.reference : null;
      const projected = !reference && projectedNames.has('initial');
      const parsed = reference ? linkedAddress(reference) : undefined;
      const label = reference ? parsed ? labelOfPort(parsed) || addressKey(parsed) : reference : undefined;
      const arm = editTopology ? () => armInitial(address, item) : undefined;
      return <div className="flex min-w-0 items-start gap-1">
        <div className="min-w-0 flex-1">{!projected &&
          <ValueField key={JSON.stringify([addressKey(address), label])} model={initialField(item, label, change)} />}</div>
        {arm && <IconButton label={studioName('pickInitial')} icon={<Crosshair />} onClick={arm} />}
        {reference && change && <IconButton label={studioName('clearLink')} icon={<X />} onClick={() => void change({ ...item, initial: 0 })} />}
      </div>;
    })() : null;
    const order = canEvaluate && chain && chain.length > 1 && isAuthored(address) ? [rawField({
      id: 'after', label: studioName('after'), value: chain[chain.indexOf(item) - 1]?.id ?? sourceAttachment,
      type: { kind: 'choice', options: [{ value: sourceAttachment, label: studioName('source') },
        ...chain.filter((entry) => entry.id !== item.id).map((entry) => ({ value: entry.id, label: attachmentName(entry) }))] },
      change: (after) => applyEvaluate('Move Attachment',
        { op: 'change', address, attachment: item, after: after === sourceAttachment ? null : String(after) }, true),
    })] : [];
    return <Inspector name={attachmentName(item)}>
      {fieldRegion}
      {projectedRows}
      {rows(addressKey(address), attachmentFields(item, model?.value.step_seconds ?? 0.01, change, projectedNames))}
      {initial}
      {rows(addressKey(address), order)}
    </Inspector>;
  }

  function ownerInspector(address: ModelAddress) {
    const list = effective.get(addressKey(address)) ?? [];
    const add = addRow([{ commands: attachmentCommands(address) }]);
    const attachmentsShown = (list.length || add) && <>{attachmentRows(list.map((item) => ({ owner: address, item })))}{add}</>;
    if (address.element.kind === 'port' && port) {
      const rename = editTopology && port.end.owner !== 'component' ? rawField({
        id: 'name', label: studioName('name'), type: { kind: 'string' }, value: port.name,
        change: async (name) => String(name).trim()
          ? (await attempt(studioName('rename'), { op: 'rename', address, name: String(name).trim() }, false))?.reason
          : studioReason('takesText', studioName('name')),
      }) : undefined;
      return <Inspector name={port.name} type={portTypeLabel(port)} attachments={attachmentsShown}>
        {rename && rows(addressKey(address), [rename])}
      </Inspector>;
    }
    const wire = address.element.kind === 'wire' ? interior?.wires?.[address.element.id] : undefined;
    return <Inspector name={wire ? `${endLabel(wire.source)} → ${endLabel(wire.target)}` : studioName('wire')}
      type={wire ? studioName('wire') : undefined} attachments={attachmentsShown} />;
  }

  const inspector = libraryItem ? <Inspector name={shortName(libraryItem.name)} type={libraryItem.category}>
    {libraryItem.description && <p className="text-shell-muted">{libraryItem.description}</p>}
    {editTopology && <Button className="justify-self-start" onClick={() => add(libraryItem)}>{studioAccessibleName('add', shortName(libraryItem.name))}</Button>}
  </Inspector> : !model ? null
    : components.length > 1 ? <Inspector name={studioFormat('elementCount', { count: components.length, name: studioName('components').toLowerCase() })}>
      {fieldRegion}
      {projectedRows}
      {rows(selectionKeys.join(), sharedFields(components.map(({ address, item }) =>
        componentFields(item, definitionOf(item), rawChange(address), projectedNames))))}
    </Inspector>
    : component && selection ? componentInspector(selection, component)
    : attachment && selection ? attachmentInspector(selection, attachment)
    : ownerAddress ? ownerInspector(ownerAddress)
    : <Inspector name={currentPath.length ? owner && 'name' in owner ? String(owner.name) : model.id : model.id}
      type={currentPath.length ? undefined : studioName('model')}>
      {fieldRegion}
      {projectedFields.length ? projectedRows : rows('model', modelFields(model.value,
        editTopology && !fieldBodies.length && !currentPath.length
          ? async (path, value) => (await attempt(studioName('step'), {
            op: 'set_field', address: fieldAddress(graph, ...path), value,
          }, false))?.reason : undefined))}
    </Inspector>;

  const outline = <ul className="space-y-0.5">
    {objects.map(([id, component]) => {
      const address = at(graph, { kind: 'component', id });
      const enter = component.kind === 'composite' ? enterComposite(id, component.name) : undefined;
      return <li key={id} className="flex">
        {enter && <IconButton label={enter.label} icon={<ChevronRight />}
          {...commandAttributes(enter)} onClick={enter.invoke} />}
        <Button variant="ghost" align="start" className="min-w-0 flex-1" pressed={selectedIds.has(addressKey(address))}
          onClick={(event) => select(address, event.shiftKey)}>{component.name}</Button>
      </li>;
    })}
    {wires.map(([id, wire]) => {
      const address = at(graph, { kind: 'wire', id });
      const title = [wire.source, wire.target].map((end) =>
        `${end.owner === 'component' ? interior?.components?.[end.component]?.name ?? end.component : end.owner}.${end.port}`).join(' → ');
      return <li key={id}>
        <Button variant="ghost" align="start" className="w-full" pressed={selectedIds.has(addressKey(address))} onClick={() => select(address)}>{title}</Button>
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
    path: model && !!resolved?.entries.length && <nav className="flex min-w-0 items-center gap-1" aria-label={studioName('modelPath')}>
      <IconButton label={studioName('modelRoot')} icon={<House />} onClick={() => go([])} />
      {resolved?.entries.map(({ id, name }, index) => <Button key={id} variant="ghost"
        onClick={() => go(currentPath.slice(0, index + 1))}>› {name}</Button>)}
    </nav>,
    body: model ? <div className="flex h-full min-h-0 flex-col" onKeyDown={(event) => {
      if (event.key === 'Escape' && (state.getState().pick || keyboardReview.current && !keyboardReview.current.signal.aborted)) {
        event.stopPropagation();
        keyboardReview.current?.abort();
        clear();
      }
    }}>
      {(issue || problem) && <p className="px-2 text-sm" role="alert">
        {issue?.reason ?? problem?.reason}
      </p>}
      {evaluate && <StudioRegion state={studioReadState(
        'model', attachments.data, attachments.error, () => { void attachments.refetch(); },
      )} />}
      {!evaluate && projection.error && <StudioRegion state={studioReadState(
        'model', projection.data, projection.error, () => { void projection.refetch(); },
      )} />}
      <div className="min-h-0 flex-1">
        <ModelCanvas key={mountKey} nodes={nodes} edges={edges} physical={!!physical} locked={locked} canEdit={editTopology}
          collapsed={isCollapsed} viewportRef={viewport} collapse={collapse} lock={lock} viewportCommands={viewportCommands}
          onCanvasPoint={(point) => { cursor.current = point; }}
          onSearch={openSearch} onNode={(node, extend) => {
            if (!pick && node.data.address.element.kind === 'component') select(node.data.address, extend);
          }} onEdge={(edge) => { if (!pick && edge.data) select(edge.data.address); }}
          onClear={clear} onBox={(cards) => state.setState({
            selection: cards.filter((node) => node.data.address.element.kind === 'component').map((node) => node.data.address),
          })} onMove={move} onContext={contextual} onConnectStart={connectStart} onConnect={connect}
          onConnectEnd={(point, valid, empty) => {
            setVerdict(undefined);
            if (!valid && empty && dragSource.current?.direction === 'output' && !dragSource.current.taskSignal) openSearch(point, dragSource.current);
            else connectionReview.current?.abort();
            dragSource.current = undefined;
          }} onLibraryDrop={(type, point, edge, reversed) => {
            if (locked) return;
            const definition = registry.get(type);
            if (!definition || !definitions.includes(definition)) return;
            if (edge) insert(definition, point, edge, reversed);
            else add(definition, point);
          }} onAttachmentDrop={(raw, edge) => {
            if (!editTopology || locked || !edge.data) return;
            let value: unknown;
            try {
              value = JSON.parse(raw);
            } catch (error) {
              if (error instanceof SyntaxError) return;
              throw error;
            }
            const parsed = ModelAddressSchema.safeParse(value);
            if (!parsed.success || parsed.data.element.kind !== 'attachment' || !inGraph(parsed.data, graph)) return;
            const address = parsed.data;
            const id = parsed.data.element.id;
            const destination = edge.data.address;
            const after = effective.get(addressKey(destination))?.filter((item) => item.id !== id).at(-1)?.id ?? null;
            void apply(studioName('moveAttachment'), { op: 'move_wire_attachment', address, destination, after })
              .then((ok) => { if (ok) select(attachmentAddress(destination, id)); });
          }} />
      </div>
    </div> : null,
    drawer: model && <section className="space-y-2 p-1 text-sm">
      <SegmentedControl<typeof drawer> aria-label={studioName('modelDrawer')} value={drawer} onValueChange={setDrawer}
        items={[{ value: 'library', label: studioName('library') }, { value: 'outline', label: studioName('outline') }]} />
      {drawer === 'library' ? <>
        <TextField aria-label={studioName('searchLibrary')} leading={<Search />} value={query} onValueChange={setQuery} />
        <StudioRegion state={studioReadState(
          'library', registryQuery.data, registryQuery.error, () => { void registryQuery.refetch(); },
        )}>
          <LeafChoices definitions={definitions} query={query} choose={add} inspect={setLibraryItem} disabled={!editTopology} browse />
        </StudioRegion>
      </> : outline}
    </section>,
    inspector,
    commands,
    overlay: search ? <Dialog ariaLabel={studioName('addComponent')} onClose={() => setSearch(undefined)} manageFocus
      panelClassName="grid max-h-[70vh] w-[min(28rem,90vw)] gap-2 overflow-auto">
      <TextField data-dialog-autofocus aria-label={studioName('components')} leading={<Search />}
        value={searchText} onValueChange={setSearchText} />
      {searchSource ? searchLibrary?.flatMap(({ component, target_port: port }) => {
        const item = registry.get(component);
        return item && searchable(item, searchText) ?
          <Button key={JSON.stringify([item.name, port])} variant="ghost" align="start" disabled={!editTopology}
            onClick={() => add(item, search, searchSource, port)}>{shortName(item.name)} · {port}</Button> : [];
      })
        : <LeafChoices definitions={definitions} query={searchText} choose={(item) => add(item, search)} disabled={!editTopology} />}
    </Dialog> : addPort ? <Dialog ariaLabel={studioName('addPort')} onClose={() => setAddPort(undefined)} manageFocus
      panelClassName="grid max-h-[70vh] w-[min(28rem,90vw)] gap-2 overflow-auto">
      <ValueField model={rawField({
        id: 'name', label: studioName('name'), type: { kind: 'string' }, value: '',
        change: async (raw) => {
          const name = String(raw).trim();
          if (!name) return studioReason('takesText', studioName('name'));
          const refusal = await attempt(studioName('addPort'), {
            op: 'add_port', graph, direction: addPort, port: { id: crypto.randomUUID(), name, type: {} },
          }, false);
          if (!refusal) setAddPort(undefined);
          return refusal?.reason;
        },
      })} />
      {addPort === 'input' && model && currentPath.flatMap((excluded, depth) => {
        const outerPath = currentPath.slice(0, depth);
        const outer = modelPath(model.value, outerPath).owner;
        const candidates = [
          ...boundaryPorts(outer.ports).filter(({ direction }) => direction === 'output'),
          ...Object.entries(outer.interior.components ?? {}).filter(([id]) => id !== excluded).flatMap(([id, component]) =>
            componentPorts(id, component, registry).filter(({ direction }) => direction === 'output')),
        ];
        return candidates.map((port) => {
          const source = wireEnd(port);
          if (source.owner !== 'inputs' && source.owner !== 'component') return null;
          return <Button key={`${depth}:${endKey(source)}`} variant="ghost" align="start" onClick={() => {
            void apply(studioName('bringSignal'), { op: 'bring_signal_in', source_graph: currentGraph(graph.document, outerPath),
              source, destination: graph }, false).then((ok) => { if (ok) setAddPort(undefined); });
          }}>{'name' in outer ? `${outer.name} › ` : ''}{source.owner === 'component' ? `${outer.interior.components?.[source.component]?.name}.` : ''}{port.name}</Button>;
        });
      })}
    </Dialog> : menu && <Menu open onOpenChange={(open) => { if (!open) setMenu(undefined); }}>
      <MenuContent aria-label={studioName('modelCommands')} anchor={{
        getBoundingClientRect: () => DOMRect.fromRect({ x: menu.x, y: menu.y, width: 0, height: 0 }),
      }}>
        {[...commands.filter((command) => command.id.startsWith('model')),
          ...topologyCommands(menu.address, menu.point), ...attachmentCommands(menu.address)]
          .filter((command) => command.enabled).map((command) =>
          <MenuItem key={command.id} {...commandAttributes(command)}
            onClick={() => { setMenu(undefined); command.invoke(); }}>{command.label}</MenuItem>)}
      </MenuContent>
    </Menu>,
  };
}
