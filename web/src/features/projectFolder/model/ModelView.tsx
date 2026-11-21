import { useQueries, useQuery } from '@tanstack/react-query';
import { MarkerType, type OnConnectStart } from '@xyflow/react';
import { useEffect, useId, useMemo, useRef, useState, type KeyboardEvent, type MouseEvent } from 'react';
import { useStore } from 'zustand';
import { fetchComponents } from '@/api/components';
import { projectModelFields, readEvaluateModel, readModelProjection, reviewModelConnections } from '@/api/projectFolders';
import { ApiRequestError, apiErrorMessage, isRefusal } from '@/api/request';
import { StudioRegion, studioReadState } from '@/components/activity/StudioRegion';
import { studioAccessibleName, studioEvaluateValueSource, studioFormat, studioName, studioReason } from '@/components/activity/studioVocabulary';
import { baseChain, documentPayload } from '../documentPayload';
import { canonicalJsonV2 } from '@/utils/canonicalJsonV2';
import { Box, Crosshair, House, Search, TriangleAlert, X } from 'lucide-react';
import clsx from 'clsx';
import { Button, IconButton } from '@/components/ui/Button';
import { focusRing } from '@/components/ui/styles';
import { Dialog } from '@/components/ui/Dialog';
import { Menu, MenuContent, MenuGroup, MenuGroupLabel, MenuItem } from '@/components/ui/Menu';
import { Inspector, InspectorAddRow } from '@/components/ui/Inspector';
import { Tree, type TreeHandle, type TreeNode } from '@/components/ui/Tree';
import { Select } from '@/components/ui/Select';
import { TextField } from '@/components/ui/TextField';
import { ValueField } from '@/components/values/ValueField';
import { fromFieldView, type ExceptionMentions } from '@/components/values/adapters/fieldView';
import { rawField } from '@/components/values/adapters/raw';
import type { FieldModel } from '@/components/values/fieldModel';
import { canvasViewportCommands, type CanvasViewport } from '@/components/canvas/StudioCanvas';
import { bindStudioCommand, commandAttributes } from '@/features/projectWindow/studioCommands';
import type {
  AddModelAttachment, ChangeModelAttachment, ComponentDefinition, EvaluateModelRequest, JsonValue,
  RemoveModelAttachment, SetField,
  ModelAddress, ModelClipboard, ModelConnectionReview, ModelDocument, ModelEditCommand, ModelFieldRequest, ModelPosition, StudioProjectSnapshot, StudioSavedTrainKey,
} from '@/generated/studioContracts';
import { ModelAddressSchema, ModelDocumentSchema, ModelEditProblemSchema } from '@/generated/studioContracts';
import type { ProjectEditController } from '../projectEditController';
import { ModelCanvas, type ModelCard, type ModelEdge } from './ModelCanvas';
import { attachmentFields, componentFields, initialField, modelFields, sharedFields, type Attachment, type FieldChange } from './ModelInspector';
import {
  accepts, addressKey, at, attachmentAddress, boundaryPorts, componentPorts, endKey, fieldAddress, graphKind, modelPath,
  holds, modelRegistry, portId, portTypeLabel, shortName, wireEnd, type Component, type Port,
} from './modelGraph';
import { choosePick, clearSelection, selectElement, useModelSelection, type ElementSelectionStore } from '@/features/projectWindow/elementSelection';
import { Drawer, DrawerSection, focusInspectorField, type DrawerSections } from '@/features/projectWindow/DrawerForms';
import { LibraryList, type LibraryItem } from '@/features/projectWindow/LibraryList';
import { awaitedValue, commandQueue } from './commandQueue';
import { modelOutline } from './modelOutline';
import { sweepValueRuns, type ConfirmRunRemoval } from '../runRemoval';

type Operation = ModelEditCommand['operation'];
type Difference = AddModelAttachment | ChangeModelAttachment | RemoveModelAttachment | SetField;
type Props = {
  model?: { id: string; value: ModelDocument };
  saved: StudioProjectSnapshot;
  editor: ProjectEditController;
  elements: ElementSelectionStore;
  active: boolean;
  editable: boolean;
  fieldEditable: boolean;
  fieldChosenSteps?: ReadonlySet<string>;
  /** Whether the plan read is of the shown revision, so that `fieldChosenSteps` names the steps it chooses. */
  planCurrent: boolean;
  fieldRows?: readonly string[];
  fieldAxes?: Record<string, string[]>;
  selectedSweepValues?: ReadonlyMap<string, ReadonlySet<string>>;
  confirmRunRemoval: ConfirmRunRemoval;
  evaluate?: Omit<EvaluateModelRequest, 'revision'>;
  problem?: { address: ModelAddress; reason: string };
  problems?: readonly { address: ModelAddress; reason: string }[];
  projectionRun?: StudioSavedTrainKey;
  path: readonly string[];
  /** Sets the path shown in `document`, the shown Model when absent. */
  setPath(path: string[], document?: string): void;
  sections: DrawerSections;
  /** Opens Problems showing only those inside the element. */
  openProblems?: (element: ModelAddress, opener: HTMLElement) => void;
  exceptions?: ExceptionMentions;
};
const sourceAttachment = '@source';
const componentDragType = 'application/x-feedbax-component';
const emptyPath: string[] = [];
const inputCard = '@inputs';
const outputCard = '@outputs';
const emptyPorts: Port[] = [];
const emptySelection: readonly ModelAddress[] = [];
let clipboard: ModelClipboard | undefined;

function idFor(end: { owner: string; component?: string }) {
  return end.owner === 'component' ? end.component! : end.owner === 'outputs' ? outputCard : inputCard;
}

/** Whether two read bodies read the same subject, whatever revision each reads it at. */
function sameSubject(previous: unknown, next: object | false | undefined) {
  if (!previous || typeof previous !== 'object' || !next) return false;
  const { revision: _previous, ...a } = previous as { revision?: string };
  const { revision: _next, ...b } = next as { revision?: string };
  return JSON.stringify(a) === JSON.stringify(b);
}

function currentGraph(composites: string[]): ModelAddress {
  return { composites, element: { kind: 'graph' } };
}

function inGraph(address: ModelAddress, graph: ModelAddress) {
  return address.composites.length === graph.composites.length &&
    address.composites.every((id, index) => id === graph.composites[index]);
}

function positionsOf(saved: StudioProjectSnapshot, document: string) {
  return saved.presentation?.documents?.[document]?.positions;
}

function positionAt(index: number): ModelPosition {
  return { x: 245 + 235 * (index % 4), y: 38 + 160 * Math.floor(index / 4) };
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

function LeafChoices({ definitions, query, choose, disabled }: {
  definitions: readonly ComponentDefinition[]; query: string; choose(definition: ComponentDefinition): void; disabled: boolean;
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
      aria-label={studioAccessibleName('add', shortName(definition.name))} disabled={disabled}
      onClick={() => choose(definition)}>{shortName(definition.name)}</Button>)}
  </section>)}</>;
}

export function useStoredModelView({ model, saved, editor, elements: state, active, editable, fieldEditable, fieldChosenSteps, planCurrent, fieldRows, fieldAxes, selectedSweepValues, confirmRunRemoval, evaluate, problem, problems, projectionRun, path, setPath, sections, openProblems, exceptions }: Props) {
  const selected = useModelSelection(state, model?.id);
  const pick = useStore(state, (current) => current.pick);
  const reveal = useStore(state, (current) => current.reveal);
  const viewport = useRef<CanvasViewport>(null);
  const cursor = useRef<ModelPosition>();
  const dragSource = useRef<Port>();
  const connectionReview = useRef<AbortController>();
  const keyboardReview = useRef<AbortController>();
  const libraryReview = useRef<Promise<ModelConnectionReview>>();
  const [run] = useState(commandQueue);
  const [plan] = useState(() => awaitedValue<{ revision: string; chosenSteps: ReadonlySet<string> }>());
  useEffect(() => {
    plan.set(planCurrent ? { revision: saved.revision, chosenSteps: fieldChosenSteps ?? new Set() } : undefined);
  }, [plan, planCurrent, saved.revision, fieldChosenSteps]);
  const [modelIssue, setIssue] = useState<{ document: string; address?: ModelAddress; reason: string }>();
  const issue = modelIssue?.document === model?.id ? modelIssue : undefined;
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
    placeholderData: (previous, query) => sameSubject(query?.queryKey[2], projectionBody) ? previous : undefined,
  });
  const projectedPorts = useMemo(() => new Map(projection.data?.ports?.map(({ address, type }) => [addressKey(address), type])), [projection.data]);
  const markedPorts = useMemo(() => new Map([
    ...(projection.data?.task_inputs ?? []).map((address) => [addressKey(address), 'task'] as const),
    ...(projection.data?.objective_ports ?? []).map((address) => [addressKey(address), 'objective'] as const),
  ]), [projection.data]);
  const attachments = useQuery({
    queryKey: ['model-evaluate-model', saved.project_id, readBody],
    queryFn: ({ signal }) => readEvaluateModel(saved.project_id, readBody!, signal),
    enabled: active && !!model && !!readBody,
    placeholderData: (previous, query) => sameSubject(query?.queryKey[2], readBody) ? previous : undefined,
  });
  const effectiveModel = evaluate ? attachments.data?.resolved_model : projection.data?.resolved_model;
  /** Why the service refuses an evaluation's read: it lands on the Model picker, and the view shows the read document. */
  const refused = evaluate && isRefusal(attachments.error) ? apiErrorMessage(attachments.error) : undefined;
  const shownModel = evaluate && !refused ? effectiveModel : model?.value;
  const resolved = useMemo(() => shownModel && modelPath(shownModel, path), [shownModel, path]);
  const currentPath = useMemo(() => resolved?.entries.map(({ id }) => id) ?? emptyPath, [resolved]);
  const graph = useMemo(() => currentGraph(currentPath), [currentPath]);
  const graphId = addressKey(graph);
  const owner = resolved?.owner;
  const interior = owner?.interior;
  const physical = interior?.physical_domain ?? null;
  /** An Evaluate edits the topology of one selected value's Model; the edit is stored on the value. */
  const editTopology = editable && (!evaluate || evaluate.scope.kind === 'axis_value');
  const selectedRuns = evaluate?.scope.kind === 'axis_value'
    ? { rows: [evaluate.run.row], axes: { [evaluate.scope.axis]: [evaluate.scope.value] } }
    : !evaluate && fieldRows?.length && fieldAxes ? { rows: [...fieldRows], axes: fieldAxes } : undefined;
  const selectedHere = selected.filter((item) => inGraph(item, graph));
  const selectedIds = new Set(selectedHere.map(addressKey));
  const selection = selectedHere[0];
  const fieldOwners = selectedHere.length > 1
    ? selectedHere.every((item) => item.element.kind === 'component') ? selectedHere : []
    : selection?.element.kind === 'attachment' || selection?.element.kind === 'component' ? [selection]
    : !selection ? [graph] : [];
  const canEvaluate = Boolean(fieldEditable && readBody && attachments.data?.editable &&
    attachments.data.revision === saved.revision);
  const canEditField = evaluate ? canEvaluate : fieldEditable;
  const fieldRun = evaluate?.run ?? projectionRun;
  const fieldBodies = model && shownModel && !libraryItem ? fieldOwners.map((owner) => ({
    document: model.id, owner, revision: saved.revision, run: fieldRun, evaluate_scope: evaluate?.scope,
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
      return previousBody && previousBody.document === fieldBody.document && addressKey(previousBody.owner) === addressKey(fieldBody.owner)
        && JSON.stringify(previousBody.evaluate_scope) === JSON.stringify(fieldBody.evaluate_scope)
        && JSON.stringify(previousBody.rows) === JSON.stringify(fieldBody.rows) ? previous : undefined;
    },
  })) });
  const fieldsPlaceholder = fieldQueries.some((query) => query.isPlaceholderData);
  const fieldsError = fieldQueries.find((query) => query.error)?.error ?? null;
  const fieldsData = fieldQueries.every((query) => query.data) ? fieldQueries.map((query) => query.data!) : undefined;
  /**
   * Edits a projected field after the view's earlier commands, once the plan of the revision it edits is current: the
   * edit carries the steps that plan lights.
   */
  async function editField(fieldBody: FieldBody, address: ModelAddress, edit: NonNullable<ModelFieldRequest['edit']>) {
    if (!canEditField || fieldsPlaceholder) return studioReason('unavailable', studioName('field'));
    return run(async () => {
      const { chosenSteps } = await plan.next(({ revision }) => revision === editor.getSnapshot().saved.revision);
      const current = editor.getSnapshot().saved;
      try {
        const result = await projectModelFields(saved.project_id, {
          ...fieldBody, revision: current.revision, address, edit, chosen_steps: [...chosenSteps],
        }, editor.signal);
        if (result.problems?.length) return result.problems[0].reason;
        if (result.command) {
          if (result.files?.length && !await confirmRunRemoval(sweepValueRuns(result.removed_values))) return undefined;
          const applied = await editor.editModel('Set Value', fieldBody.document, result.command, current, fieldBody.run && {
            step: fieldBody.run.step, chosen_steps: [...chosenSteps], files: result.files ?? [],
            ...(fieldBody.evaluate_scope?.kind === 'variant' ? { variant: fieldBody.evaluate_scope.variant } : {}),
          });
          follow(applied);
          return applied.status === 'refused' ? applied.problem.detail : undefined;
        }
        if (result.files?.length && await confirmRunRemoval(sweepValueRuns(result.removed_values))) {
          const variant = result.files.find(({ address }) => address.kind === 'variant')?.address.id;
          const dependencies = variant ? current.file_revisions.map(({ address: file }) => file).filter((file) =>
            (file.kind === 'document' && file.id === fieldBody.document) ||
            (file.kind === 'step' && chosenSteps.has(file.id)) ||
            (file.kind === 'variant' && file.id !== variant))
            : [{ kind: 'document' as const, id: fieldBody.document },
              ...(fieldBody.run ? [{ kind: 'step' as const, id: fieldBody.run.step }] : []),
              ...(attachments.data?.source ? [{ kind: 'step' as const, id: attachments.data.source.step }] : [])];
          const settled = await editor.editFiles('Set Field', result.files, { reviewedRevision: current.revision, dependencies });
          if (settled.status === 'refused') return apiErrorMessage(settled.error);
        }
        return undefined;
      } catch (error) {
        return apiErrorMessage(error);
      }
    });
  }
  const fieldName = (address: ModelAddress) => address.element.kind === 'field' ? address.element.path.map(String).join('.') : addressKey(address);
  const shownChain = model ? baseChain(saved.project.documents, model.id) : [];
  const changedText = evaluate ? studioEvaluateValueSource({ kind: 'evaluate', step: evaluate.run.step }) : studioName('changed');
  const projectedLists = fieldBodies.map((fieldBody, index) => (fieldsData?.[index]?.fields ?? []).map((field) => {
    const model = fromFieldView(field, {
      id: addressKey(field.address),
      planSources: field.sources?.filter((source) => !(source.kind === 'document' && shownChain.includes(source.name))),
      exceptions,
      changed: field.changed ? {
        text: changedText,
        reset: canEditField && !fieldsPlaceholder ? () => editField(fieldBody, field.address, { kind: 'reset' }) : undefined,
      } : undefined,
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
  const fieldScope = JSON.stringify([saved.project_id, model?.id, fieldRows, fieldOwners.map(addressKey)]);
  const ownNames = projectedLists.map((list) => new Set(list.map(({ id }) => id.split('.').at(-1)!)));
  const projectedNames = new Set([...ownNames[0] ?? []].filter((name) => ownNames.every((names) => names.has(name))));
  const addressedProblems = new Map<string, string>();
  for (const item of issue?.address ? [...(problems ?? []), { address: issue.address, reason: issue.reason }] : problems ?? []) {
    let address = item.address;
    if (!currentPath.every((id, index) => address.composites[index] === id)) continue;
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
  const libraryItems = useMemo(() => definitions.map((definition): LibraryItem => ({
    id: definition.name, name: shortName(definition.name), group: definition.category, glyph: <Box />,
    about: {
      description: definition.description, inputs: definition.input_ports ?? [], outputs: definition.output_ports ?? [],
      parameters: (definition.param_schema ?? []).map(({ name }) => name),
    },
  })), [definitions]);
  const persisted = model && positionsOf(saved, model.id);
  const presentation = model && saved.presentation?.documents?.[model.id];
  const collapsed = presentation?.collapsed;
  const flipped = presentation?.flipped;
  const waypoints = presentation?.waypoints;
  const objects = Object.entries(interior?.components ?? {});
  const wires = Object.entries(interior?.wires ?? {});
  const mountKey = JSON.stringify([saved.project_id, model?.id, graphId]);
  const handledReveal = useRef<typeof reveal>();
  useEffect(() => {
    if (!model || !active || !reveal || reveal.document !== model.id || handledReveal.current === reveal) return;
    handledReveal.current = reveal;
    if (!state.getState().reveal) return;
    state.setState({ reveal: null });
    setPath(reveal.address.composites);
  }, [active, model?.id, reveal, setPath, state]);
  useEffect(() => {
    if (resolved && active && path.length !== currentPath.length) setPath(currentPath);
  }, [resolved, active, currentPath, path, setPath]);
  const problemAddress = problem?.address;
  useEffect(() => {
    if (!model || !active || !problemAddress) return;
    const address = problemAddress.element.kind === 'field' ? at(problemAddress, problemAddress.element.parent) : problemAddress;
    setPath(address.composites);
    selectElement(state, { document: model.id, address });
  }, [model?.id, active, problemAddress, state]);
  useEffect(() => () => connectionReview.current?.abort(), [mountKey]);
  useEffect(() => {
    if (!active) return;
    return () => {
      keyboardReview.current?.abort();
      state.setState({ pick: null });
    };
  }, [active, mountKey, projectionRunId, saved.revision, state]);

  /**
   * Applies a Model edit after the view's earlier commands; a refusal is answered with its reason and the element it
   * concerns. Whether the edit is allowed is decided when it is issued, the revision it edits when it runs.
   */
  async function attempt(name: string, operation: Operation, choose = true): Promise<{ address?: ModelAddress; reason: string } | null> {
    if (!model || !editTopology) return { reason: studioReason('unavailable', name) };
    return run(async () => {
        const subject = evaluate?.run ?? projectionRun;
        const scope = subject && { step: subject.step, chosen_steps: [...(await plan.next(({ revision }) =>
          revision === editor.getSnapshot().saved.revision)).chosenSteps], ...selectedRuns };
        const reviewed = editor.getSnapshot().saved;
        setIssue(undefined);
        try {
          const result = await editor.editModel(name, model.id, { operation }, reviewed, scope);
          if (result.status === 'refused') return { address: result.problem.address, reason: result.problem.detail };
          const shown = follow(result);
          if (result.clipboard) clipboard = result.clipboard;
          if (choose && result.changed.length) {
            const selection = operation.op === 'delete' || operation.op === 'cut' || operation.op === 'ungroup' ? []
              : operation.op === 'group' ? result.changed.slice(-1) : result.changed;
            setSelection(selection, shown);
          }
          return null;
        } catch (error) {
          const problem = ModelEditProblemSchema.safeParse(error instanceof ApiRequestError ? error.detail : undefined);
          return problem.success ? { address: problem.data.address, reason: problem.data.detail } : { reason: apiErrorMessage(error) };
        }
    });
  }
  async function apply(name: string, operation: Operation, choose = true) {
    if (!model) return false;
    const refusal = await attempt(name, operation, choose);
    if (refusal) setIssue({ document: model.id, ...refusal });
    return !refusal;
  }

  async function applyEvaluate(name: string, difference: Difference, onField = false): Promise<string | undefined> {
    if (!model || !readBody || !attachments.data || attachments.data.revision !== saved.revision) {
      return studioReason('unavailable', name);
    }
    return run(async () => {
        setIssue(undefined);
        try {
          let changed: ModelAddress;
          let shown = model.id;
          const { scope } = readBody;
          const { chosenSteps } = await plan.next(({ revision }) => revision === editor.getSnapshot().saved.revision);
          const applied = await editor.editModel(name, model.id, {
            operation: difference.op === 'set_field' ? difference : { op: 'set_attachment', difference },
          }, editor.getSnapshot().saved, {
            step: readBody.run.step, chosen_steps: [...chosenSteps],
            ...(scope.kind === 'variant' ? { variant: scope.variant } : {}),
            ...(scope.kind === 'rows' ? { rows: [...scope.rows] } : {}),
            ...(scope.kind === 'axis_value' ? { rows: [readBody.run.row], axes: { [scope.axis]: [scope.value] } } : {}),
          });
          if (applied.status === 'refused') throw new Error(applied.problem.detail);
          shown = follow(applied) ?? shown;
          const [first] = applied.changed;
          if (!first) throw new Error('The edit changed nothing.');
          changed = first;
          if (difference.op !== 'set_field') {
            selectElement(state, { document: shown, address: difference.op === 'remove' ? at(changed, { kind: 'graph' }) : changed });
          }
          return undefined;
        } catch (error) {
          if (onField) return apiErrorMessage(error);
          setIssue({ document: model.id, reason: apiErrorMessage(error) });
          return undefined;
        }
    });
  }

  function route(address: ModelAddress, points: ModelPosition[], name: string) {
    void apply(name, { op: 'route_wire', address, waypoints: points }, false);
  }

  function select(address: ModelAddress, extend = false) {
    setLibraryItem(undefined);
    if (model) selectElement(state, { document: model.id, address }, extend);
  }
  function setSelection(addresses: readonly ModelAddress[], document = model?.id) {
    if (document) state.setState({ selection: addresses.map((address) => ({ document, address })) });
  }
  /**
   * After an edit that made a document based on the shown one for the shown step, the view stays where it was in it.
   * Answers the document the view now shows.
   */
  function follow(applied: { status: string; created?: readonly string[]; snapshot?: { project: { documents?: Record<string, unknown> | null } } }) {
    if (!model || applied.status !== 'applied') return model?.id;
    const documents = applied.snapshot?.project.documents;
    const next = applied.created?.find((name) => {
      const document = documents?.[name];
      return !!document && typeof document === 'object' && 'base' in document &&
        (document as { base: { document: string } }).base.document === model.id;
    });
    if (!next) return model.id;
    const from = model.id;
    state.setState(({ selection }) => ({ selection: selection.map((item) => item.document === from ? { ...item, document: next } : item) }));
    setPath(currentPath, next);
    return next;
  }
  function clear() {
    if (pick) state.setState({ pick: null });
    else clearSelection(state);
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
    if (!model || locked || !editTopology || port.direction !== 'output') return;
    if (port.taskSignal && !projectionRun) return;
    const source = port.taskSignal && projectionRun
      ? { owner: 'task' as const, run: projectionRun, signal: port.taskSignal } : wireEnd(port);
    if (source.owner !== 'inputs' && source.owner !== 'component' && source.owner !== 'task') return;
    const request = new AbortController();
    keyboardReview.current?.abort();
    keyboardReview.current = request;
    void reviewModelConnections(saved.project_id, { revision: saved.revision, document: model.id, graph, source }, request.signal).then(({ targets }) => {
      if (request.signal.aborted) return;
      const verdicts = new Map(targets.map(({ target, reason }) => [endKey(target), reason]));
      state.setState({ pick: {
        accepts: (target) => target.kind === 'port' && target.document === model.id && inGraph(target.address, graph) && target.port.direction === 'input' &&
          !(port.taskSignal && target.port.growing) &&
          verdicts.get(endKey(wireEnd(target.port))) === null,
        choose: (picked) => {
          if (picked.kind !== 'port' || picked.address.element.kind !== 'port') return;
          const { end } = picked.address.element;
          const id = idFor(end);
          const target = portMap.get(id)?.find((item) => item.direction === 'input' && item.end.port === end.port);
          if (target) connect(idFor(port.end), portId(port), id, portId(target));
        },
      } });
    }).catch((error) => { if (!request.signal.aborted) setIssue({ document: model.id, reason: apiErrorMessage(error) }); })
      .finally(() => { if (keyboardReview.current === request) keyboardReview.current = undefined; });
  }
  function openSearch(point = cursor.current ?? positionAt(objects.length), source?: Port) {
    if (!model || !editTopology) return;
    setSearchText('');
    setSearch(point);
    setSearchSource(source);
    setSearchLibrary(undefined);
    if (source) void libraryReview.current?.then((result) => setSearchLibrary(result.library)).catch((error) => {
      if (!(error instanceof DOMException && error.name === 'AbortError')) setIssue({ document: model.id, reason: apiErrorMessage(error) });
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
    if (!model || address?.element.kind !== 'wire') return;
    const source = portAt(edge.source, edge.sourceHandle ?? null);
    const target = portAt(edge.target, edge.targetHandle ?? null);
    const candidate = componentPorts('new', leaf(definition), registry);
    const input = source && candidate.find((port) => port.direction === 'input' && matchesSource(definition, port.end.port, source));
    const output = target && candidate.find((port) => port.direction === 'output' && signalFits(port, target));
    if (!input || !output) {
      setIssue({ document: model.id, address, reason: 'This component has no matching ports for the wire.' });
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
    if (!model || !editTopology || locked || !nodeId) return;
    const port = portAt(nodeId, handleId);
    dragSource.current = port;
    if (!port || port.direction !== 'output' || port.taskSignal) return;
    const source = wireEnd(port);
    if (source.owner !== 'inputs' && source.owner !== 'component') return;
    connectionReview.current?.abort();
    const request = new AbortController();
    connectionReview.current = request;
    const reviewed = reviewModelConnections(saved.project_id, { revision: saved.revision, document: model.id, graph, source }, request.signal);
    libraryReview.current = reviewed;
    void reviewed
      .then(({ targets }) => {
        if (!request.signal.aborted) setVerdict(new Map(targets.map(({ target, reason }) => [endKey(target), reason])));
      })
      .catch((error) => { if (!request.signal.aborted) setIssue({ document: model.id, reason: apiErrorMessage(error) }); });
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

  /** The attachment chains of the shown graph's wires and ports, as `source` holds them, by owner. */
  function chains(source: typeof interior) {
    const found = new Map<string, readonly Attachment[]>();
    for (const [id, wire] of Object.entries(source?.wires ?? {})) {
      found.set(addressKey(at(graph, { kind: 'wire', id })), wire.attachments ?? []);
    }
    for (const ports of portMap.values()) for (const port of ports) {
      if (port.direction === 'physical' || port.taskSignal || port.growing) continue;
      const address = at(graph, { kind: 'port', end: port.end });
      const end = port.end;
      const store = end.owner === 'component' ? source?.components?.[end.component]?.attachments : source?.attachments;
      found.set(addressKey(address), (port.direction === 'output' ? store?.sources : store?.sinks)?.[end.port] ?? []);
    }
    return found;
  }
  const effectiveInterior = effectiveModel ? modelPath(effectiveModel, currentPath).owner.interior : interior;
  const effective = chains(effectiveInterior);
  /**
   * The document the shown one's differences are from: its base, or in an evaluation the nearest document the read one
   * shares with the subject's.
   */
  const shownDocument = model && saved.project.documents?.[model.id];
  const comparedName = evaluate ? attachments.data?.changed_from ?? undefined
    : shownDocument && 'base' in shownDocument ? shownDocument.base.document : undefined;
  const compared = (() => {
    const parsed = comparedName && ModelDocumentSchema.safeParse(documentPayload(saved.project.documents, saved.composed, comparedName));
    if (!parsed || !parsed.success) return undefined;
    const reached = modelPath(parsed.data, currentPath);
    return reached.entries.length === currentPath.length ? reached.owner.interior : null;
  })();
  const ownChains = compared !== undefined ? chains(model ? modelPath(model.value, currentPath).owner.interior : undefined) : undefined;
  const comparedChains = compared ? chains(compared) : undefined;
  /** The attachment at `address` is a difference of the shown document: absent from, or other than, the one it is from. */
  function fromDifference(address: ModelAddress) {
    const element = address.element;
    if (!ownChains || element.kind !== 'attachment') return false;
    const owner = addressKey(at(address, element.parent));
    const item = ownChains.get(owner)?.find(({ id }) => id === element.id);
    const was = comparedChains?.get(owner)?.find(({ id }) => id === element.id);
    return !!item && (!was || canonicalJsonV2(was) !== canonicalJsonV2(item));
  }

  function move(nodes: ModelCard[]) {
    const placements = nodes.filter((node) => !node.data.locked)
      .map((node) => ({ address: node.data.address, position: node.position }));
    if (!placements.length) return;
    setHeld(new Map(placements.map(({ address, position }) => [addressKey(address), position])));
    void apply(studioName('moveCard'), { op: 'move_cards', placements }, false).finally(() => setHeld(undefined));
  }

  function makeCard(id: string, index: number, definition?: typeof objects[number][1]): ModelCard {
    const address = at(graph, definition ? { kind: 'component', id } : { kind: id === inputCard ? 'inputs' : 'outputs' });
    const key = addressKey(address);
    const type = definition?.kind === 'leaf' ? registry.get(definition.type) : undefined;
    const value = definition?.kind === 'leaf' ? definition.fields?.hidden_size ?? definition.fields?.output_size : undefined;
    const name = definition?.name ?? studioName(id === inputCard ? 'inputs' : 'outputs');
    const description = definition?.kind === 'leaf'
      ? [shortName(type?.name ?? definition.type), typeof value === 'number' ? value : undefined].filter((item) => item !== undefined).join(' · ')
      : definition?.kind === 'composite' ? graphKind(definition) : undefined;
    const ports = portMap.get(id) ?? emptyPorts;
    const problem = addressedProblems.get(definition ? key : addressKey(graph));
    const enter = definition?.kind === 'composite' ? enterComposite(id).invoke : undefined;
    return {
      id, type: 'modelCard', ariaLabel: name,
      domAttributes: {
        'aria-description': [description, problem].filter(Boolean).join(' · ') || undefined,
        ...(enter && { onKeyDownCapture: (event: KeyboardEvent<HTMLElement>) => {
          if (event.target !== event.currentTarget || event.key !== 'Enter') return;
          event.preventDefault();
          event.stopPropagation();
          enter();
        } }),
      },
      position: held?.get(key) ?? persisted?.[key] ?? (definition ? positionAt(index) :
        { x: id === inputCard ? 18 : 245 + 235 * Math.max(1, Math.min(objects.length, 4)), y: 38 }),
      selected: !!definition && selectedIds.has(key),
      draggable: editTopology && !locked && !pick,
      selectable: !!definition && !pick,
      data: {
        address, name, ports, collapsed: !!definition && !!collapsed?.[graphId], flipped: flipped?.[key],
        locked: locked || !editTopology, physicalConnections: !!physical, stacked: definition?.kind === 'composite',
        selected: !!definition && selectedIds.has(key),
        holdsSelection: definition?.kind === 'composite' && selected.some((item) =>
          currentPath.every((parent, offset) => item.composites[offset] === parent) && item.composites[currentPath.length] === id),
        problem,
        onPort: (port) => { if (!port.taskSignal && !port.growing) select(at(graph, { kind: 'port', end: port.end })); },
        keyboardConnect,
        canPick: pick && model ? (port) => pick.accepts({ kind: 'port', document: model.id, address: at(graph, { kind: 'port', end: port.end }), port }) : undefined,
        cancelPick: pick ? clear : undefined,
        pickPort: pick && model ? (port) => { choosePick(state, { kind: 'port', document: model.id, address: at(graph, { kind: 'port', end: port.end }), port }); } : undefined,
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
          drag: model && editTopology && !locked ? { document: model.id, address: attachmentAddress(address, item.id) } : undefined,
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
  function isAuthored(address: ModelAddress) {
    return fromDifference(address);
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
    if (!model || item.kind !== 'delay') return;
    state.setState({ pick: {
      accepts: (target) => {
        if (target.kind !== 'port' || target.document !== model.id) return false;
        const { port } = target;
        if (!inGraph(target.address, address) || port.direction !== 'output' || port.end.owner !== 'component') return false;
        const component = interior?.components?.[port.end.component];
        return component?.kind === 'leaf' && component.fields?.stateful !== false &&
          !!registry.get(component.type)?.reset_outputs?.includes(port.end.port);
      },
      choose: (target) => {
        if (target.kind === 'port') void updateAttachment(address, { ...item, initial: { kind: 'name', reference: addressKey(target.address) } });
      },
    } });
  }
  function labelOfPort(address: ModelAddress) {
    if (address.element.kind !== 'port') return '';
    const end = address.element.end;
    const candidate = portMap.get(idFor(end))?.find((item) =>
      addressKey(at(graph, { kind: 'port', end: item.end })) === addressKey(address));
    return [end.owner === 'component' ? interior?.components?.[end.component]?.name : end.owner, candidate?.name].filter(Boolean).join('.');
  }
  function attachmentCommands(address: ModelAddress) {
    const wire = address.element.kind === 'wire';
    const source = address.element.kind === 'port' &&
      (address.element.end.owner === 'inputs' || address.element.end.owner === 'component' && address.element.end.direction === 'output');
    const signalPort = address.element.kind === 'port' &&
      (address.element.end.owner !== 'component' || address.element.end.direction !== 'physical') &&
      address.element.end.owner !== 'boundary';
    if (!(wire || signalPort) || !(editTopology || canEvaluate)) return [];
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
    bindStudioCommand('modelSelectAll', () => setSelection(
      objects.map(([id]) => at(graph, { kind: 'component', id })),
    ), { enabled: !!objects.length }),
    bindStudioCommand('modelRemove', () => {
      if (selection?.element.kind === 'attachment') removeAttachment(selection);
      else void apply(studioName('modelRemove'), { op: 'delete', selection: [...selectedHere] });
    }, { enabled: selectedHere.length > 0 && (editTopology || canEvaluate && !!attachment) }),
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
    if (address.element.kind === 'inputs' || address.element.kind === 'outputs') return;
    event.preventDefault();
    event.stopPropagation();
    if (!selectedIds.has(addressKey(address))) select(address);
    setMenu({ address, x: event.clientX, y: event.clientY, point });
  }
  const fieldEditing = editTopology || canEvaluate;
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
  // A field is keyed by the field it shows, never by where its value came from: when the projection takes over a
  // field first drawn from the raw document, the same instance stays, with any draft typed meanwhile.
  const rows = (scope: string, models: readonly FieldModel[]) =>
    models.map((field) => <ValueField key={JSON.stringify([scope, field.id.split('.').at(-1)])} model={field} />);
  const fields = (raw: readonly FieldModel[]) => rows(fieldScope, [...projectedFields, ...raw]);
  const fieldRegion = fieldBodies.length > 0 && <StudioRegion state={studioReadState(
    'field', fieldsData, fieldsError, () => { for (const query of fieldQueries) void query.refetch(); },
  )} />;
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
      {fields(componentFields(item, definitionOf(item), rawChange(address), projectedNames))}
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
      {fields(attachmentFields(item, change, projectedNames))}
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
      {fields(sharedFields(components.map(({ address, item }) =>
        componentFields(item, definitionOf(item), rawChange(address), projectedNames))))}
    </Inspector>
    : component && selection ? componentInspector(selection, component)
    : attachment && selection ? attachmentInspector(selection, attachment)
    : ownerAddress ? ownerInspector(ownerAddress)
    : <Inspector name={currentPath.length ? owner && 'name' in owner ? String(owner.name) : model.id : model.id}
      type={currentPath.length ? undefined : studioName('model')}>
      {fieldRegion}
      {fields(projectedFields.length ? [] : modelFields(model.value,
        editTopology && !fieldBodies.length && !currentPath.length
          ? async (path, value) => (await attempt(studioName('step'), {
            op: 'set_field', address: fieldAddress(graph, ...path), value,
          }, false))?.reason : undefined))}
    </Inspector>;

  const outlineTree = useRef<TreeHandle>(null);
  const problemCount = (address: ModelAddress) => (problems ?? []).filter((item) => holds(address, item.address)).length;
  const outline = shownModel && model && modelOutline({
    model: shownModel, effective: effectiveModel ?? undefined, registry,
    attachmentName: (item, address) => attachmentName(item) + (isAuthored(address) ? ' ◇' : ''),
    trailing: (address) => {
      const count = problemCount(address);
      return count > 0 && <Button variant="ghost" size="mark" className="gap-0.5 text-shell-status-failure"
        aria-label={studioAccessibleName('problems', count)} onClick={(event) => {
          event.stopPropagation();
          openProblems?.(address, event.currentTarget);
        }}><TriangleAlert aria-hidden /><span className="tabular-nums">{count}</span></Button>;
    },
  });
  const outlineRow = (address: ModelAddress): string | undefined => {
    const key = addressKey(address);
    if (outline?.addresses.has(key)) return key;
    const element = address.element;
    if (element.kind === 'field' || element.kind === 'attachment') return outlineRow(at(address, element.parent));
    if (element.kind === 'port' && element.end.owner === 'component') return outlineRow(at(address, { kind: 'component', id: element.end.component }));
    const parent = address.composites.at(-1);
    return parent ? outlineRow({ ...address, composites: address.composites.slice(0, -1), element: { kind: 'component', id: parent } }) : undefined;
  };
  const pathKey = currentPath.join('\u0000');
  useEffect(() => {
    const id = currentPath.at(-1);
    if (id) outlineTree.current?.open(addressKey({ ...graph, composites: currentPath.slice(0, -1), element: { kind: 'component', id } }));
  }, [pathKey, !!outline]);
  const outlineCommands = (address: ModelAddress) => [
    ...commands.filter((command) => command.id.startsWith('model')).map((command) => command.id === 'modelRename'
      ? { ...command, invoke: () => outlineTree.current?.rename(addressKey(address)) } : command),
    ...topologyCommands(address), ...attachmentCommands(address),
  ].filter((command) => command.enabled);

  return {
    documentId: model?.id,
    refusal: refused,
    path: model && !!resolved?.entries.length && <nav className="flex min-w-0 items-center gap-1" aria-label={studioName('modelPath')}>
      <IconButton label={studioName('modelRoot')} icon={<House />} onClick={() => go([])} />
      {resolved?.entries.map(({ id, name }, index) => <Button key={id} variant="ghost"
        onClick={() => go(currentPath.slice(0, index + 1))}>› {name}</Button>)}
    </nav>,
    body: model ? <div className="flex h-full min-h-0 flex-col" onKeyDown={(event) => {
      if (event.key === 'Escape' && (pick || keyboardReview.current && !keyboardReview.current.signal.aborted)) {
        event.stopPropagation();
        keyboardReview.current?.abort();
        clear();
      }
    }}>
      {(issue || problem) && <p className="px-2 text-sm" role="alert">
        {issue?.reason ?? problem?.reason}
      </p>}
      {evaluate && !refused && <StudioRegion state={studioReadState(
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
          onClear={clear} onBox={(cards) => setSelection(
            cards.filter((node) => node.data.address.element.kind === 'component').map((node) => node.data.address),
          )} onMove={move} onContext={contextual} onConnectStart={connectStart} onConnect={connect}
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
            if (typeof value !== 'object' || !value || !('document' in value) || value.document !== model.id || !('address' in value)) return;
            const parsed = ModelAddressSchema.safeParse(value.address);
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
    drawer: model && <Drawer>
      <LibraryList sections={sections} section="model.library" name={studioName('library')} searchName="searchLibrary"
        state={studioReadState('library', registryQuery.data, registryQuery.error, () => { void registryQuery.refetch(); })}
        items={libraryItems} selected={libraryItem?.name} dragType={editTopology ? componentDragType : undefined}
        onInspect={(name) => setLibraryItem(registry.get(name))}
        onPlace={editTopology ? (name) => { const item = registry.get(name); if (item) add(item); } : undefined} />
      <DrawerSection sections={sections} id="model.outline" name={studioName('outline')}>
        {outline && <Tree ref={outlineTree} aria-label={studioName('outline')} nodes={outline.nodes}
          selected={selected.flatMap((address) => outlineRow(address) ?? [])}
          onSelectedChange={(ids) => {
            const chosen = ids.flatMap((id) => outline.addresses.get(id) ?? []);
            const last = chosen.at(-1);
            if (!last) return;
            setLibraryItem(undefined);
            if (last.composites.join('\u0000') !== currentPath.join('\u0000')) go(last.composites);
            setSelection(chosen);
          }}
          onActivate={(id) => {
            const address = outline.addresses.get(id);
            const element = address?.element;
            if (!address || !element) return;
            if (element.kind === 'component' && modelPath(shownModel!, [...address.composites, element.id]).entries.length > address.composites.length) {
              go([...address.composites, element.id]);
            } else if (!outline.parents.has(id)) {
              focusInspectorField();
            }
          }}
          onRename={editTopology ? (id, name) => {
            const address = outline.addresses.get(id);
            if (!address || !name.trim()) return false;
            void apply(studioName('rename'), { op: 'rename', address, name: name.trim() });
          } : undefined}
          menu={(node) => {
            const address = outline.addresses.get(node.id);
            return address && outlineCommands(address).map((command) =>
              <MenuItem key={command.id} {...commandAttributes(command)} onClick={command.invoke}>{command.label}</MenuItem>);
          }} />}
      </DrawerSection>
    </Drawer>,
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
            void apply(studioName('bringSignal'), { op: 'bring_signal_in', source_graph: currentGraph(outerPath),
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
