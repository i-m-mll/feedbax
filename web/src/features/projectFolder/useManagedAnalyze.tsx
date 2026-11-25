import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { baseChain, baseOf, documentPayload, stepDocument } from './documentPayload';
import { ChartLine, Database, Sigma } from 'lucide-react';
import { useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { StudioRegion, studioRegionIssue, type StudioRegionState } from '@/components/activity/StudioRegion';
import { readStudioAnalyze } from '@/api/studioAnalyze';
import { studioAccessibleName, studioName, studioRunStatus } from '@/components/activity/studioVocabulary';
import { CanvasStrip } from '@/components/canvas/CanvasStrip';
import { canvasViewportCommands, type CanvasViewport } from '@/components/canvas/StudioCanvas';
import { Button } from '@/components/ui/Button';
import { carryKind } from '@/components/ui/carry';
import { Dialog } from '@/components/ui/Dialog';
import { Inspector } from '@/components/ui/Inspector';
import { KeyField } from '@/components/ui/KeyField';
import { TextField } from '@/components/ui/TextField';
import { humanizeLabel } from '@/components/values/humanizeLabel';
import { SettingsForm } from '@/components/values/SettingsForm';
import { setIn } from '@/components/values/adapters/jsonSchema';
import { fromParamSchema } from '@/components/values/adapters/paramSchema';
import { Drawer, DrawerSection, type DrawerSections } from '@/features/projectWindow/DrawerForms';
import { LibraryList, type LibraryItem } from '@/features/projectWindow/LibraryList';
import { bindStudioCommand } from '@/features/projectWindow/studioCommands';
import {
  AnalysisDocumentSchema, type AnalysisDocument, type InputBinding, type JsonPrimitive, type JsonValue, type OperationElement,
  type ParamSchema, type PresentationPosition, type StudioAnalyzeChange, type StudioProjectSnapshot,
  type StudioSavedTrainKey, type StudioSavedTrainRuns, type SubsetTerms, type TrialMeanSelection,
} from '@/generated/studioContracts';
import { authoredPointer } from './authoredEdits';
import { ManagedDocumentCanvas, type CanvasNode } from './ManagedDocumentCanvas';
import {
  choosePick, useNamedSelection, usePick, type ElementSelectionStore,
} from '@/features/projectWindow/elementSelection';
import { shownName, type PlanStepView } from '@/features/pipeline/planView';
import { heldElements } from './analysisElements';
import { keyUnavailable } from './projectMemberName';
import { AnalyzeElementInputs } from './AnalyzeReferences';
import { SelectionRule } from './SelectionRule';
import { changedSelections, selectionProblems, stepSelections, withSubsetTerms } from './selectionEdits';
import { documentBinding, InputEditor } from './InputEditor';
import { useAuthoringCatalog } from '@/api/studioAuthoring';
import { ProjectEditConflictError, type ProjectEditController } from './projectEditController';
import type { ChangeMark } from '@/components/values/fieldModel';
import { savedRunId, stepSourceRows } from './projectWindowPlan';
import { placeElement, type Reveal } from './projectWindowProblems';
import { ChangeRow, ProblemRow } from '@/features/projectWindow/StudioProblems';
import { SavedFigureResult } from './SavedLightProducts';
import { ElementRequests, requestMember } from './ElementRequests';
import { generatedName, keyOfLabel } from './generatedName';
import type { GestureFailures } from './refusals';

type Source = OperationElement['inputs'][string];
type RefusalPlace = 'dialog' | readonly [string, string] | { readonly key: string };

const operationCarry = carryKind<LibraryItem>('operation');
const figureItem = 'velocity_summary';
const summaryTemplate = 'feedbax.VelocitySummary';

export function useManagedAnalyze({
  editor, key, projection, wholeStep, chosen, focusedView, reveal, savePosition, selection, sections, steps, failures, selectStep, mean,
}: {
  /** The clock's Mean, which every figure draws while it is picked. */
  mean?: TrialMeanSelection;
  editor: ProjectEditController;
  /** The steps an edit of the shown document is made for, when other steps read it too; the step alone otherwise. */
  chosen?: ReadonlySet<string>;
  /** Where the step's refused gestures land. */
  failures: GestureFailures;
  selection: ElementSelectionStore;
  sections: DrawerSections;
  /** The project's steps, whose outputs the step's input roles can be bound to. */
  steps: readonly PlanStepView[];
  key?: StudioSavedTrainKey;
  projection?: StudioSavedTrainRuns;
  wholeStep: boolean;
  focusedView?: string | null;
  reveal?: Reveal;
  /** Selects a step, which a blocked input links to. */
  selectStep(step: string): void;
  savePosition(document: string, node: string, position: PresentationPosition): Promise<unknown>;
}) {
  const { saved, draft, outside } = useSyncExternalStore(editor.subscribe, editor.getSnapshot);
  const step = key ? draft.steps?.[key.step] : undefined;
  const base = step?.kind === 'analyze' && step.base;
  const document = stepDocument(base);
  const authored = document ? documentPayload(draft.documents, saved.composed, document) : undefined;
  const parsed = document ? AnalysisDocumentSchema.safeParse(authored) : undefined;
  const analysis = parsed?.success ? parsed.data : undefined;
  const id = key ? savedRunId(key) : undefined;
  const scope = JSON.stringify([saved.project_id, id, document]);
  const liveScope = useRef(scope);
  liveScope.current = scope;
  const element = useNamedSelection(selection, document, reveal, placeElement(reveal?.place));
  const [dialog, setDialog] = useState<{ scope: string; mode: 'add'; point?: PresentationPosition }>();
  const [query, setQuery] = useState('');
  /** How many of its edits are waiting or in flight. */
  const [edits, setEdits] = useState(0);
  const busy = edits > 0;
  /** The refused gesture's own place, when it has one: the open dialog, a role field as `[element, role]`, or a key. */
  const [refusal, setRefusal] = useState<{ scope: string; at: RefusalPlace; reason: string }>();
  /** The operation or figure whose label (on its canvas card) or key (in the inspector) is being edited. */
  const [naming, setNaming] = useState<{ scope: string; element: string; field: 'label' | 'key' }>();
  const cursor = useRef<{ scope: string; point: PresentationPosition }>();
  /** What the operation catalog depends on. The document's own contents shape only its Changed marks, so an edit
   * refetches the review but keeps editing enabled while it does. */
  const catalogShape = step?.kind === 'analyze' ? JSON.stringify([step.base, step.sweep, step.deltas, step.inputs?.states])
    : undefined;
  const contents = document && JSON.stringify(baseChain(saved.project.documents, document)
    .map((name) => saved.project.documents?.[name]));
  const authoring = useQuery({
    queryKey: ['studio-analyze', saved.project_id, id, catalogShape, contents],
    queryFn: async ({ signal }) => ({
      shape: catalogShape,
      review: await readStudioAnalyze(saved.project_id, { revision: saved.revision, key: key! }, signal),
    }),
    enabled: !!document && !!key && key.variant === projection?.window_map?.variants[0],
    placeholderData: keepPreviousData,
    staleTime: 30_000,
  });
  const catalog = authoring.data?.review;
  /** Whether its commands apply; a gesture made while the workspace is unsettled waits for it. */
  const bindApplies = !!step && wholeStep && !outside;
  const catalogReady = !!catalog && catalog.document === document && authoring.data?.shape === catalogShape;
  /** A catalog still being read may yet allow the edit; the workspace waits for it, so a command made then is held. */
  const editApplies = bindApplies && !!analysis && !!document && (authoring.isFetching || catalogReady);
  const canBind = bindApplies;
  const canEdit = bindApplies && !!analysis && !!document && catalogReady;
  const [inspected, setInspected] = useState<{ scope: string; id: string }>();
  /** The request of the selected element shown in the inspector, when the element maps over labels. */
  const [chosenRequest, setChosenRequest] = useState<{ scope: string; element: string; member: string }>();
  const opened = dialog?.scope === scope ? dialog : undefined;
  const operations = analysis?.operations ?? {};
  const figures = analysis?.views ?? {};
  const definitions = new Map(catalog?.operations.map((operation) => [operation.key, operation]));
  const selected = element.selectedId;
  const operation = selected ? operations[selected] : undefined;
  const figure = selected ? figures[selected] : undefined;
  const inlineFigure = figure && 'data' in figure.figure;
  const resolvedFigures = new Map(projection?.window_map?.analysis_figures
    ?.filter(({ key }) => savedRunId(key) === id).map((view) => [view.name, view]));
  function figureDefinition(name: string, source: NonNullable<AnalysisDocument['views']>[string]['figure']) {
    return resolvedFigures.get(name) ?? ('data' in source ? source : undefined);
  }
  const selectedFigure = selected && figure ? figureDefinition(selected, figure.figure) : undefined;
  const definition = operation && definitions.get(operation.operation);
  const declarations = analysis?.inputs ?? {};
  const roles = selectedFigure ? selectedFigure.data.map((name) => ({ name, kind: 'dataset' as const }))
    : Object.entries(definition?.inputs ?? {}).map(([name, port]) => ({ name, kind: port.kind ?? 'dataset' }));
  const outputs = Object.entries(operations).map(([name, value]) => ({
    element: name, name: shownName(name, value), products: Object.keys(definitions.get(value.operation)?.outputs ?? {}),
  }));
  /** The Analysis document a card reads, when it reads one of its own. */
  function cardAnalysis(card: string) {
    const read = draft.steps?.[card];
    const source = read?.kind === 'analyze' ? stepDocument(read.base) : undefined;
    const parsed = source ? AnalysisDocumentSchema.safeParse(documentPayload(draft.documents, saved.composed, source)) : undefined;
    return parsed?.success ? parsed.data : undefined;
  }
  /** The other Analyze cards' operation outputs, which a dataset role can be bound to. */
  const cardOutputs = steps.filter((other) => other.kind === 'analyze' && other.id !== key?.step).flatMap((card) =>
    Object.entries(cardAnalysis(card.id)?.operations ?? {}).flatMap(([element, value]) =>
      Object.keys(definitions.get(value.operation)?.outputs ?? {}).map((product) => ({
        step: card.id, element, product, name: `${card.label}/${shownName(element, value)}.${product}`,
      }))));
  /** A role bound to another card's operation output is shown by the card and the output. */
  function boundName(role: string) {
    const bound = step?.inputs?.[role];
    if (!bound || Array.isArray(bound) || typeof bound !== 'object' || !('element' in bound) || !('step' in bound)) return role;
    const { step: card, element: source, product } = bound as { step: string; element?: string | null; product?: string | null };
    if (!source || !product) return role;
    const shownCard = steps.find(({ id }) => id === card)?.label ?? card;
    return `${shownCard}/${shownName(source, cardAnalysis(card)?.operations?.[source])}.${product}`;
  }
  const selectedInput = selected ? declarations[selected] : undefined;
  const authoringCatalog = useAuthoringCatalog().data;
  const inputRoles = authoringCatalog?.analyze_inputs;
  const figureEntries = authoringCatalog?.registries.figure_templates?.entries;
  const figureTemplates = useMemo(() => figureEntries?.filter((entry) => 'settings_schema' in entry)
    .filter(({ key }) => key !== summaryTemplate) ?? [], [figureEntries]);
  const inputRole = selectedInput && inputRoles?.[selectedInput.kind];
  const armed = usePick(selection);
  const canvasPick = useMemo(() => armed && document ? {
    accepts: (id: string) => armed.accepts({ kind: 'element', address: { document, element: { kind: 'named', id } } }),
    choose: (id: string) => { choosePick(selection, { kind: 'element', address: { document, element: { kind: 'named', id } } }); },
  } : null, [armed, document, selection]);
  /** The plan's Problems, with those the card's own read adds, such as a states input that reads no Evaluate runs. */
  const cardProblems = useMemo(() => {
    const planned = projection?.window_map?.problems ?? [];
    const seen = new Set(planned.map((problem) => JSON.stringify(problem)));
    return [...planned, ...(catalog?.problems ?? []).filter((problem) => !seen.has(JSON.stringify(problem)))];
  }, [projection?.window_map?.problems, catalog?.problems]);
  const diagnostics = cardProblems.filter((problem) => id &&
    problem.variant === key?.variant && problem.step === key?.step && problem.row === key?.row);
  const planned = new Map(projection?.window_map?.analysis_elements?.filter(({ key }) => savedRunId(key) === id)
    .map((item) => [item.name, item]));
  const catalogState: StudioRegionState = authoring.error
    ? { kind: 'failed', issue: studioRegionIssue('operations', authoring.error), retry: () => { void authoring.refetch(); } }
    : authoring.isLoading ? { kind: 'loading' } : { kind: 'result', missing: [] };

  function open(mode: 'add', point?: PresentationPosition) {
    if (!canEdit) return;
    setRefusal(undefined);
    setQuery('');
    setDialog({ scope, mode, point });
  }

  /** Runs an edit after the edits before it have saved; the controls stay editable meanwhile. */
  async function perform(work: (current: StudioProjectSnapshot) => Promise<void>, allowed = canEdit, at?: RefusalPlace) {
    if (!allowed) return;
    setEdits((count) => count + 1);
    failures.clear();
    setRefusal(undefined);
    try { await editor.command(work); }
    catch (cause) { failures.fail(cause, at && ((reason) => setRefusal({ scope, at, reason }))); }
    finally { setEdits((count) => count - 1); }
  }

  /** Store `change` of the shown document for the chosen steps, as the service computes it, as one Undo entry. */
  async function save(current: StudioProjectSnapshot, label: string, change: StudioAnalyzeChange) {
    if (!key || liveScope.current !== scope) throw new ProjectEditConflictError();
    const reviewed = await readStudioAnalyze(current.project_id, {
      revision: current.revision, key, change, chosen_steps: [...chosen ?? []],
    });
    if (reviewed.document !== document) throw new ProjectEditConflictError();
    if (!reviewed.files?.length && !reviewed.created?.length) return;
    const stored = editor.storeEdit(label, { files: reviewed.files ?? [], created: reviewed.created ?? [] }, {
      reviewedRevision: current.revision,
    });
    await editor.flush();
    const settled = await stored;
    if (settled.status === 'refused') throw settled.error;
  }
  const set = (path: readonly string[], value: JsonValue): StudioAnalyzeChange => ({ kind: 'edit', edits: [
    ...path.slice(0, -1).map((_, index) => ({
      op: 'ensure' as const, path: authoredPointer(path.slice(0, index + 1)), collection: 'object' as const,
    })),
    { op: 'replace', path: authoredPointer(path), value },
  ] });
  const changedPaths = new Set(catalog?.document === document ? catalog?.changed ?? [] : []);
  function changedAt(path: readonly string[]): ChangeMark | undefined {
    const pointer = authoredPointer(path);
    const base = baseOf(saved.project.documents, document);
    if (!changedPaths.has(pointer) || base === undefined) return undefined;
    return { from: { kind: 'base', document: base }, reset: canEdit ? async () => {
      try { await editor.command((current) => save(current, studioName('reset'), { kind: 'reset', paths: [pointer] })); return undefined; }
      catch (cause) { return cause instanceof Error ? cause.message : String(cause); }
    } : undefined };
  }
  /** Places an element named as the Library names what it is made from, numbered while another element shows that name. */
  function add(kind: 'operations' | 'views', named: string, value: Readonly<Record<string, JsonValue>>, point?: PresentationPosition) {
    void perform(async (current) => {
      const held = heldElements(editor.getSnapshot().draft.documents, current.composed);
      const shownNames = new Set(held.values());
      const label = generatedName(named, (label) => shownNames.has(label), ' ');
      const name = generatedName(keyOfLabel(label) || kind,
        (name) => name === 'states' || name in declarations || held.has(name));
      await save(current, studioName(kind === 'operations' ? 'addOperation' : 'addFigure'), set([kind, name], { ...value, label }));
      setDialog(undefined);
      if (liveScope.current === scope) {
        setInspected(undefined);
        element.onSelect(name);
      }
      // A narrowed edit leaves the step reading the document it created.
      const now = key && editor.getSnapshot().draft.steps?.[key.step]?.base;
      if (point) await savePosition(now && 'document' in now ? now.document : document!, name, point);
    }, canEdit, opened ? 'dialog' : undefined);
  }
  const addFigure = (point?: PresentationPosition) => {
    if (catalog) add('views', humanizeLabel(figureItem), { figure: catalog.figure, inputs: {} }, point);
  };
  const addOperation = (operation: string, point?: PresentationPosition) => {
    const entry = definitions.get(operation);
    if (entry) add('operations', entry.label, { operation, params: {}, inputs: {} }, point);
  };
  const addCommand = bindStudioCommand('addComponent', () => {
    if (focusedView === 'figures') addFigure();
    else open('add', cursor.current?.scope === scope ? cursor.current.point : undefined);
  }, { name: focusedView === 'figures' ? 'addFigure' : 'addOperation', enabled: editApplies });
  function wire(element: string, role: string, source: Source) {
    if (!key) return;
    void perform((current) => save(current, humanizeLabel(role), { kind: 'wire', element, role, source }), canEdit, [element, role]);
  }
  function bind(role: string, binding: InputBinding) {
    if (!key) return;
    void perform(async (current) => {
      if (liveScope.current !== scope) throw new ProjectEditConflictError();
      const inputs = current.project.steps?.[key.step]?.inputs;
      editor.editFiles(humanizeLabel(role), [{ address: { kind: 'step', id: key.step }, patches: [{
        op: inputs && role in inputs ? 'replace' : 'add', path: authoredPointer(['inputs', role]), value: binding as JsonValue,
      }] }], { reviewedRevision: current.revision });
      await editor.flush();
    }, canBind, [role, role]);
  }

  const removeCommand = bindStudioCommand('modelRemove', () => {
    if (selected) void perform((current) => save(current, studioName('modelRemove'), { kind: 'remove', element: selected }));
  }, { enabled: editApplies && !!(operation || figure) });

  const shown = (name: string) => shownName(name, operations[name] ?? figures[name]);
  const nameApplies = bindApplies && !!analysis && !!document;
  const canName = nameApplies;
  const namingNow = naming?.scope === scope ? naming : undefined;
  function renameElement(name: string, text: string) {
    const label = text === name ? null : text;
    if (!document || label === ((operations[name] ?? figures[name])?.label ?? null)) return;
    void perform(async () => {
      await editor.memberCommand(studioAccessibleName('rename', shown(name)),
        { kind: 'rename', family: 'elements', document, member: name, label }, editor.getSnapshot().saved);
    }, canName);
  }
  function changeElementKey(name: string, key: string) {
    if (!document) return false;
    void perform(async () => {
      await editor.memberCommand(studioAccessibleName('changeKey', shown(name)),
        { kind: 'change_key', family: 'elements', document, member: name, key }, editor.getSnapshot().saved);
      if (liveScope.current === scope) element.onSelect(key);
    }, canName, { key: name });
    return true;
  }
  const refused = !!key && !!projection?.window_map?.regroup_refused?.includes(key.step);
  const elementKeys = [...Object.keys(operations), ...Object.keys(figures)];
  const chosenElements = element.selectedIds.filter((name) => elementKeys.includes(name));
  /** Split: the chosen operations and figures move to a new card beside this one, with the edge between them. */
  function splitCommand(elements: readonly string[]) {
    return bindStudioCommand('split', () => {
      if (!key) return;
      void perform(async () => {
        await editor.memberCommand(studioAccessibleName('split', key.step),
          { kind: 'split', member: key.step, elements: [...elements] }, editor.getSnapshot().saved);
        if (liveScope.current === scope) selectElement(null);
      }, canName);
    }, { subject: key && shownName(key.step, draft.steps?.[key.step]), enabled: nameApplies && !!key && !refused && elements.length > 0 && elements.length < elementKeys.length });
  }
  /** The commands on an operation or figure: its context menu, and the palette's for the selected one. */
  const elementCommands = (name: string) => [
    bindStudioCommand('renameElement', () => setNaming({ scope, element: name, field: 'label' }),
      { subject: shown(name), instance: name, enabled: nameApplies }),
    bindStudioCommand('changeKey', () => {
      selectElement(name);
      setNaming({ scope, element: name, field: 'key' });
    }, { subject: shown(name), instance: name, enabled: nameApplies }),
    splitCommand(chosenElements.includes(name) ? chosenElements : [name]),
  ];
  const problemOf = (name: string) => diagnostics.find((item) => placeElement(item.place) === name)?.detail;
  const items: CanvasNode[] = [
    ...Object.keys(declarations).map((name) => ({ id: name, selectionId: name, name: boundName(name), glyph: <Database /> })),
    ...[...Object.keys(operations).map((name) => [name, <Sigma />] as const),
      ...Object.keys(figures).map((name) => [name, <ChartLine />] as const)].map(([name, glyph]): CanvasNode => ({
      id: name, selectionId: name, name: shown(name), glyph, problem: problemOf(name), status: planned.get(name)?.status,
      menu: () => elementCommands(name),
      naming: namingNow?.field === 'label' && namingNow.element === name ? {
        commit: (text) => renameElement(name, text), end: () => setNaming(undefined),
      } : undefined,
    })),
  ];
  const canvas = useRef<CanvasViewport>(null);
  /** A based document is laid out as its base is, where it does not place an element itself. */
  const positions = document ? Object.assign({}, ...[...baseChain(draft.documents, document)].reverse()
    .map((name) => saved.presentation?.documents?.[name]?.positions)) : undefined;
  const viewport = canvasViewportCommands(() => canvas.current, !!document);
  const connections = [...Object.entries(operations), ...Object.entries(figures)].flatMap(([target, value]) =>
    Object.entries(value.inputs).map(([role, source]) => {
      const from = 'input' in source ? source.input : source.element;
      const product = 'input' in source ? source.input : source.product;
      const label = product === role ? role : product + ' → ' + role;
      return { id: JSON.stringify([target, role]), source: from, target, label: label === from ? undefined : label };
    }));
  const landed = refusal?.scope === scope ? refusal : undefined;
  const roleRefusal = (element: string, role: string) =>
    landed && Array.isArray(landed.at) && landed.at[0] === element && landed.at[1] === role ? landed.reason : undefined;
  const selectedCommands = selected && (operation || figure) ? elementCommands(selected)
    : chosenElements.length ? [splitCommand(chosenElements)] : [];
  const selectedProblems = diagnostics.filter((item) => selected && placeElement(item.place) === selected);
  const selectedElement = selected ? planned.get(selected) : undefined;
  const mapped = selectedElement && selectedElement.requests.length > 1;
  const chosenMember = chosenRequest?.scope === scope && chosenRequest.element === selected ? chosenRequest.member : undefined;
  /** The request whose status and result the inspector shows: the element's own, or the chosen one of a mapped element. */
  const planOf = (name: string, member: string | undefined) => {
    const element = planned.get(name);
    return element && element.requests.length > 1
      ? element.requests.find((request) => requestMember(request) === member)
      : element?.requests[0];
  };
  const selectedPlan = selected ? planOf(selected, chosenMember) : undefined;
  /** The figure the Figures view draws: the selected one, else the first, so a window reopened there draws it. */
  const shownFigure = figure ? selected : Object.keys(figures)[0];
  const shownPlan = shownFigure && planOf(shownFigure, shownFigure === selected ? chosenMember : undefined);
  const shownStatus = selectedPlan ?? selectedElement;
  function writeParameter(path: readonly string[], value: JsonValue) {
    if (!operation || !selected) return;
    const [name, ...rest] = path;
    const next = setIn(operation.params?.[name], rest, value);
    void perform((current) => save(current, humanizeLabel(name), set(['operations', selected, 'params', name], next)));
  }
  const inspector = selected && (operation || figure || selectedInput) && <Inspector name={shown(selected)}
    type={operation ? operation.operation : figure ? studioName('figures') : humanizeLabel(selectedInput!.kind ?? 'dataset')}>
    {(operation || figure) && <KeyField value={selected} name={shown(selected)}
      editing={namingNow?.field === 'key' && namingNow.element === selected}
      onEditingChange={(editing) => setNaming(editing ? { scope, element: selected, field: 'key' } : undefined)}
      onCommit={(key) => changeElementKey(selected, key)}
      unavailable={(key) => keyUnavailable(key, (key) => heldElements(draft.documents, saved.composed).has(key))}
      refusal={landed && typeof landed.at === 'object' && 'key' in landed.at && landed.at.key === selected ? landed.reason : undefined} />}
    {selected && selectedInput && key && inputRole && <InputEditor label={humanizeLabel(selected)} role={inputRole}
      binding={step?.inputs?.[selected]} self={key.step} project={saved.project} steps={steps} picking={selection}
      refusal={roleRefusal(selected, selected)} outputs={cardOutputs} onReveal={selectStep}
      selection={(index, reference) => {
        const binding = step?.inputs?.[selected];
        const listed = Array.isArray(binding);
        const subset = steps.find(({ id }) => id === key.step)?.inputs.find(({ id }) => id === (listed ? `${selected}.${index}` : selected))?.subset;
        const source = steps.find(({ id }) => id === reference.step);
        const where = reference.where ?? {};
        const problems = selectionProblems(cardProblems, key.step, key.variant, selected, index, listed);
        const write = (terms: SubsetTerms) => {
          const next = withSubsetTerms(reference, terms);
          bind(selected, listed ? changedSelections(stepSelections(binding)!, { kind: 'replace', index, reference: next }) : next);
        };
        return {
          problems,
          runs: source && projection && !('subset' in where) && <SelectionRule
            scope={{ project: saved.project_id, revision: projection.revision, variant: key.variant }} source={source}
            terms={where} planned={subset} picking={selection}
            disabled={!canBind} problem={problems.rule}
            pick={{ rows: stepSourceRows(projection, source.id, key.variant) }} onCommit={write} />,
        };
      }}
      onChange={canBind ? (gesture) => {
        if (gesture.kind === 'select') bind(selected, gesture.binding);
        else if (gesture.kind === 'document') bind(selected, documentBinding(gesture.role, gesture.document));
      } : undefined} />}
    {selected && document && (operation ?? figure) && <AnalyzeElementInputs document={document} element={selected} roles={roles}
      inputs={(operation ?? figure)!.inputs} declarations={declarations} outputs={outputs} picking={selection}
      refusal={(role) => roleRefusal(selected, role)} onWire={canEdit ? (role, source) => wire(selected, role, source) : undefined} />}
    {selectedProblems.map((problem) => <ProblemRow key={problem.path + problem.detail} problem={problem} />)}
    {shownStatus && <p role="status">{studioRunStatus(shownStatus.status)}
      {shownStatus.reason && <span className="block text-shell-muted">{shownStatus.reason}</span>}
    </p>}
    {mapped && selected && <ElementRequests requests={selectedElement.requests} selected={chosenMember}
      onSelect={(member) => setChosenRequest(member === undefined ? undefined : { scope, element: selected, member })} />}
    {selectedPlan?.changed && <ChangeRow change={selectedPlan.changed} />}
    {operation && definition && key && <div className="grid gap-1">
      <SettingsForm key={selected} schema={definition.parameter_json_schema} value={operation.params ?? {}}
        projected={(path) => {
          const schema = path.length === 1 && definition.param_schema.find(({ name }) => name === path[0]);
          const options = schema && catalog?.variables[definition.key]?.[schema.name];
          return schema && options ? fromParamSchema(schema, operation.params?.[schema.name], canEdit
            ? (value) => writeParameter(path, value) : undefined, { options }) : undefined;
        }}
        changed={(path) => changedAt(['operations', selected, 'params', ...path])}
        write={canEdit ? writeParameter : undefined} />
    </div>}
    {figure && selected && selectedFigure && <label className="grid gap-1">
      {studioName('caption')}
      <TextField key={JSON.stringify([saved.revision, selected, selectedFigure.caption])}
        defaultValue={selectedFigure.caption ?? ''} disabled={!canEdit || !inlineFigure}
        onBlur={({ currentTarget }) => {
          const caption = currentTarget.value;
          if (inlineFigure && caption !== (selectedFigure.caption ?? '')) {
            void perform((current) => save(current, studioName('caption'), set(['views', selected, 'figure', 'caption'], caption)));
          }
        }}
        onKeyDown={(event) => { if (event.key === 'Enter') event.currentTarget.blur(); }} />
    </label>}
  </Inspector>;
  const library = useMemo((): LibraryItem[] => catalog ? [
    ...catalog.operations.map((entry) => ({
      id: entry.key, name: entry.label, group: studioName('operations'), glyph: <Sigma />, terms: entry.key,
      about: {
        description: entry.description, inputs: Object.keys(entry.inputs), outputs: Object.keys(entry.outputs),
        parameters: entry.param_schema.map(({ name }) => name),
      },
    })),
    { id: figureItem, name: humanizeLabel(figureItem), group: studioName('figures'), glyph: <ChartLine /> },
    ...figureTemplates.map((entry) => ({
      id: entry.key, name: entry.label, group: studioName('figures'), glyph: <ChartLine />, terms: entry.key,
    })),
  ] : [], [catalog, figureTemplates]);
  function selectElement(id: string | null, extend = false) {
    setInspected(undefined);
    element.onSelect(id, extend);
  }
  const shownItem = inspected?.scope === scope ? library.find((item) => item.id === inspected.id) : undefined;
  function place(id: string, point?: PresentationPosition) {
    if (id === figureItem) addFigure(point);
    else if (definitions.has(id)) addOperation(id, point);
    else {
      const template = figureTemplates.find(({ key }) => key === id);
      if (template) add('views', template.label, { figure: { template: id }, inputs: {} }, point);
    }
  }
  const libraryInspector = shownItem && <Inspector name={shownItem.name} type={shownItem.group}>
    {definitions.get(shownItem.id)?.description && <p className="text-shell-muted">{definitions.get(shownItem.id)!.description}</p>}
    {canEdit && <Button className="justify-self-start" onClick={() => place(shownItem.id)}>
      {studioAccessibleName('add', shownItem.name)}
    </Button>}
  </Inspector>;
  const drawer = <Drawer>
    <LibraryList sections={sections} section="analysis.library" name={studioName('library')} searchName="searchLibrary"
      state={catalogState} items={library} selected={shownItem?.id} carried={canEdit ? operationCarry : undefined}
      onInspect={(id) => setInspected({ scope, id })}
      onPlace={canEdit ? (id) => place(id) : undefined} />
  </Drawer>;
  const figureList = <ul className="space-y-2 p-2">{Object.entries(figures).map(([name, view]) => {
    const caption = figureDefinition(name, view.figure)?.caption;
    const result = planned.get(name);
    return <li key={name} className="flex items-center gap-2">
      <Button pressed={shownFigure === name} onClick={() => selectElement(name)}>{shown(name)}</Button>
      {caption && <span>{caption}</span>}
      {result && <span>{studioRunStatus(result.status)}</span>}
    </li>;
  })}</ul>;
  return {
    /** Whether no edit of its own is in flight and its catalog is read. */
    settled: !busy && !authoring.isFetching,
    selectedFigure: figure ? selected : undefined,
    analysis: {
      body: <CanvasStrip viewport={viewport}><div className="min-h-0 flex-1">
        <ManagedDocumentCanvas key={scope} selectedIds={element.selectedIds} onSelect={selectElement} items={items} connections={connections} viewportRef={canvas}
          positions={positions} pick={canvasPick}
          openAt={canEdit ? (point) => open('add', point) : undefined} placing={!!opened?.point}
          drop={canEdit ? { kind: operationCarry, place } : undefined}
          rememberPoint={(point) => { cursor.current = point ? { scope, point } : undefined; }} />
      </div></CanvasStrip>,
      drawer, inspector: libraryInspector || inspector,
    },
    figures: { body: <>{figureList}
      {shownFigure && shownPlan && <SavedFigureResult key={JSON.stringify([scope, shownFigure])}
        history={shownPlan} name={shownFigure} project={saved.project_id} revision={saved.revision} mean={mean} />}
    </>, drawer, inspector: libraryInspector || (figure ? inspector : undefined) },
    commands: document && focusedView === 'analysis' ? [addCommand, removeCommand, ...selectedCommands, ...viewport]
      : document && focusedView === 'figures' ? [addCommand, removeCommand, ...selectedCommands] : [],
    overlay: opened && <Dialog ariaLabel={studioName('addOperation')}
      manageFocus panelClassName="grid w-[min(26rem,90vw)] gap-3" onClose={() => setDialog(undefined)}>
      {landed?.at === 'dialog' && <p role="alert">{landed.reason}</p>}
      <TextField data-dialog-autofocus aria-label={studioName('operations')} value={query} onValueChange={setQuery} />
      {catalog?.operations.filter(({ label }) => label.toLowerCase().includes(query.toLowerCase()))
        .map((entry) => <Button key={entry.key} variant="ghost" align="start" disabled={!canEdit}
          onClick={() => addOperation(entry.key, opened.point)}>
          {entry.label}
        </Button>)}
    </Dialog>,
  };
}
