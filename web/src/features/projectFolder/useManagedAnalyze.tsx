import { useQuery } from '@tanstack/react-query';
import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { BarChart3, Database, Sigma } from 'lucide-react';
import { appTip } from '@window-shell/browser/tooltip';
import { apiErrorMessage } from '@/api/request';
import { studioRegionIssue, type StudioRegionState } from '@/components/activity/StudioRegion';
import { readStudioAnalyze } from '@/api/studioAnalyze';
import { studioName, studioRunStatus } from '@/components/activity/studioVocabulary';
import { CanvasStrip } from '@/components/canvas/CanvasStrip';
import { canvasViewportCommands, type CanvasViewport } from '@/components/canvas/StudioCanvas';
import { Button } from '@/components/ui/Button';
import { Inspector } from '@/components/ui/Inspector';
import { StatusMark } from '@/components/ui/StatusMark';
import { Tree, type TreeNode } from '@/components/ui/Tree';
import { ValueField } from '@/components/values/ValueField';
import { rawField } from '@/components/values/adapters/raw';
import { Drawer, DrawerSection, type DrawerSections } from '@/features/projectWindow/DrawerForms';
import { LibraryList, type LibraryItem } from '@/features/projectWindow/LibraryList';
import { Dialog } from '@/components/ui/Dialog';
import { Select } from '@/components/ui/Select';
import { humanizeLabel } from '@/components/values/humanizeLabel';
import { SettingsForm } from '@/components/values/SettingsForm';
import { setIn } from '@/components/values/adapters/jsonSchema';
import { fromParamSchema } from '@/components/values/adapters/paramSchema';
import { bindStudioCommand } from '@/features/projectWindow/studioCommands';
import {
  AnalysisDocumentSchema, type AnalysisDocument, type JsonPrimitive, type JsonValue, type OperationElement,
  type ParamSchema, type PresentationPosition, type StudioProjectFilePatch,
  type StudioSavedTrainKey, type StudioSavedTrainRuns,
} from '@/generated/studioContracts';
import { authoredPointer } from './authoredEdits';
import { ManagedDocumentCanvas, type CanvasNode } from './ManagedDocumentCanvas';
import { useNamedSelection, type ElementSelectionStore } from '@/features/projectWindow/elementSelection';
import { ProjectEditConflictError, type ProjectEditController } from './projectEditController';
import { savedRunId } from './projectWindowPlan';
import { problemReason, type ProblemReveal } from './projectWindowProblems';
import { SavedFigureResult } from './SavedLightProducts';

type Source = OperationElement['inputs'][string];

export function useManagedAnalyze({
  editor, key, projection, wholeStep, focusedView, reveal, savePosition, selection, sections,
}: {
  editor: ProjectEditController;
  sections: DrawerSections;
  selection: ElementSelectionStore;
  key?: StudioSavedTrainKey;
  projection?: StudioSavedTrainRuns;
  wholeStep: boolean;
  focusedView?: string | null;
  reveal?: ProblemReveal;
  savePosition(document: string, node: string, position: PresentationPosition): Promise<unknown>;
}) {
  const { saved, draft, pending, outside } = useSyncExternalStore(editor.subscribe, editor.getSnapshot);
  const step = key ? draft.steps?.[key.step] : undefined;
  const base = step?.kind === 'analyze' && step.base;
  const document = base && 'document' in base && !base.pointer ? base.document : undefined;
  const authored = document ? draft.documents?.[document]?.payload : undefined;
  const parsed = document ? AnalysisDocumentSchema.safeParse(authored) : undefined;
  const analysis = parsed?.success ? parsed.data : undefined;
  const id = key ? savedRunId(key) : undefined;
  const scope = JSON.stringify([saved.project_id, id, document]);
  const liveScope = useRef(scope);
  liveScope.current = scope;
  const element = useNamedSelection(selection, document, reveal, reveal?.target.element?.kind === 'analysisElement'
    ? reveal.target.element.id : undefined);
  const [dialog, setDialog] = useState<{ scope: string; mode: 'wire' }>();
  const [library, setLibrary] = useState<{ scope: string; key: string }>();
  const [libraryRequest, setLibraryRequest] = useState(0);
  const placement = useRef<{ scope: string; point: PresentationPosition }>();
  const [sourceIndex, setSourceIndex] = useState('');
  const [role, setRole] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<{ scope: string; message: string }>();
  const inFlight = useRef(false);
  const cursor = useRef<{ scope: string; point: PresentationPosition }>();
  const authoring = useQuery({
    queryKey: ['studio-analyze', saved.project_id, saved.revision, id],
    queryFn: ({ signal }) => readStudioAnalyze(saved.project_id, { revision: saved.revision, key: key! }, signal),
    enabled: !!document && !!key && key.row === 'default' && key.variant === projection?.window_map?.variants[0],
    staleTime: 30_000,
  });
  const catalog = authoring.data;
  const canEdit = !!analysis && !!document && !!key && !!catalog && catalog.document === document &&
    catalog.revision === saved.revision && wholeStep && !pending && !outside && !busy;
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
  const roles = selectedFigure?.data ?? Object.keys(definition?.inputs ?? {});
  const sources: { label: string; source: Source; schemas: string[] }[] = [
    { label: 'states', source: { input: 'states' }, schemas: catalog?.states.schema_ids ?? [] },
    ...Object.entries(operations).flatMap(([name, value]) => Object.entries(definitions.get(value.operation)?.outputs ?? {})
      .map(([product, port]) => ({ label: name + '.' + product, source: { element: name, product }, schemas: port.schema_ids }))),
  ];
  const diagnostics = (projection?.window_map?.problems ?? []).filter((problem) => id &&
    problem.variant === key?.variant && problem.step === key?.step && problem.row === key?.row);
  const planned = new Map(projection?.window_map?.analysis_elements?.filter(({ key }) => savedRunId(key) === id)
    .map((item) => [item.name, item]));
  const issue = error?.scope === scope ? error.message : undefined;
  const catalogState: StudioRegionState = authoring.error
    ? { kind: 'failed', issue: studioRegionIssue('operations', authoring.error), retry: () => { void authoring.refetch(); } }
    : authoring.isLoading ? { kind: 'loading' } : { kind: 'result', missing: [] };

  function open(mode: 'wire') {
    if (!canEdit) return;
    setError(undefined);
    setSourceIndex('');
    setRole(roles.length === 1 ? roles[0] : '');
    setDialog({ scope, mode });
  }
  function openLibrary(point?: PresentationPosition) {
    placement.current = point ? { scope, point } : undefined;
    setLibraryRequest((count) => count + 1);
  }

  async function perform(work: () => Promise<void>) {
    if (!canEdit || inFlight.current) return;
    inFlight.current = true;
    setBusy(true);
    setError(undefined);
    try { await work(); }
    catch (cause) { setError({ scope, message: apiErrorMessage(cause) }); }
    finally {
      inFlight.current = false;
      setBusy(false);
    }
  }

  function file(path: string[], value: JsonValue): StudioProjectFilePatch {
    let parent = authored;
    for (const segment of path.slice(0, -1)) {
      parent = parent && typeof parent === 'object' && !Array.isArray(parent) ? parent[segment] : undefined;
    }
    const leaf = path.at(-1)!;
    const existing = parent && typeof parent === 'object' && Object.prototype.hasOwnProperty.call(parent, leaf);
    return {
      address: { kind: 'document', id: document! },
      patches: [{ op: existing ? 'replace' : 'add', path: authoredPointer(['payload', ...path]), value }],
    };
  }
  async function save(label: string, files: StudioProjectFilePatch[]) {
    if (liveScope.current !== scope || editor.getSnapshot().saved.revision !== saved.revision) {
      throw new ProjectEditConflictError();
    }
    editor.editFiles(label, files, { reviewedRevision: saved.revision });
    await editor.flush();
  }
  function add(kind: 'operations' | 'views', stem: string, value: JsonValue, point?: PresentationPosition) {
    void perform(async () => {
      let name = stem;
      for (let number = 2; name === 'states' || name in operations || name in figures; number += 1) name = stem + '-' + number;
      const label = studioName(kind === 'operations' ? 'addOperation' : 'addFigure');
      const collection = analysis?.[kind];
      await save(label, [collection ? file([kind, name], value) : file([kind], { [name]: value })]);
      if (liveScope.current === scope) {
        setLibrary(undefined);
        element.onSelect(name);
      }
      if (point) await savePosition(document!, name, point);
    });
  }
  const addFigure = () => {
    if (catalog) add('views', figureStem, { figure: catalog.figure, inputs: {} });
  };
  const addOperation = (operation: string, point?: PresentationPosition) => {
    const placed = point ?? (placement.current?.scope === scope ? placement.current.point : undefined);
    placement.current = undefined;
    add('operations', operation.split('.').at(-1)!, { operation, params: {}, inputs: {} }, placed);
  };
  const place = (key: string, point?: PresentationPosition) => {
    if (key === figureEntry) addFigure();
    else addOperation(key, point);
  };
  const addCommand = bindStudioCommand('addComponent', () => {
    if (focusedView === 'figures') addFigure();
    else openLibrary(cursor.current?.scope === scope ? cursor.current.point : undefined);
  }, { name: focusedView === 'figures' ? 'addFigure' : 'addOperation', enabled: canEdit });
  const connect = bindStudioCommand('connectPorts', () => open('wire'), { enabled: canEdit && roles.length > 0 });
  function connectSelected() {
    const source = sourceIndex === '' ? undefined : sources[Number(sourceIndex)];
    if (!source || !selected || !role || !key) return;
    void perform(async () => {
      const review = await readStudioAnalyze(saved.project_id, {
        revision: saved.revision, key, wire: { element: selected, role, source: source.source },
      });
      if (review.document !== document || review.files?.length !== 1) throw new ProjectEditConflictError();
      await save(studioName('connectPorts'), review.files);
      setDialog(undefined);
    });
  }

  const problemOf = (name: string) => {
    const found = diagnostics.find((item) => item.element === name);
    return found && problemReason(found);
  };
  const items: CanvasNode[] = [
    { id: 'states', selectionId: 'states', name: 'states', glyph: <Database aria-hidden="true" /> },
    ...Object.entries(operations).map(([name, value]) => {
      const label = definitions.get(value.operation)?.label ?? value.operation;
      return {
        id: name, selectionId: name, name, glyph: <Sigma aria-hidden="true" />,
        caption: label === name ? undefined : label, problem: problemOf(name),
      };
    }),
    ...Object.keys(figures).map((name) => ({
      id: name, selectionId: name, name, glyph: <BarChart3 aria-hidden="true" />, problem: problemOf(name),
    })),
  ];
  const canvas = useRef<CanvasViewport>(null);
  const viewport = canvasViewportCommands(() => canvas.current, !!document);
  const connections = [...Object.entries(operations), ...Object.entries(figures)].flatMap(([target, value]) =>
    Object.entries(value.inputs).map(([role, source]) => ({
      id: JSON.stringify([target, role]), source: 'input' in source ? source.input : source.element, target,
      label: ('input' in source ? source.input : source.product) + ' → ' + role,
    })));
  const selectedProblems = diagnostics.filter((item) => selected && item.element === selected);
  const selectedPlan = selected ? planned.get(selected) : undefined;
  function writeParameter(path: readonly string[], value: JsonValue) {
    if (!operation || !selected) return;
    const [name, ...rest] = path;
    const next = setIn(operation.params?.[name], rest, value);
    void perform(() => save(humanizeLabel(name), [operation.params
      ? file(['operations', selected, 'params', name], next)
      : file(['operations', selected, 'params'], { [name]: next })]));
  }
  const libraryChoice = library?.scope === scope ? library.key : undefined;
  const libraryItem = libraryChoice === figureEntry ? { label: humanizeLabel(figureStem), owner: studioName('figures'), description: '' }
    : libraryChoice ? definitions.get(libraryChoice) : undefined;
  const elementInspector = selected ? <Inspector name={selected}
    type={definition?.label ?? (figure ? studioName('figures') : undefined)}>
    {selectedProblems.map((problem) => <p key={problem.path + problem.detail} role="status">{problemReason(problem)}</p>)}
    {selectedPlan && <p role="status" className="flex items-center gap-1.5">
      <StatusMark status={selectedPlan.status} />{studioRunStatus(selectedPlan.status)}
      {selectedPlan.reason && <span className="text-shell-muted">{selectedPlan.reason}</span>}
    </p>}
    {operation && definition && key && <SettingsForm key={selected} schema={definition.parameter_json_schema} value={operation.params ?? {}}
      projected={(path) => {
        const schema = path.length === 1 && definition.param_schema.find(({ name }) => name === path[0]);
        const options = schema && catalog?.variables[definition.key]?.[schema.name];
        return schema && options ? fromParamSchema(schema, operation.params?.[schema.name], canEdit
          ? (value) => writeParameter(path, value) : undefined, { options }) : undefined;
      }}
      write={canEdit ? writeParameter : undefined} />}
    {figure && selectedFigure && <ValueField key={JSON.stringify([saved.revision, selected, selectedFigure.caption])}
      model={rawField({
        id: 'caption', label: studioName('caption'), type: { kind: 'string' }, value: selectedFigure.caption ?? '',
        change: canEdit && inlineFigure ? (caption) => {
          if (caption !== (selectedFigure.caption ?? '')) {
            void perform(() => save(studioName('caption'), [file(['views', selected, 'figure', 'caption'], caption)]));
          }
        } : undefined,
      })} />}
  </Inspector> : undefined;
  const inspector = libraryItem ? <Inspector name={libraryItem.label} type={libraryItem.owner}>
    {libraryItem.description && <p className="text-shell-muted">{libraryItem.description}</p>}
  </Inspector> : elementInspector;
  const drawer = <AnalysisDrawer sections={sections} request={libraryRequest} canEdit={canEdit}
    entries={[...(catalog?.operations ?? []).map(({ key, label, owner, description }) => ({
      id: key, name: label, group: owner, terms: description,
    })), ...catalog ? [{ id: figureEntry, name: humanizeLabel(figureStem), group: studioName('figures') }] : []]}
    libraryChoice={libraryChoice} onLibraryChoice={(choice) => {
      setLibrary(choice ? { scope, key: choice } : undefined);
      if (choice) element.onSelect(null);
    }}
    place={(key) => place(key)}
    outline={items.map(({ id, name, glyph, problem }) => {
      const result = planned.get(id);
      return { id, name, glyph, trailing: problem ? <span className="inline-flex" {...appTip(problem)}>
        <StatusMark status="cannot_run" label={studioName('problems')} />
      </span> : result && <span className="inline-flex" {...appTip(studioRunStatus(result.status))}>
        <StatusMark status={result.status} label={studioRunStatus(result.status)} />
      </span> };
    })}
    selectedId={selected} onSelect={(id) => {
      setLibrary(undefined);
      element.onSelect(id);
    }} />;
  const figureList = <ul className="space-y-2 p-2">{Object.entries(figures).map(([name, view]) => {
    const caption = figureDefinition(name, view.figure)?.caption;
    const result = planned.get(name);
    return <li key={name} className="flex items-center gap-2">
      <Button pressed={selected === name} onClick={() => element.onSelect(name)}>{name}</Button>
      {caption && <span>{caption}</span>}
      {result && <span>{studioRunStatus(result.status)}</span>}
    </li>;
  })}</ul>;
  const alert = !opened && issue && <p className="p-2" role="alert">{issue}</p>;
  return {
    catalog: catalogState,
    selectedFigure: figure ? selected : undefined,
    analysis: {
      body: <CanvasStrip viewport={viewport}>{alert}<div className="min-h-0 flex-1">
        <ManagedDocumentCanvas key={scope} {...element} items={items} connections={connections} viewportRef={canvas}
          positions={document ? saved.presentation?.documents?.[document]?.positions : undefined}
          openAt={canEdit ? openLibrary : undefined} placing={busy}
          dropType={canEdit ? libraryDrag : undefined} onDropItem={(key, point) => place(key, point)}
          rememberPoint={(point) => { cursor.current = point ? { scope, point } : undefined; }} />
      </div></CanvasStrip>,
      drawer, inspector,
    },
    figures: { body: <>{alert}{figureList}
      {figure && selected && selectedPlan && <SavedFigureResult key={JSON.stringify([scope, selected])}
        history={selectedPlan} name={selected} />}
    </>, drawer, inspector: figure ? inspector : undefined },
    commands: document && focusedView === 'analysis' ? [addCommand, connect, ...viewport]
      : document && focusedView === 'figures' ? [addCommand, connect] : [],
    overlay: opened && <Dialog ariaLabel={studioName('connectPorts')}
      manageFocus panelClassName="grid w-[min(26rem,90vw)] gap-3" onClose={() => setDialog(undefined)}>
      {issue && <p role="alert">{issue}</p>}
      <form className="grid gap-2" onSubmit={(event) => { event.preventDefault(); connectSelected(); }}>
        <label className="grid gap-1">{studioName('sourcePort')}
          <Select aria-label={studioName('sourcePort')} value={sourceIndex === '' ? null : sourceIndex} onValueChange={setSourceIndex}
            items={sources.map(({ label }, index) => ({ value: String(index), label }))} />
        </label>
        <label className="grid gap-1">{studioName('inputPort')}
          <Select aria-label={studioName('inputPort')} value={role || null} onValueChange={setRole}
            items={roles.map((name) => ({ value: name, label: selected + '.' + name }))} />
        </label>
        <Button type="submit" className="justify-self-end" disabled={!canEdit || sourceIndex === '' || !role}>
          {studioName('connectPorts')}
        </Button>
      </form>
    </Dialog>,
  };
}

const figureEntry = '\u0000figure';
const figureStem = 'velocity_summary';
const libraryDrag = 'application/x-feedbax-operation';

/** The Analysis drawer: the Library of operations and figures, folded by group, and the document's Outline. */
function AnalysisDrawer({
  sections, request, canEdit, entries, libraryChoice, onLibraryChoice, place, outline, selectedId, onSelect,
}: {
  sections: DrawerSections;
  /** Changes when the Add command asks for the Library: it opens and its search takes the keyboard. */
  request: number;
  canEdit: boolean;
  entries: readonly LibraryItem[];
  libraryChoice?: string;
  onLibraryChoice(key: string | undefined): void;
  place(key: string): void;
  outline: readonly TreeNode[];
  selectedId: string | null;
  onSelect(id: string | null): void;
}) {
  const search = useRef<HTMLInputElement>(null);
  const handled = useRef(request);
  useEffect(() => {
    if (handled.current === request) return;
    handled.current = request;
    sections.setOpen('analysis.library', true);
    requestAnimationFrame(() => search.current?.focus());
  }, [request, sections]);
  return <Drawer>
    <DrawerSection sections={sections} id="analysis.library" name={studioName('library')}>
      <LibraryList items={entries} sections={sections} section="analysis.library" searchLabel={studioName('searchLibrary')}
        selected={libraryChoice} onInspect={onLibraryChoice} onPlace={canEdit ? place : undefined}
        dragType={canEdit ? libraryDrag : undefined} searchRef={search} />
    </DrawerSection>
    <DrawerSection sections={sections} id="analysis.outline" name={studioName('outline')}>
      <Tree aria-label={studioName('outline')} nodes={outline} selected={selectedId ? [selectedId] : []}
        onSelectedChange={(ids) => onSelect(ids.at(-1) ?? null)} />
    </DrawerSection>
  </Drawer>;
}
