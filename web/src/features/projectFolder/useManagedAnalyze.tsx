import { useQuery } from '@tanstack/react-query';
import { useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { apiErrorMessage } from '@/api/request';
import { studioRegionIssue, type StudioRegionState } from '@/components/activity/StudioRegion';
import { readStudioAnalyze } from '@/api/studioAnalyze';
import { studioAccessibleName, studioName, studioRunStatus } from '@/components/activity/studioVocabulary';
import { CanvasStrip } from '@/components/canvas/CanvasStrip';
import { canvasViewportCommands, type CanvasViewport } from '@/components/canvas/StudioCanvas';
import { Button } from '@/components/ui/Button';
import { Dialog } from '@/components/ui/Dialog';
import { TextField } from '@/components/ui/TextField';
import { humanizeLabel } from '@/components/values/humanizeLabel';
import { SettingsForm } from '@/components/values/SettingsForm';
import { setIn } from '@/components/values/adapters/jsonSchema';
import { fromParamSchema } from '@/components/values/adapters/paramSchema';
import { bindStudioCommand } from '@/features/projectWindow/studioCommands';
import {
  AnalysisDocumentSchema, type AnalysisDocument, type InputBinding, type JsonPrimitive, type JsonValue, type OperationElement,
  type ParamSchema, type PresentationPosition, type StudioProjectFilePatch,
  type StudioSavedTrainKey, type StudioSavedTrainRuns,
} from '@/generated/studioContracts';
import { authoredPointer } from './authoredEdits';
import { ManagedDocumentCanvas } from './ManagedDocumentCanvas';
import {
  choosePick, useNamedSelection, usePick, type ElementSelectionStore,
} from '@/features/projectWindow/elementSelection';
import type { PlanStepView } from '@/features/pipeline/planView';
import { AnalyzeElementInputs, AnalyzeRoleBinding } from './AnalyzeReferences';
import { ProjectEditConflictError, type ProjectEditController } from './projectEditController';
import { savedRunId } from './projectWindowPlan';
import { problemReason, type ProblemReveal } from './projectWindowProblems';
import { SavedFigureResult } from './SavedLightProducts';

type Source = OperationElement['inputs'][string];

export function useManagedAnalyze({
  editor, key, projection, wholeStep, focusedView, reveal, savePosition, selection, steps,
}: {
  editor: ProjectEditController;
  selection: ElementSelectionStore;
  /** The project's steps, whose outputs the step's input roles can be bound to. */
  steps: readonly PlanStepView[];
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
  const [dialog, setDialog] = useState<{ scope: string; mode: 'add'; point?: PresentationPosition }>();
  const [query, setQuery] = useState('');
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
  const canBind = !!step && wholeStep && !pending && !outside && !busy;
  const canEdit = canBind && !!analysis && !!document && !!catalog && catalog.document === document &&
    catalog.revision === saved.revision;
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
    element: name, products: Object.keys(definitions.get(value.operation)?.outputs ?? {}),
  }));
  const selectedInput = selected ? declarations[selected] : undefined;
  const armed = usePick(selection);
  const canvasPick = useMemo(() => armed && document ? {
    accepts: (id: string) => armed.accepts({ kind: 'element', address: { document, element: { kind: 'named', id } } }),
    choose: (id: string) => { choosePick(selection, { kind: 'element', address: { document, element: { kind: 'named', id } } }); },
  } : null, [armed, document, selection]);
  const diagnostics = (projection?.window_map?.problems ?? []).filter((problem) => id &&
    problem.variant === key?.variant && problem.step === key?.step && problem.row === key?.row);
  const planned = new Map(projection?.window_map?.analysis_elements?.filter(({ key }) => savedRunId(key) === id)
    .map((item) => [item.name, item]));
  const issue = error?.scope === scope ? error.message : undefined;
  const catalogState: StudioRegionState = authoring.error
    ? { kind: 'failed', issue: studioRegionIssue('operations', authoring.error), retry: () => { void authoring.refetch(); } }
    : authoring.isLoading ? { kind: 'loading' } : { kind: 'result', missing: [] };

  function open(mode: 'add', point?: PresentationPosition) {
    if (!canEdit) return;
    setError(undefined);
    setQuery('');
    setDialog({ scope, mode, point });
  }

  async function perform(work: () => Promise<void>, allowed = canEdit) {
    if (!allowed || inFlight.current) return;
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
      setDialog(undefined);
      if (liveScope.current === scope) element.onSelect(name);
      if (point) await savePosition(document!, name, point);
    });
  }
  const addFigure = () => {
    if (catalog) add('views', 'velocity_summary', { figure: catalog.figure, inputs: {} });
  };
  const addOperation = (operation: string, point?: PresentationPosition) => {
    add('operations', operation.split('.').at(-1)!, { operation, params: {}, inputs: {} }, point);
  };
  const addCommand = bindStudioCommand('addComponent', () => {
    if (focusedView === 'figures') addFigure();
    else open('add', cursor.current?.scope === scope ? cursor.current.point : undefined);
  }, { name: focusedView === 'figures' ? 'addFigure' : 'addOperation', enabled: canEdit });
  function wire(element: string, role: string, source: Source) {
    if (!key) return;
    void perform(async () => {
      const review = await readStudioAnalyze(saved.project_id, {
        revision: saved.revision, key, wire: { element, role, source },
      });
      if (review.document !== document || review.files?.length !== 1) throw new ProjectEditConflictError();
      await save(humanizeLabel(role), review.files);
    });
  }
  function bind(role: string, binding: InputBinding) {
    if (!key) return;
    void perform(async () => {
      if (liveScope.current !== scope || editor.getSnapshot().saved.revision !== saved.revision) {
        throw new ProjectEditConflictError();
      }
      editor.editFiles(humanizeLabel(role), [{ address: { kind: 'step', id: key.step }, patches: [{
        op: step?.inputs && role in step.inputs ? 'replace' : 'add', path: authoredPointer(['inputs', role]), value: binding as JsonValue,
      }] }], { reviewedRevision: saved.revision });
      await editor.flush();
    }, canBind);
  }

  const items = [
    ...Object.keys(declarations).map((name) => ({ id: name, selectionId: name, label: name })),
    ...Object.entries(operations).map(([name, value]) => ({
      id: name, selectionId: name, label: <>{name}<br />{definitions.get(value.operation)?.label ?? value.operation}</>,
      problem: diagnostics.find((item) => item.element === name)?.detail,
    })),
    ...Object.keys(figures).map((name) => ({
      id: name, selectionId: name, label: <>{name}<br />{studioName('figures')}</>,
      problem: diagnostics.find((item) => item.element === name)?.detail,
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
  const inspector = <div className="space-y-2 p-2">
    {selected && selectedInput && <AnalyzeRoleBinding role={selected} kind={selectedInput.kind} binding={step?.inputs?.[selected]}
      project={saved.project} steps={steps} picking={selection} onBind={canBind ? (binding) => bind(selected, binding) : undefined} />}
    {selected && document && (operation ?? figure) && <AnalyzeElementInputs document={document} element={selected} roles={roles}
      inputs={(operation ?? figure)!.inputs} declarations={declarations} outputs={outputs} picking={selection}
      onWire={canEdit ? (role, source) => wire(selected, role, source) : undefined} />}
    {selectedProblems.map((problem) => <p key={problem.path + problem.detail} role="status">{problemReason(problem)}</p>)}
    {selectedPlan && <p role="status">{studioRunStatus(selectedPlan.status)}
      {selectedPlan.reason && <span className="block text-shell-muted">{selectedPlan.reason}</span>}
    </p>}
    {operation && definition && key && <div className="grid gap-1">
      <SettingsForm key={selected} schema={definition.parameter_json_schema} value={operation.params ?? {}}
        projected={(path) => {
          const schema = path.length === 1 && definition.param_schema.find(({ name }) => name === path[0]);
          const options = schema && catalog?.variables[definition.key]?.[schema.name];
          return schema && options ? fromParamSchema(schema, operation.params?.[schema.name], canEdit
            ? (value) => writeParameter(path, value) : undefined, { options }) : undefined;
        }}
        write={canEdit ? writeParameter : undefined} />
    </div>}
    {figure && selected && selectedFigure && <label className="grid gap-1">
      {studioName('caption')}
      <TextField key={JSON.stringify([saved.revision, selected, selectedFigure.caption])}
        defaultValue={selectedFigure.caption ?? ''} disabled={!canEdit || !inlineFigure}
        onBlur={({ currentTarget }) => {
          const caption = currentTarget.value;
          if (inlineFigure && caption !== (selectedFigure.caption ?? '')) {
            void perform(() => save(studioName('caption'), [file(['views', selected, 'figure', 'caption'], caption)]));
          }
        }}
        onKeyDown={(event) => { if (event.key === 'Enter') event.currentTarget.blur(); }} />
    </label>}
  </div>;
  const drawer = <div className="flex flex-wrap gap-2 p-2">
    {catalog?.operations.map((entry) => <Button key={entry.key} disabled={!canEdit}
      aria-label={studioAccessibleName('add', entry.label)}
      onClick={() => addOperation(entry.key)}>
      {entry.label}
    </Button>)}
    <Button disabled={!canEdit} onClick={addFigure}>{studioName('addFigure')}</Button>
  </div>;
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
          positions={document ? saved.presentation?.documents?.[document]?.positions : undefined} pick={canvasPick}
          openAt={canEdit ? (point) => open('add', point) : undefined} placing={!!opened?.point}
          rememberPoint={(point) => { cursor.current = point ? { scope, point } : undefined; }} />
      </div></CanvasStrip>,
      drawer, inspector,
    },
    figures: { body: <>{alert}{figureList}
      {figure && selected && selectedPlan && <SavedFigureResult key={JSON.stringify([scope, selected])}
        history={selectedPlan} name={selected} />}
    </>, drawer, inspector: figure ? inspector : undefined },
    commands: document && focusedView === 'analysis' ? [addCommand, ...viewport]
      : document && focusedView === 'figures' ? [addCommand] : [],
    overlay: opened && <Dialog ariaLabel={studioName('addOperation')}
      manageFocus panelClassName="grid w-[min(26rem,90vw)] gap-3" onClose={() => setDialog(undefined)}>
      {issue && <p role="alert">{issue}</p>}
      <TextField data-dialog-autofocus aria-label={studioName('operations')} value={query} onValueChange={setQuery} />
      {catalog?.operations.filter(({ label }) => label.toLowerCase().includes(query.toLowerCase()))
        .map((entry) => <Button key={entry.key} variant="ghost" align="start" disabled={!canEdit}
          onClick={() => addOperation(entry.key, opened.point)}>
          {entry.label}
        </Button>)}
    </Dialog>,
  };
}
