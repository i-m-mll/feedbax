import { useQuery } from '@tanstack/react-query';
import { useRef, useState, useSyncExternalStore } from 'react';
import { apiErrorMessage } from '@/api/request';
import { readStudioAnalyze } from '@/api/studioAnalyze';
import { studioAccessibleName, studioName, studioRunStatus } from '@/components/activity/studioVocabulary';
import { StudioDefinitionForm, type StudioFieldControl } from '@/components/pipeline/StudioDefinitionForm';
import { Dialog } from '@/components/ui/Dialog';
import { humanizeLabel } from '@/components/values/ValueSpecField';
import { bindStudioCommand, commandAttributes } from '@/features/projectWindow/studioCommands';
import {
  AnalysisDocumentSchema, type AnalysisDocument, type JsonPrimitive, type JsonValue, type OperationElement,
  type ParamSchema, type PresentationPosition, type StudioProjectFilePatch,
  type StudioSavedTrainKey, type StudioSavedTrainRuns,
} from '@/generated/studioContracts';
import { authoredPointer } from './authoredEdits';
import { ManagedDocumentCanvas, useElementSelection } from './ManagedDocumentCanvas';
import { ProjectEditConflictError, type ProjectEditController } from './projectEditController';
import { savedRunId } from './projectWindowPlan';
import { problemReason, type ProblemReveal } from './projectWindowProblems';

type Source = OperationElement['inputs'][string];
const control = 'rounded-shell-control border border-shell-divider px-2 py-1 disabled:opacity-50';
const dialogClass = 'grid w-[min(26rem,90vw)] gap-3 rounded-shell-control border border-shell-divider bg-shell-chrome p-4 text-shell-foreground shadow-xl';

function fieldControl(schema: ParamSchema, variables?: string[]): StudioFieldControl | undefined {
  const options = variables ?? schema.options;
  if (options) return { kind: 'choice', options: options.map((value) => ({ value, label: value })) };
  if (schema.type === 'bool') return { kind: 'boolean' };
  if (schema.type === 'int' || schema.type === 'float') return {
    kind: 'number', min: schema.min ?? undefined, max: schema.max ?? undefined,
    step: schema.step ?? (schema.type === 'int' ? 1 : undefined),
  };
}

function scalar(value: unknown): JsonPrimitive {
  return typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean' ? value : null;
}

export function useManagedAnalyze({
  editor, key, projection, wholeStep, focusedView, reveal, savePosition,
}: {
  editor: ProjectEditController;
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
  const element = useElementSelection(scope, reveal, reveal?.target.element?.kind === 'analysisElement'
    ? reveal.target.element.id : undefined);
  const [dialog, setDialog] = useState<{ scope: string; mode: 'add' | 'wire'; point?: PresentationPosition }>();
  const [query, setQuery] = useState('');
  const [sourceIndex, setSourceIndex] = useState('');
  const [role, setRole] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<{ scope: string; message: string }>();
  const inFlight = useRef(false);
  const cursor = useRef<{ scope: string; point: PresentationPosition }>();
  const authoring = useQuery({
    queryKey: ['studio-analyze', saved.project_id, saved.revision, id],
    queryFn: ({ signal }) => readStudioAnalyze(saved.project_id, { revision: saved.revision, key: key! }, signal),
    enabled: !!document && !!key && key.variant === 'default' && key.row === 'default' && !draft.sweep,
    staleTime: 30_000,
    retry: false,
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
  const resolvedFigures = new Map(projection?.window_map?.analysis_figures
    ?.filter(({ key }) => savedRunId(key) === id).map((view) => [view.name, view]));
  function figureDefinition(name: string, source: NonNullable<AnalysisDocument['views']>[string]['figure']) {
    return resolvedFigures.get(name) ?? ('document' in source ? undefined : source);
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
    .map(({ name, status }) => [name, status]));
  const issue = error?.scope === scope ? error.message : authoring.error ? apiErrorMessage(authoring.error) : undefined;

  function open(mode: 'add' | 'wire', point?: PresentationPosition) {
    if (!canEdit) return;
    setError(undefined);
    setQuery('');
    setSourceIndex('');
    setRole(roles.length === 1 ? roles[0] : '');
    setDialog({ scope, mode, point });
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

  const items = [
    { id: 'states', selectionId: 'states', label: 'states' },
    ...Object.entries(operations).map(([name, value]) => ({
      id: name, selectionId: name, label: <>{name}<br />{definitions.get(value.operation)?.label ?? value.operation}</>,
      problem: diagnostics.find((item) => item.element === name)?.detail,
    })),
    ...Object.keys(figures).map((name) => ({
      id: name, selectionId: name, label: <>{name}<br />{studioName('figures')}</>,
      problem: diagnostics.find((item) => item.element === name)?.detail,
    })),
  ];
  const connections = [...Object.entries(operations), ...Object.entries(figures)].flatMap(([target, value]) =>
    Object.entries(value.inputs).map(([role, source]) => ({
      id: JSON.stringify([target, role]), source: 'input' in source ? source.input : source.element, target,
      label: ('input' in source ? source.input : source.product) + ' → ' + role,
    })));
  const selectedProblems = diagnostics.filter((item) => selected && item.element === selected);
  const inspector = <div className="space-y-2 p-2">
    {selectedProblems.map((problem) => <p key={problem.path + problem.detail} role="status">{problemReason(problem)}</p>)}
    {operation && definition && key && <StudioDefinitionForm selection={{ kind: 'steps', stepIds: [key.step] }}
      fields={definition.param_schema.map((schema) => ({
        id: schema.name, label: humanizeLabel(schema.name),
        resolved: [{ id: schema.name, value: scalar(operation.params?.[schema.name] ?? schema.default) }],
        control: fieldControl(schema, catalog?.variables[definition.key]?.[schema.name]),
        actions: canEdit ? ['set'] : [],
      }))}
      onEdit={({ fieldId, edit }) => {
        if (edit.kind !== 'set' || !selected) return;
        void perform(() => save(humanizeLabel(fieldId), [operation.params
          ? file(['operations', selected, 'params', fieldId], edit.value)
          : file(['operations', selected, 'params'], { [fieldId]: edit.value })]));
      }} />}
    {figure && selected && selectedFigure && <label className="grid gap-1">
      {studioName('caption')}
      <input className={control} key={JSON.stringify([saved.revision, selected, selectedFigure.caption])}
        defaultValue={selectedFigure.caption ?? ''} disabled={!canEdit || 'document' in figure.figure}
        onBlur={({ currentTarget }) => {
          const caption = currentTarget.value;
          if (!('document' in figure.figure) && caption !== (selectedFigure.caption ?? '')) {
            void perform(() => save(studioName('caption'), [file(['views', selected, 'figure', 'caption'], caption)]));
          }
        }}
        onKeyDown={(event) => { if (event.key === 'Enter') event.currentTarget.blur(); }} />
    </label>}
  </div>;
  const drawer = <div className="space-y-2 p-2">
    {catalog?.operations.map((entry) => <button key={entry.key} type="button" className={control} disabled={!canEdit}
      aria-label={studioAccessibleName('add', entry.label)}
      onClick={() => addOperation(entry.key)}>
      {entry.label}
    </button>)}
    <button type="button" className={control} disabled={!canEdit} onClick={addFigure}>{studioName('addFigure')}</button>
    <button type="button" className={control} {...commandAttributes(connect)} disabled={!connect.enabled} onClick={connect.invoke}>
      {connect.label}
    </button>
  </div>;
  const figureList = <ul className="space-y-2 p-2">{Object.entries(figures).map(([name, view]) => {
    const caption = figureDefinition(name, view.figure)?.caption;
    return <li key={name}>
      <button type="button" className={control} aria-pressed={selected === name} onClick={() => element.onSelect(name)}>
        {name}
      </button>
      {caption && <span className="px-2">{caption}</span>}
      {planned.has(name) && <span className="px-2">{studioRunStatus(planned.get(name)!)}</span>}
    </li>;
  })}</ul>;
  const alert = !opened && issue && <p className="p-2" role="alert">{issue}</p>;
  return {
    selectedFigure: figure ? selected : undefined,
    analysis: {
      body: <div className="flex h-full min-h-0 flex-col">{alert}<div className="min-h-0 flex-1">
        <ManagedDocumentCanvas key={scope} {...element} items={items} connections={connections}
          positions={document ? saved.presentation?.documents?.[document]?.positions : undefined}
          openAt={canEdit ? (point) => open('add', point) : undefined} placing={!!opened?.point}
          rememberPoint={(point) => { cursor.current = point ? { scope, point } : undefined; }} />
      </div></div>,
      drawer, inspector,
    },
    figures: { body: <>{alert}{figureList}</>, drawer, inspector: figure ? inspector : undefined },
    commands: document && (focusedView === 'analysis' || focusedView === 'figures') ? [addCommand, connect] : [],
    overlay: opened && <Dialog ariaLabel={studioName(opened.mode === 'add' ? 'addOperation' : 'connectPorts')}
      manageFocus panelClassName={dialogClass} onClose={() => setDialog(undefined)}>
      {issue && <p role="alert">{issue}</p>}
      {opened.mode === 'add' ? <>
        <input data-dialog-autofocus className={control} aria-label={studioName('operations')} value={query}
          onChange={(event) => setQuery(event.currentTarget.value)} />
        {catalog?.operations.filter(({ label }) => label.toLowerCase().includes(query.toLowerCase()))
          .map((entry) => <button key={entry.key} type="button" className={control} disabled={!canEdit}
            onClick={() => addOperation(entry.key, opened.point)}>
            {entry.label}
          </button>)}
      </> : <form className="grid gap-2" onSubmit={(event) => { event.preventDefault(); connectSelected(); }}>
        <label className="grid gap-1">{studioName('sourcePort')}
          <select data-dialog-autofocus className={control} value={sourceIndex} onChange={(event) => setSourceIndex(event.currentTarget.value)}>
            <option value="" disabled />{sources.map(({ label, schemas }, index) =>
              <option key={label} value={index} title={schemas.join(', ')}>{label}</option>)}
          </select>
        </label>
        <label className="grid gap-1">{studioName('targetPort')}
          <select className={control} value={role} onChange={(event) => setRole(event.currentTarget.value)}>
            <option value="" disabled />{roles.map((name) => <option key={name} value={name}>{selected}.{name}</option>)}
          </select>
        </label>
        <button type="submit" className={control} disabled={!canEdit || sourceIndex === '' || !role}>
          {studioName('connectPorts')}
        </button>
      </form>}
    </Dialog>,
  };
}
