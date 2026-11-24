import { useMemo, useState, type ReactNode } from 'react';
import { Shelf, type ShelfRow } from '@/components/timeline/Shelf';
import { batchRows, nominalRows, trialRows, type ShelfTask } from '@/features/projectFolder/task/shelfRows';
import { ProjectWindow, ProjectWindowPalette } from '@/features/projectWindow/ProjectWindow';
import { useDrawerSections } from '@/features/projectWindow/DrawerForms';
import { defaultProjectWindowState, type ProjectWindowState } from '@/features/projectWindow/projectWindowState';
import { TaskInspector } from '@/features/projectFolder/task/TaskInspector';
import { TaskOutline, type OutlineEntry, type OutlineGroup } from '@/features/projectFolder/task/TaskOutline';
import { TaskGrid } from '@/features/projectFolder/task/TaskGrid';
import { TaskDocumentBody, TaskPythonBody } from '@/features/projectFolder/task/TaskView';
import { TaskVariables } from '@/features/projectFolder/task/TaskVariables';
import type { TaskDocumentView, TaskSelection } from '@/features/projectFolder/task/taskModel';
import type { FieldModel } from '@/components/values/fieldModel';
import type { TrialShown } from '@/features/projectFolder/task/trialSample';
import { aggregate, documentView, fixtures, pythonView, templates, type DocumentSpec, type PythonSpec } from './taskFixtures';

/** Trials the stepper steps through, and the batch size All aggregates. */
const stepped = 8;
const batchSize = 64;

type Specs = Record<string, DocumentSpec | PythonSpec>;

function useTasks() {
  const [specs, setSpecs] = useState<Specs>(() => Object.fromEntries(fixtures.map(({ key, spec }) => [key, spec])));
  const [texts, setTexts] = useState<Record<string, Record<string, string>>>({});
  const edit = (key: string, change: (spec: DocumentSpec) => DocumentSpec) =>
    setSpecs((current) => current[key].kind === 'document' ? { ...current, [key]: change(current[key] as DocumentSpec) } : current);
  const view = (key: string) => {
    const spec = specs[key];
    return spec.kind === 'document'
      ? documentView(spec, texts[key] ?? {}, (id, text) => setTexts((current) => ({ ...current, [key]: { ...current[key], [id]: text } })))
      : pythonView(spec);
  };
  return { specs, view, edit };
}

/** The view's sampled trials and their aggregate, as the sampling route would return them. */
function useSample(key: string, spec: DocumentSpec | PythonSpec, seed: number) {
  return useMemo(() => {
    const fixture = fixtures.find((candidate) => candidate.key === key)!;
    if (spec.kind !== 'document' || !fixture.sample) return undefined;
    const trials = fixture.sample(seed, batchSize);
    const steps = Math.round(spec.duration / spec.stepSeconds);
    return {
      trials,
      batch: aggregate(trials, steps, spec.epochs),
    };
  }, [key, spec, seed]);
}

/** Which sampled trial is shown: none (the definition), the batch as one aggregate, or one trial by its index. */
type TrialChoice = 'definition' | 'all' | number;

/** The fixture task as the shelf draws it: its epochs' bounds and the epochs each signal holds a value over. */
function shelfTask(view: TaskDocumentView): ShelfTask {
  return {
    document: 'gallery', duration: view.duration, epochs: view.epochs.map(({ bounds }) => bounds),
    signals: view.signals.map(({ id, name, cells }) => ({
      id, name, cells: cells.flatMap((cell) => cell.kind === 'value' ? [{ first: cell.first, last: cell.last }] : []),
    })),
  };
}

/** The shelf's rows for the nominal trial, one sampled trial, or the batch. */
function shelfRows(view: TaskDocumentView, shown: TrialShown): ShelfRow[] {
  const step = view.stepSeconds ?? view.duration / 100;
  switch (shown.kind) {
    case 'definition': return nominalRows(shelfTask(view));
    case 'one': return trialRows(shelfTask(view), shown.record, step);
    case 'all': return batchRows(shelfTask(view), shown.aggregate, step);
  }
}

function shownOf(choice: TrialChoice, sample: ReturnType<typeof useSample>): TrialShown {
  if (!sample || choice === 'definition') return { kind: 'definition' };
  if (choice === 'all') return { kind: 'all', count: sample.trials.length, aggregate: sample.batch };
  return { kind: 'one', index: choice, count: stepped, record: sample.trials[choice] };
}

function moved<T extends { id: string }>(items: readonly T[], id: string, to: number) {
  const next = [...items];
  const [item] = next.splice(items.findIndex((candidate) => candidate.id === id), 1);
  next.splice(to, 0, item);
  return next;
}

/** Removing an epoch shortens the cells that span it and removes those only in it. */
function withoutEpoch(spec: DocumentSpec, id: string): DocumentSpec {
  const index = spec.epochs.findIndex((epoch) => epoch.id === id);
  const neighbour = (offset: number) => spec.epochs[index + offset]?.id;
  return {
    ...spec,
    epochs: spec.epochs.filter((epoch) => epoch.id !== id),
    repeats: spec.repeats.filter(({ first, last }) => first !== id && last !== id),
    signals: spec.signals.map((signal) => ({ ...signal, cells: signal.cells.flatMap((cell) => {
      if (cell.first === id && cell.last === id) return [];
      return [{ ...cell, first: cell.first === id ? neighbour(1) : cell.first, last: cell.last === id ? neighbour(-1) : cell.last }];
    }) })),
  };
}

const firstReason = (fields: readonly (FieldModel | undefined)[]) => fields.find((field) => field?.reason)?.reason;

/** The fixture's variables, epochs and signals as the Outline lists them. */
function outlineGroups(view: TaskDocumentView): Partial<Record<OutlineGroup, OutlineEntry[]>> {
  return {
    variables: view.variables.map(({ id, name, value }) => ({ id, name, reason: firstReason([value]) })),
    epochs: view.epochs.map(({ id, name, end }) => ({ id, name, reason: firstReason([end]) })),
    signals: view.signals.map(({ id, name, cells }) => ({
      id, name, reason: firstReason(cells.map((cell) => cell.kind === 'value' ? cell.value : undefined)),
    })),
  };
}

const outlineKinds = { variables: 'variable', epochs: 'epoch', signals: 'signal', modelInputs: 'signal' } as const;

function outlineSelection(selected: TaskSelection | undefined) {
  if (!selected || selected.kind === 'repeat') return undefined;
  const group: OutlineGroup = selected.kind === 'variable' ? 'variables'
    : selected.kind === 'epoch' || selected.kind === 'end' ? 'epochs' : 'signals';
  return { group, id: selected.kind === 'cell' ? selected.signal : selected.id };
}

function TaskWindow() {
  const tasks = useTasks();
  const [key, setKey] = useState('delayed-reaches');
  const [state, setState] = useState<ProjectWindowState>({ ...defaultProjectWindowState, mode: 'details', primaryView: 'task' });
  const sections = useDrawerSections(state, setState);
  const [selected, setSelected] = useState<TaskSelection | undefined>({ kind: 'end', id: 'cue' });
  const view = tasks.view(key);
  const shown: TrialShown = { kind: 'definition' };
  const fixture = fixtures.find((candidate) => candidate.key === key)!;
  const choose = (next: string) => {
    setKey(next);
    setSelected(undefined);
  };
  const edit = (change: (spec: DocumentSpec) => DocumentSpec) => tasks.edit(key, change);
  let counter = 0;
  const fresh = (prefix: string) => `${prefix}_${Date.now().toString(36)}${counter++}`;
  const picker = {
    label: 'task' as const, document: key, name: fixture.name,
    documents: fixtures.map(({ key: entry, name, python, problem }) => ({ key: entry, name, depth: 0, python, problem })),
    templates,
    choose,
    create: async () => key,
    duplicate: async () => key,
    rename: () => undefined,
  };
  return <ProjectWindowPalette>
    <div className="h-screen">
      <ProjectWindow state={state} onStateChange={setState} projectName="Reaching"
        views={{ available: ['runs', 'model', 'task', 'objective', 'training'], absent: {} }}
        map={null} mapDrawer={null} compactMap={<span className="px-3 text-sm text-shell-muted">train</span>} commands={[]}
        renderView={(slotView) => slotView !== 'task' ? { body: null } : {
          padded: true,
          picker,
          body: view.kind === 'document'
            ? <TaskDocumentBody
              variables={<TaskVariables variables={view.variables} selected={selected} select={setSelected} edits={{
                add: (name) => edit((current) => ({ ...current, variables: [...current.variables, { id: fresh('v'), name, type: 'Number', text: '0' }] })),
                rename: (id, name) => edit((current) => ({ ...current, variables: current.variables.map((item) => item.id === id ? { ...item, name } : item) })),
                remove: (id) => edit((current) => ({ ...current, variables: current.variables.filter((item) => item.id !== id) })),
                move: (id, to) => edit((current) => ({ ...current, variables: moved(current.variables, id, to) })),
              }} />}
              grid={<TaskGrid epochs={view.epochs} repeats={view.repeats} signals={view.signals} selected={selected} select={setSelected} edits={{
                addEpoch: (name, at) => edit((current) => {
                  const epoch = { id: fresh('e'), name, end: '0.1 s' };
                  const epochs = [...current.epochs];
                  epochs.splice(at, 0, epoch);
                  return { ...current, epochs, signals: current.signals.map((signal) => ({ ...signal, cells: [...signal.cells,
                    { first: epoch.id, last: epoch.id, text: null }].sort((a, b) =>
                    epochs.findIndex(({ id }) => id === a.first) - epochs.findIndex(({ id }) => id === b.first)) })) };
                }),
                renameEpoch: (id, name) => edit((current) => ({ ...current, epochs: current.epochs.map((item) => item.id === id ? { ...item, name } : item) })),
                removeEpochs: (ids) => edit((current) => ids.reduce(withoutEpoch, current)),
                addSignal: (name) => edit((current) => ({ ...current, signals: [...current.signals, {
                  id: fresh('s'), name, type: 'Number', cells: [{ first: current.epochs[0].id, last: current.epochs.at(-1)!.id, text: null }],
                }] })),
              }} />} />
            : <TaskPythonBody view={view} />,
          drawer: view.kind === 'document' ? <TaskOutline sections={sections} groups={outlineGroups(view)} selected={outlineSelection(selected)}
            select={(group, id) => setSelected({ kind: outlineKinds[group], id })} /> : undefined,
          inspector: view.kind === 'document' ? <TaskInspector view={view} selected={selected} shown={shown} /> : undefined,
        }} />
    </div>
  </ProjectWindowPalette>;
}

function Frame({ name, children, className }: { name: string; children: ReactNode; className?: string }) {
  return <figure className="grid content-start gap-1">
    <div className={className ?? 'rounded-control border border-shell-divider bg-shell-surface p-2'}>{children}</div>
    <figcaption className="text-xs text-shell-muted">{name}</figcaption>
  </figure>;
}

/** One task's states side by side: the shelf for the nominal trial, a sample and the batch, and the Inspector for an End, a cell, a variable and a signal's traces. */
function TaskStates({ fixtureKey, end, cell, variable, signal }: {
  fixtureKey: string; end: string; cell: { signal: string; id: string }; variable: string; signal: string;
}) {
  const tasks = useTasks();
  const spec = tasks.specs[fixtureKey];
  const view = tasks.view(fixtureKey) as TaskDocumentView;
  const sample = useSample(fixtureKey, spec, 1);
  const definition: TrialShown = { kind: 'definition' };
  const trial = shownOf(0, sample);
  const batch = shownOf('all', sample);
  const [time, setTime] = useState(0);
  const strip = (shown: TrialShown) => <Shelf rows={shelfRows(view, shown)} duration={view.duration}
    stepSeconds={view.stepSeconds} time={time} onTimeChange={setTime} selected={[]} onSelect={() => undefined} />;
  const inspector = (selected: TaskSelection, shown: TrialShown = definition) =>
    <TaskInspector view={view} selected={selected} shown={shown} />;
  const box = 'w-[244px] rounded-control border border-shell-divider bg-shell-sidebar';
  return <section className="grid gap-3" aria-label={fixtures.find(({ key }) => key === fixtureKey)!.name}>
    <h2 className="text-sm font-medium">{fixtures.find(({ key }) => key === fixtureKey)!.name}</h2>
    <div className="grid max-w-4xl gap-3">
      <Frame name="nominal">{strip(definition)}</Frame>
      <Frame name="sample 1 of 8">{strip(trial)}</Frame>
      <Frame name={`${batchSize} trials`}>{strip(batch)}</Frame>
    </div>
    <div className="flex flex-wrap items-start gap-3">
      <Frame name="End" className={box}>{inspector({ kind: 'end', id: end })}</Frame>
      <Frame name="Cell" className={box}>{inspector({ kind: 'cell', ...cell })}</Frame>
      <Frame name="Variable" className={box}>{inspector({ kind: 'variable', id: variable })}</Frame>
      <Frame name="Variable, trial 1" className={box}>{inspector({ kind: 'variable', id: variable }, trial)}</Frame>
      <Frame name="Signal, trial 1" className={box}>{inspector({ kind: 'signal', id: signal }, trial)}</Frame>
      <Frame name={`Signal, all ${batchSize}`} className={box}>{inspector({ kind: 'signal', id: signal }, batch)}</Frame>
    </div>
  </section>;
}

/**
 * The intended Task view, drawn from fixtures with the view's own parts: the window as a Train step shows it, then each
 * document task's states side by side.
 */
export function TaskViewGallery() {
  return <main className="bg-shell-surface text-shell-foreground">
    <TaskWindow />
    <div className="grid gap-8 p-6">
      <TaskStates fixtureKey="delayed-reaches" end="cue" cell={{ signal: 'goal', id: 'goal:1' }} variable="start" signal="target_pos" />
      <TaskStates fixtureKey="three-reaches" end="move" cell={{ signal: 'target', id: 'target:1' }} variable="targets" signal="target" />
    </div>
  </main>;
}
